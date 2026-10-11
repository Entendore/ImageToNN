"""
Reactive Diffusion as Feed-Forward Neural Network
with Monte Carlo Tree Search (MCTS) + Statistical Pruning
================================================================
MCTS runs in a background worker thread — UI never freezes.
PCA1-PCA4 of feed-forward network weights (V-field) computed per evaluation.
Loss landscape statistics (mean, max, std) are extracted from the ENTIRE dataset.
Top 3 V-field configurations are visualized in real-time.
"""
from __future__ import annotations
import json, logging, sys, time, threading, csv
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import cupy as cp; HAS_CUPY = True
except ImportError:
    cp = None; HAS_CUPY = False

try:
    from sklearn.datasets import load_digits, load_iris, load_wine, load_breast_cancer
    HAS_SKLEARN = True
except ImportError:
    load_digits = load_iris = load_wine = load_breast_cancer = None
    HAS_SKLEARN = False

from PySide6.QtCore import Qt, QTimer, QPoint, Signal, Slot, QRectF, QThread, QObject
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPalette, QPen, QBrush, QFont,
    QKeySequence, QShortcut, QLinearGradient, QRadialGradient, QCursor,
    QPainterPath, QMouseEvent, QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QDoubleSpinBox, QSpinBox, QPushButton,
    QComboBox, QScrollArea, QCheckBox, QGridLayout,
    QSplitter, QFileDialog, QStatusBar, QTabWidget, QSizePolicy,
    QToolBar, QMessageBox, QToolButton,
    QGraphicsScene, QGraphicsView, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsTextItem,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RD_NN")

@dataclass(frozen=True)
class AppConstants:
    DEFAULT_WIDTH: int = 256; DEFAULT_HEIGHT: int = 256
    NAN_REPLACE: float = 0.0; POSINF_REPLACE: float = 1.0; NEGINF_REPLACE: float = 0.0
    DEFAULT_SPLIT: float = 0.8; DEFAULT_MAX_SAMPLES: int = 5000
    DEFAULT_RIDGE_LAMBDA: float = 1e-4
    DEFAULT_SEED_RADIUS: int = 10; DEFAULT_SEED_COUNT: int = 10
    TREE_UPDATE_INTERVAL: int = 5     # update tree view every N UI ticks
    PCA_CLEANUP_INTERVAL: int = 50    # clean PCA data every N ticks

MODERN_QSS = """
QMainWindow { background-color: #1a1a1f; }
QWidget { color: #e4e4e8; font-family: 'Segoe UI', sans-serif; font-size: 12px; }
#sidebar { background-color: #14141a; border-right: 1px solid #2a2a35; }
#sidebar QScrollArea { border: none; background-color: transparent; }
#sidebar QScrollArea > QWidget > QWidget { background-color: transparent; }
.section-header { background-color: #1e1e26; color: #4fc3f7; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; padding: 10px 14px; border-top: 1px solid #2a2a35; border-bottom: 1px solid #2a2a35; }
.section-header:hover { background-color: #25252f; }
.section-body { background-color: #1a1a20; padding: 10px 14px; }
QPushButton { background-color: #2a2a35; color: #e4e4e8; border: 1px solid #3a3a45; border-radius: 6px; padding: 7px 14px; font-weight: 500; min-height: 22px; }
QPushButton:hover { background-color: #353542; border-color: #4fc3f7; }
QPushButton:pressed { background-color: #1a1a22; }
QPushButton:checked { background-color: #0d47a1; border-color: #4fc3f7; color: #ffffff; }
QPushButton#primary { background-color: #0d47a1; border-color: #4fc3f7; color: #ffffff; }
QPushButton#primary:hover { background-color: #1565c0; }
QDoubleSpinBox, QSpinBox, QComboBox { background-color: #14141a; border: 1px solid #3a3a45; border-radius: 5px; padding: 5px 8px; color: #e4e4e8; min-height: 20px; }
QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus { border-color: #4fc3f7; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView { background-color: #1a1a22; border: 1px solid #3a3a45; selection-background-color: #0d47a1; }
QSlider::groove:horizontal { background: #2a2a35; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal { background: #4fc3f7; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: #81d4fa; }
QSlider::sub-page:horizontal { background: #0d47a1; border-radius: 2px; }
QCheckBox { spacing: 8px; color: #c0c0d0; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #3a3a45; border-radius: 3px; background-color: #14141a; }
QCheckBox::indicator:checked { background-color: #0d47a1; border-color: #4fc3f7; }
QTabWidget::pane { background-color: #1a1a20; border: 1px solid #2a2a35; border-top: none; border-radius: 0 0 8px 8px; }
QTabBar::tab { background-color: #1e1e26; color: #808090; border: 1px solid #2a2a35; border-bottom: none; padding: 9px 18px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px; }
QTabBar::tab:selected { background-color: #1a1a20; color: #4fc3f7; border-bottom: 2px solid #4fc3f7; }
QTabBar::tab:hover:!selected { background-color: #25252f; color: #c0c0d0; }
QScrollBar:vertical { background-color: #14141a; width: 10px; border: none; }
QScrollBar::handle:vertical { background-color: #3a3a45; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background-color: #4fc3f7; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { background-color: #14141a; height: 10px; border: none; }
QScrollBar::handle:horizontal { background-color: #3a3a45; border-radius: 5px; min-width: 30px; }
QLabel { color: #b0b0c0; }
QLabel#value-display { color: #4fc3f7; font-weight: 600; font-family: 'Consolas', monospace; }
QLabel#metric-value { color: #4fc3f7; font-size: 22px; font-weight: 700; font-family: 'Consolas', monospace; }
QLabel#metric-label { color: #808090; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
QStatusBar { background-color: #0d0d12; color: #808090; border-top: 1px solid #2a2a35; font-family: 'Consolas', monospace; font-size: 11px; }
QStatusBar::item { border: none; }
QToolBar { background-color: #14141a; border-bottom: 1px solid #2a2a35; spacing: 6px; padding: 4px 8px; }
QToolBar QToolButton { background-color: transparent; color: #b0b0c0; border: 1px solid transparent; border-radius: 5px; padding: 5px 10px; font-size: 12px; }
QToolBar QToolButton:hover { background-color: #25252f; border-color: #3a3a45; color: #4fc3f7; }
QToolBar QToolButton:checked { background-color: #0d47a1; color: #ffffff; }
QToolTip { background-color: #25252f; color: #e4e4e8; border: 1px solid #4fc3f7; border-radius: 4px; padding: 5px 8px; font-size: 11px; }
QGraphicsView { background-color: #0e0e14; border: 1px solid #2a2a35; border-radius: 6px; }
"""

class OptimizationStatus(Enum):
    RUNNING = "running"; CONVERGED = "converged"; STALLED = "stalled"

@dataclass(frozen=True)
class OptimizerConfig:
    population_size: int = 6
    exploration_constant: float = 1.414
    perturb_scale: float = 0.08
    max_tree_depth: int = 20
    warmup_steps: int = 200
    convergence_window: int = 100
    convergence_threshold: float = 1e-4
    restart_threshold: int = 80
    pruning_enabled: bool = True
    prune_min_visits: int = 5
    prune_z_score: float = 1.96
    prune_margin: float = 0.0
    seed_radius: int = 10
    seed_count: int = 10
    ridge_lambda: float = 1e-4
    perturb_flags: Dict[str, bool] = field(default_factory=lambda: {"F": True, "k": True, "Du": True, "Dv": True, "dt": False})
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {"F": (0.001,0.100), "k": (0.001,0.080), "Du": (0.10,2.00), "Dv": (0.05,1.50), "dt": (0.10,2.00)})
    pca_prune_enabled: bool = True
    pca_variance_margin: float = 0.1
    pca_redundancy_threshold: float = 0.98
    def __post_init__(self):
        if self.population_size < 1: raise ValueError("pop>=1")

def _gl(rf, gf, bf, n=256):
    t = np.linspace(0,1,n); return (np.stack([np.clip(rf(t),0,1),np.clip(gf(t),0,1),np.clip(bf(t),0,1)],-1)*255).astype(np.uint8)
COLORMAPS = {
    "Ink": lambda: _gl(lambda t: 0.961-0.902*(t*t*(3-2*t)), lambda t: 0.933-0.886*(t*t*(3-2*t)), lambda t: 0.882-0.764*(t*t*(3-2*t))),
    "Grayscale": lambda: _gl(lambda t: t, lambda t: t, lambda t: t),
    "Viridis": lambda: _gl(lambda t: 0.27+1.2*t-1.8*t**2+0.8*t**3, lambda t: 1.4*t-0.6*t**2, lambda t: 0.33+0.6*t-1.3*t**2+0.7*t**3),
    "Inferno": lambda: _gl(lambda t: 1.6*t-0.6*t**2, lambda t: 0.5*t+0.8*t**2-0.3*t**3, lambda t: 0.05+1.2*t-2*t**2+0.8*t**3),
    "Magma": lambda: _gl(lambda t: 0.05+2*t-t**2, lambda t: 0.02+0.4*t+0.8*t**2-0.3*t**3, lambda t: 0.1+1.6*t-2.2*t**2+0.7*t**3),
    "Plasma": lambda: _gl(lambda t: 0.05+2.4*t-2*t**2+0.6*t**3, lambda t: 0.03+0.2*t+1.2*t**2-0.5*t**3, lambda t: 0.53+0.6*t-2*t**2+t**3),
}
INK_LUT = COLORMAPS["Ink"]()

CLASS_COLORS = [
    QColor(239,83,80), QColor(66,165,245), QColor(102,187,106), QColor(255,202,40),
    QColor(171,71,188), QColor(255,138,101), QColor(38,198,218), QColor(236,64,122),
    QColor(151,192,205), QColor(220,220,70),
]

@dataclass(frozen=True)
class BenchmarkConfig:
    name: str; num_features: int; num_classes: int; n_samples: int = 800; source: str = "synthetic"

BENCHMARK_PRESETS = {
    "Digits (64F, 10C)": BenchmarkConfig("digits",64,10,1797,"sklearn"),
    "Iris (4F, 3C)": BenchmarkConfig("iris",4,3,150,"sklearn"),
    "Wine (13F, 3C)": BenchmarkConfig("wine",13,3,178,"sklearn"),
    "Breast Cancer (30F, 2C)": BenchmarkConfig("breast_cancer",30,2,569,"sklearn"),
    "Blobs (64F, 4C)": BenchmarkConfig("Blobs",64,4,800,"synthetic"),
    "XOR (2F, 2C)": BenchmarkConfig("XOR",2,2,800,"synthetic"),
    "Linear (16F, 3C)": BenchmarkConfig("Linear",16,3,800,"synthetic"),
    "Circles (2F, 2C)": BenchmarkConfig("Circles",2,2,800,"synthetic"),
    "Moons (2F, 2C)": BenchmarkConfig("Moons",2,2,800,"synthetic"),
}

class SklearnDataset:
    _L = None
    @classmethod
    def _gl(cls):
        if cls._L is None: cls._L = {"digits":load_digits,"iris":load_iris,"wine":load_wine,"breast_cancer":load_breast_cancer}
        return cls._L
    def __init__(self, name, split=0.8):
        if not HAS_SKLEARN: raise ImportError("pip install scikit-learn")
        d=self._gl()[name](); X=d.data.astype(np.float64); y=d.target.astype(int)
        xr=X.max(0)-X.min(0); xr[xr==0]=1; X=((X-X.min(0))/xr)*2-1
        self.num_features=X.shape[1]; self.num_classes=len(np.unique(y)); self.n_samples=len(y)
        idx=np.random.default_rng(42).permutation(self.n_samples); X,y=X[idx],y[idx]
        s=int(split*self.n_samples); self.X_train,self.y_train=X[:s],y[:s]; self.X_test,self.y_test=X[s:],y[s:]

class BenchmarkDataset:
    def __init__(self, config, split=0.8):
        self.num_features=config.num_features; self.num_classes=config.num_classes; self.n_samples=config.n_samples
        self.X_train,self.y_train,self.X_test,self.y_test=self._gen(config,split)
    def _gen(self,c,split):
        rng=np.random.default_rng(42); n=c.n_samples
        if c.name=="Blobs":
            X=rng.normal(size=(n,c.num_features)); ct=rng.normal(scale=2.5,size=(c.num_classes,c.num_features))
            y=rng.integers(0,c.num_classes,size=n); X+=ct[y]
        elif c.name=="XOR": X=rng.uniform(-1.5,1.5,size=(n,2)); y=((X[:,0]*X[:,1])>0).astype(int)
        elif c.name=="Circles":
            n2=n//2; ao=rng.uniform(0,2*np.pi,n2); ai=rng.uniform(0,2*np.pi,n-n2)
            X=np.vstack([np.column_stack([np.cos(ao),np.sin(ao)])*rng.normal(3,0.3,(n2,2)),np.column_stack([np.cos(ai),np.sin(ai)])*rng.normal(1,0.2,(n-n2,2))])
            y=np.array([0]*n2+[1]*(n-n2))
        elif c.name=="Moons":
            n2=n//2; t1=rng.uniform(0,np.pi,n2); t2=rng.uniform(0,np.pi,n-n2)
            X=np.vstack([np.column_stack([np.cos(t1),np.sin(t1)])+rng.normal(0,0.15,(n2,2)),np.column_stack([1-np.cos(t2),0.5-np.sin(t2)])+rng.normal(0,0.15,(n-n2,2))])
            y=np.array([0]*n2+[1]*(n-n2))
        else:
            X=rng.normal(size=(n,c.num_features)); W=rng.normal(scale=1.5,size=(c.num_features,c.num_classes)); y=np.argmax(X@W,axis=1)
        xr=X.max(0)-X.min(0); xr[xr==0]=1; X=((X-X.min(0))/xr)*2-1
        s=int(split*n); return X[:s],y[:s],X[s:],y[s:]

class CustomBenchmarkDataset:
    def __init__(self, fp, tci=-1, ms=5000, split=0.8):
        self.num_features=0; self.num_classes=0; self.n_samples=0
        self.X_train,self.y_train,self.X_test,self.y_test=self._load(fp,tci,ms,split)
    def _load(self,fp,tci,ms,split):
        raw=np.genfromtxt(fp,delimiter=',',skip_header=1,max_rows=ms,dtype=np.float64)
        if raw.ndim==1: raw=raw.reshape(-1,1)
        raw=raw[~np.isnan(raw).any(1)]
        if raw.shape[0]<10: raise ValueError("Too small")
        if tci<0: tci=raw.shape[1]+tci
        X=raw[:,:tci]; y=raw[:,tci].astype(int)
        xr=X.max(0)-X.min(0); xr[xr==0]=1; X=((X-X.min(0))/xr)*2-1
        self.num_features=X.shape[1]; ul=np.unique(y); lm={o:n for n,o in enumerate(ul)}
        y=np.array([lm[v] for v in y]); self.num_classes=len(ul); self.n_samples=len(y)
        idx=np.random.default_rng(42).permutation(self.n_samples); X,y=X[idx],y[idx]
        s=int(split*self.n_samples); return X[:s],y[:s],X[s:],y[s:]


GS_CUDA_KERNEL = r"""
extern "C" __global__ void gs_step(const double* u, const double* v,
                                     double* u_new, double* v_new,
                                     double Du, double Dv, double F, double k,
                                     double dt, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int x1 = (x == 0) ? w-1 : x-1;
    int x2 = (x == w-1) ? 0 : x+1;
    int y1 = (y == 0) ? h-1 : y-1;
    int y2 = (y == h-1) ? 0 : y+1;
    int i = y * w + x;
    double uv = u[i], vv = v[i];
    double lu = u[y*w+x1] + u[y*w+x2] + u[y1*w+x] + u[y2*w+x] - 4.0*uv;
    double lv = v[y*w+x1] + v[y*w+x2] + v[y1*w*x] + v[y2*w+x] - 4.0*vv;
    double uvv = uv * vv * vv;
    u_new[i] = uv + dt * (Du*lu - uvv + F*(1.0 - uv));
    v_new[i] = vv + dt * (Dv*lv + uvv - (F+k)*vv);
}
"""

class GrayScott:
    def __init__(self, w=256, h=256, use_gpu=False):
        self.width=w; self.height=h; self.use_gpu=use_gpu and HAS_CUPY
        self.xp=cp if self.use_gpu else np
        self.u=self.xp.ones((h,w),dtype=self.xp.float64); self.v=self.xp.zeros((h,w),dtype=self.xp.float64)
        self.Du=1.0; self.Dv=0.5; self.F=0.055; self.k=0.062; self.dt=1.0
        self._u_buf=None; self._v_buf=None
        if self.use_gpu:
            self._gs_kernel=cp.RawKernel(GS_CUDA_KERNEL,"gs_step")
            self._u_buf=cp.empty_like(self.u); self._v_buf=cp.empty_like(self.v)

    def reset(self): self.u[:]=1.0; self.v[:]=0.0

    def seed_random(self, radius=10, count=10):
        radius = max(2, min(radius, min(self.width, self.height) // 2 - 1))
        self.reset(); rng=np.random.default_rng()
        for _ in range(count):
            cx=rng.integers(radius,self.width-radius); cy=rng.integers(radius,self.height-radius)
            r=rng.integers(max(2,radius//2),radius+1)
            yy,xx=np.ogrid[-cy:self.height-cy,-cx:self.width-cx]; m=xx*xx+yy*yy<=r*r
            if self.use_gpu: m=cp.asarray(m)
            self.v[m]=0.25+0.05*rng.random(); self.u[m]=0.50+0.05*rng.random()

    def step(self, iters=1):
        if self.use_gpu:
            w, h = self.width, self.height
            grid = ((w + 15) // 16, (h + 15) // 16)
            block = (16, 16, 1)
            for _ in range(iters):
                self._gs_kernel(
                    grid, block,
                    (self.u.ravel(), self.v.ravel(),
                     self._u_buf.ravel(), self._v_buf.ravel(),
                    self.Du, self.Dv, self.F, self.k, self.dt, w, h)
                )
                self.u, self._u_buf = self._u_buf, self.u
                self.v, self._v_buf = self._v_buf, self.v
        else:
            xp=self.xp; u,v=self.u,self.v; Du,Dv,F,k,dt=self.Du,self.Dv,self.F,self.k,self.dt
            for _ in range(iters):
                lu=xp.roll(u,1,0)+xp.roll(u,-1,0)+xp.roll(u,1,1)+xp.roll(u,-1,1)-4*u
                lv=xp.roll(v,1,0)+xp.roll(v,-1,0)+xp.roll(v,1,1)+xp.roll(v,-1,1)-4*v
                uvv=u*v*v; u=u+dt*(Du*lu-uvv+F*(1-u)); v=v+dt*(Dv*lv+uvv-(F+k)*v)
                self.u=u; self.v=v

    def get_v_np(self): return self.v.get() if self.use_gpu else self.v
    def get_u_np(self): return self.u.get() if self.use_gpu else self.u
    def get_params(self): return {"F":self.F,"k":self.k,"Du":self.Du,"Dv":self.Dv,"dt":self.dt}
    def set_params(self, p):
        for k in ("F","k","Du","Dv","dt"):
            if k in p: setattr(self,k,float(p[k]))
    def save_state(self): return {"u":self.u.copy(),"v":self.v.copy(),"params":self.get_params()}
    def restore_state(self, s): self.u=s["u"]; self.v=s["v"]; self.set_params(s["params"])


# ==============================================================================
# FEED-FORWARD NETWORK
# ==============================================================================
class FeedForwardNetwork:
    def __init__(self, sim):
        self.sim=sim; self.input_pattern="sine"; self.activation_fn="tanh"
        self.last_activations=None; self.last_output=None
        self._rng=np.random.default_rng(42)
        self.last_loss=0.0; self.last_accuracy=0.0; self._input_proj=None; self._proj_nf=0
        self.ridge_lambda=AppConstants.DEFAULT_RIDGE_LAMBDA
        self.last_pca_variance=[0.0,0.0,0.0,0.0]
        self.last_pca_projections=None
        self.last_pca_labels=None
        self.last_pca_total=0.0

    def test(self):
        w=self.sim.get_v_np(); H,W=w.shape
        if self.input_pattern=="sine": x=np.sin(np.linspace(0,4*np.pi,H))
        elif self.input_pattern=="random": x=self._rng.standard_normal(H)
        elif self.input_pattern=="impulse": x=np.zeros(H); x[H//2]=1.0
        elif self.input_pattern=="gradient": x=np.linspace(-1,1,H)
        elif self.input_pattern=="step": x=np.ones(H); x[:H//2]=-1.0
        else: x=np.zeros(H)
        am={"tanh":0,"sigmoid":1,"relu":2,"leaky_relu":3}; at=am.get(self.activation_fn,0)
        cur=x.copy(); acts=np.zeros((H,W))
        for i in range(W):
            z=w[:,i]*cur
            if at==0: cur=np.tanh(z)
            elif at==1: cur=1.0/(1.0+np.exp(-np.clip(z,-500,500)))
            elif at==2: cur=np.maximum(0,z)
            else: cur=np.where(z>0,z,0.01*z)
            acts[:,i]=cur
        self.last_output=cur.copy(); self.last_activations=acts

    def evaluate_dataset(self, dataset):
        vf=self.sim.get_v_np(); H,W=vf.shape; nf=dataset.num_features; nc=dataset.num_classes
        if self._input_proj is None or self._proj_nf!=nf:
            sc=np.sqrt(2.0/max(1,nf)); self._input_proj=self._rng.standard_normal((nf,H))*sc; self._proj_nf=nf

        vs=np.nan_to_num(vf,nan=1.0,posinf=1.0,neginf=0.0)

        # === PCA on Network Weights (V-field columns) ===
        try:
            weights = vs.T
            W_mean = weights.mean(axis=0)
            W_centered = weights - W_mean
            
            # FIXED: Normalize by max abs value to prevent SVD overflow when V-field values explode
            w_max = np.max(np.abs(W_centered))
            if w_max > 1e-6:
                W_norm = W_centered / w_max
            else:
                W_norm = W_centered
                
            U_svd, S_svd, Vt_svd = np.linalg.svd(W_norm, full_matrices=False)
            
            # Clip singular values for extreme safety
            S_svd = np.clip(S_svd, 0, 1e8)
            total_var = float((S_svd**2).sum())
            if total_var > 1e-10:
                top4_ratio = (S_svd[:4]**2) / total_var
                if len(top4_ratio) < 4:
                    top4_ratio = np.pad(top4_ratio, (0, 4 - len(top4_ratio)))
            else:
                top4_ratio = np.zeros(4)
            self.last_pca_variance = top4_ratio.tolist()
            self.last_pca_total = float(top4_ratio.sum())
            top4_vecs = Vt_svd[:4].T 
            self.last_pca_projections = (W_centered @ top4_vecs).copy() 
            self.last_pca_labels = np.arange(W, dtype=int) 
        except Exception as e:
            logger.warning(f"PCA on weights failed: {e}")
            self.last_pca_variance=[0,0,0,0]; self.last_pca_total=0.0
            self.last_pca_projections=None; self.last_pca_labels=None

        am={"tanh":np.tanh,"sigmoid":lambda z:1/(1+np.exp(-np.clip(z,-500,500))),
            "relu":lambda z:np.maximum(0,z),"leaky_relu":lambda z:np.where(z>0,z,0.01*z)}
        af=am.get(self.activation_fn,np.tanh)
        def _fwd(Z):
            for c in range(W): Z=af(Z*vs[:,c])
            return Z
            
        # === Train Readout on Train Set ===
        Ztr=dataset.X_train@self._input_proj
        Htr=_fwd(Ztr)
        Htrb=np.hstack([Htr,np.ones((Htr.shape[0],1))])
        Yoh=np.zeros((len(dataset.y_train),nc)); Yoh[np.arange(len(dataset.y_train)),dataset.y_train]=1.0
        lam=self.ridge_lambda; HtH=Htrb.T@Htrb; HtY=Htrb.T@Yoh
        try: Wo=np.linalg.solve(HtH+lam*np.eye(HtH.shape[0]),HtY)
        except np.linalg.LinAlgError: Wo=np.linalg.pinv(Htrb)@Yoh
        
        # === Evaluate on ENTIRE Dataset (Train + Test Combined) ===
        X_all = np.vstack([dataset.X_train, dataset.X_test])
        y_all = np.concatenate([dataset.y_train, dataset.y_test])
        
        Z_all = X_all @ self._input_proj
        H_all = _fwd(Z_all)
        H_all_b = np.hstack([H_all, np.ones((H_all.shape[0],1))])
        
        lg = H_all_b @ Wo
        lm2 = np.max(lg, 1, keepdims=True)
        el = np.exp(lg - lm2)
        pr = el / np.sum(el, 1, keepdims=True)
        yt = y_all 
        
        sample_losses = -np.log(pr[np.arange(len(yt)),yt] + 1e-10)
        
        loss = float(np.mean(sample_losses))
        if not np.isfinite(loss): loss = 10.0
        acc = float(np.mean(np.argmax(pr, 1) == yt))
        
        loss_std = float(np.std(sample_losses))
        loss_max = float(np.max(sample_losses))
        hidden_mean = float(H_all.mean())
        hidden_std = float(H_all.std())
        
        self.last_loss = loss; self.last_accuracy = acc
        return loss, acc, {
            "loss_std": loss_std, 
            "loss_max": loss_max, 
            "hidden_mean": hidden_mean, 
            "hidden_std": hidden_std
        }

    def get_activations_normalized(self):
        if self.last_activations is None: return None
        a=self.last_activations; mn,mx=a.min(),a.max()
        if mx-mn<1e-9: return np.zeros_like(a)
        return (a-mn)/(mx-mn)


class MCTSNode:
    _id = 0
    def __init__(self, params, parent=None, depth=0):
        self.params=dict(params); self.parent=parent; self.children=[]; self.depth=depth
        self.visits=0; self.total_loss=0.0; self.sum_loss_sq=0.0
        self.is_expanded=False; self.is_pruned=False; self.prune_reason=""
        self.last_loss=0.0; self.last_accuracy=0.0
        self.pca_variance=[0.0,0.0,0.0,0.0]
        self.pca_total=0.0
        self.pca_projections=None
        self.pca_labels=None
        self.dataset_loss_std = 0.0
        self.dataset_loss_max = 0.0
        self.dataset_hidden_mean = 0.0
        self.dataset_hidden_std = 0.0
        MCTSNode._id+=1; self.id=MCTSNode._id
    @property
    def mean_loss(self): return self.total_loss/max(1,self.visits)
    @property
    def loss_variance(self):
        if self.visits<2: return 0.0
        return max(0.0, self.sum_loss_sq/self.visits - self.mean_loss**2)
    @property
    def loss_std(self): return float(np.sqrt(self.loss_variance))
    @property
    def loss_stderr(self):
        if self.visits<2: return float('inf')
        return self.loss_std/float(np.sqrt(self.visits))
    def optimistic_bound(self, z=1.96): return self.mean_loss - z*self.loss_stderr
    def pessimistic_bound(self, z=1.96): return self.mean_loss + z*self.loss_stderr
    def ucb1(self, pv, c=1.414):
        if self.visits==0: return float('inf')
        return -self.mean_loss + c*np.sqrt(np.log(max(1,pv))/max(1,self.visits))


class MCTS:
    def __init__(self, sim, network, config=None):
        self.sim=sim; self.network=network
        self.config=config or OptimizerConfig()
        self.rng=np.random.default_rng()
        self._lock = threading.Lock()
        MCTSNode._id=0; self.root=MCTSNode(sim.get_params())
        self.best_loss=float('inf'); self.best_accuracy=0.0
        self.best_params=dict(sim.get_params()); self.best_node=None
        self.top_nodes = []  # Track top 3 nodes
        self.total_iterations=0; self.steps_without_improvement=0
        self.restart_count=0; self.status=OptimizationStatus.RUNNING
        self.loss_history=[]; self.best_loss_history=[]; self.param_history=[]
        self._recent_losses=[]; self.enabled=False; self.total_pruned=0
        self._pending_config_update = None

    def step(self, dataset=None, warmup_steps=None):
        with self._lock:
            if self._pending_config_update is not None:
                self.config = self._pending_config_update; self._pending_config_update = None
            c = self.config
            if self.status!=OptimizationStatus.RUNNING: return {"status":self.status.value,"step":self.total_iterations}
            if dataset is None: return {"status":"no_dataset","step":self.total_iterations}
            warmup=warmup_steps or c.warmup_steps; self.total_iterations+=1
        node=self.root; path=[]
        while node.is_expanded and node.children:
            viable=[ch for ch in node.children if not ch.is_pruned]
            if not viable: break
            node=max(viable, key=lambda ch: ch.ucb1(node.visits, c.exploration_constant))
            path.append(node)
            if node.depth>=c.max_tree_depth: break
        if node.is_pruned: node=self.root
        if node.visits>0 and not node.is_expanded and not node.is_pruned and node.depth<c.max_tree_depth:
            for _ in range(c.population_size):
                node.children.append(MCTSNode(self._perturb(node.params, node.depth, c), parent=node, depth=node.depth+1))
            node.is_expanded=True
            uv=[ch for ch in node.children if ch.visits==0]
            node=self.rng.choice(uv) if uv else self.rng.choice(node.children)
            path.append(node)
            
        loss, acc, ds_stats = self._evaluate(node.params, dataset, warmup, c)
        node.last_loss=loss; node.last_accuracy=acc
        node.pca_variance = list(self.network.last_pca_variance)
        node.pca_total = self.network.last_pca_total
        node.dataset_loss_std = ds_stats["loss_std"]
        node.dataset_loss_max = ds_stats["loss_max"]
        node.dataset_hidden_mean = ds_stats["hidden_mean"]
        node.dataset_hidden_std = ds_stats["hidden_std"]
        
        cur=node
        while cur:
            cur.visits+=1; cur.total_loss+=loss; cur.sum_loss_sq+=loss*loss; cur=cur.parent
        improved=False
        with self._lock:
            if loss<self.best_loss:
                if self.best_node is not None and self.best_node is not node:
                    self.best_node.pca_projections = None
                    self.best_node.pca_labels = None
                self.best_loss=loss; self.best_accuracy=acc
                self.best_params=dict(node.params); self.best_node=node
                node.pca_projections = self.network.last_pca_projections.copy() if self.network.last_pca_projections is not None else None
                node.pca_labels = self.network.last_pca_labels.copy() if self.network.last_pca_labels is not None else None
                self.sim.set_params(node.params); improved=True; self.steps_without_improvement=0
            else:
                node.pca_projections = None
                node.pca_labels = None
                self.steps_without_improvement+=1
                
            # Update Top 3 Nodes
            if not node.is_pruned:
                self.top_nodes = [tn for tn in self.top_nodes if tn.id != node.id]
                self.top_nodes.append(node)
                self.top_nodes.sort(key=lambda n: n.mean_loss)
                self.top_nodes = self.top_nodes[:3]
                
            self.loss_history.append(loss); self.best_loss_history.append(self.best_loss)
            self.param_history.append(dict(node.params)); self._recent_losses.append(loss)
            if len(self._recent_losses)>c.convergence_window: self._recent_losses.pop(0)
            pruned_iter=0
            all_nodes=self.get_all_nodes()
            if c.pruning_enabled: pruned_iter=self._run_pruning(c, all_nodes); self.total_pruned+=pruned_iter
            self._check_convergence(c)
            if self._should_restart(c): self._trigger_restart()
            total_visited=sum(1 for n in all_nodes if n.visits>0)
            total_pruned=sum(1 for n in all_nodes if n.is_pruned)
        return {"step":self.total_iterations,"status":self.status.value,"loss":loss,
                "best_loss":self.best_loss,"accuracy":acc,"best_accuracy":self.best_accuracy,
                "tree_size":len(all_nodes),"tree_depth":self._max_depth(self.root),
                "node_visits":node.visits,"node_depth":node.depth,
                "steps_without_improvement":self.steps_without_improvement,
                "restarts":self.restart_count,"improved":improved,
                "pruned_this_iter":pruned_iter,"total_pruned":total_pruned,
                "total_visited":total_visited,
                "prune_rate":total_pruned/max(1,total_visited) if total_visited else 0.0,
                "pca_variance":node.pca_variance,"pca_total":node.pca_total,
                "ds_loss_std":node.dataset_loss_std, "ds_loss_max":node.dataset_loss_max,
                "ds_hidden_mean":node.dataset_hidden_mean, "ds_hidden_std":node.dataset_hidden_std}

    def _run_pruning(self, c, all_nodes):
        if self.best_node is None or self.best_node.visits<c.prune_min_visits: return 0
        best=self.best_node; z=c.prune_z_score; margin=c.prune_margin
        threshold=best.pessimistic_bound(z)+margin
        newly=0
        
        for node in all_nodes:
            if node.is_pruned or node.visits<c.prune_min_visits or node.id==best.id: continue
            
            prune_reason = None
            
            if node.optimistic_bound(z)>threshold:
                prune_reason=f"stat: opt={node.optimistic_bound(z):.4f} > thr={threshold:.4f}"
            
            elif c.pca_prune_enabled and node.pca_total < best.pca_total - c.pca_variance_margin:
                prune_reason=f"pca: var={node.pca_total:.4f} < thr={best.pca_total - c.pca_variance_margin:.4f}"
                
            elif c.pca_prune_enabled and node.pca_total > 0:
                v1 = np.array(node.pca_variance)
                norm1 = np.linalg.norm(v1)
                if norm1 > 1e-6:
                    for other in all_nodes:
                        if other.id == node.id or other.is_pruned or other.visits == 0: continue
                        if other.mean_loss >= node.mean_loss: continue 
                        if other.pca_total <= 0: continue
                        v2 = np.array(other.pca_variance)
                        norm2 = np.linalg.norm(v2)
                        if norm2 > 1e-6:
                            sim = np.dot(v1, v2) / (norm1 * norm2)
                            if sim > c.pca_redundancy_threshold:
                                prune_reason = f"redundant: pca_sim={sim:.3f} with #{other.id}"
                                break
                                
            if prune_reason:
                node.is_pruned=True
                node.prune_reason=prune_reason
                newly+=self._prune_subtree(node)+1
                
        changed=True
        while changed:
            changed=False
            for node in all_nodes:
                if node.is_pruned or not node.children or node.id==self.root.id: continue
                if all(ch.is_pruned for ch in node.children):
                    node.is_pruned=True; node.prune_reason="dead-end"; newly+=1; changed=True
                    
        if self.root.children and all(ch.is_pruned for ch in self.root.children): self._trigger_restart()
        return newly

    def _prune_subtree(self, node):
        count=0
        for child in node.children:
            if not child.is_pruned:
                child.is_pruned=True; child.prune_reason=f"parent #{node.id} pruned"
                child.pca_projections=None; child.pca_labels=None
                count+=1+self._prune_subtree(child)
        return count

    def _evaluate(self, params, dataset, warmup, c):
        ss=self.sim.save_state()
        try:
            self.sim.reset(); self.sim.set_params(params)
            self.sim.seed_random(c.seed_radius, c.seed_count)
            self.sim.step(warmup)
            self.network.ridge_lambda = c.ridge_lambda
            return self.network.evaluate_dataset(dataset)
        except Exception as e:
            logger.error(f"MCTS eval fail: {e}")
            return 10.0, 0.0, {"loss_std": 0.0, "loss_max": 10.0, "hidden_mean": 0.0, "hidden_std": 0.0}
        finally: self.sim.restore_state(ss)

    def _perturb(self, params, depth, c):
        np_=dict(params); df=max(0.3, 1.0-depth*0.08)
        for k, dp in c.perturb_flags.items():
            if not dp: continue
            lo,hi=c.bounds[k]; sc=c.perturb_scale*(hi-lo)*df
            nv=params[k]+self.rng.normal(0,sc)
            if nv<lo: nv=lo+(lo-nv)
            elif nv>hi: nv=hi-(nv-hi)
            np_[k]=float(np.clip(nv,lo,hi))
        return np_

    def get_all_nodes(self):
        ns=[]; stack=[self.root]
        while stack: n=stack.pop(); ns.append(n); stack.extend(n.children)
        return ns

    def _max_depth(self, node):
        max_d = node.depth
        stack = [node]
        while stack:
            n = stack.pop()
            if n.depth > max_d: max_d = n.depth
            stack.extend(n.children)
        return max_d

    def _should_restart(self, c): return self.steps_without_improvement>=c.restart_threshold
    def _check_convergence(self, c):
        if len(self._recent_losses)<c.convergence_window: return
        r=self._recent_losses[-c.convergence_window:]
        if max(r)-min(r)<c.convergence_threshold: self.status=OptimizationStatus.CONVERGED
    def _trigger_restart(self):
        self.restart_count+=1; self.steps_without_improvement=0
        if hasattr(self.network,'_input_proj'): self.network._input_proj=None
        self.root=MCTSNode(self.best_params) if self.best_node else MCTSNode(self.sim.get_params())
        self.status=OptimizationStatus.RUNNING
    def request_config_update(self, new_config): self._pending_config_update = new_config

    def clear_pca_data(self, keep_node=None):
        for node in self.get_all_nodes():
            if keep_node is None or node.id != keep_node.id:
                if not node.is_pruned:
                    node.pca_projections = None
                    node.pca_labels = None

    def export_csv(self, path):
        with open(path,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["step","F","k","Du","Dv","dt","loss","best_loss"])
            for i in range(len(self.loss_history)):
                p=self.param_history[i]
                w.writerow([i,p["F"],p["k"],p["Du"],p["Dv"],p["dt"],self.loss_history[i],self.best_loss_history[i]])

    def export_best(self, fp):
        with open(fp,"w") as f: json.dump({"best_params":self.best_params,"best_loss":self.best_loss,
            "best_accuracy":self.best_accuracy,"total_iterations":self.total_iterations,
            "total_pruned":self.total_pruned,
            "best_pca_variance":self.best_node.pca_variance if self.best_node else None,
            "best_pca_total":self.best_node.pca_total if self.best_node else 0.0,
            "best_ds_loss_std": self.best_node.dataset_loss_std if self.best_node else 0.0,
            "best_ds_hidden_mean": self.best_node.dataset_hidden_mean if self.best_node else 0.0},f,indent=2)

    def save_state(self, path):
        state = {
            "best_params": self.best_params,
            "best_loss": self.best_loss,
            "best_accuracy": self.best_accuracy,
            "total_iterations": self.total_iterations,
            "total_pruned": self.total_pruned,
            "restart_count": self.restart_count,
            "loss_history": self.loss_history[-2000:],
            "best_loss_history": self.best_loss_history[-2000:],
            "param_history": self.param_history[-2000:],
        }
        with open(path, "w") as f: json.dump(state, f, indent=2)

    def load_state(self, path):
        with open(path) as f: state = json.load(f)
        self.best_params = state["best_params"]
        self.best_loss = state["best_loss"]
        self.best_accuracy = state["best_accuracy"]
        self.total_iterations = state.get("total_iterations", 0)
        self.total_pruned = state.get("total_pruned", 0)
        self.restart_count = state.get("restart_count", 0)
        self.loss_history = state.get("loss_history", [])
        self.best_loss_history = state.get("best_loss_history", [])
        self.param_history = state.get("param_history", [])
        self.sim.set_params(self.best_params)
        self.root = MCTSNode(self.best_params)
        self.best_node = self.root
        self.best_node.visits = 1
        self.best_node.total_loss = self.best_loss
        self.best_node.last_loss = self.best_loss
        self.best_node.last_accuracy = self.best_accuracy

    def reset(self):
        with self._lock:
            MCTSNode._id=0; self.root=MCTSNode(self.sim.get_params())
            self.best_loss=float('inf'); self.best_accuracy=0.0
            self.best_params=dict(self.sim.get_params()); self.best_node=None
            self.top_nodes.clear()
            self.total_iterations=0; self.steps_without_improvement=0; self.restart_count=0
            self.status=OptimizationStatus.RUNNING
            self.loss_history.clear(); self.best_loss_history.clear(); self.param_history.clear()
            self._recent_losses.clear(); self.total_pruned=0
            if hasattr(self.network,'_input_proj'): self.network._input_proj=None


class MCTSWorker(QObject):
    result_ready = Signal(dict)
    finished = Signal()
    def __init__(self, mcts, dataset_getter, warmup_getter):
        super().__init__()
        self.mcts = mcts; self._dg = dataset_getter; self._wg = warmup_getter
        self._running = False; self._iters_per_batch = 1
    @Slot()
    def run(self):
        self._running = True
        while self._running:
            ds = self._dg()
            if ds is None: time.sleep(0.01); continue
            w = self._wg()
            for _ in range(self._iters_per_batch):
                if not self._running: break
                info = self.mcts.step(dataset=ds, warmup_steps=w)
                self.result_ready.emit(info)
                if info.get("status") == "converged":
                    time.sleep(0.5)
                    break
        self.finished.emit()
    def stop(self): self._running = False
    def set_iters_per_batch(self, n): self._iters_per_batch = max(1, n)


class StatusLED(QWidget):
    def __init__(self, color="green", parent=None):
        super().__init__(parent); self.setFixedSize(12,12); self.color=color; self.active=False
    def set_active(self, a): self.active=a; self.update()
    def paintEvent(self, e):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s=min(self.width(),self.height()); c=QPoint(s//2,s//2); r=s//2-1
        cm={"green":(76,175,80),"red":(244,67,54),"blue":(79,195,247)}
        R,G,B=cm.get(self.color,(100,100,110))
        if self.active:
            g=QRadialGradient(c,r*2); g.setColorAt(0,QColor(R,G,B,180)); g.setColorAt(1,QColor(R,G,B,0))
            p.setBrush(QBrush(g)); p.setPen(Qt.PenStyle.NoPen); p.drawEllipse(c,r*2,r*2)
        gr=QRadialGradient(c-QPoint(r//3,r//3),r*1.5)
        if self.active: gr.setColorAt(0,QColor(min(255,R+80),min(255,G+80),min(255,B+80))); gr.setColorAt(1,QColor(R,G,B))
        else: gr.setColorAt(0,QColor(80,80,90)); gr.setColorAt(1,QColor(40,40,50))
        p.setBrush(QBrush(gr)); p.setPen(QPen(QColor(20,20,25),1)); p.drawEllipse(c,r,r)

class CollapsibleSection(QWidget):
    def __init__(self, title, icon="", parent=None):
        super().__init__(parent)
        self.ml=QVBoxLayout(self); self.ml.setContentsMargins(0,0,0,0); self.ml.setSpacing(0)
        self.header=QPushButton(f"  {icon}  {title}" if icon else f"  {title}")
        self.header.setObjectName("section-header"); self.header.setCheckable(True)
        self.header.setChecked(True); self.header.clicked.connect(self._tog); self.ml.addWidget(self.header)
        self.body=QWidget(); self.body.setObjectName("section-body")
        self.bl=QVBoxLayout(self.body); self.bl.setContentsMargins(10,10,10,10); self.bl.setSpacing(8); self.ml.addWidget(self.body)
    def add_widget(self, w): self.bl.addWidget(w)
    def _tog(self): self.body.setVisible(self.header.isChecked())

class MetricCard(QWidget):
    def __init__(self, label, value="—", color="#4fc3f7", parent=None):
        super().__init__(parent)
        l=QVBoxLayout(self); l.setContentsMargins(12,10,12,10); l.setSpacing(2)
        self.ll=QLabel(label.upper()); self.ll.setObjectName("metric-label"); l.addWidget(self.ll)
        self.lv=QLabel(value); self.lv.setObjectName("metric-value"); self.lv.setStyleSheet(f"color: {color};"); l.addWidget(self.lv)
        self.setStyleSheet("background-color: #1e1e26; border-radius: 6px; border: 1px solid #2a2a35;")
    def set_value(self, v): self.lv.setText(v)

class DisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumSize(128,128)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pm=QPixmap(); self._cm=INK_LUT; self._info=""
    def set_colormap(self, cm_name):
        if cm_name in COLORMAPS: self._cm = COLORMAPS[cm_name]()
    def set_info(self, t): self._info=t; self.update()
    def update_image(self, v):
        vs=np.nan_to_num(v,nan=AppConstants.NAN_REPLACE,posinf=AppConstants.POSINF_REPLACE,neginf=AppConstants.NEGINF_REPLACE)
        vn=np.clip(vs,0,1); idx=(vn*255).astype(np.uint8); rgb=self._cm[idx]
        h,w,_=rgb.shape; qi=QImage(rgb.data,w,h,3*w,QImage.Format.Format_RGB888); self._pm=QPixmap.fromImage(qi); self.update()
    def paintEvent(self, e):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform); p.fillRect(self.rect(),QColor(10,10,14))
        if not self._pm.isNull():
            sc=self._pm.scaled(self.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)
            x=(self.width()-sc.width())//2; y=(self.height()-sc.height())//2; p.drawPixmap(x,y,sc)
        if self._info:
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(QColor(10,10,14,180)))
            f=QFont("Consolas",10); f.setStyleHint(QFont.StyleHint.Monospace); p.setFont(f)
            fm=p.fontMetrics(); lines=self._info.split("\n")
            mw=max(fm.horizontalAdvance(l) for l in lines); pd=10
            p.drawRoundedRect(10,10,mw+pd*2,fm.height()*len(lines)+pd*2,4,4)
            p.setPen(QColor(79,195,247))
            for i,l in enumerate(lines): p.drawText(10+pd,10+pd+fm.ascent()+i*fm.height(),l)
        p.end()

# ==== Plots ====
class LossPlot(QWidget):
    def __init__(self, p=None):
        super().__init__(p); self.setMinimumSize(300,150); self.history=[]; self.best_history=[]; self.max_points=500
    def update_data(self,h,bh): self.history=h[-self.max_points:]; self.best_history=bh[-self.max_points:]; self.update()
    def paintEvent(self,e):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.fillRect(self.rect(),QColor(20,20,25))
        w,hh=self.width(),self.height(); m=40; pw=w-2*m; ph=hh-2*m
        if not self.history:
            p.setPen(QColor(80,80,90)); p.setFont(QFont("Segoe UI",11))
            p.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,"Waiting for MCTS data..."); p.end(); return
        av=self.history+self.best_history; mn,mx=min(av),max(av)
        if mx-mn<1e-9: mx=mn+1.0
        p.setPen(QPen(QColor(40,40,50),1,Qt.PenStyle.DotLine))
        for i in range(5):
            y=m+int(ph*i/4); p.drawLine(m,y,m+pw,y)
            p.setPen(QColor(100,100,110)); p.setFont(QFont("Consolas",9)); p.drawText(5,y+4,f"{mx-(mx-mn)*i/4:.3f}")
            p.setPen(QPen(QColor(40,40,50),1,Qt.PenStyle.DotLine))
        def _d(a,c,f=False):
            if len(a)<2: return
            pt=QPainterPath(); n=len(a); x0=m; y0=m+ph-int(ph*(a[0]-mn)/(mx-mn)); pt.moveTo(x0,y0)
            for i in range(1,n):
                x=m+int(pw*i/max(1,n-1)); y=m+ph-int(ph*(a[i]-mn)/(mx-mn)); pt.lineTo(x,y)
            if f:
                pt.lineTo(m+pw,m+ph); pt.lineTo(m,m+ph); pt.closeSubpath()
                gr=QLinearGradient(0,m,0,m+ph); gr.setColorAt(0,QColor(102,187,106,60)); gr.setColorAt(1,QColor(102,187,106,0))
                p.fillPath(pt,QBrush(gr)); p.setPen(QPen(c,2)); p.drawPath(pt)
            else: p.setPen(QPen(c,2)); p.drawPath(pt)
        _d(self.best_history,QColor(102,187,106),True); _d(self.history,QColor(239,83,80),False)
        p.setFont(QFont("Segoe UI",10))
        p.setPen(QPen(QColor(239,83,80),2)); p.drawLine(w-180,15,w-160,15); p.setPen(QColor(200,200,210)); p.drawText(w-155,19,"Current Loss")
        p.setPen(QPen(QColor(102,187,106),2)); p.drawLine(w-180,32,w-160,32); p.setPen(QColor(200,200,210)); p.drawText(w-155,36,"Best Loss"); p.end()

class PhaseSpacePlot(QWidget):
    def __init__(self,p=None):
        super().__init__(p); self.setMinimumSize(300,150); self.ph_=[]; self.bp={}; self.max_points=1000
    def update_data(self,ph,bp): self.ph_=ph[-self.max_points:]; self.bp=dict(bp); self.update()
    def paintEvent(self,e):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.fillRect(self.rect(),QColor(20,20,25))
        w,h=self.width(),self.height(); m=40; pw=w-2*m; ph=h-2*m; Fmn,Fmx,kmn,kmx=0.0,0.1,0.0,0.08
        p.setPen(QPen(QColor(40,40,50),1,Qt.PenStyle.DotLine))
        for i in range(5):
            y=m+int(ph*i/4); p.drawLine(m,y,m+pw,y); x=m+int(pw*i/4); p.drawLine(x,m,x,m+ph)
        p.setPen(QColor(100,100,110)); p.setFont(QFont("Consolas",9))
        for i in range(5):
            Fv=Fmn+(Fmx-Fmn)*i/4; x=m+int(pw*i/4); p.drawText(x-15,h-m+18,f"{Fv:.3f}")
            kv=kmx-(kmx-kmn)*i/4; y=m+int(ph*i/4); p.drawText(5,y+4,f"{kv:.3f}")
        p.setFont(QFont("Segoe UI",10)); p.setPen(QColor(150,150,160))
        p.drawText(m+pw//2-5,h-5,"F (feed rate)"); p.save(); p.translate(12,m+ph//2+30); p.rotate(-90); p.drawText(0,0,"k (kill rate)"); p.restore()
        if not self.ph_:
            p.setPen(QColor(80,80,90)); p.setFont(QFont("Segoe UI",11))
            p.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,"Waiting..."); p.end(); return
        Fs=[d["F"] for d in self.ph_]; Ks=[d["k"] for d in self.ph_]; n=len(Fs)
        for i in range(n-1):
            x1=m+int(pw*(Fs[i]-Fmn)/(Fmx-Fmn)); y1=m+ph-int(ph*(Ks[i]-kmn)/(kmx-kmn))
            x2=m+int(pw*(Fs[i+1]-Fmn)/(Fmx-Fmn)); y2=m+ph-int(ph*(Ks[i+1]-kmn)/(kmx-kmn))
            p.setPen(QPen(QColor(79,195,247,int(40+200*i/max(1,n-1))),2)); p.drawLine(x1,y1,x2,y2)
        if self.bp:
            cx=m+int(pw*(self.bp.get("F",0.05)-Fmn)/(Fmx-Fmn)); cy=m+ph-int(ph*(self.bp.get("k",0.06)-kmn)/(kmx-kmn))
            gl=QRadialGradient(cx,cy,20); gl.setColorAt(0,QColor(102,187,106,200)); gl.setColorAt(1,QColor(102,187,106,0))
            p.setBrush(QBrush(gl)); p.setPen(Qt.PenStyle.NoPen); p.drawEllipse(QPoint(cx,cy),20,20)
            p.setBrush(QColor(102,187,106)); p.setPen(QPen(QColor(255,255,255),2)); p.drawEllipse(QPoint(cx,cy),7,7)
            p.setPen(QColor(255,255,255)); p.setFont(QFont("Segoe UI",9,QFont.Weight.Bold)); p.drawText(cx+12,cy-5,"BEST")
        p.end()

class ParamTSPlot(QWidget):
    def __init__(self,p=None):
        super().__init__(p); self.setMinimumSize(300,150); self.ph_=[]; self.max_points=500
        self.colors={"F":QColor(255,100,100),"k":QColor(100,255,150),"Du":QColor(100,150,255),"Dv":QColor(255,200,100)}
    def update_data(self,ph): self.ph_=ph[-self.max_points:]; self.update()
    def paintEvent(self,e):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.fillRect(self.rect(),QColor(20,20,25))
        w,h=self.width(),self.height(); m=40; pw=w-2*m; ph=h-2*m
        if not self.ph_:
            p.setPen(QColor(80,80,90)); p.setFont(QFont("Segoe UI",11))
            p.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,"Waiting..."); p.end(); return
        p.setPen(QPen(QColor(40,40,50),1,Qt.PenStyle.DotLine))
        for i in range(5):
            y=m+int(ph*i/4); p.drawLine(m,y,m+pw,y)
        for k,c in self.colors.items():
            vals=[d.get(k,0) for d in self.ph_]; mn,mx=min(vals),max(vals)
            if mx-mn<1e-9: mx=mn+1.0; n=len(vals); p.setPen(QPen(c,2))
            for i in range(n-1):
                x1=m+int(pw*i/max(1,n-1)); y1=m+ph-int(ph*(vals[i]-mn)/(mx-mn))
                x2=m+int(pw*(i+1)/max(1,n-1)); y2=m+ph-int(ph*(vals[i+1]-mn)/(mx-mn))
                p.drawLine(x1,y1,x2,y2)
        p.setFont(QFont("Segoe UI",10)); yl=15
        for k,c in self.colors.items():
            p.setPen(QPen(c,3)); p.drawLine(w-100,yl,w-80,yl); p.setPen(QColor(200,200,210)); p.drawText(w-75,yl+4,k); yl+=18
        p.end()

class SpectrumPlot(QWidget):
    def __init__(self,p=None):
        super().__init__(p); self.setMinimumSize(300,150); self._ps=None
    def update_data(self,ps): self._ps=ps; self.update()
    def paintEvent(self,e):
        p=QPainter(self); p.fillRect(self.rect(),QColor(20,20,25))
        if self._ps is None:
            p.setPen(QColor(80,80,90)); p.setFont(QFont("Segoe UI",11))
            p.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,"Waiting..."); p.end(); return
        ps=self._ps; H,W=ps.shape; pl=np.log1p(ps); pn=pl/(pl.max()+1e-10)
        ps2=np.nan_to_num(pn,nan=0,posinf=1,neginf=0); idx=(np.clip(ps2,0,1)*255).astype(np.uint8)
        t=np.linspace(0,1,256)
        cmap=(np.stack([np.clip(1.6*t-0.6*t**2,0,1),np.clip(0.5*t+0.8*t**2-0.3*t**3,0,1),
             np.clip(0.05+1.2*t-2*t**2+0.8*t**3,0,1)],-1)*255).astype(np.uint8)
        rgb=cmap[idx]; h,w,_=rgb.shape; qi=QImage(rgb.data,w,h,3*w,QImage.Format.Format_RGB888); pm=QPixmap.fromImage(qi)
        sc=pm.scaled(self.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)
        x=(self.width()-sc.width())//2; y=(self.height()-sc.height())//2; p.drawPixmap(x,y,sc)
        p.setPen(QColor(200,200,210)); p.setFont(QFont("Segoe UI",10)); p.drawText(10,20,"Fourier Power Spectrum"); p.end()

class ActivationHeatmap(QWidget):
    def __init__(self,p=None):
        super().__init__(p); self.setMinimumSize(300,150); self._a=None
    def update_data(self,a): self._a=a; self.update()
    def paintEvent(self,e):
        p=QPainter(self); p.fillRect(self.rect(),QColor(20,20,25))
        if self._a is None:
            p.setPen(QColor(80,80,90)); p.setFont(QFont("Segoe UI",11))
            p.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,"Waiting..."); p.end(); return
        a=self._a; H,W=a.shape; as2=np.nan_to_num(a,nan=0,posinf=1,neginf=0); an=np.clip(as2,0,1); idx=(an*255).astype(np.uint8)
        rgb=COLORMAPS["Viridis"]()[idx]; h,w,_=rgb.shape; qi=QImage(rgb.data,w,h,3*w,QImage.Format.Format_RGB888); pm=QPixmap.fromImage(qi)
        sc=pm.scaled(self.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)
        x=(self.width()-sc.width())//2; y=(self.height()-sc.height())//2; p.drawPixmap(x,y,sc)
        p.setPen(QColor(200,200,210)); p.setFont(QFont("Segoe UI",10)); p.drawText(10,20,f"Activations ({H}x{W})"); p.end()

class NetworkDiagram(QWidget):
    def __init__(self,p=None):
        super().__init__(p); self.setMinimumSize(300,150); self._o=None
    def update_data(self,o): self._o=o; self.update()
    def paintEvent(self,e):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.fillRect(self.rect(),QColor(20,20,25))
        if self._o is None:
            p.setPen(QColor(80,80,90)); p.setFont(QFont("Segoe UI",11))
            p.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,"Waiting..."); p.end(); return
        w,h=self.width(),self.height(); m=40; pw=w-2*m; ph=h-2*m; o=self._o; n=len(o); mn,mx=o.min(),o.max()
        if mx-mn<1e-9: mx=mn+1.0; bw=max(1,pw//n-1)
        for i in range(n):
            v=(o[i]-mn)/(mx-mn); bh=int(ph*v); x=m+int(pw*i/n); y=m+ph-bh
            gr=QLinearGradient(x,y,x,y+bh); gr.setColorAt(0,QColor(79,195,247)); gr.setColorAt(1,QColor(13,71,161))
            p.fillRect(x,y,bw,bh,QBrush(gr))
        p.setPen(QColor(200,200,210)); p.setFont(QFont("Segoe UI",10))
        p.drawText(10,20,f"Output ({n} neurons)"); p.drawText(10,h-10,f"range: [{mn:.2f}, {mx:.2f}]"); p.end()

class PCAPlot(QWidget):
    def __init__(self,p=None):
        super().__init__(p); self.setMinimumSize(300,150); self.nodes=[]; self.best=None; self._pca=None
    def update_data(self,nodes,best): self.nodes=[n for n in nodes if n.visits>0 and not n.is_pruned]; self.best=best; self._calc(); self.update()
    def _calc(self):
        if len(self.nodes)<2: self._pca=None; return
        ks=list(self.nodes[0].params.keys()); X=np.array([[n.params[k] for k in ks] for n in self.nodes])
        mn=np.mean(X,0); sd=np.std(X,0); sd[sd<1e-10]=1; Xs=(X-mn)/sd
        cov=np.cov(Xs,rowvar=False); ev,vec=np.linalg.eigh(cov); ix=np.argsort(ev)[::-1]; ev=ev[ix]; vec=vec[:,ix]
        if len(ev)>=2:
            pr=Xs@vec[:,:2]; tv=np.sum(ev); ve=ev[:2]/tv if tv>0 else np.array([0.5,0.5]); self._pca={"pr":pr,"ve":ve}
        else: self._pca=None
    def paintEvent(self,e):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.fillRect(self.rect(),QColor(20,20,25))
        w,h=self.width(),self.height(); m=50; pw=w-2*m; ph=h-2*m
        if self._pca is None or len(self.nodes)<2:
            p.setPen(QColor(80,80,90)); p.setFont(QFont("Segoe UI",11))
            p.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,"Need >=2 active nodes"); p.end(); return
        pr=self._pca["pr"]; ve=self._pca["ve"]
        x0,x1=pr[:,0].min(),pr[:,0].max(); y0,y1=pr[:,1].min(),pr[:,1].max()
        xp=(x1-x0)*0.1 if x1>x0 else 1; yp=(y1-y0)*0.1 if y1>y0 else 1; x0-=xp; x1+=xp; y0-=yp; y1+=yp
        p.setPen(QPen(QColor(40,40,50),1,Qt.PenStyle.DotLine))
        for i in range(5):
            y=m+int(ph*i/4); p.drawLine(m,y,m+pw,y); x=m+int(pw*i/4); p.drawLine(x,m,x,m+ph)
        ls=[n.mean_loss for n in self.nodes]; lmn,lmx=min(ls),max(ls)
        if lmx-lmn<1e-9: lmx=lmn+1
        for i,n in enumerate(self.nodes):
            cx=m+int(pw*(pr[i,0]-x0)/(x1-x0)); cy=m+ph-int(ph*(pr[i,1]-y0)/(y1-y0))
            t=1.0-(n.mean_loss-lmn)/(lmx-lmn)
            c=QColor(int(239*(1-t)+102*t),int(83*(1-t)+187*t),int(80*(1-t)+106*t))
            sz=max(4,min(10,3+n.visits//3)); p.setBrush(QBrush(c)); p.setPen(QPen(QColor(255,255,255,80),1)); p.drawEllipse(QPoint(cx,cy),sz,sz)
        if self.best:
            bi=next((i for i,n in enumerate(self.nodes) if n.id==self.best.id),None)
            if bi is not None:
                cx=m+int(pw*(pr[bi,0]-x0)/(x1-x0)); cy=m+ph-int(ph*(pr[bi,1]-y0)/(y1-y0))
                gl=QRadialGradient(cx,cy,25); gl.setColorAt(0,QColor(102,187,106,200)); gl.setColorAt(1,QColor(102,187,106,0))
                p.setBrush(QBrush(gl)); p.setPen(Qt.PenStyle.NoPen); p.drawEllipse(QPoint(cx,cy),25,25)
                p.setBrush(QColor(102,187,106)); p.setPen(QPen(QColor(255,255,255),2)); p.drawEllipse(QPoint(cx,cy),8,8)
                p.setPen(QColor(255,255,255)); p.setFont(QFont("Segoe UI",9,QFont.Weight.Bold)); p.drawText(cx+14,cy-6,"BEST")
        p.setFont(QFont("Segoe UI",9)); p.setPen(QColor(180,180,190))
        p.drawText(10,20,f"Active: {len(self.nodes)}"); p.drawText(10,35,f"Var: {(ve[0]+ve[1])*100:.1f}%"); p.end()


class FFNPCAPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        self._projections = None; self._labels = None
        self._variance = [0,0,0,0]; self._num_classes = 0
    def update_data(self, projections, labels, variance, num_classes):
        self._projections = projections; self._labels = labels
        self._variance = list(variance) if len(variance) >= 4 else list(variance) + [0]*(4-len(variance))
        self._num_classes = num_classes; self.update()
    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(20, 20, 25))
        w, h = self.width(), self.height()
        if self._projections is None or self._labels is None:
            p.setPen(QColor(80, 80, 90)); p.setFont(QFont("Segoe UI", 11))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for MCTS evaluation...")
            p.end(); return
        margin = 40; scatter_h = int(h * 0.65); bar_h = h - scatter_h - 20
        sw = w - 2 * margin; sh = scatter_h - 2 * margin
        p.setPen(QPen(QColor(40, 40, 50), 1, Qt.PenStyle.DotLine))
        for i in range(5):
            y = margin + int(sh * i / 4); p.drawLine(margin, y, margin + sw, y)
            x = margin + int(sw * i / 4); p.drawLine(x, margin, x, margin + sh)
        proj = self._projections; labels = self._labels
        pc1 = proj[:, 0]; pc2 = proj[:, 1]
        x_min, x_max = float(pc1.min()), float(pc1.max())
        y_min, y_max = float(pc2.min()), float(pc2.max())
        xp = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
        yp = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        x_min -= xp; x_max += xp; y_min -= yp; y_max += yp
        if len(labels) > 1:
            p.setPen(QPen(QColor(150, 150, 160, 60), 1, Qt.PenStyle.DashLine))
            for i in range(len(labels) - 1):
                cx1 = margin + int(sw * (pc1[i] - x_min) / (x_max - x_min))
                cy1 = margin + sh - int(sh * (pc2[i] - y_min) / (y_max - y_min))
                cx2 = margin + int(sw * (pc1[i+1] - x_min) / (x_max - x_min))
                cy2 = margin + sh - int(sh * (pc2[i+1] - y_min) / (y_max - y_min))
                p.drawLine(cx1, cy1, cx2, cy2)
        for i in range(len(labels)):
            cx = margin + int(sw * (pc1[i] - x_min) / (x_max - x_min))
            cy = margin + sh - int(sh * (pc2[i] - y_min) / (y_max - y_min))
            c_idx = int(labels[i]) % len(CLASS_COLORS)
            p.setBrush(QBrush(CLASS_COLORS[c_idx])); p.setPen(QPen(QColor(255, 255, 255, 30), 1))
            p.drawEllipse(QPoint(cx, cy), 3, 3)
        p.setPen(QColor(150, 150, 160)); p.setFont(QFont("Segoe UI", 9))
        p.drawText(margin + sw // 2 - 30, scatter_h - 5, "PC1")
        p.save(); p.translate(12, margin + sh // 2); p.rotate(-90); p.drawText(0, 0, "PC2"); p.restore()
        p.setPen(QColor(79, 195, 247)); p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        p.drawText(10, 18, "FFN Weight PCA (V-Field) — Emergent Properties")
        bar_top = scatter_h + 5; bar_area_h = bar_h - 10
        bar_w = max(20, (w - 2 * margin) // 6); bar_spacing = bar_w + 10; bar_start_x = margin + 20
        p.setPen(QColor(200, 200, 210)); p.setFont(QFont("Segoe UI", 9))
        p.drawText(bar_start_x - 10, bar_top + 12, "Explained Variance:")
        max_var = max(self._variance) if max(self._variance) > 0 else 1.0
        bar_colors = [QColor(79,195,247), QColor(129,199,132), QColor(255,213,79), QColor(239,154,154)]
        for i in range(4):
            v = self._variance[i]; bh = int(bar_area_h * v / max_var) if max_var > 0 else 0
            bx = bar_start_x + i * bar_spacing; by = bar_top + bar_area_h - bh + 15
            grad = QLinearGradient(bx, by, bx, by + bh)
            grad.setColorAt(0, bar_colors[i]); grad.setColorAt(1, bar_colors[i].darker(150))
            p.setBrush(QBrush(grad)); p.setPen(QPen(QColor(255,255,255,60), 1))
            p.drawRect(bx, by, bar_w, bh)
            p.setPen(QColor(200,200,210)); p.setFont(QFont("Consolas", 8))
            p.drawText(bx, bar_top + bar_area_h + 28, f"PC{i+1}"); p.drawText(bx, by - 4, f"{v*100:.1f}%")
        total = sum(self._variance)
        p.setPen(QColor(150,150,160)); p.setFont(QFont("Consolas", 9))
        p.drawText(bar_start_x + 4 * bar_spacing + 10, bar_top + 25, f"Total: {total*100:.1f}%")
        p.end()


# ==== MCTS Tree View ====
class MCTSNodeItem(QGraphicsEllipseItem):
    def __init__(self, node, parent=None):
        super().__init__(parent); self.mcts_node=node; self._radius=10
        self._is_best=self._is_selected=self._is_hovered=False; self._on_path=False
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True); self.setAcceptHoverEvents(True); self._update_rect()
    def _update_rect(self): self.setRect(-self._radius,-self._radius,self._radius*2,self._radius*2)
    def update_appearance(self, l_min, l_max, is_best):
        n=self.mcts_node; self._is_best=is_best
        if n.is_pruned:
            self._radius=4; self.setBrush(QBrush(QColor(35,35,42,120))); self._update_rect()
            self.setPen(QPen(QColor(50,50,60,80),1,Qt.PenStyle.DashLine)); return
        if n.visits==0: self._radius=5; self.setBrush(QBrush(QColor(45,45,55)))
        else:
            t=1.0-(n.mean_loss-l_min)/max(l_max-l_min,1e-9); t=max(0.0,min(1.0,t))
            r,g,b=int(239*(1-t)+102*t),int(83*(1-t)+187*t),int(80*(1-t)+106*t)
            self._radius=max(7,min(16,7+np.log2(max(1,n.visits))*2.5))
            grad=QRadialGradient(-self._radius*0.35,-self._radius*0.35,self._radius*1.8)
            grad.setColorAt(0,QColor(min(255,r+60),min(255,g+60),min(255,b+60))); grad.setColorAt(1,QColor(r,g,b))
            self.setBrush(QBrush(grad))
        self._update_rect(); self._update_pen()
    def _update_pen(self):
        if self._is_selected: self.setPen(QPen(QColor(79,195,247),3))
        elif self._is_best: self.setPen(QPen(QColor(255,215,0),3))
        elif self._on_path: self.setPen(QPen(QColor(129,199,132,200),2))
        elif self._is_hovered: self.setPen(QPen(QColor(79,195,247,200),2))
        elif self.mcts_node.is_pruned: self.setPen(QPen(QColor(50,50,60,80),1,Qt.PenStyle.DashLine))
        elif self.mcts_node.visits>0: self.setPen(QPen(QColor(255,255,255,100),1))
        else: self.setPen(QPen(QColor(60,60,70),1))
    def set_selected(self, s): self._is_selected=s; self._update_pen()
    def set_on_path(self, on): self._on_path=on; self._update_pen()
    def hoverEnterEvent(self, e): self._is_hovered=True; self._update_pen(); super().hoverEnterEvent(e)
    def hoverLeaveEvent(self, e): self._is_hovered=False; self._update_pen(); super().hoverLeaveEvent(e)
    def mousePressEvent(self, e):
        if e.button()==Qt.MouseButton.LeftButton: self.setSelected(True)
        super().mousePressEvent(e)

class MCTSConnectionItem(QGraphicsPathItem):
    def __init__(self, x1,y1,x2,y2, visits=0, max_visits=1, is_pruned=False, on_path=False, parent=None):
        super().__init__(parent); path=QPainterPath(); mid_y=(y1+y2)/2
        path.moveTo(x1,y1); path.cubicTo(x1,mid_y,x2,mid_y,x2,y2); self.setPath(path)
        if is_pruned: self.setPen(QPen(QColor(35,35,42,25),0.5,Qt.PenStyle.DashLine))
        elif on_path: self.setPen(QPen(QColor(129,199,132,180),2.5))
        elif visits>0 and max_visits>0:
            r=visits/max_visits; self.setPen(QPen(QColor(55,55,75,int(50+120*r)),0.8+2.0*r))
        else: self.setPen(QPen(QColor(55,55,75,35),0.6))
        self.setZValue(-1)

class MCTSTreeView(QGraphicsView):
    node_clicked=Signal(object); V_SPACING=75; H_UNIT=38
    def __init__(self, parent=None):
        super().__init__(parent); self._scene=QGraphicsScene(self); self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing); self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag); self._node_items={}; self._selected_item=None
        self._path_node_ids=set()
    def update_tree(self, root, best_node, selected_node=None):
        self._scene.clear(); self._node_items.clear(); self._selected_item=None
        if root is None:
            txt=self._scene.addText("MCTS tree empty\n\nStart MCTS to build the tree", QFont("Segoe UI",11))
            txt.setDefaultTextColor(QColor(80,80,90)); return
        self._compute_widths(root,0); total_w=getattr(root,"_lw",1.0)
        x_span=max(total_w*self.H_UNIT,200); self._assign_pos(root,0,0,x_span)
        all_nodes=self._flatten(root)
        visited=[n for n in all_nodes if n.visits>0 and not n.is_pruned]
        if visited:
            losses=[n.mean_loss for n in visited]
            l_min, l_max = min(losses), max(losses)
            if l_max - l_min < 1e-9: l_max = l_min + 1.0
            max_visits = max(n.visits for n in visited)
        else:
            l_min, l_max, max_visits = 0.0, 1.0, 1
        self._path_node_ids = self._compute_path(root, selected_node or best_node)
        max_depth=max(getattr(n,"_ld",0) for n in all_nodes)
        for d in range(max_depth+1):
            y=d*self.V_SPACING
            self._scene.addLine(-2000,y,2000,y,QPen(QColor(26,26,36),1,Qt.PenStyle.DotLine)).setZValue(-3)
            lbl=self._scene.addText(f"d={d}",QFont("Consolas",8)); lbl.setDefaultTextColor(QColor(45,45,55)); lbl.setPos(-2060,y-8)
        self._draw_connections(root,max_visits); self._draw_nodes(root,l_min,l_max,best_node)
        rect=self._scene.itemsBoundingRect().adjusted(-40,-20,40,20); self._scene.setSceneRect(rect)
        self.fitInView(rect,Qt.AspectRatioMode.KeepAspectRatio)
    def _compute_path(self, root, target):
        if target is None: return set()
        path_ids = set(); cur = target
        while cur is not None: path_ids.add(cur.id); cur = cur.parent
        return path_ids
    def _compute_widths(self, root, depth):
        stack = [(root, depth)]
        while stack:
            node, d = stack.pop()
            node._ld = d
            if not node.children:
                node._lw = 1.0
            else:
                for c in node.children:
                    stack.append((c, d + 1))
        all_nodes = self._flatten(root)
        for n in reversed(all_nodes):
            if not n.children:
                n._lw = 1.0
            else:
                n._lw = max(sum(getattr(c, '_lw', 1.0) for c in n.children), 1.0)
        return root._lw
    def _assign_pos(self, root, x, y, x_span):
        stack = [(root, x, y, x_span)]
        while stack:
            node, nx, ny, ns = stack.pop()
            node._lx = nx; node._ly = ny
            if not node.children: continue
            total_w = sum(getattr(c, "_lw", 1.0) for c in node.children)
            cx = nx - ns / 2
            for child in node.children:
                cw = getattr(child, "_lw", 1.0)
                cs = ns * cw / total_w if total_w > 0 else ns / len(node.children)
                stack.append((child, cx + cs / 2, ny + self.V_SPACING, cs))
                cx += cs
    def _flatten(self, root):
        ns=[]; stack=[root]
        while stack: n=stack.pop(); ns.append(n); stack.extend(n.children)
        return ns
    def _draw_connections(self, root, max_visits):
        stack = [root]
        while stack:
            node = stack.pop()
            x = getattr(node, "_lx", 0); y = getattr(node, "_ly", 0)
            for child in node.children:
                cx = getattr(child, "_lx", 0); cy = getattr(child, "_ly", 0)
                on_path = child.id in self._path_node_ids and node.id in self._path_node_ids
                conn = MCTSConnectionItem(x, y, cx, cy, child.visits, max_visits, child.is_pruned, on_path)
                self._scene.addItem(conn)
                stack.append(child)
    def _draw_nodes(self, root, l_min, l_max, best_node):
        stack = [root]
        while stack:
            node = stack.pop()
            x = getattr(node, "_lx", 0); y = getattr(node, "_ly", 0)
            is_best = best_node is not None and node.id == best_node.id
            on_path = node.id in self._path_node_ids
            item = MCTSNodeItem(node); item.setPos(x, y)
            item.update_appearance(l_min, l_max, is_best); item.set_on_path(on_path)
            if node.is_pruned:
                tt = f"Node #{node.id}\nPRUNED: {node.prune_reason}\nVisits: {node.visits}\nDepth: {node.depth}"
            elif node.visits > 0:
                pca_str = ""
                if node.pca_variance and any(v > 0 for v in node.pca_variance):
                    pca_str = (f"\nPCA1: {node.pca_variance[0]*100:.1f}%  PCA2: {node.pca_variance[1]*100:.1f}%\n"
                               f"PCA3: {node.pca_variance[2]*100:.1f}%  PCA4: {node.pca_variance[3]*100:.1f}%\n"
                               f"Total PC1-4: {node.pca_total*100:.1f}%")
                tt = (f"Node #{node.id}\nMean Loss: {node.mean_loss:.4f} (Eval Std: {node.loss_std:.4f})\nLast Loss: {node.last_loss:.4f}\n"
                      f"Visits: {node.visits}\nDepth: {node.depth}\n"
                      f"F={node.params.get('F',0):.4f}  k={node.params.get('k',0):.4f}\n"
                      f"Du={node.params.get('Du',0):.4f}  Dv={node.params.get('Dv',0):.4f}\n"
                      f"--- Loss Landscape (Entire Dataset) ---\n"
                      f"Loss Std: {node.dataset_loss_std:.4f}  Max: {node.dataset_loss_max:.4f}\n"
                      f"Hidden Mean: {node.dataset_hidden_mean:.4f}  Std: {node.dataset_hidden_std:.4f}"
                      f"{pca_str}")
            else:
                tt = f"Node #{node.id}\nUnvisited\nDepth: {node.depth}"
            item.setToolTip(tt); self._scene.addItem(item); self._node_items[node.id] = item
            if node.visits > 0 and not node.is_pruned:
                lbl = QGraphicsTextItem(f"{node.mean_loss:.3f}"); lbl.setFont(QFont("Consolas", 7))
                lbl.setDefaultTextColor(QColor(200, 200, 210, 200)); lbl.setPos(x - 15, y + item._radius + 3)
                lbl.setZValue(1); self._scene.addItem(lbl)
                vl = QGraphicsTextItem(f"x{node.visits}"); vl.setFont(QFont("Consolas", 7))
                vl.setDefaultTextColor(QColor(129, 199, 132, 200)); vl.setPos(x + item._radius + 2, y - 6)
                vl.setZValue(1); self._scene.addItem(vl)
            stack.extend(node.children)
    def mousePressEvent(self, event):
        if event.button()==Qt.MouseButton.LeftButton:
            item=self.itemAt(event.position().toPoint())
            if isinstance(item,MCTSNodeItem):
                if self._selected_item: self._selected_item.set_selected(False)
                item.set_selected(True); self._selected_item=item; self.node_clicked.emit(item.mcts_node)
        super().mousePressEvent(event)
    def wheelEvent(self, event): self.scale(*(1.15,1.15) if event.angleDelta().y()>0 else (1/1.15,1/1.15))
    def drawBackground(self, painter, rect): painter.fillRect(rect,QColor(12,12,18))
    def drawForeground(self, painter, rect):
        painter.setPen(QColor(110,110,120)); painter.setFont(QFont("Segoe UI",9))
        painter.drawText(10,18,"Click to select | Scroll=zoom | Drag=pan")
        painter.drawText(10,33,"Green path=root to best | Gold=best | Dashed=pruned | Size=visits | Tooltip has PCA1-4")
        lx,ly,lw=10,48,120
        for i in range(lw):
            t=1.0-i/lw; r,g,b=int(239*(1-t)+102*t),int(83*(1-t)+187*t),int(80*(1-t)+106*t)
            painter.setPen(QColor(r,g,b)); painter.drawLine(lx+i,ly,lx+i,ly+8)
        painter.setPen(QColor(120,120,130)); painter.setFont(QFont("Consolas",8))
        painter.drawText(lx,ly+22,"high loss"); painter.drawText(lx+lw-50,ly+22,"low loss")


# ==============================================================================
# MAIN WINDOW
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("\u2b22 RD Neural Network \u2014 MCTS + Pruning + FFN PCA")
        self.resize(1800,1050)
        self.sim=GrayScott(AppConstants.DEFAULT_WIDTH,AppConstants.DEFAULT_HEIGHT,use_gpu=HAS_CUPY)
        self.network=FeedForwardNetwork(self.sim); self.mcts=MCTS(self.sim,self.network)
        self.frame_count=0
        self._last_fps_time=time.perf_counter(); self._last_fps_frames=0; self._fps=0.0
        self._iters_per_tick=1; self._benchmark_dataset=None
        self._current_dataset_name = "Digits (64F, 10C)" if HAS_SKLEARN else "Blobs (64F, 4C)"
        self._split_ratio=AppConstants.DEFAULT_SPLIT; self._max_samples=AppConstants.DEFAULT_MAX_SAMPLES
        self._selected_node = None
        self._worker_thread = None; self._worker = None
        self._cached_spectrum = None; self._spectrum_params_key = None
        self._show_u_field = False
        self._build_ui(); self._load_dataset(); self._refresh_best_display()
        self.ui_timer = QTimer(self); self.ui_timer.timeout.connect(self._on_ui_tick); self.ui_timer.start(100)
        self.status_bar=QStatusBar(self); self.setStatusBar(self.status_bar); self._update_status()

    def _build_ui(self):
        central=QWidget(); self.setCentralWidget(central)
        main_layout=QHBoxLayout(central); main_layout.setContentsMargins(0,0,0,0); main_layout.setSpacing(0)
        self._build_sidebar(main_layout)
        center=QWidget(); cl=QVBoxLayout(center); cl.setContentsMargins(8,8,8,8); cl.setSpacing(8)
        mr=QHBoxLayout(); mr.setSpacing(8)
        self.card_loss=MetricCard("Current Loss","\u2014","#ef5350"); self.card_best=MetricCard("Best Loss","\u2014","#66bb6a")
        self.card_acc=MetricCard("Accuracy","\u2014","#4fc3f7"); self.card_iters=MetricCard("MCTS Iters","0","#ce93d8")
        self.card_tree=MetricCard("Tree Size","0","#4dd0e1"); self.card_pruned=MetricCard("Pruned","0","#ef9a9a")
        for c in [self.card_loss,self.card_best,self.card_acc,self.card_iters,self.card_tree,self.card_pruned]: mr.addWidget(c)
        cl.addLayout(mr)
        top_split=QSplitter(Qt.Orientation.Horizontal)
        dp=QWidget(); dl=QVBoxLayout(dp); dl.setContentsMargins(0,0,0,0); dl.setSpacing(0)
        dh=QWidget(); dh.setStyleSheet("background-color: #14141a; padding: 8px 14px; border-radius: 6px 6px 0 0;")
        dhl=QHBoxLayout(dh); dhl.setContentsMargins(8,4,8,4)
        self.led_mcts=StatusLED("blue")
        self.lbl_dmode=QLabel("BEST V-FIELD (static)"); self.lbl_dmode.setStyleSheet("color: #4fc3f7; font-weight: 600; font-size: 11px;")
        dhl.addWidget(self.lbl_dmode); dhl.addStretch()
        self.btn_toggle_field = QPushButton("Show U"); self.btn_toggle_field.setCheckable(True)
        self.btn_toggle_field.setFixedWidth(80)
        self.btn_toggle_field.clicked.connect(self._on_toggle_field)
        dhl.addWidget(self.btn_toggle_field)
        self.combo_cmap = QComboBox(); self.combo_cmap.addItems(list(COLORMAPS.keys()))
        self.combo_cmap.setFixedWidth(90); self.combo_cmap.currentTextChanged.connect(self._on_cmap_changed)
        dhl.addWidget(QLabel("CMap:")); dhl.addWidget(self.combo_cmap)
        w_led=QWidget(); l_led=QHBoxLayout(w_led); l_led.setContentsMargins(6,0,6,0); l_led.setSpacing(4)
        l_led.addWidget(self.led_mcts); lb_led=QLabel("MCTS"); lb_led.setStyleSheet("color: #808090; font-size: 10px; font-weight: 600;"); l_led.addWidget(lb_led)
        dhl.addWidget(w_led)
        dl.addWidget(dh)
        self.display=DisplayWidget()
        self.display.setStyleSheet("background-color: #0a0a0e; border: 1px solid #2a2a35; border-top: none; border-radius: 0 0 6px 6px;")
        dl.addWidget(self.display,stretch=1); top_split.addWidget(dp)
        tp=QWidget(); tl=QVBoxLayout(tp); tl.setContentsMargins(0,0,0,0); tl.setSpacing(0)
        th=QWidget(); th.setStyleSheet("background-color: #14141a; padding: 8px 14px; border-radius: 6px 6px 0 0;")
        thl=QHBoxLayout(th); thl.setContentsMargins(8,4,8,4)
        lb=QLabel("MCTS TREE"); lb.setStyleSheet("color: #4fc3f7; font-weight: 600; font-size: 11px;")
        thl.addWidget(lb); thl.addStretch()
        self.lbl_sel=QLabel("No node selected"); self.lbl_sel.setStyleSheet("color: #808090; font-size: 10px;"); thl.addWidget(self.lbl_sel)
        tl.addWidget(th)
        self.tree_view=MCTSTreeView(); self.tree_view.node_clicked.connect(self._on_tree_node_clicked)
        self.tree_view.setStyleSheet("background-color: #0e0e14; border: 1px solid #2a2a35; border-top: none; border-radius: 0 0 6px 6px;")
        tl.addWidget(self.tree_view,stretch=1); top_split.addWidget(tp); top_split.setSizes([500,600]); cl.addWidget(top_split,stretch=2)
        self.tabs=QTabWidget()
        dash=QWidget(); dsl=QVBoxLayout(dash); dsl.setContentsMargins(14,14,14,14); dsl.setSpacing(12)
        self.lbl_st=QLabel("MCTS STATUS"); self.lbl_st.setStyleSheet("color: #4fc3f7; font-size: 12px; font-weight: 700; letter-spacing: 1px;")
        self.lbl_mstatus=QLabel("MCTS: idle"); self.lbl_mstatus.setWordWrap(True)
        self.lbl_mstatus.setStyleSheet("background-color: #1e1e26; padding: 12px; border-radius: 6px; color: #c0c0d0; font-family: 'Consolas', monospace; font-size: 11px; border: 1px solid #2a2a35;")
        dsl.addWidget(self.lbl_st); dsl.addWidget(self.lbl_mstatus); dsl.addStretch()
        self.tabs.addTab(dash,"Dashboard")
        self.loss_plot=LossPlot(); self.tabs.addTab(self.loss_plot,"Loss")
        self.phase_plot=PhaseSpacePlot(); self.tabs.addTab(self.phase_plot,"Phase Space")
        self.param_plot=ParamTSPlot(); self.tabs.addTab(self.param_plot,"Parameters")
        self.spectrum_plot=SpectrumPlot(); self.tabs.addTab(self.spectrum_plot,"Spectrum")
        self.activation_plot=ActivationHeatmap(); self.tabs.addTab(self.activation_plot,"Activations")
        self.network_plot=NetworkDiagram(); self.tabs.addTab(self.network_plot,"Network")
        self.pca_plot=PCAPlot(); self.tabs.addTab(self.pca_plot,"PCA Space")
        self.ffn_pca_plot=FFNPCAPlot(); self.tabs.addTab(self.ffn_pca_plot, "FFN PCA")
        
        # Top 3 Fields Tab
        self.top3_tab = QWidget()
        top3_layout = QHBoxLayout(self.top3_tab)
        self.top3_displays = [DisplayWidget() for _ in range(3)]
        self.top3_labels = [QLabel("") for _ in range(3)]
        for i in range(3):
            w = QWidget(); l = QVBoxLayout(w)
            self.top3_labels[i].setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.top3_labels[i].setStyleSheet("color: #4fc3f7; font-weight: 600; font-size: 11px; padding: 4px;")
            l.addWidget(self.top3_labels[i])
            l.addWidget(self.top3_displays[i])
            top3_layout.addWidget(w)
        self.tabs.addTab(self.top3_tab, "Top 3 Fields")
        
        vs=QSplitter(Qt.Orientation.Vertical)
        tw=QWidget(); tw.setLayout(QVBoxLayout()); tw.layout().setContentsMargins(0,0,0,0); tw.layout().addWidget(top_split); vs.addWidget(tw)
        tw2=QWidget(); tw2.setLayout(QVBoxLayout()); tw2.layout().setContentsMargins(0,0,0,0); tw2.layout().addWidget(self.tabs); vs.addWidget(tw2)
        vs.setSizes([600,300]); cl.addWidget(vs,stretch=1); main_layout.addWidget(center,stretch=1)
        self._build_toolbar(); self._connect_signals(); self._setup_shortcuts()

    def _build_sidebar(self, main_layout):
        self.sidebar=QScrollArea(); self.sidebar.setObjectName("sidebar"); self.sidebar.setWidgetResizable(True)
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar.setMinimumWidth(280); self.sidebar.setMaximumWidth(340)
        sc=QWidget(); sl=QVBoxLayout(sc); sl.setContentsMargins(0,0,0,0); sl.setSpacing(0)
        tw=QWidget(); tw.setStyleSheet("background-color: #0d0d12; padding: 14px;")
        tl=QVBoxLayout(tw); tl.setContentsMargins(14,10,14,10)
        lt=QLabel("\u2b22 RD Neural Network"); lt.setStyleSheet("color: #4fc3f7; font-size: 15px; font-weight: 700;")
        ls=QLabel("MCTS + Pruning + FFN PCA"); ls.setStyleSheet("color: #808090; font-size: 10px;")
        tl.addWidget(lt); tl.addWidget(ls); sl.addWidget(tw)
        def _sec(t,i): s=CollapsibleSection(t,i); sl.addWidget(s); return s
        def _wl(l): w=QWidget(); w.setLayout(l); w.setContentsMargins(0,0,0,0); return w
        def _dsb(mn,mx,d,st): sp=QDoubleSpinBox(); sp.setRange(mn,mx); sp.setDecimals(d); sp.setSingleStep(st); return sp
        s=_sec("DISPLAY","\u25d0")
        self.btn_eval=QPushButton("Evaluate Best"); self.btn_eval.setObjectName("primary"); s.add_widget(self.btn_eval)
        s=_sec("BEST PARAMS","\u2699")
        pg=QGridLayout(); pg.setSpacing(8)
        self.spin_F=_dsb(0.001,0.1,4,0.001); self.spin_k=_dsb(0.001,0.08,4,0.001)
        self.spin_Du=_dsb(0.1,2.0,3,0.01); self.spin_Dv=_dsb(0.05,1.5,3,0.01)
        self.spin_dt=_dsb(0.1,2.0,3,0.01)
        pg.addWidget(QLabel("F (feed)"),0,0); pg.addWidget(self.spin_F,0,1)
        pg.addWidget(QLabel("k (kill)"),1,0); pg.addWidget(self.spin_k,1,1)
        pg.addWidget(QLabel("Du"),2,0); pg.addWidget(self.spin_Du,2,1)
        pg.addWidget(QLabel("Dv"),3,0); pg.addWidget(self.spin_Dv,3,1)
        pg.addWidget(QLabel("dt"),4,0); pg.addWidget(self.spin_dt,4,1); s.add_widget(_wl(pg))
        s=_sec("DATASET","\U0001f4ca")
        bg=QGridLayout(); bg.setSpacing(6)
        self.combo_ds=QComboBox()
        for name, cfg in BENCHMARK_PRESETS.items():
            if cfg.source == "sklearn" and not HAS_SKLEARN: continue
            self.combo_ds.addItem(name)
        if HAS_SKLEARN: self.combo_ds.setCurrentText("Digits (64F, 10C)")
        bg.addWidget(QLabel("Dataset"),0,0); bg.addWidget(self.combo_ds,0,1); s.add_widget(_wl(bg))
        cl3=QVBoxLayout(); cl3.setSpacing(6)
        self.btn_load=QPushButton("\U0001f4c2 Load Custom CSV")
        self.lbl_ds=QLabel("Loading..."); self.lbl_ds.setStyleSheet("color: #808090; font-size: 10px; font-style: italic; padding: 4px;"); self.lbl_ds.setWordWrap(True)
        cl3.addWidget(self.btn_load); cl3.addWidget(self.lbl_ds); s.add_widget(_wl(cl3))
        dg=QGridLayout(); dg.setSpacing(6)
        self.slider_split=QSlider(Qt.Orientation.Horizontal); self.slider_split.setRange(50,95); self.slider_split.setValue(80)
        self.spin_max_samples=QSpinBox(); self.spin_max_samples.setRange(100,10000); self.spin_max_samples.setValue(5000)
        dg.addWidget(QLabel("Train Split %"),0,0); dg.addWidget(self.slider_split,0,1)
        dg.addWidget(QLabel("Max Samples"),1,0); dg.addWidget(self.spin_max_samples,1,1); s.add_widget(_wl(dg))
        if not HAS_SKLEARN:
            warn=QLabel("pip install scikit-learn for digits,\niris, wine, breast cancer datasets")
            warn.setStyleSheet("color: #ff9800; font-size: 10px; padding: 4px;"); warn.setWordWrap(True); s.add_widget(warn)
        s=_sec("NETWORK","\u229e")
        ng=QGridLayout(); ng.setSpacing(6)
        self.combo_input=QComboBox(); self.combo_input.addItems(["sine","random","impulse","gradient","step"])
        self.combo_act=QComboBox(); self.combo_act.addItems(["tanh","sigmoid","relu","leaky_relu"])
        self.spin_lambda=_dsb(1e-6,1e-1,6,1e-5); self.spin_lambda.setValue(AppConstants.DEFAULT_RIDGE_LAMBDA)
        ng.addWidget(QLabel("Input"),0,0); ng.addWidget(self.combo_input,0,1)
        ng.addWidget(QLabel("Activation"),1,0); ng.addWidget(self.combo_act,1,1)
        ng.addWidget(QLabel("Ridge \u03bb"),2,0); pg.addWidget(self.spin_lambda,2,1); s.add_widget(_wl(ng))
        s=_sec("MCTS SEARCH","\U0001f5b3")
        mr=QHBoxLayout()
        self.btn_mcts=QPushButton("Start MCTS"); self.btn_mcts.setObjectName("primary"); self.btn_mcts.setCheckable(True)
        self.btn_mcts_r=QPushButton("Reset")
        mr.addWidget(self.btn_mcts); mr.addWidget(self.btn_mcts_r); s.add_widget(_wl(mr))
        mg=QGridLayout(); mg.setSpacing(6)
        self.spin_warmup=QSpinBox(); self.spin_warmup.setRange(10,2000); self.spin_warmup.setValue(200)
        self.spin_children=QSpinBox(); self.spin_children.setRange(2,20); self.spin_children.setValue(6)
        self.spin_iters=QSpinBox(); self.spin_iters.setRange(1,50); self.spin_iters.setValue(1)
        self.spin_exp_c=_dsb(0.1,5.0,3,0.1); self.spin_exp_c.setValue(1.414)
        self.spin_perturb=_dsb(0.01,0.5,3,0.01); self.spin_perturb.setValue(0.08)
        self.spin_maxdepth=QSpinBox(); self.spin_maxdepth.setRange(5,50); self.spin_maxdepth.setValue(20)
        self.spin_conv_win=QSpinBox(); self.spin_conv_win.setRange(10,500); self.spin_conv_win.setValue(100)
        self.spin_conv_thr=_dsb(1e-6,1e-1,6,1e-5); self.spin_conv_thr.setValue(1e-4)
        self.spin_restart=QSpinBox(); self.spin_restart.setRange(10,500); self.spin_restart.setValue(80)
        self.spin_seed_r=QSpinBox(); self.spin_seed_r.setRange(2,50); self.spin_seed_r.setValue(10)
        self.spin_seed_c=QSpinBox(); self.spin_seed_c.setRange(1,50); self.spin_seed_c.setValue(10)
        r=0
        for lbl, w in [("Warmup",self.spin_warmup),("Children",self.spin_children),("Iters/batch",self.spin_iters),
                       ("Explo C",self.spin_exp_c),("Perturb",self.spin_perturb),("Max Depth",self.spin_maxdepth),
                       ("Conv Win",self.spin_conv_win),("Conv Thr",self.spin_conv_thr),
                       ("Restart",self.spin_restart),("Seed R",self.spin_seed_r),("Seed C",self.spin_seed_c)]:
            mg.addWidget(QLabel(lbl),r,0); mg.addWidget(w,r,1); r+=1
        s.add_widget(_wl(mg))
        s=_sec("PRUNING","\u2702")
        pg2=QGridLayout(); pg2.setSpacing(6)
        self.chk_prune=QCheckBox("Enable Loss Pruning"); self.chk_prune.setChecked(True); pg2.addWidget(self.chk_prune,0,0,1,2)
        self.spin_prune_min=QSpinBox(); self.spin_prune_min.setRange(2,30); self.spin_prune_min.setValue(5)
        self.spin_prune_z=_dsb(0.0,3.0,2,0.1); self.spin_prune_z.setValue(1.96)
        self.spin_prune_margin=_dsb(0.0,2.0,3,0.01); self.spin_prune_margin.setValue(0.0)
        pg2.addWidget(QLabel("Min Visits"),1,0); pg2.addWidget(self.spin_prune_min,1,1)
        pg2.addWidget(QLabel("Z-Score"),2,0); pg2.addWidget(self.spin_prune_z,2,1)
        pg2.addWidget(QLabel("Margin"),3,0); pg2.addWidget(self.spin_prune_margin,3,1)
        
        self.chk_pca_prune=QCheckBox("Enable PCA Pruning"); self.chk_pca_prune.setChecked(True); pg2.addWidget(self.chk_pca_prune,4,0,1,2)
        self.spin_pca_var_margin=_dsb(0.0,1.0,3,0.05); self.spin_pca_var_margin.setValue(0.1)
        self.spin_pca_redun=_dsb(0.5,1.0,3,0.01); self.spin_pca_redun.SetValue(0.98)
        pg2.addWidget(QLabel("Var Margin"),5,0); pg2.addWidget(self.spin_pca_var_margin,5,1)
        pg2.addWidget(QLabel("Redundancy Thr"),6,0); pg2.addWidget(self.spin_pca_redun,6,1)
        s.add_widget(_wl(pg2))
        
        s=_sec("EXPORT","\u2a53")
        er=QHBoxLayout()
        self.btn_exp_csv=QPushButton("CSV"); self.btn_exp_best=QPushButton("Best Params")
        self.btn_save_state=QPushButton("Save State"); self.btn_load_state=QPushButton("Load State")
        er.addWidget(self.btn_exp_csv); er.addWidget(self.btn_exp_best); s.add_widget(_wl(er))
        er2=QHBoxLayout(); er2.addWidget(self.btn_save_state); er2.addWidget(self.btn_load_state); s.add_widget(_wl(er2))
        sl.addStretch(); self.sidebar.setWidget(sc); main_layout.addWidget(self.sidebar)

    def _build_toolbar(self):
        tb=QToolBar(); tb.setMovable(False); self.addToolBar(Qt.ToolBarArea.TopToolBarArea,tb)
        def at(text,chk=False,slot=None):
            b=QToolButton(); b.setText(text); b.setCheckable(chk)
            if slot: b.clicked.connect(slot)
            tb.addWidget(b); return b
        self._tb_mcts=at("\U0001f5b3 MCTS",True,self._on_mcts_toggle)
        at("Reset",False,self._on_mcts_reset)
        at("Eval Best",False,self._on_eval_best); tb.addSeparator()
        at("? About",False,self._show_about)

    def _connect_signals(self):
        self.btn_eval.clicked.connect(self._on_eval_best)
        for sp in (self.spin_F,self.spin_k,self.spin_Du,self.spin_Dv,self.spin_dt):
            sp.valueChanged.connect(self._on_param)
        self.combo_ds.currentTextChanged.connect(self._on_ds_changed); self.btn_load.clicked.connect(self._on_load_ds)
        self.slider_split.valueChanged.connect(self._on_split_changed); self.spin_max_samples.valueChanged.connect(self._on_max_samples_changed)
        self.combo_input.currentTextChanged.connect(lambda v: setattr(self.network,'input_pattern',v))
        self.combo_act.currentTextChanged.connect(lambda v: setattr(self.network,'activation_fn',v))
        self.spin_lambda.valueChanged.connect(self._on_lambda)
        self.btn_mcts.clicked.connect(self._on_mcts_toggle)
        self.btn_mcts_r.clicked.connect(self._on_mcts_reset)
        self.spin_warmup.valueChanged.connect(lambda v: self._update_config(warmup_steps=v))
        self.spin_children.valueChanged.connect(lambda v: self._update_config(population_size=v))
        self.spin_iters.valueChanged.connect(self._on_iters_changed)
        self.spin_exp_c.valueChanged.connect(lambda v: self._update_config(exploration_constant=v))
        self.spin_perturb.valueChanged.connect(lambda v: self._update_config(perturb_scale=v))
        self.spin_maxdepth.valueChanged.connect(lambda v: self._update_config(max_tree_depth=v))
        self.spin_conv_win.valueChanged.connect(lambda v: self._update_config(convergence_window=v))
        self.spin_conv_thr.valueChanged.connect(lambda v: self._update_config(convergence_threshold=v))
        self.spin_restart.valueChanged.connect(lambda v: self._update_config(restart_threshold=v))
        self.spin_seed_r.valueChanged.connect(lambda v: self._update_config(seed_radius=v))
        self.spin_seed_c.valueChanged.connect(lambda v: self._update_config(seed_count=v))
        self.chk_prune.toggled.connect(lambda v: self._update_config(pruning_enabled=v))
        self.spin_prune_min.valueChanged.connect(lambda v: self._update_config(prune_min_visits=v))
        self.spin_prune_z.valueChanged.connect(lambda v: self._update_config(prune_z_score=v))
        self.spin_prune_margin.valueChanged.connect(lambda v: self._update_config(prune_margin=v))
        
        self.chk_pca_prune.toggled.connect(lambda v: self._update_config(pca_prune_enabled=v))
        self.spin_pca_var_margin.valueChanged.connect(lambda v: self._update_config(pca_variance_margin=v))
        self.spin_pca_redun.valueChanged.connect(lambda v: self._update_config(pca_redundancy_threshold=v))
        
        self.btn_exp_csv.clicked.connect(self._on_exp_csv); self.btn_exp_best.clicked.connect(self._on_exp_best)
        self.btn_save_state.clicked.connect(self._on_save_mcts_state)
        self.btn_load_state.clicked.connect(self._on_load_mcts_state)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("M"),self,self._on_mcts_toggle)
        QShortcut(QKeySequence("R"),self,self._on_mcts_reset)
        QShortcut(QKeySequence("E"),self,self._on_eval_best)
        QShortcut(QKeySequence("Space"),self,self._on_mcts_toggle)
        QShortcut(QKeySequence("U"),self,self._on_toggle_field)
        QShortcut(QKeySequence("Ctrl+S"),self,self._on_save_mcts_state)

    def _start_worker(self):
        if self._worker_thread is not None: return
        self._worker = MCTSWorker(self.mcts, lambda: self._benchmark_dataset, lambda: self.spin_warmup.value())
        self._worker.set_iters_per_batch(self._iters_per_tick)
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker.result_ready.connect(self._on_worker_result)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker_thread.started.connect(self._worker.run)
        self._worker_thread.start()

    def _stop_worker(self):
        if self._worker is not None: self._worker.stop()
        if self._worker_thread is not None:
            self._worker_thread.quit(); self._worker_thread.wait(3000)
        self._worker = None; self._worker_thread = None

    @Slot(dict)
    def _on_worker_result(self, info):
        self._update_mcts_status(info)
        if info.get("improved"):
            self._sync_spins(); self._refresh_best_display()

    @Slot()
    def _on_worker_finished(self):
        self.led_mcts.set_active(False)
        self.btn_mcts.setChecked(False); self.btn_mcts.setText("Start MCTS"); self._tb_mcts.setChecked(False)

    def _on_eval_best(self): self._refresh_best_display()
    def _on_param(self):
        self.sim.F=self.spin_F.value(); self.sim.k=self.spin_k.value()
        self.sim.Du=self.spin_Du.value(); self.sim.Dv=self.spin_Dv.value()
        self.sim.dt=self.spin_dt.value()
    def _on_lambda(self,v): self.network.ridge_lambda=v; self._update_config(ridge_lambda=v)
    def _on_toggle_field(self):
        self._show_u_field = not self._show_u_field
        self.btn_toggle_field.setChecked(self._show_u_field)
        self.btn_toggle_field.setText("Show V" if self._show_u_field else "Show U")
        self._refresh_best_display()
    def _on_cmap_changed(self, name):
        self.display.set_colormap(name)
        for d in self.top3_displays: d.set_colormap(name)
        self._refresh_best_display()
    def _load_dataset(self):
        name = self._current_dataset_name; cfg = BENCHMARK_PRESETS.get(name)
        if cfg is None:
            if self._benchmark_dataset is not None:
                ds = self._benchmark_dataset
                self.lbl_ds.setText(f"Custom: {ds.n_samples} samples | {ds.num_features}F {ds.num_classes}C | split {int(self._split_ratio*100)}%")
            return
        try:
            if cfg.source == "sklearn":
                if not HAS_SKLEARN:
                    self.lbl_ds.setText("pip install scikit-learn"); self.lbl_ds.setStyleSheet("color: #ef5350; font-size: 10px; padding: 4px;"); self._benchmark_dataset = None; return
                self._benchmark_dataset = SklearnDataset(cfg.name, self._split_ratio)
            else: self._benchmark_dataset = BenchmarkDataset(cfg, self._split_ratio)
            ds = self._benchmark_dataset
            src_tag = "sklearn" if cfg.source == "sklearn" else "synthetic"
            self.lbl_ds.setText(f"{cfg.name} ({src_tag})\n{ds.n_samples} samples | {ds.num_features}F {ds.num_classes}C\nsplit {int(self._split_ratio*100)}% | train={len(ds.y_train)} test={len(ds.y_test)}")
            self.lbl_ds.setStyleSheet("color: #66bb6a; font-size: 10px; padding: 4px;")
        except Exception as e:
            self.lbl_ds.setText(f"Error: {e}"); self.lbl_ds.setStyleSheet("color: #ef5350; font-size: 10px; padding: 4px;"); self._benchmark_dataset = None
    def _on_ds_changed(self, name):
        self._current_dataset_name = name; self._load_dataset(); self.mcts.reset(); self._on_mcts_reset()
    def _on_load_ds(self):
        fp,_=QFileDialog.getOpenFileName(self,"Load CSV","","CSV (*.csv)")
        if not fp: return
        try:
            cd=CustomBenchmarkDataset(fp,-1,self._max_samples,self._split_ratio); self._benchmark_dataset=cd
            self.combo_ds.blockSignals(True); self.combo_ds.setCurrentIndex(-1); self.combo_ds.blockSignals(False)
            self._current_dataset_name = ""
            self.lbl_ds.setText(f"\u2705 {Path(fp).name}\n{cd.n_samples} samples | {cd.num_features}F {cd.num_classes}C | split {int(self._split_ratio*100)}%")
            self.lbl_ds.setStyleSheet("color: #66bb6a; font-size: 10px; padding: 4px;")
            self.mcts.reset(); self._on_mcts_reset(); self.status_bar.showMessage(f"Loaded: {cd.n_samples} samples",4000)
        except ValueError as e:
            QMessageBox.critical(self,"Error",str(e))
            self.lbl_ds.setText("\u274c Failed"); self.lbl_ds.setStyleSheet("color: #ef5350; font-size: 10px; padding: 4px;")
    def _on_split_changed(self,v):
        self._split_ratio=v/100.0
        if self._current_dataset_name: self._load_dataset(); self.mcts.reset(); self._on_mcts_reset()
    def _on_max_samples_changed(self,v):
        self._max_samples=v
        if self._current_dataset_name: self._load_dataset(); self.mcts.reset(); self._on_mcts_reset()
    def _on_mcts_toggle(self):
        if self.mcts.enabled:
            self.mcts.enabled = False; self._stop_worker()
            self.btn_mcts.setChecked(False); self.btn_mcts.setText("Start MCTS")
            self._tb_mcts.setChecked(False); self.led_mcts.set_active(False)
        else:
            if self._benchmark_dataset is None:
                QMessageBox.warning(self,"No Dataset","Please select or load a dataset first.")
                self.btn_mcts.setChecked(False); return
            self.mcts.enabled = True; self.btn_mcts.setChecked(True)
            self.btn_mcts.setText("Stop MCTS"); self._tb_mcts.setChecked(True); self.led_mcts.set_active(True)
            self._start_worker()
    def _on_mcts_reset(self):
        was_running = self.mcts.enabled
        if was_running:
            self.mcts.enabled = False; self._stop_worker()
            self.btn_mcts.setChecked(False); self.btn_mcts.setText("Start MCTS")
            self._tb_mcts.setChecked(False); self.led_mcts.set_active(False)
        self.mcts.reset(); self._selected_node = None
        self._cached_spectrum = None; self._spectrum_params_key = None
        self._update_mcts_status({"step":0,"loss":0,"best_loss":0,"accuracy":0,"best_accuracy":0,
            "tree_size":0,"tree_depth":0,"node_visits":0,"node_depth":0,"status":"running",
            "steps_without_improvement":0,"restarts":0,"improved":False,"pruned_this_iter":0,
            "total_pruned":0,"total_visited":0,"prune_rate":0.0,
            "ds_loss_std":0.0, "ds_loss_max":0.0, "ds_hidden_mean":0.0, "ds_hidden_std":0.0,
            "pca_variance":[0,0,0,0], "pca_total":0.0})
        self.tree_view.update_tree(None,None,None); self._refresh_best_display()
    def _on_iters_changed(self,v):
        self._iters_per_tick = v
        if self._worker is not None: self._worker.set_iters_per_batch(v)
    def _update_config(self, **kwargs):
        self.mcts.request_config_update(replace(self.mcts.config, **kwargs))
    @Slot(object)
    def _on_tree_node_clicked(self, node):
        self._selected_node = node
        if node.is_pruned:
            self.lbl_dmode.setText(f"NODE #{node.id} PRUNED")
            self.lbl_sel.setText(f"Node #{node.id} | PRUNED: {node.prune_reason} | Visits: {node.visits}")
        elif node.visits>0:
            self.lbl_dmode.setText(f"SELECTED NODE #{node.id}  loss={node.last_loss:.4f}  acc={node.last_accuracy*100:.1f}%")
            self.lbl_sel.setText(f"Node #{node.id} | Loss: {node.mean_loss:.4f} | Visits: {node.visits} | F={node.params.get('F',0):.4f} k={node.params.get('k',0):.4f}")
            nc = 1 
            if node.pca_projections is not None:
                self.ffn_pca_plot.update_data(node.pca_projections, node.pca_labels, node.pca_variance, nc)
            elif self.mcts.best_node and self.mcts.best_node.pca_projections is not None:
                self.ffn_pca_plot.update_data(
                    self.mcts.best_node.pca_projections,
                    self.mcts.best_node.pca_labels,
                    self.mcts.best_node.pca_variance, nc)
        else:
            self.lbl_dmode.setText(f"SELECTED NODE #{node.id} (unvisited)")
            self.lbl_sel.setText(f"Node #{node.id} | Unvisited | Depth: {node.depth}")
        self._update_tree_viz()
    def _on_exp_csv(self):
        p,_=QFileDialog.getSaveFileName(self,"Export CSV","mcts.csv","CSV (*.csv)")
        if p: self.mcts.export_csv(p); self.status_bar.showMessage(f"Exported to {p}",3000)
    def _on_exp_best(self):
        p,_=QFileDialog.getSaveFileName(self,"Export Best","best.json","JSON (*.json)")
        if p: self.mcts.export_best(Path(p)); self.status_bar.showMessage(f"Exported to {p}",3000)
    def _on_save_mcts_state(self):
        p,_ = QFileDialog.getSaveFileName(self,"Save MCTS State","mcts_state.json","JSON (*.json)")
        if p:
            try: self.mcts.save_state(p); self.status_bar.showMessage(f"State saved to {p}",3000)
            except Exception as e: QMessageBox.critical(self,"Error",str(e))
    def _on_load_mcts_state(self):
        p,_ = QFileDialog.getOpenFileName(self,"Load MCTS State","","JSON (*.json)")
        if p:
            try:
                was_running = self.mcts.enabled
                if was_running: self._on_mcts_toggle()
                self.mcts.load_state(p); self._sync_spins(); self._refresh_best_display()
                self._update_plots(); self.status_bar.showMessage(f"State loaded from {p}",3000)
            except Exception as e: QMessageBox.critical(self,"Error",str(e))
    def _show_about(self):
        QMessageBox.about(self,"About",
            "\u2b22 RD Neural Network \u2014 MCTS + Pruning + FFN PCA\n\n"
            "Computes PCA1-PCA4 on feed-forward network weights (V-field)\n"
            "and extracts full dataset loss landscape statistics per node.\n\n"
            "F, k, Du, Dv all perturb by default.\n"
            "Pruning uses Loss stats AND PCA variance/redundancy.\n\n"
            "Click a tree node to see its PCA scatter and stats.\n"
            "MCTS runs in a background thread — no UI freeze.\n\n"
            "Shortcuts: M=toggle MCTS  R=reset\n"
            "           E=eval  U=toggle U/V field  Ctrl+S=save\n\n"
            "Datasets: Digits (default), Iris, Wine, Breast Cancer,\n"
            "Blobs, XOR, Linear, Circles, Moons, or custom CSV.\n\n"
            "pip install scikit-learn for built-in datasets")
    def _sync_spins(self):
        for sp,a in [(self.spin_F,"F"),(self.spin_k,"k"),(self.spin_Du,"Du"),(self.spin_Dv,"Dv"),(self.spin_dt,"dt")]:
            sp.blockSignals(True); sp.setValue(getattr(self.sim,a)); sp.blockSignals(False)

    def _on_ui_tick(self):
        self.frame_count += 1; self._update_plots(); self._update_status()
        if self.frame_count % AppConstants.PCA_CLEANUP_INTERVAL == 0:
            self.mcts.clear_pca_data(keep_node=self.mcts.best_node)

    def _refresh_best_display(self):
        ss=self.sim.save_state()
        try:
            self.sim.reset(); self.sim.set_params(self.mcts.best_params)
            self.sim.seed_random(self.mcts.config.seed_radius, self.mcts.config.seed_count)
            self.sim.step(self.spin_warmup.value())
            field = self.sim.get_u_np() if self._show_u_field else self.sim.get_v_np()
            field_name = "U-FIELD" if self._show_u_field else "V-FIELD"
            self.display.update_image(field)
            if self.mcts.best_loss < float('inf'):
                pca_str = ""
                if self.mcts.best_node and self.mcts.best_node.pca_variance:
                    pv = self.mcts.best_node.pca_variance
                    pca_str = f"\nPCA1: {pv[0]*100:.1f}%  PCA2: {pv[1]*100:.1f}%\nPCA1-4 total: {self.mcts.best_node.pca_total*100:.1f}%"
                self.display.set_info(
                    f"BEST {field_name}\n"
                    f"F={self.sim.F:.4f}  k={self.sim.k:.4f}\n"
                    f"Du={self.sim.Du:.2f}  Dv={self.sim.Dv:.2f}  dt={self.sim.dt:.2f}\n"
                    f"Loss: {self.mcts.best_loss:.4f}\n"
                    f"Acc: {self.mcts.best_accuracy*100:.1f}%"
                    f"{pca_str}")
                self.lbl_dmode.setText(f"BEST  loss={self.mcts.best_loss:.4f}  acc={self.mcts.best_accuracy*100:.1f}%")
                if self.mcts.best_node and self.mcts.best_node.pca_projections is not None:
                    nc = 1
                    self.ffn_pca_plot.update_data(self.mcts.best_node.pca_projections,
                                                  self.mcts.best_node.pca_labels,
                                                  self.mcts.best_node.pca_variance, nc)
            else:
                self.display.set_info("No best found yet\nStart MCTS to search")
            self.network.test()
            self._spectrum_params_key = None
            
            # Update Top 3 Displays
            for i in range(3):
                if i < len(self.mcts.top_nodes):
                    node = self.mcts.top_nodes[i]
                    ss2 = self.sim.save_state()
                    try:
                        self.sim.reset(); self.sim.set_params(node.params)
                        self.sim.seed_random(self.mcts.config.seed_radius, self.mcts.config.seed_count)
                        self.sim.step(self.spin_warmup.value())
                        v = self.sim.get_u_np() if self._show_u_field else self.sim.get_v_np()
                        self.top3_displays[i].update_image(v)
                        self.top3_labels[i].setText(f"#{i+1} | Loss: {node.mean_loss:.4f} | F={node.params['F']:.3f} k={node.params['k']:.3f}")
                    finally:
                        self.sim.restore_state(ss2)
                else:
                    self.top3_displays[i].update_image(np.zeros((256, 256)))
                    self.top3_labels[i].setText(f"#{i+1} | Waiting...")
                    
        finally:
            self.sim.restore_state(ss)

    def _update_tree_viz(self):
        self.tree_view.update_tree(self.mcts.root, self.mcts.best_node, self._selected_node)

    def _update_plots(self):
        loss_hist = list(self.mcts.loss_history)
        best_hist = list(self.mcts.best_loss_history)
        param_hist = list(self.mcts.param_history)
        best_params = dict(self.mcts.best_params)

        self.loss_plot.update_data(loss_hist, best_hist)
        self.phase_plot.update_data(param_hist, best_params)
        self.param_plot.update_data(param_hist)
        an=self.mcts.get_all_nodes(); self.pca_plot.update_data(an,self.mcts.best_node)
        if self.frame_count % AppConstants.TREE_UPDATE_INTERVAL == 0:
            self._update_tree_viz()
        if self.network.last_activations is not None: self.activation_plot.update_data(self.network.get_activations_normalized())
        if self.network.last_output is not None: self.network_plot.update_data(self.network.last_output)
        current_key = str(best_params)
        if current_key != self._spectrum_params_key:
            ss=self.sim.save_state()
            try:
                self.sim.reset(); self.sim.set_params(best_params)
                self.sim.seed_random(self.mcts.config.seed_radius,self.mcts.config.seed_count)
                self.sim.step(self.spin_warmup.value())
                v=self.sim.get_v_np()
                if np.all(np.isfinite(v)):
                    self._cached_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(v)))**2
                    self._spectrum_params_key = current_key
            finally: self.sim.restore_state(ss)
        if self._cached_spectrum is not None:
            self.spectrum_plot.update_data(self._cached_spectrum)

    def _update_mcts_status(self, info):
        pca_str = ""
        pv = info.get("pca_variance", [0,0,0,0])
        if pv and any(v > 0 for v in pv):
            pca_str = (f"\nPCA1: {pv[0]*100:.1f}%  PCA2: {pv[1]*100:.1f}%\n"
                       f"PCA3: {pv[2]*100:.1f}%  PCA4: {pv[3]*100:.1f}%\n"
                       f"Total PC1-4: {info.get('pca_total',0)*100:.1f}%")
        self.lbl_mstatus.setText(
            f"Step:                {info.get('step',0)}\n"
            f"Status:              {info.get('status','running').upper()}\n"
            f"Current Loss:        {info.get('loss',0):.6f}\n"
            f"Best Loss:           {info.get('best_loss',0):.6f}\n"
            f"Accuracy:            {info.get('accuracy',0)*100:.1f}%\n"
            f"Best Accuracy:       {info.get('best_accuracy',0)*100:.1f}%\n"
            f"Tree Size:           {info.get('tree_size',0)}\n"
            f"Tree Depth:          {info.get('tree_depth',0)}\n"
            f"Node Visits:         {info.get('node_visits',0)}\n"
            f"Node Depth:          {info.get('node_depth',0)}\n"
            f"No improvement:      {info.get('steps_without_improvement',0)}\n"
            f"Restarts:            {info.get('restarts',0)}\n"
            f"Pruned (this iter):  {info.get('pruned_this_iter',0)}\n"
            f"Total Pruned:        {info.get('total_pruned',0)}\n"
            f"Total Visited:       {info.get('total_visited',0)}\n"
            f"Prune Rate:          {info.get('prune_rate',0)*100:.1f}%\n"
            f"--- Loss Landscape (Entire Dataset) ---\n"
            f"Dataset Loss Std:    {info.get('ds_loss_std',0):.4f}\n"
            f"Dataset Loss Max:    {info.get('ds_loss_max',0):.4f}\n"
            f"Hidden Mean:         {info.get('ds_hidden_mean',0):.4f}\n"
            f"Hidden Std:          {info.get('ds_hidden_std',0):.4f}"
            f"{pca_str}")
        self.card_loss.set_value(f"{info.get('loss',0):.4f}")
        self.card_best.set_value(f"{info.get('best_loss',0):.4f}")
        self.card_acc.set_value(f"{info.get('accuracy',0)*100:.1f}%")
        self.card_iters.set_value(f"{info.get('step',0)}")
        self.card_tree.set_value(f"{info.get('tree_size',0)}")
        self.card_pruned.set_value(f"{info.get('total_pruned',0)}")

    def _update_status(self):
        now=time.perf_counter(); self._last_fps_frames+=1; el=now-self._last_fps_time
        if el>=1.0: self._fps=self._last_fps_frames/el; self._last_fps_time=now; self._last_fps_frames=0
        g="GPU" if self.sim.use_gpu else "CPU"; m="MCTS:ON" if self.mcts.enabled else "MCTS:OFF"
        d=f"DS:{self._benchmark_dataset.n_samples}" if self._benchmark_dataset else "DS:none"
        p=f"PRN:{self.mcts.total_pruned}" if self.mcts.config.pruning_enabled else ""
        thrd = "THREADED" if self._worker_thread is not None else ""
        fld = "U-FIELD" if self._show_u_field else "V-FIELD"
        self.status_bar.showMessage(f"Tick {self.frame_count}  |  {self._fps:.1f} FPS  |  {g}  |  {self.sim.width}x{self.sim.height}  |  {m} {p}  {d}  {thrd}  {fld}")

    def closeEvent(self, event):
        self._stop_worker(); super().closeEvent(event)


def create_example_dataset():
    rng=np.random.default_rng(42); X=rng.normal(size=(200,8))
    centers=rng.normal(scale=2.0,size=(3,8)); y=rng.integers(0,3,size=200); X+=centers[y]
    with open("example_dataset.csv","w") as f:
        f.write(",".join([f"f{i}" for i in range(8)]+["target"])+"\n")
        for row in np.hstack([X,y.reshape(-1,1)]): f.write(",".join(f"{v:.6f}" for v in row)+"\n")
    print("Created example_dataset.csv")

def main():
    app=QApplication(sys.argv); app.setApplicationName("RD_NN_MCTS_PCA"); app.setStyleSheet(MODERN_QSS)
    pal=QPalette()
    for role,val in [(QPalette.ColorRole.Window,QColor(26,26,31)),(QPalette.ColorRole.WindowText,QColor(228,228,232)),
                     (QPalette.ColorRole.Base,QColor(20,20,26)),(QPalette.ColorRole.Text,QColor(228,228,232)),
                     (QPalette.ColorRole.Button,QColor(42,42,53)),(QPalette.ColorRole.ButtonText,QColor(228,228,232)),
                     (QPalette.ColorRole.Highlight,QColor(13,71,161)),(QPalette.ColorRole.HighlightedText,QColor(255,255,255))]:
        pal.setColor(role,val)
    app.setPalette(pal); win=MainWindow(); win.show(); sys.exit(app.exec())

if __name__=="__main__":
    if not Path("example_dataset.csv").exists(): create_example_dataset()
    main()