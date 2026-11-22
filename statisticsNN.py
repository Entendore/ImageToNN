"""
Loss Term Study — Complete Research Platform
=============================================
A PySide6 application for studying loss terms in stochastic neural networks.

Features:
  • 20 loss terms across 6 categories
  • 12 visualization canvases
  • Multi-run comparison
  • Auto-weight optimizer
  • Phase scheduling (early/mid/late)
  • Per-term convergence detection
  • Session save/load (JSON)
  • Export suite (PNG/CSV)
  • Dark mode

No backpropagation — all learning via REINFORCE on scalar loss.
"""

from __future__ import annotations
import sys
import os
import time
import traceback
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QFont, QColor, QBrush, QPalette
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QLabel, QComboBox, QPushButton,
    QSpinBox, QDoubleSpinBox, QMessageBox, QProgressBar,
    QTextEdit, QSplitter, QStatusBar, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QToolBar, QLineEdit, QFormLayout, QScrollArea, QToolButton,
    QStyleFactory,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.max_open_warning'] = 0


# ===========================================================================
# ACTIVATIONS & HELPERS
# ===========================================================================

def relu(x): return np.maximum(0, x)
def tanh_(x): return np.tanh(x)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -80, 80)))
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
def identity(x): return x

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-8)

ACTIVATIONS = {"relu": relu, "tanh": tanh_, "sigmoid": sigmoid, "gelu": gelu, "identity": identity}
ACT_NAMES = list(ACTIVATIONS.keys())
N_ACTS = len(ACT_NAMES)


# ===========================================================================
# LOSS TERMS (20 terms)
# ===========================================================================

class LossTerm:
    name = "unnamed"
    category = "uncategorized"
    description = ""
    default_weight = 0.0
    default_enabled = False
    phase = "always"  # "always", "early", "mid", "late"

    def __init__(self):
        self.weight = self.default_weight
        self.enabled = self.default_enabled
        self.history: List[float] = []
        self.weighted_history: List[float] = []
        self.converged = False
        self.convergence_epoch: Optional[int] = None

    def compute(self, net, cache, L_task, epoch, total_epochs) -> float:
        raise NotImplementedError

    def record(self, raw: float, epoch: int):
        self.history.append(float(raw))
        self.weighted_history.append(float(raw * self.weight))
        self._check_convergence(epoch)

    def _check_convergence(self, epoch: int, window: int = 10, tol: float = 1e-4):
        if self.converged or len(self.history) < window + 2:
            return
        recent = np.array(self.history[-window:])
        if (recent.max() - recent.min()) < tol:
            self.converged = True
            self.convergence_epoch = epoch

    def reset(self):
        self.history.clear()
        self.weighted_history.clear()
        self.converged = False
        self.convergence_epoch = None


# --- Regularization ---
class KLPriorTerm(LossTerm):
    name = "KL to Prior"; category = "Regularization"
    description = "KL(q||p) where q=N(μ,σ²), p=N(0,1). Prevents drift."
    default_weight = 1e-3; default_enabled = True; phase = "always"
    def compute(self, net, cache, L_task, ep, tot):
        return sum(0.5*(np.exp(2*l.log_sig_W)+l.mu_W**2-1-2*l.log_sig_W).sum()
                   for l in net.layers if hasattr(l,'mu_W') and hasattr(l,'log_sig_W'))

class EntropyTerm(LossTerm):
    name = "Entropy"; category = "Regularization"
    description = "Negative entropy of weight distributions. Encourages exploration."
    default_weight = -0.01; default_enabled = False; phase = "early"
    def compute(self, net, cache, L_task, ep, tot):
        return sum((0.5*np.log(2*np.pi*np.e)+l.log_sig_W).sum()
                   for l in net.layers if hasattr(l,'log_sig_W'))

class L2Term(LossTerm):
    name = "L2 Decay"; category = "Regularization"
    description = "Σμ². Prevents weight explosion."
    default_weight = 1e-4; default_enabled = True; phase = "always"
    def compute(self, net, cache, L_task, ep, tot):
        return sum((l.mu_W**2).sum() for l in net.layers if hasattr(l,'mu_W'))

class L1Term(LossTerm):
    name = "L1 Sparsity"; category = "Regularization"
    description = "Σ|μ|. Encourages sparse weights."
    default_weight = 1e-4; default_enabled = False; phase = "late"
    def compute(self, net, cache, L_task, ep, tot):
        return sum(np.abs(l.mu_W).sum() for l in net.layers if hasattr(l,'mu_W'))

# --- Structural ---
class ActDiversityTerm(LossTerm):
    name = "Act Diversity"; category = "Structural"
    description = "Entropy of avg activation distribution. Prevents collapse."
    default_weight = -0.1; default_enabled = True; phase = "always"
    def compute(self, net, cache, L_task, ep, tot):
        t = 0.0
        for l in net.layers:
            if hasattr(l,'act_logits'):
                p = np.exp(l.act_logits - l.act_logits.max(-1,keepdims=True))
                p = p/(p.sum(-1,keepdims=True)+1e-8)
                avg = p.mean(axis=tuple(range(len(p.shape)-1)))
                t += -np.sum(avg*np.log(avg+1e-8))
        return t

class ActCoherenceTerm(LossTerm):
    name = "Act Coherence"; category = "Structural"
    description = "Large weights→unbounded act, small→bounded."
    default_weight = 0.05; default_enabled = False; phase = "mid"
    def compute(self, net, cache, L_task, ep, tot):
        t = 0.0
        for l in net.layers:
            if hasattr(l,'act_logits') and hasattr(l,'mu_W'):
                p = np.exp(l.act_logits-l.act_logits.max(-1,keepdims=True))
                p = p/(p.sum(-1,keepdims=True)+1e-8)
                wn = np.abs(l.mu_W)/(np.abs(l.mu_W).max()+1e-8)
                t += (wn*(p[...,1]+p[...,2])+(1-wn)*(p[...,0]+p[...,4])).sum()
        return t

class ActSaturationTerm(LossTerm):
    name = "Act Saturation"; category = "Structural"
    description = "Penalizes pre-activations in saturation zone."
    default_weight = 0.01; default_enabled = False; phase = "always"
    def compute(self, net, cache, L_task, ep, tot):
        if not cache: return 0.0
        return sum(np.maximum(0,np.abs(c.get('z',0))-2.0).sum() for c in cache if isinstance(c,dict))

class DeadNeuronTerm(LossTerm):
    name = "Dead Neurons"; category = "Structural"
    description = "Count of neurons with near-zero activity."
    default_weight = 0.01; default_enabled = False; phase = "mid"
    def compute(self, net, cache, L_task, ep, tot):
        if not cache: return 0.0
        return sum(float((np.abs(c.get('z',0)).mean(0)<1e-3).sum()) for c in cache if isinstance(c,dict))

class OrthogonalityTerm(LossTerm):
    name = "Orthogonality"; category = "Structural"
    description = "||W^TW - I||_F. Encourages orthogonal weights."
    default_weight = 0.001; default_enabled = False; phase = "always"
    def compute(self, net, cache, L_task, ep, tot):
        t = 0.0
        for l in net.layers:
            if hasattr(l,'mu_W'):
                W = l.mu_W
                if W.shape[0]>=W.shape[1]: g=W.T@W; I=np.eye(W.shape[1])
                else: g=W@W.T; I=np.eye(W.shape[0])
                t += ((g-I)**2).sum()
        return t

class WeightBalanceTerm(LossTerm):
    name = "Weight Balance"; category = "Structural"
    description = "Variance of mean |W| across layers."
    default_weight = 0.01; default_enabled = False; phase = "always"
    def compute(self, net, cache, L_task, ep, tot):
        m = [float(np.abs(l.mu_W).mean()) for l in net.layers if hasattr(l,'mu_W')]
        return float(np.var(m)) if len(m)>=2 else 0.0

# --- Stability ---
class ScoreVarianceTerm(LossTerm):
    name = "Score Variance"; category = "Stability"
    description = "Variance of REINFORCE scores. High=noisy."
    default_weight = 0.001; default_enabled = False; phase = "always"
    def compute(self, net, cache, L_task, ep, tot):
        return sum((1.0/(np.exp(2*l.log_sig_W)+1e-8)).mean() for l in net.layers if hasattr(l,'log_sig_W'))

class SpectralNormTerm(LossTerm):
    name = "Spectral Norm"; category = "Stability"
    description = "Approx largest singular value. Prevents explosion."
    default_weight = 0.001; default_enabled = False; phase = "always"
    def compute(self, net, cache, L_task, ep, tot):
        t = 0.0
        for l in net.layers:
            if hasattr(l,'mu_W'):
                u = np.random.randn(l.mu_W.shape[0]).astype(np.float32)
                v = l.mu_W.T@u; v=v/(np.linalg.norm(v)+1e-8)
                t += np.linalg.norm(l.mu_W@v)**2
        return t

class MomentumTerm(LossTerm):
    name = "Param Momentum"; category = "Stability"
    description = "||θ_t - θ_{t-1}||². Penalizes jumps."
    default_weight = 0.001; default_enabled = False; phase = "always"
    def __init__(self):
        super().__init__(); self._last = {}
    def compute(self, net, cache, L_task, ep, tot):
        t = 0.0; new = {}
        for i,l in enumerate(net.layers):
            if hasattr(l,'mu_W'):
                k = f"l{i}"; new[k] = l.mu_W.copy()
                if k in self._last: t += ((l.mu_W-self._last[k])**2).sum()
        self._last = new; return t

class LossSmoothTerm(LossTerm):
    name = "Loss Smooth"; category = "Stability"
    description = "(L_t - L_{t-1})². Penalizes erratic loss."
    default_weight = 0.1; default_enabled = False; phase = "always"
    def __init__(self):
        super().__init__(); self._last = None
    def compute(self, net, cache, L_task, ep, tot):
        if self._last is None: self._last=L_task; return 0.0
        v = (L_task-self._last)**2; self._last=L_task; return v

# --- Robustness ---
class ConsistencyTerm(LossTerm):
    name = "Consistency"; category = "Robustness"
    description = "Variance of pre-activations (proxy for prediction variance)."
    default_weight = 0.01; default_enabled = False; phase = "always"
    def compute(self, net, cache, L_task, ep, tot):
        if not cache: return 0.0
        return sum(float(c.get('z',np.zeros(1)).var()) for c in cache if isinstance(c,dict))

class AdversarialTerm(LossTerm):
    name = "Adversarial Sens."; category = "Robustness"
    description = "Σσ (higher σ = more sensitive to perturbation)."
    default_weight = 0.01; default_enabled = False; phase = "late"
    def compute(self, net, cache, L_task, ep, tot):
        return sum(np.exp(l.log_sig_W).sum() for l in net.layers if hasattr(l,'log_sig_W'))

# --- Information ---
class InterLayerCorrTerm(LossTerm):
    name = "Inter-Layer Corr"; category = "Information"
    description = "Correlation between adjacent layer weight patterns."
    default_weight = 0.01; default_enabled = False; phase = "mid"
    def compute(self, net, cache, L_task, ep, tot):
        t=0.0; prev=None
        for l in net.layers:
            if hasattr(l,'mu_W'):
                f=l.mu_W.ravel()
                if prev is not None and len(f)==len(prev):
                    t+=abs(np.corrcoef(f,prev)[0,1])
                prev=f
        return t

class InfoBottleneckTerm(LossTerm):
    name = "Info Bottleneck"; category = "Information"
    description = "-Σlog σ. Compresses representations."
    default_weight = 0.001; default_enabled = False; phase = "late"
    def compute(self, net, cache, L_task, ep, tot):
        return sum(-l.log_sig_W.sum() for l in net.layers if hasattr(l,'log_sig_W'))

class MDLTerm(LossTerm):
    name = "MDL"; category = "Information"
    description = "Bits to encode model. Simpler = better."
    default_weight = 1e-5; default_enabled = False; phase = "late"
    def compute(self, net, cache, L_task, ep, tot):
        return sum(np.log(1+np.abs(l.mu_W)).sum()+(-l.log_sig_W).sum()
                   for l in net.layers if hasattr(l,'mu_W') and hasattr(l,'log_sig_W'))

class LabelSmoothingTerm(LossTerm):
    name = "Label Smooth"; category = "Task"
    description = "Penalizes extreme logits (overconfidence)."
    default_weight = 0.1; default_enabled = False; phase = "always"
    def compute(self, net, cache, L_task, ep, tot):
        if not cache: return 0.0
        last = cache[-1] if cache else {}
        if not isinstance(last,dict) or 'z' not in last: return 0.0
        return float(np.maximum(0,np.abs(last['z'])-5.0).sum())


ALL_TERMS = [
    KLPriorTerm, EntropyTerm, L2Term, L1Term,
    ActDiversityTerm, ActCoherenceTerm, ActSaturationTerm, DeadNeuronTerm,
    OrthogonalityTerm, WeightBalanceTerm,
    ScoreVarianceTerm, SpectralNormTerm, MomentumTerm, LossSmoothTerm,
    ConsistencyTerm, AdversarialTerm,
    InterLayerCorrTerm, InfoBottleneckTerm, MDLTerm, LabelSmoothingTerm,
]
CATEGORIES = ["Regularization","Structural","Stability","Robustness","Information","Task"]
CAT_COLORS = {"Regularization":"#4C72B0","Structural":"#DD8452","Stability":"#55A868",
              "Robustness":"#C44E52","Information":"#8172B2","Task":"#937860"}

PRESETS = {
    "Minimal": {"KL to Prior":(True,1e-3),"L2 Decay":(True,1e-4)},
    "Balanced": {"KL to Prior":(True,1e-3),"L2 Decay":(True,1e-4),"Act Diversity":(True,-0.1),"Entropy":(True,-0.01)},
    "Sparse": {"KL to Prior":(True,1e-3),"L1 Sparsity":(True,1e-3),"Dead Neurons":(True,0.01),"MDL":(True,1e-5)},
    "Stable": {"KL to Prior":(True,1e-3),"L2 Decay":(True,1e-4),"Spectral Norm":(True,0.001),"Param Momentum":(True,0.001),"Loss Smooth":(True,0.1)},
    "Robust": {"KL to Prior":(True,1e-3),"Consistency":(True,0.01),"Adversarial Sens.":(True,0.01),"Label Smooth":(True,0.1),"Orthogonality":(True,0.001)},
    "Scheduled": {"Entropy":(True,-0.05),"Act Diversity":(True,-0.1),"Act Coherence":(True,0.05),
                  "L1 Sparsity":(True,1e-3),"Info Bottleneck":(True,0.001),"Adversarial Sens.":(True,0.01)},
    "All": {cls().name:(True,cls().default_weight) for cls in ALL_TERMS},
}


# ===========================================================================
# LOSS TERM MANAGER
# ===========================================================================

class LossTermManager:
    def __init__(self):
        self.terms: Dict[str, LossTerm] = {}
        for cls in ALL_TERMS:
            t = cls()
            self.terms[t.name] = t

    def apply_preset(self, name):
        for t in self.terms.values():
            t.enabled = False
            t.weight = t.default_weight
        if name in PRESETS:
            for n,(en,w) in PRESETS[name].items():
                if n in self.terms:
                    self.terms[n].enabled = en
                    self.terms[n].weight = w

    def is_active(self, term_name: str, epoch: int, total_epochs: int) -> bool:
        t = self.terms.get(term_name)
        if t is None or not t.enabled:
            return False
        phase = epoch / max(1, total_epochs)
        if t.phase == "always": return True
        if t.phase == "early": return phase < 0.33
        if t.phase == "mid": return 0.2 <= phase < 0.75
        if t.phase == "late": return phase >= 0.6
        return True

    def compute_total(self, net, cache, L_task, ep, tot) -> Tuple[float, Dict[str,float]]:
        bd = {"task": float(L_task)}
        L = float(L_task)
        for t in self.terms.values():
            if not self.is_active(t.name, ep, tot):
                bd[t.name] = 0.0
                bd[f"{t.name}*w"] = 0.0
                continue
            try:
                v = t.compute(net, cache, L_task, ep, tot)
                wv = t.weight * v
                L += wv
                bd[t.name] = float(v)
                bd[f"{t.name}*w"] = float(wv)
                t.record(v, ep)
            except Exception:
                bd[t.name] = 0.0
        return L, bd

    def reset(self):
        for t in self.terms.values():
            t.reset()

    def enabled_names(self):
        return [n for n,t in self.terms.items() if t.enabled]


# ===========================================================================
# SESSION PERSISTENCE
# ===========================================================================

@dataclass
class TrainingSession:
    name: str
    preset: str
    dataset: str
    network_arch: str
    epochs: int
    final_val_loss: float
    final_val_acc: float
    term_histories: Dict[str, List[float]]
    term_weights: Dict[str, float]
    term_enabled: Dict[str, bool]
    final_loss: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class SessionStore:
    def __init__(self):
        self.sessions: List[TrainingSession] = []

    def add(self, session: TrainingSession):
        self.sessions.append(session)

    def save_to_file(self, path: str):
        data = [s.to_dict() for s in self.sessions]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.sessions = [TrainingSession.from_dict(d) for d in data]


# ===========================================================================
# AUTO-WEIGHT OPTIMIZER
# ===========================================================================

class WeightOptimizer:
    def __init__(self, mgr: LossTermManager):
        self.mgr = mgr
        self.results: List[Dict] = []

    def run_random_search(self, n_configs: int, weight_range: Tuple[float,float] = (1e-5, 1e-1),
                          seed: int = 42) -> List[Dict]:
        rng = np.random.default_rng(seed)
        configs = []
        enabled = self.mgr.enabled_names()
        for i in range(n_configs):
            config = {}
            for name in enabled:
                log_w = rng.uniform(np.log10(weight_range[0]), np.log10(weight_range[1]))
                w = 10 ** log_w
                if rng.random() < 0.3:
                    w = -w
                config[name] = w
            configs.append(config)
        return configs

    def record_result(self, config: Dict, final_loss: float, final_acc: float):
        self.results.append({
            "config": config,
            "final_loss": final_loss,
            "final_acc": final_acc,
        })

    def best_config(self, metric: str = "final_loss") -> Optional[Dict]:
        if not self.results: return None
        best = min(self.results, key=lambda r: r[metric])
        return best["config"]


# ===========================================================================
# SIMPLE NETWORK
# ===========================================================================

class SimpleLayer:
    def __init__(self, fi, fo, rng):
        s = 0.1*np.sqrt(2.0/fi)
        self.mu_W = rng.normal(0,s,(fi,fo)).astype(np.float32)
        self.log_sig_W = np.full((fi,fo),np.log(s),dtype=np.float32)
        self.act_logits = np.zeros((fi,fo,N_ACTS),dtype=np.float32)
        self.act_logits[:,:,0]=1.0
        self.mu_b = np.zeros(fo,dtype=np.float32)
        self.log_sig_b = np.full(fo,np.log(0.05),dtype=np.float32)
        self.coupling = np.full(fo,0.3,dtype=np.float32)
        self.rng = rng

    def forward(self, x, training=True):
        eps = self.rng.standard_normal(self.mu_W.shape).astype(np.float32)
        W = self.mu_W + np.exp(self.log_sig_W)*eps
        b = self.mu_b + np.exp(self.log_sig_b)*self.rng.standard_normal(self.mu_b.shape).astype(np.float32)
        z = x@W+b
        a = relu(z)
        return a, {"W":W,"b":b,"z":z,"x":x}

    def reinforce_update(self, adv, cache, lr):
        W=cache["W"]; s2=np.exp(2*self.log_sig_W); d=W-self.mu_W
        self.mu_W -= lr*adv*d/(s2+1e-8)
        self.log_sig_W -= lr*adv*(d**2/(s2+1e-8)-1.0)
        sb2=np.exp(2*self.log_sig_b); db=cache["b"]-self.mu_b
        self.mu_b -= lr*adv*db/(sb2+1e-8)
        self.log_sig_b -= lr*adv*(db**2/(sb2+1e-8)-1.0)


class SimpleNet:
    def __init__(self, in_dim, sizes, out_dim, rng):
        self.layers = []
        dims = [in_dim]+sizes+[out_dim]
        for i in range(len(sizes)+1):
            self.layers.append(SimpleLayer(dims[i],dims[i+1],rng))

    def forward(self, X, training=True):
        a=X; caches=[]
        for l in self.layers:
            a,c=l.forward(a,training); caches.append(c)
        return a, caches

    def train_step(self, X, y_oh, lr):
        pred, caches = self.forward(X, True)
        probs = softmax(pred)
        L = float(-np.mean(np.sum(y_oh*np.log(probs+1e-12),axis=1)))
        if not hasattr(self,"_bl"): self._bl=L
        self._bl=0.9*self._bl+0.1*L; adv=L-self._bl
        for l,c in zip(self.layers,caches): l.reinforce_update(adv,c,lr)
        return L, caches


def load_data(name):
    import sklearn.datasets as ds
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    loaders={"iris":ds.load_iris,"digits":ds.load_digits,"wine":ds.load_wine,"breast_cancer":ds.load_breast_cancer}
    X,y=loaders[name](return_X_y=True)
    X=StandardScaler().fit_transform(X).astype(np.float32)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    cls=np.unique(y)
    return dict(X_train=Xtr,X_test=Xte,y_train=ytr,y_test=yte,
                y_train_oh=(ytr[:,None]==cls).astype(np.float32),
                y_test_oh=(yte[:,None]==cls).astype(np.float32),
                is_classification=True,n_classes=len(cls),input_features=Xtr.shape[1])


# ===========================================================================
# 12 VISUALIZATION CANVASES
# ===========================================================================

class TermEvolutionCanvas(FigureCanvas):
    def __init__(self, mgr):
        self.mgr=mgr; self.fig=Figure(figsize=(6,3.5),tight_layout=True); super().__init__(self.fig)
    def update(self, sessions: Optional[List[TrainingSession]] = None):
        self.fig.clear(); ax=self.fig.add_subplot(111)
        if sessions:
            for sess in sessions:
                for name, hist in sess.term_histories.items():
                    if hist:
                        ax.plot(range(1,len(hist)+1), hist,
                                label=f"{sess.name}/{name}", lw=1, alpha=0.7)
            ax.legend(fontsize=5, ncol=2, loc="upper right")
        else:
            enabled=[t for t in self.mgr.terms.values() if t.enabled and t.history]
            if not enabled:
                ax.text(0.5,0.5,"No data",ha="center",va="center"); ax.axis("off")
                self.draw_idle(); return
            cols=cm.tab20(np.linspace(0,1,max(len(enabled),1)))
            for i,t in enumerate(enabled):
                ax.plot(range(1,len(t.history)+1),t.history,label=t.name,color=cols[i],lw=1.3)
                if t.converged and t.convergence_epoch:
                    ax.axvline(t.convergence_epoch, color=cols[i], linestyle=":", alpha=0.5)
            ax.legend(fontsize=6,ncol=2,loc="upper right")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Raw value"); ax.set_title("Term Evolution")
        ax.grid(True,alpha=0.3)
        self.draw_idle()

class StackedAreaCanvas(FigureCanvas):
    def __init__(self, mgr):
        self.mgr=mgr; self.fig=Figure(figsize=(6,3.5),tight_layout=True); super().__init__(self.fig)
    def update(self):
        self.fig.clear(); ax=self.fig.add_subplot(111)
        enabled=[t for t in self.mgr.terms.values() if t.enabled and t.weighted_history]
        if not enabled:
            ax.text(0.5,0.5,"No data",ha="center",va="center"); ax.axis("off"); self.draw_idle(); return
        n=len(enabled[0].weighted_history); x=np.arange(1,n+1)
        cols=cm.Set3(np.linspace(0,1,max(len(enabled),1)))
        pos=np.zeros(n); neg=np.zeros(n)
        for i,t in enumerate(enabled):
            vals=np.array(t.weighted_history)
            p=np.where(vals>0,vals,0); ng=np.where(vals<0,vals,0)
            ax.fill_between(x,pos,pos+p,color=cols[i],alpha=0.7,label=t.name)
            pos+=p; ax.fill_between(x,neg+ng,neg,color=cols[i],alpha=0.4); neg+=ng
        ax.axhline(0,color="black",lw=0.8)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Weighted contribution"); ax.set_title("Stacked Contribution")
        ax.legend(fontsize=5,ncol=2,loc="upper right"); ax.grid(True,alpha=0.3)
        self.draw_idle()

class CorrelationHeatmapCanvas(FigureCanvas):
    def __init__(self, mgr):
        self.mgr=mgr; self.fig=Figure(figsize=(5,4),tight_layout=True); super().__init__(self.fig)
    def update(self):
        self.fig.clear(); ax=self.fig.add_subplot(111)
        enabled=[t for t in self.mgr.terms.values() if t.enabled and len(t.history)>5]
        if len(enabled)<2:
            ax.text(0.5,0.5,"Need ≥2 terms with >5 epochs",ha="center",va="center")
            ax.axis("off"); self.draw_idle(); return
        names=[t.name for t in enabled]
        data=np.array([t.history for t in enabled])
        min_len=min(len(t.history) for t in enabled)
        data=data[:,:min_len]
        corr=np.corrcoef(data)
        im=ax.imshow(corr,cmap="RdBu_r",vmin=-1,vmax=1,aspect="auto")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=45,ha="right",fontsize=6)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=6)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j,i,f"{corr[i,j]:.2f}",ha="center",va="center",fontsize=5,
                        color="white" if abs(corr[i,j])>0.5 else "black")
        plt.colorbar(im,ax=ax,shrink=0.8)
        ax.set_title("Term Correlation"); self.draw_idle()

class RadarCanvas(FigureCanvas):
    def __init__(self, mgr):
        self.mgr=mgr; self.fig=Figure(figsize=(5,4),tight_layout=True); super().__init__(self.fig)
    def update(self):
        self.fig.clear(); ax=self.fig.add_subplot(111,polar=True)
        enabled=[t for t in self.mgr.terms.values() if t.enabled and t.history]
        if not enabled:
            ax.text(0,0,"No data",ha="center",va="center"); self.draw_idle(); return
        names=[t.name for t in enabled]
        vals=[]
        for t in enabled:
            v=t.history[-1] if t.history else 0
            mn=min(t.history); mx=max(t.history)
            vals.append((v-mn)/(mx-mn+1e-12))
        N=len(names)
        angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist()
        vals+=vals[:1]; angles+=angles[:1]
        ax.plot(angles,vals,"o-",lw=2,color="#4C72B0")
        ax.fill(angles,vals,alpha=0.25,color="#4C72B0")
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(names,fontsize=6)
        ax.set_ylim(0,1); ax.set_title("Term Radar (normalized)",pad=20)
        self.draw_idle()

class SensitivityCanvas(FigureCanvas):
    def __init__(self, mgr):
        self.mgr=mgr; self.fig=Figure(figsize=(6,3),tight_layout=True); super().__init__(self.fig)
    def update(self):
        self.fig.clear(); ax=self.fig.add_subplot(111)
        enabled=[t for t in self.mgr.terms.values() if t.enabled and t.history]
        if not enabled:
            ax.text(0.5,0.5,"No data",ha="center",va="center"); ax.axis("off"); self.draw_idle(); return
        names=[t.name for t in enabled]
        sens=[]
        for t in enabled:
            if t.weighted_history and t.weight!=0:
                sens.append(abs(np.mean(t.weighted_history))/abs(t.weight))
            else: sens.append(0)
        colors=[CAT_COLORS.get(t.category,"#888") for t in enabled]
        ax.barh(names,sens,color=colors,alpha=0.8)
        ax.set_xlabel("Sensitivity"); ax.set_title("Weight Sensitivity"); ax.grid(True,alpha=0.3,axis="x")
        ax.tick_params(axis="y",labelsize=7)
        self.draw_idle()

class ViolinCanvas(FigureCanvas):
    def __init__(self, mgr):
        self.mgr=mgr; self.fig=Figure(figsize=(7,4),tight_layout=True); super().__init__(self.fig)
    def update(self):
        self.fig.clear()
        enabled=[t for t in self.mgr.terms.values() if t.enabled and len(t.history)>5]
        if not enabled:
            ax=self.fig.add_subplot(111)
            ax.text(0.5,0.5,"No data",ha="center",va="center"); ax.axis("off"); self.draw_idle(); return
        ax=self.fig.add_subplot(111)
        names=[t.name for t in enabled]
        data=[t.history for t in enabled]
        positions=np.arange(1,len(names)+1)
        parts=ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#4C72B0'); pc.set_alpha(0.5)
        ax.boxplot(data, positions=positions, widths=0.15,
                   boxprops=dict(color='black', alpha=0.7),
                   medianprops=dict(color='red', lw=1.5))
        ax.set_xticks(positions); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Term value"); ax.set_title("Distribution (violin + box)")
        ax.grid(True, alpha=0.3, axis="y")
        self.fig.tight_layout()
        self.draw_idle()

class CumulativeCanvas(FigureCanvas):
    def __init__(self, mgr):
        self.mgr=mgr; self.fig=Figure(figsize=(6,3.5),tight_layout=True); super().__init__(self.fig)
    def update(self):
        self.fig.clear(); ax=self.fig.add_subplot(111)
        enabled=[t for t in self.mgr.terms.values() if t.enabled and t.weighted_history]
        if not enabled:
            ax.text(0.5,0.5,"No data",ha="center",va="center"); ax.axis("off"); self.draw_idle(); return
        cols=cm.tab20(np.linspace(0,1,max(len(enabled),1)))
        for i,t in enumerate(enabled):
            cum=np.cumsum(t.weighted_history)
            ax.plot(range(1,len(cum)+1),cum,label=t.name,color=cols[i],lw=1.5)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Cumulative contribution")
        ax.set_title("Cumulative Impact"); ax.legend(fontsize=6,ncol=2)
        ax.grid(True,alpha=0.3); ax.axhline(0,color="black",lw=0.6)
        self.draw_idle()

class Phase3DCanvas(FigureCanvas):
    def __init__(self, mgr):
        self.mgr=mgr; self.fig=Figure(figsize=(6,5),tight_layout=True); super().__init__(self.fig)
    def update(self):
        self.fig.clear()
        enabled=[t for t in self.mgr.terms.values() if t.enabled and len(t.history)>5]
        if not enabled:
            ax=self.fig.add_subplot(111)
            ax.text(0.5,0.5,"No data",ha="center",va="center"); ax.axis("off"); self.draw_idle(); return
        ax=self.fig.add_subplot(111, projection='3d')
        cols=cm.tab20(np.linspace(0,1,max(len(enabled),1)))
        for i,t in enumerate(enabled):
            vals=np.array(t.history)
            grads=np.diff(vals); grads=np.concatenate([[0],grads])
            accels=np.diff(grads); accels=np.concatenate([[0],accels])
            ax.scatter(vals, grads, accels, c=[cols[i]]*len(vals), s=6, alpha=0.6, label=t.name)
            ax.plot(vals, grads, accels, color=cols[i], alpha=0.3, lw=0.8)
        ax.set_xlabel("Value", fontsize=8); ax.set_ylabel("Gradient", fontsize=8)
        ax.set_zlabel("Acceleration", fontsize=8)
        ax.set_title("3D Phase Portrait", fontsize=10)
        ax.legend(fontsize=6, loc="upper left")
        ax.view_init(elev=20, azim=45)
        self.draw_idle()

class InteractionGraphCanvas(FigureCanvas):
    def __init__(self, mgr):
        self.mgr=mgr; self.fig=Figure(figsize=(6,5),tight_layout=True); super().__init__(self.fig)
    def update(self):
        self.fig.clear(); ax=self.fig.add_subplot(111)
        enabled=[t for t in self.mgr.terms.values() if t.enabled and len(t.history)>5]
        if len(enabled)<3:
            ax.text(0.5,0.5,"Need ≥3 terms",ha="center",va="center")
            ax.axis("off"); self.draw_idle(); return
        names=[t.name for t in enabled]
        data=np.array([t.history for t in enabled])
        min_len=min(len(t.history) for t in enabled)
        data=data[:,:min_len]
        corr=np.corrcoef(data)
        n=len(names)
        np.random.seed(42)
        pos=np.random.rand(n,2)
        for _ in range(100):
            forces=np.zeros_like(pos)
            for i in range(n):
                for j in range(i+1,n):
                    d=pos[i]-pos[j]; dist=np.linalg.norm(d)+1e-6
                    f_rep = 0.01 / (dist**2)
                    forces[i] += f_rep * d/dist
                    forces[j] -= f_rep * d/dist
                    if abs(corr[i,j]) > 0.3:
                        f_att = 0.005 * abs(corr[i,j]) * dist
                        forces[i] -= f_att * d/dist
                        forces[j] += f_att * d/dist
            pos += forces * 0.1
            pos = np.clip(pos, 0.1, 0.9)
        for i in range(n):
            for j in range(i+1,n):
                c=corr[i,j]
                if abs(c) > 0.3:
                    color="green" if c>0 else "red"
                    ax.plot([pos[i,0],pos[j,0]],[pos[i,1],pos[j,1]],
                            color=color, alpha=abs(c), lw=1+2*abs(c))
        for i,name in enumerate(names):
            color=CAT_COLORS.get(enabled[i].category,"#888")
            ax.add_patch(plt.Circle(pos[i], 0.03, facecolor=color, edgecolor='black', alpha=0.8))
            ax.text(pos[i,0], pos[i,1], name[:10], ha="center", va="center",
                    fontsize=6, fontweight="bold")
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
        ax.axis("off")
        ax.set_title("Term Interaction Graph\n(green=positive, red=negative)", fontsize=10)
        self.draw_idle()

class ParetoCanvas(FigureCanvas):
    def __init__(self, mgr):
        self.mgr=mgr; self.fig=Figure(figsize=(6,4),tight_layout=True); super().__init__(self.fig)
    def update(self):
        self.fig.clear(); ax=self.fig.add_subplot(111)
        enabled=[t for t in self.mgr.terms.values() if t.enabled and t.weighted_history]
        if not enabled:
            ax.text(0.5,0.5,"No data",ha="center",va="center"); ax.axis("off"); self.draw_idle(); return
        x_vals=[]; y_vals=[]; names=[]; colors=[]
        for t in enabled:
            if t.weighted_history:
                x_vals.append(abs(np.mean(t.weighted_history)))
                y_vals.append(abs(t.history[-1]) if t.history else 0)
                names.append(t.name)
                colors.append(CAT_COLORS.get(t.category,"#888"))
        ax.scatter(x_vals, y_vals, c=colors, s=80, alpha=0.7, edgecolors='black')
        for i,name in enumerate(names):
            ax.annotate(name[:12], (x_vals[i],y_vals[i]), fontsize=6,
                        xytext=(5,5), textcoords='offset points')
        pts=np.array(list(zip(x_vals,y_vals)))
        pareto=[]
        for i,(x,y) in enumerate(pts):
            if not any((pts[:,0]<=x) & (pts[:,1]<=y) & ((pts[:,0]<x) | (pts[:,1]<y))):
                pareto.append(i)
        if pareto:
            p_pts=pts[pareto]
            order=np.argsort(p_pts[:,0])
            ax.plot(p_pts[order,0], p_pts[order,1], 'k--', lw=1.5, alpha=0.6, label="Pareto")
        ax.set_xlabel("|mean weighted contribution|")
        ax.set_ylabel("|current value|")
        ax.set_title("Pareto Frontier"); ax.grid(True,alpha=0.3); ax.legend(fontsize=8)
        self.draw_idle()

class ConvergenceTimelineCanvas(FigureCanvas):
    def __init__(self, mgr):
        self.mgr=mgr; self.fig=Figure(figsize=(6,3),tight_layout=True); super().__init__(self.fig)
    def update(self):
        self.fig.clear(); ax=self.fig.add_subplot(111)
        enabled=[t for t in self.mgr.terms.values() if t.enabled and t.history]
        if not enabled:
            ax.text(0.5,0.5,"No data",ha="center",va="center"); ax.axis("off"); self.draw_idle(); return
        names=[t.name for t in enabled]
        y_pos=np.arange(len(names))
        for i,t in enumerate(enabled):
            color=CAT_COLORS.get(t.category,"#888")
            end=len(t.history)
            if t.converged and t.convergence_epoch:
                ax.barh(i, t.convergence_epoch, left=0, color=color, alpha=0.4, height=0.6)
                ax.barh(i, end-t.convergence_epoch, left=t.convergence_epoch,
                        color=color, alpha=0.8, height=0.6)
                ax.plot(t.convergence_epoch, i, 'k*', markersize=10)
            else:
                ax.barh(i, end, left=0, color=color, alpha=0.6, height=0.6)
        ax.set_yticks(y_pos); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Epoch"); ax.set_title("Convergence Timeline (* = converged)")
        ax.grid(True, alpha=0.3, axis="x")
        self.draw_idle()

class PhaseScheduleCanvas(FigureCanvas):
    def __init__(self, mgr):
        self.mgr=mgr; self.fig=Figure(figsize=(6,3),tight_layout=True); super().__init__(self.fig)
    def update(self, total_epochs: int = 100):
        self.fig.clear(); ax=self.fig.add_subplot(111)
        enabled=[t for t in self.mgr.terms.values() if t.enabled]
        if not enabled:
            ax.text(0.5,0.5,"No terms enabled",ha="center",va="center"); ax.axis("off"); self.draw_idle(); return
        names=[t.name for t in enabled]
        y_pos=np.arange(len(names))
        for i,t in enumerate(enabled):
            color=CAT_COLORS.get(t.category,"#888")
            if t.phase == "always":
                ax.barh(i, 1.0, left=0, color=color, alpha=0.8, height=0.6)
            elif t.phase == "early":
                ax.barh(i, 0.33, left=0, color=color, alpha=0.8, height=0.6)
            elif t.phase == "mid":
                ax.barh(i, 0.55, left=0.2, color=color, alpha=0.8, height=0.6)
            elif t.phase == "late":
                ax.barh(i, 0.4, left=0.6, color=color, alpha=0.8, height=0.6)
        ax.set_yticks(y_pos); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlim(0,1); ax.set_xlabel("Training progress (0=start, 1=end)")
        ax.set_title("Phase Schedule")
        ax.axvline(0.33, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0.66, color='gray', linestyle=':', alpha=0.5)
        if len(names) > 0:
            ax.text(0.16, len(names)-0.3, "early", fontsize=7, ha="center", color="gray")
            ax.text(0.50, len(names)-0.3, "mid", fontsize=7, ha="center", color="gray")
            ax.text(0.83, len(names)-0.3, "late", fontsize=7, ha="center", color="gray")
        ax.grid(True, alpha=0.3, axis="x")
        self.draw_idle()


# ===========================================================================
# TRAINING WORKER
# ===========================================================================

class TrainWorker(QObject):
    progress=Signal(int,int); metrics=Signal(dict); term_breakdown=Signal(dict)
    finished=Signal(dict); error=Signal(str); log=Signal(str)

    def __init__(self, net, data, epochs, bs, lr, mgr, session_name="run"):
        super().__init__()
        self.net=net; self.data=data; self.epochs=epochs; self.bs=bs; self.lr=lr
        self.mgr=mgr; self.session_name=session_name; self._stop=False

    def stop(self): self._stop=True

    def run(self):
        try:
            Xtr=self.data["X_train"]; yoh=self.data["y_train_oh"]
            n=Xtr.shape[0]
            self.log.emit(f"[{self.session_name}] Training {self.epochs} epochs | lr={self.lr}")
            final_loss=0; final_vl=0; final_va=0
            for ep in range(1,self.epochs+1):
                if self._stop: break
                idx=np.random.permutation(n); losses=[]
                for s in range(0,n,self.bs):
                    ii=idx[s:s+self.bs]
                    L,caches=self.net.train_step(Xtr[ii],yoh[ii],self.lr)
                    L_aug,bd=self.mgr.compute_total(self.net,caches,L,ep,self.epochs)
                    losses.append(L_aug)
                tl=float(np.mean(losses))
                pred,_=self.net.forward(self.data["X_test"],False)
                probs=softmax(pred)
                vl=float(-np.mean(np.sum(self.data["y_test_oh"]*np.log(probs+1e-12),axis=1)))
                va=float(np.mean(np.argmax(probs,1)==np.argmax(self.data["y_test_oh"],1)))
                final_loss=tl; final_vl=vl; final_va=va
                self.metrics.emit({"epoch":ep,"train_loss":tl,"val_loss":vl,"val_acc":va})
                self.term_breakdown.emit(bd)
                self.progress.emit(ep,self.epochs)
                if ep%5==0: self.log.emit(f"[{self.session_name}] ep {ep:3d} train={tl:.4f} val={vl:.4f} acc={va*100:.1f}%")
            session = TrainingSession(
                name=self.session_name,
                preset="custom",
                dataset=self.data.get("name","?"),
                network_arch=f"{self.data['input_features']}→net→{self.data['n_classes']}",
                epochs=self.epochs,
                final_val_loss=final_vl,
                final_val_acc=final_va,
                term_histories={n_:list(t.history) for n_,t in self.mgr.terms.items() if t.history},
                term_weights={n_:t.weight for n_,t in self.mgr.terms.items()},
                term_enabled={n_:t.enabled for n_,t in self.mgr.terms.items()},
                final_loss=final_loss,
            )
            self.finished.emit({"session": session})
        except Exception:
            self.error.emit(traceback.format_exc())


# ===========================================================================
# MAIN WINDOW
# ===========================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Loss Term Study — Research Platform")
        self.resize(1700,1000)
        self.data=None; self.net=None; self.worker=None; self.thread=None
        self.mgr=LossTermManager()
        self.mgr.apply_preset("Balanced")
        self.store = SessionStore()
        self.optimizer = WeightOptimizer(self.mgr)
        self.dark_mode = False
        self._build_ui()

    def _build_ui(self):
        central=QWidget(); self.setCentralWidget(central)
        root=QHBoxLayout(central)
        splitter=QSplitter(Qt.Horizontal); root.addWidget(splitter)

        # ============ LEFT: Controls ============
        left=QTabWidget(); left.setMinimumWidth(440)

        # --- Tab 1: Setup ---
        t1=QWidget(); v1=QVBoxLayout(t1)
        gb=QGroupBox("Dataset")
        g=QVBoxLayout(gb)
        self.sk_combo=QComboBox()
        for k in ["iris","digits","wine","breast_cancer"]: self.sk_combo.addItem(k)
        load_btn=QPushButton("Load"); load_btn.clicked.connect(self._load_data)
        g.addWidget(self.sk_combo); g.addWidget(load_btn); v1.addWidget(gb)
        self.data_info=QLabel("<i>No dataset</i>"); self.data_info.setWordWrap(True); v1.addWidget(self.data_info)

        gb2=QGroupBox("Network")
        g2=QVBoxLayout(gb2)
        h=QHBoxLayout()
        h.addWidget(QLabel("Hidden sizes"))
        self.sizes_edit=QComboBox(); self.sizes_edit.setEditable(True)
        self.sizes_edit.setEditText("32,16")
        h.addWidget(self.sizes_edit); g2.addLayout(h)
        v1.addWidget(gb2)

        gb3=QGroupBox("Training")
        g3=QVBoxLayout(gb3)
        h2=QHBoxLayout()
        h2.addWidget(QLabel("epochs")); self.ep_spin=QSpinBox(); self.ep_spin.setRange(1,500); self.ep_spin.setValue(80)
        h2.addWidget(self.ep_spin)
        h2.addWidget(QLabel("batch")); self.bs_spin=QSpinBox(); self.bs_spin.setRange(1,256); self.bs_spin.setValue(32)
        h2.addWidget(self.bs_spin)
        h2.addWidget(QLabel("lr")); self.lr_spin=QDoubleSpinBox(); self.lr_spin.setDecimals(4); self.lr_spin.setRange(1e-4,0.1); self.lr_spin.setValue(0.005)
        h2.addWidget(self.lr_spin); g3.addLayout(h2)
        h3=QHBoxLayout()
        h3.addWidget(QLabel("Session name"))
        self.session_name_edit=QLineEdit("run_1")
        h3.addWidget(self.session_name_edit); g3.addLayout(h3)
        v1.addWidget(gb3)

        btns=QHBoxLayout()
        self.build_btn=QPushButton("Build"); self.build_btn.clicked.connect(self._build_net)
        self.train_btn=QPushButton("Train"); self.train_btn.clicked.connect(self._start_training)
        self.stop_btn=QPushButton("Stop"); self.stop_btn.clicked.connect(self._stop); self.stop_btn.setEnabled(False)
        btns.addWidget(self.build_btn); btns.addWidget(self.train_btn); btns.addWidget(self.stop_btn)
        v1.addLayout(btns)
        self.progress=QProgressBar(); v1.addWidget(self.progress)

        multi=QGroupBox("Multi-Run Analysis")
        ml=QVBoxLayout(multi)
        mb=QHBoxLayout()
        self.run_multi_btn=QPushButton("Run 3 Configs"); self.run_multi_btn.clicked.connect(self._run_multi)
        mb.addWidget(self.run_multi_btn)
        self.clear_sessions_btn=QPushButton("Clear Sessions"); self.clear_sessions_btn.clicked.connect(self._clear_sessions)
        mb.addWidget(self.clear_sessions_btn); ml.addLayout(mb)
        self.session_list=QComboBox(); ml.addWidget(QLabel("Compare sessions:"))
        ml.addWidget(self.session_list)
        v1.addWidget(multi)
        v1.addStretch()
        left.addTab(t1,"Setup")

        # --- Tab 2: Loss Terms ---
        t2=QWidget(); v2=QVBoxLayout(t2)
        preset_row=QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_combo=QComboBox()
        for k in PRESETS: self.preset_combo.addItem(k)
        self.preset_combo.setCurrentText("Balanced")
        preset_row.addWidget(self.preset_combo)
        apply_btn=QPushButton("Apply"); apply_btn.clicked.connect(self._apply_preset)
        preset_row.addWidget(apply_btn); v2.addLayout(preset_row)

        scroll=QScrollArea(); scroll.setWidgetResizable(True)
        scroll_widget=QWidget(); self.term_layout=QFormLayout(scroll_widget)
        self.term_widgets={}
        scroll.setWidget(scroll_widget)
        v2.addWidget(scroll)
        self._build_term_controls()
        left.addTab(t2,"Loss Terms")

        # --- Tab 3: Optimizer ---
        t3=QWidget(); v3=QVBoxLayout(t3)
        v3.addWidget(QLabel("<b>Auto-Weight Optimizer</b>"))
        v3.addWidget(QLabel("Random search over term weights."))
        oh=QHBoxLayout()
        oh.addWidget(QLabel("Configs")); self.opt_n_spin=QSpinBox()
        self.opt_n_spin.setRange(3,50); self.opt_n_spin.setValue(10)
        oh.addWidget(self.opt_n_spin); v3.addLayout(oh)
        self.run_opt_btn=QPushButton("Run Optimizer"); self.run_opt_btn.clicked.connect(self._run_optimizer)
        v3.addWidget(self.run_opt_btn)
        self.opt_result=QTextEdit(); self.opt_result.setReadOnly(True)
        self.opt_result.setMaximumHeight(200); v3.addWidget(self.opt_result)
        v3.addStretch()
        left.addTab(t3,"Optimizer")

        # --- Tab 4: Sessions ---
        t4=QWidget(); v4=QVBoxLayout(t4)
        sb=QHBoxLayout()
        save_btn=QPushButton("Save Sessions"); save_btn.clicked.connect(self._save_sessions)
        load_btn=QPushButton("Load Sessions"); load_btn.clicked.connect(self._load_sessions)
        sb.addWidget(save_btn); sb.addWidget(load_btn); v4.addLayout(sb)
        self.session_table=QTableWidget(); self.session_table.setColumnCount(5)
        self.session_table.setHorizontalHeaderLabels(["Name","Dataset","Epochs","Val Loss","Val Acc"])
        self.session_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        v4.addWidget(self.session_table)
        self._refresh_session_table()
        left.addTab(t4,"Sessions")

        # Log
        self.log_view=QTextEdit(); self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas",8)); self.log_view.setMaximumHeight(100)
        v1.addWidget(QLabel("<b>Log</b>")); v1.addWidget(self.log_view)

        splitter.addWidget(left)

        # ============ RIGHT: 12 chart tabs ============
        right=QTabWidget()
        charts = [
            ("📈 Evolution", TermEvolutionCanvas, "Raw term values over time"),
            ("📊 Stacked", StackedAreaCanvas, "Weighted contribution (stacked)"),
            ("🔥 Correlation", CorrelationHeatmapCanvas, "Pairwise term correlation"),
            ("🎯 Radar", RadarCanvas, "Normalized snapshot"),
            ("⚡ Sensitivity", SensitivityCanvas, "Impact per unit weight"),
            ("🎻 Violin", ViolinCanvas, "Distribution (violin + box)"),
            ("📐 Cumulative", CumulativeCanvas, "Running sum of contributions"),
            ("🌀 3D Phase", Phase3DCanvas, "Value × Gradient × Acceleration"),
            ("🕸 Interaction", InteractionGraphCanvas, "Force-directed term network"),
            ("⚖ Pareto", ParetoCanvas, "Efficiency frontier"),
            ("✓ Convergence", ConvergenceTimelineCanvas, "When each term converged"),
            ("📅 Schedule", PhaseScheduleCanvas, "Phase-based activation"),
        ]
        self.canvases = {}
        for title, cls, desc in charts:
            w=QWidget(); v=QVBoxLayout(w)
            canvas = cls(self.mgr)
            self.canvases[title] = canvas
            v.addWidget(QLabel(f"<b>{title}</b> — <i>{desc}</i>"))
            v.addWidget(canvas)
            right.addTab(w, title)
        splitter.addWidget(right)
        splitter.setSizes([440,1260])

        # ============ TOOLBAR ============
        tb=QToolBar(); self.addToolBar(tb)
        dark_btn=QToolButton(); dark_btn.setText("🌓 Dark Mode")
        dark_btn.clicked.connect(self._toggle_dark); tb.addWidget(dark_btn)
        tb.addSeparator()
        exp_png=QToolButton(); exp_png.setText("📷 Export PNG")
        exp_png.clicked.connect(self._export_png); tb.addWidget(exp_png)
        exp_csv=QToolButton(); exp_csv.setText("📊 Export CSV")
        exp_csv.clicked.connect(self._export_csv); tb.addWidget(exp_csv)

        self.setStatusBar(QStatusBar()); self.statusBar().showMessage("Ready.")

    def _build_term_controls(self):
        for name in list(self.term_widgets.keys()):
            widgets = self.term_widgets.pop(name, {})
            for w in widgets.values():
                if hasattr(w, 'setParent'): w.setParent(None)
        # Clear layout
        while self.term_layout.rowCount() > 0:
            self.term_layout.removeRow(0)

        for cat in CATEGORIES:
            header=QLabel(f"<b>━━ {cat} ━━</b>")
            header.setStyleSheet("background-color: #e8e8e8; padding: 4px;")
            self.term_layout.addRow(header)
            for name,t in self.mgr.terms.items():
                if t.category!=cat: continue
                row=QWidget(); hl=QHBoxLayout(row); hl.setContentsMargins(0,0,0,0)
                cb=QCheckBox(); cb.setChecked(t.enabled)
                cb.stateChanged.connect(lambda s, n=name: self._on_toggle(n, s))
                cb.setToolTip(t.description)
                hl.addWidget(cb)
                ws=QDoubleSpinBox(); ws.setDecimals(5); ws.setRange(-10,10); ws.setSingleStep(0.001)
                ws.setValue(t.weight); ws.setFixedWidth(90)
                ws.valueChanged.connect(lambda v, n=name: self._on_weight(n, v))
                hl.addWidget(ws)
                ps=QComboBox(); ps.addItems(["always","early","mid","late"])
                ps.setCurrentText(t.phase); ps.setFixedWidth(80)
                ps.currentTextChanged.connect(lambda p, n=name: self._on_phase(n, p))
                hl.addWidget(ps)
                hl.addStretch()
                self.term_layout.addRow(f"{name}:", row)
                self.term_widgets[name] = {"cb":cb, "ws":ws, "ps":ps}

    def _on_toggle(self, name, state):
        self.mgr.terms[name].enabled = (state == Qt.Checked)
    def _on_weight(self, name, value):
        self.mgr.terms[name].weight = value
    def _on_phase(self, name, phase):
        self.mgr.terms[name].phase = phase

    def _apply_preset(self):
        self.mgr.apply_preset(self.preset_combo.currentText())
        for name,t in self.mgr.terms.items():
            if name in self.term_widgets:
                self.term_widgets[name]["cb"].blockSignals(True)
                self.term_widgets[name]["ws"].blockSignals(True)
                self.term_widgets[name]["ps"].blockSignals(True)
                self.term_widgets[name]["cb"].setChecked(t.enabled)
                self.term_widgets[name]["ws"].setValue(t.weight)
                self.term_widgets[name]["ps"].setCurrentText(t.phase)
                self.term_widgets[name]["cb"].blockSignals(False)
                self.term_widgets[name]["ws"].blockSignals(False)
                self.term_widgets[name]["ps"].blockSignals(False)

    def _load_data(self):
        try:
            self.data=load_data(self.sk_combo.currentText())
            self.data["name"]=self.sk_combo.currentText()
            self.data_info.setText(f"<b>{self.data['name']}</b>: {self.data['X_train'].shape[0]} train, {self.data['input_features']} features, {self.data['n_classes']} classes")
        except Exception as e: QMessageBox.critical(self,"Error",str(e))

    def _build_net(self):
        if not self.data: QMessageBox.warning(self,"","Load data first."); return
        try:
            sizes=[int(x.strip()) for x in self.sizes_edit.currentText().split(",") if x.strip()]
        except: QMessageBox.warning(self,"","Invalid sizes."); return
        self.net=SimpleNet(self.data["input_features"],sizes,self.data["n_classes"],np.random.default_rng(42))
        self.mgr.reset(); self._log(f"Built: {self.data['input_features']}→{sizes}→{self.data['n_classes']}")

    def _start_training(self):
        if not self.net: QMessageBox.warning(self,"","Build first."); return
        self.train_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.progress.setMaximum(self.ep_spin.value()); self.progress.setValue(0)
        self.mgr.reset()
        session_name = self.session_name_edit.text() or "run"
        self.worker=TrainWorker(self.net,self.data,self.ep_spin.value(),self.bs_spin.value(),
                                self.lr_spin.value(),self.mgr,session_name)
        self.thread=QThread(); self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_prog)
        self.worker.metrics.connect(self._on_met)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_err)
        self.worker.finished.connect(self.thread.quit)
        self.thread.start()

    def _stop(self):
        if self.worker: self.worker.stop()

    def _on_prog(self,e,t): self.progress.setValue(e)

    def _on_met(self,m):
        for c in self.canvases.values(): c.update()
        if "📅 Schedule" in self.canvases:
            self.canvases["📅 Schedule"].update(total_epochs=self.ep_spin.value())

    def _on_done(self, result):
        self.train_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        if "session" in result:
            self.store.add(result["session"])
            self._refresh_session_table()
            self._refresh_session_list()
            self._log(f"✓ Session '{result['session'].name}' saved (val_acc={result['session'].final_val_acc*100:.1f}%)")
        self._log("Training finished.")

    def _on_err(self,tb):
        self.train_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        QMessageBox.critical(self,"Error",tb); self._log(tb)

    def _log(self,msg): self.log_view.append(msg)

    def _run_multi(self):
        if not self.data: QMessageBox.warning(self,"","Load data first."); return
        presets=["Minimal","Balanced","Robust"]
        base_name=self.session_name_edit.text() or "run"
        sizes=[int(x.strip()) for x in self.sizes_edit.currentText().split(",") if x.strip()]
        for i,p in enumerate(presets):
            self.mgr.apply_preset(p)
            self.mgr.reset()
            net=SimpleNet(self.data["input_features"],sizes,self.data["n_classes"],np.random.default_rng(42+i))
            name=f"{base_name}_{p}"
            self._log(f"Running {name}...")
            Xtr=self.data["X_train"]; yoh=self.data["y_train_oh"]
            n=Xtr.shape[0]; bs=self.bs_spin.value(); lr=self.lr_spin.value()
            epochs=self.ep_spin.value()
            losses=[]
            for ep in range(1,epochs+1):
                idx=np.random.permutation(n); ep_losses=[]
                for s in range(0,n,bs):
                    ii=idx[s:s+bs]
                    L,caches=net.train_step(Xtr[ii],yoh[ii],lr)
                    L_aug,bd=self.mgr.compute_total(net,caches,L,ep,epochs)
                    ep_losses.append(L_aug)
                losses.extend(ep_losses)
            pred,_=net.forward(self.data["X_test"],False)
            probs=softmax(pred)
            vl=float(-np.mean(np.sum(self.data["y_test_oh"]*np.log(probs+1e-12),axis=1)))
            va=float(np.mean(np.argmax(probs,1)==np.argmax(self.data["y_test_oh"],1)))
            sess=TrainingSession(name=name,preset=p,dataset=self.data["name"],
                                 network_arch=f"{self.data['input_features']}→{sizes}→{self.data['n_classes']}",
                                 epochs=epochs,final_val_loss=vl,final_val_acc=va,
                                 term_histories={n_:list(t.history) for n_,t in self.mgr.terms.items() if t.history},
                                 term_weights={n_:t.weight for n_,t in self.mgr.terms.items()},
                                 term_enabled={n_:t.enabled for n_,t in self.mgr.terms.items()},
                                 final_loss=float(np.mean(losses)))
            self.store.add(sess)
            self._log(f"  ✓ {name}: val_acc={va*100:.1f}%")
        self._refresh_session_table()
        self._refresh_session_list()
        self.canvases["📈 Evolution"].update(self.store.sessions)

    def _clear_sessions(self):
        self.store.sessions.clear()
        self._refresh_session_table()
        self._refresh_session_list()
        self.canvases["📈 Evolution"].update()

    def _refresh_session_table(self):
        self.session_table.setRowCount(0)
        for s in self.store.sessions:
            r=self.session_table.rowCount(); self.session_table.insertRow(r)
            self.session_table.setItem(r,0,QTableWidgetItem(s.name))
            self.session_table.setItem(r,1,QTableWidgetItem(s.dataset))
            self.session_table.setItem(r,2,QTableWidgetItem(str(s.epochs)))
            self.session_table.setItem(r,3,QTableWidgetItem(f"{s.final_val_loss:.4f}"))
            self.session_table.setItem(r,4,QTableWidgetItem(f"{s.final_val_acc*100:.1f}%"))

    def _refresh_session_list(self):
        self.session_list.clear()
        for s in self.store.sessions:
            self.session_list.addItem(s.name)

    def _run_optimizer(self):
        if not self.data: QMessageBox.warning(self,"","Load data first."); return
        if not self.net: QMessageBox.warning(self,"","Build network first."); return
        self.opt_result.clear()
        self._log("Running weight optimizer...")
        n=self.opt_n_spin.value()
        configs=self.optimizer.run_random_search(n)
        self.optimizer.results.clear()
        sizes=[int(x.strip()) for x in self.sizes_edit.currentText().split(",") if x.strip()]
        Xtr=self.data["X_train"]; yoh=self.data["y_train_oh"]
        n_=Xtr.shape[0]; bs=self.bs_spin.value(); lr=self.lr_spin.value(); epochs=self.ep_spin.value()

        for i,cfg in enumerate(configs):
            for name,w in cfg.items():
                if name in self.mgr.terms:
                    self.mgr.terms[name].weight=w
                    self.mgr.terms[name].enabled=True
            self.mgr.reset()
            net=SimpleNet(self.data["input_features"],sizes,self.data["n_classes"],np.random.default_rng(i))
            losses=[]
            for ep in range(1,epochs+1):
                idx=np.random.permutation(n_); ep_losses=[]
                for s in range(0,n_,bs):
                    ii=idx[s:s+bs]
                    L,caches=net.train_step(Xtr[ii],yoh[ii],lr)
                    L_aug,_=self.mgr.compute_total(net,caches,L,ep,epochs)
                    ep_losses.append(L_aug)
                losses.extend(ep_losses)
            pred,_=net.forward(self.data["X_test"],False)
            probs=softmax(pred)
            vl=float(-np.mean(np.sum(self.data["y_test_oh"]*np.log(probs+1e-12),axis=1)))
            va=float(np.mean(np.argmax(probs,1)==np.argmax(self.data["y_test_oh"],1)))
            self.optimizer.record_result(cfg,vl,va)
            if (i+1)%3==0:
                best=min(r['final_loss'] for r in self.optimizer.results)
                self._log(f"  optimizer: {i+1}/{n} done, best so far val_loss={best:.4f}")

        best=self.optimizer.best_config()
        self.opt_result.append("<b>Best configuration:</b>")
        for name,w in best.items():
            self.opt_result.append(f"  {name}: {w:.5f}")
        self.opt_result.append(f"<br><b>Final val loss:</b> {min(r['final_loss'] for r in self.optimizer.results):.4f}")
        self._log("Optimizer complete.")

    def _save_sessions(self):
        path,_=QFileDialog.getSaveFileName(self,"Save Sessions","","JSON (*.json)")
        if path:
            self.store.save_to_file(path)
            self._log(f"Saved {len(self.store.sessions)} sessions to {path}")

    def _load_sessions(self):
        path,_=QFileDialog.getOpenFileName(self,"Load Sessions","","JSON (*.json)")
        if path:
            self.store.load_from_file(path)
            self._refresh_session_table()
            self._refresh_session_list()
            self._log(f"Loaded {len(self.store.sessions)} sessions")

    def _export_png(self):
        path,_=QFileDialog.getSaveFileName(self,"Export PNG","","PNG (*.png)")
        if path:
            for c in self.canvases.values():
                if c.isVisible():
                    c.fig.savefig(path, dpi=150, bbox_inches='tight')
                    self._log(f"Exported to {path}")
                    return

    def _export_csv(self):
        path,_=QFileDialog.getSaveFileName(self,"Export CSV","","CSV (*.csv)")
        if path:
            import csv
            with open(path,'w',newline='') as f:
                w=csv.writer(f)
                enabled=[t for t in self.mgr.terms.values() if t.enabled and t.history]
                if not enabled: return
                max_len=max(len(t.history) for t in enabled)
                w.writerow(["epoch"]+[t.name for t in enabled])
                for i in range(max_len):
                    row=[i+1]
                    for t in enabled:
                        row.append(t.history[i] if i<len(t.history) else "")
                    w.writerow(row)
            self._log(f"Exported to {path}")

    def _toggle_dark(self):
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            QApplication.setStyle(QStyleFactory.create("Fusion"))
            p=QPalette()
            p.setColor(QPalette.Window, QColor(53,53,53))
            p.setColor(QPalette.WindowText, Qt.white)
            p.setColor(QPalette.Base, QColor(25,25,25))
            p.setColor(QPalette.AlternateBase, QColor(53,53,53))
            p.setColor(QPalette.ToolTipBase, Qt.white)
            p.setColor(QPalette.ToolTipText, Qt.white)
            p.setColor(QPalette.Text, Qt.white)
            p.setColor(QPalette.Button, QColor(53,53,53))
            p.setColor(QPalette.ButtonText, Qt.white)
            p.setColor(QPalette.BrightText, Qt.red)
            p.setColor(QPalette.Link, QColor(42,130,218))
            p.setColor(QPalette.Highlight, QColor(42,130,218))
            p.setColor(QPalette.HighlightedText, Qt.black)
            QApplication.setPalette(p)
        else:
            QApplication.setPalette(QApplication.style().standardPalette())


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()