"""
Reactive Diffusion as Feed-Forward Neural Network
with Advanced Monte Carlo Parameter Search + Dataset Benchmarks
================================================================
Production-Grade Single-File Implementation with:
  - Gray-Scott reaction-diffusion simulation (CPU + CUDA)
  - Feed-forward network interpretation (V-field = weights, NO TRAINING)
  - Advanced Evolutionary/MC optimizer with population-based search (DE + Adaptive Jitter)
  - Extreme Learning Machine (ELM) Ridge Regression Output Layer (Analytically Solved)
  - Custom dataset benchmarking (CSV input) with Mini-Batching
  - Modern dark GUI with 9 analysis tabs + PCA visualization

Architecture:
  - RD Simulation → V-field image → Neural Network weights (frozen)
  - Dataset → Forward pass through V-field → Hidden State → Ridge Regression Output → Loss
  - MC/DE Optimizer → Minimizes loss by evolving RD parameters

Requirements:
    pip install PySide6 numpy scipy
    pip install cupy-cuda12x        # optional GPU
    pip install imageio             # optional GIF export
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

# ==============================================================================
# 1. OPTIONAL DEPENDENCIES
# ==============================================================================
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    imageio = None
    HAS_IMAGEIO = False

# ==============================================================================
# 2. PYSIDE6 IMPORTS
# ==============================================================================
from PySide6.QtCore import Qt, QTimer, QPoint, QSize, Signal, Slot
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPalette, QPen, QBrush, QFont,
    QKeySequence, QShortcut, QLinearGradient, QRadialGradient, QCursor,
    QIcon, QPainterPath, QMouseEvent,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QDoubleSpinBox, QSpinBox, QPushButton,
    QComboBox, QScrollArea, QFrame, QCheckBox, QGridLayout,
    QSplitter, QFileDialog, QStatusBar, QTabWidget, QSizePolicy,
    QToolBar, QMessageBox, QToolButton,
)

# ==============================================================================
# 3. LOGGING & CONSTANTS
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RD_NN")


@dataclass(frozen=True)
class AppConstants:
    """Centralized configuration for magic numbers and defaults."""
    DEFAULT_WIDTH: int = 256
    DEFAULT_HEIGHT: int = 256
    MIN_STEPS_PER_FRAME: int = 1
    MAX_STEPS_PER_FRAME: int = 200
    DEFAULT_STEPS_PER_FRAME: int = 20
    MAX_GIF_FRAMES: int = 300
    NAN_REPLACE: float = 0.0
    POSINF_REPLACE: float = 1.0
    NEGINF_REPLACE: float = 0.0


# ==============================================================================
# 4. MODERN QSS STYLING
# ==============================================================================
MODERN_QSS = """
/* === Base === */
QMainWindow { background-color: #1a1a1f; }
QWidget { color: #e4e4e8; font-family: 'Segoe UI', 'SF Pro Display', -apple-system, sans-serif; font-size: 12px; }
/* === Sidebar === */
#sidebar { background-color: #14141a; border-right: 1px solid #2a2a35; }
#sidebar QScrollArea { border: none; background-color: transparent; }
#sidebar QScrollArea > QWidget > QWidget { background-color: transparent; }
/* === Collapsible Sections === */
.section-header { background-color: #1e1e26; color: #4fc3f7; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; padding: 10px 14px; border-top: 1px solid #2a2a35; border-bottom: 1px solid #2a2a35; }
.section-header:hover { background-color: #25252f; }
.section-body { background-color: #1a1a20; padding: 10px 14px; }
/* === Group Boxes === */
QGroupBox { background-color: #1e1e26; border: 1px solid #2a2a35; border-radius: 8px; margin-top: 12px; padding: 14px 10px 10px 10px; font-weight: 600; color: #b0b0c0; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 2px 10px; color: #4fc3f7; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
/* === Buttons === */
QPushButton { background-color: #2a2a35; color: #e4e4e8; border: 1px solid #3a3a45; border-radius: 6px; padding: 7px 14px; font-weight: 500; min-height: 22px; }
QPushButton:hover { background-color: #353542; border-color: #4fc3f7; }
QPushButton:pressed { background-color: #1a1a22; }
QPushButton:checked { background-color: #0d47a1; border-color: #4fc3f7; color: #ffffff; }
QPushButton#primary { background-color: #0d47a1; border-color: #4fc3f7; color: #ffffff; }
QPushButton#primary:hover { background-color: #1565c0; }
QPushButton#danger { background-color: #5d1a1a; border-color: #ff5252; }
QPushButton#danger:hover { background-color: #7a2020; }
/* === Inputs === */
QDoubleSpinBox, QSpinBox, QComboBox { background-color: #14141a; border: 1px solid #3a3a45; border-radius: 5px; padding: 5px 8px; color: #e4e4e8; min-height: 20px; }
QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus { border-color: #4fc3f7; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView { background-color: #1a1a22; border: 1px solid #3a3a45; selection-background-color: #0d47a1; selection-color: #ffffff; }
/* === Sliders === */
QSlider::groove:horizontal { background: #2a2a35; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal { background: #4fc3f7; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: #81d4fa; }
QSlider::sub-page:horizontal { background: #0d47a1; border-radius: 2px; }
/* === Checkboxes === */
QCheckBox { spacing: 8px; color: #c0c0d0; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #3a3a45; border-radius: 3px; background-color: #14141a; }
QCheckBox::indicator:checked { background-color: #0d47a1; border-color: #4fc3f7; image: none; }
/* === Tabs === */
QTabWidget::pane { background-color: #1a1a20; border: 1px solid #2a2a35; border-top: none; border-radius: 0 0 8px 8px; }
QTabBar::tab { background-color: #1e1e26; color: #808090; border: 1px solid #2a2a35; border-bottom: none; padding: 9px 18px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px; }
QTabBar::tab:selected { background-color: #1a1a20; color: #4fc3f7; border-bottom: 2px solid #4fc3f7; }
QTabBar::tab:hover:!selected { background-color: #25252f; color: #c0c0d0; }
/* === Scrollbars === */
QScrollBar:vertical { background-color: #14141a; width: 10px; border: none; }
QScrollBar::handle:vertical { background-color: #3a3a45; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background-color: #4fc3f7; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { background-color: #14141a; height: 10px; border: none; }
QScrollBar::handle:horizontal { background-color: #3a3a45; border-radius: 5px; min-width: 30px; }
/* === Labels === */
QLabel { color: #b0b0c0; }
QLabel#value-display { color: #4fc3f7; font-weight: 600; font-family: 'Consolas', 'Monaco', monospace; }
QLabel#metric-value { color: #4fc3f7; font-size: 22px; font-weight: 700; font-family: 'Consolas', 'Monaco', monospace; }
QLabel#metric-label { color: #808090; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
/* === Status Bar === */
QStatusBar { background-color: #0d0d12; color: #808090; border-top: 1px solid #2a2a35; font-family: 'Consolas', 'Monaco', monospace; font-size: 11px; }
QStatusBar::item { border: none; }
/* === Toolbar === */
QToolBar { background-color: #14141a; border-bottom: 1px solid #2a2a35; spacing: 6px; padding: 4px 8px; }
QToolBar QToolButton { background-color: transparent; color: #b0b0c0; border: 1px solid transparent; border-radius: 5px; padding: 5px 10px; font-size: 12px; }
QToolBar QToolButton:hover { background-color: #25252f; border-color: #3a3a45; color: #4fc3f7; }
QToolBar QToolButton:checked { background-color: #0d47a1; color: #ffffff; }
QToolBar QToolButton:pressed { background-color: #1a1a22; }
/* === Tooltips === */
QToolTip { background-color: #25252f; color: #e4e4e8; border: 1px solid #4fc3f7; border-radius: 4px; padding: 5px 8px; font-size: 11px; }
"""


# ==============================================================================
# 5. PROTOCOLS & ENUMS
# ==============================================================================
class SimulationProtocol(Protocol):
    use_gpu: bool
    width: int
    height: int
    def get_params(self) -> Dict[str, float]: ...
    def set_params(self, params: Dict[str, float]) -> None: ...
    def step(self, iterations: int, save_prev: bool = False) -> None: ...
    def get_v_np(self) -> np.ndarray: ...
    def get_prev_v(self) -> Optional[np.ndarray]: ...
    def save_state(self) -> Dict[str, Any]: ...
    def restore_state(self, state: Dict[str, Any]) -> None: ...


class NetworkProtocol(Protocol):
    def test(self) -> None: ...
    @property
    def last_output(self) -> Optional[np.ndarray]: ...
    @property
    def last_activations(self) -> Optional[np.ndarray]: ...


class OptimizationStatus(Enum):
    RUNNING = "running"
    CONVERGED = "converged"
    STALLED = "stalled"
    FAILED = "failed"


@dataclass(frozen=True)
class Preset:
    name: str
    F: float
    k: float
    Du: float = 1.0
    Dv: float = 0.5
    seed_radius: int = 10
    seed_count: int = 10
    
    def __post_init__(self):
        if not (0.001 <= self.F <= 0.1):
            raise ValueError(f"F must be in [0.001, 0.1], got {self.F}")
        if not (0.001 <= self.k <= 0.08):
            raise ValueError(f"k must be in [0.001, 0.08], got {self.k}")


@dataclass(frozen=True)
class Candidate:
    params: Dict[str, float]
    fitness: float
    fitness_components: Dict[str, float]
    age: int = 0
    candidate_id: int = field(default_factory=lambda: id(object()))
    
    def __post_init__(self):
        if not np.isfinite(self.fitness):
            raise ValueError(f"Candidate fitness must be finite, got {self.fitness}")


@dataclass(frozen=True)
class OptimizerConfig:
    population_size: int = 5
    initial_temperature: float = 0.015
    min_temperature: float = 0.0005
    max_temperature: float = 0.100
    target_acceptance: float = 0.234
    temp_adjust_rate: float = 0.01
    restart_threshold: int = 50
    max_restarts: int = 10
    momentum_coeff: float = 0.3
    diversity_weight: float = 0.2
    fitness_sharing_alpha: float = 1.0
    fitness_sharing_sigma: float = 0.1
    w_var: float = 1.0
    w_mean: float = 1.0
    w_grad: float = 1.0
    w_sparsity: float = 0.5
    w_pattern: float = 0.3
    w_spectral_entropy: float = 1.5
    w_wavelength: float = 1.0
    convergence_window: int = 100
    convergence_threshold: float = 1e-4
    perturb_flags: Dict[str, bool] = field(default_factory=lambda: {
        "F": True, "k": True, "Du": False, "Dv": False, "dt": False
    })
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "F": (0.001, 0.100), "k": (0.001, 0.080),
        "Du": (0.10, 2.00), "Dv": (0.05, 1.50), "dt": (0.10, 2.00)
    })
    param_scales: Dict[str, float] = field(default_factory=lambda: {
        "F": 1.0, "k": 1.0, "Du": 0.5, "Dv": 0.5, "dt": 0.3
    })
    
    def __post_init__(self):
        if self.population_size < 1:
            raise ValueError("population_size must be >= 1")
        if not (0.0 < self.initial_temperature <= 1.0):
            raise ValueError("initial_temperature must be in (0, 1]")
        if self.min_temperature >= self.max_temperature:
            raise ValueError("min_temperature must be < max_temperature")


# ==============================================================================
# 6. PRESETS & COLORMAPS
# ==============================================================================
PRESETS: Dict[str, Preset] = {
    "Coral":    Preset("Coral",    0.0545, 0.0620, seed_radius=12, seed_count=8),
    "Mitosis":  Preset("Mitosis",  0.0367, 0.0649, seed_radius=6,  seed_count=15),
    "Spots":    Preset("Spots",    0.0350, 0.0650, seed_radius=10, seed_count=10),
    "Worms":    Preset("Worms",    0.0780, 0.0610, seed_radius=8,  seed_count=12),
    "Waves":    Preset("Waves",    0.0140, 0.0545, seed_radius=15, seed_count=6),
    "Maze":     Preset("Maze",     0.0290, 0.0570, seed_radius=20, seed_count=4),
    "Holes":    Preset("Holes",    0.0390, 0.0650, seed_radius=8,  seed_count=12),
    "Solitons": Preset("Solitons", 0.0300, 0.0620, seed_radius=10, seed_count=10),
    "Pulsing":  Preset("Pulsing",  0.0250, 0.0600, seed_radius=14, seed_count=6),
    "U-Skate":  Preset("U-Skate",  0.0620, 0.0609, seed_radius=10, seed_count=10),
}

def _generate_lut(r_func, g_func, b_func, size=256) -> np.ndarray:
    """Helper to generate consistent colormaps."""
    t = np.linspace(0, 1, size)
    r = np.clip(r_func(t), 0, 1)
    g = np.clip(g_func(t), 0, 1)
    b = np.clip(b_func(t), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)

COLORMAPS = {
    "Ink": lambda: _generate_lut(lambda t: 0.961 - 0.902 * (t * t * (3 - 2 * t)),
                                 lambda t: 0.933 - 0.886 * (t * t * (3 - 2 * t)),
                                 lambda t: 0.882 - 0.764 * (t * t * (3 - 2 * t))),
    "Grayscale": lambda: _generate_lut(lambda t: t, lambda t: t, lambda t: t),
    "Viridis": lambda: _generate_lut(lambda t: 0.27 + 1.20*t - 1.80*t**2 + 0.80*t**3,
                                     lambda t: 1.40*t - 0.60*t**2,
                                     lambda t: 0.33 + 0.60*t - 1.30*t**2 + 0.70*t**3),
    "Inferno": lambda: _generate_lut(lambda t: 1.60*t - 0.60*t**2,
                                     lambda t: 0.50*t + 0.80*t**2 - 0.30*t**3,
                                     lambda t: 0.05 + 1.20*t - 2.00*t**2 + 0.80*t**3),
    "Magma": lambda: _generate_lut(lambda t: 0.05 + 2.00*t - 1.00*t**2,
                                   lambda t: 0.02 + 0.40*t + 0.80*t**2 - 0.30*t**3,
                                   lambda t: 0.10 + 1.60*t - 2.20*t**2 + 0.70*t**3),
    "Sepia": lambda: _generate_lut(lambda t: 0.15 + 0.80*t, lambda t: 0.08 + 0.72*t, lambda t: 0.04 + 0.55*t),
    "Plasma": lambda: _generate_lut(lambda t: 0.05 + 2.40*t - 2.00*t**2 + 0.60*t**3,
                                    lambda t: 0.03 + 0.20*t + 1.20*t**2 - 0.50*t**3,
                                    lambda t: 0.53 + 0.60*t - 2.00*t**2 + 1.00*t**3),
}

OVERLAY_CMAPS = {
    "Hot (red/yellow)": lambda: _generate_lut(lambda t: np.clip(t * 2.5, 0, 1),
                                               lambda t: np.clip((t - 0.4) * 2.5, 0, 1),
                                               lambda t: np.clip((t - 0.7) * 3.3, 0, 1)),
    "Cool (cyan/magenta)": lambda: _generate_lut(lambda t: np.clip(t, 0, 1),
                                                  lambda t: np.clip(1 - t, 0, 1) * 0.6,
                                                  lambda t: np.clip(1 - t * 0.5, 0, 1)),
    "Electric (blue/purple)": lambda: _generate_lut(lambda t: np.clip(t * t * 2, 0, 1),
                                                    lambda t: np.clip(t, 0, 1) * 0.8,
                                                    lambda t: np.clip(0.3 + t * 0.7, 0, 1)),
}


# ==============================================================================
# 7. DATASET BENCHMARKS
# ==============================================================================
@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    num_features: int
    num_classes: int
    n_samples: int = 800

BENCHMARK_PRESETS: Dict[str, BenchmarkConfig] = {
    "Blobs (64F, 4C)": BenchmarkConfig("Blobs", 64, 4),
    "XOR Non-Linear (2F, 2C)": BenchmarkConfig("XOR", 2, 2),
    "Linear Separable (16F, 3C)": BenchmarkConfig("Linear", 16, 3),
}

class BenchmarkDataset:
    """Lightweight, dependency-free synthetic dataset generator for benchmarking."""
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.X_train, self.y_train, self.X_test, self.y_test = self._generate()

    def _generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        n = self.config.n_samples
        
        if self.config.name == "Blobs":
            X = rng.normal(size=(n, self.config.num_features))
            centers = rng.normal(scale=2.5, size=(self.config.num_classes, self.config.num_features))
            y = rng.integers(0, self.config.num_classes, size=n)
            X += centers[y]
        elif self.config.name == "XOR":
            X = rng.uniform(-1.5, 1.5, size=(n, 2))
            y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
        else:  # Linear
            X = rng.normal(size=(n, self.config.num_features))
            W_true = rng.normal(scale=1.5, size=(self.config.num_features, self.config.num_classes))
            logits = X @ W_true
            y = np.argmax(logits, axis=1)
            
        split = int(0.8 * n)
        return X[:split], y[:split], X[split:], y[split:]


class CustomBenchmarkDataset:
    """Loads, normalizes, and splits external CSV datasets for benchmarking."""
    def __init__(self, filepath: str, target_col_idx: int = -1, max_samples: int = 5000):
        self.filepath = filepath
        self.num_features = 0
        self.num_classes = 0
        self.n_samples = 0
        self.X_train, self.y_train, self.X_test, self.y_test = self._load_and_preprocess(filepath, target_col_idx, max_samples)

    def _load_and_preprocess(self, filepath: str, target_col_idx: int, max_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            raw_data = np.genfromtxt(filepath, delimiter=',', skip_header=1, max_rows=max_samples, dtype=np.float64)
            if raw_data.ndim == 1:
                raw_data = raw_data.reshape(-1, 1)
            valid_mask = ~np.isnan(raw_data).any(axis=1)
            raw_data = raw_data[valid_mask]
            if raw_data.shape[0] < 10:
                raise ValueError("Dataset too small after cleaning (< 10 valid rows).")
            if target_col_idx < 0:
                target_col_idx = raw_data.shape[1] + target_col_idx
            if target_col_idx >= raw_data.shape[1] or target_col_idx < 0:
                raise ValueError(f"Invalid target column index: {target_col_idx}")

            X = raw_data[:, :target_col_idx]
            y = raw_data[:, target_col_idx].astype(int)
            
            x_min, x_max = X.min(axis=0), X.max(axis=0)
            x_range = x_max - x_min
            x_range[x_range == 0] = 1.0
            X = ((X - x_min) / x_range) * 2.0 - 1.0
            
            self.num_features = X.shape[1]
            self.num_classes = len(np.unique(y))
            self.n_samples = len(y)
            
            rng = np.random.default_rng(42)
            indices = rng.permutation(self.n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]
            
            split = int(0.8 * self.n_samples)
            return X_shuffled[:split], y_shuffled[:split], X_shuffled[split:], y_shuffled[split:]
        except Exception as e:
            raise ValueError(f"Failed to load dataset '{filepath}': {e}")


# ==============================================================================
# 8. CUDA KERNELS
# ==============================================================================
GS_CUDA_KERNEL = r"""
extern "C" __global__
void gs_step(double* u, double* v, double Du, double Dv, double F, double k,
             double dt, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int x1 = (x == 0) ? width - 1 : x - 1;
    int x2 = (x == width - 1) ? 0 : x + 1;
    int y1 = (y == 0) ? height - 1 : y - 1;
    int y2 = (y == height - 1) ? 0 : y + 1;
    int idx = y * width + x;
    double u_val = u[idx];
    double v_val = v[idx];
    double lap_u = u[y*width+x1] + u[y*width+x2] + u[y1*width+x] + u[y2*width+x] - 4.0*u_val;
    double lap_v = v[y*width+x1] + v[y*width+x2] + v[y1*width+x] + v[y2*width+x] - 4.0*v_val;
    double uvv = u_val * v_val * v_val;
    u[idx] = u_val + dt * (Du*lap_u - uvv + F*(1.0 - u_val));
    v[idx] = v_val + dt * (Dv*lap_v + uvv - (F + k)*v_val);
}
"""

FFN_CUDA_KERNEL = r"""
extern "C" __global__
void ffn_forward(const double* weights, const double* x_in, double* out,
                 int H, int W, int act_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H) return;
    double val = x_in[idx];
    for (int w = 0; w < W; w++) {
        val = val * weights[idx + w * H];
        if (act_type == 0) val = tanh(val);
        else if (act_type == 1) val = 1.0 / (1.0 + exp(-val));
        else if (act_type == 2) val = val > 0 ? val : 0.0;
        else if (act_type == 3) val = val > 0 ? val : 0.01 * val;
    }
    out[idx] = val;
}
"""


# ==============================================================================
# 9. CORE SIMULATION
# ==============================================================================
class GrayScott:
    """Gray-Scott reaction-diffusion simulation core."""
    def __init__(self, width: int = AppConstants.DEFAULT_WIDTH, 
                 height: int = AppConstants.DEFAULT_HEIGHT, use_gpu: bool = False):
        self.width = width
        self.height = height
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = cp if self.use_gpu else np
        
        self.u = self.xp.ones((height, width), dtype=self.xp.float64)
        self.v = self.xp.zeros((height, width), dtype=self.xp.float64)
        
        self.Du: float = 1.0
        self.Dv: float = 0.5
        self.F: float = 0.055
        self.k: float = 0.062
        self.dt: float = 1.0
        self._prev_v: Optional[np.ndarray] = None
        
        if self.use_gpu:
            self._gs_kernel = cp.RawKernel(GS_CUDA_KERNEL, "gs_step")

    def set_gpu(self, use_gpu: bool) -> None:
        want_gpu = use_gpu and HAS_CUPY
        if want_gpu == self.use_gpu:
            return
        u_np = self.u.get() if self.use_gpu else np.array(self.u)
        v_np = self.v.get() if self.use_gpu else np.array(self.v)
        self.use_gpu = want_gpu
        self.xp = cp if want_gpu else np
        self.u = self.xp.array(u_np, dtype=self.xp.float64)
        self.v = self.xp.array(v_np, dtype=self.xp.float64)
        if self.use_gpu:
            self._gs_kernel = cp.RawKernel(GS_CUDA_KERNEL, "gs_step")

    def reset(self) -> None:
        self.u[:] = 1.0
        self.v[:] = 0.0
        self._prev_v = None

    def seed_random(self, radius: int = 10, count: int = 10) -> None:
        self.reset()
        rng = np.random.default_rng()
        for _ in range(count):
            cx = rng.integers(radius, self.width - radius)
            cy = rng.integers(radius, self.height - radius)
            r = rng.integers(max(2, radius // 2), radius + 1)
            yy, xx = np.ogrid[-cy:self.height-cy, -cx:self.width-cx]
            mask_np = xx*xx + yy*yy <= r*r
            mask = cp.asarray(mask_np) if self.use_gpu else mask_np
            self.v[mask] = 0.25 + 0.05 * rng.random()
            self.u[mask] = 0.50 + 0.05 * rng.random()

    def seed_point(self, x: int, y: int, radius: int = 8, intensity: float = 0.35) -> None:
        yy, xx = np.ogrid[-y:self.height-y, -x:self.width-x]
        mask_np = xx*xx + yy*yy <= radius*radius
        mask = cp.asarray(mask_np) if self.use_gpu else mask_np
        self.v[mask] = intensity
        self.u[mask] = 0.50

    def step(self, iterations: int = 1, save_prev: bool = False) -> None:
        if save_prev and iterations > 0:
            self._prev_v = self.v.copy() if not self.use_gpu else self.v.get()
            
        if self.use_gpu and iterations > 1:
            block = (16, 16)
            grid = ((self.width + block[0] - 1) // block[0],
                    (self.height + block[1] - 1) // block[1])
            for _ in range(iterations):
                self._gs_kernel(grid, block, (
                    self.u, self.v, self.Du, self.Dv, self.F, self.k, self.dt,
                    self.width, self.height,
                ))
        else:
            xp = self.xp
            u, v = self.u, self.v
            Du, Dv, F, k, dt = self.Du, self.Dv, self.F, self.k, self.dt
            for _ in range(iterations):
                lap_u = (xp.roll(u, 1, 0) + xp.roll(u, -1, 0) + xp.roll(u, 1, 1) + xp.roll(u, -1, 1) - 4*u)
                lap_v = (xp.roll(v, 1, 0) + xp.roll(v, -1, 0) + xp.roll(v, 1, 1) + xp.roll(v, -1, 1) - 4*v)
                uvv = u * v * v
                u += dt * (Du*lap_u - uvv + F*(1 - u))
                v += dt * (Dv*lap_v + uvv - (F + k)*v)

    def get_u_np(self) -> np.ndarray:
        return self.u.get() if self.use_gpu else self.u

    def get_v_np(self) -> np.ndarray:
        return self.v.get() if self.use_gpu else self.v

    def get_prev_v(self) -> Optional[np.ndarray]:
        return self._prev_v

    def get_params(self) -> Dict[str, float]:
        return {"F": self.F, "k": self.k, "Du": self.Du, "Dv": self.Dv, "dt": self.dt}

    def set_params(self, p: Dict[str, float]) -> None:
        for key in ("F", "k", "Du", "Dv", "dt"):
            if key in p:
                setattr(self, key, float(p[key]))

    def save_state(self) -> Dict[str, Any]:
        return {"u": self.u.copy(), "v": self.v.copy(), "params": self.get_params()}

    def restore_state(self, state: Dict[str, Any]) -> None:
        self.u = state["u"]
        self.v = state["v"]
        self.set_params(state["params"])


class RippleSystem:
    """Automated perturbation system for the simulation."""
    def __init__(self, sim: GrayScott):
        self.sim = sim
        self.enabled = False
        self.interval = 30
        self.frame_counter = 0
        self.mode = "random"
        self.radius = 5
        self.intensity = 0.35
        self.grid_spacing = 40
        self.grid_offset_x = 0
        self.grid_offset_y = 0
        self._spiral_t = 0.0

    def tick(self) -> None:
        if not self.enabled:
            return
        self.frame_counter += 1
        if self.frame_counter >= self.interval:
            self.frame_counter = 0
            self._place_drop()

    def _place_drop(self) -> None:
        rng = np.random.default_rng()
        if self.mode == "random":
            x = rng.integers(self.radius, self.sim.width - self.radius)
            y = rng.integers(self.radius, self.sim.height - self.radius)
            self.sim.seed_point(x, y, self.radius, self.intensity)
        elif self.mode == "grid":
            self.sim.seed_point(self.grid_offset_x, self.grid_offset_y, self.radius, self.intensity)
            self.grid_offset_x += self.grid_spacing
            if self.grid_offset_x >= self.sim.width:
                self.grid_offset_x = self.grid_spacing // 2
                self.grid_offset_y += self.grid_spacing
                if self.grid_offset_y >= self.sim.height:
                    self.grid_offset_y = self.grid_spacing // 2
        elif self.mode == "spiral":
            self._spiral_t += 0.15
            r = 20 + self._spiral_t * 3
            cx, cy = self.sim.width // 2, self.sim.height // 2
            x = int(cx + r * np.cos(self._spiral_t))
            y = int(cy + r * np.sin(self._spiral_t))
            if 0 <= x < self.sim.width and 0 <= y < self.sim.height:
                self.sim.seed_point(x, y, self.radius, self.intensity)

    def reset_grid(self) -> None:
        self.grid_offset_x = self.grid_spacing // 2
        self.grid_offset_y = self.grid_spacing // 2
        self._spiral_t = 0.0


class FeedForwardNetwork:
    """Interprets the V field as weights for a feed-forward network (ELM-based)."""
    def __init__(self, sim: GrayScott):
        self.sim = sim
        self.input_pattern = "sine"
        self.activation_fn = "tanh"
        self.last_activations: Optional[np.ndarray] = None
        self.last_output: Optional[np.ndarray] = None
        self.last_input: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(42)
        
        # Benchmark state
        self.benchmark_mode = False
        self.benchmark_dataset: Optional[object] = None
        self.last_accuracy = 0.0
        self.last_loss = 0.0
        self._input_proj: Optional[np.ndarray] = None
        
        if self.sim.use_gpu:
            self._ffn_kernel = cp.RawKernel(FFN_CUDA_KERNEL, "ffn_forward")

    def test(self) -> None:
        """Standard test for intrinsic pattern evaluation."""
        xp = self.sim.xp
        weights = self.sim.get_v_np()
        H, W = weights.shape
        
        if self.input_pattern == "sine":
            x = np.sin(np.linspace(0, 4*np.pi, H))
        elif self.input_pattern == "random":
            x = self._rng.standard_normal(H)
        elif self.input_pattern == "impulse":
            x = np.zeros(H); x[H // 2] = 1.0
        elif self.input_pattern == "gradient":
            x = np.linspace(-1, 1, H)
        elif self.input_pattern == "step":
            x = np.ones(H); x[:H // 2] = -1.0
        else:
            x = np.zeros(H)
            
        self.last_input = x.copy()
        x_dev = xp.array(x, dtype=xp.float64)
        w_dev = xp.array(weights, dtype=xp.float64, order="F")
        out_dev = xp.zeros(H, dtype=xp.float64)
        
        act_map = {"tanh": 0, "sigmoid": 1, "relu": 2, "leaky_relu": 3}
        act_type = act_map.get(self.activation_fn, 0)
        
        if self.sim.use_gpu:
            block = (256,)
            grid = ((H + block[0] - 1) // block[0],)
            self._ffn_kernel(grid, block, (w_dev, x_dev, out_dev, H, W, act_type))
            self.last_output = out_dev.get()
            self.last_activations = np.tile(self.last_output, (W, 1)).T
        else:
            current = x.copy()
            activations = np.zeros((H, W))
            for i in range(W):
                w = weights[:, i]
                z = w * current
                if act_type == 0: current = np.tanh(z)
                elif act_type == 1: current = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
                elif act_type == 2: current = np.maximum(0, z)
                else: current = np.where(z > 0, z, 0.01 * z)
                activations[:, i] = current
            self.last_output = current.copy()
            self.last_activations = activations

    def evaluate_benchmark(self, dataset: object) -> Tuple[float, float]:
        """
        Pure forward pass through V-field. 
        Uses Regularized ELM (Ridge Regression) and Mini-batching for speed and stability.
        NO BACKPROPAGATION.
        """
        if not self.benchmark_mode or dataset is None:
            return 0.0, 10.0

        v_field = self.sim.get_v_np()  # Shape: (H, W)
        H, W = v_field.shape
        
        X_train = dataset.X_train
        y_train = dataset.y_train
        X_test = dataset.X_test
        y_test = dataset.y_test
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        
        # Input projection: Map dataset features to H neurons
        if self._input_proj is None or self._input_proj.shape[0] != num_features:
            scale = np.sqrt(2.0 / num_features)
            self._input_proj = self._rng.standard_normal((num_features, H)) * scale
            
        # Mini-batch Sub-sampling (4x Speed Increase)
        batch_size = min(200, len(X_train))
        test_batch_size = min(100, len(X_test))
        
        idx_train = self._rng.choice(len(X_train), batch_size, replace=False)
        idx_test = self._rng.choice(len(X_test), test_batch_size, replace=False)
        
        X_train_batch = X_train[idx_train]
        y_train_batch = y_train[idx_train]
        X_test_batch = X_test[idx_test]
        y_test_batch = y_test[idx_test]
        
        # Project input data to network width H
        Z_train = X_train_batch @ self._input_proj
        Z_test = X_test_batch @ self._input_proj
        
        # Sanitize V-field
        v_safe = np.nan_to_num(v_field, nan=1.0, posinf=1.0, neginf=0.0)
        
        # Activation function
        act_map = {
            "tanh": np.tanh, 
            "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))), 
            "relu": lambda x: np.maximum(0, x),
            "leaky_relu": lambda x: np.where(x > 0, x, 0.01 * x)
        }
        act_fn = act_map.get(self.activation_fn, np.tanh)

        # Forward pass through V-field columns (layers)
        def _forward_pass(Z):
            for x_col in range(W):
                weights_layer = v_safe[:, x_col]  # Shape: (H,)
                Z = act_fn(Z * weights_layer)
            return Z
        
        H_train = _forward_pass(Z_train)
        H_test = _forward_pass(Z_test)
        
        # Add bias term for the linear output layer
        H_train_b = np.hstack([H_train, np.ones((H_train.shape[0], 1))])
        H_test_b = np.hstack([H_test, np.ones((H_test.shape[0], 1))])
        
        # One-hot encode training labels
        Y_train_oh = np.zeros((len(y_train_batch), num_classes))
        Y_train_oh[np.arange(len(y_train_batch)), y_train_batch] = 1.0
        
        # Ridge Regression (L2 Regularized ELM)
        lambda_ridge = 1e-4  # Regularization strength
        HtH = H_train_b.T @ H_train_b
        HtY = H_train_b.T @ Y_train_oh
        
        try:
            W_out = np.linalg.solve(HtH + lambda_ridge * np.eye(HtH.shape[0]), HtY)
        except np.linalg.LinAlgError:
            W_out = np.linalg.pinv(H_train_b) @ Y_train_oh
        
        # Calculate logits
        logits = H_test_b @ W_out
        
        # Softmax for classification
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Calculate accuracy
        preds = np.argmax(probs, axis=1)
        accuracy = float(np.mean(preds == y_test_batch))
        
        # Calculate cross-entropy loss
        log_probs = np.log(probs[np.arange(len(y_test_batch)), y_test_batch] + 1e-10)
        loss = float(-np.mean(log_probs))
        
        self.last_accuracy = accuracy
        self.last_loss = loss
        return accuracy, loss

    def get_activations_normalized(self) -> Optional[np.ndarray]:
        if self.last_activations is None:
            return None
        a = self.last_activations
        mn, mx = a.min(), a.max()
        if mx - mn < 1e-9:
            return np.zeros_like(a)
        return (a - mn) / (mx - mn)


# ==============================================================================
# 10. OPTIMIZATION ENGINE
# ==============================================================================
class PerturbationEngine:
    def __init__(self, config: OptimizerConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.velocity: Dict[str, float] = {k: 0.0 for k in config.bounds}
        self.temperature = config.initial_temperature
        self.accept_count = 0
        self.reject_count = 0
        self._recent_accepts: List[bool] = []
        self._adaptive_scales: Dict[str, float] = dict(config.param_scales)

    def generate_candidate(self, current_params: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        new_params = dict(current_params)
        new_velocity = dict(self.velocity)
        for key, do_perturb in self.config.perturb_flags.items():
            if not do_perturb:
                continue
            lo, hi = self.config.bounds[key]
            scale = self.temperature * (hi - lo) * self._adaptive_scales[key]
            perturbation = self.rng.normal(0, scale)
            perturbation += self.config.momentum_coeff * self.velocity[key]
            new_velocity[key] = perturbation
            new_val = current_params[key] + perturbation
            
            # Reflective boundary handling
            if new_val < lo: new_val = lo + (lo - new_val)
            elif new_val > hi: new_val = hi - (new_val - hi)
            new_params[key] = float(np.clip(new_val, lo, hi))
        return new_params, new_velocity

    def update_temperature(self, accepted: bool) -> None:
        self._recent_accepts.append(accepted)
        if len(self._recent_accepts) > 50:
            self._recent_accepts.pop(0)
        if accepted: self.accept_count += 1
        else: self.reject_count += 1
        
        if len(self._recent_accepts) > 10:
            accept_rate = sum(self._recent_accepts) / len(self._recent_accepts)
            if accept_rate > self.config.target_acceptance + 0.05:
                self.temperature *= (1.0 + self.config.temp_adjust_rate)
            elif accept_rate < self.config.target_acceptance - 0.05:
                self.temperature *= (1.0 - self.config.temp_adjust_rate)
                
        self.temperature = float(np.clip(
            self.temperature, self.config.min_temperature, self.config.max_temperature
        ))

    def update_adaptive_scales(self, param_importance: Dict[str, float]) -> None:
        for key, importance in param_importance.items():
            if key in self._adaptive_scales:
                self._adaptive_scales[key] = self.config.param_scales[key] / (1.0 + importance)

    def reheat(self) -> None:
        self.temperature = max(self.temperature, 0.05)
        self.velocity = {k: 0.0 for k in self.velocity}
        self._recent_accepts.clear()


class FitnessEvaluator:
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self._grid_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.last_power_spectrum: Optional[np.ndarray] = None

    def _get_grid(self, H: int, W: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (H, W)
        if key not in self._grid_cache:
            cy, cx = H // 2, W // 2
            Y, X = np.ogrid[-cy:H-cy, -cx:W-cx]
            dist = np.sqrt(X**2 + Y**2)
            dist[cy, cx] = np.inf
            self._grid_cache[key] = (Y, X, dist)
        return self._grid_cache[key]

    def evaluate(self, sim: SimulationProtocol, network: NetworkProtocol,
                 warmup_steps: int, benchmark_result: Optional[Tuple[float, float]] = None) -> Tuple[float, Dict[str, float]]:
        
        components = {}
        
        # Benchmark evaluation (if enabled)
        if benchmark_result is not None:
            accuracy, loss = benchmark_result
            components["accuracy"] = accuracy
            components["loss"] = loss
            fitness = -loss * 10.0
            return float(fitness) if np.isfinite(fitness) else -np.inf, components
        
        # Intrinsic evaluation (existing logic for pattern quality)
        if network.last_output is None:
            return 0.0, {}
            
        out = network.last_output
        v_field = sim.get_v_np()
        H, W = v_field.shape
        
        if not np.all(np.isfinite(out)) or not np.all(np.isfinite(v_field)):
            return -np.inf, {}

        var = float(np.var(out))
        mean_abs = float(np.mean(np.abs(out)))
        sparsity = float(np.mean(np.abs(out) > 1e-3))
        grad = 0.0
        if network.last_activations is not None and network.last_activations.shape[1] > 1:
            grad = float(np.mean(np.abs(np.diff(network.last_activations, axis=1))))

        f = np.fft.fft2(v_field)
        fshift = np.fft.fftshift(f)
        power_spectrum = np.abs(fshift)**2
        self.last_power_spectrum = power_spectrum.copy()
        
        cy, cx = H // 2, W // 2
        power_spectrum[cy-1:cy+2, cx-1:cx+2] = 0
        power_sum = np.sum(power_spectrum)
        
        if power_sum < 1e-10:
            return 0.0, {"spectral_entropy": 0.0, "wavelength": 0.0}
            
        prob = power_spectrum / power_sum
        prob = prob[prob > 1e-12]
        spectral_entropy = float(-np.sum(prob * np.log(prob + 1e-10)))
        max_entropy = np.log(prob.size)
        norm_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0.0
        entropy_score = float(np.exp(-10 * (norm_entropy - 0.55)**2))

        _, _, dist = self._get_grid(H, W)
        mask = (dist > 3) & (dist < H // 4)
        wavelength_score = 0.1
        wavelength = 0.0
        if np.any(mask):
            masked_power = power_spectrum * mask
            max_idx = np.unravel_index(np.argmax(masked_power), masked_power.shape)
            dominant_freq = dist[max_idx]
            wavelength = H / dominant_freq if dominant_freq > 0 else 0.0
            if 8 <= wavelength <= 40: wavelength_score = 1.0
            elif 5 <= wavelength < 8 or 40 < wavelength <= 60: wavelength_score = 0.5

        v_std = float(np.std(v_field))
        v_gradient = float(np.mean(np.abs(np.gradient(v_field))))
        pattern = v_std * (1.0 + v_gradient)

        components = {
            "var": var, "mean": mean_abs, "grad": grad, "sparsity": sparsity,
            "pattern": pattern, "spectral_entropy": norm_entropy, "wavelength": wavelength,
        }
        fitness = (
            self.config.w_var * var + self.config.w_mean * mean_abs +
            self.config.w_grad * grad + self.config.w_sparsity * sparsity +
            self.config.w_pattern * pattern +
            self.config.w_spectral_entropy * entropy_score +
            self.config.w_wavelength * wavelength_score
        )
        return float(fitness) if np.isfinite(fitness) else -np.inf, components


class PopulationManager:
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.population: List[Candidate] = []
        self._candidate_counter = 0

    def compute_diversity_bonus(self, candidate_params: Dict[str, float]) -> float:
        if not self.population:
            return 0.0
        total_sharing = 0.0
        for c in self.population:
            dist_sq = 0.0
            for key in candidate_params:
                lo, hi = self.config.bounds[key]
                scale = hi - lo
                dist_sq += ((candidate_params[key] - c.params[key]) / scale) ** 2
            dist = np.sqrt(dist_sq)
            sigma = self.config.fitness_sharing_sigma
            if dist < sigma:
                sharing = (1.0 - (dist / sigma) ** self.config.fitness_sharing_alpha)
                total_sharing += sharing
        return float(1.0 / (1.0 + total_sharing)) * self.config.diversity_weight

    def update_population(self, new_candidate: Candidate) -> None:
        self._candidate_counter += 1
        new_candidate = Candidate(
            params=new_candidate.params, fitness=new_candidate.fitness,
            fitness_components=new_candidate.fitness_components, age=0,
            candidate_id=self._candidate_counter,
        )
        aged_population = []
        for c in self.population:
            aged_population.append(Candidate(
                params=c.params, fitness=c.fitness,
                fitness_components=c.fitness_components, age=c.age + 1,
                candidate_id=c.candidate_id,
            ))
        aged_population.append(new_candidate)
        aged_population.sort(key=lambda c: c.fitness, reverse=True)
        self.population = aged_population[:self.config.population_size]

    def get_best(self) -> Optional[Candidate]:
        return self.population[0] if self.population else None

    def get_population_diversity(self) -> float:
        if len(self.population) < 2:
            return 0.0
        total_dist = 0.0
        count = 0
        for i, c1 in enumerate(self.population):
            for c2 in self.population[i+1:]:
                dist_sq = 0.0
                for key in c1.params:
                    lo, hi = self.config.bounds[key]
                    scale = hi - lo
                    dist_sq += ((c1.params[key] - c2.params[key]) / scale) ** 2
                total_dist += np.sqrt(dist_sq)
                count += 1
        return total_dist / count if count > 0 else 0.0


class MonteCarloOptimizer:
    def __init__(self, sim: SimulationProtocol, network: NetworkProtocol,
                 config: Optional[OptimizerConfig] = None,
                 checkpoint_dir: Optional[Path] = None):
        self.sim = sim
        self.network = network
        self.config = config or OptimizerConfig()
        self.checkpoint_dir = checkpoint_dir
        self.rng = np.random.default_rng()
        
        self.perturbation = PerturbationEngine(self.config, self.rng)
        self.evaluator = FitnessEvaluator(self.config)
        self.population_mgr = PopulationManager(self.config)
        
        self.current_params = sim.get_params()
        self.current_fitness = 0.0
        self.best_candidate: Optional[Candidate] = None
        self.total_steps = 0
        self.steps_without_improvement = 0
        self.restart_count = 0
        self.status = OptimizationStatus.RUNNING
        
        self.fitness_history: List[float] = []
        self.param_history: List[Dict[str, float]] = []
        self._recent_fitnesses: List[float] = []
        self.enabled = False
        
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._load_checkpoint()

    def step(self, warmup_steps: int = 200, benchmark_result: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        if self.status != OptimizationStatus.RUNNING:
            return {"status": self.status.value, "step": self.total_steps}
            
        self.total_steps += 1
        
        # Hybrid Differential Evolution (DE) / Monte Carlo Search
        if len(self.population_mgr.population) >= 3 and self.rng.random() < 0.8:
            pop = self.population_mgr.population
            idxs = self.rng.choice(len(pop), 3, replace=False)
            a_params, b_params, c_params = pop[idxs[0]].params, pop[idxs[1]].params, pop[idxs[2]].params
            best_params = self.population_mgr.get_best().params
            
            # Adaptive Jitter to prevent premature convergence
            current_diversity = self.population_mgr.get_population_diversity()
            jitter_scale = 0.0
            if current_diversity < 0.05:
                # If population collapses, inject scaled random noise to keep evolution alive
                jitter_scale = 0.05 * (0.05 - current_diversity) 
                
            candidate_params = {}
            new_velocity = {}
            for key in self.config.perturb_flags:
                if not self.config.perturb_flags[key]:
                    candidate_params[key] = self.current_params[key]
                    new_velocity[key] = 0.0
                    continue
                
                lo, hi = self.config.bounds[key]
                F_scale = 0.4 + 0.5 * self.rng.random()  # DE mutation factor [0.4, 0.9]
                # DE/best/1 mutation strategy
                mutated_val = best_params[key] + F_scale * (b_params[key] - c_params[key])
                
                # Inject jitter if diversity is too low
                if jitter_scale > 0:
                    mutated_val += self.rng.normal(0, jitter_scale * (hi - lo))
                
                # Binomial crossover
                if self.rng.random() < 0.9:
                    candidate_params[key] = float(np.clip(mutated_val, lo, hi))
                else:
                    candidate_params[key] = self.current_params[key]
                    
                new_velocity[key] = candidate_params[key] - self.current_params[key]
        else:
            # Fallback to standard adaptive MC if population is too small
            candidate_params, new_velocity = self.perturbation.generate_candidate(self.current_params)

        sim_state = self.sim.save_state()
        
        try:
            self.sim.set_params(candidate_params)
            self.sim.step(warmup_steps)
            
            if benchmark_result is not None:
                base_fitness, components = self.evaluator.evaluate(
                    self.sim, self.network, warmup_steps, benchmark_result=benchmark_result
                )
            else:
                self.network.test()
                base_fitness, components = self.evaluator.evaluate(self.sim, self.network, warmup_steps)
                
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            base_fitness = -np.inf
            components = {}
        finally:
            self.sim.restore_state(sim_state)
            
        if not np.isfinite(base_fitness):
            self.perturbation.update_temperature(False)
            self.steps_without_improvement += 1
            return self._build_result(False, base_fitness, components)
            
        diversity_bonus = self.population_mgr.compute_diversity_bonus(candidate_params)
        adjusted_fitness = base_fitness + diversity_bonus
        delta = adjusted_fitness - self.current_fitness
        accepted = delta > 0 or self.rng.random() < np.exp(delta / max(self.perturbation.temperature, 1e-9))
        
        if accepted:
            self.current_params = candidate_params
            self.current_fitness = adjusted_fitness
            self.perturbation.velocity = new_velocity
            self.steps_without_improvement = 0
            
            new_candidate = Candidate(
                params=candidate_params, fitness=base_fitness,
                fitness_components=components, age=0,
            )
            self.population_mgr.update_population(new_candidate)
            
            current_best = self.population_mgr.get_best()
            if current_best and (self.best_candidate is None or current_best.fitness > self.best_candidate.fitness):
                self.best_candidate = current_best
                self.sim.set_params(current_best.params)
        else:
            self.steps_without_improvement += 1
            
        self.perturbation.update_temperature(accepted)
        self._check_convergence(adjusted_fitness)
        
        if self._should_restart():
            self._trigger_restart()
            
        self.fitness_history.append(self.current_fitness)
        self.param_history.append(dict(self.current_params))
        self._recent_fitnesses.append(adjusted_fitness)
        if len(self._recent_fitnesses) > self.config.convergence_window:
            self._recent_fitnesses.pop(0)
            
        if self.checkpoint_dir and self.total_steps % 100 == 0:
            self._save_checkpoint()
            
        return self._build_result(accepted, adjusted_fitness, components)

    def _should_restart(self) -> bool:
        if self.steps_without_improvement < self.config.restart_threshold:
            return False
        diversity = self.population_mgr.get_population_diversity()
        if diversity < 0.01:
            return True
        if self.restart_count >= self.config.max_restarts:
            self.status = OptimizationStatus.STALLED
            return False
        return True

    def _check_convergence(self, fitness: float) -> None:
        if len(self._recent_fitnesses) < self.config.convergence_window:
            return
        recent = self._recent_fitnesses[-self.config.convergence_window:]
        fitness_range = max(recent) - min(recent)
        if fitness_range < self.config.convergence_threshold:
            self.status = OptimizationStatus.CONVERGED

    def _build_result(self, accepted: bool, fitness: float, components: Dict[str, float]) -> Dict[str, Any]:
        return {
            "step": self.total_steps, "status": self.status.value,
            "accepted": accepted, "fitness": fitness,
            "best_fitness": self.best_candidate.fitness if self.best_candidate else 0.0,
            "temperature": self.perturbation.temperature,
            "restarts": self.restart_count,
            "population_diversity": self.population_mgr.get_population_diversity(),
            "components": components,
        }

    def _trigger_restart(self) -> None:
        self.restart_count += 1
        self.steps_without_improvement = 0
        self.perturbation.reheat()
        
        # Reset the Input Projection on restart
        # This forces the network to try a new structural "lottery ticket"
        if hasattr(self.network, '_input_proj'):
            self.network._input_proj = None 
            
        if self.best_candidate:
            restart_params = dict(self.best_candidate.params)
            param_importance = self._analyze_parameter_importance()
            self.perturbation.update_adaptive_scales(param_importance)
            
            for key in restart_params:
                lo, hi = self.config.bounds[key]
                importance = param_importance.get(key, 1.0)
                scale = 0.1 * (hi - lo) / (1.0 + importance)
                restart_params[key] += self.rng.normal(0, scale)
                restart_params[key] = float(np.clip(restart_params[key], lo, hi))
                
            self.current_params = restart_params
            self.current_fitness = 0.0
            self.population_mgr.population = [
                Candidate(params=dict(self.best_candidate.params),
                          fitness=self.best_candidate.fitness,
                          fitness_components=self.best_candidate.fitness_components, age=0)
            ]

    def _analyze_parameter_importance(self) -> Dict[str, float]:
        if len(self.param_history) < 20:
            return {k: 1.0 for k in self.config.bounds}
            
        recent_params = self.param_history[-20:]
        recent_fitnesses = self.fitness_history[-20:]
        importance = {}
        
        for key in self.config.bounds:
            param_values = [p[key] for p in recent_params]
            param_std = np.std(param_values)
            fitness_std = np.std(recent_fitnesses)
            if param_std > 1e-6 and fitness_std > 1e-6:
                corr = np.corrcoef(param_values, recent_fitnesses)[0, 1]
                importance[key] = abs(corr) if np.isfinite(corr) else 1.0
            else:
                importance[key] = 1.0
        return importance

    def _save_checkpoint(self) -> None:
        if not self.checkpoint_dir:
            return
        checkpoint = {
            "total_steps": self.total_steps, "current_params": self.current_params,
            "current_fitness": self.current_fitness, "best_candidate": self.best_candidate,
            "temperature": self.perturbation.temperature, "restart_count": self.restart_count,
            "status": self.status.value,
            "fitness_history": self.fitness_history[-1000:],
            "param_history": self.param_history[-1000:],
        }
        with open(self.checkpoint_dir / "optimizer_checkpoint.pkl", "wb") as f:
            pickle.dump(checkpoint, f)

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_dir:
            return
        checkpoint_file = self.checkpoint_dir / "optimizer_checkpoint.pkl"
        if not checkpoint_file.exists():
            return
        try:
            with open(checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)
            self.total_steps = checkpoint.get("total_steps", 0)
            self.current_params = checkpoint.get("current_params", self.sim.get_params())
            self.current_fitness = checkpoint.get("current_fitness", 0.0)
            self.best_candidate = checkpoint.get("best_candidate")
            self.perturbation.temperature = checkpoint.get("temperature", self.config.initial_temperature)
            self.restart_count = checkpoint.get("restart_count", 0)
            self.status = OptimizationStatus(checkpoint.get("status", "running"))
            self.fitness_history = checkpoint.get("fitness_history", [])
            self.param_history = checkpoint.get("param_history", [])
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

    def export_pareto_front(self, filepath: Path) -> None:
        pareto_data = [{
            "params": c.params, "fitness": c.fitness,
            "components": c.fitness_components, "age": c.age,
        } for c in self.population_mgr.population]
        with open(filepath, "w") as f:
            json.dump(pareto_data, f, indent=2)

    def export_csv(self, path: str) -> None:
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "F", "k", "Du", "Dv", "dt", "fitness"])
            for i, (fit, p) in enumerate(zip(self.fitness_history, self.param_history)):
                w.writerow([i, p["F"], p["k"], p["Du"], p["Dv"], p["dt"], fit])

    def reset(self) -> None:
        self.current_params = self.sim.get_params()
        self.current_fitness = 0.0
        self.best_candidate = None
        self.total_steps = 0
        self.steps_without_improvement = 0
        self.restart_count = 0
        self.status = OptimizationStatus.RUNNING
        self.fitness_history.clear()
        self.param_history.clear()
        self._recent_fitnesses.clear()
        self.population_mgr.population.clear()
        self.perturbation.temperature = self.config.initial_temperature
        self.perturbation.velocity = {k: 0.0 for k in self.current_params}
        self.perturbation.accept_count = 0
        self.perturbation.reject_count = 0


# ==============================================================================
# 11. GUI UTILITIES & CUSTOM WIDGETS
# ==============================================================================
class StatusLED(QWidget):
    """Animated status indicator LED."""
    def __init__(self, color: str = "green", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self.color = color
        self.active = False

    def set_active(self, active: bool) -> None:
        self.active = active
        self.update()

    def set_color(self, color: str) -> None:
        self.color = color
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = min(self.width(), self.height())
        center = QPoint(size // 2, size // 2)
        radius = size // 2 - 1

        color_map = {
            "green": (76, 175, 80), "red": (244, 67, 54), "yellow": (255, 193, 7),
            "blue": (79, 195, 247), "gray": (100, 100, 110),
        }
        r, g, b = color_map.get(self.color, color_map["gray"])

        if self.active:
            glow = QRadialGradient(center, radius * 2)
            glow.setColorAt(0, QColor(r, g, b, 180))
            glow.setColorAt(0.5, QColor(r, g, b, 60))
            glow.setColorAt(1, QColor(r, g, b, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center, radius * 2, radius * 2)

        grad = QRadialGradient(center - QPoint(radius//3, radius//3), radius * 1.5)
        if self.active:
            grad.setColorAt(0, QColor(min(255, r+80), min(255, g+80), min(255, b+80)))
            grad.setColorAt(1, QColor(r, g, b))
        else:
            grad.setColorAt(0, QColor(80, 80, 90))
            grad.setColorAt(1, QColor(40, 40, 50))
            
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(20, 20, 25), 1))
        painter.drawEllipse(center, radius, radius)
        painter.end()


class CollapsibleSection(QWidget):
    """Collapsible sidebar section with header."""
    def __init__(self, title: str, icon: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layout_main = QVBoxLayout(self)
        self.layout_main.setContentsMargins(0, 0, 0, 0)
        self.layout_main.setSpacing(0)

        self.header = QPushButton(f"  {icon}  {title}" if icon else f"  {title}")
        self.header.setObjectName("section-header")
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.clicked.connect(self._toggle)
        self.layout_main.addWidget(self.header)

        self.body = QWidget()
        self.body.setObjectName("section-body")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(10, 10, 10, 10)
        self.body_layout.setSpacing(8)
        self.layout_main.addWidget(self.body)

    def add_widget(self, widget: QWidget) -> None:
        self.body_layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        container = QWidget()
        container.setLayout(layout)
        container.setObjectName("section-body")
        self.body_layout.addWidget(container)

    @Slot()
    def _toggle(self) -> None:
        self.body.setVisible(self.header.isChecked())


class MetricCard(QWidget):
    """Compact metric display card."""
    def __init__(self, label: str, value: str = "—", color: str = "#4fc3f7", parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)

        self.lbl_label = QLabel(label.upper())
        self.lbl_label.setObjectName("metric-label")
        layout.addWidget(self.lbl_label)

        self.lbl_value = QLabel(value)
        self.lbl_value.setObjectName("metric-value")
        self.lbl_value.setStyleSheet(f"color: {color};")
        layout.addWidget(self.lbl_value)

        self.setStyleSheet("background-color: #1e1e26; border-radius: 6px; border: 1px solid #2a2a35;")

    def set_value(self, value: str) -> None:
        self.lbl_value.setText(value)


# ==============================================================================
# 12. GUI PLOTS & DISPLAY
# ==============================================================================
class DisplayWidget(QWidget):
    """Main simulation display with overlay rendering and safe array handling."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(256, 256)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap = QPixmap()
        self._colormap = COLORMAPS["Ink"]()
        self._overlay = None
        self._overlay_opacity = 0.0
        self._ghost_v: Optional[np.ndarray] = None
        self._ghost_opacity = 0.0
        self.clicked = None
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self._info_text = ""

    def set_colormap(self, name: str) -> None:
        if name in COLORMAPS:
            self._colormap = COLORMAPS[name]()

    def set_overlay(self, name: str, opacity: float) -> None:
        if name in OVERLAY_CMAPS and opacity > 0:
            self._overlay = OVERLAY_CMAPS[name]()
            self._overlay_opacity = opacity
        else:
            self._overlay = None
            self._overlay_opacity = 0.0

    def set_ghost(self, v: Optional[np.ndarray], opacity: float) -> None:
        self._ghost_v = v
        self._ghost_opacity = opacity

    def set_info(self, text: str) -> None:
        self._info_text = text
        self.update()

    def update_image(self, v: np.ndarray) -> None:
        H, W = v.shape
        v_safe = np.nan_to_num(v, nan=AppConstants.NAN_REPLACE, 
                               posinf=AppConstants.POSINF_REPLACE, 
                               neginf=AppConstants.NEGINF_REPLACE)
        v_norm = np.clip(v_safe, 0.0, 1.0)
        indices = (v_norm * 255).astype(np.uint8)
        
        rgb = self._colormap[indices]
        
        if self._overlay is not None and self._overlay_opacity > 0:
            overlay_rgb = self._overlay[indices]
            alpha = self._overlay_opacity
            rgb = ((1 - alpha) * rgb + alpha * overlay_rgb).astype(np.uint8)
            
        if self._ghost_v is not None and self._ghost_opacity > 0:
            g_safe = np.nan_to_num(self._ghost_v, nan=AppConstants.NAN_REPLACE, 
                                   posinf=AppConstants.POSINF_REPLACE, 
                                   neginf=AppConstants.NEGINF_REPLACE)
            g_norm = np.clip(g_safe, 0.0, 1.0)
            g_idx = (g_norm * 255).astype(np.uint8)
            g_rgb = self._colormap[g_idx]
            alpha = self._ghost_opacity
            rgb = ((1 - alpha) * rgb + alpha * g_rgb).astype(np.uint8)
            
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(10, 10, 14))
        
        if not self._pixmap.isNull():
            scaled = self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            
        if self._info_text:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(10, 10, 14, 180)))
            font = QFont("Consolas", 10)
            font.setStyleHint(QFont.StyleHint.Monospace)
            painter.setFont(font)
            fm = painter.fontMetrics()
            lines = self._info_text.split("\n")
            max_w = max(fm.horizontalAdvance(line) for line in lines)
            padding = 10
            box_h = fm.height() * len(lines) + padding * 2
            box_w = max_w + padding * 2
            painter.drawRoundedRect(10, 10, box_w, box_h, 4, 4)
            painter.setPen(QColor(79, 195, 247))
            for i, line in enumerate(lines):
                painter.drawText(10 + padding, 10 + padding + fm.ascent() + i * fm.height(), line)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.clicked:
            pos = event.position().toPoint()
            self.clicked(pos.x(), pos.y())


class FitnessPlot(QWidget):
    """Real-time fitness history plot with gradient fill."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(300, 150)
        self.history: List[float] = []
        self.best_history: List[float] = []
        self.max_points = 500

    def update_data(self, history: List[float], best_history: List[float]) -> None:
        self.history = history[-self.max_points:]
        self.best_history = best_history[-self.max_points:]
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(20, 20, 25))
        w, h = self.width(), self.height()
        margin = 40
        plot_w = w - 2 * margin
        plot_h = h - 2 * margin

        if not self.history:
            painter.setPen(QColor(80, 80, 90))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for MC data...")
            painter.end()
            return

        all_vals = self.history + self.best_history
        mn, mx = min(all_vals), max(all_vals)
        if mx - mn < 1e-9:
            mx = mn + 1.0

        painter.setPen(QPen(QColor(40, 40, 50), 1, Qt.PenStyle.DotLine))
        for i in range(5):
            y = margin + int(plot_h * i / 4)
            painter.drawLine(margin, y, margin + plot_w, y)
            val = mx - (mx - mn) * i / 4
            painter.setPen(QColor(100, 100, 110))
            painter.setFont(QFont("Consolas", 9))
            painter.drawText(5, y + 4, f"{val:.3f}")

        if len(self.best_history) > 1:
            path = QPainterPath()
            n = len(self.best_history)
            x0 = margin
            y0 = margin + plot_h - int(plot_h * (self.best_history[0] - mn) / (mx - mn))
            path.moveTo(x0, y0)
            for i in range(1, n):
                x = margin + int(plot_w * i / max(1, n - 1))
                y = margin + plot_h - int(plot_h * (self.best_history[i] - mn) / (mx - mn))
                path.lineTo(x, y)
            path.lineTo(margin + plot_w, margin + plot_h)
            path.lineTo(margin, margin + plot_h)
            path.closeSubpath()
            grad = QLinearGradient(0, margin, 0, margin + plot_h)
            grad.setColorAt(0, QColor(255, 180, 80, 60))
            grad.setColorAt(1, QColor(255, 180, 80, 0))
            painter.fillPath(path, QBrush(grad))
            painter.setPen(QPen(QColor(255, 180, 80), 2))
            painter.drawPath(path)

        if len(self.history) > 1:
            path = QPainterPath()
            n = len(self.history)
            x0 = margin
            y0 = margin + plot_h - int(plot_h * (self.history[0] - mn) / (mx - mn))
            path.moveTo(x0, y0)
            for i in range(1, n):
                x = margin + int(plot_w * i / max(1, n - 1))
                y = margin + plot_h - int(plot_h * (self.history[i] - mn) / (mx - mn))
                path.lineTo(x, y)
            painter.setPen(QPen(QColor(79, 195, 247), 2))
            painter.drawPath(path)

        painter.setFont(QFont("Segoe UI", 10))
        painter.setPen(QPen(QColor(79, 195, 247), 2))
        painter.drawLine(w - 180, 15, w - 160, 15)
        painter.setPen(QColor(200, 200, 210))
        painter.drawText(w - 155, 19, "Current")
        painter.setPen(QPen(QColor(255, 180, 80), 2))
        painter.drawLine(w - 180, 32, w - 160, 32)
        painter.setPen(QColor(200, 200, 210))
        painter.drawText(w - 155, 36, "Best")
        painter.end()


class PhaseSpacePlot(QWidget):
    """F vs k trajectory plot with gradient trail."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(300, 150)
        self.param_history: List[Dict[str, float]] = []
        self.max_points = 1000

    def update_data(self, param_history: List[Dict[str, float]]) -> None:
        self.param_history = param_history[-self.max_points:]
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(20, 20, 25))
        w, h = self.width(), self.height()
        margin = 40
        plot_w = w - 2 * margin
        plot_h = h - 2 * margin
        F_min, F_max = 0.0, 0.1
        k_min, k_max = 0.0, 0.08

        painter.setPen(QPen(QColor(40, 40, 50), 1, Qt.PenStyle.DotLine))
        for i in range(5):
            y = margin + int(plot_h * i / 4)
            painter.drawLine(margin, y, margin + plot_w, y)
            x = margin + int(plot_w * i / 4)
            painter.drawLine(x, margin, x, margin + plot_h)

        painter.setPen(QColor(100, 100, 110))
        painter.setFont(QFont("Consolas", 9))
        for i in range(5):
            F_val = F_min + (F_max - F_min) * i / 4
            x = margin + int(plot_w * i / 4)
            painter.drawText(x - 15, h - margin + 18, f"{F_val:.3f}")
            k_val = k_max - (k_max - k_min) * i / 4
            y = margin + int(plot_h * i / 4)
            painter.drawText(5, y + 4, f"{k_val:.3f}")

        painter.setFont(QFont("Segoe UI", 10))
        painter.setPen(QColor(150, 150, 160))
        painter.drawText(margin + plot_w // 2 - 5, h - 5, "F (feed rate)")
        painter.save()
        painter.translate(12, margin + plot_h // 2 + 30)
        painter.rotate(-90)
        painter.drawText(0, 0, "k (kill rate)")
        painter.restore()

        if not self.param_history:
            painter.setPen(QColor(80, 80, 90))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for trajectory...")
            painter.end()
            return

        F_vals = [p["F"] for p in self.param_history]
        k_vals = [p["k"] for p in self.param_history]
        n = len(F_vals)

        for i in range(n - 1):
            x1 = margin + int(plot_w * (F_vals[i] - F_min) / (F_max - F_min))
            y1 = margin + plot_h - int(plot_h * (k_vals[i] - k_min) / (k_max - k_min))
            x2 = margin + int(plot_w * (F_vals[i + 1] - F_min) / (F_max - F_min))
            y2 = margin + plot_h - int(plot_h * (k_vals[i + 1] - k_min) / (k_max - k_min))
            alpha = int(40 + 200 * i / max(1, n - 1))
            painter.setPen(QPen(QColor(79, 195, 247, alpha), 2))
            painter.drawLine(x1, y1, x2, y2)

        if n > 0:
            cx = margin + int(plot_w * (F_vals[-1] - F_min) / (F_max - F_min))
            cy = margin + plot_h - int(plot_h * (k_vals[-1] - k_min) / (k_max - k_min))
            glow = QRadialGradient(cx, cy, 15)
            glow.setColorAt(0, QColor(79, 195, 247, 150))
            glow.setColorAt(1, QColor(79, 195, 247, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPoint(cx, cy), 15, 15)
            painter.setBrush(QColor(255, 100, 100))
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawEllipse(QPoint(cx, cy), 5, 5)
        painter.end()


class ParameterTimeSeriesPlot(QWidget):
    """Parameter values over time with individual scaling."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(300, 150)
        self.param_history: List[Dict[str, float]] = []
        self.max_points = 500
        self.colors = {
            "F": QColor(255, 100, 100), "k": QColor(100, 255, 150),
            "Du": QColor(100, 150, 255), "Dv": QColor(255, 200, 100),
        }

    def update_data(self, param_history: List[Dict[str, float]]) -> None:
        self.param_history = param_history[-self.max_points:]
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(20, 20, 25))
        w, h = self.width(), self.height()
        margin = 40
        plot_w = w - 2 * margin
        plot_h = h - 2 * margin

        if not self.param_history:
            painter.setPen(QColor(80, 80, 90))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for data...")
            painter.end()
            return

        painter.setPen(QPen(QColor(40, 40, 50), 1, Qt.PenStyle.DotLine))
        for i in range(5):
            y = margin + int(plot_h * i / 4)
            painter.drawLine(margin, y, margin + plot_w, y)

        for key, color in self.colors.items():
            vals = [p.get(key, 0) for p in self.param_history]
            if not vals:
                continue
            mn, mx = min(vals), max(vals)
            if mx - mn < 1e-9:
                mx = mn + 1.0
            n = len(vals)
            painter.setPen(QPen(color, 2))
            for i in range(n - 1):
                x1 = margin + int(plot_w * i / max(1, n - 1))
                y1 = margin + plot_h - int(plot_h * (vals[i] - mn) / (mx - mn))
                x2 = margin + int(plot_w * (i + 1) / max(1, n - 1))
                y2 = margin + plot_h - int(plot_h * (vals[i + 1] - mn) / (mx - mn))
                painter.drawLine(x1, y1, x2, y2)

        painter.setFont(QFont("Segoe UI", 10))
        y_legend = 15
        for key, color in self.colors.items():
            painter.setPen(QPen(color, 3))
            painter.drawLine(w - 100, y_legend, w - 80, y_legend)
            painter.setPen(QColor(200, 200, 210))
            painter.drawText(w - 75, y_legend + 4, key)
            y_legend += 18
        painter.end()


class SpectrumPlot(QWidget):
    """2D Fourier power spectrum visualization."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(300, 150)
        self._power_spectrum: Optional[np.ndarray] = None

    def update_data(self, power_spectrum: Optional[np.ndarray]) -> None:
        self._power_spectrum = power_spectrum
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 25))
        if self._power_spectrum is None:
            painter.setPen(QColor(80, 80, 90))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for spectrum...")
            painter.end()
            return

        ps = self._power_spectrum
        H, W = ps.shape
        ps_log = np.log1p(ps)
        ps_norm = ps_log / (ps_log.max() + 1e-10)
        ps_safe = np.nan_to_num(ps_norm, nan=0.0, posinf=1.0, neginf=0.0)
        indices = (np.clip(ps_safe, 0, 1) * 255).astype(np.uint8)

        t = np.linspace(0, 1, 256)
        r = np.clip(1.60*t - 0.60*t**2, 0, 1)
        g = np.clip(0.50*t + 0.80*t**2 - 0.30*t**3, 0, 1)
        b = np.clip(0.05 + 1.20*t - 2.00*t**2 + 0.80*t**3, 0, 1)
        cmap = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
        rgb = cmap[indices]

        h, w_img, _ = rgb.shape
        qimg = QImage(rgb.data, w_img, h, 3 * w_img, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

        cx = x + scaled.width() // 2
        cy = y + scaled.height() // 2
        painter.setPen(QPen(QColor(79, 195, 247, 180), 1))
        painter.drawLine(cx - 8, cy, cx + 8, cy)
        painter.drawLine(cx, cy - 8, cx, cy + 8)

        painter.setPen(QColor(200, 200, 210))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(10, 20, "Fourier Power Spectrum (log scale)")
        painter.end()


class ActivationHeatmap(QWidget):
    """Network activation heatmap."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(300, 150)
        self._activations: Optional[np.ndarray] = None

    def update_data(self, activations: Optional[np.ndarray]) -> None:
        self._activations = activations
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 25))
        if self._activations is None:
            painter.setPen(QColor(80, 80, 90))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for activations...")
            painter.end()
            return
            
        a = self._activations
        H, W = a.shape
        a_safe = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
        a_norm = np.clip(a_safe, 0.0, 1.0)
        indices = (a_norm * 255).astype(np.uint8)
        
        rgb = COLORMAPS["Viridis"]()[indices]
        h, w_img, _ = rgb.shape
        qimg = QImage(rgb.data, w_img, h, 3 * w_img, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.setPen(QColor(200, 200, 210))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(10, 20, f"Activations ({H}×{W})")
        painter.end()


class NetworkDiagram(QWidget):
    """Network output bar chart."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(300, 150)
        self._output: Optional[np.ndarray] = None

    def update_data(self, output: Optional[np.ndarray]) -> None:
        self._output = output
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(20, 20, 25))
        if self._output is None:
            painter.setPen(QColor(80, 80, 90))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for output...")
            painter.end()
            return
            
        w, h = self.width(), self.height()
        margin = 40
        plot_w = w - 2 * margin
        plot_h = h - 2 * margin
        out = self._output
        n = len(out)
        mn, mx = out.min(), out.max()
        if mx - mn < 1e-9:
            mx = mn + 1.0
            
        bar_w = max(1, plot_w // n - 1)
        for i in range(n):
            val = (out[i] - mn) / (mx - mn)
            bar_h = int(plot_h * val)
            x = margin + int(plot_w * i / n)
            y = margin + plot_h - bar_h
            grad = QLinearGradient(x, y, x, y + bar_h)
            grad.setColorAt(0, QColor(79, 195, 247))
            grad.setColorAt(1, QColor(13, 71, 161))
            painter.fillRect(x, y, bar_w, bar_h, QBrush(grad))
            
        painter.setPen(QColor(200, 200, 210))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(10, 20, f"Output ({n} neurons)")
        painter.drawText(10, h - 10, f"range: [{mn:.2f}, {mx:.2f}]")
        painter.end()


class PopulationPlot(QWidget):
    """Population candidates in F-k space with fitness coloring."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(300, 150)
        self.population: List[Candidate] = []
        self.best: Optional[Candidate] = None

    def update_data(self, population: List[Candidate], best: Optional[Candidate]) -> None:
        self.population = population
        self.best = best
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(20, 20, 25))
        w, h = self.width(), self.height()
        margin = 40
        plot_w = w - 2 * margin
        plot_h = h - 2 * margin
        F_min, F_max = 0.0, 0.1
        k_min, k_max = 0.0, 0.08

        painter.setPen(QPen(QColor(40, 40, 50), 1, Qt.PenStyle.DotLine))
        for i in range(5):
            y = margin + int(plot_h * i / 4)
            painter.drawLine(margin, y, margin + plot_w, y)
            x = margin + int(plot_w * i / 4)
            painter.drawLine(x, margin, x, margin + plot_h)

        painter.setPen(QColor(100, 100, 110))
        painter.setFont(QFont("Consolas", 9))
        for i in range(5):
            F_val = F_min + (F_max - F_min) * i / 4
            x = margin + int(plot_w * i / 4)
            painter.drawText(x - 15, h - margin + 18, f"{F_val:.3f}")
            k_val = k_max - (k_max - k_min) * i / 4
            y = margin + int(plot_h * i / 4)
            painter.drawText(5, y + 4, f"{k_val:.3f}")

        painter.setFont(QFont("Segoe UI", 10))
        painter.setPen(QColor(150, 150, 160))
        painter.drawText(margin + plot_w // 2 - 5, h - 5, "F (feed rate)")
        painter.save()
        painter.translate(12, margin + plot_h // 2 + 30)
        painter.rotate(-90)
        painter.drawText(0, 0, "k (kill rate)")
        painter.restore()

        if not self.population:
            painter.setPen(QColor(80, 80, 90))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Population empty")
            painter.end()
            return

        fits = [c.fitness for c in self.population]
        f_min, f_max = min(fits), max(fits)
        if f_max - f_min < 1e-9:
            f_max = f_min + 1.0

        for c in self.population:
            cx = margin + int(plot_w * (c.params["F"] - F_min) / (F_max - F_min))
            cy = margin + plot_h - int(plot_h * (c.params["k"] - k_min) / (k_max - k_min))
            t = (c.fitness - f_min) / (f_max - f_min)
            color = QColor(int(30 + 50 * t), int(100 + 150 * t), 247)
            size = max(4, 12 - c.age // 5)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
            painter.drawEllipse(QPoint(cx, cy), size, size)

        if self.best:
            cx = margin + int(plot_w * (self.best.params["F"] - F_min) / (F_max - F_min))
            cy = margin + plot_h - int(plot_h * (self.best.params["k"] - k_min) / (k_max - k_min))
            glow = QRadialGradient(cx, cy, 20)
            glow.setColorAt(0, QColor(255, 180, 80, 200))
            glow.setColorAt(1, QColor(255, 180, 80, 0))
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPoint(cx, cy), 20, 20)
            painter.setBrush(QColor(255, 180, 80))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawEllipse(QPoint(cx, cy), 7, 7)
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(cx + 12, cy - 5, "BEST")
        painter.end()


class PCAPlot(QWidget):
    """PCA visualization of population in principal component space."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(300, 150)
        self.population: List[Candidate] = []
        self.best: Optional[Candidate] = None
        self._pca_result: Optional[Dict[str, Any]] = None

    def update_data(self, population: List[Candidate], best: Optional[Candidate]) -> None:
        self.population = population
        self.best = best
        self._compute_pca()
        self.update()

    def _compute_pca(self) -> None:
        """Compute PCA on population parameters."""
        if len(self.population) < 2:
            self._pca_result = None
            return

        param_keys = list(self.population[0].params.keys())
        X = np.array([[c.params[k] for k in param_keys] for c in self.population])

        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std < 1e-10] = 1.0  # Avoid division by zero
        X_std = (X - mean) / std

        cov = np.cov(X_std, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        if len(eigenvalues) >= 2:
            projections = X_std @ eigenvectors[:, :2]
            total_var = np.sum(eigenvalues)
            var_explained = eigenvalues[:2] / total_var if total_var > 0 else np.array([0.5, 0.5])

            self._pca_result = {
                "projections": projections, "var_explained": var_explained,
                "param_keys": param_keys, "mean": mean, "std": std,
            }
        else:
            self._pca_result = None

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(20, 20, 25))
        w, h = self.width(), self.height()
        margin = 50
        plot_w = w - 2 * margin
        plot_h = h - 2 * margin

        if self._pca_result is None or len(self.population) < 2:
            painter.setPen(QColor(80, 80, 90))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Need ≥2 candidates for PCA")
            painter.end()
            return

        proj = self._pca_result["projections"]
        var_exp = self._pca_result["var_explained"]

        pc1_min, pc1_max = proj[:, 0].min(), proj[:, 0].max()
        pc2_min, pc2_max = proj[:, 1].min(), proj[:, 1].max()

        pc1_pad = (pc1_max - pc1_min) * 0.1 if pc1_max > pc1_min else 1.0
        pc2_pad = (pc2_max - pc2_min) * 0.1 if pc2_max > pc2_min else 1.0
        pc1_min -= pc1_pad; pc1_max += pc1_pad
        pc2_min -= pc2_pad; pc2_max += pc2_pad

        painter.setPen(QPen(QColor(40, 40, 50), 1, Qt.PenStyle.DotLine))
        for i in range(5):
            y = margin + int(plot_h * i / 4)
            painter.drawLine(margin, y, margin + plot_w, y)
            x = margin + int(plot_w * i / 4)
            painter.drawLine(x, margin, x, margin + plot_h)

        painter.setPen(QColor(100, 100, 110))
        painter.setFont(QFont("Consolas", 9))
        for i in range(5):
            pc1_val = pc1_min + (pc1_max - pc1_min) * i / 4
            x = margin + int(plot_w * i / 4)
            painter.drawText(x - 20, h - margin + 18, f"{pc1_val:.2f}")
            pc2_val = pc2_max - (pc2_max - pc2_min) * i / 4
            y = margin + int(plot_h * i / 4)
            painter.drawText(5, y + 4, f"{pc2_val:.2f}")

        painter.setFont(QFont("Segoe UI", 10))
        painter.setPen(QColor(150, 150, 160))
        painter.drawText(margin + plot_w // 2 - 40, h - 5, f"PC1 ({var_exp[0]*100:.1f}% var)")
        painter.save()
        painter.translate(12, margin + plot_h // 2 + 50)
        painter.rotate(-90)
        painter.drawText(0, 0, f"PC2 ({var_exp[1]*100:.1f}% var)")
        painter.restore()

        fits = [c.fitness for c in self.population]
        f_min, f_max = min(fits), max(fits)
        if f_max - f_min < 1e-9:
            f_max = f_min + 1.0

        for i, c in enumerate(self.population):
            pc1_val, pc2_val = proj[i, 0], proj[i, 1]
            cx = margin + int(plot_w * (pc1_val - pc1_min) / (pc1_max - pc1_min))
            cy = margin + plot_h - int(plot_h * (pc2_val - pc2_min) / (pc2_max - pc2_min))
            t = (c.fitness - f_min) / (f_max - f_min)
            color = QColor(int(30 + 50 * t), int(100 + 150 * t), 247)
            size = max(5, 14 - c.age // 5)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
            painter.drawEllipse(QPoint(cx, cy), size, size)

        if self.best:
            best_idx = next((i for i, c in enumerate(self.population) if c.candidate_id == self.best.candidate_id), None)
            if best_idx is not None:
                pc1_val, pc2_val = proj[best_idx, 0], proj[best_idx, 1]
                cx = margin + int(plot_w * (pc1_val - pc1_min) / (pc1_max - pc1_min))
                cy = margin + plot_h - int(plot_h * (pc2_val - pc2_min) / (pc2_max - pc2_min))
                glow = QRadialGradient(cx, cy, 25)
                glow.setColorAt(0, QColor(255, 180, 80, 200))
                glow.setColorAt(1, QColor(255, 180, 80, 0))
                painter.setBrush(QBrush(glow))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPoint(cx, cy), 25, 25)
                painter.setBrush(QColor(255, 180, 80))
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.drawEllipse(QPoint(cx, cy), 8, 8)
                painter.setPen(QColor(255, 255, 255))
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.drawText(cx + 14, cy - 6, "BEST")

        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor(180, 180, 190))
        painter.drawText(10, 20, f"Population: {len(self.population)} candidates")
        painter.drawText(10, 35, f"Total variance explained: {(var_exp[0] + var_exp[1])*100:.1f}%")
        painter.end()


# ==============================================================================
# 13. MAIN APPLICATION WINDOW
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⬢ Reaction-Diffusion Neural Network — Advanced MC Search + Benchmarks")
        self.resize(1700, 1000)

        self.sim = GrayScott(AppConstants.DEFAULT_WIDTH, AppConstants.DEFAULT_HEIGHT, use_gpu=HAS_CUPY)
        self.ripples = RippleSystem(self.sim)
        self.network = FeedForwardNetwork(self.sim)
        self.mc = MonteCarloOptimizer(self.sim, self.network)

        self.running = False
        self.steps_per_frame = AppConstants.DEFAULT_STEPS_PER_FRAME
        self.frame_count = 0
        self.colormap_name = "Ink"
        self._last_fps_time = time.perf_counter()
        self._last_fps_frames = 0
        self._fps = 0.0
        self._nn_test_interval = 100
        self._nn_step_counter = 0
        self._auto_nn_test = True
        self._ghost_enabled = False
        self._ghost_opacity = 0.3
        self._overlay_cmap = ""
        self._overlay_opacity = 0.0
        self._recording = False
        self._gif_frames: List[np.ndarray] = []
        self._best_fitness_history: List[float] = []
        
        # Benchmark state
        self._benchmark_enabled = False
        self._benchmark_dataset: Optional[object] = None
        self._current_dataset_name = "Blobs (64F, 4C)"

        self._build_ui()
        self._apply_preset("Coral")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self.status = QStatusBar(self)
        self.setStatusBar(self.status)
        self._update_status()

    def _build_ui(self) -> None:
        """Construct the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._build_sidebar(main_layout)
        self._build_center_panel(main_layout)
        self._build_toolbar()
        self._connect_signals()
        self._setup_shortcuts()

    def _build_sidebar(self, main_layout) -> None:
        """Construct the collapsible sidebar."""
        self.sidebar = QScrollArea()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setWidgetResizable(True)
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar.setMinimumWidth(280)
        self.sidebar.setMaximumWidth(320)

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        title_widget = QWidget()
        title_widget.setStyleSheet("background-color: #0d0d12; padding: 14px;")
        title_layout = QVBoxLayout(title_widget)
        title_layout.setContentsMargins(14, 10, 14, 10)
        lbl_title = QLabel("⬢ RD Neural Network")
        lbl_title.setStyleSheet("color: #4fc3f7; font-size: 15px; font-weight: 700;")
        lbl_subtitle = QLabel("Advanced Monte Carlo Search + Benchmarks")
        lbl_subtitle.setStyleSheet("color: #808090; font-size: 10px;")
        title_layout.addWidget(lbl_title)
        title_layout.addWidget(lbl_subtitle)
        sidebar_layout.addWidget(title_widget)

        # Simulation Section
        sec_sim = CollapsibleSection("SIMULATION", "▶")
        btn_row1 = QHBoxLayout()
        self.btn_play = QPushButton("▶  Play")
        self.btn_play.setObjectName("primary")
        self.btn_play.setCheckable(True)
        self.btn_step = QPushButton("Step")
        btn_row1.addWidget(self.btn_play)
        btn_row1.addWidget(self.btn_step)
        sec_sim.add_widget(self._wrap_layout(btn_row1))

        btn_row2 = QHBoxLayout()
        self.btn_reset = QPushButton("Reset")
        self.btn_seed = QPushButton("Seed")
        btn_row2.addWidget(self.btn_reset)
        btn_row2.addWidget(self.btn_seed)
        sec_sim.add_widget(self._wrap_layout(btn_row2))

        steps_layout = QGridLayout()
        steps_layout.setSpacing(6)
        steps_layout.addWidget(QLabel("Steps / frame"), 0, 0)
        self.lbl_steps_val = QLabel(str(self.steps_per_frame))
        self.lbl_steps_val.setObjectName("value-display")
        self.lbl_steps_val.setAlignment(Qt.AlignmentFlag.AlignRight)
        steps_layout.addWidget(self.lbl_steps_val, 0, 1)
        self.slider_steps = QSlider(Qt.Orientation.Horizontal)
        self.slider_steps.setRange(AppConstants.MIN_STEPS_PER_FRAME, AppConstants.MAX_STEPS_PER_FRAME)
        self.slider_steps.setValue(self.steps_per_frame)
        steps_layout.addWidget(self.slider_steps, 1, 0, 1, 2)
        sec_sim.add_widget(self._wrap_layout(steps_layout))

        preset_layout = QGridLayout()
        preset_layout.setSpacing(6)
        preset_layout.addWidget(QLabel("Preset"), 0, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()))
        preset_layout.addWidget(self.preset_combo, 0, 1)
        sec_sim.add_widget(self._wrap_layout(preset_layout))

        cmap_layout = QGridLayout()
        cmap_layout.setSpacing(6)
        cmap_layout.addWidget(QLabel("Colormap"), 0, 0)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(list(COLORMAPS.keys()))
        cmap_layout.addWidget(self.colormap_combo, 0, 1)
        sec_sim.add_widget(self._wrap_layout(cmap_layout))
        sidebar_layout.addWidget(sec_sim)

        # Parameters Section
        sec_params = CollapsibleSection("PARAMETERS", "⚙")
        param_grid = QGridLayout()
        param_grid.setSpacing(8)
        self.spin_F = self._create_spin(0.001, 0.100, 4, 0.001)
        self.spin_k = self._create_spin(0.001, 0.080, 4, 0.001)
        self.spin_Du = self._create_spin(0.1, 2.0, 3, 0.01)
        self.spin_Dv = self._create_spin(0.05, 1.5, 3, 0.01)
        
        param_grid.addWidget(QLabel("F (feed)"), 0, 0); param_grid.addWidget(self.spin_F, 0, 1)
        param_grid.addWidget(QLabel("k (kill)"), 1, 0); param_grid.addWidget(self.spin_k, 1, 1)
        param_grid.addWidget(QLabel("Du"), 2, 0); param_grid.addWidget(self.spin_Du, 2, 1)
        param_grid.addWidget(QLabel("Dv"), 3, 0); param_grid.addWidget(self.spin_Dv, 3, 1)
        sec_params.add_widget(self._wrap_layout(param_grid))
        sidebar_layout.addWidget(sec_params)

        # Ripples Section
        sec_ripple = CollapsibleSection("RIPPLES", "◉")
        self.chk_ripple = QCheckBox("Enable ripples")
        sec_ripple.add_widget(self.chk_ripple)
        ripple_grid = QGridLayout()
        ripple_grid.setSpacing(6)
        self.combo_ripple_mode = QComboBox()
        self.combo_ripple_mode.addItems(["random", "grid", "spiral"])
        self.spin_ripple_interval = QSpinBox(); self.spin_ripple_interval.setRange(5, 200); self.spin_ripple_interval.setValue(30)
        self.spin_ripple_radius = QSpinBox(); self.spin_ripple_radius.setRange(2, 30); self.spin_ripple_radius.setValue(5)
        ripple_grid.addWidget(QLabel("Mode"), 0, 0); ripple_grid.addWidget(self.combo_ripple_mode, 0, 1)
        ripple_grid.addWidget(QLabel("Interval"), 1, 0); ripple_grid.addWidget(self.spin_ripple_interval, 1, 1)
        ripple_grid.addWidget(QLabel("Radius"), 2, 0); ripple_grid.addWidget(self.spin_ripple_radius, 2, 1)
        sec_ripple.add_widget(self._wrap_layout(ripple_grid))
        sidebar_layout.addWidget(sec_ripple)

        # Network Section
        sec_net = CollapsibleSection("NETWORK", "⊞")
        net_grid = QGridLayout()
        net_grid.setSpacing(6)
        self.combo_input = QComboBox(); self.combo_input.addItems(["sine", "random", "impulse", "gradient", "step"])
        self.combo_activation = QComboBox(); self.combo_activation.addItems(["tanh", "sigmoid", "relu", "leaky_relu"])
        net_grid.addWidget(QLabel("Input"), 0, 0); net_grid.addWidget(self.combo_input, 0, 1)
        net_grid.addWidget(QLabel("Activation"), 1, 0); net_grid.addWidget(self.combo_activation, 1, 1)
        sec_net.add_widget(self._wrap_layout(net_grid))
        net_btn_row = QHBoxLayout()
        self.chk_auto_test = QCheckBox("Auto-test"); self.chk_auto_test.setChecked(True)
        self.btn_test = QPushButton("Test Now")
        net_btn_row.addWidget(self.chk_auto_test); net_btn_row.addWidget(self.btn_test)
        sec_net.add_widget(self._wrap_layout(net_btn_row))
        sidebar_layout.addWidget(sec_net)

        # Monte Carlo Section
        sec_mc = CollapsibleSection("MONTE CARLO", "∿")
        mc_btn_row = QHBoxLayout()
        self.btn_mc_start = QPushButton("Start MC"); self.btn_mc_start.setObjectName("primary"); self.btn_mc_start.setCheckable(True)
        self.btn_mc_reset = QPushButton("Reset")
        mc_btn_row.addWidget(self.btn_mc_start); mc_btn_row.addWidget(self.btn_mc_reset)
        sec_mc.add_widget(self._wrap_layout(mc_btn_row))
        mc_grid = QGridLayout()
        mc_grid.setSpacing(6)
        self.spin_warmup = QSpinBox(); self.spin_warmup.setRange(10, 2000); self.spin_warmup.setValue(200)
        self.spin_pop_size = QSpinBox(); self.spin_pop_size.setRange(1, 20); self.spin_pop_size.setValue(5)
        mc_grid.addWidget(QLabel("Warmup"), 0, 0); mc_grid.addWidget(self.spin_warmup, 0, 1)
        mc_grid.addWidget(QLabel("Pop size"), 1, 0); mc_grid.addWidget(self.spin_pop_size, 1, 1)
        sec_mc.add_widget(self._wrap_layout(mc_grid))
        sidebar_layout.addWidget(sec_mc)

        # Benchmark Section
        sec_bench = CollapsibleSection("BENCHMARK", "📊")
        self.chk_benchmark = QCheckBox("Enable Dataset Benchmark")
        sec_bench.add_widget(self.chk_benchmark)
        
        bench_grid = QGridLayout()
        bench_grid.setSpacing(6)
        self.combo_benchmark = QComboBox()
        self.combo_benchmark.addItems(list(BENCHMARK_PRESETS.keys()))
        bench_grid.addWidget(QLabel("Synthetic"), 0, 0)
        bench_grid.addWidget(self.combo_benchmark, 0, 1)
        sec_bench.add_widget(self._wrap_layout(bench_grid))
        
        # Custom Dataset Loader
        custom_layout = QVBoxLayout()
        custom_layout.setSpacing(6)
        self.btn_load_custom = QPushButton("📂 Load Custom CSV")
        self.btn_load_custom.setToolTip("Expected: Comma-separated, numeric, last column = target label.")
        self.lbl_dataset_info = QLabel("No custom dataset loaded.")
        self.lbl_dataset_info.setStyleSheet("color: #808090; font-size: 10px; font-style: italic; padding: 4px; word-wrap: break-word;")
        self.lbl_dataset_info.setWordWrap(True)
        
        custom_layout.addWidget(self.btn_load_custom)
        custom_layout.addWidget(self.lbl_dataset_info)
        sec_bench.add_widget(self._wrap_layout(custom_layout))
        
        sidebar_layout.addWidget(sec_bench)

        # View Section
        sec_view = CollapsibleSection("VIEW", "◐")
        self.chk_ghost = QCheckBox("Ghost overlay")
        sec_view.add_widget(self.chk_ghost)
        ghost_layout = QGridLayout()
        self.slider_ghost = QSlider(Qt.Orientation.Horizontal); self.slider_ghost.setRange(0, 100); self.slider_ghost.setValue(30)
        ghost_layout.addWidget(QLabel("Opacity"), 0, 0); ghost_layout.addWidget(self.slider_ghost, 0, 1)
        sec_view.add_widget(self._wrap_layout(ghost_layout))
        overlay_layout = QGridLayout()
        self.combo_overlay = QComboBox(); self.combo_overlay.addItems(["None"] + list(OVERLAY_CMAPS.keys()))
        self.slider_overlay = QSlider(Qt.Orientation.Horizontal); self.slider_overlay.setRange(0, 100); self.slider_overlay.setValue(0)
        overlay_layout.addWidget(QLabel("Overlay"), 0, 0); overlay_layout.addWidget(self.combo_overlay, 0, 1)
        overlay_layout.addWidget(QLabel("Opacity"), 1, 0); overlay_layout.addWidget(self.slider_overlay, 1, 1)
        sec_view.add_widget(self._wrap_layout(overlay_layout))
        sidebar_layout.addWidget(sec_view)

        # Export Section
        sec_export = CollapsibleSection("EXPORT", "⤓")
        self.btn_record = QPushButton("Record GIF"); self.btn_record.setCheckable(True)
        sec_export.add_widget(self.btn_record)
        export_row = QHBoxLayout()
        self.btn_export_csv = QPushButton("CSV"); self.btn_export_pareto = QPushButton("Pareto")
        export_row.addWidget(self.btn_export_csv); export_row.addWidget(self.btn_export_pareto)
        sec_export.add_widget(self._wrap_layout(export_row))
        sidebar_layout.addWidget(sec_export)

        sidebar_layout.addStretch()
        self.sidebar.setWidget(sidebar_content)
        main_layout.addWidget(self.sidebar)

    def _build_center_panel(self, main_layout) -> None:
        """Construct the center display and tabs panel."""
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(8)

        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(8)
        self.card_fitness = MetricCard("Current Fitness", "—", "#4fc3f7")
        self.card_best = MetricCard("Best Fitness", "—", "#ffb74d")
        self.card_acc = MetricCard("Accuracy", "—", "#66bb6a")
        self.card_loss = MetricCard("Loss", "—", "#ef5350")
        self.card_temp = MetricCard("Temperature", "—", "#81c784")
        self.card_steps = MetricCard("MC Steps", "0", "#ce93d8")
        self.card_diversity = MetricCard("Diversity", "—", "#4dd0e1")
        for card in [self.card_fitness, self.card_best, self.card_acc, self.card_loss, self.card_temp, self.card_steps, self.card_diversity]:
            metrics_row.addWidget(card)
        center_layout.addLayout(metrics_row)

        center_splitter = QSplitter(Qt.Orientation.Horizontal)

        display_panel = QWidget()
        display_layout = QVBoxLayout(display_panel)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(0)

        display_header = QWidget()
        display_header.setStyleSheet("background-color: #14141a; padding: 8px 14px; border-radius: 6px 6px 0 0;")
        dh_layout = QHBoxLayout(display_header)
        dh_layout.setContentsMargins(8, 4, 8, 4)
        self.led_running = StatusLED("green")
        self.led_mc = StatusLED("blue")
        self.led_rec = StatusLED("red")
        dh_layout.addWidget(QLabel("SIMULATION"))
        dh_layout.addStretch()
        dh_layout.addWidget(self._make_led_label("RUN", self.led_running))
        dh_layout.addWidget(self._make_led_label("MC", self.led_mc))
        dh_layout.addWidget(self._make_led_label("REC", self.led_rec))
        display_layout.addWidget(display_header)

        self.display = DisplayWidget()
        self.display.clicked = self._on_display_click
        self.display.setStyleSheet("background-color: #0a0a0e; border: 1px solid #2a2a35; border-top: none; border-radius: 0 0 6px 6px;")
        display_layout.addWidget(self.display, stretch=1)
        center_splitter.addWidget(display_panel)

        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(480)
        
        dashboard = QWidget()
        dash_layout = QVBoxLayout(dashboard)
        dash_layout.setContentsMargins(14, 14, 14, 14)
        dash_layout.setSpacing(12)
        self.lbl_status_title = QLabel("OPTIMIZER STATUS")
        self.lbl_status_title.setStyleSheet("color: #4fc3f7; font-size: 12px; font-weight: 700; letter-spacing: 1px;")
        self.lbl_mc_status = QLabel("MC: idle")
        self.lbl_mc_status.setWordWrap(True)
        self.lbl_mc_status.setStyleSheet("background-color: #1e1e26; padding: 12px; border-radius: 6px; color: #c0c0d0; font-family: 'Consolas', monospace; font-size: 11px; border: 1px solid #2a2a35;")
        dash_layout.addWidget(self.lbl_status_title)
        dash_layout.addWidget(self.lbl_mc_status)
        dash_layout.addStretch()
        
        self.tabs.addTab(dashboard, "Dashboard")
        self.fitness_plot = FitnessPlot(); self.tabs.addTab(self.fitness_plot, "Fitness")
        self.phase_plot = PhaseSpacePlot(); self.tabs.addTab(self.phase_plot, "Phase Space")
        self.param_plot = ParameterTimeSeriesPlot(); self.tabs.addTab(self.param_plot, "Parameters")
        self.spectrum_plot = SpectrumPlot(); self.tabs.addTab(self.spectrum_plot, "Spectrum")
        self.activation_plot = ActivationHeatmap(); self.tabs.addTab(self.activation_plot, "Activations")
        self.network_plot = NetworkDiagram(); self.tabs.addTab(self.network_plot, "Network")
        self.population_plot = PopulationPlot(); self.tabs.addTab(self.population_plot, "Population")
        self.pca_plot = PCAPlot(); self.tabs.addTab(self.pca_plot, "PCA Space")

        center_splitter.addWidget(self.tabs)
        center_splitter.setSizes([900, 500])
        center_layout.addWidget(center_splitter, stretch=1)
        main_layout.addWidget(center_widget, stretch=1)

    def _build_toolbar(self) -> None:
        """Construct the top toolbar."""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        def add_tb(text: str, checkable: bool = False, slot=None):
            btn = QToolButton()
            btn.setText(text)
            btn.setCheckable(checkable)
            if slot:
                btn.clicked.connect(slot)
            toolbar.addWidget(btn)
            return btn

        self._tb_play = add_tb("▶ Play", True, self._on_play)
        add_tb("Step", False, self._on_step)
        add_tb("Reset", False, self._on_reset)
        add_tb("Seed", False, self._on_seed)
        toolbar.addSeparator()
        self._tb_mc = add_tb("∿ MC", True, self._on_mc_toggle)
        add_tb("MC Reset", False, self._on_mc_reset)
        toolbar.addSeparator()
        self._tb_rec = add_tb("● REC", True, self._on_record_toggle)
        toolbar.addSeparator()
        add_tb("⛶ Fullscreen", False, self._toggle_fullscreen)
        add_tb("? About", False, self._show_about)

    def _wrap_layout(self, layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        w.setContentsMargins(0, 0, 0, 0)
        return w

    def _create_spin(self, min_val: float, max_val: float, decimals: int, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        return spin

    def _make_led_label(self, text: str, led: StatusLED) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(4)
        layout.addWidget(led)
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #808090; font-size: 10px; font-weight: 600; letter-spacing: 0.5px;")
        layout.addWidget(lbl)
        return w

    def _connect_signals(self) -> None:
        self.btn_play.clicked.connect(self._on_play)
        self.btn_step.clicked.connect(self._on_step)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_seed.clicked.connect(self._on_seed)
        self.slider_steps.valueChanged.connect(self._on_steps_changed)
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        for spin in (self.spin_F, self.spin_k, self.spin_Du, self.spin_Dv):
            spin.valueChanged.connect(self._on_param_changed)
        self.chk_ripple.toggled.connect(self._on_ripple_toggled)
        self.combo_ripple_mode.currentTextChanged.connect(self._on_ripple_mode_changed)
        self.spin_ripple_interval.valueChanged.connect(self._on_ripple_interval_changed)
        self.spin_ripple_radius.valueChanged.connect(self._on_ripple_radius_changed)
        self.combo_input.currentTextChanged.connect(self._on_input_changed)
        self.combo_activation.currentTextChanged.connect(self._on_activation_changed)
        self.chk_auto_test.toggled.connect(self._on_auto_test_toggled)
        self.btn_test.clicked.connect(self._on_test_now)
        self.btn_mc_start.clicked.connect(self._on_mc_toggle)
        self.btn_mc_reset.clicked.connect(self._on_mc_reset)
        self.spin_pop_size.valueChanged.connect(self._on_pop_size_changed)
        self.chk_ghost.toggled.connect(self._on_ghost_toggled)
        self.slider_ghost.valueChanged.connect(self._on_ghost_opacity_changed)
        self.combo_overlay.currentTextChanged.connect(self._on_overlay_changed)
        self.slider_overlay.valueChanged.connect(self._on_overlay_opacity_changed)
        self.btn_record.clicked.connect(self._on_record_toggle)
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        self.btn_export_pareto.clicked.connect(self._on_export_pareto)
        self.chk_benchmark.toggled.connect(self._on_benchmark_toggled)
        self.combo_benchmark.currentTextChanged.connect(self._on_benchmark_dataset_changed)
        self.btn_load_custom.clicked.connect(self._on_load_custom_dataset)

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("Space"), self, self._on_play)
        QShortcut(QKeySequence("S"), self, self._on_step)
        QShortcut(QKeySequence("R"), self, self._on_reset)
        QShortcut(QKeySequence("F11"), self, self._toggle_fullscreen)
        QShortcut(QKeySequence("M"), self, self._on_mc_toggle)

    @Slot()
    def _on_play(self) -> None:
        self.running = not self.running
        self.btn_play.setChecked(self.running)
        self.btn_play.setText("⏸  Pause" if self.running else "▶  Play")
        self._tb_play.setChecked(self.running)
        self._tb_play.setText("⏸ Pause" if self.running else "▶ Play")
        self.led_running.set_active(self.running)
        self.timer.start(16) if self.running else self.timer.stop()

    @Slot()
    def _on_step(self) -> None:
        self._do_sim_steps(self.steps_per_frame)
        self._update_visuals()

    @Slot()
    def _on_reset(self) -> None:
        self.sim.reset()
        self._update_visuals()

    @Slot()
    def _on_seed(self) -> None:
        preset_name = self.preset_combo.currentText()
        if preset_name in PRESETS:
            p = PRESETS[preset_name]
            self.sim.seed_random(p.seed_radius, p.seed_count)
        else:
            self.sim.seed_random(10, 10)
        self._update_visuals()

    @Slot(int)
    def _on_steps_changed(self, value: int) -> None:
        self.steps_per_frame = value
        self.lbl_steps_val.setText(str(value))

    @Slot(str)
    def _on_colormap_changed(self, name: str) -> None:
        self.colormap_name = name
        self.display.set_colormap(name)
        self._update_visuals()

    @Slot()
    def _on_param_changed(self) -> None:
        self.sim.F = self.spin_F.value()
        self.sim.k = self.spin_k.value()
        self.sim.Du = self.spin_Du.value()
        self.sim.Dv = self.spin_Dv.value()

    @Slot(bool)
    def _on_ripple_toggled(self, checked: bool) -> None:
        self.ripples.enabled = checked

    @Slot(str)
    def _on_ripple_mode_changed(self, mode: str) -> None:
        self.ripples.mode = mode
        self.ripples.reset_grid()

    @Slot(int)
    def _on_ripple_interval_changed(self, value: int) -> None:
        self.ripples.interval = value

    @Slot(int)
    def _on_ripple_radius_changed(self, value: int) -> None:
        self.ripples.radius = value

    @Slot(str)
    def _on_input_changed(self, pattern: str) -> None:
        self.network.input_pattern = pattern

    @Slot(str)
    def _on_activation_changed(self, fn: str) -> None:
        self.network.activation_fn = fn

    @Slot(bool)
    def _on_auto_test_toggled(self, checked: bool) -> None:
        self._auto_nn_test = checked

    @Slot()
    def _on_test_now(self) -> None:
        self.network.test()
        self._update_nn_plots()

    @Slot()
    def _on_mc_toggle(self) -> None:
        self.mc.enabled = not self.mc.enabled
        self.btn_mc_start.setChecked(self.mc.enabled)
        self.btn_mc_start.setText("Stop MC" if self.mc.enabled else "Start MC")
        self._tb_mc.setChecked(self.mc.enabled)
        self.led_mc.set_active(self.mc.enabled)
        if self.mc.enabled and not self.running:
            self._on_play()

    @Slot()
    def _on_mc_reset(self) -> None:
        self.mc.reset()
        self._best_fitness_history.clear()
        self._update_mc_status({
            "step": 0, "fitness": 0.0, "best_fitness": 0.0,
            "temperature": self.mc.perturbation.temperature,
            "population_size": 0, "restarts": 0,
            "population_diversity": 0.0, "status": "running",
        })

    @Slot(int)
    def _on_pop_size_changed(self, value: int) -> None:
        self.mc.config = OptimizerConfig(
            population_size=value,
            initial_temperature=self.mc.config.initial_temperature,
            min_temperature=self.mc.config.min_temperature,
            max_temperature=self.mc.config.max_temperature,
            target_acceptance=self.mc.config.target_acceptance,
            temp_adjust_rate=self.mc.config.temp_adjust_rate,
            restart_threshold=self.mc.config.restart_threshold,
            max_restarts=self.mc.config.max_restarts,
            momentum_coeff=self.mc.config.momentum_coeff,
            diversity_weight=self.mc.config.diversity_weight,
            fitness_sharing_alpha=self.mc.config.fitness_sharing_alpha,
            fitness_sharing_sigma=self.mc.config.fitness_sharing_sigma,
            w_var=self.mc.config.w_var, w_mean=self.mc.config.w_mean,
            w_grad=self.mc.config.w_grad, w_sparsity=self.mc.config.w_sparsity,
            w_pattern=self.mc.config.w_pattern, w_spectral_entropy=self.mc.config.w_spectral_entropy,
            w_wavelength=self.mc.config.w_wavelength, convergence_window=self.mc.config.convergence_window,
            convergence_threshold=self.mc.config.convergence_threshold,
            perturb_flags=dict(self.mc.config.perturb_flags),
            bounds=dict(self.mc.config.bounds), param_scales=dict(self.mc.config.param_scales),
        )
        self.mc.population_mgr = PopulationManager(self.mc.config)

    @Slot(bool)
    def _on_ghost_toggled(self, checked: bool) -> None:
        self._ghost_enabled = checked
        if not checked:
            self.display.set_ghost(None, 0.0)
            self._update_visuals()

    @Slot(int)
    def _on_ghost_opacity_changed(self, value: int) -> None:
        self._ghost_opacity = value / 100.0
        self._update_visuals()

    @Slot(str)
    def _on_overlay_changed(self, name: str) -> None:
        self._overlay_cmap = name
        if name == "None":
            self.display.set_overlay("", 0.0)
        else:
            self.display.set_overlay(name, self._overlay_opacity)
        self._update_visuals()

    @Slot(int)
    def _on_overlay_opacity_changed(self, value: int) -> None:
        self._overlay_opacity = value / 100.0
        if self._overlay_cmap and self._overlay_cmap != "None":
            self.display.set_overlay(self._overlay_cmap, self._overlay_opacity)
        self._update_visuals()

    @Slot()
    def _on_record_toggle(self) -> None:
        self._recording = not self._recording
        self.btn_record.setChecked(self._recording)
        self.btn_record.setText("Stop REC" if self._recording else "Record GIF")
        self._tb_rec.setChecked(self._recording)
        self.led_rec.set_active(self._recording)
        if not self._recording and self._gif_frames:
            self._save_gif()

    @Slot()
    def _on_export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "mc_history.csv", "CSV Files (*.csv)")
        if path:
            self.mc.export_csv(path)
            self.status.showMessage(f"Exported to {path}", 3000)

    @Slot()
    def _on_export_pareto(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Pareto Front", "pareto.json", "JSON Files (*.json)")
        if path:
            self.mc.export_pareto_front(Path(path))
            self.status.showMessage(f"Exported to {path}", 3000)

    @Slot()
    def _toggle_fullscreen(self) -> None:
        self.showFullScreen() if not self.isFullScreen() else self.showNormal()

    @Slot()
    def _show_about(self) -> None:
        QMessageBox.about(
            self, "About",
            "⬢ Reaction-Diffusion Neural Network\n"
            "with Advanced Monte Carlo Search + Benchmarks\n\n"
            "A research tool exploring the intersection of\n"
            "Turing patterns and neural computation.\n\n"
            "Features:\n"
            "  • Gray-Scott RD simulation (CPU + CUDA)\n"
            "  • Feed-forward network interpretation (NO TRAINING)\n"
            "  • Population-based MC optimizer (DE + Adaptive Jitter)\n"
            "  • Regularized ELM (Ridge Regression) Output Layer\n"
            "  • Custom dataset benchmarking (CSV input)\n"
            "  • 9 analysis tabs with real-time plots"
        )

    def _on_display_click(self, x: int, y: int) -> None:
        dw, dh = self.display.width(), self.display.height()
        scaled = min(dw, dh)
        offset_x = (dw - scaled) // 2
        offset_y = (dh - scaled) // 2
        sx = int((x - offset_x) / scaled * self.sim.width)
        sy = int((y - offset_y) / scaled * self.sim.height)
        if 0 <= sx < self.sim.width and 0 <= sy < self.sim.height:
            self.sim.seed_point(sx, sy, radius=8, intensity=0.35)
            self._update_visuals()

    def _apply_preset(self, name: str) -> None:
        if name not in PRESETS:
            return
        p = PRESETS[name]
        self.sim.F, self.sim.k, self.sim.Du, self.sim.Dv = p.F, p.k, p.Du, p.Dv
        self.sim.seed_random(p.seed_radius, p.seed_count)
        self._sync_param_spins()
        self._update_visuals()

    def _sync_param_spins(self) -> None:
        for spin, attr in [(self.spin_F, "F"), (self.spin_k, "k"), (self.spin_Du, "Du"), (self.spin_Dv, "Dv")]:
            spin.blockSignals(True)
            spin.setValue(getattr(self.sim, attr))
            spin.blockSignals(False)

    def _load_synthetic_benchmark(self) -> None:
        config = BENCHMARK_PRESETS.get(self._current_dataset_name, BENCHMARK_PRESETS["Blobs (64F, 4C)"])
        self._benchmark_dataset = BenchmarkDataset(config)
        self.network.benchmark_dataset = self._benchmark_dataset
        self.lbl_dataset_info.setText("Using synthetic dataset.")

    @Slot(bool)
    def _on_benchmark_toggled(self, checked: bool) -> None:
        self._benchmark_enabled = checked
        self.network.benchmark_mode = checked
        if checked and self._benchmark_dataset is None:
            self._load_synthetic_benchmark()
        self._update_visuals()

    @Slot(str)
    def _on_benchmark_dataset_changed(self, name: str) -> None:
        self._current_dataset_name = name
        self._load_synthetic_benchmark()
        if self._benchmark_enabled:
            self.mc.reset()
            self._on_mc_reset()

    @Slot()
    def _on_load_custom_dataset(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Custom Dataset CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not filepath:
            return
            
        try:
            custom_data = CustomBenchmarkDataset(filepath, target_col_idx=-1, max_samples=5000)
            self._benchmark_dataset = custom_data
            self.network.benchmark_dataset = custom_data
            self.network.benchmark_mode = True
            self.chk_benchmark.setChecked(True)
            
            info = (f"✅ Loaded: {Path(filepath).name}\n"
                    f"Samples: {custom_data.n_samples} | Features: {custom_data.num_features}\n"
                    f"Classes: {custom_data.num_classes} | Split: 80/20")
            self.lbl_dataset_info.setText(info)
            self.lbl_dataset_info.setStyleSheet("color: #66bb6a; font-size: 10px; padding: 4px;")
            
            self.mc.reset()
            self._on_mc_reset()
            self.status.showMessage(f"Successfully loaded dataset: {custom_data.n_samples} samples", 4000)
            
        except ValueError as e:
            QMessageBox.critical(self, "Dataset Load Error", str(e))
            self.lbl_dataset_info.setText("❌ Failed to load dataset.")
            self.lbl_dataset_info.setStyleSheet("color: #ef5350; font-size: 10px; padding: 4px;")

    # ==========================================================================
    # MAIN LOOP & HELPERS
    # ==========================================================================
    @Slot()
    def _tick(self) -> None:
        """Main simulation tick, delegated to clean helper methods."""
        self.frame_count += 1
        self._do_sim_steps(self.steps_per_frame)
        self.ripples.tick()

        if self.mc.enabled:
            self._update_mc_step()

        if self._auto_nn_test and self._nn_step_counter >= self._nn_test_interval:
            self._nn_step_counter = 0
            self.network.test()

        self._update_visuals()
        self._update_status()

        if self._recording:
            self._record_frame()

    def _do_sim_steps(self, n: int) -> None:
        self.sim.step(n, save_prev=self._ghost_enabled)

    def _update_mc_step(self) -> None:
        warmup = self.spin_warmup.value()
        
        benchmark_result = None
        if self._benchmark_enabled and self._benchmark_dataset is not None:
            benchmark_result = self.network.evaluate_benchmark(self._benchmark_dataset)
            
        info = self.mc.step(warmup_steps=warmup, benchmark_result=benchmark_result)
        self._update_mc_status(info)
        
        if info.get("accepted"):
            self._sync_param_spins()
        if self.mc.best_candidate:
            self._best_fitness_history.append(self.mc.best_candidate.fitness)
        else:
            self._best_fitness_history.append(0.0)

    def _update_visuals(self) -> None:
        v = self.sim.get_v_np()
        self.display.update_image(v)
        
        info = f"F={self.sim.F:.4f}  k={self.sim.k:.4f}\nDu={self.sim.Du:.2f}  Dv={self.sim.Dv:.2f}"
        self.display.set_info(info)

        if self._ghost_enabled and self.sim.get_prev_v() is not None:
            prev_v = self.sim.get_prev_v()
            prev_v_safe = np.nan_to_num(prev_v, nan=AppConstants.NAN_REPLACE, posinf=AppConstants.POSINF_REPLACE, neginf=AppConstants.NEGINF_REPLACE)
            self.display.set_ghost(np.clip(prev_v_safe, 0.0, 1.0), self._ghost_opacity)
        else:
            self.display.set_ghost(None, 0.0)

        self._update_plots()

    def _record_frame(self) -> None:
        if not HAS_IMAGEIO:
            return
        v = self.sim.get_v_np()
        v_safe = np.nan_to_num(v, nan=AppConstants.NAN_REPLACE, posinf=AppConstants.POSINF_REPLACE, neginf=AppConstants.NEGINF_REPLACE)
        v_norm = np.clip(v_safe, 0.0, 1.0)
        indices = (v_norm * 255).astype(np.uint8)
        rgb = self.display._colormap[indices]
        self._gif_frames.append(rgb)
        if len(self._gif_frames) > AppConstants.MAX_GIF_FRAMES:
            self._gif_frames.pop(0)

    def _update_plots(self) -> None:
        self.fitness_plot.update_data(self.mc.fitness_history, self._best_fitness_history)
        self.phase_plot.update_data(self.mc.param_history)
        self.param_plot.update_data(self.mc.param_history)
        self.population_plot.update_data(self.mc.population_mgr.population, self.mc.best_candidate)
        self.pca_plot.update_data(self.mc.population_mgr.population, self.mc.best_candidate)
        if self.network.last_activations is not None:
            self.activation_plot.update_data(self.network.get_activations_normalized())
        if self.network.last_output is not None:
            self.network_plot.update_data(self.network.last_output)
        if hasattr(self.mc.evaluator, "last_power_spectrum"):
            self.spectrum_plot.update_data(self.mc.evaluator.last_power_spectrum)

    def _update_nn_plots(self) -> None:
        if self.network.last_activations is not None:
            self.activation_plot.update_data(self.network.get_activations_normalized())
        if self.network.last_output is not None:
            self.network_plot.update_data(self.network.last_output)

    def _update_mc_status(self, info: Dict[str, Any]) -> None:
        acc_rate = (self.mc.perturbation.accept_count / max(1, self.mc.perturbation.accept_count + self.mc.perturbation.reject_count))
        
        acc = info.get("components", {}).get("accuracy", 0.0)
        loss = info.get("components", {}).get("loss", 0.0)
        
        bench_text = ""
        if self._benchmark_enabled:
            bench_text = f"Acc:         {acc*100:.1f}%\nLoss:        {loss:.4f}\n"
            
        self.lbl_mc_status.setText(
            f"Step:        {info.get('step', 0)}\n"
            f"Status:      {info.get('status', 'running').upper()}\n"
            f"Fitness:     {info.get('fitness', 0.0):.6f}\n"
            f"Best:        {info.get('best_fitness', 0.0):.6f}\n"
            f"{bench_text}"
            f"Temperature: {info.get('temperature', 0.0):.5f}\n"
            f"Population:  {len(self.mc.population_mgr.population)}\n"
            f"Diversity:   {info.get('population_diversity', 0.0):.4f}\n"
            f"Restarts:    {info.get('restarts', 0)}\n"
            f"Accept rate: {acc_rate*100:.1f}%"
        )
        
        self.card_fitness.set_value(f"{info.get('fitness', 0.0):.4f}")
        self.card_best.set_value(f"{info.get('best_fitness', 0.0):.4f}")
        self.card_acc.set_value(f"{acc*100:.1f}%")
        self.card_loss.set_value(f"{loss:.4f}")
        self.card_temp.set_value(f"{info.get('temperature', 0.0):.4f}")
        self.card_steps.set_value(f"{info.get('step', 0)}")
        self.card_diversity.set_value(f"{info.get('population_diversity', 0.0):.3f}")

    def _update_status(self) -> None:
        now = time.perf_counter()
        self._last_fps_frames += 1
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps = self._last_fps_frames / elapsed
            self._last_fps_time = now
            self._last_fps_frames = 0
            
        gpu_str = "GPU" if self.sim.use_gpu else "CPU"
        mc_str = "MC:ON" if self.mc.enabled else "MC:OFF"
        rec_str = "● REC" if self._recording else ""
        bench_str = "BENCH:ON" if self._benchmark_enabled else ""
        self.status.showMessage(
            f"Frame {self.frame_count}  |  {self._fps:.1f} FPS  |  {gpu_str}  |  "
            f"{self.sim.width}×{self.sim.height}  |  F={self.sim.F:.4f} k={self.sim.k:.4f}  |  {mc_str}  {bench_str}  {rec_str}"
        )

    def _save_gif(self) -> None:
        if not HAS_IMAGEIO or not self._gif_frames:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save GIF", "recording.gif", "GIF Files (*.gif)")
        if path:
            try:
                imageio.mimsave(path, self._gif_frames, fps=15)
                self.status.showMessage(f"Saved GIF to {path}", 3000)
            except Exception as e:
                logger.error(f"Failed to save GIF: {e}")
        self._gif_frames.clear()


# ==============================================================================
# 14. ENTRY POINT
# ==============================================================================
def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("RD_NN_MC_Production_Benchmarks")
    app.setStyleSheet(MODERN_QSS)

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(26, 26, 31))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(228, 228, 232))
    pal.setColor(QPalette.ColorRole.Base, QColor(20, 20, 26))
    pal.setColor(QPalette.ColorRole.Text, QColor(228, 228, 232))
    pal.setColor(QPalette.ColorRole.Button, QColor(42, 42, 53))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(228, 228, 232))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(13, 71, 161))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


# ==============================================================================
# 15. EXAMPLE DATASET GENERATOR
# ==============================================================================
def create_example_dataset() -> None:
    """Creates a simple example CSV dataset for testing."""
    rng = np.random.default_rng(42)
    n_samples = 200
    n_features = 8
    n_classes = 3
    
    # Generate synthetic data with 3 classes
    X = rng.normal(size=(n_samples, n_features))
    centers = rng.normal(scale=2.0, size=(n_classes, n_features))
    y = rng.integers(0, n_classes, size=n_samples)
    X += centers[y]
    
    # Save to CSV
    header = ",".join([f"feature_{i}" for i in range(n_features)] + ["target"])
    data = np.hstack([X, y.reshape(-1, 1)])
    
    with open("example_dataset.csv", "w") as f:
        f.write(header + "\n")
        for row in data:
            f.write(",".join([f"{val:.6f}" for val in row]) + "\n")
    
    print("✅ Created example_dataset.csv")
    print(f"   Samples: {n_samples} | Features: {n_features} | Classes: {n_classes}")
    print("   You can load this file using the '📂 Load Custom CSV' button.")


if __name__ == "__main__":
    # Create example dataset if it doesn't exist
    if not Path("example_dataset.csv").exists():
        create_example_dataset()
    
    main()