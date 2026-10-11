"""
GA Image-Neural Network Evolver — Enhanced Complete Edition v4.6
================================================================
Features:
  - Fixed UnboundLocalError by explicit variable initialization
  - Streamlined UI: Trimmed visualization dropdown for development focus
  - Modern Visual Polish (Tokyo Night dark theme aesthetic)
  - 16 Activation Functions (Swish, Mish, ELU, SELU, etc.)
  - Auto-save & Auto-load Settings (ga_settings.json in CWD)
  - Async File I/O (QThreadPool) for zero UI freeze
  - Native Qt Thread Interruption for safe engine shutdown
  - Optimized Garbage Collection
  - Copy Config to Clipboard & Clear Visualizations buttons
  - HSV genome (Hue=activation, Sat=weight, Val=bias)
  - Memetic local search, NSGA-II, SBX, Polynomial Mutation
"""

import sys, os, time, traceback, csv, json, datetime, gc, math
import numpy as np
from contextlib import contextmanager
from collections import deque, defaultdict
from PySide6.QtCore import (Qt, QThread, Signal, QTimer, QPoint, QRectF, QLineF,
                            QPointF, QElapsedTimer, QSize, QByteArray, QRunnable, 
                            QThreadPool, QMetaObject, Q_ARG)
from PySide6.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QFont,
                           QKeySequence, QShortcut, QLinearGradient, QPolygonF,
                           QPalette, QPainterPath, QBrush, QGradient, QRadialGradient,
                           QClipboard)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
    QGroupBox, QProgressBar, QMessageBox, QCheckBox, QTabWidget, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit, QScrollArea, QFrame,
    QInputDialog, QSplitter, QSizePolicy, QGridLayout, QSlider, QStackedWidget)
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.datasets import (load_iris, load_breast_cancer, load_digits,
                              load_wine, load_diabetes)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve)
import pandas as pd

# Optional dependencies
try:
    import cupy as cp; CUPY_AVAILABLE = True
except ImportError:
    cp = None; CUPY_AVAILABLE = False

try:
    import psutil; PSUTIL_AVAILABLE = True
    try: psutil.cpu_percent(interval=None)
    except Exception: pass
except ImportError:
    psutil = None; PSUTIL_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ── Constants & Setup ──────────────────────────────────────────────────────
AUTOSAVE_DIR = os.getcwd()
CHECKPOINT_DIR = os.path.join(AUTOSAVE_DIR, "checkpoints")
SETTINGS_FILE = os.path.join(AUTOSAVE_DIR, "ga_settings.json")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DARK_STYLE = """
QMainWindow, QWidget { background: #1a1b26; color: #a9b1d6; }
QGroupBox { border: 1px solid #2a2e42; border-radius: 6px; margin-top: 12px; padding-top: 12px; font-weight: bold; color: #7aa2f7; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; background: #1a1b26; }
QLabel { color: #a9b1d6; background: transparent; }
QLabel#TitleLabel { color: #7aa2f7; font-size: 16px; font-weight: bold; }
QLabel#StatusTitle { color: #bb9af7; font-weight: bold; }
QPushButton { background: #2a2e42; color: #c0caf5; border: 1px solid #3b4056;
              border-radius: 4px; padding: 6px 12px; min-height: 22px; font-weight: bold; }
QPushButton:hover { background: #3b4056; border-color: #7aa2f7; color: #7aa2f7; }
QPushButton:pressed { background: #414868; }
QPushButton:disabled { color: #414868; background: #1f2233; border-color: #2a2e42; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #1f2233; color: #c0caf5; border: 1px solid #2a2e42;
    border-radius: 4px; padding: 4px 6px; min-height: 22px; }
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border-color: #7aa2f7; }
QTableWidget { background: #1a1b26; color: #c0caf5; gridline-color: #2a2e42;
               alternate-background-color: #1f2233; border: 1px solid #2a2e42; border-radius: 4px; }
QTableWidget::item { background: #1f2233; }
QTableWidget::item:selected { background: #7aa2f7; color: #1a1b26; }
QProgressBar { background: #1f2233; border: 1px solid #2a2e42; border-radius: 4px; text-align: center; color: #1a1b26; font-weight: bold; }
QProgressBar::chunk { background: #7aa2f7; border-radius: 3px; }
QTabWidget::pane { border: 1px solid #2a2e42; border-radius: 4px; top: -1px; }
QTabBar::tab { background: #1f2233; color: #565f89; padding: 8px 16px;
               border: 1px solid #2a2e42; border-bottom: none; 
               border-top-left-radius: 4px; border-top-right-radius: 4px; }
QTabBar::tab:selected { background: #2a2e42; color: #7aa2f7; }
QTabBar::tab:hover:!selected { background: #242838; }
QScrollArea { border: none; }
QScrollBar:vertical { background: #1a1b26; width: 10px; margin: 0px; }
QScrollBar::handle:vertical { background: #2a2e42; border-radius: 4px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background: #3b4056; }
QScrollBar:horizontal { background: #1a1b26; height: 10px; margin: 0px; }
QScrollBar::handle:horizontal { background: #2a2e42; border-radius: 4px; min-width: 20px; }
QCheckBox { color: #a9b1d6; spacing: 6px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; }
QCheckBox::indicator:unchecked { background: #1f2233; border: 1px solid #3b4056; }
QCheckBox::indicator:checked { background: #7aa2f7; border: 1px solid #7aa2f7; }
QSplitter::handle { background: #2a2e42; }
QSplitter::handle:hover { background: #7aa2f7; }
QToolTip { background: #1f2233; color: #c0caf5; border: 1px solid #7aa2f7; border-radius: 4px; padding: 4px; }
"""

def format_bytes(n):
    if n <= 0: return "0 B"
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    f = float(n)
    for u in units[:-1]:
        if abs(f) < 1024:
            return f"{int(f)} {u}" if f == int(f) else f"{f:.1f} {u}"
        f /= 1024
    return f"{f:.1f} {units[-1]}"

def get_gpu_memory():
    if not CUPY_AVAILABLE: return 0, 0
    try:
        f, t = cp.cuda.runtime.memGetInfo()
        return t - f, t
    except RuntimeError:
        return 0, 0

def get_process_memory():
    if not PSUTIL_AVAILABLE: return 0
    try: return psutil.Process().memory_info().rss
    except Exception: return 0

def get_system_memory():
    if not PSUTIL_AVAILABLE: return 0, 0
    try:
        vm = psutil.virtual_memory()
        return vm.available, vm.total
    except Exception: return 0, 0

def get_cpu_usage():
    if not PSUTIL_AVAILABLE: return 0.0
    try: return max(0.0, psutil.cpu_percent(interval=None))
    except Exception: return 0.0

def estimate_peak_memory(ps, bs, H, W, od, cs=32):
    c = min(cs, ps)
    pb = ps * H * W * 3 * 8
    fb = c * bs * W * 8
    ab = c * bs * W * 8 * 2
    pr = c * bs * od * 8
    return {'population': pb, 'forward': fb, 'activations': ab,
            'predictions': pr, 'total': pb + fb + ab + pr}

class Profiler:
    def __init__(self):
        self.timings = {}
        self.calls = {}
        self.gen_start = 0
        self.gen_total = 0
        self.history = deque(maxlen=200)

    def start_gen(self):
        self.timings = {}
        self.calls = {}
        self.gen_start = time.perf_counter()

    def end_gen(self):
        self.gen_total = time.perf_counter() - self.gen_start
        self.history.append(self.gen_total)

    @contextmanager
    def time(self, n):
        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
        self.timings[n] = self.timings.get(n, 0) + dt
        self.calls[n] = self.calls.get(n, 0) + 1

    def get_data(self):
        tp = sum(self.timings.values())
        avg_gen = np.mean(self.history) if self.history else 0
        return {
            'timings': dict(self.timings),
            'calls': dict(self.calls),
            'gen_total': self.gen_total,
            'total_profiled': tp,
            'overhead': max(0, self.gen_total - tp),
            'avg_gen_time': avg_gen,
        }

ACTIVATION_NAMES = ['sigmoid', 'tanh', 'relu', 'leaky', 'linear',
                    'gaussian', 'sin', 'step', 'swish', 'mish',
                    'elu', 'selu', 'softplus', 'hard_sig', 'tanhshrink', 'softsign']
N_ACTIVATIONS = 16

CLASS_NAMES = {
    'digits': ['0','1','2','3','4','5','6','7','8','9'],
    'iris': ['Setosa','Versicolor','Virginica'],
    'wine': ['C0','C1','C2'],
    'breast_cancer': ['Malignant','Benign'],
    'xor': ['0','1'],
    'classification': ['A','B'],
    'spiral': ['A','B']
}

def get_class_names(n, c):
    k = n.lower().replace(' ', '_')
    if k in CLASS_NAMES and len(CLASS_NAMES[k]) >= c:
        return CLASS_NAMES[k][:c]
    return [f'Class {i}' for i in range(c)]

def mann_kendall_trend(data):
    n = len(data)
    if n < 4: return 0, 1.0
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(data[j] - data[i])
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return z, p

def nsga2_sort(objectives):
    n = len(objectives)
    fronts = [[]]
    rank = np.zeros(n, dtype=int)
    counts = np.zeros(n, dtype=int)
    dominated = [[] for _ in range(n)]
    for p in range(n):
        for q in range(n):
            if p == q: continue
            if np.all(objectives[p] <= objectives[q]) and np.any(objectives[p] < objectives[q]):
                dominated[p].append(q)
            elif np.all(objectives[q] <= objectives[p]) and np.any(objectives[q] < objectives[p]):
                counts[p] += 1
        if counts[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated[p]:
                counts[q] -= 1
                if counts[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    crowding = np.zeros(n)
    for front in fronts:
        if len(front) <= 2:
            crowding[front] = float('inf')
            continue
        for obj in range(objectives.shape[1]):
            sorted_idx = sorted(front, key=lambda x: objectives[x, obj])
            crowding[sorted_idx[0]] = float('inf')
            crowding[sorted_idx[-1]] = float('inf')
            vals = objectives[sorted_idx, obj]
            rng = vals[-1] - vals[0]
            if rng > 1e-12:
                for k in range(1, len(sorted_idx) - 1):
                    crowding[sorted_idx[k]] += (vals[k+1] - vals[k-1]) / rng
    return rank, crowding, fronts

class AdaptiveOperatorSelector:
    def __init__(self, operators, credit='average', p_min=0.05):
        self.operators = list(operators)
        self.n = len(self.operators)
        self.credit_assignment = credit
        self.p_min = p_min
        self.q = np.ones(self.n) * 0.5
        self.t = 0
        self.recent_rewards = {op: deque(maxlen=20) for op in self.operators}

    def get_probabilities(self):
        p_max = 1 - (self.n - 1) * self.p_min
        probs = np.maximum(self.q / self.q.sum() if self.q.sum() > 0 else
                          np.ones(self.n) / self.n, 0)
        if probs.sum() == 0:
            probs = np.ones(self.n) / self.n
        probs = self.p_min + probs / probs.sum() * (p_max - self.p_min)
        return probs / probs.sum()

    def select(self):
        self.t += 1
        return np.random.choice(self.n, p=self.get_probabilities())

    def update(self, op_idx, reward):
        self.recent_rewards[self.operators[op_idx]].append(reward)
        if self.credit_assignment == 'average':
            r = np.mean(self.recent_rewards[self.operators[op_idx]]) if \
                self.recent_rewards[self.operators[op_idx]] else 0
        elif self.credit_assignment == 'extreme':
            r = max(self.recent_rewards[self.operators[op_idx]]) if \
                self.recent_rewards[self.operators[op_idx]] else 0
        else:
            r = reward
        alpha = 1.0 / max(1, self.t / 10)
        self.q[op_idx] = (1 - alpha) * self.q[op_idx] + alpha * r

    def get_stats(self):
        return {op: {'q': self.q[i], 'p': self.get_probabilities()[i],
                     'avg_reward': np.mean(self.recent_rewards[op]) if self.recent_rewards[op] else 0}
                for i, op in enumerate(self.operators)}

class Species:
    def __init__(self, representative, members=None, threshold=0.3):
        self.representative = representative
        self.members = members or []
        self.threshold = threshold
        self.best_fitness = -float('inf')
        self.staleness = 0

    def distance(self, genome):
        diff = np.mean(np.abs(genome - self.representative))
        return diff

    def try_add(self, genome, idx):
        if self.distance(genome) < self.threshold:
            self.members.append((idx, genome))
            return True
        return False

def speciate(population, threshold=0.3):
    species_list = []
    for i, g in enumerate(population):
        placed = False
        for sp in species_list:
            if sp.try_add(g, i):
                placed = True
                break
        if not placed:
            sp = Species(g, [(i, g)], threshold)
            species_list.append(sp)
    return species_list

def _hsv_to_rgb_vec(hsv):
    h = hsv[..., 0] % 1.0
    s = np.clip(hsv[..., 1], 0, 1)
    v = np.clip(hsv[..., 2], 0, 1)
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return (np.clip(np.stack([r, g, b], axis=-1), 0, 1) * 255).astype(np.uint8)

def hsv_array_to_pixmap(arr, us=4):
    h, w, _ = arr.shape
    rgb = np.ascontiguousarray(_hsv_to_rgb_vec(arr))
    qi = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi.copy()).scaled(
        w * us, h * us, Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation)

def _create_heatmap_pixmap(arr, cmap='coolwarm', vmin=None, vmax=None):
    if vmin is None: vmin = float(arr.min())
    if vmax is None: vmax = float(arr.max())
    vr = max(vmax - vmin, 1e-12)
    norm = np.clip((arr - vmin) / vr, 0, 1)
    if cmap == 'coolwarm':
        r = (255 * norm).astype(np.uint8)
        b = (255 * (1 - norm)).astype(np.uint8)
        g = (255 * (1 - np.abs(2 * norm - 1))).astype(np.uint8)
    elif cmap == 'diverging':
        norm = np.clip(arr / max(abs(vmin), abs(vmax), 1e-12), -1, 1)
        norm_mapped = (norm + 1) / 2.0
        r = (255 * norm_mapped).astype(np.uint8)
        b = (255 * (1 - norm_mapped)).astype(np.uint8)
        g = (255 * (1 - np.abs(2 * norm_mapped - 1))).astype(np.uint8)
    else:
        r = (255 * norm).astype(np.uint8)
        b = (255 * (1 - norm)).astype(np.uint8)
        g = (255 * (1 - np.abs(2 * norm - 1))).astype(np.uint8)
    rgb = np.ascontiguousarray(np.stack([r, g, b], axis=-1))
    h, w = rgb.shape[:2]
    qi = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi.copy())

# ── Dataset Manager ────────────────────────────────────────────────────────
class DatasetManager:
    def __init__(self):
        self.current_dataset = None
        self.dataset_info = {}
        self.scaler = StandardScaler()
        self.is_classification = False
        self.num_classes = 0
        self.class_weights = None

    def unload(self):
        self.current_dataset = None
        self.dataset_info = {}
        self.is_classification = False
        self.num_classes = 0
        self.class_weights = None

    def get_dataset_info(self):
        return self.dataset_info

    def _compute_class_weights(self, y_onehot):
        if not self.is_classification: return None
        counts = np.sum(y_onehot, axis=0)
        counts = np.maximum(counts, 1)
        total = np.sum(counts)
        return total / (self.num_classes * counts)

    def _process_targets(self, y):
        u = np.unique(y)
        if len(u) <= 20 and all(float(v).is_integer() for v in u):
            self.is_classification = True
            self.num_classes = len(u)
            m = {v: i for i, v in enumerate(u)}
            r = np.array([m[v] for v in y], dtype=np.int64)
            onehot = np.eye(self.num_classes, dtype=np.float64)[r]
            self.class_weights = self._compute_class_weights(onehot)
            return onehot
        self.is_classification = False
        self.num_classes = 1
        return y.astype(np.float64).reshape(-1, 1)

    def augment_data(self, X, y, n_augment=0, noise_std=0.05):
        if n_augment <= 0 or not self.is_classification:
            return X, y
        rng = np.random.default_rng(42)
        aug_X = X + rng.normal(0, noise_std, X.shape)
        return np.vstack([X, aug_X]), np.vstack([y, y])

    def load_builtin(self, name, n=200, seed=42):
        rng = np.random.default_rng(seed)
        if name == "xor":
            X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
            y = np.array([0,1,1,0], dtype=np.float64)
        elif name == "sine":
            X = rng.uniform(0, 2*np.pi, (n, 1)).astype(np.float64)
            y = np.sin(X).flatten()
        elif name == "quadratic":
            X = rng.uniform(-2, 2, (n, 1)).astype(np.float64)
            y = (X**2).flatten()
        elif name == "classification":
            X = np.vstack([rng.normal(-1, .5, (n//2, 2)),
                          rng.normal(1, .5, (n//2, 2))]).astype(np.float64)
            y = np.hstack([np.zeros(n//2), np.ones(n//2)])
        elif name == "high_dim":
            X = rng.normal(0, 1, (n, 10)).astype(np.float64)
            y = np.sum(X[:, :3]**2, axis=1)
        elif name == "spiral":
            X = np.zeros((2*n, 2), dtype=np.float64)
            y = np.zeros(2*n, dtype=np.float64)
            for i in range(n):
                r, t = i/n*5, 1.75*i/n*2*np.pi
                X[i] = [r*np.cos(t), r*np.sin(t)]
            for i in range(n):
                r, t = i/n*5, 1.75*i/n*2*np.pi + np.pi
                X[n+i] = [r*np.cos(t), r*np.sin(t)]
                y[n+i] = 1
        else:
            X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
            y = np.array([0,1,1,0], dtype=np.float64)
        y = self._process_targets(y)
        self._store(X, y, 'builtin', name)
        return X, y

    def load_sklearn(self, name):
        loaders = {"iris": load_iris, "breast_cancer": load_breast_cancer,
                   "digits": load_digits, "wine": load_wine, "diabetes": load_diabetes}
        if name not in loaders: return None, None
        data = loaders[name]()
        X = self.scaler.fit_transform(data.data.astype(np.float64))
        y = self._process_targets(data.target.astype(np.float64))
        self._store(X, y, 'sklearn', name)
        return X, y

    def load_csv(self, path, tc=-1):
        try:
            df = pd.read_csv(path)
            nc = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(nc) < 2: return None, None
            if isinstance(tc, str):
                y = df[tc].values.astype(np.float64)
                feats = [c for c in nc if c != tc]
            else:
                idx = tc if tc >= 0 else len(df.columns) + tc
                col = df.columns[idx]
                if col not in nc: return None, None
                y = df[col].values.astype(np.float64)
                feats = [c for c in nc if c != col]
            if not feats: return None, None
            X = self.scaler.fit_transform(df[feats].values.astype(np.float64))
            y = self._process_targets(y)
            self._store(X, y, 'csv', os.path.basename(path))
            return X, y
        except Exception:
            traceback.print_exc()
            return None, None

    def load_images(self, directory, size=32):
        try:
            if not PIL_AVAILABLE:
                QMessageBox.critical(None, "Error", "Pillow library is required for image loading.")
                return None, None
            imgs, labels = [], []
            classes = sorted(d for d in os.listdir(directory)
                           if os.path.isdir(os.path.join(directory, d)))
            exts = ('.png', '.jpg', '.jpeg', '.bmp')
            if not classes:
                files = [f for f in os.listdir(directory) if f.lower().endswith(exts)]
                for i, f in enumerate(files):
                    im = PILImage.open(os.path.join(directory, f)).convert('L').resize((size, size))
                    imgs.append(np.array(im).flatten().astype(np.float64))
                    labels.append(i)
            else:
                for ci, cn in enumerate(classes):
                    for f in os.listdir(os.path.join(directory, cn)):
                        if f.lower().endswith(exts):
                            im = PILImage.open(os.path.join(directory, cn, f)).convert('L').resize((size, size))
                            imgs.append(np.array(im).flatten().astype(np.float64))
                            labels.append(ci)
            if not imgs: return None, None
            X = self.scaler.fit_transform(np.array(imgs, dtype=np.float64))
            y = self._process_targets(np.array(labels, dtype=np.float64))
            self._store(X, y, 'image', os.path.basename(directory))
            return X, y
        except Exception:
            traceback.print_exc()
            return None, None

    def _store(self, X, y, dt, n):
        self.current_dataset = (X, y)
        self.dataset_info = {
            'type': dt, 'name': n, 'n_samples': X.shape[0], 'n_features': X.shape[1],
            'n_outputs': y.shape[1], 'is_classification': self.is_classification,
            'class_weights': self.class_weights.tolist() if self.class_weights is not None else None
        }

class PCAProcessor:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.pca = None
        self.fitted = False
        self.actual_n = n_components

    def fit(self, X):
        n = min(self.n_components, X.shape[0], X.shape[1])
        n = max(1, n)
        self.actual_n = n
        self.pca = SklearnPCA(n_components=n)
        self.pca.fit(X)
        self.fitted = True
        return self

    def transform(self, X):
        return self.pca.transform(X) if self.fitted else X

    def get_components(self):
        return self.pca.components_ if self.fitted else None

    def get_ev(self):
        return self.pca.explained_variance_ratio_ if self.fitted else None

# ── Async Task for Saving ─────────────────────────────────────────────────
class SaveTask(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = kwargs.pop('signals', None)

    def run(self):
        try:
            self.fn(*self.args, **self.kwargs)
        except Exception as e:
            traceback.print_exc()

# ── View Manager (Dropdown instead of Tabs) ────────────────────────────────
class ViewManager(QWidget):
    currentChanged = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(4)

        self.header_layout = QHBoxLayout()
        self.header_layout.setContentsMargins(5, 2, 5, 2)
        
        self.view_combo = QComboBox()
        self.view_combo.setMinimumHeight(32)
        self.view_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.view_combo.setStyleSheet("font-weight:bold; padding-left: 8px;")
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        
        self.header_layout.addWidget(QLabel("📊 View:"))
        self.header_layout.addWidget(self.view_combo, 1)
        
        self.clear_btn = QPushButton("🧹 Clear")
        self.clear_btn.setToolTip("Clear data from plots.")
        self.clear_btn.clicked.connect(self._clear_views)
        self.header_layout.addWidget(self.clear_btn)
        
        self.main_layout.addLayout(self.header_layout)

        self.stack = QStackedWidget()
        self.main_layout.addWidget(self.stack)
        
        self._item_count = 0

    def addTab(self, widget, name):
        self.stack.addWidget(widget)
        self.view_combo.addItem(name)
        idx = self._item_count
        self._item_count += 1
        return idx

    def addSeparator(self):
        self.view_combo.insertSeparator(self._item_count)
        self._item_count += 1

    def setTabText(self, index, text):
        if 0 <= index < self.view_combo.count():
            if self.view_combo.itemData(index) is None:
                self.view_combo.setItemText(index, text)

    def currentIndex(self):
        return self.view_combo.currentIndex()

    def setCurrentIndex(self, index):
        self.view_combo.setCurrentIndex(index)

    def _on_view_changed(self, index):
        self.stack.setCurrentIndex(index)
        self.currentChanged.emit(index)

    def _clear_views(self):
        p = self.parent()
        while p is not None and not isinstance(p, QMainWindow):
            p = p.parent()
        if p:
            p.clear_visualizations()

# ── Architecture Viewer ────────────────────────────────────────────────────
class ArchitectureViewer(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.setMinimumHeight(450)
        self.states = {}
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._decay)
        self.timer.start(50)
        self.pulse = 0.0

    def update_state(self, state):
        if state in ("evaluate", "test_eval"):
            for s in ["Evaluate", "Input", "Hidden", "Output", "Loss"]:
                self.states[s] = 1.0
        elif state == "fitness":
            self.states["Fitness"] = 1.0
            self.states["Loss"] = max(self.states.get("Loss", 0), 0.5)
        elif state == "selection": self.states["Selection"] = 1.0
        elif state == "crossover": self.states["Crossover"] = 1.0
        elif state == "mutation": self.states["Mutation"] = 1.0
        elif state == "memetic": self.states["Memetic"] = 1.0
        elif state == "pca_init": self.states["Evaluate"] = 0.5

    def _decay(self):
        self.pulse = (self.pulse + 0.1) % 1.0
        for k in list(self.states.keys()):
            self.states[k] = max(0.0, self.states[k] - 0.1)
        self.update()

    def _draw_node(self, p, rect, name, energy, is_ff=True):
        bg = self.palette().color(QPalette.ColorRole.Window)
        is_dark = bg.lightness() < 128
        base_col = QColor("#1f2233") if is_dark else QColor("#e0e0e0")
        active_col = QColor("#7aa2f7") if is_ff else QColor("#bb9af7")
        r = base_col.red() + (active_col.red() - base_col.red()) * energy
        g = base_col.green() + (active_col.green() - base_col.green()) * energy
        b = base_col.blue() + (active_col.blue() - base_col.blue()) * energy
        col = QColor(int(r), int(g), int(b))
        if energy > 0.1:
            glow = QColor(col); glow.setAlphaF(energy * 0.5)
            p.setBrush(glow); p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(rect.adjusted(-4, -4, 4, 4), 8, 8)
        p.setBrush(col)
        p.setPen(QPen(QColor("#3b4056") if is_dark else QColor("#aaa"), 1))
        p.drawRoundedRect(rect, 5, 5)
        text_col = QColor("#fff") if energy > 0.4 or is_dark else QColor("#333")
        p.setPen(text_col)
        p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, name)

    def _draw_arrow(self, p, sr, er, energy=0.0, dashed=False):
        is_dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128
        base_col = QColor("#565f89") if is_dark else QColor("#888")
        active_col = QColor("#e0af68")
        r = base_col.red() + (active_col.red() - base_col.red()) * energy
        g = base_col.green() + (active_col.green() - base_col.green()) * energy
        b = base_col.blue() + (active_col.blue() - base_col.blue()) * energy
        col = QColor(int(r), int(g), int(b))
        pen_style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
        p.setPen(QPen(col, 1.5 + energy * 1.5, pen_style))
        p1 = QPointF(sr.right(), sr.center().y())
        p2 = QPointF(er.left(), er.center().y())
        p.drawLine(p1, p2)
        p.setBrush(col); p.setPen(Qt.PenStyle.NoPen)
        pts = [QPointF(p2.x(), p2.y()),
               QPointF(p2.x()-6, p2.y()-4),
               QPointF(p2.x()-6, p2.y()+4)]
        p.drawPolygon(QPolygonF(pts))

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        bg = self.palette().color(QPalette.ColorRole.Window)
        p.fillRect(0, 0, w, h, bg)
        p.setPen(self.palette().color(QPalette.ColorRole.Text))
        p.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        p.drawText(20, 30, "Feedforward Pipeline (Neural Network Evaluation)")
        p.drawText(20, h // 2 + 20, "Genetic Algorithm Pipeline (Optimization Loop)")
        n_w, n_h = 100, 40
        step_x = w / 9.0
        ff_y = 60
        ff_nodes = {
            "Input": QRectF(step_x * 1 - n_w/2, ff_y, n_w, n_h),
            "Hidden": QRectF(step_x * 2.5 - n_w/2, ff_y, n_w, n_h),
            "Output": QRectF(step_x * 4 - n_w/2, ff_y, n_w, n_h),
            "Loss": QRectF(step_x * 5.5 - n_w/2, ff_y, n_w, n_h)
        }
        ga_y = h // 2 + 50
        ga_nodes = {
            "Population": QRectF(step_x * 1 - n_w/2, ga_y, n_w, n_h),
            "Evaluate": QRectF(step_x * 2.5 - n_w/2, ga_y, n_w, n_h),
            "Fitness": QRectF(step_x * 4 - n_w/2, ga_y, n_w, n_h),
            "Selection": QRectF(step_x * 5.5 - n_w/2, ga_y, n_w, n_h),
            "Crossover": QRectF(step_x * 7 - n_w/2, ga_y, n_w, n_h),
            "Mutation": QRectF(step_x * 8.5 - n_w/2, ga_y, n_w, n_h)
        }
        ff_energies = [self.states.get("Input", 0), self.states.get("Hidden", 0),
                       self.states.get("Output", 0)]
        self._draw_arrow(p, ff_nodes["Input"], ff_nodes["Hidden"], ff_energies[0])
        self._draw_arrow(p, ff_nodes["Hidden"], ff_nodes["Output"], ff_energies[1])
        self._draw_arrow(p, ff_nodes["Output"], ff_nodes["Loss"], ff_energies[2])
        ga_energies = [
            self.states.get("Evaluate", 0),
            self.states.get("Fitness", 0),
            self.states.get("Selection", 0),
            self.states.get("Crossover", 0),
            self.states.get("Mutation", 0)
        ]
        self._draw_arrow(p, ga_nodes["Population"], ga_nodes["Evaluate"], ga_energies[0])
        self._draw_arrow(p, ga_nodes["Evaluate"], ga_nodes["Fitness"], ga_energies[1])
        self._draw_arrow(p, ga_nodes["Fitness"], ga_nodes["Selection"], ga_energies[2])
        self._draw_arrow(p, ga_nodes["Selection"], ga_nodes["Crossover"], ga_energies[3])
        self._draw_arrow(p, ga_nodes["Crossover"], ga_nodes["Mutation"], ga_energies[4])
        pen = QPen(QColor("#565f89"), 1.5, Qt.PenStyle.DashLine)
        p.setPen(pen)
        path = QPainterPath()
        start = QPointF(ga_nodes["Mutation"].center().x(), ga_nodes["Mutation"].bottom() + 10)
        end = QPointF(ga_nodes["Population"].center().x(), ga_nodes["Population"].bottom() + 10)
        ctrl1 = QPointF(start.x() - 50, h - 20)
        ctrl2 = QPointF(end.x() + 50, h - 20)
        path.moveTo(start)
        path.cubicTo(ctrl1, ctrl2, end)
        p.drawPath(path)
        p.setBrush(QColor("#565f89")); p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(QPolygonF([end, QPointF(end.x()+6, end.y()-4), QPointF(end.x()+6, end.y()+4)]))
        e_eval = self.states.get("Evaluate", 0)
        e_fit = self.states.get("Fitness", 0)
        self._draw_arrow(p, ga_nodes["Evaluate"], ff_nodes["Input"], e_eval, dashed=True)
        self._draw_arrow(p, ff_nodes["Loss"], ga_nodes["Fitness"], e_fit, dashed=True)
        for name, rect in ff_nodes.items():
            self._draw_node(p, rect, name, self.states.get(name, 0.0), is_ff=True)
        for name, rect in ga_nodes.items():
            self._draw_node(p, rect, name, self.states.get(name, 0.0), is_ff=False)
        if self.states.get("Memetic", 0) > 0.1:
            p.setPen(QPen(QColor("#f7768e"), 2, Qt.PenStyle.DashLine))
            p.drawEllipse(ga_nodes["Selection"].center(), 35, 35)
            p.setPen(QColor("#f7768e"))
            p.setFont(QFont("Arial", 8, QFont.Weight.Bold))
            p.drawText(QRectF(ga_nodes["Selection"].center().x()-40,
                              ga_nodes["Selection"].center().y()-30, 80, 16),
                       Qt.AlignmentFlag.AlignCenter, "⚡ Memetic")
        p.end()

# ── Visualization Widgets ──────────────────────────────────────────────────
class ConvergencePlot(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.tl, self.tel, self.ta, self.tea = [], [], [], []
        self.tl_th, self.ta_th, self.clf, self.ma, self.mw = None, None, False, True, 20
        self.pareto_fronts = []
        self.setMinimumHeight(200)

    def set_thresholds(self, lt, at, cl):
        self.tl_th = lt; self.ta_th = at; self.clf = cl

    def add_point(self, tl, tel, ta, tea):
        self.tl.append(tl); self.tel.append(tel)
        self.ta.append(ta); self.tea.append(tea)
        if len(self.tl) > 2000:
            self.tl = self.tl[-2000:]; self.tel = self.tel[-2000:]
            self.ta = self.ta[-2000:]; self.tea = self.tea[-2000:]
        self.update()

    def clear_data(self):
        self.tl = []; self.tel = []; self.ta = []; self.tea = []
        self.pareto_fronts = []
        self.update()

    @staticmethod
    def _ma(v, w):
        return np.convolve(v, np.ones(w)/w, 'valid').tolist() if len(v) >= w else v

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        bg = self.palette().color(QPalette.ColorRole.Window)
        p.fillRect(0, 0, w, h, bg)
        if len(self.tl) < 2:
            p.setPen(self.palette().color(QPalette.ColorRole.Text))
            p.setFont(QFont("Arial", 10))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Convergence plot")
            p.end(); return
        ml, mr, mt, mb = 60, 110, 15, 35
        pw, ph = w - ml - mr, h - mt - mb
        grid_col = self.palette().color(QPalette.ColorRole.Mid)
        p.setPen(QPen(grid_col, 1))
        for i in range(1, 5):
            y = mt + ph * i / 5
            p.drawLine(ml, int(y), ml + pw, int(y))
        text_col = self.palette().color(QPalette.ColorRole.Text)
        p.setPen(QPen(text_col, 1))
        p.drawLine(ml, mt, ml, mt + ph)
        p.drawLine(ml, mt + ph, ml + pw, mt + ph)
        n = len(self.tl)
        al = self.tl + self.tel
        mx, mn = max(al), min(al)
        lr = max(mx - mn, 1e-12)

        def _d(v, c, d=False, t=2, a=255):
            st = Qt.PenStyle.DashLine if d else Qt.PenStyle.SolidLine
            cl = QColor(c); cl.setAlpha(a)
            p.setPen(QPen(cl, t, st))
            pr = None
            for i in range(len(v)):
                x = ml + i / max(len(v)-1, 1) * pw
                y = mt + (1 - (v[i] - mn) / lr) * ph
                if pr: p.drawLine(QPointF(pr[0], pr[1]), QPointF(x, y))
                pr = (x, y)

        if self.tl_th and mn <= self.tl_th <= mx:
            ty = mt + (1 - (self.tl_th - mn) / lr) * ph
            p.setPen(QPen(QColor("#f7768e"), 1, Qt.PenStyle.DashLine))
            p.drawLine(ml, int(ty), ml + pw, int(ty))
        _d(self.tl, "#7aa2f7"); _d(self.tel, "#e0af68")
        if self.ma and n > self.mw:
            _d(self._ma(self.tl, self.mw), "#7aa2f7", t=3, a=100)
            _d(self._ma(self.tel, self.mw), "#e0af68", t=3, a=100)
        if self.clf and any(a > 0 for a in self.tea):
            if self.ta_th and self.ta_th > 0:
                ay = mt + (1 - self.ta_th) * ph
                p.setPen(QPen(QColor("#9ece6a"), 1, Qt.PenStyle.DashLine))
                p.drawLine(ml, int(ay), ml + pw, int(ay))
            _d(self.ta, "#9ece6a", d=True); _d(self.tea, "#73daca", d=True)
        p.setPen(text_col); p.setFont(QFont("Arial", 8))
        p.drawText(ml, h - 15, "Gen 1")
        p.drawText(ml + pw - 50, h - 15, f"Gen {n}")
        p.drawText(5, mt + 10, f"max:{mx:.4f}")
        p.drawText(5, mt + ph - 5, f"min:{mn:.4f}")
        lx, ly = ml + pw + 10, mt + 5
        for lb, c, d in [("Train Loss", "#7aa2f7", False),
                         ("Test Loss", "#e0af68", False)] + \
                        [("Train Acc", "#9ece6a", True),
                         ("Test Acc", "#73daca", True)] * self.clf:
            st = Qt.PenStyle.DashLine if d else Qt.PenStyle.SolidLine
            p.setPen(QPen(QColor(c), 2, st))
            p.drawLine(lx, ly + 4, lx + 20, ly + 4)
            p.setPen(text_col)
            p.drawText(lx + 25, ly + 8, lb)
            ly += 16
        p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        p.setPen(QColor("#7aa2f7")); p.drawText(ml + 5, mt + 15, f"TrLoss:{self.tl[-1]:.6f}")
        p.setPen(QColor("#e0af68")); p.drawText(ml + 5, mt + 30, f"TeLoss:{self.tel[-1]:.6f}")
        if self.clf:
            p.setPen(QColor("#9ece6a")); p.drawText(ml + 5, mt + 45, f"TrAcc:{self.ta[-1]*100:.2f}%")
            p.setPen(QColor("#73daca")); p.drawText(ml + 5, mt + 60, f"TeAcc:{self.tea[-1]*100:.2f}%")
        p.end()

class PCAComponentsViewer(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.components = None; self.ev = None; self.nf = 0
        main = QVBoxLayout(self)
        main.setContentsMargins(4, 4, 4, 4)
        self.summary = QLabel("PCA fitted on train set when evolution starts.")
        self.summary.setWordWrap(True)
        main.addWidget(self.summary)
        self.cl = QLabel()
        self.cl.setMinimumHeight(180)
        self.cl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main.addWidget(self.cl)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        sc = QWidget()
        self.cl2 = QVBoxLayout(sc)
        self.cl2.setSpacing(4)
        self.scroll.setWidget(sc)
        main.addWidget(self.scroll, 1)

    def set_pca(self, c, ev, nf):
        self.components = c; self.ev = ev; self.nf = nf
        self._r()

    def clear_view(self):
        self.components = None; self.ev = None
        while self.cl2.count():
            it = self.cl2.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        self.summary.setText("PCA fitted on train set when evolution starts.")
        self.cl.clear()

    def _r(self):
        while self.cl2.count():
            it = self.cl2.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        if self.components is None: return
        nc = self.components.shape[0]
        cum = np.cumsum(self.ev)
        self.summary.setText(f"✓ PCA: {nc} from {self.nf} | Total: {cum[-1]*100:.2f}%")
        for i in range(nc):
            self.cl2.addWidget(self._card(i, cum))
        self.cl2.addStretch()
        self._chart(cum)

    def _card(self, idx, cum):
        c = QFrame()
        c.setFrameStyle(QFrame.Shape.Box)
        l = QHBoxLayout(c)
        l.setContentsMargins(6, 4, 6, 4)
        lb = QLabel(f"PC{idx+1}")
        lb.setFixedWidth(45)
        lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lb.setStyleSheet("font-weight:bold;font-size:14px;color:#7aa2f7;")
        l.addWidget(lb)
        pix = self._pix(self.components[idx])
        il = QLabel(); il.setPixmap(pix)
        l.addWidget(il, 1)
        ev = self.ev[idx]; cu = cum[idx]
        nm = float(np.linalg.norm(self.components[idx]))
        nz = int(np.sum(np.abs(self.components[idx]) > 1e-8))
        st = QLabel(f"Var:{ev*100:>6.2f}%\nCum:{cu*100:>6.2f}%\nNorm:{nm:.4f}\nNZ:{nz}/{len(self.components[idx])}")
        st.setFixedWidth(130)
        st.setStyleSheet("font-family:monospace;font-size:11px;")
        l.addWidget(st)
        return c

    def _pix(self, comp, us=8):
        vmn, vmx = float(comp.min()), float(comp.max())
        r = max(vmx - vmn, 1e-12)
        n = (comp - vmn) / r
        sq = int(np.sqrt(len(n)))
        h, w = (sq, sq) if sq*sq == len(n) else (1, len(n))
        n2 = n.reshape(h, w)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        ml = n2 < 0.5
        tl = n2 * 2
        mh = ~ml
        th = (n2 - 0.5) * 2
        rgb[..., 0] = np.where(ml, (255*tl).astype(np.uint8), 255)
        rgb[..., 1] = np.where(ml, (255*tl).astype(np.uint8), (255*(1-th)).astype(np.uint8))
        rgb[..., 2] = np.where(ml, 255, (255*(1-th)).astype(np.uint8)
)
        rgb = np.ascontiguousarray(rgb)
        qi = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qi.copy()).scaled(
            max(w*us, 120), max(h*us, 24),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

    def _chart(self, cum):
        n = len(self.ev)
        cw, ch = 600, 200
        pm = QPixmap(cw, ch)
        pm.fill(QColor("#fff"))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        ml, mr, mt, mb = 50, 150, 20, 35
        pw, ph = cw - ml - mr, ch - mt - mb
        p.setPen(QPen(QColor("#eee"), 1))
        for i in range(1, 5):
            p.drawLine(ml, int(mt + ph*i/5), ml+pw, int(mt + ph*i/5))
        p.setPen(QPen(QColor("#ccc"), 1))
        p.drawLine(ml, mt, ml, mt+ph)
        p.drawLine(ml, mt+ph, ml+pw, mt+ph)
        bw = pw / n
        mx = max(self.ev)
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            x = ml + i * bw
            bh = (self.ev[i] / max(mx, 1e-12)) * ph
            r2 = self.ev[i] / max(mx, 1e-12)
            p.setBrush(QColor(int(100+100*(1-r2)), int(100+100*(1-r2)), int(200+55*r2)))
            p.drawRect(QRectF(x + bw*.1, mt + ph - bh, bw*.8, bh))
        p.setPen(QPen(QColor("#cc4400"), 2))
        pr = None
        for i in range(n):
            x = ml + (i+.5) * bw
            y = mt + (1 - cum[i]) * ph
            if pr: p.drawLine(QPointF(pr[0], pr[1]), QPointF(x, y))
            pr = (x, y)
        p.setBrush(QColor("#cc4400")); p.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            x = ml + (i+.5) * bw
            y = mt + (1 - cum[i]) * ph
            p.drawEllipse(QPointF(x, y), 3, 3)
        p.setPen(QColor("#333")); p.setFont(QFont("Arial", 8))
        for i in range(6):
            p.drawText(5, int(mt + ph*i/5 + 4), f"{mx*(1-i/5)*100:.1f}%")
            p.drawText(ml + pw + 5, int(mt + ph*i/5 + 4), f"{(1-i/5)*100:.0f}%")
        for i in range(n):
            p.drawText(int(ml + (i+.5)*bw - 12), int(mt + ph + 14), f"PC{i+1}")
        p.end()
        self.cl.setPixmap(pm)

class ActivationLegend(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.setMinimumHeight(36); self.setMaximumHeight(36)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        bg = self.palette().color(QPalette.ColorRole.Window)
        p.fillRect(0, 0, w, h, bg)
        bw = w / N_ACTIVATIONS
        for i, nm in enumerate(ACTIVATION_NAMES):
            hue = (i + .5) / N_ACTIVATIONS
            rgb = _hsv_to_rgb_vec(np.array([[[hue, .85, .9]]]))[0, 0]
            p.setBrush(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
            p.setPen(QPen(self.palette().color(QPalette.ColorRole.Mid), 1))
            p.drawRect(int(i*bw), 0, int(bw), h-12)
            p.setPen(self.palette().color(QPalette.ColorRole.Text))
            p.setFont(QFont("Arial", 7))
            p.drawText(int(i*bw + 2), h-2, nm)
        p.end()

class ActivationStats(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.counts = np.zeros(N_ACTIVATIONS)
        self.setMinimumHeight(70); self.setMaximumHeight(70)

    def update_counts(self, c):
        self.counts = np.array(c, dtype=float); self.update()

    def clear(self):
        self.counts = np.zeros(N_ACTIVATIONS); self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        bg = self.palette().color(QPalette.ColorRole.Window)
        p.fillRect(0, 0, w, h, bg)
        total = max(self.counts.sum(), 1)
        bw = w / N_ACTIVATIONS
        mh = h - 16
        for i in range(N_ACTIVATIONS):
            hue = (i + .5) / N_ACTIVATIONS
            rgb = _hsv_to_rgb_vec(np.array([[[hue, .85, .9]]]))[0, 0]
            pct = self.counts[i] / total
            bh = int(pct * mh)
            p.setBrush(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(int(i*bw + 2), int(mh - bh), int(bw - 4), int(bh))
            p.setPen(self.palette().color(QPalette.ColorRole.Text))
            p.setFont(QFont("Arial", 7))
            p.drawText(int(i*bw + 2), int(mh + 12),
                      f"{ACTIVATION_NAMES[i][:4]} {pct*100:.0f}%")
        p.end()

class NetworkTopologyView(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.image = None; self.setMinimumHeight(350)

    def set_image(self, img):
        self.image = img; self.update()

    def clear(self):
        self.image = None; self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        bg = self.palette().color(QPalette.ColorRole.Window)
        p.fillRect(0, 0, w, h, bg)
        if self.image is None:
            p.setPen(self.palette().color(QPalette.ColorRole.Text))
            p.setFont(QFont("Arial", 11))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                      "Network topology\nColor=activation Size=weight Brightness=bias")
            p.end(); return
        H, W, _ = self.image.shape
        ml = min(H - 1, 15)
        ls = max(1, (H - 1) // ml)
        mn = min(W, 20)
        ns = max(1, W // mn)
        m = 40
        pw, ph = w - 2*m, h - 2*m
        lw = pw / ml
        grid_col = self.palette().color(QPalette.ColorRole.Mid)
        p.setPen(QPen(grid_col.lighter(), 1))
        for li in range(ml - 1):
            x1 = m + li*lw + lw/2
            x2 = m + (li+1)*lw + lw/2
            for nj in range(mn):
                y = m + nj * (ph/mn) + ph/(2*mn)
                p.drawLine(int(x1), int(y), int(x2), int(y))
        for li in range(ml):
            i = li * ls
            if i >= H - 1: break
            x = m + li*lw + lw/2
            for nj in range(mn):
                j = nj * ns
                if j >= W: break
                y = m + nj * (ph/mn) + ph/(2*mn)
                hue = self.image[i, j, 0]
                sat = self.image[i, j, 1]
                val = self.image[i, j, 2]
                rgb = _hsv_to_rgb_vec(np.array([[[hue, .85, max(.3, val)]]]))[0, 0]
                r = 3 + sat * 7
                if sat > .7:
                    g = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
                    g.setAlpha(50)
                    p.setBrush(g); p.setPen(Qt.PenStyle.NoPen)
                    p.drawEllipse(int(x - r*2), int(y - r*2), int(r*4), int(r*4))
                p.setBrush(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
                p.setPen(QPen(QColor(80, 80, 80), 1))
                p.drawEllipse(int(x - r), int(y - r), int(r*2), int(r*2))
        p.end()

class EvolutionHistory(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.snapshots = []
        l = QVBoxLayout(self)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        c = QWidget()
        self.grid = QVBoxLayout(c)
        self.grid.setSpacing(4)
        self.scroll.setWidget(c)
        l.addWidget(self.scroll)
        self._ph()

    def _ph(self):
        lb = QLabel("Snapshots appear every 50 gens")
        lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lb.setStyleSheet("color:#565f89;padding:20px;")
        self.grid.addWidget(lb)

    def add_snapshot(self, g, l, img):
        while self.grid.count():
            it = self.grid.takeAt(0)
            it.widget().deleteLater() if it.widget() else None
        self.snapshots.append((g, l, img))
        if len(self.snapshots) > 30:
            self.snapshots = self.snapshots[-30:]
        for idx, (gg, ll, im) in enumerate(self.snapshots):
            pix = hsv_array_to_pixmap(im, 2)
            lb = QLabel(); lb.setPixmap(pix)
            lb.setToolTip(f"Gen {gg} | Loss {ll:.6f}")
            lb.setStyleSheet("border:1px solid #2a2e42;")
            self.grid.addWidget(lb)

    def clear_history(self):
        self.snapshots = []
        while self.grid.count():
            it = self.grid.takeAt(0)
            it.widget().deleteLater() if it.widget() else None
        self._ph()

class GeneImportanceView(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.hm = None; self.setMinimumHeight(250)

    def update_heatmap(self, h):
        self.hm = h; self.update()

    def clear(self):
        self.hm = None; self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        bg = self.palette().color(QPalette.ColorRole.Window)
        p.fillRect(0, 0, w, h, bg)
        if self.hm is None:
            p.setPen(self.palette().color(QPalette.ColorRole.Text))
            p.setFont(QFont("Arial", 10))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                      "Gene Importance\nRed=high change Blue=stable")
            p.end(); return
        hm = self.hm
        if hm.ndim == 3: hm = np.mean(hm, axis=2)
        pm = _create_heatmap_pixmap(hm, cmap='coolwarm')
        p.drawPixmap(10, 10, pm.scaled(w-20, h-40,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        text_col = self.palette().color(QPalette.ColorRole.Text)
        p.setPen(text_col); p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        p.drawText(10, h-10, "Low Change")
        p.drawText(w-70, h-10, "High Change")
        p.end()

class WeightHistogram(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.weights = None; self.setMinimumHeight(150)

    def update_weights(self, img):
        self.weights = img[..., 1].flatten(); self.update()

    def clear(self):
        self.weights = None; self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        bg = self.palette().color(QPalette.ColorRole.Window)
        p.fillRect(0, 0, w, h, bg)
        text_col = self.palette().color(QPalette.ColorRole.Text)
        if self.weights is None:
            p.setPen(text_col)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Weight Distribution")
            p.end(); return
        nb = 20
        counts, edges = np.histogram(self.weights, bins=nb, range=(0, 1))
        mc = max(counts) if max(counts) > 0 else 1
        m = 30
        pw, ph = w - 2*m, h - 2*m
        bw = pw / nb
        p.setPen(text_col); p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        p.drawText(m, 15, "Weight Distribution (Saturation)")
        p.setPen(Qt.PenStyle.NoPen)
        for i, c in enumerate(counts):
            x = m + i * bw
            bh = (c / mc) * ph
            val = edges[i]
            rgb = _hsv_to_rgb_vec(np.array([[[.6, val, .8]]]))[0, 0]
            p.setBrush(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
            p.drawRect(QRectF(x + 1, m + ph - bh, bw - 2, bh))
        p.setPen(text_col); p.setFont(QFont("Arial", 7))
        for i in range(0, nb + 1, 4):
            p.drawText(int(m + i*bw - 10), int(m + ph + 12), f"{(i/nb)*2-1:+.1f}")
        p.end()

class ActivationTimeline(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.history = []; self.setMinimumHeight(200)

    def add_point(self, g, c):
        self.history.append((g, np.array(c, dtype=float)))
        if len(self.history) > 300:
            self.history = self.history[-300:]
        self.update()

    def clear(self):
        self.history = []; self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        bg = self.palette().color(QPalette.ColorRole.Window)
        p.fillRect(0, 0, w, h, bg)
        text_col = self.palette().color(QPalette.ColorRole.Text)
        if len(self.history) < 2:
            p.setPen(text_col)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Activation Timeline")
            p.end(); return
        n = len(self.history)
        m = 40
        pw, ph = w - 2*m, h - 2*m
        totals = [c.sum() for _, c in self.history]
        mt = max(totals) if max(totals) > 0 else 1
        bottom = np.zeros(n)
        for a in range(N_ACTIVATIONS):
            frac = np.array([c[a] / mt for _, c in self.history])
            top = bottom + frac
            hue = (a + .5) / N_ACTIVATIONS
            rgb = _hsv_to_rgb_vec(np.array([[[hue, .7, .85]]]))[0, 0]
            cl = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            cl.setAlpha(180)
            p.setBrush(cl); p.setPen(Qt.PenStyle.NoPen)
            pts = []
            for i in range(n):
                x = m + i / max(n-1, 1) * pw
                y = m + (1 - top[i]) * ph
                pts.append(QPointF(x, y))
            for i in range(n-1, -1, -1):
                x = m + i / max(n-1, 1) * pw
                y = m + (1 - bottom[i]) * ph
                pts.append(QPointF(x, y))
            p.drawPolygon(QPolygonF(pts))
            bottom = top
        p.end()

class FitnessDiversityPlot(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.points = []; self.setMinimumHeight(200)

    def add_point(self, d, f):
        self.points.append((d, f))
        if len(self.points) > 500:
            self.points = self.points[-500:]
        self.update()

    def clear(self):
        self.points = []; self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        bg = self.palette().color(QPalette.ColorRole.Window)
        p.fillRect(0, 0, w, h, bg)
        text_col = self.palette().color(QPalette.ColorRole.Text)
        if len(self.points) < 2:
            p.setPen(text_col)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Fitness vs Diversity")
            p.end(); return
        m = 45
        pw, ph = w - 2*m, h - 2*m
        grid_col = self.palette().color(QPalette.ColorRole.Mid)
        p.setPen(QPen(grid_col, 1))
        for i in range(1, 5):
            y = m + ph * i / 5
            p.drawLine(m, int(y), m + pw, int(y))
        p.setPen(QPen(text_col, 1))
        p.drawLine(m, m, m, m+ph)
        p.drawLine(m, m+ph, m+pw, m+ph)
        ds = [d for d, _ in self.points]
        fs = [f for _, f in self.points]
        dmn, dmx = min(ds), max(ds)
        fmn, fmx = min(fs), max(fs)
        dr = max(dmx - dmn, 1e-12)
        fr = max(fmx - fmn, 1e-12)
        n = len(self.points)
        for i, (d, f) in enumerate(self.points):
            x = m + (d - dmn) / dr * pw
            y = m + (1 - (f - fmn) / fr) * ph
            t = i / max(n - 1, 1)
            p.setBrush(QColor(int(50 + 205*t), 100, int(50 + 205*(1-t))))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(x, y), 3, 3)
        bi = int(np.argmin(fs))
        bx = m + (ds[bi] - dmn) / dr * pw
        by = m + (1 - (fs[bi] - fmn) / fr) * ph
        p.setBrush(QColor(255, 215, 0))
        p.setPen(QPen(QColor(0, 0, 0), 1))
        p.drawEllipse(QPointF(bx, by), 5, 5)
        p.end()

class NeuronContributionView(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.contrib = None; self.setMinimumHeight(250)

    def set_contribution(self, c):
        self.contrib = c; self.update()

    def clear(self):
        self.contrib = None; self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        bg = self.palette().color(QPalette.ColorRole.Window)
        p.fillRect(0, 0, w, h, bg)
        text_col = self.palette().color(QPalette.ColorRole.Text)
        if self.contrib is None:
            p.setPen(text_col)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                      "Neuron Contribution\nRed=important Blue=unimportant")
            p.end(); return
        hm = self.contrib
        pm = _create_heatmap_pixmap(hm, cmap='diverging',
                                     vmin=float(hm.min()), vmax=float(hm.max()))
        p.drawPixmap(10, 10, pm.scaled(w-20, h-40,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        p.setPen(text_col); p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        p.drawText(10, h-10, "Inhibitory")
        p.drawText(w-60, h-10, "Excitatory")
        p.end()

class PopulationPCAView(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.coords = None; self.losses = None; self.setMinimumHeight(250)

    def set_data(self, c, l):
        self.coords = c; self.losses = l; self.update()

    def clear(self):
        self.coords = None; self.losses = None; self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        bg = self.palette().color(QPalette.ColorRole.Window)
        p.fillRect(0, 0, w, h, bg)
        text_col = self.palette().color(QPalette.ColorRole.Text)
        if self.coords is None:
            p.setPen(text_col)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                      "Population PCA\nRed=bad Green=good Gold=best")
            p.end(); return
        m = 45; cw = 20
        pw, ph = w - 2*m - cw, h - 2*m
        grid_col = self.palette().color(QPalette.ColorRole.Mid)
        p.setPen(QPen(grid_col, 1))
        for i in range(1, 5):
            y = m + ph * i / 5
            p.drawLine(m, int(y), m + pw, int(y))
            x = m + pw * i / 5
            p.drawLine(int(x), m, int(x), m + ph)
        p.setPen(QPen(text_col, 1))
        p.drawLine(m, m, m, m+ph)
        p.drawLine(m, m+ph, m+pw, m+ph)
        xmn, xmx = self.coords[:, 0].min(), self.coords[:, 0].max()
        ymn, ymx = self.coords[:, 1].min(), self.coords[:, 1].max()
        xr = max(xmx - xmn, 1e-12)
        yr = max(ymx - ymn, 1e-12)
        lmn, lmx = self.losses.min(), self.losses.max()
        lr = max(lmx - lmn, 1e-12)
        for i in range(len(self.coords)):
            x = m + (self.coords[i, 0] - xmn) / xr * pw
            y = m + (1 - (self.coords[i, 1] - ymn) / yr) * ph
            t = 1 - (self.losses[i] - lmn) / lr
            p.setBrush(QColor(int(255*(1-t)), int(255*t), 50))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(x, y), 4, 4)
        bi = int(np.argmin(self.losses))
        bx = m + (self.coords[bi, 0] - xmn) / xr * pw
        by = m + (1 - (self.coords[bi, 1] - ymn) / yr) * ph
        p.setBrush(QColor(255, 215, 0))
        p.setPen(QPen(QColor(0, 0, 0), 1))
        p.drawEllipse(QPointF(bx, by), 6, 6)
        cb_x = m + pw + 15
        grad = QLinearGradient(cb_x, m, cb_x, m+ph)
        grad.setColorAt(0, QColor(50, 255, 50))
        grad.setColorAt(1, QColor(255, 50, 50))
        p.fillRect(QRectF(cb_x, m, 15, ph), grad)
        p.setPen(text_col); p.drawRect(QRectF(cb_x, m, 15, ph))
        p.setFont(QFont("Arial", 8))
        p.drawText(cb_x - 5, m - 5, "Good")
        p.drawText(cb_x - 5, m + ph + 12, "Bad")
        p.drawText(m + pw//2 - 20, h - 10, "PC1")
        p.save()
        p.translate(15, m + ph//2 + 20)
        p.rotate(-90)
        p.drawText(0, 0, "PC2")
        p.restore()
        p.end()

class EvolutionSlideshow(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.snapshots = []; self.cur = 0; self.playing = False
        self.setMinimumHeight(350)
        l = QVBoxLayout(self)
        ctrl = QHBoxLayout()
        self.pb = QPushButton("▶ Play"); self.pb.clicked.connect(self._tg)
        self.prv = QPushButton("◀"); self.prv.clicked.connect(self._pv)
        self.nxt = QPushButton("▶"); self.nxt.clicked.connect(self._nx)
        self.sp = QSpinBox(); self.sp.setRange(100, 5000)
        self.sp.setValue(500); self.sp.setSuffix(" ms")
        ctrl.addWidget(self.pb); ctrl.addWidget(self.prv)
        ctrl.addWidget(self.nxt); ctrl.addWidget(QLabel("Speed:"))
        ctrl.addWidget(self.sp); ctrl.addStretch()
        l.addLayout(ctrl)
        self.il = QLabel("No snapshots")
        self.il.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.il.setMinimumHeight(250)
        l.addWidget(self.il)
        self.info = QLabel(""); self.info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l.addWidget(self.info)
        self.timer = QTimer(); self.timer.timeout.connect(self._nx)

    def set_snapshots(self, sn):
        self.snapshots = sn; self.cur = 0; self._sh()

    def _tg(self):
        if self.playing:
            self.timer.stop(); self.playing = False; self.pb.setText("▶ Play")
        else:
            self.timer.start(self.sp.value()); self.playing = True; self.pb.setText("⏸ Pause")

    def _nx(self):
        if not self.snapshots: return
        self.cur = (self.cur + 1) % len(self.snapshots); self._sh()

    def _pv(self):
        if not self.snapshots: return
        self.cur = (self.cur - 1) % len(self.snapshots); self._sh()

    def _sh(self):
        if not self.snapshots: return
        g, l, img = self.snapshots[self.cur]
        self.il.setPixmap(hsv_array_to_pixmap(img, 4))
        self.info.setText(f"Gen {g} | Loss {l:.6f} | {self.cur+1}/{len(self.snapshots)}")

    def clear(self):
        self.snapshots = []; self.cur = 0
        self.il.setText("No snapshots"); self.info.setText("")

class LossLandscapeView(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.landscape = None; self.setMinimumHeight(350)
        l = QVBoxLayout(self)
        btn_row = QHBoxLayout()
        self.btn = QPushButton("🔬 Compute Loss Landscape")
        self.btn.setEnabled(False)
        self.btn.clicked.connect(self._compute_request)
        btn_row.addWidget(self.btn)
        self.btn3d = QPushButton("🌐 3D View")
        self.btn3d.setEnabled(False)
        self.btn3d.clicked.connect(self._show_3d)
        btn_row.addWidget(self.btn3d)
        l.addLayout(btn_row)
        self.info = QLabel("Perturbs best in 2 random directions\nBlue=low loss Red=high loss Gold=best")
        l.addWidget(self.info)

    def _compute_request(self):
        pass

    def set_enabled(self, en):
        self.btn.setEnabled(en)

    def set_landscape(self, ls):
        self.landscape = ls
        self.btn3d.setEnabled(True)
        self.update()

    def _show_3d(self):
        if self.landscape is None or not MATPLOTLIB_AVAILABLE: return
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        n = self.landscape.shape[0]
        X, Y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
        surf = ax.plot_surface(X, Y, self.landscape, cmap='coolwarm',
                              alpha=0.8, edgecolor='none')
        ax.set_xlabel('Direction 1')
        ax.set_ylabel('Direction 2')
        ax.set_zlabel('Loss')
        ax.set_title('3D Loss Landscape')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def clear(self):
        self.landscape = None; self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing |
                       QPainter.RenderHint.SmoothPixmapTransform)
        w, h = self.width(), self.height()
        bg = self.palette().color(QPalette.ColorRole.Window)
        p.fillRect(0, 0, w, h, bg)
        if self.landscape is None:
            p.end(); return
        hm = self.landscape
        n = hm.shape[0]
        m = 50; cw = 20
        pw, ph = w - 2*m - cw, h - 2*m
        vmn, vmx = float(hm.min()), float(hm.max())
        pm = _create_heatmap_pixmap(hm, cmap='coolwarm', vmin=vmn, vmax=vmx)
        p.drawPixmap(QRectF(m, m, pw, ph), pm.scaled(pw, ph,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        text_col = self.palette().color(QPalette.ColorRole.Text)
        p.setPen(QPen(text_col, 1))
        p.drawRect(QRectF(m, m, pw, ph))
        cb_x = m + pw + 15
        grad = QLinearGradient(cb_x, m, cb_x, m+ph)
        grad.setColorAt(0, QColor(0, 0, 255))
        grad.setColorAt(0.5, QColor(255, 255, 255))
        grad.setColorAt(1, QColor(255, 0, 0))
        p.fillRect(QRectF(cb_x, m, 15, ph), grad)
        p.setPen(text_col); p.drawRect(QRectF(cb_x, m, 15, ph))
        p.setFont(QFont("Arial", 8))
        p.drawText(cb_x - 5, m - 5, f"{vmn:.2f}")
        p.drawText(cb_x - 5, m + ph + 12, f"{vmx:.2f}")
        cx = m + (n//2) * (pw/n)
        cy = m + (n//2) * (ph/n)
        p.setBrush(QColor(255, 215, 0))
        p.setPen(QPen(QColor(0, 0, 0), 1))
        p.drawEllipse(QPointF(cx, cy), 6, 6)
        p.setPen(text_col); p.setFont(QFont("Arial", 9))
        p.drawText(m + pw//2 - 50, h - 15, "Perturbation Direction 1")
        p.save()
        p.translate(15, m + ph//2 + 50); p.rotate(-90)
        p.drawText(0, 0, "Perturbation Direction 2")
        p.restore()
        p.end()

class ConfusionMatrixWidget(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.cm = None
        self.class_names = []
        self.setMinimumHeight(300)
        l = QVBoxLayout(self)
        self.info_label = QLabel("Confusion matrix appears after evolution (classification only)")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("padding:20px;color:#565f89;")
        l.addWidget(self.info_label)
        self.metrics_label = QLabel("")
        self.metrics_label.setStyleSheet("font-family:monospace;font-size:11px;padding:5px;")
        l.addWidget(self.metrics_label)

    def set_confusion_matrix(self, cm, class_names):
        self.cm = cm
        self.class_names = class_names[:cm.shape[0]]
        self.info_label.setText("")
        if cm.shape[0] == cm.shape[1]:
            report_lines = ["Classification Report:"]
            for i in range(cm.shape[0]):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-12)
                name = self.class_names[i] if i < len(self.class_names) else f"C{i}"
                report_lines.append(f"  {name:>12}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} (n={cm[i,:].sum()})")
            acc = np.trace(cm) / max(cm.sum(), 1)
            report_lines.append(f"  Overall Accuracy: {acc*100:.2f}%")
            self.metrics_label.setText("\n".join(report_lines))
        self.update()

    def clear(self):
        self.cm = None
        self.class_names = []
        self.info_label.setText("Confusion matrix appears after evolution (classification only)")
        self.metrics_label.setText("")
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        bg = self.palette().color(QPalette.ColorRole.Window)
        p.fillRect(0, 0, w, h, bg)
        if self.cm is None:
            p.end(); return
        n = self.cm.shape[0]
        margin = 60
        plot_size = min(w, h) - 2 * margin
        cell_size = plot_size / n
        text_col = self.palette().color(QPalette.ColorRole.Text)
        cm_max = self.cm.max() if self.cm.max() > 0 else 1
        for i in range(n):
            for j in range(n):
                x = margin + j * cell_size
                y = margin + i * cell_size
                val = self.cm[i, j]
                intensity = val / cm_max
                if i == j:
                    color = QColor(int(100 + 155 * (1 - intensity)),
                                   int(200 + 55 * intensity), 100)
                else:
                    color = QColor(255, int(150 + 105 * (1 - intensity)),
                                  int(150 + 105 * (1 - intensity)))
                p.fillRect(QRectF(x, y, cell_size, cell_size), color)
                p.setPen(QPen(text_col, 1))
                p.drawRect(QRectF(x, y, cell_size, cell_size))
                p.setPen(text_col)
                p.setFont(QFont("Arial", max(7, int(cell_size / 5))))
                p.drawText(QRectF(x, y, cell_size, cell_size),
                          Qt.AlignmentFlag.AlignCenter, str(int(val)))
        p.setPen(text_col)
        p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        for i in range(n):
            name = self.class_names[i] if i < len(self.class_names) else f"C{i}"
            p.drawText(QRectF(0, margin + i * cell_size, margin - 5, cell_size),
                      Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, name)
            p.save()
            p.translate(margin + i * cell_size + cell_size/2, margin + plot_size + 5)
            p.rotate(45)
            p.drawText(QRectF(-50, 0, 100, 20), Qt.AlignmentFlag.AlignCenter, name)
            p.restore()
        p.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        p.drawText(QRectF(margin, h - 25, plot_size, 20),
                  Qt.AlignmentFlag.AlignCenter, "Predicted →")
        p.save()
        p.translate(20, margin + plot_size / 2)
        p.rotate(-90)
        p.drawText(QRectF(-50, -10, 100, 20), Qt.AlignmentFlag.AlignCenter, "← Actual")
        p.restore()
        p.end()

class ParetoFrontWidget(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.fronts = []
        self.objectives = None
        self.setMinimumHeight(250)

    def set_pareto_data(self, objectives, fronts):
        self.objectives = objectives
        self.fronts = fronts
        self.update()

    def clear(self):
        self.objectives = None
        self.fronts = []
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        bg = self.palette().color(QPalette.ColorRole.Window)
        p.fillRect(0, 0, w, h, bg)
        text_col = self.palette().color(QPalette.ColorRole.Text)
        if self.objectives is None or len(self.fronts) == 0:
            p.setPen(text_col)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                      "Pareto Front (NSGA-II)\nRequires Multi-objective fitness")
            p.end(); return
        m = 45
        pw, ph = w - 2*m, h - 2*m
        n_obj = self.objectives.shape[1]
        if n_obj < 2:
            p.setPen(text_col)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                      "Need ≥2 objectives for Pareto visualization")
            p.end(); return
        x_data = self.objectives[:, 0]
        y_data = self.objectives[:, 1]
        xmn, xmx = x_data.min(), x_data.max()
        ymn, ymx = y_data.min(), y_data.max()
        xr = max(xmx - xmn, 1e-12)
        yr = max(ymx - ymn, 1e-12)
        grid_col = self.palette().color(QPalette.ColorRole.Mid)
        p.setPen(QPen(grid_col, 1))
        for i in range(1, 5):
            y = m + ph * i / 5
            p.drawLine(m, int(y), m + pw, int(y))
        p.setPen(QPen(text_col, 1))
        p.drawLine(m, m, m, m+ph)
        p.drawLine(m, m+ph, m+pw, m+ph)
        colors = [QColor(255, 50, 50), QColor(50, 200, 50), QColor(50, 100, 255),
                  QColor(255, 165, 0), QColor(128, 0, 128)]
        for fi, front in enumerate(self.fronts[:5]):
            col = colors[fi % len(colors)]
            for idx in front:
                x = m + (x_data[idx] - xmn) / xr * pw
                y = m + (1 - (y_data[idx] - ymn) / yr) * ph
                p.setBrush(col)
                p.setPen(QPen(col.darker(), 1))
                size = 6 if fi == 0 else 4
                p.drawEllipse(QPointF(x, y), size, size)
            if fi == 0 and len(front) > 1:
                sorted_front = sorted(front, key=lambda i: x_data[i])
                p.setPen(QPen(col, 1, Qt.PenStyle.DashLine))
                prev = None
                for idx in sorted_front:
                    x = m + (x_data[idx] - xmn) / xr * pw
                    y = m + (1 - (y_data[idx] - ymn) / yr) * ph
                    if prev:
                        p.drawLine(QPointF(prev[0], prev[1]), QPointF(x, y))
                    prev = (x, y)
        p.setPen(text_col)
        p.setFont(QFont("Arial", 9))
        p.drawText(m + pw//2 - 30, h - 10, "Objective 1 (Loss)")
        p.save()
        p.translate(15, m + ph//2 + 30)
        p.rotate(-90)
        p.drawText(0, 0, "Objective 2 (Complexity)")
        p.restore()
        lx, ly = m + pw - 100, m + 5
        for fi in range(min(len(self.fronts), 5)):
            col = colors[fi % len(colors)]
            p.setBrush(col)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(lx, ly + fi * 15), 4, 4)
            p.setPen(text_col)
            p.setFont(QFont("Arial", 8))
            p.drawText(lx + 10, ly + fi * 15 + 4, f"Front {fi+1} ({len(self.fronts[fi])})")
        p.end()

class OperatorStatsWidget(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self.stats = {}
        self.setMinimumHeight(200)
        l = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Operator", "Q-Value", "Probability", "Avg Reward"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        l.addWidget(self.table)
        self.info = QLabel("Adaptive Operator Selection (AOS) statistics")
        self.info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info.setStyleSheet("color:#565f89;padding:5px;")
        l.addWidget(self.info)

    def update_stats(self, stats):
        self.stats = stats
        self.table.setRowCount(0)
        for op, data in sorted(stats.items(), key=lambda x: -x[1]['p']):
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(op))
            self.table.setItem(r, 1, QTableWidgetItem(f"{data['q']:.4f}"))
            self.table.setItem(r, 2, QTableWidgetItem(f"{data['p']*100:.1f}%"))
            self.table.setItem(r, 3, QTableWidgetItem(f"{data['avg_reward']:.6f}"))

    def clear(self):
        self.stats = {}
        self.table.setRowCount(0)

class PredictionViewer(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        self._state = None
        self._cn = None
        self._rng = np.random.default_rng()
        l = QVBoxLayout(self)
        l.setContentsMargins(4, 4, 4, 4)
        self.banner = QLabel("🧪 Prediction Demo\nAfter evolution, predictions appear here.")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("font-weight:bold;font-size:13px;padding:10px;"
                                  "background:#242838;border-radius:5px;color:#7aa2f7;")
        l.addWidget(self.banner)
        self.summary = QLabel("")
        self.summary.setStyleSheet("font-size:14px;padding:6px;")
        l.addWidget(self.summary)
        br = QHBoxLayout()
        self.nb = QPushButton("🎲 New"); self.nb.setEnabled(False)
        self.nb.clicked.connect(self._rn)
        self.sc = QPushButton("↕ Sort Conf"); self.sc.setEnabled(False)
        self.sc.clicked.connect(self._sc)
        self.scc = QPushButton("✓ Sort Corr"); self.scc.setEnabled(False)
        self.scc.clicked.connect(self._scc)
        self.ensemble_btn = QPushButton("👥 Ensemble"); self.ensemble_btn.setEnabled(False)
        self.ensemble_btn.clicked.connect(self._ensemble_pred)
        br.addWidget(self.nb); br.addWidget(self.sc); br.addWidget(self.scc)
        br.addWidget(self.ensemble_btn); br.addStretch()
        l.addLayout(br)
        sp = QHBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["#", "Input", "Predicted", "Actual", "Conf/Err", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(250)
        sp.addWidget(self.table, 3)
        self.hl = QLabel("Confidence distribution")
        self.hl.setMinimumWidth(200)
        self.hl.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.hl.setStyleSheet("border:1px solid #2a2e42;background:#1a1b26;")
        sp.addWidget(self.hl, 1)
        l.addLayout(sp, 1)
        self._lp = None; self._lt = None; self._lc = False

    def set_class_names(self, n):
        self._cn = n

    def show_predictions(self, st, il=False):
        self._state = st
        self._lp = st['preds']; self._lt = st['targets']; self._lc = st['is_clf']
        self._d(st['preds'], st['targets'], st['inputs'], st['is_clf'],
                st['feature_shape'], il)
        self._dh()
        if 'ensemble_preds' in st:
            self.ensemble_btn.setEnabled(True)

    def _rn(self):
        if not self._state: return
        st = self._state
        n = len(st['X_test'])
        ns = min(24, n)
        idx = self._rng.choice(n, ns, replace=False)
        idx.sort()
        ps = evaluate_single(st['X_test'][idx], st['best_individual'], st['H'],
                            st['W'], st['output_dim'], st['use_gpu'])
        self._lp = ps; self._lt = st['y_test'][idx]; self._lc = st['is_clf']
        self._d(ps, st['y_test'][idx], st['X_test_orig'][idx], st['is_clf'],
                st['feature_shape'])
        self._dh()

    def _sc(self):
        if not self._state: return
        st = self._state
        p, t, i = st['preds'], st['targets'], st['inputs']
        if st['is_clf']:
            e = np.exp(p - np.max(p, axis=1, keepdims=True))
            c = e / e.sum(axis=1, keepdims=True)
            o = np.argsort(-np.max(c, axis=1))
        else:
            o = np.argsort(np.abs(p[:, 0] - t[:, 0]))
        self._d(p[o], t[o], i[o], st['is_clf'], st['feature_shape'])

    def _scc(self):
        if not self._state: return
        st = self._state
        p, t, i = st['preds'], st['targets'], st['inputs']
        if st['is_clf']:
            c = (np.argmax(p, axis=1) == np.argmax(t, axis=1)).astype(int)
            o = np.argsort(-c)
        else:
            o = np.argsort(np.abs(p[:, 0] - t[:, 0]))
        self._d(p[o], t[o], i[o], st['is_clf'], st['feature_shape'])

    def _ensemble_pred(self):
        if not self._state or 'ensemble_preds' not in self._state: return
        st = self._state
        ep = st['ensemble_preds']
        self._d(ep, st['targets'], st['inputs'], st['is_clf'], st['feature_shape'])
        self.banner.setText(self.banner.text().replace("🧪", "👥 Ensemble"))

    def _d(self, preds, targets, inputs, is_clf, fs, il=False):
        self.table.setRowCount(0)
        n = len(preds)
        nc = 0; te = 0.0
        nm = self._cn or [f'Class {i}' for i in range(preds.shape[1])] if is_clf else None
        for i in range(n):
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(str(i + 1)))
            if fs is not None:
                il2 = QLabel()
                il2.setPixmap(self._th(inputs[i], fs))
                il2.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setCellWidget(r, 1, il2)
            else:
                f = inputs[i]
                fs2 = ", ".join(f"{v:.2f}" for v in f[:5]) + (", …" if len(f) > 5 else "")
                it = QTableWidgetItem(fs2)
                it.setToolTip(str(f.tolist()))
                self.table.setItem(r, 1, it)
            if is_clf:
                pc = int(np.argmax(preds[i]))
                e = np.exp(preds[i] - np.max(preds[i]))
                cf = e / e.sum()
                ac = int(np.argmax(targets[i]))
                pn = nm[pc] if nm and pc < len(nm) else f"Class {pc}"
                an = nm[ac] if nm and ac < len(nm) else f"Class {ac}"
                self.table.setItem(r, 2, QTableWidgetItem(pn))
                self.table.setItem(r, 3, QTableWidgetItem(an))
                ci = QTableWidgetItem(f"{cf[pc]*100:.1f}%")
                c = cf[pc]
                ci.setBackground(QColor(200, 255, 200) if c > .8 else
                                QColor(255, 255, 200) if c > .5 else
                                QColor(255, 200, 200))
                self.table.setItem(r, 4, ci)
                if pc == ac:
                    nc += 1
                    it2 = QTableWidgetItem("✓"); it2.setForeground(QColor("#9ece6a"))
                else:
                    it2 = QTableWidgetItem("✗"); it2.setForeground(QColor("#f7768e"))
                self.table.setItem(r, 5, it2)
            else:
                pv = float(preds[i][0]); av = float(targets[i][0])
                er = pv - av; te += abs(er)
                self.table.setItem(r, 2, QTableWidgetItem(f"{pv:.4f}"))
                self.table.setItem(r, 3, QTableWidgetItem(f"{av:.4f}"))
                self.table.setItem(r, 4, QTableWidgetItem(f"{er:+.4f}"))
                ae = abs(er)
                it2 = QTableWidgetItem(f"{ae:.4f}")
                it2.setForeground(QColor("#9ece6a") if ae < .05 else
                                  QColor("#e0af68") if ae < .15 else
                                  QColor("#f7768e"))
                self.table.setItem(r, 5, it2)
        px = "🔴 LIVE — " if il else ""
        if is_clf:
            a = nc / n * 100
            self.banner.setText(f"{px}🧪 {n} samples | Acc: {a:.1f}%")
            self.summary.setText(f"📊 Acc: {a:.1f}% ({nc}/{n})")
        else:
            m = te / n
            self.banner.setText(f"{px}🧪 {n} samples | MAE: {m:.6f}")
            self.summary.setText(f"📊 MAE: {m:.6f}")
        self.nb.setEnabled(True); self.sc.setEnabled(True); self.scc.setEnabled(True)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def _dh(self):
        if self._lp is None or not self._lc:
            self.hl.setText("Confidence dist\n(classification only)")
            return
        e = np.exp(self._lp - np.max(self._lp, axis=1, keepdims=True))
        cfs = e / e.sum(axis=1, keepdims=True)
        mc = np.max(cfs, axis=1)
        bins = np.linspace(0, 1, 11)
        counts, _ = np.histogram(mc, bins=bins)
        mc2 = max(counts) if max(counts) > 0 else 1
        cw, ch = 200, 250
        pm = QPixmap(cw, ch); pm.fill(QColor("#1a1b26"))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor("#a9b1d6"))
        p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        p.drawText(10, 15, "Confidence Dist")
        p.setFont(QFont("Arial", 8))
        p.drawText(10, 28, f"n={len(mc)} mean={np.mean(mc)*100:.1f}%")
        ml, mb = 10, 25
        pw, ph = cw - 2*ml, ch - 60
        bw = pw / len(counts)
        p.setPen(Qt.PenStyle.NoPen)
        for i, c in enumerate(counts):
            x = ml + i * bw
            bh = (c / mc2) * ph
            r = bins[i]
            p.setBrush(QColor(122, 162, 247) if r > .8 else
                       QColor(224, 175, 104) if r > .5 else
                       QColor(247, 118, 142))
            p.drawRect(QRectF(x + 2, mb + ph - bh, bw - 4, bh))
        p.setPen(QColor("#565f89"))
        p.drawLine(ml, mb + ph, ml + pw, mb + ph)
        p.setPen(QColor("#a9b1d6")); p.setFont(QFont("Arial", 7))
        for i in range(0, 11, 2):
            p.drawText(int(ml + i*bw - 8), int(mb + ph + 12), f"{bins[i]*100:.0f}%")
        p.drawText(10, ch - 5, "Conf→")
        p.end()
        self.hl.setPixmap(pm)

    def _th(self, feats, shape, sz=48):
        h, w = shape
        n = h * w
        f = np.pad(feats, (0, max(0, n - len(feats))))[:n] if len(feats) < n else feats[:n]
        img = f.reshape(h, w)
        vmn, vmx = float(img.min()), float(img.max())
        nm = ((img - vmn) / max(vmx - vmn, 1e-8) * 255).astype(np.uint8)
        rgb = np.ascontiguousarray(np.stack([nm, nm, nm], axis=-1))
        qi = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qi.copy()).scaled(sz, sz,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

    def clear_view(self):
        self.table.setRowCount(0)
        self.summary.setText("")
        self.nb.setEnabled(False); self.sc.setEnabled(False); self.scc.setEnabled(False)
        self.ensemble_btn.setEnabled(False)
        self.banner.setText("🧪 Prediction Demo\nAfter evolution, predictions appear here.")
        self.hl.setText("Confidence dist\n(classification only)")
        self._state = None; self._lp = None

class DiagnosticsWidget(QWidget):
    def __init__(self, p=None):
        super().__init__(p)
        l = QVBoxLayout(self)
        l.setContentsMargins(6, 6, 6, 6)
        gr = QHBoxLayout()
        for title, attr, style in [
            ("GPU VRAM", "vram", "stop:0 #7aa2f7,stop:0.7 #e0af68,stop:1 #f7768e"),
            ("System RAM", "ram", "stop:0 #7aa2f7,stop:0.7 #e0af68,stop:1 #f7768e"),
            ("CPU Usage", "cpu", "stop:0 #9ece6a,stop:0.7 #e0af68,stop:1 #f7768e")]:
            gb = QGroupBox(title)
            vl = QVBoxLayout(gb)
            lbl = QLabel("N/A")
            lbl.setStyleSheet("font-family:monospace;font-size:13px;font-weight:bold;color:#c0caf5;")
            vl.addWidget(lbl)
            bar = QProgressBar(); bar.setRange(0, 100); bar.setTextVisible(False)
            bar.setStyleSheet(f"QProgressBar{{background:#1f2233;border-radius:4px;}}"
                            f"QProgressBar::chunk{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,{style});border-radius:4px;}}")
            vl.addWidget(bar)
            gr.addWidget(gb)
            setattr(self, f"{attr}_label", lbl)
            setattr(self, f"{attr}_bar", bar)
        l.addLayout(gr)
        est = QGroupBox("Estimated Peak Memory")
        el = QVBoxLayout(est)
        self.est_label = QLabel("N/A")
        self.est_label.setStyleSheet("font-family:monospace;font-size:12px;color:#c0caf5;")
        el.addWidget(self.est_label)
        self.est_bar = QProgressBar(); self.est_bar.setRange(0, 100)
        self.est_bar.setTextVisible(False)
        self.est_bar.setStyleSheet("QProgressBar{background:#1f2233;border-radius:4px;}"
                                    "QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #7aa2f7,stop:0.8 #e0af68,stop:1 #f7768e);border-radius:4px;}")
        el.addWidget(self.est_bar)
        self.est_detail = QLabel("")
        self.est_detail.setStyleSheet("font-family:monospace;font-size:10px;color:#565f89;")
        el.addWidget(self.est_detail)
        l.addWidget(est)
        ab = QGroupBox("Array Sizes (live)")
        al = QVBoxLayout(ab)
        self.at = QTableWidget(); self.at.setColumnCount(4)
        self.at.setHorizontalHeaderLabels(["Array", "Shape", "Size", "%VRAM"])
        self.at.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        al.addWidget(self.at)
        l.addWidget(ab)
        pb = QGroupBox("Mini Profiler")
        pl = QVBoxLayout(pb)
        self.pt = QTableWidget(); self.pt.setColumnCount(5)
        self.pt.setHorizontalHeaderLabels(["Op", "ms", "%", "Calls", "Bar"])
        self.pt.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        pl.addWidget(self.pt)
        self.gt = QLabel("idle")
        self.gt.setStyleSheet("font-family:monospace;font-weight:bold;padding:4px;color:#7aa2f7;")
        pl.addWidget(self.gt)
        l.addWidget(pb)
        l.addStretch()
        self._um()

    def _um(self):
        u, t = get_gpu_memory()
        if t > 0:
            p = int(u / t * 100)
            self.vram_label.setText(f"{format_bytes(u)} / {format_bytes(t)} ({p}%)")
            self.vram_bar.setValue(p)
        else:
            self.vram_label.setText("No GPU"); self.vram_bar.setValue(0)
        rss = get_process_memory()
        av, st = get_system_memory()
        if rss > 0 and st > 0:
            p = int(rss / st * 100)
            self.ram_label.setText(f"Process: {format_bytes(rss)}\nAvail: {format_bytes(av)}/{format_bytes(st)} ({p}%)")
            self.ram_bar.setValue(min(100, p))
        else:
            self.ram_label.setText("psutil N/A"); self.ram_bar.setValue(0)
        cpu = get_cpu_usage()
        if cpu >= 0:
            self.cpu_label.setText(f"{cpu:.1f}%")
            self.cpu_bar.setValue(int(cpu))
        else:
            self.cpu_label.setText("N/A"); self.cpu_bar.setValue(0)

    def update_estimate(self, me, gt):
        total = me['total']; pb = me['population']; ab = me.get('activations', 0)
        if gt > 0:
            pct = total / gt * 100; pp = pb / gt * 100
            if pb > gt:
                self.est_label.setText(f"⚠️ POP>VRAM!\n  Pop:{format_bytes(pb)}>{format_bytes(gt)}")
                col = "#f7768e"
            else:
                self.est_label.setText(f"Est:{format_bytes(total)}/{format_bytes(gt)} ({pct:.1f}%)\n  HSV:{format_bytes(pb)} ({pp:.1f}%) Acts:{format_bytes(ab)}")
                col = "#f7768e" if pct > 80 else "#e0af68" if pct > 50 else "#7aa2f7"
            self.est_label.setStyleSheet(f"font-family:monospace;font-size:12px;color:{col};font-weight:bold;")
            self.est_bar.setValue(int(min(100, pct)))
        else:
            self.est_label.setText(f"Est:{format_bytes(total)} (RAM)\n  HSV:{format_bytes(pb)}")
            self.est_bar.setValue(0)
        self.est_detail.setText(f"Pop:{format_bytes(me['population'])} Fwd:{format_bytes(me['forward'])} Acts:{format_bytes(ab)} Preds:{format_bytes(me['predictions'])}")

    def update_profiler(self, data):
        tm = data.get('timings', {})
        cl = data.get('calls', {})
        gt = data.get('gen_total', 0)
        tp = data.get('total_profiled', 0)
        oh = data.get('overhead', 0)
        asz = data.get('array_sizes', {})
        sp = data.get('shapes', {})
        self.at.setRowCount(0)
        vt = get_gpu_memory()[1]
        for nm, sz in sorted(asz.items(), key=lambda x: -x[1]):
            r = self.at.rowCount()
            self.at.insertRow(r)
            self.at.setItem(r, 0, QTableWidgetItem(nm))
            self.at.setItem(r, 1, QTableWidgetItem(str(sp.get(nm, ''))))
            self.at.setItem(r, 2, QTableWidgetItem(format_bytes(sz)))
            self.at.setItem(r, 3, QTableWidgetItem(f"{sz/vt*100:.2f}%" if vt > 0 else "N/A"))
        self.pt.setRowCount(0)
        tt = gt if gt > 0 else 1e-12
        ao = dict(tm)
        if oh > 0: ao['other'] = oh
        cs = {'evaluate': QColor(0, 102, 204), 'test_eval': QColor(204, 68, 0),
              'ga': QColor(0, 170, 0), 'pca_init': QColor(170, 0, 170),
              'selection': QColor(255, 152, 0), 'crossover': QColor(156, 39, 176),
              'mutation': QColor(0, 193, 0), 'adaptive': QColor(63, 81, 181),
              'crowding': QColor(121, 85, 72), 'fitness': QColor(255, 193, 7),
              'novelty': QColor(255, 64, 129), 'migration': QColor(0, 188, 212),
              'fitness_sharing': QColor(255, 87, 34), 'elite_mut': QColor(255, 87, 34),
              'memetic': QColor(255, 0, 255), 'speciation': QColor(0, 128, 128),
              'nsga2': QColor(128, 0, 128), 'other': QColor(150, 150, 150)}
        for nm, t in sorted(ao.items(), key=lambda x: -x[1]):
            r = self.pt.rowCount()
            self.pt.insertRow(r)
            ms = t * 1000; pct = t / tt * 100
            self.pt.setItem(r, 0, QTableWidgetItem(nm))
            self.pt.setItem(r, 1, QTableWidgetItem(f"{ms:.2f}"))
            self.pt.setItem(r, 2, QTableWidgetItem(f"{pct:.1f}%"))
            self.pt.setItem(r, 3, QTableWidgetItem(str(cl.get(nm, '-'))))
            bar = QTableWidgetItem("█" * max(1, int(pct / 3)))
            bar.setForeground(cs.get(nm, QColor(100, 100, 100)))
            self.pt.setItem(r, 4, bar)
        avg = data.get('avg_gen_time', 0)
        self.gt.setText(f"⏱ Gen:{gt*1000:.2f}ms (avg:{avg*1000:.1f}ms) | Profiled:{tp*1000:.2f}ms | Overhead:{oh*1000:.2f}ms")

# ── Network Evaluation ─────────────────────────────────────────────────────
def _pad_truncate(X, W):
    c = X.shape[1]
    if c < W:
        p = np.zeros((X.shape[0], W), dtype=np.float64)
        p[:, :c] = X
        return p
    elif c > W:
        return X[:, :W].copy()
    return X.astype(np.float64).copy()

def _act_sigmoid(x, xp): return 1.0 / (1.0 + xp.exp(-xp.clip(x, -50, 50)))
def _act_tanh(x, xp): return xp.tanh(xp.clip(x, -50, 50))
def _act_relu(x, xp): return xp.maximum(0, x)
def _act_leaky(x, xp): return xp.where(x > 0, x, 0.01 * x)
def _act_linear(x, xp): return x
def _act_gaussian(x, xp): return xp.exp(-xp.clip(x, -50, 50) ** 2)
def _act_sin(x, xp): return xp.sin(xp.clip(x, -50, 50))
def _act_step(x, xp): return xp.where(x > 0, 1.0, 0.0)
def _act_swish(x, xp): return x * (1.0 / (1.0 + xp.exp(-xp.clip(x, -50, 50))))
def _act_mish(x, xp): return x * xp.tanh(xp.log1p(xp.exp(xp.clip(x, -50, 50))))
def _act_elu(x, xp): return xp.where(x > 0, x, xp.exp(xp.clip(x, -50, 0)) - 1)
def _act_selu(x, xp):
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * xp.where(x > 0, x, alpha * (xp.exp(xp.clip(x, -50, 0)) - 1))
def _act_softplus(x, xp): return xp.log1p(xp.exp(xp.clip(x, -50, 50)))
def _act_hard_sigmoid(x, xp): return xp.clip(x * 0.2 + 0.5, 0, 1)
def _act_tanhshrink(x, xp): return x - xp.tanh(xp.clip(x, -50, 50))
def _act_softsign(x, xp): return x / (1.0 + xp.abs(x))

ACT_FNS = [_act_sigmoid, _act_tanh, _act_relu, _act_leaky, _act_linear, _act_gaussian,
           _act_sin, _act_step, _act_swish, _act_mish, _act_elu, _act_selu,
           _act_softplus, _act_hard_sigmoid, _act_tanhshrink, _act_softsign]

def _apply_acts_batch(x, hue, ug):
    xp = cp if (ug and CUPY_AVAILABLE) else np
    ai = (hue * N_ACTIVATIONS).astype(xp.int64) % N_ACTIVATIONS
    u = xp.unique(ai)
    if len(u) == 1:
        return ACT_FNS[int(u[0])](x, xp)
    out = xp.empty_like(x)
    ai_b = xp.broadcast_to(ai[:, None, :], x.shape)
    for a in range(N_ACTIVATIONS):
        mask = (ai_b == a)
        if mask.any():
            out[mask] = ACT_FNS[a](x[mask], xp)
    return out

def _apply_acts_single(x, hue, ug):
    xp = cp if (ug and CUPY_AVAILABLE) else np
    ai = (hue * N_ACTIVATIONS).astype(xp.int64) % N_ACTIVATIONS
    u = xp.unique(ai)
    if len(u) == 1:
        return ACT_FNS[int(u[0])](x, xp)
    out = xp.empty_like(x)
    for a in range(N_ACTIVATIONS):
        m = (ai == a)
        if m.any():
            out[:, m] = ACT_FNS[a](x[:, m], xp)
    return out

def _eval_chunk(X, pc, H, W, od, ug):
    xp = cp if (ug and CUPY_AVAILABLE) else np
    px = xp.asarray(np.ascontiguousarray(pc))
    P, B = px.shape[0], X.shape[0]
    x = xp.broadcast_to(xp.asarray(X), (P, B, X.shape[1]))
    for i in range(H - 1):
        hue = px[:, i, :, 0]; sat = px[:, i, :, 1]; val = px[:, i, :, 2]
        w = sat * 2 - 1; b = (val - 0.5) * 2
        x = x * w[:, None, :] + b[:, None, :]
        x = _apply_acts_batch(x, hue, ug)
    if W >= od:
        return x[:, :, :od]
    p = xp.zeros((x.shape[0], x.shape[1], od), dtype=np.float64)
    p[:, :, :W] = x
    return p

def evaluate_population(X, pop, H, W, od, ug, cs=32):
    ps = pop.shape[0]
    xp = cp if (ug and CUPY_AVAILABLE) else np
    if ps <= cs:
        pr = _eval_chunk(X, pop, H, W, od, ug)
        res = cp.asnumpy(pr) if (ug and CUPY_AVAILABLE) else pr
        del pr
        if ug and CUPY_AVAILABLE:
            try: cp.get_default_memory_pool().free_all_blocks()
            except RuntimeError: pass
        return res
    Xd = xp.asarray(X) if (ug and CUPY_AVAILABLE) else X
    out = []
    for i in range(0, ps, cs):
        pr = _eval_chunk(Xd, pop[i:i+cs], H, W, od, ug)
        out.append(cp.asnumpy(pr) if (ug and CUPY_AVAILABLE) else pr)
        if ug and CUPY_AVAILABLE:
            del pr
            try: cp.get_default_memory_pool().free_all_blocks()
            except RuntimeError: pass
    return np.concatenate(out, axis=0)

def evaluate_single(X, img, H, W, od, ug):
    xp = cp if (ug and CUPY_AVAILABLE) else np
    im = xp.asarray(np.ascontiguousarray(img))
    x = xp.asarray(X)
    for i in range(H - 1):
        hue = im[i, :, 0]; sat = im[i, :, 1]; val = im[i, :, 2]
        w = sat * 2 - 1; b = (val - 0.5) * 2
        x = x * w + b
        x = _apply_acts_single(x, hue, ug)
    if W >= od:
        pr = x[:, :od]
    else:
        pr = xp.zeros((x.shape[0], od), dtype=np.float64)
        pr[:, :W] = x
    res = cp.asnumpy(pr) if (ug and CUPY_AVAILABLE) else pr
    del pr, im, x
    if ug and CUPY_AVAILABLE:
        try: cp.get_default_memory_pool().free_all_blocks()
        except RuntimeError: pass
    return res

def _compute_accuracy(p, y):
    if p.shape[1] < 2: return 0.0
    return float(np.mean(np.argmax(p, axis=1) == np.argmax(y, axis=1)))

# ── SBX & Polynomial Mutation ──────────────────────────────────────────────
def sbx_crossover_vec(p1, p2, eta=20, rng=None):
    if rng is None: rng = np.random.default_rng()
    c1 = p1.copy(); c2 = p2.copy()
    diff = np.abs(p1 - p2)
    mask = diff > 1e-14
    x1 = np.minimum(p1, p2); x2 = np.maximum(p1, p2)
    rand = rng.random(p1.shape)
    beta = 1.0 + 2.0 * (x1 - 0.0) / np.maximum(x2 - x1, 1e-14)
    alpha = 2.0 - np.power(np.maximum(beta, 1e-14), -(eta + 1.0))
    beta_q = np.where(rand <= 1.0 / np.maximum(alpha, 1e-14),
                      np.power(np.maximum(rand * alpha, 1e-14), 1.0 / (eta + 1.0)),
                      np.power(np.maximum(1.0 / (2.0 - rand * alpha), 1e-14), 1.0 / (eta + 1.0)))
    c1_v = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
    c2_v = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))
    c1 = np.where(mask, np.clip(c1_v, 0, 1), c1)
    c2 = np.where(mask, np.clip(c2_v, 0, 1), c2)
    swap = rng.random(p1.shape) < 0.5
    final_c = np.where(swap, c2, c1)
    final_c[..., 0] = final_c[..., 0] % 1.0
    return final_c

def polynomial_mutation(ind, eta_m=20, prob=None, rng=None):
    if rng is None: rng = np.random.default_rng()
    if prob is None:
        prob = 1.0 / ind.size
    mut = ind.copy()
    mask = rng.random(ind.shape) < prob
    delta = rng.random(ind.shape)
    delta_q = np.where(delta < 0.5,
                       np.power(2.0 * delta, 1.0 / (eta_m + 1.0)) - 1.0,
                       1.0 - np.power(2.0 * (1.0 - delta), 1.0 / (eta_m + 1.0)))
    mut = np.where(mask, np.clip(ind + delta_q * 0.5, 0, 1), mut)
    mut[..., 0] = mut[..., 0] % 1.0
    return mut

def nelder_mead_search(best_img, X, y, H, W, od, ug, max_iter=50, rng=None):
    if rng is None: rng = np.random.default_rng()
    n_dims = min(20, best_img.size)
    current = best_img.copy()
    current_loss = float(np.mean((evaluate_single(X, current, H, W, od, ug) - y) ** 2))
    step = 0.05
    coords = rng.choice(best_img.size, n_dims, replace=False)
    for iteration in range(max_iter):
        improved = False
        for c in coords:
            ci, cj, ck = c // (W * 3), (c // 3) % W, c % 3
            for delta in [step, -step]:
                trial = current.copy()
                trial[ci, cj, ck] = np.clip(trial[ci, cj, ck] + delta, 0, 1) if ck > 0 else (trial[ci, cj, ck] + delta) % 1.0
                trial_loss = float(np.mean((evaluate_single(X, trial, H, W, od, ug) - y) ** 2))
                if trial_loss < current_loss:
                    current = trial
                    current_loss = trial_loss
                    improved = True
                    break
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break
    return current, current_loss

# ── Engine ─────────────────────────────────────────────────────────────────
class Engine(QThread):
    generation_done = Signal(int, float, float, object, float, float, float)
    image_ready = Signal(object)
    pca_ready = Signal(object, object, int)
    dataset_loaded = Signal(dict)
    finished_run = Signal()
    error_occurred = Signal(str)
    stop_reason = Signal(str)
    progress_update = Signal(int)
    stats_update = Signal(float, float)
    predictions_ready = Signal(object)
    live_predictions = Signal(object)
    profiler_update = Signal(dict)
    stopping_status = Signal(str)
    activation_stats = Signal(object)
    image_snapshot = Signal(int, float, object)
    topology_update = Signal(object)
    gene_importance_update = Signal(object)
    weight_dist_update = Signal(object)
    fit_div_update = Signal(float, float)
    neuron_contribution = Signal(object)
    population_pca = Signal(object, object)
    fitness_breakdown = Signal(str, float, float, float, float)
    state_changed = Signal(str)
    confusion_matrix_ready = Signal(object, list)
    pareto_update = Signal(object, list)
    operator_stats = Signal(dict)
    checkpoint_saved = Signal(str)
    landscape_ready = Signal(object)

    def __init__(self, p=None):
        super().__init__(p)
        self.params = {}
        self.preloaded_dataset = None
        self.loaded_image = None
        self.profiler = Profiler()
        self.aos_selector = None
        self.checkpoint_path = None
        self.resume_mode = False
        self.start_gen = 0
        self.best_individual_history = []

    def _emit_state(self, s):
        now = time.perf_counter()
        if now - getattr(self, '_last_state_t', 0) > 0.05:
            self.state_changed.emit(s)
            self._last_state_t = now

    def _build_ps(self, bi, Xp, yp, Xo, H, W, od, ug, ic, rng, top_k=5, pop=None):
        n = len(Xp)
        ns = min(24, n)
        si = sorted(rng.choice(n, ns, replace=False).tolist())
        Xs = np.ascontiguousarray(Xp[si])
        ys = np.ascontiguousarray(yp[si])
        Xso = np.ascontiguousarray(Xo[si])
        ps = evaluate_single(Xs, bi, H, W, od, ug)
        nf = Xso.shape[1]
        sq = int(np.sqrt(nf))
        fs = (sq, sq) if sq * sq == nf else None
        ensemble_preds = None
        if pop is not None and top_k > 1:
            top_imgs = pop[:top_k]
            all_preds = []
            for img in top_imgs:
                p = evaluate_single(Xs, img, H, W, od, ug)
                all_preds.append(p)
            ensemble_preds = np.mean(all_preds, axis=0)
        return {
            'preds': ps, 'targets': ys, 'inputs': Xso, 'is_clf': ic, 'feature_shape': fs,
            'best_individual': bi.copy(), 'X_test': Xp, 'y_test': yp, 'X_test_orig': Xo,
            'H': H, 'W': W, 'output_dim': od, 'use_gpu': ug,
            'ensemble_preds': ensemble_preds
        }

    def compute_landscape(self, best_img, X, y, H, W, od, use_gpu, n=15, sigma=0.3):
        try:
            rng = np.random.default_rng(0)
            d1 = rng.normal(0, 1, best_img.shape)
            d2 = rng.normal(0, 1, best_img.shape)
            d1 /= max(np.linalg.norm(d1), 1e-12)
            d2 /= max(np.linalg.norm(d2), 1e-12)
            grid = np.linspace(-sigma, sigma, n)
            perturbed = np.empty((n * n, *best_img.shape), dtype=np.float64)
            k = 0
            for a in grid:
                for b in grid:
                    perturbed[k] = np.clip(best_img + a * d1 + b * d2, 0, 1)
                    perturbed[k][..., 0] %= 1.0
                    k += 1
            preds = evaluate_population(X, perturbed, H, W, od, use_gpu and CUPY_AVAILABLE,
                                        cs=min(32, n * n))
            losses = np.mean((preds - y) ** 2, axis=(1, 2)).reshape(n, n)
            self.landscape_ready.emit(losses)
            return losses
        except Exception:
            traceback.print_exc()
            return None

    def _save_checkpoint(self, gen, best_img, best_loss, pop, cel):
        try:
            ckpt = {
                'generation': gen,
                'best_individual': best_img.tolist(),
                'best_loss': best_loss,
                'population': pop.tolist(),
                'elite_losses': cel.tolist() if cel is not None else None,
                'params': {k: v for k, v in self.params.items()
                          if isinstance(v, (int, float, str, bool, type(None)))},
                'timestamp': datetime.datetime.now().isoformat(),
            }
            fn = os.path.join(CHECKPOINT_DIR, f"ckpt_gen{gen}.json")
            with open(fn, 'w') as f:
                json.dump(ckpt, f)
            ckpts = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('ckpt_')],
                          key=lambda x: int(x.split('_gen')[1].split('.')[0]))
            for old in ckpts[:-5]:
                os.remove(os.path.join(CHECKPOINT_DIR, old))
            self.checkpoint_saved.emit(fn)
        except Exception:
            pass

    def run(self):
        # Initialize state variables at the very beginning to prevent UnboundLocalError
        cel = None
        sm = None
        il = None
        last_preds = None
        try:
            p = self.params
            H, W, ps = p["img_height"], p["img_width"], p["pop_size"]
            mr, ef, sd = p["mut_rate"], p["elite_frac"], p["seed"]
            ug, up, pc = p["use_gpu"], p["use_pca"], p["pca_components"]
            ic = p["is_classification"]
            lt, at, dm = p["loss_threshold"], p["acc_threshold"], p["delay_ms"]
            pa = p.get("pca_alpha", .01)
            bs, tf = p.get("batch_size", 256), p.get("test_frac", .2)
            patience, mi = p.get("patience", 200), p.get("min_improvement", .001)
            cw, mg = p.get("conv_window", 50), p.get("min_generations", 50)
            ts, ct = p.get("tournament_size", 3), p.get("crossover_type", "uniform")
            md = p.get("mutation_dist", "gaussian")
            sa, dr, dcr = p.get("self_adaptive_mut", True), p.get("restart_on_stagnation", True), p.get("crowding", False)
            ni, mi2 = p.get("n_islands", 1), p.get("migration_interval", 50)
            ft, aw = p.get("fitness_type", "Loss only"), p.get("adaptive_weights", True)
            l2w, l1w = p.get("l2_weight", .01), p.get("l1_weight", 0)
            gw, adw = p.get("gap_weight", .05), p.get("act_div_weight", .05)
            lf, oi, ds = p.get("log_file", ""), p.get("opposition_init", False), p.get("fitness_sharing", False)
            log_mode = p.get("log_mode", "w")
            dn = (ft == "Novelty") or p.get("novelty_search", False)
            use_memetic = p.get("use_memetic", False)
            memetic_interval = p.get("memetic_interval", 100)
            use_nsga2 = p.get("use_nsga2", False)
            use_speciation = p.get("use_speciation", False)
            speciation_threshold = p.get("speciation_threshold", 0.3)
            use_aos = p.get("use_aos", False)
            use_statistical_stop = p.get("use_statistical_stop", False)
            ckpt_interval = p.get("checkpoint_interval", 100)
            use_augmentation = p.get("use_augmentation", False)
            aug_factor = p.get("aug_factor", 2)
            class_weights = p.get("class_weights", None)

            X, y = self.preloaded_dataset
            X = X.astype(np.float64)
            y = y.astype(np.float64)

            if use_augmentation and ic:
                rng_aug = np.random.default_rng(42)
                aug_X = X + rng_aug.normal(0, 0.05, X.shape)
                X = np.vstack([X, aug_X])
                y = np.vstack([y, y])

            ns2, nf, od = X.shape[0], X.shape[1], y.shape[1]
            ht = tf > 0 and ns2 >= 10
            if ht:
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=tf, random_state=sd,
                                                       stratify=y if (ic and ns2 > 20) else None)
            else:
                Xtr, Xte, ytr, yte = X, X, y, y
            ntr, nte = Xtr.shape[0], Xte.shape[0]
            Xtro, Xteo = Xtr.copy(), Xte.copy()
            info = p.get("dataset_info", {}).copy()
            info['n_train'], info['n_test'], info['batch_size'], info['has_test_set'] = ntr, nte, min(bs, ntr), ht
            self.dataset_loaded.emit(info)

            pw2, pev = None, None
            if up and pc > 0:
                self._emit_state("pca_init")
                with self.profiler.time("pca_init"):
                    pp = PCAProcessor(n_components=pc)
                    pp.fit(Xtr)
                    Xtr = pp.transform(Xtr).astype(np.float64)
                    Xte = pp.transform(Xte).astype(np.float64)
                    pw2 = pp.get_components(); pev = pp.get_ev()
                    self.pca_ready.emit(pw2, pev, nf)

            Xtr = _pad_truncate(Xtr, W)
            Xte = _pad_truncate(Xte, W)
            if od > W:
                self.error_occurred.emit(f"Output dim {od} > W={W}. Increase Hidden Width to ≥ {od}.")
                return

            self._csv_file, self._csv_writer = None, None
            if lf:
                wh = not os.path.exists(lf) or log_mode == 'w'
                self._csv_file = open(lf, 'w' if log_mode == 'w' else 'a', newline='')
                self._csv_writer = csv.writer(self._csv_file)
                if wh:
                    self._csv_writer.writerow(['gen', 'train_loss', 'mean_train_loss',
                                               'train_acc', 'test_loss', 'test_acc',
                                               'diversity', 'gen_rate', 'act_div',
                                               'memetic_loss', 'n_species'])

            if use_aos:
                operators = ['uniform', 'arithmetic', 'blx', 'sbx', 'single_point']
                self.aos_selector = AdaptiveOperatorSelector(operators)

            rng = np.random.default_rng(sd)
            isz = ps // ni if ni > 1 else ps

            if self.resume_mode and os.path.exists(self.checkpoint_path or ""):
                try:
                    with open(self.checkpoint_path) as f:
                        ckpt = json.load(f)
                    pop = np.array(ckpt['population'])
                    best_individual = np.array(ckpt['best_individual'])
                    best_loss_ever = ckpt['best_loss']
                    self.start_gen = ckpt['generation']
                    g = self.start_gen
                except Exception:
                    self.resume_mode = False
                    g = 0
                    best_individual = None
                    best_loss_ever = float('inf')

            if not self.resume_mode:
                if self.loaded_image is not None and self.loaded_image.shape == (H, W, 3):
                    pop = np.tile(self.loaded_image, (ps, 1, 1, 1)).astype(np.float64)
                    pop += rng.normal(0, .1, pop.shape)
                else:
                    pop = rng.uniform(0, 1, (ps, H, W, 3)).astype(np.float64)
                    hue_bins = rng.integers(0, N_ACTIVATIONS, (ps, H, W))
                    hue_jitter = rng.uniform(0, 1, (ps, H, W)) / N_ACTIVATIONS
                    pop[..., 0] = (hue_bins + hue_jitter) / N_ACTIVATIONS
                pop[..., 0] = pop[..., 0] % 1
                pop[..., 1] = np.clip(pop[..., 1], 0, 1)
                pop[..., 2] = np.clip(pop[..., 2], 0, 1)
                best_individual = None
                best_loss_ever = float('inf')
                g = 0

            if oi and not self.resume_mode:
                po = 1 - pop.copy()
                po[..., 0] = po[..., 0] % 1
                ne = min(100, ntr)
                Xe, ye = Xtr[:ne], ytr[:ne]
                with self.profiler.time("opp_init"):
                    pr1 = evaluate_population(Xe, pop, H, W, od, ug and CUPY_AVAILABLE, 32)
                    pr2 = evaluate_population(Xe, po, H, W, od, ug and CUPY_AVAILABLE, 32)
                    l1 = np.mean((pr1 - ye) ** 2, axis=(1, 2))
                    l2 = np.mean((pr2 - ye) ** 2, axis=(1, 2))
                    pop[l2 < l1] = po[l2 < l1]
                del pr1, pr2, po, l1, l2
                gc.collect()

            imr = np.full(ps, mr) if sa else None
            gi = np.zeros((H, W, 3))
            asc, rc = 0, 0
            btg = float('inf')
            btg2 = 0
            self._bta_ever = 0.0
            stagnation = 0
            ecc, eac, adv = 0, 0, 0
            elh, eah = [], []
            n_species = 1

            SAFETY = 100000
            ab = min(bs, ntr)
            pX, pY, pO = (Xte if ht else Xtr, yte if ht else ytr, Xteo if ht else Xtro)

            # EXPLICIT INITIALIZATION BEFORE LOOP
            cel = None
            sm = None
            il = None
            last_preds = None

            while not self.isInterruptionRequested() and g < SAFETY:
                self.profiler.start_gen()
                if ab < ntr:
                    bi = rng.choice(ntr, ab, replace=False)
                    Xb, yb = Xtr[bi], ytr[bi]
                else:
                    Xb, yb = Xtr, ytr

                ne = max(2, int(ef * ps))
                self._emit_state("evaluate")
                with self.profiler.time("evaluate"):
                    if cel is not None and ne > 0:
                        po2 = evaluate_population(Xb, pop[ne:], H, W, od, ug and CUPY_AVAILABLE,
                                                  max(8, min(32, ps - ne)))
                        last_preds = po2
                        dl2 = po2 - yb
                        ll2 = np.mean(dl2 ** 2, axis=(1, 2))
                        if class_weights is not None and ic:
                            cw_arr = np.array(class_weights)
                            weighted_sq = (dl2 ** 2) * cw_arr[None, None, :]
                            ll2 = np.mean(weighted_sq, axis=(1, 2))
                        tl = np.empty(ps)
                        tl[:ne] = cel
                        tl[ne:] = ll2
                        del po2, dl2, ll2
                    else:
                        preds = evaluate_population(Xb, pop, H, W, od, ug and CUPY_AVAILABLE,
                                                    max(8, min(32, ps)))
                        last_preds = preds
                        d = preds - yb
                        tl = np.mean(d ** 2, axis=(1, 2))
                        if class_weights is not None and ic:
                            cw_arr = np.array(class_weights)
                            weighted_sq = (d ** 2) * cw_arr[None, None, :]
                            tl = np.mean(weighted_sq, axis=(1, 2))
                        del preds, d
                    if ug and CUPY_AVAILABLE:
                        try: cp.get_default_memory_pool().free_all_blocks()
                        except RuntimeError: pass
                
                if g % 10 == 0:
                    gc.collect()

                ft2 = ft.lower().replace(" ", "_")
                fph = "fixed"

                if use_nsga2 and ft2 in ("multi-objective", "multi"):
                    self._emit_state("nsga2")
                    with self.profiler.time("nsga2"):
                        objs = np.zeros((ps, 3))
                        objs[:, 0] = tl
                        wts = pop[..., 1] * 2 - 1
                        objs[:, 1] = np.mean(wts ** 2, axis=(1, 2, 3))
                        hf = (pop[..., 0] * N_ACTIVATIONS).astype(int) % N_ACTIVATIONS
                        one_hot = np.zeros((ps, N_ACTIVATIONS))
                        for a in range(N_ACTIVATIONS):
                            one_hot[:, a] = np.sum(hf == a, axis=(1, 2))
                        pa2 = one_hot / np.maximum(one_hot.sum(axis=1, keepdims=True), 1)
                        ent = -np.sum(pa2 * np.log(pa2 + 1e-12), axis=1) / np.log(N_ACTIVATIONS)
                        objs[:, 2] = 1.0 - ent
                        rank, crowding, fronts = nsga2_sort(objs)
                        fit = rank.astype(float) - crowding / max(ps, 1)
                        self.pareto_update.emit(objs, fronts)
                        fph = "nsga2"
                    del wts, hf, one_hot, pa2, ent
                elif dn:
                    self._emit_state("fitness")
                    with self.profiler.time("fitness"):
                        nrf = min(50, ntr)
                        rX, ry = Xtr[:nrf], ytr[:nrf]
                        rp = evaluate_population(rX, pop, H, W, od, ug and CUPY_AVAILABLE,
                                                  max(8, min(32, ps)))
                        bh = rp.reshape(ps, -1)
                        kn = min(15, ps - 1)
                        diff = bh[:, None, :] - bh[None, :, :]
                        ds2 = np.sqrt(np.sum(diff ** 2, axis=-1))
                        np.fill_diagonal(ds2, np.inf)
                        nov = np.mean(np.sort(ds2, axis=1)[:, :kn], axis=1)
                        fit = -0.7 * nov + 0.3 * tl
                        del rp, bh, diff, ds2, nov
                    fph = "novelty"
                elif ft2 == "rank":
                    self._emit_state("fitness")
                    with self.profiler.time("fitness"):
                        rk = np.argsort(np.argsort(tl)).astype(float)
                        tmp = max(ps * 0.3, 5.0)
                        fit = -np.exp(-rk / tmp)
                    fph = "rank"
                elif ft2 in ("multi-objective", "multi"):
                    self._emit_state("fitness")
                    with self.profiler.time("fitness"):
                        fit = tl.copy()
                        if aw:
                            pr = min(1.0, g / 500.0)
                            if pr < 0.25:
                                fph = "explore"
                                wl, wl2, wl1, wg2, wa = 0.5, 0, 0, 0, 0.15
                            elif pr < 0.75:
                                fph = "exploit"
                                wl, wl2, wl1, wg2, wa = 0.8, l2w, l1w, gw, 0.05
                            else:
                                fph = "refine"
                                wl, wl2, wl1, wg2, wa = 0.9, l2w * 1.5, l1w, gw * 2, 0
                        else:
                            fph = "fixed"
                            wl, wl2, wl1, wg2, wa = 1.0, l2w, l1w, gw, adw
                        fit = wl * fit
                        l2v = l1v = av2 = 0.0
                        if wl2 > 0:
                            wts = pop[..., 1] * 2 - 1
                            l2a = np.mean(wts ** 2, axis=(1, 2, 3))
                            fit += wl2 * l2a
                            l2v = float(np.mean(l2a))
                            del wts, l2a
                        if wl1 > 0:
                            wts = pop[..., 1] * 2 - 1
                            l1a = np.mean(np.abs(wts), axis=(1, 2, 3))
                            fit += wl1 * l1a
                            l1v = float(np.mean(l1a))
                            del wts, l1a
                        if wg2 > 0:
                            ml2 = float(np.mean(tl))
                            sl = float(np.std(tl)) + 1e-8
                            z = (tl - ml2) / sl
                            ov = np.where(z < -1.5, np.abs(z) - 1.5, 0.0)
                            fit += wg2 * ov
                            del z, ov
                        if wa > 0:
                            hf = (pop[..., 0] * N_ACTIVATIONS).astype(int) % N_ACTIVATIONS
                            one_hot = np.zeros((ps, N_ACTIVATIONS), dtype=np.float64)
                            for a in range(N_ACTIVATIONS):
                                one_hot[:, a] = np.sum(hf == a, axis=(1, 2))
                            pa2 = one_hot / np.maximum(one_hot.sum(axis=1, keepdims=True), 1)
                            ent = -np.sum(pa2 * np.log(pa2 + 1e-12), axis=1) / np.log(N_ACTIVATIONS)
                            fit -= wa * ent
                            av2 = float(wa)
                            del hf, one_hot, pa2, ent
                        if pw2 is not None and pa > 0:
                            nc = min(pw2.shape[0], H)
                            nf2 = min(pw2.shape[1], W)
                            pr2 = pop[:, :nc, :nf2, 1]
                            pw3 = pw2[:nc, :nf2]
                            dots = np.sum(pr2 * pw3[None, :, :], axis=(1, 2))
                            pn = np.sqrt(np.sum(pr2 ** 2, axis=(1, 2))) + 1e-8
                            wn = np.sqrt(np.sum(pw3 ** 2)) + 1e-8
                            fit -= pa * np.abs(dots) / (pn * wn)
                            del pr2, pw3, dots, pn, wn
                        self.fitness_breakdown.emit(fph, float(np.mean(tl)), l2v, l1v, av2)
                else:
                    fit = tl.copy()
                    if pw2 is not None and pa > 0:
                        nc = min(pw2.shape[0], H)
                        nf2 = min(pw2.shape[1], W)
                        pr2 = pop[:, :nc, :nf2, 1]
                        pw3 = pw2[:nc, :nf2]
                        dots = np.sum(pr2 * pw3[None, :, :], axis=(1, 2))
                        pn = np.sqrt(np.sum(pr2 ** 2, axis=(1, 2))) + 1e-8
                        wn = np.sqrt(np.sum(pw3 ** 2)) + 1e-8
                        fit -= pa * np.abs(dots) / (pn * wn)
                        del pr2, pw3, dots, pn, wn

                if ds:
                    with self.profiler.time("fitness_sharing"):
                        fp = pop.reshape(ps, -1)
                        ss = np.std(fp) * 0.5 + 1e-8
                        if ps <= 100:
                            diff = fp[:, None, :] - fp[None, :, :]
                            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
                            sh = np.where(dists < ss, 1 - dists / ss, 0)
                            np.fill_diagonal(sh, 0)
                            fit = fit / (1 + np.sum(sh, axis=1))
                            del diff, dists, sh
                        else:
                            nsh = 50
                            sidx = rng.choice(ps, nsh, replace=False)
                            fp_s = fp[sidx]
                            diff = fp[:, None, :] - fp_s[None, :, :]
                            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
                            sh = np.where(dists < ss, 1 - dists / ss, 0)
                            fit = fit / (1 + np.sum(sh, axis=1))
                            del fp_s, diff, dists, sh
                        del fp

                if use_speciation and g % 20 == 0:
                    with self.profiler.time("speciation"):
                        species_list = speciate(pop, speciation_threshold)
                        n_species = len(species_list)

                order = np.argsort(fit)
                bi2 = int(order[0])
                btl = float(tl[bi2])
                if il is None: il = btl
                if btl < best_loss_ever - 1e-10:
                    best_loss_ever = btl
                    stagnation = 0
                else:
                    stagnation += 1
                cel = tl[order[:ne]].copy()

                if use_memetic and g > 0 and g % memetic_interval == 0 and best_individual is not None:
                    self._emit_state("memetic")
                    with self.profiler.time("memetic"):
                        ne2 = min(50, nte if ht else ntr)
                        Xe2 = (Xte if ht else Xtr)[:ne2]
                        ye2 = (yte if ht else ytr)[:ne2]
                        improved_img, improved_loss = nelder_mead_search(
                            pop[bi2], Xe2, ye2, H, W, od, ug and CUPY_AVAILABLE,
                            max_iter=30, rng=rng)
                        if improved_loss < btl:
                            pop[bi2] = improved_img
                            tl[bi2] = improved_loss
                            btl = improved_loss

                if ic:
                    if last_preds is not None and bi2 < last_preds.shape[0] and last_preds.shape[0] == ps:
                        ta = _compute_accuracy(last_preds[bi2], yb)
                    else:
                        bp = evaluate_single(Xb, pop[bi2], H, W, od, ug and CUPY_AVAILABLE)
                        ta = _compute_accuracy(bp, yb)
                        del bp
                else:
                    ta = 0.0

                self._emit_state("test_eval")
                with self.profiler.time("test_eval"):
                    if ht:
                        tp = evaluate_single(Xte, pop[bi2], H, W, od, ug and CUPY_AVAILABLE)
                        tl2 = float(np.mean((tp - yte) ** 2))
                        tacc = _compute_accuracy(tp, yte) if ic else 0.0
                        if ic:
                            pred_labels = np.argmax(tp, axis=1)
                            true_labels = np.argmax(yte, axis=1)
                            cm = confusion_matrix(true_labels, pred_labels, labels=list(range(od)))
                            cn = get_class_names(info.get('name', ''), od)
                            self.confusion_matrix_ready.emit(cm, cn)
                        del tp
                    else:
                        tl2 = btl
                        tacc = ta
                    if tl2 < btg:
                        btg = tl2
                        best_individual = pop[bi2].copy()
                        btg2 = g + 1
                        asc += 1
                        self._bta_ever = max(self._bta_ever, tacc) if ic else 0
                        if asc % 10 == 0:
                            try: np.save(os.path.join(AUTOSAVE_DIR, "best_autosave.npy"), best_individual)
                            except Exception: pass
                        self.weight_dist_update.emit(best_individual.copy())
                        self.best_individual_history.append((g, best_individual.copy(), btg))
                        if len(self.best_individual_history) > 20:
                            self.best_individual_history = self.best_individual_history[-20:]

                self.image_ready.emit(pop[bi2].copy())
                self.generation_done.emit(g + 1, btl, float(np.mean(tl)), pw2, ta, tl2, tacc)
                cl = tl2 if ht else btl
                lp = (1 - (cl - lt) / (il - lt)) * 100 if il > lt else (100 if cl <= lt else 0)
                ap = (tacc / max(at, .01)) * 100 if ic else 0
                self.progress_update.emit(int(max(0, min(99, max(lp, ap)))))
                el = max(time.time() - self._t0 if hasattr(self, '_t0') else 0, 1e-6)
                self.stats_update.emit(float(np.std(pop)), (g + 1) / el)
                self.fit_div_update.emit(float(np.std(pop)), tl2 if ht else btl)

                if g % 10 == 0 or g == 0:
                    hf = pop[..., 0].flatten()
                    ac3 = np.bincount((hf * N_ACTIVATIONS).astype(int) % N_ACTIVATIONS, minlength=N_ACTIVATIONS)
                    self.activation_stats.emit(ac3)
                    adv = int(np.sum(ac3 > 0))
                    if best_individual is not None:
                        self.topology_update.emit(best_individual.copy())
                if g % 50 == 0 and best_individual is not None:
                    self.image_snapshot.emit(g, btg, best_individual.copy())
                if g % 50 == 0 and g > 0:
                    with self.profiler.time("pop_pca"):
                        npc = min(ps, 80)
                        sidx = rng.choice(ps, npc, replace=False)
                        fl = pop[sidx].reshape(npc, -1)
                        p2 = SklearnPCA(n_components=2)
                        co = p2.fit_transform(fl)
                        self.population_pca.emit(co.astype(np.float64), tl[sidx].astype(np.float64))
                        del fl, p2, co
                if g > 0 and g % 100 == 0 and best_individual is not None:
                    with self.profiler.time("neuron_contrib"):
                        ne2 = min(50, nte if ht else ntr)
                        Xe2 = (Xte if ht else Xtr)[:ne2]
                        ye2 = (yte if ht else ytr)[:ne2]
                        op2 = evaluate_single(Xe2, best_individual, H, W, od, ug and CUPY_AVAILABLE)
                        ol = np.mean((op2 - ye2) ** 2)
                        tn = (H - 1) * W
                        ns3 = min(tn, 200)
                        sm2 = rng.choice(tn, ns3, replace=False)
                        contrib = np.zeros((H - 1, W))
                        for idx2 in sm2:
                            i2, j2 = idx2 // W, idx2 % W
                            md = best_individual.copy()
                            md[i2, j2, 1] = 0.5
                            mp = evaluate_single(Xe2, md, H, W, od, ug and CUPY_AVAILABLE)
                            ml = np.mean((mp - ye2) ** 2)
                            contrib[i2, j2] = ol - ml
                        self.neuron_contribution.emit(contrib)
                        del op2, md, mp, contrib

                if use_aos and self.aos_selector and g % 50 == 0:
                    self.operator_stats.emit(self.aos_selector.get_stats())

                cur_loss = tl2 if ht else btl
                cur_acc = tacc if ht else ta
                ebg = g
                if cur_loss < btg: ebg = g
                elh.append(cur_loss)
                if len(elh) > cw + 10: elh = elh[-(cw + 10):]
                if ic:
                    eah.append(cur_acc)
                    if len(eah) > cw + 10: eah = eah[-(cw + 10):]
                pl = patience - (g - ebg)
                cs2 = "improving"
                csl = 0.0
                if len(elh) >= cw:
                    h2 = cw // 2
                    oa = np.mean(elh[-cw:-h2])
                    na = np.mean(elh[-h2:])
                    ri = (oa - na) / max(oa, 1e-12) if oa > 1e-12 else 0
                    csl = (na - oa) / max(oa, 1e-12)
                    if ri < mi:
                        ecc += 1
                        cs2 = f"plateau({ecc})"
                    else:
                        ecc = 0
                        cs2 = "improving"
                    if ic and len(eah) >= cw:
                        oa2 = np.mean(eah[-cw:-h2])
                        na2 = np.mean(eah[-h2:])
                        if na2 - oa2 < 0.001: eac += 1
                        else: eac = 0
                else:
                    cs2 = f"warmup({len(elh)}/{cw})"

                stat_stop = False
                if use_statistical_stop and len(elh) >= 30:
                    z, p_val = mann_kendall_trend(elh[-30:])
                    if p_val > 0.05 and g > mg:
                        stat_stop = True
                        cs2 = f"stat-conv(p={p_val:.3f})"

                eta = ""
                if len(elh) >= 20:
                    r2 = elh[-20:]
                    ir = max(1e-15, (r2[0] - r2[-1]) / 20)
                    if cur_loss > lt and ir > 1e-10:
                        rg = int((cur_loss - lt) / ir)
                        gr = (g + 1) / max(el, 1e-6)
                        es = rg / max(gr, .01)
                        if es < 60: eta = f" | ETA:~{int(es)}s"
                        elif es < 3600: eta = f" | ETA:~{es/60:.0f}m"
                        else: eta = f" | ETA:~{es/3600:.1f}h"
                    elif ir <= 1e-10: eta = " | ETA:converging"

                self.stopping_status.emit(
                    f"Best:TeLoss={btg:.6f}(gen {btg2})"
                    + (f" TeAcc={self._bta_ever*100:.1f}%" if ic else "")
                    + f" | Patience:{max(0, pl)}/{patience} | {cs2}"
                    + (f" | AccConv:{eac}" if ic else "")
                    + f" | ActDiv:{adv}/{N_ACTIVATIONS} | Fit:{fph}"
                    + (f" | Species:{n_species}" if use_speciation else "")
                    + eta
                )

                if g >= mg:
                    if ht and tl2 <= lt:
                        sm = f"✅ Loss:{tl2:.6f}≤{lt}"; break
                    if ic and (tacc if ht else ta) >= at:
                        sm = f"✅ Acc:{(tacc if ht else ta)*100:.2f}%≥{at*100:.2f}%"; break
                    if g - ebg >= patience:
                        sm = f"⏹️ Early stop:no improvement for {patience} gens"; break
                    if len(elh) >= cw and ecc >= 3:
                        sm = f"📉 Converged:improvement<{mi*100:.2f}%"; break
                    if ic and eac >= 5:
                        sm = f"📉 Acc plateau:<0.1% for {eac} windows"; break
                    if stat_stop:
                        sm = f"📊 Statistical convergence (Mann-Kendall p>0.05)"; break

                if g > 0 and g % ckpt_interval == 0 and best_individual is not None:
                    self._save_checkpoint(g, best_individual, btg, pop, cel)

                self._emit_state("ga")
                with self.profiler.time("ga"):
                    children = np.empty_like(pop)
                    children[:ne] = pop[order[:ne]]
                    nmr = np.empty(ps) if sa else None
                    if sa: nmr[:ne] = imr[order[:ne]]

                    if dr and stagnation > patience * 2:
                        rc += 1
                        bi3 = pop[order[0]].copy()
                        pop[:] = bi3[None, :, :, :]
                        ds2 = max(0.05, 0.4 - 0.05 * rc)
                        pop[..., 1] += rng.normal(0, ds2, pop[..., 1].shape)
                        pop[..., 2] += rng.normal(0, ds2, pop[..., 2].shape)
                        rf = max(0.1, 0.3 - 0.05 * rc)
                        arm = rng.random((ps, H, W)) < rf
                        pop[..., 0][arm] = rng.uniform(0, 1, int(arm.sum()))
                        pop[..., 0] = pop[..., 0] % 1
                        pop[..., 1] = np.clip(pop[..., 1], 0, 1)
                        pop[..., 2] = np.clip(pop[..., 2], 0, 1)
                        if sa: imr[:] = mr * 2
                        stagnation = 0
                        children[:] = pop
                        elh, ecc, ebg, gi = [], 0, g, np.zeros((H, W, 3))
                    else:
                        with self.profiler.time("elite_mut"):
                            er = mr * 0.1
                            for ei in range(ne):
                                em = rng.random((H, W, 3)) < er
                                en = int(np.count_nonzero(em))
                                if en > 0:
                                    children[ei][em] += rng.normal(0, 0.02, en)
                                    children[ei][..., 0] = children[ei][..., 0] % 1
                                    children[ei][..., 1] = np.clip(children[ei][..., 1], 0, 1)
                                    children[ei][..., 2] = np.clip(children[ei][..., 2], 0, 1)

                        no = ps - ne
                        ta2 = min(ts, ps)
                        self._emit_state("selection")
                        with self.profiler.time("selection"):
                            ca = rng.integers(0, ps, (no, ta2))
                            cb = rng.integers(0, ps, (no, ta2))
                            pa = ca[np.arange(no), np.argmin(fit[ca], axis=1)]
                            pb = cb[np.arange(no), np.argmin(fit[cb], axis=1)]

                        self._emit_state("crossover")
                        self._emit_state("mutation")
                        for idx in range(no):
                            pa2, pb2 = pop[pa[idx]], pop[pb[idx]]
                            if use_aos and self.aos_selector:
                                op_idx = self.aos_selector.select()
                                ct_actual = self.aos_selector.operators[op_idx]
                            else:
                                ct_actual = ct
                            with self.profiler.time("crossover"):
                                if p.get("multi_parent", True) and rng.random() < 0.2:
                                    c3 = rng.integers(0, ps, ta2)
                                    pc = pop[c3[np.argmin(fit[c3])]]
                                    w1, w2, w3 = rng.dirichlet([1.0, 1.0, 1.0])
                                    child = w1 * pa2 + w2 * pb2 + w3 * pc
                                elif ct_actual == "arithmetic":
                                    al = rng.uniform(0.3, 0.7)
                                    child = al * pa2 + (1 - al) * pb2
                                elif ct_actual in ("blx-α", "blx-a", "blx"):
                                    ab2 = 0.5
                                    cmn = np.minimum(pa2, pb2)
                                    cmx = np.maximum(pa2, pb2)
                                    df = cmx - cmn
                                    child = (cmn - df * ab2) + rng.random((H, W, 3)) * df * (1 + 2 * ab2)
                                elif ct_actual == "sbx":
                                    child = sbx_crossover_vec(pa2, pb2, eta=20, rng=rng)
                                elif ct_actual == "single-point":
                                    pt = rng.integers(1, H * W * 3)
                                    child = np.concatenate([pa2.flatten()[:pt], pb2.flatten()[pt:]]).reshape(H, W, 3)
                                else:
                                    mask = rng.random((H, W, 3)) < 0.5
                                    child = np.where(mask, pa2, pb2)

                            with self.profiler.time("adaptive"):
                                if sa:
                                    cr = (imr[pa[idx]] + imr[pb[idx]]) / 2.0
                                    if rng.random() < 0.1: cr *= rng.uniform(0.8, 1.2)
                                    cr = max(0.001, min(1.0, cr))
                                    nmr[ne + idx] = cr
                                else:
                                    cr = mr
                                er2 = cr * (1.0 - min(1.0, g / 2000.0))
                                if stagnation > 30: er2 = min(1.0, er2 * 3.0)

                            with self.profiler.time("mutation"):
                                if md == "polynomial":
                                    child = polynomial_mutation(child, eta_m=20,
                                                               prob=er2 / max(H * W * 3, 1),
                                                               rng=rng)
                                else:
                                    am = rng.random((H, W)) < (er2 * 0.05)
                                    nam = int(np.count_nonzero(am))
                                    if nam > 0:
                                        child[..., 0][am] = rng.uniform(0, 1, nam)
                                    wm = rng.random((H, W)) < er2
                                    nwm = int(np.count_nonzero(wm))
                                    if nwm > 0:
                                        wsc = 0.15 if stagnation > 30 else 0.08
                                        if md == "cauchy":
                                            child[..., 1][wm] += rng.standard_cauchy(nwm) * wsc
                                        else:
                                            child[..., 1][wm] += rng.normal(0, wsc, nwm)
                                        child[..., 2][wm] += rng.normal(0, wsc * 0.5, nwm)
                                child[..., 0] = child[..., 0] % 1
                                child[..., 1] = np.clip(child[..., 1], 0, 1)
                                child[..., 2] = np.clip(child[..., 2], 0, 1)

                            if use_aos and self.aos_selector:
                                child_loss = tl[pa[idx]]
                                parent_loss = min(tl[pa[idx]], tl[pb[idx]])
                                reward = max(0, parent_loss - child_loss)
                                self.aos_selector.update(op_idx, reward)

                            with self.profiler.time("crowding"):
                                ai2 = ne + idx
                                if dcr and ai2 > ne:
                                    fc = child.flatten()
                                    fne = children[ne:ai2].reshape(ai2 - ne, -1)
                                    if len(fne) > 0:
                                        ds4 = np.sum((fne - fc[None, :]) ** 2, axis=1)
                                        ms = int(np.argmin(ds4)) + ne
                                        if ms < len(fit) and fit[ai2] < fit[ms]:
                                            children[ms] = child
                                        else:
                                            children[ai2] = child
                                    else:
                                        children[ai2] = child
                                else:
                                    children[ai2] = child

                    if sa: imr = nmr
                    if ni > 1 and g > 0 and g % mi2 == 0:
                        with self.profiler.time("migration"):
                            for i in range(ni):
                                ni2 = (i + 1) % ni
                                si2 = i * isz
                                ei2 = min(si2 + isz, ps)
                                sn2 = ni2 * isz
                                en2 = min(sn2 + isz, ps)
                                mi3 = (order >= si2) & (order < ei2)
                                if not mi3.any(): continue
                                bg = order[mi3][0]
                                wg = en2 - 1
                                children[wg] = children[bg].copy()
                                if sa and nmr is not None: nmr[wg] = nmr[bg]

                    if g > 0 and g % 100 == 0:
                        nim = max(2, ps // 10)
                        wi = order[-nim:]
                        pop2 = rng.uniform(0, 1, (nim, H, W, 3))
                        pop2[..., 0] = ((pop2[..., 0] * N_ACTIVATIONS).astype(int) +
                                       rng.uniform(0, 1, (nim, H, W)) / N_ACTIVATIONS) / N_ACTIVATIONS
                        pop2[..., 0] = pop2[..., 0] % 1
                        if sa: imr[wi] = mr * 2
                        children[wi] = pop2

                    if g % 10 == 0:
                        ch = np.mean(np.abs(children - pop), axis=0)
                        gi += ch
                        self.gene_importance_update.emit(gi.copy())
                        del ch

                    if self._csv_writer and (g % 10 == 0 or g == 0):
                        self._csv_writer.writerow([g + 1, btl, float(np.mean(tl)), ta, tl2, tacc,
                                                  float(np.std(pop)), f"{(g+1)/el:.2f}", adv,
                                                  btl if (use_memetic and g % memetic_interval == 0) else '',
                                                  n_species])
                        self._csv_file.flush()
                    pop = children
                    del children, fit, order, tl, cel, last_preds
                    last_preds = None

                self.profiler.end_gen()
                pd2 = self.profiler.get_data()
                pd2['array_sizes'] = {
                    'Population(HSV)': pop.nbytes,
                    'Input batch': Xb.nbytes,
                    'Predictions': 0,
                    'Test set(X)': Xte.nbytes,
                    'Best indiv': best_individual.nbytes if best_individual is not None else 0,
                    'PCA weights': pw2.nbytes if pw2 is not None else 0
                }
                pd2['shapes'] = {
                    'Population': pop.shape,
                    'Input batch': Xb.shape,
                    'Predictions': (0,),
                    'Test set(X)': Xte.shape
                }
                self.profiler_update.emit(pd2)

                if g > 0 and g % 50 == 0 and best_individual is not None:
                    sorted_pop = pop[np.argsort(tl if 'tl' in dir() else np.zeros(ps))] if ps > 5 else pop
                    st = self._build_ps(best_individual, pX, pY, pO, H, W, od,
                                       ug and CUPY_AVAILABLE, ic, rng, top_k=5, pop=pop[order[:5]])
                    self.live_predictions.emit(st)
                    del st
                g += 1
                
                if ug and CUPY_AVAILABLE:
                    try: cp.get_default_memory_pool().free_all_blocks()
                    except RuntimeError: pass
                    
                remaining = max(0, dm)
                while remaining > 0 and not self.isInterruptionRequested():
                    chunk = min(50, remaining)
                    self.msleep(chunk)
                    remaining -= chunk

            if best_individual is not None:
                st = self._build_ps(best_individual, pX, pY, pO, H, W, od,
                                   ug and CUPY_AVAILABLE, ic, rng, top_k=5,
                                   pop=pop[order[:5]] if 'order' in dir() else None)
                self.predictions_ready.emit(st)
                del st
            if sm is None and g >= SAFETY: sm = f"Safety cap({SAFETY} gens)"
            self.progress_update.emit(100)
            self.stop_reason.emit(sm or "Stopped by user")
            self.finished_run.emit()
        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(f"Engine error: {e}")
            self.finished_run.emit()
        finally:
            if hasattr(self, '_csv_file') and self._csv_file:
                self._csv_file.close()
            if self.params.get("use_gpu") and CUPY_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.cuda.Stream.null.synchronize()
                except Exception: pass
            gc.collect()

# ── MainWindow with Dropdown View Manager ─────────────────────────────────
class MainWindow(QMainWindow):
    _PRESETS = [None, (16, 16), (32, 32), (64, 64), (128, 128), (256, 256),
                (512, 512), (1024, 1024), (2048, 2048), (3840, 2160), (1920, 1080),
                (256, 128), (128, 256), (64, 32), (32, 64)]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GA Image-NN Evolver — Enhanced Complete Edition v4.6")
        self.setMinimumSize(1200, 800)
        self.resize(1600, 900)
        self.engine = None
        self.current_pca_weights = None
        self.dataset_manager = DatasetManager()
        self.best_image = None
        self.generation_stats = []
        self._div, self._grate = 0.0, 0.0
        self._t0 = None
        self._btl_ever, self._bta_ever = float('inf'), 0.0
        self._li = None
        self._pti = None
        self._lpd = None
        self._dark = False
        self.thread_pool = QThreadPool.globalInstance()

        c = QWidget()
        self.setCentralWidget(c)
        root = QHBoxLayout(c)
        root.setContentsMargins(5, 5, 5, 5)
        root.setSpacing(5)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self.splitter)

        left_widget = QWidget()
        left_v_layout = QVBoxLayout(left_widget)
        left_v_layout.setContentsMargins(0,0,0,0)
        left_v_layout.setSpacing(4)

        self.config_tabs = QTabWidget()
        self.config_tabs.setTabPosition(QTabWidget.TabPosition.North)
        left_v_layout.addWidget(self.config_tabs, 1)

        l_data = self._create_config_tab("📊 Data")
        l_net = self._create_config_tab("🧠 Network")
        l_evo = self._create_config_tab("🧬 Evolution")
        l_adv = self._create_config_tab("🚀 Advanced")

        # --- DATA TAB ---
        ds = QGroupBox("Dataset")
        dl = QFormLayout(ds)
        self.dtc = QComboBox()
        self.dtc.addItems(["Built-in", "Sklearn", "CSV", "Image Dir"])
        self.dtc.currentIndexChanged.connect(self._tc)
        self.dtc.setToolTip("Select the source of the dataset.")
        dl.addRow("Type:", self.dtc)
        self.bc = QComboBox()
        self.bc.addItems(["XOR", "Sine", "Quadratic", "Classification", "High-Dim", "Spiral"])
        self.bc.currentIndexChanged.connect(self._alb)
        self.bc.setToolTip("Select a built-in synthetic dataset.")
        dl.addRow("Built-in:", self.bc)
        self.sc = QComboBox()
        self.sc.addItems(["Digits", "Iris", "Breast Cancer", "Wine", "Diabetes"])
        self.sc.currentIndexChanged.connect(self._als)
        self.sc.setToolTip("Select a standard scikit-learn dataset.")
        dl.addRow("Sklearn:", self.sc)
        self.cpe = QLineEdit(); self.cpe.setPlaceholderText("CSV...")
        self.cpe.setToolTip("Path to the CSV file.")
        self.cb = QPushButton("..."); self.cb.clicked.connect(self.bcsv)
        dl.addRow("CSV:", self._hb(self.cpe, self.cb))
        self.tcs = QSpinBox(); self.tcs.setRange(-1, 1000); self.tcs.setValue(-1)
        self.tcs.setToolTip("Column index for the target variable (-1 for last).")
        dl.addRow("Target col:", self.tcs)
        self.ide = QLineEdit(); self.ide.setPlaceholderText("Image dir...")
        self.ide.setToolTip("Path to the directory containing image classes.")
        self.ib = QPushButton("..."); self.ib.clicked.connect(self.bidir)
        dl.addRow("Images:", self._hb(self.ide, self.ib))
        self.lb = QPushButton("Load Dataset"); self.lb.clicked.connect(self.ld)
        self.lb.setStyleSheet("font-weight:bold;")
        dl.addRow("", self.lb)
        self.dil = QLabel("Auto-loading...")
        self.dil.setWordWrap(True)
        self.dil.setStyleSheet("background:#1f2233;padding:5px;color:#a9b1d6;")
        dl.addRow("", self.dil)
        l_data.addWidget(ds)
        self._stv(0)

        ab2 = QGroupBox("Acceleration")
        al = QFormLayout(ab2)
        self.gcb = QCheckBox("CuPy GPU")
        self.gcb.setChecked(CUPY_AVAILABLE)
        self.gcb.setEnabled(CUPY_AVAILABLE)
        self.gcb.setToolTip("Enable GPU acceleration using CuPy (requires Nvidia GPU).")
        al.addRow(self.gcb)
        ps = []
        if CUPY_AVAILABLE:
            try:
                n = cp.cuda.runtime.getDeviceProperties(0)['name']
                ps.append(f"GPU:{n.decode() if isinstance(n, bytes) else n}")
            except Exception:
                ps.append("GPU:Available")
        else:
            ps.append("GPU:N/A")
        ps.append("psutil:✓" if PSUTIL_AVAILABLE else "psutil:✗")
        ps.append("matplotlib:✓" if MATPLOTLIB_AVAILABLE else "matplotlib:✗")
        al.addRow("", QLabel(" | ".join(ps)))
        l_data.addWidget(ab2)

        eb = QGroupBox("Evaluation")
        el = QFormLayout(eb)
        self.tss = QDoubleSpinBox(); self.tss.setRange(0, .5)
        self.tss.setSingleStep(.05); self.tss.setValue(.20)
        self.tss.setSuffix("(0=none)")
        self.tss.setToolTip("Fraction of data to use for testing (0.0 to 0.5).")
        el.addRow("Test split:", self.tss)
        self.bss = QSpinBox(); self.bss.setRange(4, 100000)
        self.bss.setValue(256); self.bss.valueChanged.connect(self._arch)
        self.bss.setToolTip("Number of samples per batch during evaluation.")
        el.addRow("Batch:", self.bss)
        lr = QHBoxLayout()
        self.le = QLineEdit(); self.le.setPlaceholderText("log.csv")
        self.le.setToolTip("Path to save the CSV evolution log.")
        self.lb2 = QPushButton("..."); self.lb2.clicked.connect(self._bl)
        lr.addWidget(self.le); lr.addWidget(self.lb2)
        el.addRow("Log CSV:", lr)
        
        self.lmc = QComboBox()
        self.lmc.addItems(["Overwrite (w)", "Append (a)"])
        self.lmc.setToolTip("Overwrite clears the log on start. Append keeps old data.")
        el.addRow("Log Mode:", self.lmc)
        
        l_data.addWidget(eb)
        l_data.addStretch()

        # --- NETWORK TAB ---
        pb = QGroupBox("Input Projection (PCA)")
        pl = QFormLayout(pb)
        self.pcb = QCheckBox("Enable PCA"); self.pcb.setChecked(True)
        self.pcb.stateChanged.connect(self._arch)
        self.pcb.setToolTip("Reduces input dimensions to speed up training and prevent overfitting.")
        pl.addRow(self.pcb)
        self.ps = QSpinBox(); self.ps.setRange(1, 256); self.ps.setValue(8)
        self.ps.valueChanged.connect(self._arch)
        self.ps.setToolTip("Number of principal components to keep.")
        pl.addRow("Components:", self.ps)
        self.pas = QDoubleSpinBox(); self.pas.setRange(0, 1)
        self.pas.setSingleStep(.01); self.pas.setValue(.01)
        self.pas.setToolTip("Penalty weight for PCA misalignment.")
        pl.addRow("PCA α:", self.pas)
        l_net.addWidget(pb)

        ab3 = QGroupBox("Hidden Layers (H×W×3 HSV)")
        ar = QFormLayout(ab3)
        self.rc = QComboBox()
        self.rc.addItems(["Custom", "16×16", "32×32", "64×64", "128×128", "256×256",
                         "512×512", "1024×1024", "2048×2048", "3840×2160 (4K)",
                         "1920×1080", "256×128", "128×256", "64×32", "32×64"])
        self.rc.currentIndexChanged.connect(self._rp)
        self.rc.setToolTip("Preset image dimensions for the hidden layers.")
        ar.addRow("Preset:", self.rc)
        wh = QHBoxLayout()
        self.iw = QSpinBox(); self.iw.setRange(4, 3840); self.iw.setValue(16)
        self.iw.valueChanged.connect(self._ow)
        self.iw.setToolTip("Width of the HSV image (number of neurons per layer).")
        self.ih = QSpinBox(); self.ih.setRange(4, 2160); self.ih.setValue(16)
        self.ih.valueChanged.connect(self._oh)
        self.ih.setToolTip("Height of the HSV image (number of hidden layers).")
        wh.addWidget(QLabel("W:")); wh.addWidget(self.iw)
        wh.addWidget(QLabel("H:")); wh.addWidget(self.ih)
        ar.addRow("Hidden:", wh)
        self.al2 = QLabel(); self.al2.setWordWrap(True)
        self.al2.setStyleSheet("font-family:monospace;background:#1f2233;padding:6px;color:#a9b1d6;")
        ar.addRow("", self.al2)
        self.vel = QLabel("VRAM:N/A")
        self.vel.setStyleSheet("font-family:monospace;font-size:11px;color:#a9b1d6;")
        ar.addRow("", self.vel)
        self.veb = QProgressBar(); self.veb.setRange(0, 100); self.veb.setTextVisible(False)
        ar.addRow("", self.veb)
        self.abc = QCheckBox("Auto-adjust batch"); self.abc.setChecked(True)
        self.abc.stateChanged.connect(self._arch)
        self.abc.ToolTip="Automatically reduce batch size if VRAM is insufficient."
        ar.addRow("", self.abc)
        self.ss = QSpinBox(); self.ss.setRange(0, 99999); self.ss.setValue(42)
        self.ss.ToolTip="Random seed for reproducibility."
        ar.addRow("Seed:", self.ss)
        l_net.addWidget(ab3)
        l_net.addStretch()

        # --- EVOLUTION TAB ---
        fb = QGroupBox("Fitness Function")
        fl = QFormLayout(fb)
        self.fc = QComboBox()
        self.fc.addItems(["Loss only", "Multi-objective", "Rank-based", "Novelty"])
        self.fc.ToolTip="Function used to evaluate individuals."
        fl.addRow("Function:", self.fc)
        self.awc = QCheckBox("Adaptive phase weights"); self.awc.setChecked(True)
        self.awc.ToolTip="Automatically adjust weights between exploration and exploitation."
        fl.addRow(self.awc)
        self.l2s = QDoubleSpinBox(); self.l2s.setRange(0, 1)
        self.l2s.setSingleStep(.01); self.l2s.setValue(.01)
        self.l2s.ToolTip="L2 regularization penalty for large weights."
        fl.addRow("L2 reg:", self.l2s)
        self.l1s = QDoubleSpinBox(); self.l1s.setRange(0, 1)
        self.l1s.setSingleStep(.01); self.l1s.setValue(0)
        self.l1s.ToolTip="L1 regularization penalty to encourage sparse weights."
        fl.addRow("L1 sparse:", self.l1s)
        self.gps = QDoubleSpinBox(); self.gps.setRange(0, 1)
        self.gps.setSingleStep(.01); self.gps.setValue(.05)
        self.gps.ToolTip="Penalty for large gaps between train and test loss."
        fl.addRow("Overfit penalty:", self.gps)
        self.ads = QDoubleSpinBox(); self.ads.setRange(0, 1)
        self.ads.setSingleStep(.01); self.ads.setValue(.05)
        self.ads.ToolTip="Reward for using a diverse set of activation functions."
        fl.addRow("Act diversity:", self.ads)
        l_evo.addWidget(fb)

        gb = QGroupBox("Genetic Algorithm")
        gl = QFormLayout(gb)
        self.pop = QSpinBox(); self.pop.setRange(4, 500); self.pop.setValue(50)
        self.pop.valueChanged.connect(self._arch)
        self.pop.ToolTip="Number of individuals in the population."
        gl.addRow("Population:", self.pop)
        self.el = QSpinBox(); self.el.setRange(1, 50); self.el.setValue(10); self.el.setSuffix(" %")
        self.el.ToolTip="Percentage of top individuals kept unchanged."
        gl.addRow("Elitism:", self.el)
        self.mu = QDoubleSpinBox(); self.mu.setRange(0, 1)
        self.mu.setSingleStep(.01); self.mu.setValue(.10)
        self.mu.ToolTip="Probability of a gene mutating."
        gl.addRow("Mutation:", self.mu)
        self.ts2 = QSpinBox(); self.ts2.setRange(1, 10); self.ts2.setValue(3)
        self.ts2.ToolTip="Number of individuals competing in tournament selection."
        gl.addRow("Tournament:", self.ts2)
        self.cc = QComboBox()
        self.cc.addItems(["Uniform", "Arithmetic", "BLX-α", "SBX", "Single-point"])
        self.cc.ToolTip="Crossover operator for mixing parents."
        gl.addRow("Crossover:", self.cc)
        self.mdc = QComboBox()
        self.mdc.addItems(["Gaussian", "Cauchy", "Polynomial"])
        self.mdc.ToolTip="Distribution used for mutation steps."
        gl.addRow("Mut dist:", self.mdc)
        self.amc = QCheckBox("Self-adaptive mut"); self.amc.setChecked(True)
        self.amc.ToolTip="Individuals learn their own optimal mutation rate."
        gl.addRow(self.amc)
        self.rcb = QCheckBox("Restart on stagnation"); self.rcb.setChecked(True)
        self.rcb.ToolTip="Restart population if no improvement for a long time."
        gl.addRow(self.rcb)
        self.crcb = QCheckBox("Crowding"); self.crcb.setChecked(False)
        self.crcb.ToolTip="Replace similar individuals to maintain diversity."
        gl.addRow(self.crcb)
        self.shcb = QCheckBox("Fitness sharing"); self.shcb.setChecked(False)
        self.shcb.ToolTip="Reduce fitness of individuals in dense areas."
        gl.addRow(self.shcb)
        self.mpc = QCheckBox("Multi-parent crossover"); self.mpc.setChecked(True)
        self.mpc.ToolTip="Use 3 parents for crossover instead of 2."
        gl.addRow(self.mpc)
        self.isl = QSpinBox(); self.isl.setRange(1, 10); self.isl.setValue(1)
        self.isl.ToolTip="Number of isolated sub-populations."
        gl.addRow("Islands:", self.isl)
        self.mis2 = QSpinBox(); self.mis2.setRange(10, 500); self.mis2.setValue(50)
        self.mis2.ToolTip="Generations between island migrations."
        gl.addRow("Migration:", self.mis2)
        self.opc = QCheckBox("Opposition init"); self.opc.setChecked(False)
        self.opc.ToolTip="Initialize half the population with opposite values."
        gl.addRow(self.opc)
        self.dy = QSpinBox(); self.dy.setRange(0, 2000); self.dy.setValue(10); self.dy.setSuffix(" ms")
        self.dy.ToolTip="Delay between generations to keep UI responsive."
        gl.addRow("Delay:", self.dy)
        l_evo.addWidget(gb)

        sb = QGroupBox("Stopping Criteria")
        sl = QFormLayout(sb)
        self.lts = QDoubleSpinBox(); self.lts.setRange(.000001, 100)
        self.lts.setDecimals(6); self.lts.setSingleStep(.001); self.lts.setValue(.01)
        self.lts.ToolTip="Stop if test loss drops below this value."
        sl.addRow("Loss≤:", self.lts)
        self.ats = QDoubleSpinBox(); self.ats.setRange(0, 1)
        self.ats.setSingleStep(.01); self.ats.setValue(.90)
        self.ats.ToolTip="Stop if test accuracy reaches this value."
        sl.addRow("Acc≥:", self.ats)
        self.ps2 = QSpinBox(); self.ps2.setRange(10, 10000)
        self.ps2.setValue(200); self.ps2.setSuffix(" gens")
        self.ps2.ToolTip="Stop if no improvement for this many generations."
        sl.addRow("Patience:", self.ps2)
        self.mis = QDoubleSpinBox(); self.mis.setRange(0, 1)
        self.mis.setDecimals(4); self.mis.setSingleStep(.001); self.mis.setValue(.001)
        self.mis.ToolTip="Minimum percentage improvement to reset patience."
        sl.addRow("Min improve:", self.mis)
        self.cws = QSpinBox(); self.cws.setRange(10, 1000); self.cws.setValue(50)
        self.cws.ToolTip="Window size for calculating moving averages."
        sl.addRow("Conv window:", self.cws)
        self.mgs = QSpinBox(); self.mgs.setRange(0, 10000); self.mgs.setValue(50)
        self.mgs.ToolTip="Minimum generations before stopping is allowed."
        sl.addRow("Min gens:", self.mgs)
        l_evo.addWidget(sb)
        l_evo.addStretch()

        # --- ADVANCED TAB ---
        adv_gb = QGroupBox("🚀 Advanced Features")
        adv_l = QFormLayout(adv_gb)
        self.memetic_cb = QCheckBox("Memetic local search (Nelder-Mead)")
        self.memetic_cb.setChecked(False)
        self.memetic_cb.ToolTip="Apply local search to the best individual to fine-tune weights."
        adv_l.addRow(self.memetic_cb)
        self.memetic_interval_sp = QSpinBox()
        self.memetic_interval_sp.setRange(10, 1000)
        self.memetic_interval_sp.setValue(100)
        self.memetic_interval_sp.setSuffix(" gens")
        adv_l.addRow("  Memetic interval:", self.memetic_interval_sp)
        self.nsga2_cb = QCheckBox("NSGA-II (multi-obj sorting)")
        self.nsga2_cb.setChecked(False)
        self.nsga2_cb.ToolTip="Use Non-dominated Sorting for true multi-objective optimization."
        adv_l.addRow(self.nsga2_cb)
        self.speciation_cb = QCheckBox("Speciation (NEAT-style)")
        self.speciation_cb.setChecked(False)
        self.speciation_cb.ToolTip="Group similar individuals to protect innovation."
        adv_l.addRow(self.speciation_cb)
        self.spec_thresh_sp = QDoubleSpinBox()
        self.spec_thresh_sp.setRange(0.01, 1.0)
        self.spec_thresh_sp.setSingleStep(0.05)
        self.spec_thresh_sp.setValue(0.3)
        self.spec_thresh_sp.ToolTip="Distance threshold for grouping into species."
        adv_l.addRow("  Species threshold:", self.spec_thresh_sp)
        self.aos_cb = QCheckBox("Adaptive Operator Selection")
        self.aos_cb.setChecked(False)
        self.aos_cb.ToolTip="Auto-learn which crossover/mutation operators work best."
        adv_l.addRow(self.aos_cb)
        self.stat_stop_cb = QCheckBox("Statistical stopping (Mann-Kendall)")
        self.stat_stop_cb.setChecked(False)
        self.stat_stop_cb.ToolTip="Stop if trend test indicates convergence."
        adv_l.addRow(self.stat_stop_cb)
        self.aug_cb = QCheckBox("Data augmentation")
        self.aug_cb.setChecked(False)
        self.aug_cb.ToolTip="Add Gaussian noise to inputs to prevent overfitting."
        adv_l.addRow(self.aug_cb)
        self.aug_factor_sp = QSpinBox()
        self.aug_factor_sp.setRange(1, 10)
        self.aug_factor_sp.setValue(2)
        adv_l.addRow("  Aug factor:", self.aug_factor_sp)
        self.ckpt_sp = QSpinBox()
        self.ckpt_sp.setRange(10, 10000)
        self.ckpt_sp.setValue(100)
        self.ckpt_sp.setSuffix(" gens")
        self.ckpt_sp.ToolTip="Interval to save population checkpoints for resuming."
        adv_l.addRow("Checkpoint interval:", self.ckpt_sp)
        l_adv.addWidget(adv_gb)
        l_adv.addStretch()

        ctrl_frame = QFrame()
        ctrl_frame.setFrameShape(QFrame.Shape.StyledPanel)
        ctrl_layout = QVBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(5, 5, 5, 5)

        br = QHBoxLayout()
        self.sb = QPushButton("Start"); self.sb.setEnabled(False)
        self.sb.clicked.connect(self.start)
        self.sb.setStyleSheet("font-weight:bold;")
        self.stb = QPushButton("Stop"); self.stb.setEnabled(False)
        self.stb.clicked.connect(self.stop)
        self.svb = QPushButton("Save Image"); self.svb.setEnabled(False)
        self.svb.clicked.connect(self.save_image)
        self.bmb = QPushButton("🔖"); self.bmb.setEnabled(False)
        self.bmb.clicked.connect(self.qbm)
        self.bmb.setToolTip("Bookmark current best image.")
        self.prb = QPushButton("✂ Prune"); self.prb.setEnabled(False)
        self.prb.clicked.connect(self.exp_pruned)
        self.rsb = QPushButton("▶ Resume"); self.rsb.setEnabled(False)
        self.rsb.clicked.connect(self.resume_run)
        br.addWidget(self.sb); br.addWidget(self.stb); br.addWidget(self.svb)
        br.addWidget(self.bmb); br.addWidget(self.prb); br.addWidget(self.rsb)
        ctrl_layout.addLayout(br)

        mr = QHBoxLayout()
        self.smb = QPushButton("💾 Model"); self.smb.setEnabled(False)
        self.smb.clicked.connect(self.save_model)
        self.lmb = QPushButton("📁 Model"); self.lmb.clicked.connect(self.load_model)
        self.epb = QPushButton("📤 .py"); self.epb.setEnabled(False)
        self.epb.clicked.connect(self.exp_net)
        self.scfg = QPushButton("⚙ Save"); self.scfg.clicked.connect(self.save_config)
        self.lcfg = QPushButton("⚙ Load"); self.lcfg.clicked.connect(self.load_config)
        self.copy_btn = QPushButton("📋 Copy")
        self.copy_btn.setToolTip("Copy current configuration JSON to clipboard.")
        self.copy_btn.clicked.connect(self.copy_config)
        self.sdb = QPushButton("🩺 Save Diag"); self.sdb.setEnabled(False)
        self.sdb.clicked.connect(self.save_diagnostics)
        self.lddb = QPushButton("🩺 Load Diag"); self.lddb.clicked.connect(self.load_diagnostics)
        self.rst_btn = QPushButton("🔄 Defaults")
        self.rst_btn.setToolTip("Reset all settings to default values and delete auto-save.")
        self.rst_btn.clicked.connect(self.reset_defaults)
        self.db = QPushButton("🌙"); self.db.setCheckable(True)
        self.db.clicked.connect(self.tdark)
        mr.addWidget(self.smb); mr.addWidget(self.lmb); mr.addWidget(self.epb)
        mr.addWidget(self.scfg); mr.addWidget(self.lcfg); mr.addWidget(self.copy_btn)
        mr.addWidget(self.sdb); mr.addWidget(self.lddb); mr.addWidget(self.rst_btn); mr.addWidget(self.db)
        ctrl_layout.addLayout(mr)

        self.lml = QLabel("Initializing...")
        self.lml.setStyleSheet("color:#7aa2f7;font-size:11px;font-weight:bold;")
        ctrl_layout.addWidget(self.lml)

        left_v_layout.addWidget(ctrl_frame)
        self.splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_v_layout = QVBoxLayout(right_widget)
        right_v_layout.setContentsMargins(0,0,0,0)
        
        self.tabs = ViewManager()
        right_v_layout.addWidget(self.tabs)
        self.splitter.addWidget(right_widget)
        
        self.splitter.setSizes([500, 1100])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        # Instantiate all widgets (even if hidden) to prevent AttributeError
        self.arch = ArchitectureViewer()
        self.il = QLabel("Evolving Image")
        self.pg = QProgressBar()
        self.st = QLabel("Idle.Auto-loading...")
        self.dl2 = QLabel("VRAM:N/A|RAM:N/A|CPU:N/A|Profiler:idle")
        self.ss2 = QLabel("")
        self.al = ActivationLegend()
        self.ast = ActivationStats()
        self.pred = PredictionViewer()
        self.sl = QLabel("Statistics")
        self.cp = ConvergencePlot()
        self.dt = QTableWidget()
        self.dw = DiagnosticsWidget()
        self.topo = NetworkTopologyView()
        self.cmw = ConfusionMatrixWidget()
        self.fdp = FitnessDiversityPlot()
        self.llv = LossLandscapeView()
        self.wh = WeightHistogram()
        self.ncv = NeuronContributionView()
        self.ppv = PopulationPCAView()
        self.pfw = ParetoFrontWidget()
        self.osw = OperatorStatsWidget()
        self.hist = EvolutionHistory()
        self.ssh = EvolutionSlideshow()
        self.pv = PCAComponentsViewer()
        self.giv = GeneImportanceView()
        self.atl = ActivationTimeline()

        # Add only the most important to the View Manager
        self.tabs.addTab(self.arch, "Architecture")

        evo = QWidget()
        el2 = QVBoxLayout(evo)
        el2.addWidget(self.il)
        el2.addWidget(self.pg)
        el2.addWidget(self.st)
        el2.addWidget(self.dl2)
        el2.addWidget(self.ss2)
        el2.addWidget(self.al)
        el2.addWidget(self.ast)
        self.tabs.addTab(evo, "Evolution Status")

        self._pti = self.tabs.addTab(self.pred, "Predictions")

        stw = QWidget()
        stl = QVBoxLayout(stw)
        stl.addWidget(self.sl)
        self.tabs.addTab(stw, "Statistics")

        self.tabs.addSeparator()

        self.tabs.addTab(self.cp, "Convergence")

        dst = QWidget()
        dsl = QVBoxLayout(dst)
        dsl.addWidget(self.dt)
        self.tabs.addTab(dst, "Dataset Info")

        self.tabs.addTab(self.dw, "Diagnostics")

        self.tabs.addSeparator()

        self.tabs.addTab(self.topo, "Network Topology")
        self.tabs.addTab(self.cmw, "Confusion Matrix")

        self.tabs.currentChanged.connect(self._tt)

        self._arch()
        
        self._load_app_settings()
        if self.dtc.currentIndex() in [0, 1]:
            QTimer.singleShot(100, self.ld)

        self.mt = QTimer()
        self.mt.timeout.connect(self._um)
        self.mt.start(1000)

        QShortcut(QKeySequence("Escape"), self).activated.connect(self.stop)
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self._ss)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._sv)
        QShortcut(QKeySequence("Ctrl+D"), self).activated.connect(self.save_diagnostics)
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(self.resume_run)

    def _create_config_tab(self, name):
        tab = QWidget()
        v = QVBoxLayout(tab)
        v.setContentsMargins(0,0,0,0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        l = QVBoxLayout(content)
        l.setContentsMargins(2,2,2,2)
        l.setSpacing(6)
        scroll.setWidget(content)
        v.addWidget(scroll)
        self.config_tabs.addTab(tab, name)
        return l

    @staticmethod
    def _hb(*ws):
        h = QHBoxLayout()
        for w in ws: h.addWidget(w)
        return h

    def _save_app_settings(self):
        try:
            settings = self._sce()
            settings['_ui_state'] = {
                'geometry': self.saveGeometry().data().hex(),
                'window_state': self.saveState().data().hex(),
                'dark_mode': self._dark,
                'splitter_sizes': self.splitter.sizes(),
                'config_tab_index': self.config_tabs.currentIndex(),
                'viz_tab_index': self.tabs.currentIndex()
            }
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Failed to save app settings: {e}")

    def _load_app_settings(self):
        if not os.path.exists(SETTINGS_FILE):
            self._ald()
            return
        try:
            with open(SETTINGS_FILE) as f:
                settings = json.load(f)
            self.load_config_from_dict(settings)
            
            ui_state = settings.get('_ui_state', {})
            if 'geometry' in ui_state:
                self.restoreGeometry(QByteArray.fromHex(bytes(ui_state['geometry'], encoding='ascii')))
            if 'window_state' in ui_state:
                self.restoreState(QByteArray.fromHex(bytes(ui_state['window_state'], encoding='ascii')))
            if 'dark_mode' in ui_state and ui_state['dark_mode']:
                self.tdark()
            if 'splitter_sizes' in ui_state:
                self.splitter.setSizes(ui_state['splitter_sizes'])
            if 'config_tab_index' in ui_state:
                self.config_tabs.setCurrentIndex(ui_state['config_tab_index'])
            if 'viz_tab_index' in ui_state:
                self.tabs.setCurrentIndex(ui_state['viz_tab_index'])
            
            self._arch()
        except Exception as e:
            print(f"Failed to load app settings: {e}")
            self._ald()

    def reset_defaults(self):
        if os.path.exists(SETTINGS_FILE):
            try: os.remove(SETTINGS_FILE)
            except: pass
        self.dtc.setCurrentIndex(1)
        self.sc.setCurrentIndex(0)
        self.iw.setValue(16); self.ih.setValue(16)
        self.pop.setValue(50); self.el.setValue(10); self.mu.setValue(0.10)
        self.ts2.setValue(3); self.cc.setCurrentIndex(0); self.mdc.setCurrentIndex(0)
        self.amc.setChecked(True); self.rcb.setChecked(True)
        self.crcb.setChecked(False); self.shcb.setChecked(False); self.mpc.setChecked(True)
        self.isl.setValue(1); self.mis2.setValue(50); self.opc.setChecked(False)
        self.dy.setValue(10)
        self.lts.setValue(0.01); self.ats.setValue(0.90)
        self.ps2.setValue(200); self.mis.setValue(0.001); self.cws.setValue(50); self.mgs.setValue(50)
        self.tss.setValue(0.20); self.bss.setValue(256); self.ss.setValue(42)
        self.pcb.setChecked(True); self.ps.setValue(8); self.pas.setValue(0.01)
        self.fc.setCurrentIndex(1); self.awc.setChecked(True)
        self.l2s.setValue(0.01); self.l1s.setValue(0.0); self.gps.setValue(0.05); self.ads.setValue(0.05)
        self.memetic_cb.setChecked(False); self.nsga2_cb.setChecked(False)
        self.speciation_cb.setChecked(False); self.aos_cb.setChecked(False)
        self.stat_stop_cb.setChecked(False); self.aug_cb.setChecked(False)
        self.lmc.setCurrentIndex(0)
        self._arch()
        self._ald()
        QMessageBox.information(self, "Reset", "Settings reset to defaults.")

    def copy_config(self):
        try:
            cfg = self._sce()
            cb = QApplication.clipboard()
            cb.setText(json.dumps(cfg, indent=2))
            self.st.setText("📋 Config copied to clipboard!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def clear_visualizations(self):
        self.cp.clear_data()
        self.hist.clear_history()
        self.ssh.clear()
        self.ast.clear()
        self.fdp.clear()
        self.ncv.clear()
        self.ppv.clear()
        self.llv.clear()
        self.topo.clear()
        self.wh.clear()
        self.cmw.clear()
        self.pfw.clear()
        self.osw.clear()
        self.pred.clear_view()
        self.dt.setRowCount(0)
        self.sl.setText("Statistics\n\n(Cleared)")
        self.il.setText("Evolving Image\n\n(Cleared)")
        self.st.setText("Idle")
        self.ss2.setText("")
        self.dl2.setText("VRAM:N/A|RAM:N/A|CPU:N/A|Profiler:idle")
        self.generation_stats = []
        self._btl_ever = float('inf')
        self._bta_ever = 0.0
        self._lpd = None
        QMessageBox.information(self, "Clear", "Visualizations cleared.")

    def _ss(self):
        if self.sb.isEnabled(): self.start()

    def _sv(self):
        if self.svb.isEnabled(): self.save_image()
        elif self.smb.isEnabled(): self.save_model()

    def _tt(self, i):
        if i == self._pti: self.tabs.setTabText(self._pti, "Predictions")

    def _fp(self):
        if self.tabs.currentIndex() != self._pti:
            self.tabs.setTabText(self._pti, "🔴 Predictions")

    def _um(self):
        self.dw._um()
        self._uc()

    def _uc(self):
        u, t = get_gpu_memory()
        vs = f"VRAM:{format_bytes(u)}/{format_bytes(t)} ({u/t*100:.0f}%)" if t > 0 else "VRAM:N/A"
        rss = get_process_memory()
        rs = f"RAM:{format_bytes(rss)}" if rss > 0 else "RAM:N/A"
        cpu = get_cpu_usage()
        cs = f"CPU:{cpu:.0f}%" if cpu >= 0 else "CPU:N/A"
        if self._lpd and self._lpd.get('gen_total', 0) > 0:
            d = self._lpd
            gt = d['gen_total']
            ps = []
            for n in ['evaluate', 'test_eval', 'selection', 'crossover', 'mutation',
                      'ga', 'pca_init', 'fitness', 'migration', 'memetic', 'nsga2',
                      'speciation']:
                if n in d.get('timings', {}):
                    p = d['timings'][n] / gt * 100
                    sh = {'evaluate':'E','test_eval':'T','selection':'sel',
                          'crossover':'xov','mutation':'mut','ga':'GA',
                          'pca_init':'PCA','fitness':'fit','migration':'mig',
                          'memetic':'⚡','nsga2':'NSGA','speciation':'Spc'}[n]
                    ps.append(f"{sh}:{p:.0f}%")
            avg = d.get('avg_gen_time', 0)
            ps2 = f"⏱{gt*1000:.0f}ms(avg:{avg*1000:.0f}ms)({' '.join(ps[:6])})"
        else:
            ps2 = "Profiler:idle"
        self.dl2.setText(f"{vs}|{rs}|{cs}|{ps2}")

    def _rp(self, idx):
        if idx < 0 or idx >= len(self._PRESETS) or self._PRESETS[idx] is None: return
        w, h = self._PRESETS[idx]
        self.iw.blockSignals(True); self.iw.setValue(w); self.iw.blockSignals(False)
        self.ih.blockSignals(True); self.ih.setValue(h); self.ih.blockSignals(False)
        self._arch()

    def _ow(self):
        self.rc.blockSignals(True); self.rc.setCurrentIndex(0); self.rc.blockSignals(False)
        self._arch()

    def _oh(self):
        self.rc.blockSignals(True); self.rc.setCurrentIndex(0); self.rc.blockSignals(False)
        self._arch()

    def _ald(self):
        self.dtc.blockSignals(True); self.dtc.setCurrentIndex(1); self.dtc.blockSignals(False)
        self.sc.blockSignals(True); self.sc.setCurrentIndex(0); self.sc.blockSignals(False)
        self._stv(1)
        self._dls("digits")

    def _tc(self, idx):
        self._stv(idx)
        if self.dataset_manager.current_dataset is not None:
            self.dataset_manager.unload()
            self.sb.setEnabled(False)
            self.dil.setText("Unloaded...")
            self.dt.setRowCount(0)
            self.pred.clear_view()
            self._arch()
        if idx == 0: self._alb()
        elif idx == 1: self._als()

    def _stv(self, idx):
        self.bc.setVisible(idx == 0)
        self.sc.setVisible(idx == 1)
        self.cpe.setVisible(idx == 2)
        self.cb.setVisible(idx == 2)
        self.tcs.setVisible(idx == 2)
        self.ide.setVisible(idx == 3)
        self.ib.setVisible(idx == 3)

    def _alb(self):
        if self.dtc.currentIndex() != 0 or not hasattr(self, 'dil'): return
        n = self.bc.currentText().lower().split()[0]
        if n == "high-dim": n = "high_dim"
        self._dlb(n)

    def _als(self):
        if self.dtc.currentIndex() != 1 or not hasattr(self, 'dil'): return
        n = self.sc.currentText().lower().replace(" ", "_")
        self._dls(n)

    def _dlb(self, n):
        X, y = self.dataset_manager.load_builtin(n)
        self._pl(X, y)

    def _dls(self, n):
        X, y = self.dataset_manager.load_sklearn(n)
        self._pl(X, y)

    def _pl(self, X, y):
        if X is None or y is None or len(X) == 0:
            self.dil.setText("Failed.")
            self.sb.setEnabled(False)
            return
        info = self.dataset_manager.get_dataset_info()
        self._udt(info)
        self.dil.setText(f"Auto-loaded:{info['name']}\n  In:{info['n_features']}→Out:{info['n_outputs']}{'classes' if info['is_classification'] else 'outputs'}")
        self.sb.setEnabled(True)
        mp = min(X.shape[0], X.shape[1])
        self.ps.setRange(1, max(1, mp))
        if self.ps.value() > mp: self.ps.setValue(mp)
        if info.get('is_classification'):
            self.pred.set_class_names(get_class_names(info['name'], info['n_outputs']))
        self._ast()
        self._arch()
        self.sb.setFocus()

    def _ast(self):
        info = self.dataset_manager.get_dataset_info()
        if not info: return
        if info['is_classification']:
            nc = info['n_outputs']
            ch = 1. / nc
            self.ats.setValue(min(.99, round(ch + .8 * (1 - ch), 2)))
        self.lts.setValue(.01)

    def bcsv(self):
        fd = QFileDialog()
        fd.setFileMode(QFileDialog.FileMode.ExistingFile)
        fd.setNameFilter("CSV (*.csv);;All (*.*)")
        fd.setDirectory(os.getcwd())
        if fd.exec(): self.cpe.setText(fd.selectedFiles()[0])

    def bidir(self):
        fd = QFileDialog()
        fd.setFileMode(QFileDialog.FileMode.Directory)
        fd.setDirectory(os.getcwd())
        if fd.exec(): self.ide.setText(fd.selectedFiles()[0])

    def _bl(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Log",
            os.path.join(os.getcwd(), "log.csv"), "CSV (*.csv)")
        if fn: self.le.setText(fn)

    def ld(self):
        idx = self.dtc.currentIndex()
        if idx == 0:
            n = self.bc.currentText().lower().split()[0]
            if n == "high-dimensional": n = "high_dim"
            X, y = self.dataset_manager.load_builtin(n)
        elif idx == 1:
            n = self.sc.currentText().lower().replace(" ", "_")
            X, y = self.dataset_manager.load_sklearn(n)
        elif idx == 2:
            if not self.cpe.text():
                QMessageBox.warning(self, "No File", "Select CSV."); return
            X, y = self.dataset_manager.load_csv(self.cpe.text(), self.tcs.value())
        elif idx == 3:
            if not self.ide.text():
                QMessageBox.warning(self, "No Dir", "Select dir."); return
            X, y = self.dataset_manager.load_images(self.ide.text(), self.iw.value())
        else:
            return
        self._pl(X, y)

    def _udt(self, info):
        self.dt.setRowCount(0)
        for k, v in info.items():
            r = self.dt.rowCount()
            self.dt.insertRow(r)
            self.dt.setItem(r, 0, QTableWidgetItem(str(k)))
            self.dt.setItem(r, 1, QTableWidgetItem(str(v)))

    def _arch(self):
        if not hasattr(self, 'iw') or not hasattr(self, 'ih'): return
        W = self.iw.value(); H = self.ih.value()
        info = self.dataset_manager.get_dataset_info()
        al = " | ".join(f"{i}:{n}" for i, n in enumerate(ACTIVATION_NAMES))
        if not info:
            self.al2.setText(f"Hidden:{H}×{W}×3={H*W*3:,} genes\n{H-1} layers×{W} neurons\nHue=act Sat=wt Val=bias\nActs:{al}")
            self.lml.setText(f"📊 None | 🧬 {self.pop.value()} Pop | 🧠 {H}x{W} | ⚡ {'GPU' if self.gcb.isChecked() else 'CPU'}")
            return
        ni, no = info['n_features'], info['n_outputs']
        ps3 = f"PCA:{ni}→{self.ps.value()}" if self.pcb.isChecked() else f"pad:{ni}"
        self.al2.setText(f"━Input:{ni}({info['name']})\n  └→{ps3}→{W}\n━Hidden:{H}×{W}×3={H*W*3:,} genes\n  {H-1} layers×{W} neurons\n  Hue=act Sat=wt[-1,1] Val=bias[-1,1]\n━Output:{no}{'classes' if info['is_classification'] else 'outputs'}\n  └→first {min(no, W)} of {W}")
        self.al2.setStyleSheet("font-family:monospace;background:#fdd;padding:6px;" if no > W else "font-family:monospace;background:#1f2233;padding:6px;color:#a9b1d6;")
        pop = self.pop.value()
        bs = self.bss.value()
        me = estimate_peak_memory(pop, bs, H, W, no)
        gt = get_gpu_memory()[1]
        if self.abc.isChecked() and gt > 0:
            tg = gt * .7
            lo, hi = 4, bs
            while lo < hi:
                mid = (lo + hi + 1) // 2
                est = estimate_peak_memory(pop, mid, H, W, no)['total']
                if est <= tg: lo = mid
                else: hi = mid - 1
            if lo < bs:
                self.bss.blockSignals(True); self.bss.setValue(lo); self.bss.blockSignals(False)
                bs = lo
                me = estimate_peak_memory(pop, bs, H, W, no)
        pb = pop * H * W * 3 * 8
        if gt > 0 and pb > gt:
            sp2 = max(4, int(gt * .5 / (H * W * 3 * 8)))
            self.vel.setText(f"⚠️Pop:{format_bytes(pb)}>VRAM({format_bytes(gt)})!\n  Reduce pop to≤{sp2}")
            self.vel.setStyleSheet("font-family:monospace;font-size:11px;color:#f7768e;font-weight:bold;")
            self.veb.setValue(100)
        elif gt > 0:
            pct = me['total'] / gt * 100
            self.vel.setText(f"Est:{format_bytes(me['total'])}/{format_bytes(gt)}({pct:.1f}%)\n  HSV:{format_bytes(pb)}")
            self.veb.setValue(int(min(100, pct)))
            col = "#f7768e" if pct > 80 else "#e0af68" if pct > 50 else "#7aa2f7"
            self.vel.setStyleSheet(f"font-family:monospace;font-size:11px;color:{col};font-weight:bold;")
        else:
            self.vel.setText(f"Est:{format_bytes(me['total'])}(RAM)")
            self.veb.setValue(0)
        self.dw.update_estimate(me, gt)
        
        ds_name = info.get('name', 'None')
        mode = "GPU" if self.gcb.isChecked() else "CPU"
        self.lml.setText(f"📊 {ds_name} | 🧬 {pop} Pop | 🧠 {H}x{W} | ⚡ {mode}")

    def _sce(self):
        return {
            'dataset_type': self.dtc.currentText(),
            'builtin': self.bc.currentText(),
            'sklearn': self.sc.currentText(),
            'csv_path': self.cpe.text(),
            'image_dir': self.ide.text(),
            'img_width': self.iw.value(),
            'img_height': self.ih.value(),
            'pca_enabled': self.pcb.isChecked(),
            'pca_components': self.ps.value(),
            'pca_alpha': self.pas.value(),
            'pop_size': self.pop.value(),
            'elite_frac': self.el.value(),
            'mut_rate': self.mu.value(),
            'tournament_size': self.ts2.value(),
            'crossover_type': self.cc.currentText(),
            'mutation_dist': self.mdc.currentText(),
            'self_adaptive': self.amc.isChecked(),
            'restart': self.rcb.isChecked(),
            'crowding': self.crcb.isChecked(),
            'fitness_sharing': self.shcb.isChecked(),
            'multi_parent': self.mpc.isChecked(),
            'n_islands': self.isl.value(),
            'migration_interval': self.mis2.value(),
            'opposition_init': self.opc.isChecked(),
            'loss_threshold': self.lts.value(),
            'acc_threshold': self.ats.value(),
            'patience': self.ps2.value(),
            'min_improvement': self.mis.value(),
            'conv_window': self.cws.value(),
            'min_generations': self.mgs.value(),
            'test_split': self.tss.value(),
            'batch_size': self.bss.value(),
            'seed': self.ss.value(),
            'delay_ms': self.dy.value(),
            'gpu': self.gcb.isChecked(),
            'fitness_type': self.fc.currentText(),
            'adaptive_weights': self.awc.isChecked(),
            'l2_weight': self.l2s.value(),
            'l1_weight': self.l1s.value(),
            'gap_weight': self.gps.value(),
            'act_div_weight': self.ads.value(),
            'log_file': self.le.text(),
            'log_mode': 'w' if self.lmc.currentText().startswith('Overwrite') else 'a',
            'use_memetic': self.memetic_cb.isChecked(),
            'memetic_interval': self.memetic_interval_sp.value(),
            'use_nsga2': self.nsga2_cb.isChecked(),
            'use_speciation': self.speciation_cb.isChecked(),
            'speciation_threshold': self.spec_thresh_sp.value(),
            'use_aos': self.aos_cb.isChecked(),
            'use_statistical_stop': self.stat_stop_cb.isChecked(),
            'use_augmentation': self.aug_cb.isChecked(),
            'aug_factor': self.aug_factor_sp.value(),
            'checkpoint_interval': self.ckpt_sp.value(),
        }

    def save_config(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save Config",
            os.path.join(os.getcwd(), "config.json"), "JSON (*.json)")
        if not fn: return
        with open(fn, 'w') as f:
            json.dump(self._sce(), f, indent=2)
        self.st.setText(f"⚙ Saved {fn}")

    def load_config(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load Config", os.getcwd(), "JSON (*.json)")
        if not fn: return
        with open(fn) as f:
            c = json.load(f)
        self.load_config_from_dict(c)
        self._arch()
        self.st.setText(f"⚙ Loaded {fn}")

    def load_config_from_dict(self, c):
        for key, widget_attr in [
            ('dataset_type', 'dtc'), ('builtin', 'bc'), ('sklearn', 'sc'),
            ('img_width', 'iw'), ('img_height', 'ih'),
            ('pca_enabled', 'pcb'), ('pca_components', 'ps'), ('pca_alpha', 'pas'),
            ('pop_size', 'pop'), ('elite_frac', 'el'), ('mut_rate', 'mu'),
            ('tournament_size', 'ts2'), ('crossover_type', 'cc'),
            ('mutation_dist', 'mdc'), ('self_adaptive', 'amc'),
            ('restart', 'rcb'), ('crowding', 'crcb'), ('fitness_sharing', 'shcb'),
            ('multi_parent', 'mpc'), ('n_islands', 'isl'),
            ('migration_interval', 'mis2'), ('opposition_init', 'opc'),
            ('loss_threshold', 'lts'), ('acc_threshold', 'ats'),
            ('patience', 'ps2'), ('min_improvement', 'mis'),
            ('conv_window', 'cws'), ('min_generations', 'mgs'),
            ('test_split', 'tss'), ('batch_size', 'bss'),
            ('seed', 'ss'), ('delay_ms', 'dy'), ('gpu', 'gcb'),
            ('fitness_type', 'fc'), ('adaptive_weights', 'awc'),
            ('l2_weight', 'l2s'), ('l1_weight', 'l1s'),
            ('gap_weight', 'gps'), ('act_div_weight', 'ads'),
        ]:
            w = getattr(self, widget_attr, None)
            if w is None: continue
            if isinstance(w, QCheckBox):
                w.setChecked(c.get(key, False))
            elif isinstance(w, QComboBox):
                val = c.get(key, w.currentText())
                if isinstance(val, int):
                    w.setCurrentIndex(val)
                else:
                    w.setCurrentText(val)
            elif isinstance(w, QSpinBox):
                w.setValue(c.get(key, w.value()))
            elif isinstance(w, QDoubleSpinBox):
                w.setValue(c.get(key, w.value()))
            elif isinstance(w, QLineEdit):
                w.setText(c.get(key, ''))
        
        self.cpe.setText(c.get('csv_path', ''))
        self.ide.setText(c.get('image_dir', ''))
        
        log_mode_str = "Overwrite (w)" if c.get('log_mode', 'w') == 'w' else "Append (a)"
        self.lmc.setCurrentText(log_mode_str)
        
        self.memetic_cb.setChecked(c.get('use_memetic', False))
        self.memetic_interval_sp.setValue(c.get('memetic_interval', 100))
        self.nsga2_cb.setChecked(c.get('use_nsga2', False))
        self.speciation_cb.setChecked(c.get('use_speciation', False))
        self.spec_thresh_sp.setValue(c.get('speciation_threshold', 0.3))
        self.aos_cb.setChecked(c.get('use_aos', False))
        self.stat_stop_cb.setChecked(c.get('use_statistical_stop', False))
        self.aug_cb.setChecked(c.get('use_augmentation', False))
        self.aug_factor_sp.setValue(c.get('aug_factor', 2))
        self.ckpt_sp.setValue(c.get('checkpoint_interval', 100))

    def _execute_save_diagnostics(self, fn, bundle):
        with open(fn, 'w') as f:
            json.dump(bundle, f, indent=2)

    def save_diagnostics(self):
        if self.best_image is None:
            QMessageBox.warning(self, "No Image", "Evolve or load a model first.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save Diagnostics",
            os.path.join(os.getcwd(), "diagnostics.json"), "JSON (*.json)")
        if not fn: return
        try:
            H, W, _ = self.best_image.shape
            info = self.dataset_manager.get_dataset_info() or {}
            vu, vt = get_gpu_memory()
            rss = get_process_memory()
            av, st = get_system_memory()
            cpu = get_cpu_usage()
            last = self.generation_stats[-1] if self.generation_stats else {}
            best_gen = min(self.generation_stats,
                          key=lambda x: x.get('te_loss', x.get('tr_loss', float('inf')))) \
                          if self.generation_stats else {}
            bundle = {
                "version": "4.6",
                "timestamp": datetime.datetime.now().isoformat(),
                "settings": self._sce(),
                "network": {
                    "img_height": H, "img_width": W, "n_layers": H - 1,
                    "n_neurons_per_layer": W, "n_genes": H * W * 3,
                    "image_shape": [H, W, 3],
                    "activations": ACTIVATION_NAMES,
                    "n_activation_types": N_ACTIVATIONS,
                    "encoding": {"hue": "activation", "saturation": "weight[-1,1]",
                                "value": "bias[-1,1]"},
                },
                "image": self.best_image.tolist(),
                "diagnostics": {
                    "best_train_loss_ever": float(self._btl_ever) if self._btl_ever != float('inf') else None,
                    "best_test_acc_ever": float(self._bta_ever),
                    "final_generation": last.get('gen', 0),
                    "final_train_loss": last.get('tr_loss'),
                    "final_test_loss": last.get('te_loss'),
                    "final_train_acc": last.get('tr_acc'),
                    "final_test_acc": last.get('te_acc'),
                    "best_generation": best_gen.get('gen'),
                    "best_test_loss": best_gen.get('te_loss'),
                    "best_test_acc": best_gen.get('te_acc'),
                    "total_generations_logged": len(self.generation_stats),
                    "elapsed_time_sec": (time.time() - self._t0) if self._t0 else None,
                    "diversity": float(self._div),
                    "gen_rate": float(self._grate),
                    "dataset_info": info,
                    "system_snapshot": {
                        "gpu": {"vram_used": vu, "vram_total": vt, "cupy_available": CUPY_AVAILABLE},
                        "ram": {"process_rss": rss, "system_avail": av,
                               "system_total": st, "psutil_available": PSUTIL_AVAILABLE},
                        "cpu_percent": cpu,
                    },
                    "profiler": self._lpd or {},
                    "generation_stats_tail": self.generation_stats[-100:],
                    "convergence_points": {
                        "train_loss": self.cp.tl[-200:],
                        "test_loss": self.cp.tel[-200:],
                        "train_acc": self.cp.ta[-200:],
                        "test_acc": self.cp.tea[-200:],
                    },
                },
            }
            task = SaveTask(self._execute_save_diagnostics, fn, bundle)
            self.thread_pool.start(task)
            self.st.setText(f"🩺 Diagnostics saving to: {fn}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Save Error", str(e))

    def load_diagnostics(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load Diagnostics", os.getcwd(),
                                           "JSON (*.json);;All (*.*)")
        if not fn: return
        try:
            with open(fn) as f:
                bundle = json.load(f)
            c = bundle.get("settings", {})
            if c:
                self.load_config_from_dict(c)
            img = np.array(bundle["image"], dtype=np.float64) if "image" in bundle else None
            if img is not None and img.ndim == 3 and img.shape[2] == 3:
                self.iw.blockSignals(True); self.iw.setValue(img.shape[1]); self.iw.blockSignals(False)
                self.ih.blockSignals(True); self.ih.setValue(img.shape[0]); self.ih.blockSignals(False)
                self._li = img; self.best_image = img
                us = max(2, 500 // max(1, img.shape[0]))
                self.il.setPixmap(hsv_array_to_pixmap(img, us))
                self.svb.setEnabled(True); self.smb.setEnabled(True)
                self.sdb.setEnabled(True); self.bmb.setEnabled(True)
                self.prb.setEnabled(True); self.epb.setEnabled(True)
                self.rsb.setEnabled(True)
                self.llv.set_enabled(True)
                self.topo.set_image(img)
                self.wh.update_weights(img)
            d = bundle.get("diagnostics", {})
            self._btl_ever = d.get("best_train_loss_ever") or float('inf')
            self._bta_ever = d.get("best_test_acc_ever") or 0.0
            self._lpd = d.get("profiler") or None
            if self._lpd: self.dw.update_profiler(self._lpd)
            conv = d.get("convergence_points", {})
            self.cp.clear_data()
            tl = conv.get("train_loss", [])
            tel = conv.get("test_loss", [])
            ta = conv.get("train_acc", [])
            tea = conv.get("test_acc", [])
            n = min(len(tl), len(tel), len(ta), len(tea))
            for i in range(n):
                self.cp.add_point(tl[i], tel[i], ta[i], tea[i])
            gs = d.get("generation_stats_tail", [])
            self.generation_stats = list(gs)
            self._ust()
            self.lml.setText(f"🩺 Loaded diag | gen {d.get('final_generation', '?')} | best loss {self._btl_ever if self._btl_ever != float('inf') else 0:.6f}")
            self.st.setText(f"🩺 Diagnostics loaded from {fn}")
            self._arch()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Load Error", str(e))

    def tdark(self):
        if self._dark:
            self.setStyleSheet(""); self._dark = False; self.db.setText("🌙")
        else:
            self.setStyleSheet(DARK_STYLE); self._dark = True; self.db.setText("☀")

    def _set_ctrl(self, en):
        self.config_tabs.setEnabled(en)
        widgets = [self.smb, self.lmb, self.epb, self.scfg, self.lcfg, self.copy_btn, self.db, self.sdb, self.lddb, self.rst_btn]
        for w in widgets:
            w.setEnabled(en)

    def _execute_save_model(self, fn, img, st2=None):
        if fn.endswith('.json'):
            with open(fn, 'w') as f:
                json.dump(st2, f)
        else:
            np.save(fn, img)

    def save_model(self):
        if self.best_image is None: return
        fn, _ = QFileDialog.getSaveFileName(self, "Save",
            os.path.join(os.getcwd(), "model.npy"), "NumPy (*.npy);;JSON (*.json)")
        if not fn: return
        if fn.endswith('.json'):
            st2 = {'image': self.best_image.tolist(), 'shape': list(self.best_image.shape),
                  'best_loss': self._btl_ever, 'best_acc': self._bta_ever,
                  'gen': self.generation_stats[-1]['gen'] if self.generation_stats else 0,
                  'config': self._sce()}
            task = SaveTask(self._execute_save_model, fn, self.best_image.copy(), st2)
        else:
            task = SaveTask(self._execute_save_model, fn, self.best_image.copy())
        self.thread_pool.start(task)
        self.st.setText(f"💾 Model saving to {fn}...")

    def load_model(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load", os.getcwd(),
                                            "NumPy (*.npy);;JSON (*.json);;All (*.*)")
        if not fn: return
        if fn.endswith('.json'):
            with open(fn) as f:
                st2 = json.load(f)
            img = np.array(st2['image'])
            cfg = st2.get('config', {})
            if cfg:
                self.iw.blockSignals(True); self.iw.setValue(cfg.get('img_width', img.shape[1])); self.iw.blockSignals(False)
                self.ih.blockSignals(True); self.ih.setValue(cfg.get('img_height', img.shape[0])); self.ih.blockSignals(False)
            self._li = img
            self.lml.setText(f"📁Ckpt:gen{st2.get('gen', '?')}|loss{st2.get('best_loss', 0):.6f}")
            self._arch()
        else:
            img = np.load(fn)
            if img.ndim != 3 or img.shape[2] != 3:
                QMessageBox.warning(self, "Error", "Expected HSV (H,W,3)"); return
            if img.shape != (self.ih.value(), self.iw.value(), 3):
                ret = QMessageBox.warning(self, "Mismatch",
                    f"Model {img.shape[0]}×{img.shape[1]}×3 vs {self.ih.value()}×{self.iw.value()}×3. Adjust?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if ret == QMessageBox.StandardButton.Yes:
                    self.iw.blockSignals(True); self.iw.setValue(img.shape[1]); self.iw.blockSignals(False)
                    self.ih.blockSignals(True); self.ih.setValue(img.shape[0]); self.ih.blockSignals(False)
                    self._arch()
                else:
                    return
            self._li = img
            self.lml.setText(f"📁{os.path.basename(fn)}({img.shape[0]}×{img.shape[1]}×3)")
        self.st.setText("📁 Loaded — seeds next run")

    def _execute_exp_net(self, fn, img, npy_path, od):
        H, W, _ = img.shape
        np.save(npy_path, img)
        code = f'''"""
Evolved Network: {H} layers x {W} neurons
Genome: {H}x{W}x3 HSV = {H*W*3} genes
Hue=activation({N_ACTIVATIONS}) Sat=weight[-1,1] Val=bias[-1,1]
Generated by GA Image-NN Evolver v4.6
"""
import os
import numpy as np

N_ACT={N_ACTIVATIONS}
ACTS=[
    lambda x:1/(1+np.exp(-np.clip(x,-50,50))),
    lambda x:np.tanh(np.clip(x,-50,50)),
    lambda x:np.maximum(0,x),
    lambda x:np.where(x>0,x,0.01*x),
    lambda x:x,
    lambda x:np.exp(-np.clip(x,-50,50)**2),
    lambda x:np.sin(x),
    lambda x:np.where(x>0,1.0,0.0),
    lambda x:x*(1/(1+np.exp(-np.clip(x,-50,50)))),
    lambda x:x*np.tanh(np.log1p(np.exp(np.clip(x,-50,50)))),
    lambda x:np.where(x>0,x,np.exp(np.clip(x,-50,0))-1),
    lambda x:1.0507009873554805*np.where(x>0,x,1.6732632423543772*(np.exp(np.clip(x,-50,0))-1)),
    lambda x:np.log1p(np.exp(np.clip(x,-50,50))),
    lambda x:np.clip(x*0.2+0.5,0,1),
    lambda x:x-np.tanh(np.clip(x,-50,50)),
    lambda x:x/(1+np.abs(x))
]
IMAGE=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), {os.path.basename(npy_path)!r})

def forward(X, output_dim={od}):
    H,W,_=IMAGE.shape
    x=np.zeros((X.shape[0],W))
    x[:,:min(X.shape[1],W)]=X[:,:W]
    for i in range(H-1):
        w=IMAGE[i,:,1]*2-1
        b=(IMAGE[i,:,2]-0.5)*2
        x=x*w+b
        a=(IMAGE[i,:,0]*{N_ACTIVATIONS}).astype(int)%{N_ACTIVATIONS}
        for k in range({N_ACTIVATIONS}):
            m=a==k
            if m.any(): x[:,m]=ACTS[k](x[:,m])
    return x[:,:output_dim]

def predict(X):
    p=forward(X)
    return np.argmax(p,axis=1) if p.shape[1]>1 else p[:,0]
'''
        with open(fn, 'w') as f:
            f.write(code)

    def exp_net(self):
        if self.best_image is None: return
        img = self.best_image
        H, W, _ = img.shape
        info = self.dataset_manager.get_dataset_info()
        od = info.get('n_outputs', 1) if info else 1
        fn, _ = QFileDialog.getSaveFileName(self, "Export",
            os.path.join(os.getcwd(), "network.py"), "Python (*.py)")
        if not fn: return
        npy_path = fn.replace('.py', '_weights.npy')
        
        task = SaveTask(self._execute_exp_net, fn, img.copy(), npy_path, od)
        self.thread_pool.start(task)
        self.st.setText(f"📤 Exporting to {fn}...")

    def _execute_exp_pruned(self, fn, pr, npy_path, od, np2, tot):
        H, W, _ = pr.shape
        np.save(npy_path, pr)
        code = f'''"""Pruned: {np2}/{tot} zeroed ({np2/tot*100:.1f}%)"""
import os
import numpy as np
N_ACT={N_ACTIVATIONS}
ACTS=[
    lambda x:1/(1+np.exp(-np.clip(x,-50,50))),
    lambda x:np.tanh(np.clip(x,-50,50)),
    lambda x:np.maximum(0,x),
    lambda x:np.where(x>0,x,0.01*x),
    lambda x:x,
    lambda x:np.exp(-np.clip(x,-50,50)**2),
    lambda x:np.sin(x),
    lambda x:np.where(x>0,1.0,0.0),
    lambda x:x*(1/(1+np.exp(-np.clip(x,-50,50)))),
    lambda x:x*np.tanh(np.log1p(np.exp(np.clip(x,-50,50)))),
    lambda x:np.where(x>0,x,np.exp(np.clip(x,-50,0))-1),
    lambda x:1.0507009873554805*np.where(x>0,x,1.6732632423543772*(np.exp(np.clip(x,-50,0))-1)),
    lambda x:np.log1p(np.exp(np.clip(x,-50,50))),
    lambda x:np.clip(x*0.2+0.5,0,1),
    lambda x:x-np.tanh(np.clip(x,-50,50)),
    lambda x:x/(1+np.abs(x))
]
IMAGE=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), {os.path.basename(npy_path)!r}))
def forward(X):
    H,W,_=IMAGE.shape
    x=np.zeros((X.shape[0],W))
    x[:,:min(X.shape[1],W)]=X[:,:W]
    for i in range(H-1):
        w=IMAGE[i,:,1]*2-1
        b=(IMAGE[i,:,2]-.5)*2
        x=x*w+b
        a=(IMAGE[i,:,0]*{N_ACTIVATIONS}).astype(int)%{N_ACTIVATIONS}
        for k in range({N_ACTIVATIONS}):
            m=a==k
            if m.any(): x[:,m]=ACTS[k](x[:,m])
    return x[:,:{od}]
'''
        with open(fn, 'w') as f:
            f.write(code)

    def exp_pruned(self):
        if self.best_image is None:
            QMessageBox.warning(self, "No Image", "Evolve first."); return
        if self.ncv.contrib is None:
            QMessageBox.warning(self, "No Data", "Neuron contribution computed every 100 gens."); return
        th, ok = QInputDialog.getDouble(self, "Prune",
            "Zero neurons with |contribution|<:", .001, 0, 1, 4)
        if not ok: return
        pr = self.best_image.copy()
        c = self.ncv.contrib
        np2 = 0
        tot = c.shape[0] * c.shape[1]
        for i in range(min(c.shape[0], pr.shape[0] - 1)):
            for j in range(min(c.shape[1], pr.shape[1])):
                if abs(c[i, j]) < th:
                    pr[i, j, 1] = .5
                    np2 += 1
        fn, _ = QFileDialog.getSaveFileName(self, "Export Pruned",
            os.path.join(os.getcwd(), "pruned.py"), "Python (*.py)")
        if not fn: return
        H, W, _ = pr.shape
        info = self.dataset_manager.get_dataset_info()
        od = info.get('n_outputs', 1) if info else 1
        npy_path = fn.replace('.py', '_pruned_weights.npy')
        
        task = SaveTask(self._execute_exp_pruned, fn, pr, npy_path, od, np2, tot)
        self.thread_pool.start(task)
        self.st.setText(f"✂ Pruning & Exporting... ({np2}/{tot} zeroed)")

    def _execute_qbm(self, fn, img):
        pix = hsv_array_to_pixmap(img, 4)
        if pix.save(fn):
            np.save(fn.replace('.png', '.npy'), img)

    def qbm(self):
        if self.best_image is None: return
        g = self.generation_stats[-1]['gen'] if self.generation_stats else 0
        l = self._btl_ever
        fn = os.path.join(os.getcwd(), f"bm_gen{g}_l{l:.4f}_{time.strftime('%H%M%S')}.png")
        
        task = SaveTask(self._execute_qbm, fn, self.best_image.copy())
        self.thread_pool.start(task)
        self.st.setText(f"🔖 Bookmarking to {os.path.basename(fn)}...")

    def _llc(self):
        if self.best_image is None: return
        self.llv.info.setText("Computing...")
        QApplication.processEvents()
        info = self.dataset_manager.get_dataset_info()
        H = self.ih.value(); W = self.iw.value()
        od = info.get('n_outputs', 1) if info else 1
        ug = self.gcb.isChecked()
        X = self.dataset_manager.current_dataset[0][:50]
        y = self.dataset_manager.current_dataset[1][:50]
        X = _pad_truncate(X.astype(np.float64), W)
        ls = self.engine.compute_landscape(self.best_image, X, y, H, W, od,
                                           ug and CUPY_AVAILABLE) if self.engine else None
        if ls is not None:
            self.llv.set_landscape(ls)
            self.llv.info.setText(f"Done: {ls.shape[0]}×{ls.shape[1]}")
        else:
            self.llv.info.setText("Engine not available")

    def resume_run(self):
        ckpts = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('ckpt_')],
                      key=lambda x: int(x.split('_gen')[1].split('.')[0]))
        if not ckpts:
            QMessageBox.information(self, "No Checkpoints", "No checkpoints found.")
            return
        latest = ckpts[-1]
        gen = int(latest.split('_gen')[1].split('.')[0])
        ret = QMessageBox.question(self, "Resume",
            f"Resume from checkpoint at gen {gen}?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ret != QMessageBox.StandardButton.Yes: return
        self.engine = Engine()
        self.engine.params = self._build_params()
        self.engine.preloaded_dataset = self.dataset_manager.current_dataset
        self.engine.resume_mode = True
        self.engine.checkpoint_path = os.path.join(CHECKPOINT_DIR, latest)
        self.engine._t0 = time.time() - gen * 0.1
        self._connect_engine_signals()
        self._set_ctrl(False)
        self.sb.setEnabled(False); self.stb.setEnabled(True)
        self.pg.setValue(0)
        self.st.setText(f"Resuming from gen {gen}...")
        self.engine.start()

    def _build_params(self):
        W = self.iw.value(); H = self.ih.value()
        info = self.dataset_manager.get_dataset_info()
        ug = self.gcb.isChecked()
        params = dict(
            img_height=H, img_width=W, pop_size=self.pop.value(),
            elite_frac=self.el.value() / 100., mut_rate=self.mu.value(),
            seed=self.ss.value(), delay_ms=self.dy.value(), use_gpu=ug,
            use_pca=self.pcb.isChecked(), pca_components=self.ps.value(),
            pca_alpha=self.pas.value(),
            is_classification=self.dataset_manager.is_classification,
            loss_threshold=self.lts.value(), acc_threshold=self.ats.value(),
            batch_size=self.bss.value(), test_frac=self.tss.value(),
            dataset_info=self.dataset_manager.get_dataset_info(),
            patience=self.ps2.value(), min_improvement=self.mis.value(),
            conv_window=self.cws.value(), min_generations=self.mgs.value(),
            tournament_size=self.ts2.value(),
            crossover_type=self.cc.currentText().lower(),
            mutation_dist=self.mdc.currentText().lower(),
            self_adaptive_mut=self.amc.isChecked(),
            restart_on_stagnation=self.rcb.isChecked(),
            crowding=self.crcb.isChecked(),
            fitness_sharing=self.shcb.isChecked(),
            multi_parent=self.mpc.isChecked(),
            n_islands=self.isl.value(),
            migration_interval=self.mis2.value(),
            opposition_init=self.opc.isChecked(),
            fitness_type=self.fc.currentText(),
            adaptive_weights=self.awc.isChecked(),
            l2_weight=self.l2s.value(), l1_weight=self.l1s.value(),
            gap_weight=self.gps.value(), act_div_weight=self.ads.value(),
            log_file=self.le.text(),
            log_mode='w' if self.lmc.currentText().startswith('Overwrite') else 'a',
            use_memetic=self.memetic_cb.isChecked(),
            memetic_interval=self.memetic_interval_sp.value(),
            use_nsga2=self.nsga2_cb.isChecked(),
            use_speciation=self.speciation_cb.isChecked(),
            speciation_threshold=self.spec_thresh_sp.value(),
            use_aos=self.aos_cb.isChecked(),
            use_statistical_stop=self.stat_stop_cb.isChecked(),
            use_augmentation=self.aug_cb.isChecked(),
            aug_factor=self.aug_factor_sp.value(),
            checkpoint_interval=self.ckpt_sp.value(),
            class_weights=self.dataset_manager.class_weights,
        )
        return params

    def _connect_engine_signals(self):
        self.engine.generation_done.connect(self.on_gen)
        self.engine.image_ready.connect(self.on_img)
        self.engine.dataset_loaded.connect(lambda i: self._udt(i))
        self.engine.finished_run.connect(self.on_fin)
        self.engine.error_occurred.connect(self.on_err)
        self.engine.stop_reason.connect(lambda r: self.st.setText(f"Stopped:{r}"))
        self.engine.progress_update.connect(self.pg.setValue)
        self.engine.stats_update.connect(self.on_st)
        self.engine.predictions_ready.connect(self.on_pred)
        self.engine.live_predictions.connect(self.on_lpred)
        self.engine.profiler_update.connect(self.on_prof)
        self.engine.stopping_status.connect(self.on_ss)
        self.engine.activation_stats.connect(self.on_as)
        self.engine.image_snapshot.connect(self.on_snap)
        self.engine.topology_update.connect(self.on_topo)
        self.engine.gene_importance_update.connect(self.on_gi)
        self.engine.weight_dist_update.connect(self.on_wd)
        self.engine.fit_div_update.connect(self.on_fd)
        self.engine.neuron_contribution.connect(self.on_nc)
        self.engine.population_pca.connect(self.on_pp)
        self.engine.fitness_breakdown.connect(self.on_fb)
        self.engine.state_changed.connect(self.arch.update_state)
        self.engine.confusion_matrix_ready.connect(self.on_cm)
        self.engine.pareto_update.connect(self.on_pareto)
        self.engine.operator_stats.connect(self.on_os)
        self.engine.checkpoint_saved.connect(self.on_ckpt)
        self.engine.landscape_ready.connect(self.on_landscape)

    def start(self):
        if self.engine is not None and self.engine.isRunning():
            QMessageBox.warning(self, "Running", "Wait."); return
        if self.dataset_manager.current_dataset is None:
            QMessageBox.warning(self, "No Data", "Load dataset first."); return
        W = self.iw.value(); H = self.ih.value()
        info = self.dataset_manager.get_dataset_info()
        if info['n_outputs'] > W:
            QMessageBox.warning(self, "Too Small", f"Output {info['n_outputs']}>W={W}"); return
        ug = self.gcb.isChecked()
        gn = H * W
        if gn > 65536 and not ug:
            ret = QMessageBox.warning(self, "⚠️ Extreme",
                f"Hidden {H}×{W}={gn:,} genes.\nNo GPU=EXTREMELY slow.\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ret != QMessageBox.StandardButton.Yes: return
        elif gn > 4096 and not ug:
            ret = QMessageBox.warning(self, "Perf",
                f"Hidden {H}×{W}={gn:,}. No GPU slow. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ret != QMessageBox.StandardButton.Yes: return
        if gn > 1000000:
            ret = QMessageBox.warning(self, "Massive",
                f"{gn:,} genes! Pop {self.pop.value()} needs {format_bytes(self.pop.value() * gn * 3 * 8)}. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ret != QMessageBox.StandardButton.Yes: return
        params = self._build_params()
        self.engine = Engine()
        self.engine.params = params
        self.engine.preloaded_dataset = self.dataset_manager.current_dataset
        self.engine.loaded_image = self._li
        self.engine._t0 = time.time()
        self.engine._bta_ever = 0.
        self._connect_engine_signals()
        self._set_ctrl(False)
        self.sb.setEnabled(False); self.stb.setEnabled(True)
        self.svb.setEnabled(False); self.smb.setEnabled(False)
        self.bmb.setEnabled(False); self.prb.setEnabled(False)
        self.epb.setEnabled(False); self.sdb.setEnabled(False)
        self.rsb.setEnabled(False)
        self.pg.setValue(0)
        ic = params['is_classification']
        self.cp.set_thresholds(params['loss_threshold'], params['acc_threshold'], ic)
        self.cp.clear_data()
        self.pv.clear_view(); self.pred.clear_view()
        self.ast.clear(); self.giv.clear(); self.wh.clear()
        self.atl.clear(); self.fdp.clear(); self.ncv.clear()
        self.ppv.clear(); self.ssh.clear(); self.llv.clear()
        self.topo.clear(); self.hist.clear_history()
        self.cmw.clear(); self.pfw.clear(); self.osw.clear()
        self._t0 = time.time()
        self._btl_ever = float('inf'); self._bta_ever = 0.
        self._lpd = None
        n = info['n_samples']
        tf = params['test_frac']
        bs = params['batch_size']
        if tf > 0 and n >= 10:
            ntr = int(n * (1 - tf)); nte = n - ntr
            ss = f"Train:{ntr} Test:{nte}"
        else:
            ss = f"All:{n}"
        ab = min(bs, ntr if (tf > 0 and n >= 10) else n)
        t = "Classification" if ic else "Regression"
        m = "GPU" if ug else "CPU"
        sd = " +seed" if self._li is not None else ""
        me = estimate_peak_memory(params['pop_size'], ab, H, W, info['n_outputs'])
        ft2 = self.fc.currentText()
        features = []
        if params.get('use_memetic'): features.append("⚡Memetic")
        if params.get('use_nsga2'): features.append("NSGA-II")
        if params.get('use_speciation'): features.append("Speciation")
        if params.get('use_aos'): features.append("AOS")
        if params.get('use_statistical_stop'): features.append("StatStop")
        if params.get('use_augmentation'): features.append("Aug")
        feat_str = f" [{', '.join(features)}]" if features else ""
        self.st.setText(f"Running:{H}×{W},{params['pop_size']}pop,{H-1}layers×{W}neurons|{m}|{t}{sd}{feat_str}\n  In:{info['n_features']}→Out:{info['n_outputs']}|{ss}|Batch:{ab}|Est:{format_bytes(me['total'])}\n  GA:tour={self.ts2.value()} cross={self.cc.currentText()} mut={self.mdc.currentText()}\n  Fitness:{ft2}\n  Stop:loss≤{params['loss_threshold']}" + (f"|acc≥{params['acc_threshold']*100:.0f}%" if ic else "") + f"|patience={params['patience']}")
        self.generation_stats = []
        self.engine.start()

    def stop(self):
        if self.engine and self.engine.isRunning():
            self.engine.requestInterruption()
            self.stb.setEnabled(False)
            self.st.setText("Stopping... (waiting for current gen to finish)")

    def on_err(self, m):
        self.st.setText(f"Error:{m}")
        QMessageBox.critical(self, "Error", m)
        self._set_ctrl(True)
        self.sb.setEnabled(True); self.stb.setEnabled(False)

    def on_st(self, d, r):
        self._div = d; self._grate = r

    def on_ss(self, st):
        self.ss2.setText(st)

    def on_as(self, c):
        self.ast.update_counts(c)
        self.atl.add_point(self.generation_stats[-1]['gen'] if self.generation_stats else 0, c)

    def on_snap(self, g, l, img):
        self.hist.add_snapshot(g, l, img)
        self.ssh.set_snapshots(self.hist.snapshots)

    def on_topo(self, img):
        self.topo.set_image(img)

    def on_gi(self, h):
        self.giv.update_heatmap(h)

    def on_wd(self, img):
        self.wh.update_weights(img)

    def on_fd(self, d, f):
        self.fdp.add_point(d, f)

    def on_nc(self, c):
        self.ncv.set_contribution(c)

    def on_pp(self, c, l):
        self.ppv.set_data(c, l)

    def on_fb(self, ph, l, l2, l1, ad):
        parts = [f"Phase:{ph}", f"loss:{l:.4f}"]
        if l2 > 0: parts.append(f"L2:{l2:.4f}")
        if l1 > 0: parts.append(f"L1:{l1:.4f}")
        if ad > 0: parts.append(f"actDiv:{ad:.3f}")
        base = self.dl2.text().split(" | Fitness:")[0]
        self.dl2.setText(base + " | Fitness: " + " ".join(parts))

    def on_cm(self, cm, cn):
        self.cmw.set_confusion_matrix(cm, cn)

    def on_pareto(self, objs, fronts):
        self.pfw.set_pareto_data(objs, fronts)

    def on_os(self, stats):
        self.osw.update_stats(stats)

    def on_ckpt(self, path):
        self.rsb.setEnabled(True)
        self.lml.setText(f"💾 Checkpoint: {os.path.basename(path)}")

    def on_landscape(self, ls):
        self.llv.set_landscape(ls)

    def on_gen(self, g, tl, mt, pw, ta, tel, tea):
        self._btl_ever = min(self._btl_ever, tel)
        self._bta_ever = max(self._bta_ever, tea)
        el = time.time() - self._t0 if self._t0 else 0
        ps = [f"Gen {g}", f"TrLoss:{tl:.6f}"]
        if ta > 0: ps.append(f"TrAcc:{ta*100:.1f}%")
        ps.append(f"TeLoss:{tel:.6f}")
        if tea > 0: ps.append(f"TeAcc:{tea*100:.1f}%")
        ps.append(f"{self._grate:.1f}gen/s")
        ps.append(f"⏱{el:.0f}s")
        self.st.setText(" | ".join(ps))
        self.generation_stats.append({'gen': g, 'tr_loss': tl, 'te_loss': tel,
                                      'tr_acc': ta, 'te_acc': tea})
        if len(self.generation_stats) > 1000:
            self.generation_stats = self.generation_stats[-1000:]
        self.cp.add_point(tl, tel, ta, tea)
        if g % 10 == 0: self._ust()

    def on_img(self, img):
        self.best_image = img
        us = max(2, 500 // max(1, img.shape[0]))
        self.il.setPixmap(hsv_array_to_pixmap(img, us))
        self.svb.setEnabled(True); self.smb.setEnabled(True)
        self.bmb.setEnabled(True); self.prb.setEnabled(True)
        self.epb.setEnabled(True); self.llv.set_enabled(True)
        self.sdb.setEnabled(True)

    def on_pca(self, c, ev, nf):
        self.current_pca_weights = c
        self.pv.set_pca(c, ev, nf)

    def on_lpred(self, st):
        if self.dataset_manager.is_classification:
            self.pred.set_class_names(get_class_names(
                self.dataset_manager.get_dataset_info().get('name', ''),
                self.dataset_manager.num_classes))
        self.pred.show_predictions(st, is_live=True)
        self._fp()

    def on_pred(self, st):
        if self.dataset_manager.is_classification:
            self.pred.set_class_names(get_class_names(
                self.dataset_manager.get_dataset_info().get('name', ''),
                self.dataset_manager.num_classes))
        self.pred.show_predictions(st, is_live=False)
        self._fp()

    def on_prof(self, d):
        self._lpd = d
        self.dw.update_profiler(d)

    def _ust(self):
        if not self.generation_stats: return
        el = time.time() - self._t0 if self._t0 else 0
        t = f"Evolution Statistics:\n\nGens:{len(self.generation_stats)}\nElapsed:{el:.1f}s\nDiversity:{self._div:.4f}\nRate:{self._grate:.1f}gen/s\nBest test loss:{self._btl_ever:.6f}\nBest test acc:{self._bta_ever*100:.2f}%\n\n"
        t += f"{'Gen':>4}|{'TrLoss':>10}|{'TeLoss':>10}|{'TrAcc':>6}|{'TeAcc':>6}|{'Gap':>6}\n" + "-"*60 + "\n"
        for st in self.generation_stats[-10:]:
            gap = (st['tr_acc'] - st['te_acc']) * 100
            t += (f"{st['gen']:>4}|{st['tr_loss']:>10.6f}|{st['te_loss']:>10.6f}|{st['tr_acc']*100:>5.1f}%|{st['te_acc']*100:>5.1f}%|{gap:>+5.1f}%\n")
        b = min(self.generation_stats, key=lambda x: x['te_loss'])
        t += f"\nBest:Gen {b['gen']}|TeLoss {b['te_loss']:.6f}"
        if b['te_acc'] > 0: t += f"|TeAcc {b['te_acc']*100:.2f}%"
        t += f"\n\nActivations: {' | '.join(ACTIVATION_NAMES)}\n"
        self.sl.setText(t)

    def on_fin(self):
        self._set_ctrl(True)
        self.sb.setEnabled(True); self.stb.setEnabled(False)
        self.rsb.setEnabled(True)
        self.pg.setValue(100)
        self._ust()
        self.setWindowTitle("✅ Complete — GA Image-NN Evolver v4.6")
        self.activateWindow(); self.raise_()
        if sys.platform == 'win32':
            try:
                import winsound
                winsound.Beep(800, 200); winsound.Beep(1000, 200); winsound.Beep(1200, 300)
            except Exception: pass
        elif sys.platform == 'darwin':
            try: os.system('afplay /System/Library/Sounds/Glass.aiff &')
            except Exception: pass
        QTimer.singleShot(3000, lambda: self.setWindowTitle("GA Image-NN Evolver — Enhanced Complete Edition v4.6"))

    def save_image(self):
        if self.best_image is None: return
        fn, _ = QFileDialog.getSaveFileName(self, "Save",
            os.path.join(os.getcwd(), "evolved.png"), "PNG (*.png);;All (*.*)")
        if not fn: return
        task = SaveTask(self._execute_qbm, fn, self.best_image.copy())
        self.thread_pool.start(task)
        self.st.setText(f"Saving image to {fn}...")

    def closeEvent(self, ev):
        self._save_app_settings()
        if self.engine is not None and self.engine.isRunning():
            self.engine.requestInterruption()
            self.st.setText("Stopping engine...")
            timeout_ms = 5000
            while self.engine.isRunning() and timeout_ms > 0:
                QApplication.processEvents()
                self.engine.wait(100)
                timeout_ms -= 100
            if self.engine.isRunning():
                self.engine.terminate()
                self.engine.wait(500)
        self.mt.stop()
        gc.collect()
        if CUPY_AVAILABLE:
            try: cp.get_default_memory_pool().free_all_blocks()
            except Exception: pass
        super().closeEvent(ev)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    w = MainWindow()
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        try:
            with open(sys.argv[1]) as f:
                cfg = json.load(f)
            if 'img_width' in cfg: w.iw.setValue(cfg['img_width'])
            if 'img_height' in cfg: w.ih.setValue(cfg['img_height'])
            if 'pop_size' in cfg: w.pop.setValue(cfg['pop_size'])
            w._arch()
            print(f"✓ Loaded config from {sys.argv[1]}")
            if cfg.get('auto_start', False) and w.sb.isEnabled(): w.start()
        except Exception as e:
            print(f"⚠ Config error: {e}")
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()