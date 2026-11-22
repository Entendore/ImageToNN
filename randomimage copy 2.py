import sys
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
import time
import pickle
from datetime import datetime
import traceback

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QTabWidget, QPushButton, QLabel,
                                QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox,
                                QFormLayout, QProgressBar, QTextEdit, QStatusBar,
                                QMenuBar, QMenu, QFileDialog, QMessageBox,
                                QSplitter, QFrame, QCheckBox, QSlider, QScrollArea,
                                QToolBar, QToolButton, QStyle, QInputDialog,
                                QListWidget, QListWidgetItem, QDialog, QDialogButtonBox,
                                QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QSize, QTimer, QSettings
from PySide6.QtGui import QFont, QColor, QPalette, QAction, QIcon, QKeySequence, QShortcut

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════
# THEME MANAGEMENT
# ═══════════════════════════════════════════════
class ThemeManager:
    LIGHT_THEME = {
        'bg': '#FAFAFA', 'fg': '#264653', 'accent': '#2A9D8F',
        'danger': '#E63946', 'warning': '#F4A261', 'success': '#2A9D8F',
        'border': '#bdc3c7', 'card': '#FFFFFF', 'hover': '#ecf0f1',
        'plot_bg': '#FAFAFA', 'plot_grid': '#E0E0E0'
    }
    
    DARK_THEME = {
        'bg': '#1E1E1E', 'fg': '#E0E0E0', 'accent': '#4ECDC4',
        'danger': '#FF6B6B', 'warning': '#FFD93D', 'success': '#6BCF7F',
        'border': '#3E3E3E', 'card': '#2D2D2D', 'hover': '#3A3A3A',
        'plot_bg': '#2D2D2D', 'plot_grid': '#4A4A4A'
    }
    
    @staticmethod
    def get_stylesheet(theme_name='light'):
        theme = ThemeManager.LIGHT_THEME if theme_name == 'light' else ThemeManager.DARK_THEME
        return f"""
            QMainWindow {{ background-color: {theme['bg']}; }}
            QWidget {{ color: {theme['fg']}; }}
            QTabWidget::pane {{ border: 1px solid {theme['border']}; background-color: {theme['card']}; }}
            QTabBar::tab {{
                background-color: {theme['hover']}; padding: 12px 24px; margin-right: 2px;
                border-top-left-radius: 6px; border-top-right-radius: 6px;
                font-weight: bold; color: {theme['fg']};
            }}
            QTabBar::tab:selected {{ background-color: {theme['accent']}; color: white; }}
            QTabBar::tab:hover {{ background-color: {theme['border']}; }}
            QGroupBox {{
                font-weight: bold; font-size: 13px; border: 2px solid {theme['accent']};
                border-radius: 8px; margin-top: 12px; padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 12px; padding: 0 8px; color: {theme['accent']};
            }}
            QPushButton {{
                background-color: {theme['accent']}; color: white; padding: 10px 20px;
                border-radius: 6px; font-weight: bold; border: none;
            }}
            QPushButton:hover {{ background-color: {theme['accent']}dd; }}
            QPushButton:pressed {{ background-color: {theme['accent']}bb; }}
            QPushButton:disabled {{ background-color: {theme['border']}; color: {theme['fg']}88; }}
            QLineEdit, QComboBox {{
                padding: 8px; border: 2px solid {theme['border']}; border-radius: 5px;
                background-color: {theme['card']}; color: {theme['fg']};
            }}
            QProgressBar {{
                border: 2px solid {theme['border']}; border-radius: 5px; text-align: center;
                font-weight: bold; background-color: {theme['card']};
            }}
            QProgressBar::chunk {{ background-color: {theme['accent']}; }}
            QTextEdit {{
                background-color: {theme['card']}; color: {theme['fg']};
                border: 1px solid {theme['border']}; border-radius: 5px; padding: 5px;
            }}
            QScrollArea {{ border: none; background-color: {theme['bg']}; }}
            QStatusBar {{ background-color: {theme['hover']}; color: {theme['fg']}; font-weight: bold; }}
            QMenuBar {{ background-color: {theme['hover']}; color: {theme['fg']}; }}
            QMenuBar::item:selected {{ background-color: {theme['accent']}; color: white; }}
            QMenu {{ background-color: {theme['card']}; color: {theme['fg']}; border: 1px solid {theme['border']}; }}
            QMenu::item:selected {{ background-color: {theme['accent']}; color: white; }}
            QToolBar {{ background-color: {theme['hover']}; border: none; spacing: 10px; padding: 5px; }}
            QToolButton {{ background-color: transparent; border: none; padding: 8px; border-radius: 5px; }}
            QToolButton:hover {{ background-color: {theme['border']}; }}
            QCheckBox {{ font-size: 12px; padding: 4px; }}
        """

# ═══════════════════════════════════════════════
# ACTIVATION FUNCTIONS (24 Types)
# ═══════════════════════════════════════════════
def relu(x): return np.maximum(0, x)
def linear(x): return x
def leaky_relu(x, a=0.01): return np.where(x > 0, x, a * x)
def prelu(x, a=0.25): return np.where(x > 0, x, a * x)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))
def tanh_act(x): return np.tanh(np.clip(x, -100, 100))
def softsign(x): return x / (1.0 + np.abs(x))
def elu(x, a=1.0): return np.where(x > 0, x, a * (np.exp(np.clip(x, -100, 0)) - 1.0))
def softplus(x): return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -100, 20))))
def swish(x): return x * sigmoid(x)
def hard_sigmoid(x): return np.clip(x / 6.0 + 0.5, 0.0, 1.0)
def hard_swish(x): return x * hard_sigmoid(x)
def mish(x): return x * np.tanh(softplus(x))
def gelu(x):
    x = np.clip(x, -10, 10)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
def bent_identity(x): return (np.sqrt(x**2 + 1.0) - 1.0) / 2.0 + x
def square(x): return np.clip(x, -10, 10)**2
def cube(x): return np.clip(x, -10, 10)**3
def abs_act(x): return np.abs(x)
def sign(x): return np.sign(x)
def step(x): return np.where(x > 0, 1.0, 0.0)
def gaussian(x): return np.exp(-np.clip(x, -10, 10)**2)
def sinc(x): return np.sinc(np.clip(x, -10*np.pi, 10*np.pi) / np.pi)
def cosine(x): return np.cos(np.clip(x, -10*np.pi, 10*np.pi))
def sine(x): return np.sin(np.clip(x, -10*np.pi, 10*np.pi))

ACTIVATION_MAP = [
    (15, relu, "ReLU"),         (30, sigmoid, "Sigmoid"),
    (45, tanh_act, "Tanh"),     (60, leaky_relu, "LeakyReLU"),
    (75, swish, "Swish"),       (90, linear, "Linear"),
    (105, gelu, "GELU"),        (120, elu, "ELU"),
    (135, softplus, "Softplus"),(150, mish, "Mish"),
    (165, hard_sigmoid, "HardSigmoid"), (180, softsign, "Softsign"),
    (195, prelu, "PReLU"),      (210, hard_swish, "HardSwish"),
    (225, square, "Square"),    (240, cube, "Cube"),
    (255, abs_act, "Abs"),      (270, sign, "Sign"),
    (285, step, "Step"),        (300, gaussian, "Gaussian"),
    (315, bent_identity, "BentIdentity"), (330, sinc, "Sinc"),
    (345, cosine, "Cosine"),    (360, sine, "Sine")
]
ACTIVATION_FUNCS = [f for _, f, _ in ACTIVATION_MAP]

def hue_to_activation_index(hue_deg):
    for i, (threshold, _, _) in enumerate(ACTIVATION_MAP):
        if hue_deg < threshold: return i
    return 5

# ═══════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════
class LossFunctions:
    @staticmethod
    def binary_cross_entropy(y_true, logits):
        logits = np.clip(logits, -100, 100)
        loss = np.maximum(logits, 0) - logits * y_true + np.log1p(np.exp(-np.abs(logits)))
        return np.mean(loss)
    
    @staticmethod
    def focal_loss(y_true, logits, gamma=2.0, alpha=0.25):
        logits = np.clip(logits, -100, 100)
        p = 1.0 / (1.0 + np.exp(-logits))
        bce = np.maximum(logits, 0) - logits * y_true + np.log1p(np.exp(-np.abs(logits)))
        p_t = p * y_true + (1 - p) * (1 - y_true)
        focal_weight = (1 - p_t) ** gamma
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        loss = alpha_t * focal_weight * bce
        return np.mean(loss)
    
    @staticmethod
    def hinge_loss(y_true, logits):
        y_signed = 2 * y_true - 1
        loss = np.maximum(0, 1 - y_signed * logits)
        return np.mean(loss)
    
    @staticmethod
    def huber_loss(y_true, logits, delta=1.0):
        preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -100, 100)))
        error = np.abs(y_true - preds)
        quadratic = 0.5 * error**2
        linear = delta * error - 0.5 * delta**2
        loss = np.where(error <= delta, quadratic, linear)
        return np.mean(loss)
    
    @staticmethod
    def log_cosh_loss(y_true, logits):
        preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -100, 100)))
        diff = preds - y_true
        loss = np.abs(diff) + np.log1p(np.exp(-2 * np.abs(diff))) - np.log(2)
        return np.mean(loss)
    
    @staticmethod
    def get_all_losses(y_true, logits):
        return {
            "BCE": LossFunctions.binary_cross_entropy(y_true, logits),
            "Focal (γ=2)": LossFunctions.focal_loss(y_true, logits, gamma=2.0),
            "Focal (γ=5)": LossFunctions.focal_loss(y_true, logits, gamma=5.0),
            "Hinge": LossFunctions.hinge_loss(y_true, logits),
            "Huber (δ=1)": LossFunctions.huber_loss(y_true, logits, delta=1.0),
            "Log-Cosh": LossFunctions.log_cosh_loss(y_true, logits),
        }

# ═══════════════════════════════════════════════
# BIAS CONFIG & ENGINE
# ═══════════════════════════════════════════════
class BiasConfig:
    def __init__(self, mapping='centered', spatial=False, depth_scale=True, act_aware=True):
        self.mapping = mapping
        self.spatial = spatial
        self.depth_scale = depth_scale
        self.act_aware = act_aware

class BiasEngine:
    @staticmethod
    def compute_layer_bias(V_map, layer_idx, config, act_indices_layer):
        if config.mapping == 'centered':
            bias = 2.0 * V_map - 1.0
        elif config.mapping == 'sigmoidal':
            bias = 2.0 * (1.0 / (1.0 + np.exp(-12 * (V_map - 0.5)))) - 1.0
        elif config.mapping == 'gaussian':
            bias = np.exp(-((V_map - 0.5) ** 2) / 0.05) - 0.5
        else:
            bias = V_map.copy()

        if config.spatial:
            if V_map.ndim == 2:
                pad = 1
                padded = np.pad(V_map, pad, mode='edge')
                local_mean = (padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                              padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
                              padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]) / 9.0
                bias = V_map - local_mean
            else:
                pad = 1
                padded = np.pad(V_map, pad, mode='edge')
                local_mean = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
                bias = V_map - local_mean

        if config.depth_scale:
            bias *= (1.0 / np.sqrt(layer_idx + 1))

        if config.act_aware:
            multipliers = np.ones_like(bias)
            for idx in [1, 2, 10, 11]:
                multipliers[act_indices_layer == idx] = 0.2
            for idx in [18, 19]:
                multipliers[act_indices_layer == idx] = 2.0
            bias *= multipliers

        return bias

def build_network_from_hsv(H_arr, S_arr, V_arr, bias_config):
    num_neurons, num_layers = H_arr.shape
    weights = np.zeros((num_layers, num_neurons))
    biases = np.zeros((num_layers, num_neurons))
    act_indices = np.zeros((num_layers, num_neurons), dtype=int)
    
    for l in range(num_layers):
        h_col = H_arr[:, l]
        s_col = S_arr[:, l]
        v_col = V_arr[:, l]
        weights[l, :] = s_col
        act_indices[l, :] = [hue_to_activation_index(h) for h in h_col]
        biases[l, :] = BiasEngine.compute_layer_bias(v_col, l, bias_config, act_indices[l, :])
        
    return weights, biases, act_indices

def evaluate_network(X, y, weights, biases, act_indices):
    batch_size = X.shape[0]
    num_layers, layer_neurons = weights.shape
    current_X = X.copy()
    
    for l in range(num_layers):
        prev_sum = np.sum(current_X, axis=1, keepdims=True)
        Z = prev_sum * weights[l] + biases[l]
        
        next_X = np.zeros_like(Z)
        for i in range(layer_neurons):
            act_func = ACTIVATION_FUNCS[act_indices[l, i]]
            next_X[:, i] = act_func(Z[:, i])
        current_X = next_X
    
    raw_mean = np.mean(current_X, axis=1)
    raw_std = np.std(current_X, axis=1) + 1e-8
    logits = (raw_mean / raw_std) * np.sqrt(layer_neurons)
    
    return logits

# ═══════════════════════════════════════════════
# INDEPENDENT IMAGE GENERATOR
# ═══════════════════════════════════════════════
class ImageGenerator:
    """
    Completely independent random HSV image generator.
    Has NO knowledge of neural networks or how images will be used.
    """
    
    def __init__(self, height=10, width=8, seed=None):
        self.height = height
        self.width = width
        self.rng = np.random.RandomState(seed)
        self.generated_count = 0
    
    def generate_single(self):
        """Generate a single random HSV image."""
        H = self.rng.uniform(0, 360, (self.height, self.width))
        S = self.rng.uniform(0, 1, (self.height, self.width))
        V = self.rng.uniform(0, 1, (self.height, self.width))
        self.generated_count += 1
        return H, S, V
    
    def generate_batch(self, count):
        """Generate multiple random HSV images."""
        return [self.generate_single() for _ in range(count)]
    
    def generate_stream(self):
        """Generator that yields images indefinitely."""
        while True:
            yield self.generate_single()
    
    def reset_seed(self, seed=None):
        """Reset the random number generator."""
        self.rng = np.random.RandomState(seed)
        self.generated_count = 0
    
    def get_stats(self):
        """Return generation statistics."""
        return {
            'total_generated': self.generated_count,
            'height': self.height,
            'width': self.width,
            'image_shape': (self.height, self.width, 3)
        }


# ═══════════════════════════════════════════════
# NETWORK BUILDER (Consumes images, no generation)
# ═══════════════════════════════════════════════
class NetworkBuilder:
    """
    Builds neural networks from HSV images.
    Only consumes images, never generates them.
    """
    
    def __init__(self, bias_config):
        self.bias_config = bias_config
        self.built_count = 0
    
    def build_from_image(self, H, S, V):
        """Build a network from a single HSV image."""
        weights, biases, act_indices = build_network_from_hsv(
            H, S, V, self.bias_config
        )
        self.built_count += 1
        return weights, biases, act_indices
    
    def build_batch(self, images):
        """Build networks from a batch of images."""
        return [self.build_from_image(H, S, V) for H, S, V in images]
    
    def get_stats(self):
        """Return building statistics."""
        return {
            'total_built': self.built_count,
            'bias_mapping': self.bias_config.mapping,
            'spatial': self.bias_config.spatial,
            'depth_scale': self.bias_config.depth_scale,
            'act_aware': self.bias_config.act_aware
        }


# ═══════════════════════════════════════════════
# NETWORK EVALUATOR (Independent component)
# ═══════════════════════════════════════════════
class NetworkEvaluator:
    """
    Evaluates networks on datasets.
    Only consumes networks, never builds or generates them.
    """
    
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        self.evaluated_count = 0
    
    def evaluate(self, weights, biases, act_indices):
        """Evaluate a single network."""
        logits = evaluate_network(
            self.X_test, self.y_test, 
            weights, biases, act_indices
        )
        losses = LossFunctions.get_all_losses(self.y_test, logits)
        self.evaluated_count += 1
        return logits, losses
    
    def evaluate_batch(self, networks):
        """Evaluate a batch of networks."""
        return [self.evaluate(w, b, a) for w, b, a in networks]
    
    def get_stats(self):
        """Return evaluation statistics."""
        return {
            'total_evaluated': self.evaluated_count,
            'test_samples': len(self.X_test)
        }


# ═══════════════════════════════════════════════
# DATASET MANAGER
# ═══════════════════════════════════════════════
class DatasetManager:
    @staticmethod
    def load_dataset(name, n_samples=1500, noise=0.25, random_state=42):
        if name == "make_moons":
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        elif name == "make_circles":
            X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
        elif name == "make_blobs":
            X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=noise, random_state=random_state)
        elif name == "make_classification":
            X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2,
                                      n_redundant=0, n_classes=2, random_state=random_state)
        else:
            raise ValueError(f"Unknown dataset: {name}")
        return X, y
    
    @staticmethod
    def get_available_datasets():
        return ["make_moons", "make_circles", "make_blobs", "make_classification"]


# ═══════════════════════════════════════════════
# EXPERIMENT WORKER (Orchestrates independent components)
# ═══════════════════════════════════════════════
class ExperimentWorker(QObject):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(dict)
    error = Signal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._is_running = True
    
    def stop(self):
        self._is_running = False
    
    def run(self):
        try:
            # ── STEP 1: Initialize independent components ──
            self.status.emit("Initializing independent components...")
            
            # Image Generator - completely independent
            image_gen = ImageGenerator(
                height=self.config['height'],
                width=self.config['width'],
                seed=None
            )
            
            # Load dataset (separate from image generation)
            self.status.emit(f"Loading {self.config['dataset']} dataset...")
            X_full, y_full = DatasetManager.load_dataset(
                self.config['dataset'],
                n_samples=self.config.get('n_samples', 1500),
                noise=self.config.get('noise', 0.25)
            )
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_full, y_full, test_size=0.3, random_state=42
            )
            
            scaler = MinMaxScaler() if self.config.get('scaler', 'minmax') == 'minmax' else StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Network Builder - only consumes images
            bias_config = BiasConfig(
                mapping=self.config['bias_mapping'],
                spatial=self.config['spatial'],
                depth_scale=self.config['depth_scale'],
                act_aware=self.config['act_aware']
            )
            net_builder = NetworkBuilder(bias_config)
            
            # Network Evaluator - only consumes networks
            evaluator = NetworkEvaluator(X_test, y_test)
            
            # ── STEP 2: Run experiment with decoupled pipeline ──
            all_losses = {name: [] for name in ["BCE", "Focal (γ=2)", "Focal (γ=5)", 
                                                 "Hinge", "Huber (δ=1)", "Log-Cosh"]}
            
            network_stats = {
                'weight_mean': [], 'weight_std': [],
                'bias_mean': [], 'bias_std': [],
                'act_diversity': []
            }
            
            best_networks = {name: {'loss': float('inf'), 'idx': -1, 'stats': None, 
                                   'hsv': None, 'weights': None, 'biases': None, 'act_indices': None} 
                            for name in all_losses}
            
            start_time = time.time()
            N_IMAGES = self.config['trials']
            
            for i in range(N_IMAGES):
                if not self._is_running:
                    self.status.emit("Experiment stopped by user")
                    return
                
                # ── PHASE 1: Generate image (completely random, no network knowledge) ──
                H_rand, S_rand, V_rand = image_gen.generate_single()
                
                # ── PHASE 2: Build network from image (consumes image, no generation) ──
                weights, biases, act_indices = net_builder.build_from_image(
                    H_rand, S_rand, V_rand
                )
                
                # ── PHASE 3: Evaluate network (consumes network, no building/generation) ──
                logits, losses = evaluator.evaluate(weights, biases, act_indices)
                
                # ── PHASE 4: Track results ──
                for name, val in losses.items():
                    all_losses[name].append(val)
                    if val < best_networks[name]['loss']:
                        best_networks[name]['loss'] = val
                        best_networks[name]['idx'] = i
                        best_networks[name]['stats'] = {
                            'weight_mean': np.mean(weights),
                            'weight_std': np.std(weights),
                            'bias_mean': np.mean(biases),
                            'bias_std': np.std(biases),
                            'act_diversity': len(np.unique(act_indices))
                        }
                        best_networks[name]['hsv'] = (H_rand.copy(), S_rand.copy(), V_rand.copy())
                        best_networks[name]['weights'] = weights.copy()
                        best_networks[name]['biases'] = biases.copy()
                        best_networks[name]['act_indices'] = act_indices.copy()
                
                network_stats['weight_mean'].append(np.mean(weights))
                network_stats['weight_std'].append(np.std(weights))
                network_stats['bias_mean'].append(np.mean(biases))
                network_stats['bias_std'].append(np.std(biases))
                network_stats['act_diversity'].append(len(np.unique(act_indices)))
                
                self.progress.emit(int((i + 1) / N_IMAGES * 100))
                
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (N_IMAGES - i - 1) / rate if rate > 0 else 0
                self.status.emit(f"Trial {i+1}/{N_IMAGES} | {rate:.1f} trials/s | ETA: {eta:.0f}s")
            
            # Convert to numpy arrays
            for name in all_losses:
                all_losses[name] = np.array(all_losses[name])
            for key in network_stats:
                network_stats[key] = np.array(network_stats[key])
            
            results = {
                'all_losses': all_losses,
                'network_stats': network_stats,
                'best_networks': best_networks,
                'config': self.config,
                'timestamp': datetime.now().isoformat(),
                'elapsed_time': time.time() - start_time,
                'component_stats': {
                    'image_generator': image_gen.get_stats(),
                    'network_builder': net_builder.get_stats(),
                    'network_evaluator': evaluator.get_stats()
                }
            }
            
            self.status.emit("Experiment complete!")
            self.finished.emit(results)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)
            self.status.emit("Experiment failed!")


# ═══════════════════════════════════════════════
# ENHANCED SPINBOX WIDGETS
# ═══════════════════════════════════════════════
class EnhancedSpinBox(QSpinBox):
    """Professional spinbox with adaptive stepping, large ranges, and modifier key support."""
    
    def __init__(self, parent=None, default=10, min_val=1, max_val=100000, 
                 step=1, suffix="", prefix="", tooltip=""):
        super().__init__(parent)
        self._default = default
        self._base_step = step
        self._suffix = suffix
        self._prefix = prefix
        
        self.setRange(min_val, max_val)
        self.setValue(default)
        self.setSingleStep(step)
        self.setToolTip(tooltip)
        self.setKeyboardTracking(True)
        self.setButtonSymbols(QSpinBox.UpDownArrows)
        
        if suffix:
            self.setSuffix(f" {suffix}")
        if prefix:
            self.setPrefix(f"{prefix} ")
        
        self.lineEdit().setAlignment(Qt.AlignRight)
        
        self.setStyleSheet("""
            QSpinBox {
                padding: 8px 12px; border: 2px solid #bdc3c7; border-radius: 6px;
                background-color: white; font-size: 13px; font-weight: bold;
                min-height: 20px; min-width: 80px;
            }
            QSpinBox:focus { border-color: #2A9D8F; background-color: #f0fffe; }
            QSpinBox:hover { border-color: #2A9D8F; }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 24px; border: none; background-color: #ecf0f1; border-radius: 3px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover { background-color: #2A9D8F; }
            QSpinBox::up-arrow {
                image: none; border-left: 5px solid transparent; border-right: 5px solid transparent;
                border-bottom: 6px solid #264653; width: 0; height: 0;
            }
            QSpinBox::down-arrow {
                image: none; border-left: 5px solid transparent; border-right: 5px solid transparent;
                border-top: 6px solid #264653; width: 0; height: 0;
            }
            QSpinBox::up-button:hover QSpinBox::up-arrow { border-bottom-color: white; }
            QSpinBox::down-button:hover QSpinBox::down-arrow { border-top-color: white; }
        """)
    
    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.lineEdit().selectAll()
    
    def wheelEvent(self, event):
        """Adaptive stepping based on modifier keys and current value magnitude."""
        modifiers = event.modifiers()
        delta = event.angleDelta().y()
        direction = 1 if delta > 0 else -1
        
        if modifiers & Qt.ShiftModifier:
            step = self._base_step * 100
        elif modifiers & Qt.ControlModifier:
            step = self._base_step
        else:
            current = self.value()
            if current >= 10000:
                step = self._base_step * 100
            elif current >= 1000:
                step = self._base_step * 10
            elif current >= 100:
                step = self._base_step * 5
            else:
                step = self._base_step
        
        new_val = self.value() + direction * step
        self.setValue(max(self.minimum(), min(self.maximum(), new_val)))
    
    def keyPressEvent(self, event):
        """Allow Ctrl+Up/Down for fine stepping, Shift+Up/Down for coarse."""
        modifiers = event.modifiers()
        
        if event.key() in (Qt.Key_Up, Qt.Key_Down):
            direction = 1 if event.key() == Qt.Key_Up else -1
            
            if modifiers & Qt.ShiftModifier:
                step = self._base_step * 100
            elif modifiers & Qt.ControlModifier:
                step = self._base_step
            else:
                current = self.value()
                if current >= 10000:
                    step = self._base_step * 100
                elif current >= 1000:
                    step = self._base_step * 10
                elif current >= 100:
                    step = self._base_step * 5
                else:
                    step = self._base_step
            
            new_val = self.value() + direction * step
            self.setValue(max(self.minimum(), min(self.maximum(), new_val)))
            return
        
        super().keyPressEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        """Double-click to reset to default value."""
        self.setValue(self._default)
        self.lineEdit().selectAll()
    
    def reset(self):
        """Reset to default value."""
        self.setValue(self._default)


class EnhancedDoubleSpinBox(QDoubleSpinBox):
    """Professional double spinbox with adaptive stepping and better precision."""
    
    def __init__(self, parent=None, default=0.25, min_val=0.0, max_val=1000.0,
                 step=0.01, decimals=3, suffix="", tooltip=""):
        super().__init__(parent)
        self._default = default
        self._base_step = step
        
        self.setRange(min_val, max_val)
        self.setValue(default)
        self.setSingleStep(step)
        self.setDecimals(decimals)
        self.setToolTip(tooltip)
        self.setKeyboardTracking(True)
        
        if suffix:
            self.setSuffix(f" {suffix}")
        
        self.lineEdit().setAlignment(Qt.AlignRight)
        
        self.setStyleSheet("""
            QDoubleSpinBox {
                padding: 8px 12px; border: 2px solid #bdc3c7; border-radius: 6px;
                background-color: white; font-size: 13px; font-weight: bold;
                min-height: 20px; min-width: 80px;
            }
            QDoubleSpinBox:focus { border-color: #2A9D8F; background-color: #f0fffe; }
            QDoubleSpinBox:hover { border-color: #2A9D8F; }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 24px; border: none; background-color: #ecf0f1; border-radius: 3px;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background-color: #2A9D8F; }
            QDoubleSpinBox::up-arrow {
                image: none; border-left: 5px solid transparent; border-right: 5px solid transparent;
                border-bottom: 6px solid #264653; width: 0; height: 0;
            }
            QDoubleSpinBox::down-arrow {
                image: none; border-left: 5px solid transparent; border-right: 5px solid transparent;
                border-top: 6px solid #264653; width: 0; height: 0;
            }
        """)
    
    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.lineEdit().selectAll()
    
    def wheelEvent(self, event):
        modifiers = event.modifiers()
        delta = event.angleDelta().y()
        direction = 1 if delta > 0 else -1
        
        if modifiers & Qt.ShiftModifier:
            step = self._base_step * 100
        elif modifiers & Qt.ControlModifier:
            step = self._base_step * 0.1
        else:
            step = self._base_step * 10
        
        new_val = self.value() + direction * step
        self.setValue(max(self.minimum(), min(self.maximum(), new_val)))
    
    def mouseDoubleClickEvent(self, event):
        self.setValue(self._default)
        self.lineEdit().selectAll()
    
    def reset(self):
        self.setValue(self._default)


class ConfigRow(QWidget):
    """A labeled configuration row with optional reset button and description."""
    
    def __init__(self, label_text, widget, description="", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        
        label = QLabel(label_text)
        label.setMinimumWidth(180)
        label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(label)
        
        layout.addWidget(widget, 1)
        
        if hasattr(widget, 'reset'):
            reset_btn = QPushButton("↺")
            reset_btn.setFixedSize(28, 28)
            reset_btn.setToolTip("Reset to default")
            reset_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ecf0f1; border: 1px solid #bdc3c7;
                    border-radius: 14px; font-size: 14px; font-weight: bold; color: #264653;
                }
                QPushButton:hover { background-color: #E63946; color: white; border-color: #E63946; }
            """)
            reset_btn.clicked.connect(widget.reset)
            layout.addWidget(reset_btn)
        
        if description:
            self.setToolTip(description)


# ═══════════════════════════════════════════════
# PROFESSIONAL PLOT GENERATORS
# ═══════════════════════════════════════════════
class PlotGenerator:
    COLORS = ['#E63946', '#F4A261', '#E9C46A', '#2A9D8F', '#264653', '#A8DADC']
    
    @staticmethod
    def create_multi_loss_dashboard(results, theme='light'):
        all_losses = results['all_losses']
        best_networks = results['best_networks']
        N_IMAGES = len(all_losses['BCE'])
        loss_names = list(all_losses.keys())
        theme_colors = ThemeManager.LIGHT_THEME if theme == 'light' else ThemeManager.DARK_THEME
        
        fig = Figure(figsize=(24, 15), facecolor=theme_colors['plot_bg'])
        fig.suptitle("🎯 Random Image-Derived Networks: Multi-Loss Function Analysis",
                    fontsize=22, fontweight="bold", y=0.98, color=theme_colors['fg'])
        
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
        
        # Plot 1: Loss Trajectories
        ax1 = fig.add_subplot(gs[0, :2])
        window = max(5, N_IMAGES // 15)
        for idx, name in enumerate(loss_names):
            vals = all_losses[name]
            rolling_mean = np.convolve(vals, np.ones(window)/window, mode='valid')
            rolling_std = np.array([np.std(vals[i:i+window]) for i in range(len(rolling_mean))])
            x_range = range(window-1, window-1 + len(rolling_mean))
            
            ax1.plot(x_range, rolling_mean, color=PlotGenerator.COLORS[idx], 
                    linewidth=2.5, label=f'{name} (μ={np.mean(vals):.3f})', alpha=0.9)
            ax1.fill_between(x_range, rolling_mean - rolling_std, rolling_mean + rolling_std, 
                            color=PlotGenerator.COLORS[idx], alpha=0.15)
        
        ax1.set_title("Loss Trajectories with ±1σ Confidence Bands", 
                     fontsize=15, fontweight="bold", pad=12, color=theme_colors['fg'])
        ax1.set_xlabel("Trial Number", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Loss Value", fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10, ncol=2, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(0, N_IMAGES)
        ax1.set_facecolor(theme_colors['plot_bg'])
        
        # Plot 2: Violin Plots
        ax2 = fig.add_subplot(gs[0, 2])
        violin_data = [all_losses[name] for name in loss_names]
        parts = ax2.violinplot(violin_data, showmeans=True, showmedians=True, showextrema=True)
        
        for pc, color in zip(parts['bodies'], PlotGenerator.COLORS):
            pc.set_facecolor(color)
            pc.set_alpha(0.75)
            pc.set_edgecolor(theme_colors['fg'])
            pc.set_linewidth(1.5)
        
        if parts['cmeans']:
            parts['cmeans'].set_color(theme_colors['danger'])
            parts['cmeans'].set_linewidth(2.5)
        if parts['cmedians']:
            parts['cmedians'].set_color('white')
            parts['cmedians'].set_linewidth(2.5)
        
        ax2.set_xticks(range(1, len(loss_names) + 1))
        ax2.set_xticklabels(loss_names, rotation=45, ha='right', fontsize=10)
        ax2.set_title("Loss Distribution (Violin)", fontsize=15, fontweight="bold", pad=12, color=theme_colors['fg'])
        ax2.set_ylabel("Loss Value", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.set_facecolor(theme_colors['plot_bg'])
        
        # Plot 3: Correlation Heatmap
        ax3 = fig.add_subplot(gs[1, 0])
        corr_matrix = np.zeros((len(loss_names), len(loss_names)))
        for i, name1 in enumerate(loss_names):
            for j, name2 in enumerate(loss_names):
                corr_matrix[i, j] = np.corrcoef(all_losses[name1], all_losses[name2])[0, 1]
        
        im = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax3.set_xticks(range(len(loss_names)))
        ax3.set_yticks(range(len(loss_names)))
        ax3.set_xticklabels([name[:8] for name in loss_names], rotation=45, ha='right', fontsize=9)
        ax3.set_yticklabels([name[:8] for name in loss_names], fontsize=9)
        
        for i in range(len(loss_names)):
            for j in range(len(loss_names)):
                ax3.text(j, i, f'{corr_matrix[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=9, fontweight="bold")
        
        ax3.set_title("Loss Correlation Matrix", fontsize=15, fontweight="bold", pad=12, color=theme_colors['fg'])
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
        # Plot 4: Parallel Coordinates
        ax4 = fig.add_subplot(gs[1, 1])
        normalized_losses = np.zeros((N_IMAGES, len(loss_names)))
        for idx, name in enumerate(loss_names):
            vals = all_losses[name]
            normalized_losses[:, idx] = (vals - np.min(vals)) / (np.max(vals) - np.min(vals) + 1e-8)
        
        top_indices = np.argsort(all_losses['BCE'])[:20]
        for i in top_indices:
            ax4.plot(range(len(loss_names)), normalized_losses[i], 
                    color=theme_colors['fg'], alpha=0.2, linewidth=1)
        
        best_idx = np.argmin(all_losses['BCE'])
        ax4.plot(range(len(loss_names)), normalized_losses[best_idx], 
                color=theme_colors['danger'], linewidth=3.5, label='Best Network', zorder=10,
                marker='o', markersize=8)
        
        ax4.set_xticks(range(len(loss_names)))
        ax4.set_xticklabels([name[:8] for name in loss_names], rotation=45, ha='right', fontsize=9)
        ax4.set_ylabel("Normalized Loss (0=best, 1=worst)", fontsize=11, fontweight='bold')
        ax4.set_title("Top 20 Networks (Parallel Coordinates)", fontsize=15, fontweight="bold", pad=12, color=theme_colors['fg'])
        ax4.legend(fontsize=10, loc='upper right')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_ylim(-0.05, 1.05)
        ax4.set_facecolor(theme_colors['plot_bg'])
        
        # Plot 5: Radar Chart
        ax5 = fig.add_subplot(gs[1, 2], polar=True)
        angles = np.linspace(0, 2 * np.pi, len(loss_names), endpoint=False).tolist()
        angles += angles[:1]
        
        mean_vals = [np.mean(all_losses[name]) for name in loss_names]
        median_vals = [np.median(all_losses[name]) for name in loss_names]
        best_vals = [np.min(all_losses[name]) for name in loss_names]
        
        max_val = max(max(mean_vals), max(median_vals), max(best_vals))
        mean_norm = [v / max_val for v in mean_vals] + [mean_vals[0] / max_val]
        median_norm = [v / max_val for v in median_vals] + [median_vals[0] / max_val]
        best_norm = [v / max_val for v in best_vals] + [best_vals[0] / max_val]
        
        ax5.plot(angles, mean_norm, 'o-', linewidth=2.5, label='Mean', color=theme_colors['danger'], markersize=6)
        ax5.fill(angles, mean_norm, alpha=0.15, color=theme_colors['danger'])
        ax5.plot(angles, median_norm, 's-', linewidth=2.5, label='Median', color=theme_colors['fg'], markersize=6)
        ax5.fill(angles, median_norm, alpha=0.15, color=theme_colors['fg'])
        ax5.plot(angles, best_norm, '^-', linewidth=2.5, label='Best', color=theme_colors['success'], markersize=6)
        ax5.fill(angles, best_norm, alpha=0.15, color=theme_colors['success'])
        
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels([name[:8] for name in loss_names], fontsize=9)
        ax5.set_title("Loss Profile Radar\n(Normalized)", fontsize=15, fontweight="bold", pad=25, color=theme_colors['fg'])
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax5.set_facecolor(theme_colors['plot_bg'])
        
        # Plot 6: Best Network Statistics
        ax6 = fig.add_subplot(gs[2, :])
        metrics = ['Weight Mean', 'Weight Std', 'Bias Mean', 'Bias Std', 'Act Diversity']
        x_pos = np.arange(len(metrics))
        width = 0.13
        
        for idx, name in enumerate(loss_names):
            stats = best_networks[name]['stats']
            values = [
                stats['weight_mean'], stats['weight_std'],
                stats['bias_mean'], stats['bias_std'],
                stats['act_diversity'] / 24.0
            ]
            ax6.bar(x_pos + idx * width, values, width, label=name, 
                   color=PlotGenerator.COLORS[idx], alpha=0.85, edgecolor=theme_colors['fg'], linewidth=0.8)
        
        ax6.set_xlabel('Network Metric', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax6.set_title('Best Network Statistics per Loss Function', 
                     fontsize=15, fontweight="bold", pad=12, color=theme_colors['fg'])
        ax6.set_xticks(x_pos + width * 2.5)
        ax6.set_xticklabels(metrics, fontsize=11)
        ax6.legend(loc='upper right', fontsize=9, ncol=3)
        ax6.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax6.set_facecolor(theme_colors['plot_bg'])
        
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
    
    @staticmethod
    def create_architecture_analysis(results, theme='light'):
        all_losses = results['all_losses']
        network_stats = results['network_stats']
        loss_names = list(all_losses.keys())
        theme_colors = ThemeManager.LIGHT_THEME if theme == 'light' else ThemeManager.DARK_THEME
        
        fig = Figure(figsize=(24, 16), facecolor=theme_colors['plot_bg'])
        fig.suptitle("🏗️ Network Architecture Analysis: How Random HSV Maps to Performance",
                    fontsize=22, fontweight="bold", y=0.98, color=theme_colors['fg'])
        
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
        
        # Plot 1: Loss vs Weight Statistics
        ax1 = fig.add_subplot(gs[0, 0])
        scatter = ax1.scatter(network_stats['weight_mean'], all_losses['BCE'], 
                             c=all_losses['BCE'], cmap='viridis', alpha=0.7, 
                             s=60, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Mean Weight (Saturation)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('BCE Loss', fontsize=11, fontweight='bold')
        ax1.set_title('Loss vs Mean Weight', fontsize=14, fontweight="bold", pad=10, color=theme_colors['fg'])
        ax1.grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(scatter, ax=ax1, label='BCE Loss')
        ax1.set_facecolor(theme_colors['plot_bg'])
        
        # Plot 2: Loss vs Bias Statistics
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(network_stats['bias_std'], all_losses['BCE'], 
                             c=network_stats['weight_std'], cmap='plasma', alpha=0.7, 
                             s=60, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Bias Std (Value Channel)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('BCE Loss', fontsize=11, fontweight='bold')
        ax2.set_title('Loss vs Bias Variance', fontsize=14, fontweight="bold", pad=10, color=theme_colors['fg'])
        ax2.grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(scatter, ax=ax2, label='Weight Std')
        ax2.set_facecolor(theme_colors['plot_bg'])
        
        # Plot 3: Loss vs Activation Diversity
        ax3 = fig.add_subplot(gs[0, 2])
        scatter = ax3.scatter(network_stats['act_diversity'], all_losses['BCE'], 
                             c=all_losses['Hinge'], cmap='coolwarm', alpha=0.7, 
                             s=60, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Activation Function Diversity (0-24)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('BCE Loss', fontsize=11, fontweight='bold')
        ax3.set_title('Loss vs Activation Diversity', fontsize=14, fontweight="bold", pad=10, color=theme_colors['fg'])
        ax3.grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(scatter, ax=ax3, label='Hinge Loss')
        ax3.set_facecolor(theme_colors['plot_bg'])
        
        # Plot 4: 2D Loss Landscape
        ax4 = fig.add_subplot(gs[1, :2])
        scatter = ax4.scatter(network_stats['weight_mean'], network_stats['bias_mean'], 
                             c=all_losses['BCE'], cmap='hot', alpha=0.75, 
                             s=90, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('Mean Weight (Saturation)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Mean Bias (Value)', fontsize=12, fontweight='bold')
        ax4.set_title('2D Loss Landscape: Weight-Bias Space', fontsize=15, fontweight="bold", pad=10, color=theme_colors['fg'])
        ax4.grid(True, alpha=0.3, linestyle='--')
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('BCE Loss', fontsize=11)
        
        best_idx = np.argmin(all_losses['BCE'])
        ax4.scatter(network_stats['weight_mean'][best_idx], network_stats['bias_mean'][best_idx], 
                   c='lime', s=400, marker='*', edgecolors='black', linewidth=2.5, 
                   zorder=10, label='Best Network')
        ax4.legend(fontsize=11, loc='upper right')
        ax4.set_facecolor(theme_colors['plot_bg'])
        
        # Plot 5: Loss Distribution by Activation Diversity (FIXED for matplotlib 3.9+)
        ax5 = fig.add_subplot(gs[1, 2])
        diversity_bins = [(0, 8, 'Low (1-8)'), (8, 16, 'Medium (9-16)'), (16, 24, 'High (17-24)')]
        
        box_data = []
        box_labels = []
        for low, high, label in diversity_bins:
            mask = (network_stats['act_diversity'] >= low) & (network_stats['act_diversity'] <= high)
            if np.sum(mask) > 0:
                box_data.append(all_losses['BCE'][mask])
                box_labels.append(label)
        
        if len(box_data) > 0:
            bp = ax5.boxplot(box_data, patch_artist=True, showmeans=True)
            colors_list = [theme_colors['fg'], theme_colors['success'], theme_colors['warning']]
            for patch, color in zip(bp['boxes'], colors_list[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
                patch.set_edgecolor('black')
            
            # Set tick labels separately (matplotlib 3.9+ compatible)
            ax5.set_xticks(range(1, len(box_labels) + 1))
            ax5.set_xticklabels(box_labels)
        
        ax5.set_xlabel('Activation Diversity', fontsize=11, fontweight='bold')
        ax5.set_ylabel('BCE Loss', fontsize=11, fontweight='bold')
        ax5.set_title('Loss by Activation Diversity', fontsize=14, fontweight="bold", pad=10, color=theme_colors['fg'])
        ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax5.set_facecolor(theme_colors['plot_bg'])
        
        # Plot 6: Correlation Heatmap
        ax6 = fig.add_subplot(gs[2, :2])
        stats_names = ['Weight Mean', 'Weight Std', 'Bias Mean', 'Bias Std', 'Act Diversity']
        stats_array = np.array([
            network_stats['weight_mean'], network_stats['weight_std'],
            network_stats['bias_mean'], network_stats['bias_std'],
            network_stats['act_diversity']
        ])
        
        corr_matrix = np.zeros((len(stats_names), len(loss_names)))
        for i in range(len(stats_names)):
            for j in range(len(loss_names)):
                corr_matrix[i, j] = np.corrcoef(stats_array[i], all_losses[loss_names[j]])[0, 1]
        
        im = ax6.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax6.set_xticks(range(len(loss_names)))
        ax6.set_yticks(range(len(stats_names)))
        ax6.set_xticklabels(loss_names, rotation=45, ha='right', fontsize=10)
        ax6.set_yticklabels(stats_names, fontsize=11)
        
        for i in range(len(stats_names)):
            for j in range(len(loss_names)):
                ax6.text(j, i, f'{corr_matrix[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=10, fontweight="bold")
        
        ax6.set_title('Correlation: Network Statistics vs Loss Functions', 
                     fontsize=15, fontweight="bold", pad=10, color=theme_colors['fg'])
        plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
        
        # Plot 7: Summary Table
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        table_data = []
        for name in loss_names:
            vals = all_losses[name]
            table_data.append([
                name, f'{np.mean(vals):.4f}', f'{np.std(vals):.4f}',
                f'{np.min(vals):.4f}', f'{np.max(vals):.4f}'
            ])
        
        table = ax7.table(cellText=table_data, 
                         colLabels=['Loss Function', 'Mean', 'Std', 'Min', 'Max'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)
        
        for i in range(5):
            table[(0, i)].set_facecolor(theme_colors['fg'])
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        for i in range(1, len(loss_names) + 1):
            for j in range(5):
                table[(i, j)].set_facecolor(theme_colors['hover'] if i % 2 == 0 else theme_colors['card'])
        
        ax7.set_title('Summary Statistics', fontsize=15, fontweight="bold", pad=20, color=theme_colors['fg'])
        
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
    
    @staticmethod
    def create_temporal_analysis(results, theme='light'):
        all_losses = results['all_losses']
        best_networks = results['best_networks']
        N_IMAGES = len(all_losses['BCE'])
        loss_names = list(all_losses.keys())
        theme_colors = ThemeManager.LIGHT_THEME if theme == 'light' else ThemeManager.DARK_THEME
        
        fig = Figure(figsize=(24, 14), facecolor=theme_colors['plot_bg'])
        fig.suptitle("📈 Temporal Analysis: How Random Networks Discover Good Solutions Over Time",
                    fontsize=22, fontweight="bold", y=0.98, color=theme_colors['fg'])
        
        gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
        
        # Plot 1: Running Best Loss
        ax1 = fig.add_subplot(gs[0, :2])
        for idx, name in enumerate(loss_names):
            vals = all_losses[name]
            running_best = np.minimum.accumulate(vals)
            ax1.plot(range(N_IMAGES), running_best, color=PlotGenerator.COLORS[idx], 
                    linewidth=3, label=f'{name} (final={running_best[-1]:.3f})', alpha=0.9)
        
        ax1.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Running Best Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Discovery of Best Networks Over Time', 
                     fontsize=15, fontweight="bold", pad=10, color=theme_colors['fg'])
        ax1.legend(loc='upper right', fontsize=10, ncol=2)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(0, N_IMAGES)
        ax1.set_facecolor(theme_colors['plot_bg'])
        
        for idx, name in enumerate(loss_names):
            best_trial = np.argmin(all_losses[name])
            ax1.scatter(best_trial, all_losses[name][best_trial], 
                       color=PlotGenerator.COLORS[idx], s=200, marker='*', 
                       edgecolors='black', linewidth=1.5, zorder=10)
        
        # Plot 2: Improvement Rate
        ax2 = fig.add_subplot(gs[0, 2])
        improvement_rates = []
        for name in loss_names:
            vals = all_losses[name]
            running_best = np.minimum.accumulate(vals)
            rate = (running_best[0] - running_best[-1]) / N_IMAGES
            improvement_rates.append(rate)
        
        bars = ax2.barh(range(len(loss_names)), improvement_rates, 
                       color=PlotGenerator.COLORS, alpha=0.85, edgecolor='black', linewidth=1)
        ax2.set_yticks(range(len(loss_names)))
        ax2.set_yticklabels(loss_names, fontsize=10)
        ax2.set_xlabel('Improvement Rate (loss/trial)', fontsize=11, fontweight='bold')
        ax2.set_title('Speed of Best Network Discovery', 
                     fontsize=14, fontweight="bold", pad=10, color=theme_colors['fg'])
        ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        for i, (bar, rate) in enumerate(zip(bars, improvement_rates)):
            ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{rate:.5f}', va='center', fontsize=9, fontweight='bold')
        
        ax2.set_facecolor(theme_colors['plot_bg'])
        
        # Plot 3: Rolling Statistics
        ax3 = fig.add_subplot(gs[1, 0])
        window = max(5, N_IMAGES // 10)
        for idx, name in enumerate(loss_names[:3]):
            vals = all_losses[name]
            rolling_mean = np.convolve(vals, np.ones(window)/window, mode='valid')
            rolling_std = np.array([np.std(vals[i:i+window]) for i in range(len(rolling_mean))])
            x_range = range(window-1, window-1 + len(rolling_mean))
            
            ax3.plot(x_range, rolling_mean, color=PlotGenerator.COLORS[idx], 
                    linewidth=2.5, label=f'{name} mean')
            ax3.fill_between(x_range, rolling_mean - rolling_std, rolling_mean + rolling_std, 
                            color=PlotGenerator.COLORS[idx], alpha=0.25)
        
        ax3.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Loss Value', fontsize=11, fontweight='bold')
        ax3.set_title(f'Rolling Statistics (window={window})', 
                     fontsize=14, fontweight="bold", pad=10, color=theme_colors['fg'])
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_facecolor(theme_colors['plot_bg'])
        
        # Plot 4: Loss Variance Over Time
        ax4 = fig.add_subplot(gs[1, 1])
        window = max(5, N_IMAGES // 10)
        for idx, name in enumerate(loss_names):
            vals = all_losses[name]
            rolling_var = np.array([np.var(vals[i:i+window]) for i in range(N_IMAGES - window + 1)])
            ax4.plot(range(window-1, window-1 + len(rolling_var)), rolling_var, 
                    color=PlotGenerator.COLORS[idx], linewidth=2, label=name, alpha=0.85)
        
        ax4.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Variance', fontsize=11, fontweight='bold')
        ax4.set_title('Loss Variance Over Time (Stability)', 
                     fontsize=14, fontweight="bold", pad=10, color=theme_colors['fg'])
        ax4.legend(fontsize=8, ncol=2, loc='upper right')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_facecolor(theme_colors['plot_bg'])
        
        # Plot 5: Network Fingerprints
        ax5 = fig.add_subplot(gs[1, 2])
        metrics = ['Weight Mean', 'Weight Std', 'Bias Mean', 'Bias Std', 'Act Diversity']
        x_pos = np.arange(len(metrics))
        
        for idx, name in enumerate(loss_names):
            stats = best_networks[name]['stats']
            values = [
                stats['weight_mean'], stats['weight_std'],
                stats['bias_mean'], stats['bias_std'],
                stats['act_diversity'] / 24.0
            ]
            ax5.plot(x_pos, values, 'o-', color=PlotGenerator.COLORS[idx], 
                    linewidth=2.5, markersize=9, label=name, alpha=0.85)
        
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(metrics, rotation=45, ha='right', fontsize=9)
        ax5.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax5.set_title('Best Network "Fingerprints"', 
                     fontsize=14, fontweight="bold", pad=10, color=theme_colors['fg'])
        ax5.legend(fontsize=7, ncol=2, loc='upper right')
        ax5.grid(True, alpha=0.3, linestyle='--')
        ax5.set_facecolor(theme_colors['plot_bg'])
        
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
    
    @staticmethod
    def create_network_inspector(best_networks, loss_name, theme='light'):
        net = best_networks[loss_name]
        weights = net['weights']
        biases = net['biases']
        act_indices = net['act_indices']
        H, S, V = net['hsv']
        
        theme_colors = ThemeManager.LIGHT_THEME if theme == 'light' else ThemeManager.DARK_THEME
        
        fig = Figure(figsize=(20, 14), facecolor=theme_colors['plot_bg'])
        fig.suptitle(
            f"🔬 Network Inspector: Best {loss_name} Network (Trial #{net['idx']}, Loss={net['loss']:.4f})",
            fontsize=18, fontweight="bold", y=0.98, color=theme_colors['fg']
        )
        
        gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
        
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(H, cmap='hsv', vmin=0, vmax=360, aspect='auto')
        ax1.set_title('Hue Map (Activation Selection)', fontsize=12, fontweight='bold', color=theme_colors['fg'])
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(S, cmap='Reds', vmin=0, vmax=1, aspect='auto')
        ax2.set_title('Saturation Map (Weights)', fontsize=12, fontweight='bold', color=theme_colors['fg'])
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(V, cmap='gray', vmin=0, vmax=1, aspect='auto')
        ax3.set_title('Value Map (Biases)', fontsize=12, fontweight='bold', color=theme_colors['fg'])
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(weights.flatten(), bins=30, color=theme_colors['fg'], alpha=0.75, edgecolor='black')
        ax4.set_xlabel('Weight Value', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax4.set_title('Weight Distribution', fontsize=12, fontweight='bold', color=theme_colors['fg'])
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_facecolor(theme_colors['plot_bg'])
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(biases.flatten(), bins=30, color=theme_colors['danger'], alpha=0.75, edgecolor='black')
        ax5.set_xlabel('Bias Value', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax5.set_title('Bias Distribution', fontsize=12, fontweight='bold', color=theme_colors['fg'])
        ax5.grid(True, alpha=0.3, linestyle='--')
        ax5.set_facecolor(theme_colors['plot_bg'])
        
        ax6 = fig.add_subplot(gs[1, 2])
        unique, counts = np.unique(act_indices, return_counts=True)
        act_names = [ACTIVATION_MAP[i][2] for i in unique]
        colors = [PlotGenerator.COLORS[i % len(PlotGenerator.COLORS)] for i in range(len(unique))]
        
        ax6.bar(range(len(unique)), counts, color=colors, alpha=0.85, edgecolor='black')
        ax6.set_xticks(range(len(unique)))
        ax6.set_xticklabels(act_names, rotation=45, ha='right', fontsize=9)
        ax6.set_ylabel('Usage Count', fontsize=11, fontweight='bold')
        ax6.set_title('Activation Function Usage', fontsize=12, fontweight='bold', color=theme_colors['fg'])
        ax6.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax6.set_facecolor(theme_colors['plot_bg'])
        
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig


# ═══════════════════════════════════════════════
# GUI TABS
# ═══════════════════════════════════════════════
class SetupTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()
    
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # ── Left Panel: Configuration ──
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Dataset Group
        dataset_group = QGroupBox("📦 Dataset Configuration")
        dataset_form = QVBoxLayout()
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(DatasetManager.get_available_datasets())
        self.dataset_combo.setToolTip("Select the dataset to use for evaluation")
        dataset_form.addWidget(ConfigRow("Dataset:", self.dataset_combo, "Choose which sklearn dataset to evaluate on"))
        
        self.n_samples_spin = EnhancedSpinBox(
            default=1500, min_val=200, max_val=50000, step=100,
            tooltip="Number of samples in the dataset (200 - 50,000)"
        )
        dataset_form.addWidget(ConfigRow("Samples:", self.n_samples_spin,
                                         "Scroll: adaptive | Ctrl+scroll: fine | Shift+scroll: coarse | Dbl-click: reset"))
        
        self.noise_spin = EnhancedDoubleSpinBox(
            default=0.25, min_val=0.0, max_val=2.0, step=0.01, decimals=3,
            tooltip="Noise level in the dataset (0.0 - 2.0)"
        )
        dataset_form.addWidget(ConfigRow("Noise:", self.noise_spin,
                                         "Ctrl+scroll: fine (0.001) | Scroll: normal (0.1) | Shift+scroll: coarse (1.0)"))
        
        self.scaler_combo = QComboBox()
        self.scaler_combo.addItems(['minmax', 'standard'])
        self.scaler_combo.setToolTip("Feature scaling method")
        dataset_form.addWidget(ConfigRow("Scaler:", self.scaler_combo))
        
        dataset_group.setLayout(dataset_form)
        left_layout.addWidget(dataset_group)
        
        # Network Group
        network_group = QGroupBox("🧠 Network Architecture")
        network_form = QVBoxLayout()
        
        self.height_spin = EnhancedSpinBox(
            default=10, min_val=2, max_val=512, step=1,
            tooltip="Number of neurons per layer (2 - 512)"
        )
        network_form.addWidget(ConfigRow("Height (Neurons):", self.height_spin,
                                         "Number of neurons in each layer. Dbl-click to reset to 10."))
        
        self.width_spin = EnhancedSpinBox(
            default=8, min_val=2, max_val=256, step=1,
            tooltip="Number of layers (2 - 256)"
        )
        network_form.addWidget(ConfigRow("Width (Layers):", self.width_spin,
                                         "Number of hidden layers. Dbl-click to reset to 8."))
        
        network_group.setLayout(network_form)
        left_layout.addWidget(network_group)
        
        # Experiment Group
        experiment_group = QGroupBox("🔬 Experiment Settings")
        experiment_form = QVBoxLayout()
        
        self.trials_spin = EnhancedSpinBox(
            default=200, min_val=10, max_val=100000, step=10,
            tooltip="Number of random networks to evaluate (10 - 100,000)"
        )
        experiment_form.addWidget(ConfigRow("Trials:", self.trials_spin,
                                            "Adaptive scroll: <100→+10, <1000→+50, <10000→+100, else +1000"))
        
        self.bias_combo = QComboBox()
        self.bias_combo.addItems(['centered', 'sigmoidal', 'gaussian', 'linear'])
        self.bias_combo.setToolTip("How to map Value channel to bias values")
        experiment_form.addWidget(ConfigRow("Bias Mapping:", self.bias_combo))
        
        self.spatial_check = QCheckBox("Spatial Smoothing")
        self.spatial_check.setChecked(True)
        self.spatial_check.setToolTip("Apply local averaging to bias values")
        experiment_form.addWidget(self.spatial_check)
        
        self.depth_check = QCheckBox("Depth Scaling")
        self.depth_check.setChecked(True)
        self.depth_check.setToolTip("Scale biases by 1/sqrt(layer_depth)")
        experiment_form.addWidget(self.depth_check)
        
        self.act_check = QCheckBox("Activation Awareness")
        self.act_check.setChecked(True)
        self.act_check.setToolTip("Adjust biases based on activation function type")
        experiment_form.addWidget(self.act_check)
        
        experiment_group.setLayout(experiment_form)
        left_layout.addWidget(experiment_group)
        
        # Reset All button
        reset_all_btn = QPushButton("↺  Reset All to Defaults")
        reset_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #ecf0f1; color: #264653; padding: 10px;
                border: 2px solid #bdc3c7; border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background-color: #E63946; color: white; border-color: #E63946; }
        """)
        reset_all_btn.clicked.connect(self.reset_all)
        left_layout.addWidget(reset_all_btn)
        
        left_layout.addStretch()
        
        # ── Right Panel: Controls ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        control_group = QGroupBox("🚀 Execution Controls")
        control_inner = QVBoxLayout()
        
        self.run_btn = QPushButton("▶  START EXPERIMENT")
        self.run_btn.setMinimumHeight(60)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #2A9D8F; color: white; font-size: 18px;
                font-weight: bold; border-radius: 8px; border: none;
            }
            QPushButton:hover { background-color: #21867a; }
            QPushButton:pressed { background-color: #1a6b61; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.run_btn.clicked.connect(self.start_experiment)
        
        self.stop_btn = QPushButton("⏹  STOP")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #E63946; color: white; font-size: 16px;
                font-weight: bold; border-radius: 8px; border: none;
            }
            QPushButton:hover { background-color: #c1303b; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.stop_btn.clicked.connect(self.stop_experiment)
        
        self.progress = QProgressBar()
        self.progress.setMinimumHeight(35)
        
        self.status_label = QLabel("Ready to start experiment")
        self.status_label.setStyleSheet("font-size: 13px; padding: 8px;")
        self.status_label.setWordWrap(True)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        
        control_inner.addWidget(self.run_btn)
        control_inner.addWidget(self.stop_btn)
        control_inner.addWidget(QLabel("Progress:"))
        control_inner.addWidget(self.progress)
        control_inner.addWidget(self.status_label)
        control_inner.addWidget(QLabel("Log:"))
        control_inner.addWidget(self.log_text)
        
        control_group.setLayout(control_inner)
        right_layout.addWidget(control_group)
        
        # Quick Presets
        preset_group = QGroupBox("⚡ Quick Presets")
        preset_layout = QVBoxLayout()
        
        presets = [
            ("🐇 Quick Test (50 trials)", {"trials": 50, "height": 6, "width": 4}),
            ("📊 Standard (200 trials)", {"trials": 200, "height": 10, "width": 8}),
            ("🔬 Deep Scan (1000 trials)", {"trials": 1000, "height": 16, "width": 12}),
            ("🚀 Full Search (10,000 trials)", {"trials": 10000, "height": 24, "width": 16}),
        ]
        
        for name, params in presets:
            btn = QPushButton(name)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #264653; color: white; padding: 8px;
                    border-radius: 5px; text-align: left; font-size: 12px;
                }
                QPushButton:hover { background-color: #2A9D8F; }
            """)
            btn.clicked.connect(lambda checked, p=params: self.apply_preset(p))
            preset_layout.addWidget(btn)
        
        preset_group.setLayout(preset_layout)
        right_layout.addWidget(preset_group)
        
        # Assemble
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 1)
    
    def apply_preset(self, params):
        if 'trials' in params:
            self.trials_spin.setValue(params['trials'])
        if 'height' in params:
            self.height_spin.setValue(params['height'])
        if 'width' in params:
            self.width_spin.setValue(params['width'])
    
    def reset_all(self):
        for widget in self.findChildren((EnhancedSpinBox, EnhancedDoubleSpinBox)):
            widget.reset()
        self.bias_combo.setCurrentText('centered')
        self.dataset_combo.setCurrentText('make_moons')
        self.scaler_combo.setCurrentText('minmax')
        self.spatial_check.setChecked(True)
        self.depth_check.setChecked(True)
        self.act_check.setChecked(True)
    
    def start_experiment(self):
        config = {
            'dataset': self.dataset_combo.currentText(),
            'n_samples': self.n_samples_spin.value(),
            'noise': self.noise_spin.value(),
            'scaler': self.scaler_combo.currentText(),
            'height': self.height_spin.value(),
            'width': self.width_spin.value(),
            'trials': self.trials_spin.value(),
            'bias_mapping': self.bias_combo.currentText(),
            'spatial': self.spatial_check.isChecked(),
            'depth_scale': self.depth_check.isChecked(),
            'act_aware': self.act_check.isChecked()
        }
        
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        self.log_text.clear()
        
        self.parent.run_experiment(config)
    
    def stop_experiment(self):
        self.parent.stop_experiment()
        self.stop_btn.setEnabled(False)
    
    def update_progress(self, value):
        self.progress.setValue(value)
    
    def update_status(self, msg):
        self.status_label.setText(msg)
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    
    def experiment_finished(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(100)


class PlotTab(QWidget):
    def __init__(self, title, parent):
        super().__init__()
        self.title = title
        self.parent = parent
        self.figure = None
        self.canvas = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        header = QLabel(self.title)
        header.setStyleSheet("font-size: 20px; font-weight: bold; padding: 12px; border-radius: 6px;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_layout.setAlignment(Qt.AlignCenter)
        
        self.placeholder = QLabel("⏳ Run an experiment to see results here")
        self.placeholder.setStyleSheet("font-size: 18px; padding: 50px;")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.plot_layout.addWidget(self.placeholder)
        
        scroll.setWidget(self.plot_widget)
        layout.addWidget(scroll)
        
        toolbar_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("💾 Save Plot")
        self.save_btn.setEnabled(False)
        self.save_btn.setToolTip("Save this plot as an image file")
        self.save_btn.clicked.connect(self.save_plot)
        
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setToolTip("Regenerate this plot with current theme")
        self.refresh_btn.clicked.connect(self.refresh_plot)
        
        toolbar_layout.addWidget(self.save_btn)
        toolbar_layout.addWidget(self.refresh_btn)
        toolbar_layout.addStretch()
        
        layout.addLayout(toolbar_layout)
    
    def set_figure(self, fig):
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
        if self.placeholder:
            self.plot_layout.removeWidget(self.placeholder)
            self.placeholder.deleteLater()
            self.placeholder = None
        
        self.figure = fig
        self.canvas = FigureCanvas(fig)
        self.canvas.setMinimumWidth(1200)
        self.plot_layout.addWidget(self.canvas)
        
        self.save_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
    
    def save_plot(self):
        if not self.figure:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", f"{self.title.replace(' ', '_')}.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        if filename:
            self.figure.savefig(filename, dpi=200, bbox_inches='tight')
            QMessageBox.information(self, "Saved", f"Plot saved to:\n{filename}")
    
    def refresh_plot(self):
        if self.parent.results:
            self.parent.generate_plot(self.title)


class InspectorTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.canvas = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        header = QLabel("🔬 Network Inspector")
        header.setStyleSheet("font-size: 20px; font-weight: bold; padding: 12px; border-radius: 6px;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("Select Best Network by Loss Function:"))
        
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["BCE", "Focal (γ=2)", "Focal (γ=5)", "Hinge", "Huber (δ=1)", "Log-Cosh"])
        self.loss_combo.setMinimumWidth(200)
        self.loss_combo.setToolTip("Choose which loss function's best network to inspect")
        self.loss_combo.currentTextChanged.connect(self.inspect_network)
        select_layout.addWidget(self.loss_combo)
        
        self.inspect_btn = QPushButton("🔍 Inspect")
        self.inspect_btn.setEnabled(False)
        self.inspect_btn.setToolTip("Generate detailed visualization of the selected network")
        self.inspect_btn.clicked.connect(self.inspect_network)
        select_layout.addWidget(self.inspect_btn)
        select_layout.addStretch()
        
        layout.addLayout(select_layout)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_layout.setAlignment(Qt.AlignCenter)
        
        self.placeholder = QLabel("⏳ Run an experiment first, then select a loss function to inspect")
        self.placeholder.setStyleSheet("font-size: 18px; padding: 50px;")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.plot_layout.addWidget(self.placeholder)
        
        scroll.setWidget(self.plot_widget)
        layout.addWidget(scroll)
    
    def enable(self):
        self.inspect_btn.setEnabled(True)
    
    def inspect_network(self):
        if not self.parent.results:
            return
        
        loss_name = self.loss_combo.currentText()
        theme = 'dark' if self.parent.dark_mode else 'light'
        fig = PlotGenerator.create_network_inspector(
            self.parent.results['best_networks'], loss_name, theme
        )
        
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
        if self.placeholder:
            self.plot_layout.removeWidget(self.placeholder)
            self.placeholder.deleteLater()
            self.placeholder = None
        
        self.canvas = FigureCanvas(fig)
        self.canvas.setMinimumWidth(1200)
        self.plot_layout.addWidget(self.canvas)


class AboutTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        title = QLabel("🎨 Image-to-Neural Network Explorer")
        title.setStyleSheet("font-size: 36px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Random Image-Derived Neural Network Explorer v3.0")
        subtitle.setStyleSheet("font-size: 20px; margin-bottom: 30px;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        info = QLabel("""
        <div style='max-width: 800px; text-align: center; font-size: 15px; line-height: 1.8;'>
        <p>This application explores how random HSV images can be converted into neural network architectures.</p>
        <p><b>How it works:</b></p>
        <ul style='text-align: left;'>
        <li><b>Hue (H)</b> → Activation function selection (24 types)</li>
        <li><b>Saturation (S)</b> → Weight magnitude</li>
        <li><b>Value (V)</b> → Bias values (with multiple mapping strategies)</li>
        </ul>
        <p><b>Architecture (v3.0):</b></p>
        <ul style='text-align: left;'>
        <li>🎲 <b>ImageGenerator</b> - Generates random HSV images (NO network knowledge)</li>
        <li>🏗️ <b>NetworkBuilder</b> - Consumes images, builds networks (NO image generation)</li>
        <li>📊 <b>NetworkEvaluator</b> - Consumes networks, evaluates on data (NO building)</li>
        </ul>
        <p><b>New Features in v3.0:</b></p>
        <ul style='text-align: left;'>
        <li>🎲 <b>Decoupled Architecture</b> - Image generation completely independent from networks</li>
        <li>🔢 <b>Enhanced Spinboxes</b> - Support up to 100,000 with adaptive stepping</li>
        <li>⚡ <b>Quick Presets</b> - One-click experiment configurations</li>
        <li>↺ <b>Reset Buttons</b> - Per-widget and global reset to defaults</li>
        <li>🌗 Dark/Light theme toggle</li>
        <li>📊 Multiple dataset support</li>
        <li>⌨️ Keyboard shortcuts (Ctrl+S, Ctrl+R, Ctrl+Q, Ctrl+T)</li>
        </ul>
        </div>
        """)
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)
        
        layout.addStretch()


# ═══════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.results = None
        self.worker = None
        self.thread = None
        self.dark_mode = False
        self.settings = QSettings("ImageToNN", "Explorer")
        self.init_ui()
        self.create_menus()
        self.create_toolbar()
        self.load_settings()
    
    def init_ui(self):
        self.setWindowTitle("🎨 Image-to-Neural Network Explorer v3.0")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        self.update_theme()
        
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        
        self.setup_tab = SetupTab(self)
        self.results_tab = PlotTab("📊 Multi-Loss Comparison Dashboard", self)
        self.arch_tab = PlotTab("🏗️ Network Architecture Analysis", self)
        self.temporal_tab = PlotTab("📈 Temporal & Convergence Analysis", self)
        self.inspector_tab = InspectorTab(self)
        self.about_tab = AboutTab()
        
        self.tabs.addTab(self.setup_tab, "⚙️ Setup")
        self.tabs.addTab(self.results_tab, "📊 Results")
        self.tabs.addTab(self.arch_tab, "🏗️ Architecture")
        self.tabs.addTab(self.temporal_tab, "📈 Temporal")
        self.tabs.addTab(self.inspector_tab, "🔬 Inspector")
        self.tabs.addTab(self.about_tab, "ℹ️ About")
        
        self.setCentralWidget(self.tabs)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def create_menus(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("📁 File")
        
        save_action = QAction("💾 Save Results", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        load_action = QAction("📂 Load Results", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_results)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("❌ Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        view_menu = menubar.addMenu("👁️ View")
        
        theme_action = QAction("🌗 Toggle Dark/Light Mode", self)
        theme_action.setShortcut("Ctrl+T")
        theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(theme_action)
        
        view_menu.addSeparator()
        
        refresh_action = QAction("🔄 Refresh All Plots", self)
        refresh_action.setShortcut("Ctrl+R")
        refresh_action.triggered.connect(self.refresh_all_plots)
        view_menu.addAction(refresh_action)
        
        help_menu = menubar.addMenu("❓ Help")
        
        about_action = QAction("ℹ️ About", self)
        about_action.triggered.connect(lambda: self.tabs.setCurrentIndex(5))
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        theme_btn = QToolButton()
        theme_btn.setText("🌗 Theme")
        theme_btn.setToolTip("Toggle dark/light mode (Ctrl+T)")
        theme_btn.clicked.connect(self.toggle_theme)
        toolbar.addWidget(theme_btn)
        
        toolbar.addSeparator()
        
        refresh_btn = QToolButton()
        refresh_btn.setText("🔄 Refresh")
        refresh_btn.setToolTip("Refresh all plots (Ctrl+R)")
        refresh_btn.clicked.connect(self.refresh_all_plots)
        toolbar.addWidget(refresh_btn)
    
    def update_theme(self):
        theme_name = 'dark' if self.dark_mode else 'light'
        self.setStyleSheet(ThemeManager.get_stylesheet(theme_name))
    
    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.update_theme()
        self.save_settings()
        if self.results:
            self.refresh_all_plots()
    
    def save_settings(self):
        self.settings.setValue("dark_mode", self.dark_mode)
    
    def load_settings(self):
        self.dark_mode = self.settings.value("dark_mode", False, type=bool)
        self.update_theme()
    
    def run_experiment(self, config):
        self.thread = QThread()
        self.worker = ExperimentWorker(config)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.setup_tab.update_progress)
        self.worker.status.connect(self.setup_tab.update_status)
        self.worker.status.connect(self.status_bar.showMessage)
        self.worker.finished.connect(self.on_experiment_finished)
        self.worker.error.connect(self.on_experiment_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()
    
    def stop_experiment(self):
        if self.worker:
            self.worker.stop()
    
    def on_experiment_finished(self, results):
        self.results = results
        self.setup_tab.experiment_finished()
        
        elapsed = results.get('elapsed_time', 0)
        self.status_bar.showMessage(
            f"Experiment complete! {len(results['all_losses']['BCE'])} trials evaluated in {elapsed:.1f}s"
        )
        
        self.generate_plot("📊 Multi-Loss Comparison Dashboard")
        self.generate_plot("🏗️ Network Architecture Analysis")
        self.generate_plot("📈 Temporal & Convergence Analysis")
        
        self.inspector_tab.enable()
        self.tabs.setCurrentIndex(1)
        
        QMessageBox.information(self, "Success", 
            f"Experiment completed successfully!\n\n"
            f"Dataset: {results['config']['dataset']}\n"
            f"Trials: {len(results['all_losses']['BCE'])}\n"
            f"Time: {elapsed:.1f}s\n\n"
            f"Best BCE Loss: {results['best_networks']['BCE']['loss']:.4f}\n"
            f"Best Hinge Loss: {results['best_networks']['Hinge']['loss']:.4f}\n\n"
            f"Component Stats:\n"
            f"  • Images Generated: {results['component_stats']['image_generator']['total_generated']}\n"
            f"  • Networks Built: {results['component_stats']['network_builder']['total_built']}\n"
            f"  • Networks Evaluated: {results['component_stats']['network_evaluator']['total_evaluated']}")
    
    def on_experiment_error(self, error_msg):
        self.setup_tab.experiment_finished()
        self.status_bar.showMessage("Experiment failed!")
        QMessageBox.critical(self, "Error", f"Experiment failed:\n\n{error_msg}")
    
    def generate_plot(self, title):
        if not self.results:
            return
        
        theme = 'dark' if self.dark_mode else 'light'
        
        if title == "📊 Multi-Loss Comparison Dashboard":
            fig = PlotGenerator.create_multi_loss_dashboard(self.results, theme)
            self.results_tab.set_figure(fig)
        elif title == "🏗️ Network Architecture Analysis":
            fig = PlotGenerator.create_architecture_analysis(self.results, theme)
            self.arch_tab.set_figure(fig)
        elif title == "📈 Temporal & Convergence Analysis":
            fig = PlotGenerator.create_temporal_analysis(self.results, theme)
            self.temporal_tab.set_figure(fig)
    
    def refresh_all_plots(self):
        if self.results:
            self.generate_plot("📊 Multi-Loss Comparison Dashboard")
            self.generate_plot("🏗️ Network Architecture Analysis")
            self.generate_plot("📈 Temporal & Convergence Analysis")
            self.status_bar.showMessage("All plots refreshed")
    
    def save_results(self):
        if not self.results:
            QMessageBox.warning(self, "No Results", "Run an experiment first!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "experiment_results.pkl",
            "Pickle Files (*.pkl);;All Files (*.*)"
        )
        if filename:
            with open(filename, 'wb') as f:
                pickle.dump(self.results, f)
            self.status_bar.showMessage(f"Results saved to {filename}")
            QMessageBox.information(self, "Saved", f"Results saved to:\n{filename}")
    
    def load_results(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Results", "",
            "Pickle Files (*.pkl);;All Files (*.*)"
        )
        if filename:
            try:
                with open(filename, 'rb') as f:
                    self.results = pickle.load(f)
                self.refresh_all_plots()
                self.inspector_tab.enable()
                self.tabs.setCurrentIndex(1)
                self.status_bar.showMessage(f"Results loaded from {filename}")
                QMessageBox.information(self, "Loaded", "Results loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load results:\n{str(e)}")


# ═══════════════════════════════════════════════
# APPLICATION ENTRY POINT
# ═══════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Image-to-NN Explorer")
    app.setApplicationVersion("3.0.0")
    app.setOrganizationName("ImageToNN")
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()