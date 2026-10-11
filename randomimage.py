# main.py
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.colors
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from scipy.ndimage import sobel, gaussian_filter, laplace
from scipy.integrate import cumulative_trapezoid
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QPushButton, QLabel, QSpinBox, 
                                QComboBox, QTabWidget, QScrollArea, QSplitter, 
                                QGroupBox, QCheckBox, QDoubleSpinBox, QFrame)
from PySide6.QtCore import Qt, QThread, Signal

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════
# LIBRARY 1: HSV IMAGE GENERATION
# ═══════════════════════════════════════════════
class HSVImageLibrary:
    @staticmethod
    def generate_image(height=10, width=8, seed=None, pattern='random', allowed_hues=None):
        rng = np.random.RandomState(seed)
        
        if pattern == 'random':
            H = rng.uniform(0, 360, (height, width))
            S = rng.uniform(0, 1, (height, width))
            V = rng.uniform(0, 1, (height, width))
        elif pattern == 'gradient':
            x = np.linspace(0, 1, width)
            y = np.linspace(0, 1, height)
            X, Y = np.meshgrid(x, y)
            H = (X * 360 + rng.uniform(-20, 20, (height, width))) % 360
            S = np.clip(Y + rng.uniform(-0.1, 0.1, (height, width)), 0, 1)
            V = np.clip((X + Y) / 2 + rng.uniform(-0.1, 0.1, (height, width)), 0, 1)
        elif pattern == 'sine_waves':
            x = np.linspace(0, 4 * np.pi, width)
            y = np.linspace(0, 4 * np.pi, height)
            X, Y = np.meshgrid(x, y)
            H = ((np.sin(X) + np.cos(Y)) * 90 + 180 + rng.uniform(-10, 10, (height, width))) % 360
            S = np.clip(0.5 + 0.5 * np.sin(X * 1.5) * np.cos(Y * 0.5), 0, 1)
            V = np.clip(0.5 + 0.5 * np.cos(X * 0.5 + Y * 1.5), 0, 1)
        elif pattern == 'voronoi':
            n_points = max(5, (height * width) // 10)
            points = rng.rand(n_points, 2) * np.array([height, width])
            colors_H = rng.uniform(0, 360, n_points)
            colors_S = rng.uniform(0.2, 1.0, n_points)
            colors_V = rng.uniform(0.2, 1.0, n_points)
            H, S, V = np.zeros((height, width)), np.zeros((height, width)), np.zeros((height, width))
            for i in range(height):
                for j in range(width):
                    dists = np.linalg.norm(points - np.array([i, j]), axis=1)
                    min_idx = np.argmin(dists)
                    H[i, j] = colors_H[min_idx]
                    S[i, j] = colors_S[min_idx]
                    V[i, j] = colors_V[min_idx]
            H += rng.uniform(-5, 5, (height, width))
            S += rng.uniform(-0.05, 0.05, (height, width))
            V += rng.uniform(-0.05, 0.05, (height, width))
        elif pattern == 'perlin':
            scale = max(2, min(height, width) // 4)
            low_res_H = rng.uniform(0, 360, (scale, scale))
            low_res_S = rng.uniform(0.2, 1.0, (scale, scale))
            low_res_V = rng.uniform(0.2, 1.0, (scale, scale))
            H = np.repeat(np.repeat(low_res_H, height//scale + 1, axis=0), width//scale + 1, axis=1)[:height, :width]
            S = np.repeat(np.repeat(low_res_S, height//scale + 1, axis=0), width//scale + 1, axis=1)[:height, :width]
            V = np.repeat(np.repeat(low_res_V, height//scale + 1, axis=0), width//scale + 1, axis=1)[:height, :width]
            H += rng.uniform(-15, 15, (height, width))
            S += rng.uniform(-0.1, 0.1, (height, width))
            V += rng.uniform(-0.1, 0.1, (height, width))
            
        if allowed_hues is not None and len(allowed_hues) > 0:
            allowed_hues_arr = np.array(allowed_hues)
            H_flat = H.flatten()
            extended_hues = np.concatenate([allowed_hues_arr, allowed_hues_arr - 360, allowed_hues_arr + 360])
            dists = np.abs(H_flat[:, None] - extended_hues[None, :])
            closest_idx = np.argmin(dists, axis=1)
            H = extended_hues[closest_idx] % 360
            H = H.reshape(height, width)
            
        H = np.clip(H, 0, 360)
        S = np.clip(S, 0, 1)
        V = np.clip(V, 0, 1)
        return H, S, V

# ═══════════════════════════════════════════════
# LIBRARY 2: FEED-FORWARD NETWORK GENERATION
# ═══════════════════════════════════════════════
class FFNetworkLibrary:
    @staticmethod
    def _relu(x): return np.maximum(0, x)
    
    @staticmethod
    def _leaky_relu(x, a=0.01): return np.where(x > 0, x, a * x)
    
    @staticmethod
    def _elu(x, a=1.0): return np.where(x > 0, x, a * (np.exp(np.clip(x, -50, 50)) - 1))
    
    @staticmethod
    def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))
    
    @staticmethod
    def _tanh(x): return np.tanh(np.clip(x, -100, 100))
    
    @staticmethod
    def _silu(x): return x * (1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))) # Swish
    
    @staticmethod
    def _gelu(x): return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def _softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0) # Numerically stable
    
    @staticmethod
    def _linear(x): return x
    
    # 9 Activations -> 360 / 9 = 40 degrees interval
    ACTIVATION_MAP = [
        (40,  _relu.__func__, "ReLU"),
        (80,  _leaky_relu.__func__, "LeakyReLU"),
        (120, _elu.__func__, "ELU"),
        (160, _sigmoid.__func__, "Sigmoid"),
        (200, _tanh.__func__, "Tanh"),
        (240, _silu.__func__, "SiLU"),
        (280, _gelu.__func__, "GELU"),
        (320, _softplus.__func__, "Softplus"),
        (360, _linear.__func__, "Linear")
    ]
    HUE_THRESHOLDS = np.array([t for t, _, _ in ACTIVATION_MAP])
    ACTIVATION_FUNCS = [f for _, f, _ in ACTIVATION_MAP]

    @staticmethod
    def get_activation_choices():
        choices = []
        for i, (thresh, _, name) in enumerate(FFNetworkLibrary.ACTIVATION_MAP):
            prev_thresh = FFNetworkLibrary.ACTIVATION_MAP[i-1][0] if i > 0 else 0
            center_hue = (prev_thresh + thresh) / 2.0
            choices.append({'index': i, 'name': name, 'hue': center_hue})
        return choices

    @staticmethod
    def build_network(H, S, V, bias_mapping='centered', depth_scale=True, allowed_indices=None):
        num_neurons, num_layers = H.shape
        
        # Pre-allocate arrays with explicit network shape: (Layers, Neurons)
        weights = np.zeros((num_layers, num_neurons))
        biases = np.zeros((num_layers, num_neurons))
        act_indices = np.zeros((num_layers, num_neurons), dtype=int)
        
        if allowed_indices is None:
            allowed_indices = list(range(len(FFNetworkLibrary.ACTIVATION_FUNCS)))
            
        choices = FFNetworkLibrary.get_activation_choices()
        allowed_choices = [c for c in choices if c['index'] in allowed_indices]
        allowed_centers = np.array([c['hue'] for c in allowed_choices])
        allowed_idxs = np.array([c['index'] for c in allowed_choices])
        
        # Pre-calculate extended hues for closest-match logic
        extended_hues = np.concatenate([allowed_centers, allowed_centers - 360, allowed_centers + 360])
        
        # Iterate through image: Y-axis (i) = Neuron, X-axis (j) = Layer
        for i in range(num_neurons):
            for j in range(num_layers):
                layer_idx = j
                neuron_idx = i
                
                # 1. Map Saturation to Weight
                weights[layer_idx, neuron_idx] = S[i, j]
                
                # 2. Map Value to Bias
                v_val = V[i, j]
                if bias_mapping == 'centered':
                    b = 2.0 * v_val - 1.0
                else:
                    b = v_val
                
                if depth_scale:
                    # j is 0-indexed, so layer 1 is j=0 -> sqrt(1), layer 2 is j=1 -> sqrt(2)
                    b *= (1.0 / np.sqrt(j + 1.0))
                    
                biases[layer_idx, neuron_idx] = b
                
                # 3. Map Hue to Activation Function iteratively
                h_val = H[i, j]
                dists = np.abs(extended_hues - h_val)
                closest_ext_idx = np.argmin(dists)
                closest_act_pos = closest_ext_idx % len(allowed_centers)
                
                act_indices[layer_idx, neuron_idx] = allowed_idxs[closest_act_pos]
            
        return weights, biases, act_indices

    @staticmethod
    def evaluate_network(X, y, weights, biases, act_indices):
        current_X = X.copy()
        num_layers = weights.shape[0]
        
        for l in range(num_layers):
            prev_sum = np.sum(current_X, axis=1, keepdims=True)
            Z = prev_sum * weights[l] + biases[l]
            
            next_X = np.zeros_like(Z)
            unique_acts = np.unique(act_indices[l, :])
            for act_idx in unique_acts:
                mask = (act_indices[l, :] == act_idx)
                act_func = FFNetworkLibrary.ACTIVATION_FUNCS[act_idx]
                next_X[:, mask] = act_func(Z[:, mask])
            current_X = next_X
            
        raw_mean = np.mean(current_X, axis=1)
        raw_std = np.std(current_X, axis=1) + 1e-8
        logits = (raw_mean / raw_std) * np.sqrt(current_X.shape[1])
        
        loss = np.mean(np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits))))
        return logits, loss, current_X

    @staticmethod
    def load_dataset(name="make_moons", n_samples=500, noise=0.25):
        if name == "make_moons":
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        elif name == "make_circles":
            X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
        elif name == "xor":
            # Custom XOR dataset generator
            rng = np.random.RandomState(42)
            # Pick random corners: (0,0) and (1,1) are class 0; (0,1) and (1,0) are class 1
            corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            labels = np.array([0, 1, 1, 0])
            
            indices = rng.randint(0, 4, size=n_samples)
            X = corners[indices].astype(np.float64)
            y = labels[indices].astype(np.float64)
            
            # Add Gaussian noise to spread out the clusters
            X += rng.normal(0, noise, X.shape)
        else:
            X, y = make_classification(n_samples=n_samples, n_features=4, n_informative=4, n_redundant=0, random_state=42)
            
        if X.shape[1] < 4:
            noise_dims = 4 - X.shape[1]
            X = np.hstack((X, np.random.randn(n_samples, noise_dims) * 0.1))
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = MinMaxScaler()
        return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test

    @staticmethod
    def run_experiment(config, allowed_hues, allowed_indices):
        X_train, X_test, y_train, y_test = FFNetworkLibrary.load_dataset(
            config['dataset'], config['n_samples'], config['noise']
        )
        
        results = {'all_losses': [], 'best_bce': float('inf'), 'best_network': None, 'config': config}
        
        for i in range(config['trials']):
            H, S, V = HSVImageLibrary.generate_image(
                config['height'], config['width'], pattern=config['pattern'], allowed_hues=allowed_hues
            )
            weights, biases, act_indices = FFNetworkLibrary.build_network(
                H, S, V, bias_mapping=config['bias_mapping'], allowed_indices=allowed_indices
            )
            logits, loss, hidden_acts = FFNetworkLibrary.evaluate_network(X_test, y_test, weights, biases, act_indices)
            
            results['all_losses'].append(loss)
            if loss < results['best_bce']:
                results['best_bce'] = loss
                results['best_network'] = {
                    'weights': weights, 'biases': biases, 'act_indices': act_indices,
                    'hidden_acts': hidden_acts, 'hsv': (H, S, V)
                }
                
        results['y_test'] = y_test
        return results

# ═══════════════════════════════════════════════
# LIBRARY 3: IMAGE PLOTS (WITH MATH CHARTS)
# ═══════════════════════════════════════════════
class ImagePlotLibrary:
    @staticmethod
    def plot_cv_analysis(H, S, V, theme_colors):
        H_norm = H / 360.0
        fig = Figure(figsize=(16, 6), facecolor=theme_colors['plot_bg'])
        fig.suptitle("🛰️ Image CV Analysis", fontsize=16, color=theme_colors['fg'])
        gs = GridSpec(1, 3, figure=fig, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(sobel(H_norm), cmap='viridis', aspect='auto')
        ax1.set_title("Hue Sobel Edges", color=theme_colors['fg'])
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(gaussian_filter(S, sigma=1.5), cmap='magma', aspect='auto')
        ax2.set_title("Saturation Gaussian Blur", color=theme_colors['fg'])
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(laplace(V), cmap='coolwarm', aspect='auto')
        ax3.set_title("Value Laplacian", color=theme_colors['fg'])
        
        for ax in fig.axes:
            ax.tick_params(colors=theme_colors['fg'])
            ax.set_facecolor(theme_colors['plot_bg'])
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    @staticmethod
    def plot_kernels(H, S, V, theme_colors):
        fig = Figure(figsize=(16, 6), facecolor=theme_colors['plot_bg'])
        fig.suptitle("🧩 Image Distributions", fontsize=16, color=theme_colors['fg'])
        gs = GridSpec(1, 3, figure=fig, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(H.flatten(), bins=24, color='#E63946', alpha=0.8)
        ax1.set_title("Hue Distribution", color=theme_colors['fg'])

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(S.flatten(), bins=30, color='#F4A261', alpha=0.8)
        ax2.set_title("Saturation Distribution", color=theme_colors['fg'])

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(V.flatten(), bins=30, color='#2A9D8F', alpha=0.8)
        ax3.set_title("Value Distribution", color=theme_colors['fg'])
        
        for ax in fig.axes:
            ax.tick_params(colors=theme_colors['fg'])
            ax.set_facecolor(theme_colors['plot_bg'])
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    @staticmethod
    def plot_image_math(H, S, V, theme_colors):
        """NEW: Plots 1st-4th order spatial derivatives and integrals of the 2D image."""
        fig = Figure(figsize=(12, 16), facecolor=theme_colors['plot_bg'])
        fig.suptitle("🧮 Image Calculus: 1st-4th Order Derivatives & Integrals (Value Channel)", fontsize=16, color=theme_colors['fg'])
        gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Use the Value channel as the spatial domain for calculus
        img = V
        labels = ["1st", "2nd", "3rd", "4th"]
        
        # Calculate Derivatives (Spatial Gradient Magnitudes)
        current_d = img.copy()
        derivs = []
        for _ in range(4):
            dx = np.gradient(current_d, axis=0)
            dy = np.gradient(current_d, axis=1)
            current_d = np.sqrt(dx**2 + dy**2)
            derivs.append(current_d)
            
        # Calculate Integrals (2D Cumulative Trapezoidal Integration)
        current_i = img.copy()
        integs = []
        for _ in range(4):
            current_i = cumulative_trapezoid(current_i, axis=0, initial=0)
            current_i = cumulative_trapezoid(current_i, axis=1, initial=0)
            integs.append(current_i)
            
        for i in range(4):
            # Plot Derivative
            ax_d = fig.add_subplot(gs[i, 0])
            d_clip = np.clip(derivs[i], 0, np.percentile(derivs[i], 95))
            im_d = ax_d.imshow(d_clip, cmap='magma', aspect='auto')
            ax_d.set_title(f"{labels[i]} Derivative (Edges)", color=theme_colors['fg'])
            fig.colorbar(im_d, ax=ax_d, fraction=0.046)
            
            # Plot Integral
            ax_i = fig.add_subplot(gs[i, 1])
            # Normalize integrals to [-1, 1] for consistent colormap scaling as area grows
            i_norm = integs[i]
            max_val = np.max(np.abs(i_norm)) + 1e-8
            i_norm = i_norm / max_val
            im_i = ax_i.imshow(i_norm, cmap='coolwarm', aspect='auto')
            ax_i.set_title(f"{labels[i]} Integral (Volume)", color=theme_colors['fg'])
            fig.colorbar(im_i, ax=ax_i, fraction=0.046)
            
        for ax in fig.axes:
            ax.tick_params(colors=theme_colors['fg'])
            ax.set_facecolor(theme_colors['plot_bg'])
                
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

# ═══════════════════════════════════════════════
# LIBRARY 4: NETWORK PLOTS
# ═══════════════════════════════════════════════
class NetworkPlotLibrary:
    @staticmethod
    def plot_architecture(weights, biases, act_indices, theme_colors):
        fig = Figure(figsize=(18, 10), facecolor=theme_colors['plot_bg'])
        fig.suptitle("🧠 Full Network Inspector: Architecture & Distributions", fontsize=18, color=theme_colors['fg'])
        gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(weights, cmap='viridis', aspect='auto')
        ax1.set_title("Weights Heatmap", color=theme_colors['fg'])
        fig.colorbar(im1, ax=ax1, fraction=0.046)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(biases, cmap='coolwarm', aspect='auto')
        ax2.set_title("Biases Heatmap", color=theme_colors['fg'])
        fig.colorbar(im2, ax=ax2, fraction=0.046)
        
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(act_indices, cmap='tab20', aspect='auto')
        ax3.set_title("Activations Heatmap", color=theme_colors['fg'])
        fig.colorbar(im3, ax=ax3, fraction=0.046)
        
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(weights.flatten(), bins=30, color='#E63946', alpha=0.8, edgecolor='black')
        ax4.set_title("Weights Distribution", color=theme_colors['fg'])
        ax4.set_xlabel("Weight Value", color=theme_colors['fg'])
        ax4.set_ylabel("Frequency", color=theme_colors['fg'])
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(biases.flatten(), bins=30, color='#2A9D8F', alpha=0.8, edgecolor='black')
        ax5.set_title("Biases Distribution", color=theme_colors['fg'])
        ax5.set_xlabel("Bias Value", color=theme_colors['fg'])
        ax5.set_ylabel("Frequency", color=theme_colors['fg'])
        
        ax6 = fig.add_subplot(gs[1, 2])
        unique, counts = np.unique(act_indices, return_counts=True)
        act_names = [FFNetworkLibrary.ACTIVATION_MAP[i][2] for i in unique]
        ax6.bar(range(len(unique)), counts, color='#F4A261', alpha=0.8, edgecolor='black')
        ax6.set_xticks(range(len(unique)))
        ax6.set_xticklabels(act_names, rotation=45, ha='right')
        ax6.set_title("Activation Function Usage", color=theme_colors['fg'])
        ax6.set_ylabel("Count", color=theme_colors['fg'])
        
        for ax in fig.axes:
            ax.tick_params(colors=theme_colors['fg'])
            ax.set_facecolor(theme_colors['plot_bg'])
            if ax not in [ax1, ax2, ax3]:
                ax.grid(True, alpha=0.2, linestyle='--', color=theme_colors['fg'])
                
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    @staticmethod
    def plot_network_pca(hidden_activations, y_test, theme_colors):
        pca = PCA(n_components=4)
        X_pca = pca.fit_transform(hidden_activations)
        
        fig = Figure(figsize=(16, 12), facecolor=theme_colors['plot_bg'])
        fig.suptitle("🧠 Latent Space PCA Analysis (Network Hidden Layer)", fontsize=18, color=theme_colors['fg'])
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='coolwarm', alpha=0.7, edgecolors='none')
        ax1.set_title(f"PC1 vs PC2 ({pca.explained_variance_ratio_[0]*100 + pca.explained_variance_ratio_[1]*100:.1f}%)", color=theme_colors['fg'])
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(X_pca[:, 1], X_pca[:, 2], c=y_test, cmap='viridis', alpha=0.7, edgecolors='none')
        ax2.set_title(f"PC2 vs PC3 ({pca.explained_variance_ratio_[1]*100 + pca.explained_variance_ratio_[2]*100:.1f}%)", color=theme_colors['fg'])
        
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(X_pca[:, 2], X_pca[:, 3], c=y_test, cmap='plasma', alpha=0.7, edgecolors='none')
        ax3.set_title(f"PC3 vs PC4 ({pca.explained_variance_ratio_[2]*100 + pca.explained_variance_ratio_[3]*100:.1f}%)", color=theme_colors['fg'])
        
        ax4 = fig.add_subplot(gs[1, 1])
        bars = ax4.bar(range(1, 5), pca.explained_variance_ratio_, color=['#E63946', '#F4A261', '#2A9D8F', '#264653'], alpha=0.8, edgecolor='black')
        ax4.set_title("Explained Variance Ratio", color=theme_colors['fg'])
        ax4.set_xticks([1, 2, 3, 4])
        ax4.set_xlabel("Principal Component", color=theme_colors['fg'])
        ax4.set_ylabel("Ratio", color=theme_colors['fg'])
        for bar in bars:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval*100:.1f}%", ha='center', color=theme_colors['fg'], fontweight='bold')
        
        for ax in fig.axes:
            ax.tick_params(colors=theme_colors['fg'])
            ax.set_facecolor(theme_colors['plot_bg'])
            ax.grid(True, alpha=0.2, linestyle='--', color=theme_colors['fg'])
                
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    @staticmethod
    def plot_loss_history(losses, theme_colors):
        fig = Figure(figsize=(16, 6), facecolor=theme_colors['plot_bg'])
        fig.suptitle("📉 Experiment Performance Metrics", fontsize=18, color=theme_colors['fg'])
        gs = GridSpec(1, 2, figure=fig, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        trials = range(1, len(losses) + 1)
        ax1.plot(trials, losses, color='#E63946', linewidth=2, label='Trial Loss', alpha=0.7)
        running_best = np.minimum.accumulate(losses)
        ax1.plot(trials, running_best, color='#2A9D8F', linewidth=3, linestyle='--', label='Running Best')
        ax1.set_title("Loss Convergence Over Trials", color=theme_colors['fg'])
        ax1.set_xlabel("Trial Number", color=theme_colors['fg'])
        ax1.set_ylabel("BCE Loss", color=theme_colors['fg'])
        ax1.legend()
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(losses, bins=30, color='#F4A261', alpha=0.8, edgecolor='black')
        min_loss = np.min(losses)
        mean_loss = np.mean(losses)
        ax2.axvline(min_loss, color='#2A9D8F', linestyle='--', linewidth=2, label=f'Min Loss: {min_loss:.4f}')
        ax2.axvline(mean_loss, color='#E63946', linestyle='--', linewidth=2, label=f'Avg Loss: {mean_loss:.4f}')
        ax2.set_title("Distribution of Network Losses", color=theme_colors['fg'])
        ax2.set_xlabel("BCE Loss", color=theme_colors['fg'])
        ax2.set_ylabel("Frequency", color=theme_colors['fg'])
        ax2.legend()
        
        for ax in fig.axes:
            ax.tick_params(colors=theme_colors['fg'])
            ax.set_facecolor(theme_colors['plot_bg'])
            ax.grid(True, alpha=0.2, linestyle='--', color=theme_colors['fg'])
                
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    @staticmethod
    def plot_activation_math(act_indices, theme_colors):
        """Plots 1st-4th order derivatives and integrals of activations used."""
        unique_acts = np.unique(act_indices)
        
        fig = Figure(figsize=(16, 6), facecolor=theme_colors['plot_bg'])
        fig.suptitle("🧮 Activation Functions Math (Derivatives & Integrals)", fontsize=16, color=theme_colors['fg'])
        gs = GridSpec(1, 2, figure=fig, wspace=0.3)
        
        x = np.linspace(-5, 5, 1000)
        math_colors = ['#E63946', '#F4A261', '#2A9D8F', '#264653']
        labels = ['1st Order', '2nd Order', '3rd Order', '4th Order']
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title("Derivatives (1st to 4th)", color=theme_colors['fg'])
        ax1.set_xlabel("x", color=theme_colors['fg'])
        ax1.set_ylabel("f'(x)", color=theme_colors['fg'])
        ax1.axhline(0, color=theme_colors['fg'], alpha=0.3, linestyle='-')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title("Integrals (1st to 4th)", color=theme_colors['fg'])
        ax2.set_xlabel("x", color=theme_colors['fg'])
        ax2.set_ylabel("∫f(x)dx", color=theme_colors['fg'])
        ax2.axhline(0, color=theme_colors['fg'], alpha=0.3, linestyle='-')
        
        for act_idx in unique_acts:
            act_func = FFNetworkLibrary.ACTIVATION_FUNCS[act_idx]
            act_name = FFNetworkLibrary.ACTIVATION_MAP[act_idx][2]
            y = act_func(x)
            
            current_y = y.copy()
            for order in range(4):
                current_y = np.gradient(current_y, x)
                clipped_y = np.clip(current_y, -10, 10)
                ax1.plot(x, clipped_y, color=math_colors[order], linewidth=1.5, 
                         label=f'{act_name} - {labels[order]}', alpha=0.8)
            
            current_y = y.copy()
            for order in range(4):
                current_y = cumulative_trapezoid(current_y, x, initial=0)
                clipped_y = np.clip(current_y, -50, 50)
                ax2.plot(x, clipped_y, color=math_colors[order], linewidth=1.5, 
                         label=f'{act_name} - {labels[order]}', alpha=0.8)
        
        ax1.legend(fontsize=7, loc='upper left')
        ax2.legend(fontsize=7, loc='upper left')
        
        for ax in fig.axes:
            ax.tick_params(colors=theme_colors['fg'])
            ax.set_facecolor(theme_colors['plot_bg'])
            ax.grid(True, alpha=0.2, linestyle='--', color=theme_colors['fg'])
                
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

# ═══════════════════════════════════════════════
# LIBRARY 5: DATASET PLOTS
# ═══════════════════════════════════════════════
class DatasetPlotLibrary:
    @staticmethod
    def plot_dataset_analysis(X, y, config, theme_colors):
        fig = Figure(figsize=(12, 8), facecolor=theme_colors['plot_bg'])
        fig.suptitle(f"📊 Dataset Analysis: {config['dataset']} ({config['n_samples']} samples)", fontsize=16, color=theme_colors['fg'])
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.7, edgecolors='none')
        ax1.set_title("Feature 1 vs Feature 2", color=theme_colors['fg'])
        ax1.set_xlabel("Feature 1", color=theme_colors['fg'])
        ax1.set_ylabel("Feature 2", color=theme_colors['fg'])

        pca = PCA(n_components=4)
        X_pca = pca.fit_transform(X)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolors='none')
        ax2.set_title(f"PCA (PC1 vs PC2) - {pca.explained_variance_ratio_[0]*100:.1f}% / {pca.explained_variance_ratio_[1]*100:.1f}%", color=theme_colors['fg'])
        ax2.set_xlabel("PC1", color=theme_colors['fg'])
        ax2.set_ylabel("PC2", color=theme_colors['fg'])

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(X_pca[:, 2], X_pca[:, 3], c=y, cmap='plasma', alpha=0.7, edgecolors='none')
        ax3.set_title(f"PCA (PC3 vs PC4) - {pca.explained_variance_ratio_[2]*100:.1f}% / {pca.explained_variance_ratio_[3]*100:.1f}%", color=theme_colors['fg'])
        ax3.set_xlabel("PC3", color=theme_colors['fg'])
        ax3.set_ylabel("PC4", color=theme_colors['fg'])

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(X[:, 0], bins=30, color='#E63946', alpha=0.7, label='Feature 1')
        ax4.hist(X[:, 1], bins=30, color='#2A9D8F', alpha=0.7, label='Feature 2')
        ax4.set_title("Feature Distributions", color=theme_colors['fg'])
        ax4.set_xlabel("Value", color=theme_colors['fg'])
        ax4.set_ylabel("Frequency", color=theme_colors['fg'])
        ax4.legend()

        for ax in fig.axes:
            ax.tick_params(colors=theme_colors['fg'])
            ax.set_facecolor(theme_colors['plot_bg'])
            ax.grid(True, alpha=0.2, linestyle='--', color=theme_colors['fg'])
            
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

# ═══════════════════════════════════════════════
# GUI APPLICATION
# ═══════════════════════════════════════════════
class ThemeManager:
    LIGHT = {'bg': '#FAFAFA', 'fg': '#264653', 'accent': '#2A9D8F', 'plot_bg': '#FAFAFA', 'border': '#bdc3c7'}

class ExperimentWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, config, allowed_hues, allowed_indices):
        super().__init__()
        self.config = config
        self.allowed_hues = allowed_hues
        self.allowed_indices = allowed_indices

    def run(self):
        try:
            results = FFNetworkLibrary.run_experiment(self.config, self.allowed_hues, self.allowed_indices)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

class ImageGenerationTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.current_image = None
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(QLabel("🖼️ Image Generation (Lib 1)"))
        
        self.h_spin = QSpinBox(); self.h_spin.setValue(10)
        self.w_spin = QSpinBox(); self.w_spin.setValue(8)
        controls_layout.addWidget(QLabel("Height:")); controls_layout.addWidget(self.h_spin)
        controls_layout.addWidget(QLabel("Width:")); controls_layout.addWidget(self.w_spin)
        
        controls_layout.addWidget(QLabel("Generation Algorithm:"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(['random', 'gradient', 'sine_waves', 'voronoi', 'perlin'])
        controls_layout.addWidget(self.pattern_combo)
        
        act_group = QGroupBox("Allowed Activations (Hue Types)")
        act_layout = QVBoxLayout()
        self.act_checks = []
        for choice in FFNetworkLibrary.get_activation_choices():
            cb = QCheckBox(f"{choice['name']} (Hue: {int(choice['hue'])}°)")
            cb.setChecked(True)
            self.act_checks.append((choice['index'], choice['hue'], cb))
            act_layout.addWidget(cb)
        act_group.setLayout(act_layout)
        controls_layout.addWidget(act_group)
        
        gen_btn = QPushButton("🎲 Generate HSV Image")
        gen_btn.setStyleSheet("background-color: #2A9D8F; color: white; padding: 10px; font-weight: bold;")
        gen_btn.clicked.connect(self.generate)
        controls_layout.addWidget(gen_btn)
        controls_layout.addStretch()
        
        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("No image generated yet")
        self.ax.axis('off')
        
        splitter.addWidget(controls)
        splitter.addWidget(self.canvas)
        layout.addWidget(splitter)

    def get_allowed_info(self):
        indices, hues = [], []
        for idx, hue, cb in self.act_checks:
            if cb.isChecked():
                indices.append(idx)
                hues.append(hue)
        return indices, hues

    def generate(self):
        h, w = self.h_spin.value(), self.w_spin.value()
        pattern = self.pattern_combo.currentText()
        allowed_indices, allowed_hues = self.get_allowed_info()
        
        if not allowed_hues:
            self.main_window.statusBar().showMessage("Error: Select at least one activation function.")
            return
            
        self.current_image = HSVImageLibrary.generate_image(h, w, pattern=pattern, allowed_hues=allowed_hues)
        self.main_window.statusBar().showMessage(f"Generated '{pattern}' image restricted to {len(allowed_hues)} hues.")
        
        self.ax.clear()
        H, S, V = self.current_image
        rgb = matplotlib.colors.hsv_to_rgb(np.stack([H/360.0, S, V], axis=-1))
        self.ax.imshow(rgb, aspect='auto')
        self.ax.set_title(f"Generated {h}x{w} HSV Image")
        self.canvas.draw()

class NetworkGenerationTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.network_data = None
        self.experiment_results = None
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(QLabel("🧠 Network Generation (Lib 2)"))
        
        single_group = QGroupBox("Single Network Inspector")
        single_layout = QVBoxLayout(single_group)
        self.bias_combo = QComboBox(); self.bias_combo.addItems(['centered', 'linear'])
        single_layout.addWidget(QLabel("Bias Mapping:")); single_layout.addWidget(self.bias_combo)
        
        build_btn = QPushButton("🏗️ Build Single Network")
        build_btn.setStyleSheet("background-color: #264653; color: white; padding: 10px; font-weight: bold;")
        build_btn.clicked.connect(self.build)
        single_layout.addWidget(build_btn)
        controls_layout.addWidget(single_group)

        exp_group = QGroupBox("🚀 Bulk Data Generation (For Charts)")
        exp_layout = QVBoxLayout(exp_group)
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(['make_moons', 'make_circles', 'xor', 'make_classification'])
        exp_layout.addWidget(QLabel("Dataset:")); exp_layout.addWidget(self.dataset_combo)
        
        self.trials_spin = QSpinBox(); self.trials_spin.setRange(10, 5000); self.trials_spin.setValue(100)
        exp_layout.addWidget(QLabel("Trials:")); exp_layout.addWidget(self.trials_spin)
        
        self.noise_spin = QDoubleSpinBox(); self.noise_spin.setRange(0.0, 1.0); self.noise_spin.setValue(0.25); self.noise_spin.setSingleStep(0.05)
        exp_layout.addWidget(QLabel("Dataset Noise:")); exp_layout.addWidget(self.noise_spin)
        
        self.run_exp_btn = QPushButton("▶ Run Experiment (Generate Data)")
        self.run_exp_btn.setStyleSheet("background-color: #2A9D8F; color: white; padding: 10px; font-weight: bold;")
        self.run_exp_btn.clicked.connect(self.run_experiment)
        exp_layout.addWidget(self.run_exp_btn)
        controls_layout.addWidget(exp_group)
        
        controls_layout.addStretch()
        self.log_label = QLabel("Status: Waiting...")
        controls_layout.addWidget(self.log_label)
        
        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("No network built yet")
        self.ax.axis('off')
        
        splitter.addWidget(controls)
        splitter.addWidget(self.canvas)
        layout.addWidget(splitter)

    def build(self):
        img_tab = self.main_window.image_gen_tab
        if not img_tab.current_image:
            self.log_label.setText("Status: Generate an image first!")
            return
            
        H, S, V = img_tab.current_image
        allowed_indices, _ = img_tab.get_allowed_info()
        
        self.network_data = FFNetworkLibrary.build_network(
            H, S, V, bias_mapping=self.bias_combo.currentText(), allowed_indices=allowed_indices
        )
        self.log_label.setText("Status: Network built successfully.")
        
        self.ax.clear()
        self.ax.imshow(self.network_data[0], cmap='viridis', aspect='auto')
        self.ax.set_title("Network Weights")
        self.canvas.draw()

    def run_experiment(self):
        img_tab = self.main_window.image_gen_tab
        if not img_tab.current_image:
            self.log_label.setText("Status: Generate an image config first!")
            return
            
        allowed_indices, allowed_hues = img_tab.get_allowed_info()
        
        config = {
            'height': img_tab.h_spin.value(),
            'width': img_tab.w_spin.value(),
            'pattern': img_tab.pattern_combo.currentText(),
            'bias_mapping': self.bias_combo.currentText(),
            'dataset': self.dataset_combo.currentText(),
            'noise': self.noise_spin.value(),
            'trials': self.trials_spin.value(),
            'n_samples': 500
        }
        
        self.log_label.setText(f"Status: Running {config['trials']} trials...")
        self.run_exp_btn.setEnabled(False)
        
        self.worker = ExperimentWorker(config, allowed_hues, allowed_indices)
        self.worker.finished.connect(self.on_experiment_finished)
        self.worker.error.connect(self.on_experiment_error)
        self.worker.start()

    def on_experiment_finished(self, results):
        self.experiment_results = results
        self.network_data = results['best_network']
        self.log_label.setText(f"Status: Done! Best BCE: {results['best_bce']:.4f}")
        self.run_exp_btn.setEnabled(True)
        self.main_window.statusBar().showMessage("Experiment data generated. Charts are ready!")

    def on_experiment_error(self, err):
        self.log_label.setText(f"Status: Error - {err}")
        self.run_exp_btn.setEnabled(True)

class ChartsTab(QWidget):
    def __init__(self, main_window, is_image_tab=True):
        super().__init__()
        self.main_window = main_window
        self.is_image_tab = is_image_tab
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        btn_layout = QHBoxLayout()
        
        if self.is_image_tab:
            btn1 = QPushButton("🛰️ Generate CV Analysis")
            btn1.clicked.connect(self.plot_cv)
            btn2 = QPushButton("🧩 Generate Histograms")
            btn2.clicked.connect(self.plot_kernels)
            btn3 = QPushButton("🧮 Image Math (Deriv/Integ)") # NEW BUTTON
            btn3.clicked.connect(self.plot_img_math)
            btn_layout.addWidget(btn1)
            btn_layout.addWidget(btn2)
            btn_layout.addWidget(btn3)
        else:
            btn1 = QPushButton("🏗️ Full Network Inspector")
            btn1.clicked.connect(self.plot_arch)
            btn2 = QPushButton("📉 Loss History & Metrics")
            btn2.clicked.connect(self.plot_loss_history)
            btn3 = QPushButton("🧮 Activations Math (Deriv/Integ)")
            btn3.clicked.connect(self.plot_act_math)
            btn_layout.addWidget(btn1)
            btn_layout.addWidget(btn2)
            btn_layout.addWidget(btn3)
            
        layout.addLayout(btn_layout)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.v_layout = QVBoxLayout(self.container)
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)

    def plot_cv(self):
        img = self.main_window.image_gen_tab.current_image
        if not img: return
        fig = ImagePlotLibrary.plot_cv_analysis(*img, ThemeManager.LIGHT)
        self._display_figure(fig)

    def plot_kernels(self):
        img = self.main_window.image_gen_tab.current_image
        if not img: return
        fig = ImagePlotLibrary.plot_kernels(*img, ThemeManager.LIGHT)
        self._display_figure(fig)

    def plot_img_math(self):
        img = self.main_window.image_gen_tab.current_image
        if not img: return
        fig = ImagePlotLibrary.plot_image_math(*img, ThemeManager.LIGHT)
        self._display_figure(fig)

    def plot_arch(self):
        if not self.main_window.network_gen_tab.network_data: return
        best_net = self.main_window.network_gen_tab.network_data
        fig = NetworkPlotLibrary.plot_architecture(best_net['weights'], best_net['biases'], best_net['act_indices'], ThemeManager.LIGHT)
        self._display_figure(fig)

    def plot_loss_history(self):
        if not self.main_window.network_gen_tab.experiment_results: return
        losses = self.main_window.network_gen_tab.experiment_results['all_losses']
        fig = NetworkPlotLibrary.plot_loss_history(losses, ThemeManager.LIGHT)
        self._display_figure(fig)

    def plot_act_math(self):
        if not self.main_window.network_gen_tab.network_data: return
        act_indices = self.main_window.network_gen_tab.network_data['act_indices']
        fig = NetworkPlotLibrary.plot_activation_math(act_indices, ThemeManager.LIGHT)
        self._display_figure(fig)

    def _display_figure(self, fig):
        for i in reversed(range(self.v_layout.count())): 
            self.v_layout.itemAt(i).widget().setParent(None)
        canvas = FigureCanvas(fig)
        self.v_layout.addWidget(canvas)

class DatasetTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.current_dataset = None
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(QLabel("📊 Dataset Configuration"))
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(['make_moons', 'make_circles', 'xor', 'make_classification'])
        controls_layout.addWidget(QLabel("Dataset:")); controls_layout.addWidget(self.dataset_combo)
        
        self.samples_spin = QSpinBox(); self.samples_spin.setRange(50, 10000); self.samples_spin.setValue(500)
        controls_layout.addWidget(QLabel("Samples:")); controls_layout.addWidget(self.samples_spin)
        
        self.noise_spin = QDoubleSpinBox(); self.noise_spin.setRange(0.0, 1.0); self.noise_spin.setValue(0.25); self.noise_spin.setSingleStep(0.05)
        controls_layout.addWidget(QLabel("Noise:")); controls_layout.addWidget(self.noise_spin)
        
        gen_btn = QPushButton("🔄 Load & Plot Dataset")
        gen_btn.setStyleSheet("background-color: #264653; color: white; padding: 10px; font-weight: bold;")
        gen_btn.clicked.connect(self.generate)
        controls_layout.addWidget(gen_btn)
        controls_layout.addStretch()
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.v_layout = QVBoxLayout(self.container)
        self.scroll.setWidget(self.container)
        
        splitter.addWidget(controls)
        splitter.addWidget(self.scroll)
        layout.addWidget(splitter)

    def generate(self):
        config = {
            'dataset': self.dataset_combo.currentText(),
            'n_samples': self.samples_spin.value(),
            'noise': self.noise_spin.value()
        }
        
        X_train, X_test, y_train, y_test = FFNetworkLibrary.load_dataset(
            config['dataset'], config['n_samples'], config['noise']
        )
        
        X = np.vstack((X_train, X_test))
        y = np.hstack((y_train, y_test))
        
        self.current_dataset = {'X': X, 'y': y, 'config': config}
        
        fig = DatasetPlotLibrary.plot_dataset_analysis(X, y, config, ThemeManager.LIGHT)
        
        for i in reversed(range(self.v_layout.count())): 
            self.v_layout.itemAt(i).widget().setParent(None)
            
        canvas = FigureCanvas(fig)
        self.v_layout.addWidget(canvas)
        self.main_window.statusBar().showMessage(f"Dataset '{config['dataset']}' loaded and plotted.")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image-to-Neural Network Explorer (Upgraded Image Charts)")
        self.resize(1200, 800)
        
        tabs = QTabWidget()
        self.image_gen_tab = ImageGenerationTab(self)
        self.network_gen_tab = NetworkGenerationTab(self)
        self.image_charts_tab = ChartsTab(self, is_image_tab=True)
        self.network_charts_tab = ChartsTab(self, is_image_tab=False)
        self.dataset_tab = DatasetTab(self) 
        
        tabs.addTab(self.image_gen_tab, "1. Image Gen Lib")
        tabs.addTab(self.network_gen_tab, "2. Network Gen Lib")
        tabs.addTab(self.image_charts_tab, "3. Image Plots Lib")
        tabs.addTab(self.network_charts_tab, "4. Network Plots Lib")
        tabs.addTab(self.dataset_tab, "5. Dataset Charts Lib") 
        
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Ready")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())