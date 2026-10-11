import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. ACTIVATION FUNCTIONS (24 Types)
# ─────────────────────────────────────────────
def relu(x): return np.maximum(0, x)
def linear(x): return x
def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
def prelu(x, alpha=0.25): return np.where(x > 0, x, alpha * x)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))
def tanh_act(x): return np.tanh(np.clip(x, -100, 100))
def softsign(x): return x / (1.0 + np.abs(x))
def elu(x, alpha=1.0): return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -100, 0)) - 1.0))
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

def hue_to_activation(hue_deg):
    for i, (threshold, func, name) in enumerate(ACTIVATION_MAP):
        if hue_deg < threshold: return func, name, i
    return linear, "Linear", 5


# ─────────────────────────────────────────────
# 2. THE VALUE (BIAS) ENGINE
# ─────────────────────────────────────────────
class BiasConfig:
    """Configuration for how the Value channel is transformed into biases."""
    def __init__(self, mapping='centered', spatial=False, depth_scale=True, act_aware=True):
        self.mapping = mapping      # 'raw', 'centered', 'sigmoidal', 'gaussian'
        self.spatial = spatial      # Use local contrast (edges) instead of absolute V
        self.depth_scale = depth_scale # Scale by 1/sqrt(depth)
        self.act_aware = act_aware  # Auto-scale based on activation function

class BiasEngine:
    @staticmethod
    def compute_layer_bias(V_map, layer_idx, config):
        # 1. Base Mapping
        if config.mapping == 'centered':
            bias = 2.0 * V_map - 1.0
        elif config.mapping == 'sigmoidal':
            bias = 2.0 * (1.0 / (1.0 + np.exp(-12 * (V_map - 0.5)))) - 1.0
        elif config.mapping == 'gaussian':
            bias = np.exp(-((V_map - 0.5) ** 2) / 0.05) - 0.5
        else: # raw
            bias = V_map.copy()

        # 2. Spatial Context (Local Contrast / High-Pass Filter)
        if config.spatial:
            pad = 1
            padded = np.pad(V_map, pad, mode='edge')
            # Calculate 3x3 local mean
            local_mean = (padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                          padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
                          padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]) / 9.0
            # Bias becomes the difference from the local neighborhood (Edge detection)
            bias = V_map - local_mean 

        # 3. Depth Scaling (Prevents exploding gradients in deep nets)
        if config.depth_scale:
            scale = 1.0 / np.sqrt(layer_idx + 1)
            bias *= scale

        return bias


# ─────────────────────────────────────────────
# 3. NETWORK ARCHITECTURE (Vectorized)
# ─────────────────────────────────────────────
class Neuron:
    def __init__(self, row, col, weight, base_bias, activation_name, act_fn, config):
        self.weight = weight
        self.activation_name = activation_name
        self.activation_fn = act_fn
        
        # 4. Activation-Aware Injection
        if config.act_aware:
            if activation_name in ['Sigmoid', 'Tanh', 'HardSigmoid', 'Softsign']:
                self.effective_bias = base_bias * 0.2 # Prevent saturation
            elif activation_name in ['Step', 'Sign']:
                self.effective_bias = base_bias * 2.0 # Ensure threshold crossing
            else:
                self.effective_bias = base_bias
        else:
            self.effective_bias = base_bias

class Layer:
    def __init__(self, layer_index, V_map, H_map, S_map, bias_config):
        self.index = layer_index
        self.neurons = []
        
        # Compute the spatial/depth-mapped bias for the whole layer at once
        self.base_bias_map = BiasEngine.compute_layer_bias(V_map, layer_index, bias_config)
        
        height, width = V_map.shape
        for r in range(height):
            for c in range(width):
                # Note: We only use the column 'c' for the current layer's pixels
                # But V_map is the whole image. We extract the specific column.
                pass 
                
        # Actually, V_map passed here should be the whole image, but we only build 
        # neurons for the current column. Let's fix the build logic in the Network class.

class ImageFeedForwardNetwork:
    def __init__(self, bias_config):
        self.layers = []
        self.bias_config = bias_config

    def build_from_image(self, H, S, V):
        height, width = H.shape
        self.num_layers = width
        self.neurons_per_layer = height

        for col in range(width):
            layer = Layer(layer_index=col, V_map=V, H_map=H, S_map=S, bias_config=self.bias_config)
            
            # Build neurons specifically for this column
            for row in range(height):
                hue = H[row, col]
                act_fn, act_name, _ = hue_to_activation(hue)
                weight = S[row, col]
                base_bias = layer.base_bias_map[row, col]
                
                neuron = Neuron(row, col, weight, base_bias, act_name, act_fn, self.bias_config)
                layer.neurons.append(neuron)
                
            self.layers.append(layer)

    def forward(self, input_data=None):
        current = input_data
        for layer in self.layers:
            if current is None:
                inputs = np.zeros(len(layer.neurons))
            else:
                inputs = np.full(len(layer.neurons), np.sum(current))
                
            # Vectorized forward pass
            weights = np.array([n.weight for n in layer.neurons])
            biases = np.array([n.effective_bias for n in layer.neurons])
            
            z = weights * inputs + biases
            
            outputs = np.zeros_like(z)
            for i, n in enumerate(layer.neurons):
                outputs[i] = n.activation_fn(z[i])
                
            current = outputs
        return current

    def summary(self):
        print("=" * 75)
        print(f"  IMAGE-DERIVED NEURAL NETWORK (Enhanced Value Engine)")
        print(f"  Layers: {self.num_layers}  |  Neurons/layer: {self.neurons_per_layer}")
        print(f"  Bias Config: Map={self.bias_config.mapping}, Spatial={self.bias_config.spatial}, "
              f"DepthScale={self.bias_config.depth_scale}, ActAware={self.bias_config.act_aware}")
        print("=" * 75)


# ─────────────────────────────────────────────
# 4. VISUALIZATION (Bias Pipeline)
# ─────────────────────────────────────────────
def visualize_network(network, H, S, V, image_path):
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle("Enhanced Value (Bias) Pipeline & Network Mapping", fontsize=16, fontweight="bold")

    # 1. Original Image
    orig = np.array(Image.open(image_path).convert("RGB"))
    axes[0, 0].imshow(orig)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # 2. Raw Value Channel
    im0 = axes[0, 1].imshow(V, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("1. Raw Value Channel (0-1)")
    axes[0, 1].axis("off")

    # 3. Mapped Bias (Layer 0 example)
    layer0 = network.layers[0]
    mapped_bias = layer0.base_bias_map
    im1 = axes[0, 2].imshow(mapped_bias, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0, 2].set_title("2. Mapped Bias (Centered/Gaussian)")
    axes[0, 2].axis("off")
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046)

    # 4. Activation Function Map
    cmap = plt.cm.hsv
    act_rgb = np.zeros((*H.shape, 3))
    for r in range(H.shape[0]):
        for c in range(H.shape[1]):
            _, _, idx = hue_to_activation(H[r, c])
            act_rgb[r, c] = cmap(idx / 24.0)[:3]
    axes[1, 0].imshow(act_rgb)
    axes[1, 0].set_title("Hue → Activation Functions (24 Types)")
    axes[1, 0].axis("off")

    # 5. Saturation (Weights)
    im2 = axes[1, 1].imshow(S, cmap="hot", vmin=0, vmax=1)
    axes[1, 1].set_title("Saturation → Weights")
    axes[1, 1].axis("off")
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    # 6. Final Effective Bias (Layer 0, after Act-Aware scaling)
    final_bias = np.array([[n.effective_bias for n in layer.neurons] for layer in network.layers[:1]])[0]
    im3 = axes[1, 2].imshow(final_bias, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 2].set_title("3. Final Effective Bias (Act-Aware)")
    axes[1, 2].axis("off")
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)

    # 7, 8, 9: Network Outputs & Stats
    output = network.forward()
    axes[2, 0].bar(range(len(output)), output, color="steelblue")
    axes[2, 0].set_title("Final Layer Activations")
    axes[2, 0].set_xlabel("Neuron Index")
    
    # Histogram of all biases in the network
    all_biases = np.array([n.effective_bias for layer in network.layers for n in layer.neurons])
    axes[2, 1].hist(all_biases, bins=50, color="purple", alpha=0.7)
    axes[2, 1].set_title("Distribution of All Biases in Network")
    axes[2, 1].set_xlabel("Bias Value")
    axes[2, 1].set_ylabel("Frequency")

    # Depth scaling visualization
    depths = np.arange(network.num_layers)
    scales = 1.0 / np.sqrt(depths + 1)
    axes[2, 2].plot(depths, scales, marker='o', color="green")
    axes[2, 2].set_title("Depth-Adaptive Scaling Factor")
    axes[2, 2].set_xlabel("Layer Depth")
    axes[2, 2].set_ylabel("Scale Multiplier")

    plt.tight_layout()
    plt.savefig("network_enhanced_value_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
# 5. EXECUTION
# ─────────────────────────────────────────────
if __name__ == "__main__":
    sample_path = "sample_input.png"
    try:
        Image.open(sample_path)
    except FileNotFoundError:
        print("  Creating sample image...")
        w, h = 60, 40
        img_array = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                img_array[y, x] = [
                    int(255 * (x / w)), 
                    int(255 * (y / h)), 
                    int(128 + 127 * np.sin(x * y / 50))
                ]
        Image.fromarray(img_array).save(sample_path)

    img = Image.open(sample_path).convert("RGB")
    img_hsv = img.convert("HSV")
    hsv_array = np.array(img_hsv)

    H = hsv_array[:, :, 0].astype(float) / 255.0 * 360.0
    S = hsv_array[:, :, 1].astype(float) / 255.0
    V = hsv_array[:, :, 2].astype(float) / 255.0

    # ── EXPERIMENT WITH DIFFERENT BIAS CONFIGS ──
    # Try changing these parameters to see how the network behavior shifts!
    config = BiasConfig(
        mapping='centered',     # Options: 'raw', 'centered', 'sigmoidal', 'gaussian'
        spatial=True,           # True = Edge-aware bias, False = Absolute brightness bias
        depth_scale=True,       # True = Scale down in deeper layers
        act_aware=True          # True = Auto-adjust for Sigmoid/Tanh/Step
    )

    network = ImageFeedForwardNetwork(bias_config=config)
    network.build_from_image(H, S, V)
    network.summary()

    print("\n  Running forward pass...")
    output = network.forward()
    print(f"  Output mean: {np.mean(output):.4f}, std: {np.std(output):.4f}")

    visualize_network(network, H, S, V, sample_path)