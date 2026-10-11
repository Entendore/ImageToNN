"""
Icosphere-Encoded Neural Network Weight Evolution via Genetic Algorithm
======================================================================
GPU-ACCELERATED VERSION (CuPy + Numba)
======================================================================
Acceleration Stack:
  - CuPy: GPU array operations (matrix math, forward passes)
  - Numba: JIT compilation for Python loops and custom kernels
  - Falls back to NumPy + pure Python if libraries unavailable
"""

# =====================================================================
#  BACKEND DETECTION & IMPORTS
# =====================================================================

try:
    import cupy as xp
    CUPY_AVAILABLE = True
    print("✅ CuPy detected: GPU acceleration enabled.")
except ImportError:
    import numpy as xp
    CUPY_AVAILABLE = False
    print("⚠️ CuPy not found. Falling back to NumPy (CPU).")

try:
    from numba import njit, cuda
    from numba import types as nb_types
    NUMBA_AVAILABLE = True
    print("✅ Numba detected: JIT compilation enabled.")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️ Numba not found. Using pure Python (slower).")
    # Mock decorators for fallback
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

import numpy as np  # Always need numpy for CPU operations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.filterwarnings("ignore")

# =====================================================================
#  1. ICOSPHERE GENERATION (CPU-bound, JIT-compiled with Numba)
# =====================================================================

@njit(cache=True)
def _subdivide_numba(vertices, faces):
    """Numba-JIT compiled icosphere subdivision (CPU)."""
    n_verts = len(vertices)
    n_faces = len(faces)
    
    # Pre-allocate for worst case (4x faces, 3x vertices)
    new_verts = np.zeros((n_verts * 3, 3), dtype=np.float32)
    new_faces = np.zeros((n_faces * 4, 3), dtype=np.int32)
    
    # Copy initial vertices
    new_verts[:n_verts] = vertices
    
    # Midpoint cache (simple hash table)
    mid_cache = {}
    vert_count = n_verts
    face_count = 0
    
    for f_idx in range(n_faces):
        a, b, c = faces[f_idx]
        
        # Get or create midpoints
        def get_mid(i, j):
            nonlocal vert_count
            key = (min(i, j), max(i, j))
            if key in mid_cache:
                return mid_cache[key]
            
            mid = (vertices[i] + vertices[j]) / 2.0
            mid /= np.linalg.norm(mid)
            mid_cache[key] = vert_count
            new_verts[vert_count] = mid
            vert_count += 1
            return mid_cache[key]
        
        ab = get_mid(a, b)
        bc = get_mid(b, c)
        ca = get_mid(c, a)
        
        # Create 4 new faces
        new_faces[face_count] = [a, ab, ca]
        new_faces[face_count + 1] = [b, bc, ab]
        new_faces[face_count + 2] = [c, ca, bc]
        new_faces[face_count + 3] = [ab, bc, ca]
        face_count += 4
    
    return new_verts[:vert_count], new_faces[:face_count]


def create_icosahedron():
    """Return the 12 vertices and 20 faces of a unit icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=np.float32)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]
    ], dtype=np.int32)
    return verts, faces


def create_icosphere(level=1):
    """Build an icosphere with Numba acceleration."""
    verts, faces = create_icosahedron()
    for _ in range(level):
        if NUMBA_AVAILABLE:
            verts, faces = _subdivide_numba(verts, faces)
        else:
            # Fallback to pure Python
            vert_list = list(verts)
            mid_cache = {}
            new_faces = []
            
            def midpoint(i, j):
                key = (min(i, j), max(i, j))
                if key in mid_cache:
                    return mid_cache[key]
                mid = (vert_list[i] + vert_list[j]) / 2.0
                mid /= np.linalg.norm(mid)
                mid_cache[key] = len(vert_list)
                vert_list.append(mid)
                return mid_cache[key]
            
            for f in faces:
                a, b, c = f
                ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
                new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
            
            verts = np.array(vert_list, dtype=np.float32)
            faces = np.array(new_faces, dtype=np.int32)
    
    # Transfer to GPU if CuPy available
    return xp.asarray(verts, dtype=xp.float32), faces


# =====================================================================
#  2. NUMBA-JIT COMPILLED NEURAL NETWORK OPERATIONS
# =====================================================================

@njit(cache=True, parallel=True)
def _forward_pass_numba(X, weights_list, biases_list, activation_type):
    """
    JIT-compiled forward pass for CPU execution.
    activation_type: 0=linear, 1=tanh, 2=relu, 3=sigmoid
    """
    a = X.copy()
    n_layers = len(weights_list)
    
    for layer_idx in range(n_layers):
        W = weights_list[layer_idx]
        b = biases_list[layer_idx]
        n_out = W.shape[1]
        
        # Matrix multiplication + bias
        z = np.zeros((a.shape[0], n_out), dtype=np.float32)
        for i in range(a.shape[0]):
            for j in range(n_out):
                z[i, j] = b[j]
                for k in range(a.shape[1]):
                    z[i, j] += a[i, k] * W[k, j]
        
        # Activation (skip for output layer)
        if layer_idx < n_layers - 1:
            if activation_type == 1:  # tanh
                a = np.tanh(z)
            elif activation_type == 2:  # relu
                a = np.maximum(0, z)
            elif activation_type == 3:  # sigmoid
                a = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            else:
                a = z
        else:
            a = z
    
    return a


@njit(cache=True, parallel=True)
def _compute_mse_numba(pred, target):
    """JIT-compiled MSE loss computation."""
    diff = pred - target
    return np.mean(diff * diff)


# =====================================================================
#  3. GENERIC FROZEN FEED-FORWARD NEURAL NETWORK
# =====================================================================

class FrozenFFNN:
    def __init__(self, layer_sizes, activation="tanh"):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.shapes = []
        for i in range(len(layer_sizes) - 1):
            self.shapes.append((layer_sizes[i], layer_sizes[i + 1]))
        
        self.total_params = sum((ins * outs) + outs for ins, outs in self.shapes)
        self.weights = []
        self.biases = []
        
        # Map activation to int for Numba
        self.activation_id = {"linear": 0, "tanh": 1, "relu": 2, "sigmoid": 3}.get(activation, 0)

    def load_magnitudes(self, magnitudes):
        """Map a 1-D magnitude array onto the network's weights and biases."""
        m = xp.asarray(magnitudes, dtype=xp.float32).ravel()
        if len(m) < self.total_params:
            reps = self.total_params // len(m) + 1
            m = xp.tile(m, reps)
        m = m[:self.total_params]

        self.weights = []
        self.biases = []
        idx = 0
        for ins, outs in self.shapes:
            n_w = ins * outs
            self.weights.append(m[idx:idx + n_w].reshape((ins, outs)))
            idx += n_w
            self.biases.append(m[idx:idx + outs])
            idx += outs

    def forward(self, X):
        """Forward pass with automatic backend selection."""
        if CUPY_AVAILABLE:
            # Use CuPy (GPU)
            a = X
            n_layers = len(self.shapes)
            for i, (ins, outs) in enumerate(self.shapes):
                z = a @ self.weights[i] + self.biases[i]
                if i < n_layers - 1:
                    if self.activation == "tanh":
                        a = xp.tanh(z)
                    elif self.activation == "relu":
                        a = xp.maximum(0, z)
                    elif self.activation == "sigmoid":
                        a = 1.0 / (1.0 + xp.exp(-xp.clip(z, -500, 500)))
                    else:
                        a = z
                else:
                    a = z
            return a
        elif NUMBA_AVAILABLE:
            # Use Numba JIT (CPU)
            X_cpu = X.get() if hasattr(X, 'get') else X
            weights_cpu = [w.get() if hasattr(w, 'get') else w for w in self.weights]
            biases_cpu = [b.get() if hasattr(b, 'get') else b for b in self.biases]
            result = _forward_pass_numba(X_cpu, weights_cpu, biases_cpu, self.activation_id)
            return xp.asarray(result)
        else:
            # Pure NumPy fallback
            a = X
            n_layers = len(self.shapes)
            for i, (ins, outs) in enumerate(self.shapes):
                z = a @ self.weights[i] + self.biases[i]
                if i < n_layers - 1:
                    if self.activation == "tanh":
                        a = np.tanh(z)
                    elif self.activation == "relu":
                        a = np.maximum(0, z)
                    elif self.activation == "sigmoid":
                        a = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
                    else:
                        a = z
                else:
                    a = z
            return a

    def compute_loss(self, X, Y):
        """Universal Mean Squared Error loss."""
        pred = self.forward(X)
        if CUPY_AVAILABLE:
            return xp.mean(xp.square(pred - Y))
        elif NUMBA_AVAILABLE:
            pred_cpu = pred.get() if hasattr(pred, 'get') else pred
            Y_cpu = Y.get() if hasattr(Y, 'get') else Y
            return xp.asarray(_compute_mse_numba(pred_cpu, Y_cpu))
        else:
            return np.mean((pred - Y) ** 2)


# =====================================================================
#  4. GENETIC ALGORITHM (Numba-optimized breeding)
# =====================================================================

@njit(cache=True, parallel=True)
def _tournament_selection_numba(fitness, tournament_k, pop_size):
    """JIT-compiled tournament selection."""
    selected = np.zeros(pop_size, dtype=np.int32)
    for i in range(pop_size):
        best_idx = 0
        best_fit = fitness[0]
        for _ in range(tournament_k):
            idx = np.random.randint(0, pop_size)
            if fitness[idx] > best_fit:
                best_fit = fitness[idx]
                best_idx = idx
        selected[i] = best_idx
    return selected


@njit(cache=True, parallel=True)
def _crossover_numba(population, parent_indices, crossover_rate, n_genes):
    """JIT-compiled crossover operation."""
    new_pop = np.zeros_like(population)
    for i in range(len(parent_indices) // 2):
        p1_idx = parent_indices[i * 2]
        p2_idx = parent_indices[i * 2 + 1]
        
        if np.random.rand() < crossover_rate:
            for g in range(n_genes):
                if np.random.rand() < 0.5:
                    new_pop[i * 2, g] = population[p1_idx, g]
                    new_pop[i * 2 + 1, g] = population[p2_idx, g]
                else:
                    new_pop[i * 2, g] = population[p2_idx, g]
                    new_pop[i * 2 + 1, g] = population[p1_idx, g]
        else:
            new_pop[i * 2] = population[p1_idx]
            new_pop[i * 2 + 1] = population[p2_idx]
    
    return new_pop


@njit(cache=True, parallel=True)
def _mutation_numba(population, mutation_rate, mutation_scale):
    """JIT-compiled mutation operation."""
    for i in range(population.shape[0]):
        for g in range(population.shape[1]):
            if np.random.rand() < mutation_rate:
                population[i, g] += np.random.randn() * mutation_scale
    return population


class GeneticAlgorithm:
    def __init__(self, pop_size, n_genes, mutation_rate=0.15, 
                 mutation_scale=0.3, crossover_rate=0.7, tournament_k=5, elitism=2):
        self.pop_size = pop_size
        self.n_genes = n_genes
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.crossover_rate = crossover_rate
        self.tournament_k = tournament_k
        self.elitism = elitism
        
        self.population = xp.random.randn(pop_size, n_genes).astype(xp.float32) * 0.5

    def step(self, fitness):
        """Evolve one generation with automatic backend selection."""
        if CUPY_AVAILABLE:
            # GPU path (CuPy)
            new_pop = []
            elite_idx = xp.argsort(fitness)[-self.elitism:]
            for ei in elite_idx:
                new_pop.append(self.population[ei].copy())

            while len(new_pop) < self.pop_size:
                idxs = xp.random.choice(self.pop_size, self.tournament_k, replace=False)
                i1 = int(idxs[xp.argmax(fitness[idxs])])
                idxs = xp.random.choice(self.pop_size, self.tournament_k, replace=False)
                i2 = int(idxs[xp.argmax(fitness[idxs])])
                
                if xp.random.rand() < self.crossover_rate:
                    mask = xp.random.rand(self.n_genes) < 0.5
                    c1 = xp.where(mask, self.population[i1], self.population[i2])
                    c2 = xp.where(mask, self.population[i2], self.population[i1])
                else:
                    c1, c2 = self.population[i1].copy(), self.population[i2].copy()
                
                # Mutation
                mask = xp.random.rand(self.n_genes) < self.mutation_rate
                noise = xp.random.randn(self.n_genes).astype(xp.float32) * self.mutation_scale
                c1 = c1 + mask * noise
                new_pop.append(c1)
                
                if len(new_pop) < self.pop_size:
                    mask = xp.random.rand(self.n_genes) < self.mutation_rate
                    noise = xp.random.randn(self.n_genes).astype(xp.float32) * self.mutation_scale
                    c2 = c2 + mask * noise
                    new_pop.append(c2)

            self.population = xp.array(new_pop[:self.pop_size])
        
        elif NUMBA_AVAILABLE:
            # CPU path with Numba JIT
            pop_cpu = self.population.get() if hasattr(self.population, 'get') else self.population
            fit_cpu = fitness.get() if hasattr(fitness, 'get') else fitness
            
            # Tournament selection
            parent_indices = _tournament_selection_numba(fit_cpu, self.tournament_k, self.pop_size)
            
            # Crossover
            new_pop = _crossover_numba(pop_cpu, parent_indices, self.crossover_rate, self.n_genes)
            
            # Mutation
            new_pop = _mutation_numba(new_pop, self.mutation_rate, self.mutation_scale)
            
            # Elitism
            elite_idx = np.argsort(fit_cpu)[-self.elitism:]
            for i, ei in enumerate(elite_idx):
                new_pop[i] = pop_cpu[ei]
            
            self.population = xp.asarray(new_pop)
        
        else:
            # Pure Python fallback
            new_pop = []
            elite_idx = np.argsort(fitness)[-self.elitism:]
            for ei in elite_idx:
                new_pop.append(self.population[ei].copy())

            while len(new_pop) < self.pop_size:
                idxs = np.random.choice(self.pop_size, self.tournament_k, replace=False)
                i1 = idxs[np.argmax(fitness[idxs])]
                idxs = np.random.choice(self.pop_size, self.tournament_k, replace=False)
                i2 = idxs[np.argmax(fitness[idxs])]
                
                if np.random.rand() < self.crossover_rate:
                    mask = np.random.rand(self.n_genes) < 0.5
                    c1 = np.where(mask, self.population[i1], self.population[i2])
                    c2 = np.where(mask, self.population[i2], self.population[i1])
                else:
                    c1, c2 = self.population[i1].copy(), self.population[i2].copy()
                
                mask = np.random.rand(self.n_genes) < self.mutation_rate
                c1[mask] += np.random.randn(mask.sum()) * self.mutation_scale
                new_pop.append(c1)
                
                if len(new_pop) < self.pop_size:
                    mask = np.random.rand(self.n_genes) < self.mutation_rate
                    c2[mask] += np.random.randn(mask.sum()) * self.mutation_scale
                    new_pop.append(c2)

            self.population = np.array(new_pop[:self.pop_size])
        
        return self.population


# =====================================================================
#  5. MAIN EVOLUTION LOOP
# =====================================================================

def to_cpu(arr):
    """Safely convert CuPy array to NumPy for Matplotlib plotting."""
    return arr.get() if hasattr(arr, 'get') else arr

def main():
    xp.random.seed(42)
    np.random.seed(42)

    # =================================================================
    # USER DATASET INJECTION POINT
    # =================================================================
    n_samples = 2000
    n_features = 10
    n_targets = 3
    
    X = xp.random.randn(n_samples, n_features).astype(xp.float32)
    Y = xp.zeros((n_samples, n_targets), dtype=xp.float32)
    Y[:, 0] = X[:, 0]**2 + xp.sin(X[:, 1] * X[:, 2])
    Y[:, 1] = X[:, 2] * X[:, 3] - X[:, 0]
    Y[:, 2] = xp.tanh(X[:, 4] + X[:, 5])
    
    Y = (Y - xp.mean(Y, axis=0)) / (xp.std(Y, axis=0) + 1e-8)
    # =================================================================

    LAYER_SIZES = [n_features, 32, 16, n_targets]
    
    SUBDIV_LEVEL = 2  # Higher resolution icosphere (162 vertices)
    unit_dirs, faces = create_icosphere(SUBDIV_LEVEL)
    n_verts = len(unit_dirs)
    
    print(f"\n{'='*60}")
    print(f"Acceleration Stack:")
    print(f"  CuPy (GPU): {'✅' if CUPY_AVAILABLE else '❌'}")
    print(f"  Numba (JIT): {'✅' if NUMBA_AVAILABLE else '❌'}")
    print(f"{'='*60}")
    print(f"Dataset: X shape {X.shape}, Y shape {Y.shape}")
    print(f"Network: {LAYER_SIZES} ({sum((i*o)+o for i, o in FrozenFFNN(LAYER_SIZES).shapes)} params)")
    print(f"Icosphere: {n_verts} vertices -> mapped to network weights\n")

    nn = FrozenFFNN(LAYER_SIZES, activation="tanh")
    ga = GeneticAlgorithm(pop_size=300, n_genes=n_verts, mutation_rate=0.2, mutation_scale=0.25)

    best_loss_history = []
    mean_loss_history = []

    print("Starting accelerated evolution...")
    for gen in range(400):
        fitness = xp.empty(ga.pop_size, dtype=xp.float32)
        for i in range(ga.pop_size):
            nn.load_magnitudes(ga.population[i])
            loss = nn.compute_loss(X, Y)
            fitness[i] = -loss

        best_loss = float(-xp.max(fitness))
        mean_loss = float(-xp.mean(fitness))
        best_loss_history.append(best_loss)
        mean_loss_history.append(mean_loss)

        if gen % 50 == 0 or gen == 399:
            print(f"Gen {gen:4d} | Best Loss: {best_loss:.4f} | Mean Loss: {mean_loss:.4f}")

        ga.step(fitness)

    best_idx = int(xp.argmax(fitness))
    best_mags = ga.population[best_idx]
    nn.load_magnitudes(best_mags)
    final_loss = float(nn.compute_loss(X, Y))
    print(f"\nEvolution complete. Final Best Loss: {final_loss:.4f}")

    # =================================================================
    # VISUALIZATION
    # =================================================================
    best_mags_cpu = to_cpu(best_mags)
    unit_dirs_cpu = to_cpu(unit_dirs)
    X_cpu = to_cpu(X)
    Y_cpu = to_cpu(Y)
    
    nn.load_magnitudes(xp.random.randn(n_verts).astype(xp.float32) * 0.5)
    pred_random_cpu = to_cpu(nn.forward(X)[:, 0])
    
    nn.load_magnitudes(best_mags)
    pred_evolved_cpu = to_cpu(nn.forward(X)[:, 0])
    
    w1_cpu = to_cpu(nn.weights[0])

    fig = plt.figure(figsize=(16, 10))

    # 1. Initial Icosphere
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    pts_init = unit_dirs_cpu * 1.0
    mesh1 = Poly3DCollection(pts_init[faces], alpha=0.4, edgecolor="gray", linewidth=0.2)
    ax1.add_collection3d(mesh1)
    ax1.scatter(pts_init[:, 0], pts_init[:, 1], pts_init[:, 2], s=5, c="k")
    ax1.set_title("Initial Icosphere\n(Magnitudes = 1.0)"); ax1.set_xlim(-2, 2); ax1.set_ylim(-2, 2); ax1.set_zlim(-2, 2)

    # 2. Evolved Icosphere
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    pts_evolved = unit_dirs_cpu * best_mags_cpu[:, None]
    mag_faces = [best_mags_cpu[f].mean() for f in faces]
    colors = plt.cm.coolwarm((mag_faces - min(mag_faces)) / (max(mag_faces) - min(mag_faces) + 1e-8))
    mesh2 = Poly3DCollection(pts_evolved[faces], facecolors=colors, alpha=0.6, edgecolor="k", linewidth=0.2)
    ax2.add_collection3d(mesh2)
    ax2.scatter(pts_evolved[:, 0], pts_evolved[:, 1], pts_evolved[:, 2], s=5, c="k")
    ax2.set_title(f"Evolved Icosphere\n(Loss = {final_loss:.4f})"); ax2.set_xlim(-3, 3); ax2.set_ylim(-3, 3); ax2.set_zlim(-3, 3)

    # 3. Loss Curve
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(best_loss_history, label="Best Loss", lw=2, color="blue")
    ax3.plot(mean_loss_history, label="Mean Loss", lw=1, alpha=0.6, color="orange")
    ax3.set_xlabel("Generation"); ax3.set_ylabel("MSE Loss")
    ax3.set_title("GA Optimization Curve"); ax3.legend(); ax3.grid(True, alpha=0.3)

    # 4. Parity Plot
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(Y_cpu[:, 0], pred_random_cpu, alpha=0.3, s=10, label="Random Init")
    ax4.scatter(Y_cpu[:, 0], pred_evolved_cpu, alpha=0.5, s=10, color="red", label="Evolved")
    min_val, max_val = min(Y_cpu[:, 0].min(), pred_evolved_cpu.min()), max(Y_cpu[:, 0].max(), pred_evolved_cpu.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, label="Perfect Prediction")
    ax4.set_xlabel("Actual Y (Dim 0)"); ax4.set_ylabel("Predicted Y (Dim 0)")
    ax4.set_title("Predicted vs Actual (Generic Parity Plot)"); ax4.legend(); ax4.grid(True, alpha=0.3)

    # 5. Magnitude Distribution
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(best_mags_cpu, bins=40, color="teal", alpha=0.7, edgecolor="black")
    ax5.set_xlabel("Vertex Magnitude (= NN Weight)")
    ax5.set_ylabel("Frequency")
    ax5.set_title("Distribution of Evolved Weights")
    ax5.axvline(0, color="red", ls="--", lw=1)
    ax5.grid(True, alpha=0.3)

    # 6. Weight Matrix Visualization
    ax6 = fig.add_subplot(2, 3, 6)
    im = ax6.imshow(w1_cpu, aspect="auto", cmap="coolwarm")
    ax6.set_title(f"Evolved Weights: Layer 1\nShape: {w1_cpu.shape}")
    ax6.set_xlabel("Output Neurons"); ax6.set_ylabel("Input Features")
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("generic_icosphere_ga_cupy_numba.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()