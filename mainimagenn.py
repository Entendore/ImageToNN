#!/usr/bin/env python3
"""
RGBA Intelligence Search: Discovering the Most Intelligent Solid-Color Image
via Hierarchical Monte Carlo Tree Search

This is a complete, self-contained implementation that:
1. Treats every RGBA color as a neural network configuration
2. Searches the 4.3 billion color space using H-MCTS
3. Evaluates network intelligence via expressivity and benchmarks
4. Produces visualizations of the intelligence landscape

Author: AI Research Assistant
Date: 2026
"""

import os
import sys
import time
import json
import hashlib
import pickle
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import math
import random
import logging

import numpy as np
import yaml
from tqdm import tqdm

# Try to import JAX, fall back to NumPy if not available
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, random
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not available, falling back to NumPy (slower)")
    import numpy as jnp
    def jit(fn): return fn
    def vmap(fn): return fn
    random = np.random

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import accuracy_score
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False

try:
    from scipy import stats
    from scipy.spatial.distance import pdist, squareform
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "project": {
        "name": "rgba_intelligence",
        "seed": 42,
        "output_dir": "./outputs",
        "device": "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
        "dtype": "float32"
    },
    "encoding": {
        "channels": 4,
        "bit_depth": 8,
        "red": {
            "ranges": [
                [0, 85, "relu"],
                [86, 170, "sigmoid"],
                [171, 255, "tanh"]
            ]
        },
        "green": {
            "range": [0.0, 2.0]
        },
        "blue": {
            "range": [-1.0, 1.0]
        },
        "alpha": {
            "ranges": [
                [0, 63, {"depth": 2, "width": 32, "dropout": 0.0}],
                [64, 127, {"depth": 3, "width": 64, "dropout": 0.1}],
                [128, 191, {"depth": 4, "width": 128, "dropout": 0.2}],
                [192, 255, {"depth": 6, "width": 256, "dropout": 0.3}]
            ]
        }
    },
    "network": {
        "input_dim": 784,
        "output_dim": 10
    },
    "mcts": {
        "total_iterations": 1000,
        "rollout_size": 10,
        "exploration_constant": 1.4,
        "max_depth": 6,
        "branching_factor": 2,
        "rave_enabled": True,
        "rave_constant": 1.0,
        "early_stopping": True,
        "early_stopping_threshold": 2.0
    },
    "grid": {
        "step_size": 32
    },
    "evaluation": {
        "batch_size": 100,
        "input_noise_types": ["gaussian", "perlin", "binary"],
        "intelligence_weights": {
            "expressivity": 0.4,
            "representational_capacity": 0.4,
            "stability": 0.2
        }
    },
    "datasets": {
        "xor_16x16": {"enabled": True},
        "mnist_28x28": {"enabled": False}
    },
    "curriculum": {
        "enabled": False,
        "stages": []
    },
    "surrogate": {
        "enabled": False,
        "initial_evaluations": 100,
        "retrain_every": 200
    },
    "refinement": {
        "enabled": False,
        "budget": 100
    },
    "pareto": {
        "enabled": False,
        "iterations": 500
    },
    "statistics": {
        "enabled": True,
        "top_percentile": 5
    },
    "visualization": {
        "enabled": True,
        "formats": ["png"]
    },
    "performance": {
        "cache_enabled": True,
        "cache_max_size_gb": 5
    }
}

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

RGBA = Tuple[int, int, int, int]
ColorBounds = Tuple[RGBA, RGBA]

@dataclass(frozen=True)
class ArchitectureConfig:
    depth: int
    width: int
    dropout: float = 0.0

@dataclass(frozen=True)
class NetworkConfig:
    activation: str
    weight_scale: float
    bias_value: float
    architecture: ArchitectureConfig
    input_dim: int
    output_dim: int

@dataclass
class MCTSNode:
    node_id: int
    bounds: ColorBounds
    depth: int
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    visit_count: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0
    is_expanded: bool = False
    rave_total: float = 0.0
    rave_count: int = 0

@dataclass
class EvaluationResult:
    color: RGBA
    intelligence_score: float
    expressivity: float
    representational_capacity: float
    stability: float
    metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class SearchResult:
    best_color: RGBA
    best_score: float
    total_evaluations: int
    total_time_seconds: float
    tree_size: int
    all_evaluations: List[EvaluationResult] = field(default_factory=list)

# ============================================================================
# CORE MODULE: COLOR ENCODING
# ============================================================================

class ColorEncoder:
    """Encodes RGBA colors into neural network configurations."""
    
    def __init__(self, config: dict):
        self.config = config
        self.red_ranges = config['encoding']['red']['ranges']
        self.green_range = config['encoding']['green']['range']
        self.blue_range = config['encoding']['blue']['range']
        self.alpha_ranges = config['encoding']['alpha']['ranges']
    
    def decode(self, color: RGBA) -> NetworkConfig:
        """Decode RGBA color into NetworkConfig."""
        r, g, b, a = color
        
        # Decode activation (Red)
        activation = "relu"
        for low, high, act in self.red_ranges:
            if low <= r <= high:
                activation = act
                break
        
        # Decode weight scale (Green)
        weight_scale = self._map_range(g, 0, 255, 
                                       self.green_range[0], self.green_range[1])
        
        # Decode bias (Blue)
        bias_value = self._map_range(b, 0, 255,
                                     self.blue_range[0], self.blue_range[1])
        
        # Decode architecture (Alpha)
        architecture = ArchitectureConfig(depth=3, width=128)
        for low, high, arch_dict in self.alpha_ranges:
            if low <= a <= high:
                architecture = ArchitectureConfig(**arch_dict)
                break
        
        return NetworkConfig(
            activation=activation,
            weight_scale=weight_scale,
            bias_value=bias_value,
            architecture=architecture,
            input_dim=self.config['network']['input_dim'],
            output_dim=self.config['network']['output_dim']
        )
    
    @staticmethod
    def _map_range(value: int, in_min: int, in_max: int,
                   out_min: float, out_max: float) -> float:
        normalized = (value - in_min) / (in_max - in_min)
        return out_min + normalized * (out_max - out_min)

# ============================================================================
# CORE MODULE: NETWORK
# ============================================================================

class UPFNN:
    """Uniformly-Parameterized Feedforward Network."""
    
    def __init__(self, config: NetworkConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.params = self._initialize_params()
    
    def _initialize_params(self) -> dict:
        """Initialize base random weights with color-derived scaling."""
        np.random.seed(self.seed)
        params = {}
        cfg = self.config
        
        for i in range(cfg.architecture.depth + 1):
            if i == 0:
                in_dim, out_dim = cfg.input_dim, cfg.architecture.width
            elif i == cfg.architecture.depth:
                in_dim, out_dim = cfg.architecture.width, cfg.output_dim
            else:
                in_dim = out_dim = cfg.architecture.width
            
            # Base random weights
            W_base = np.random.randn(in_dim, out_dim).astype(np.float32)
            
            # Apply global weight scale
            params[f'W_{i}'] = W_base * cfg.weight_scale
            params[f'b_{i}'] = np.full(out_dim, cfg.bias_value, dtype=np.float32)
        
        return params
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        cfg = self.config
        h = x
        
        for i in range(cfg.architecture.depth + 1):
            W = self.params[f'W_{i}']
            b = self.params[f'b_{i}']
            h = h @ W + b
            
            # Apply activation (except on output layer)
            if i < cfg.architecture.depth:
                h = self._apply_activation(h, cfg.activation)
        
        return h
    
    @staticmethod
    def _apply_activation(x: np.ndarray, activation: str) -> np.ndarray:
        if activation == "relu":
            return np.maximum(0, x)
        elif activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif activation == "tanh":
            return np.tanh(x)
        elif activation == "gelu":
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
        else:
            return x

# ============================================================================
# SEARCH MODULE: MCTS TREE
# ============================================================================

class MCTSTree:
    """Memory-efficient MCTS tree."""
    
    def __init__(self, root_bounds: ColorBounds, max_nodes: int = 1_000_000):
        self.max_nodes = max_nodes
        self.nodes: List[Optional[MCTSNode]] = [None] * max_nodes
        self.next_id = 0
        
        # Create root node
        root = MCTSNode(
            node_id=0,
            bounds=root_bounds,
            depth=0,
            parent_id=None
        )
        self.nodes[0] = root
        self.next_id = 1
        self.root_id = 0
    
    @property
    def root(self) -> MCTSNode:
        return self.nodes[self.root_id]
    
    def get_node(self, node_id: int) -> MCTSNode:
        return self.nodes[node_id]
    
    def add_child(self, parent_id: int, bounds: ColorBounds) -> MCTSNode:
        """Add a child node."""
        if self.next_id >= self.max_nodes:
            raise MemoryError("Tree capacity exceeded")
        
        parent = self.nodes[parent_id]
        child = MCTSNode(
            node_id=self.next_id,
            bounds=bounds,
            depth=parent.depth + 1,
            parent_id=parent_id
        )
        self.nodes[self.next_id] = child
        parent.children_ids.append(self.next_id)
        self.next_id += 1
        return child
    
    def update_node(self, node_id: int, reward: float):
        """Update node statistics."""
        node = self.nodes[node_id]
        node.visit_count += 1
        node.total_reward += reward
        node.mean_reward = node.total_reward / node.visit_count
    
    def partition_space(self, node_id: int, branching: int = 2):
        """Partition node's bounding box into children."""
        node = self.nodes[node_id]
        if node.is_expanded:
            return
        
        min_corner, max_corner = node.bounds
        children_bounds = self._subdivide_bounds(min_corner, max_corner, branching)
        
        for child_bounds in children_bounds:
            self.add_child(node_id, child_bounds)
        
        node.is_expanded = True
    
    @staticmethod
    def _subdivide_bounds(min_corner: RGBA, max_corner: RGBA,
                          branching: int) -> List[ColorBounds]:
        """Subdivide 4D bounding box."""
        bounds_list = []
        
        def generate_splits(channel_idx, current_min, current_max):
            if channel_idx == 4:
                bounds_list.append((tuple(current_min), tuple(current_max)))
                return
            
            low = min_corner[channel_idx]
            high = max_corner[channel_idx]
            step = (high - low) / branching
            
            for i in range(branching):
                new_min = list(current_min)
                new_max = list(current_max)
                new_min[channel_idx] = int(low + i * step)
                new_max[channel_idx] = int(low + (i + 1) * step)
                generate_splits(channel_idx + 1, new_min, new_max)
        
        generate_splits(0, list(min_corner), list(max_corner))
        return bounds_list
    
    def get_best_node(self) -> MCTSNode:
        """Return node with highest mean reward."""
        best = None
        best_score = -float('inf')
        for node in self.nodes[:self.next_id]:
            if node is not None and node.visit_count > 0:
                if node.mean_reward > best_score:
                    best_score = node.mean_reward
                    best = node
        return best
    
    @staticmethod
    def center_of_bounds(node: MCTSNode) -> RGBA:
        """Get the center color of a node's bounding box."""
        min_c, max_c = node.bounds
        return tuple((min_c[i] + max_c[i]) // 2 for i in range(4))
    
    @property
    def size(self) -> int:
        return self.next_id

# ============================================================================
# SEARCH MODULE: MCTS POLICIES
# ============================================================================

class UCB1Policy:
    """Upper Confidence Bound policy."""
    
    def __init__(self, exploration_constant: float = 1.4):
        self.C = exploration_constant
    
    def select_child(self, parent: MCTSNode, tree: MCTSTree) -> MCTSNode:
        """Select child with highest UCB1 score."""
        best_child = None
        best_score = -float('inf')
        
        for child_id in parent.children_ids:
            child = tree.get_node(child_id)
            if child.visit_count == 0:
                return child
            
            ucb1 = child.mean_reward + self.C * math.sqrt(
                math.log(parent.visit_count + 1) / (child.visit_count + 1)
            )
            
            if ucb1 > best_score:
                best_score = ucb1
                best_child = child
        
        return best_child

class RAVEPolicy:
    """RAVE-augmented UCB1 policy."""
    
    def __init__(self, exploration_constant: float = 1.4,
                 rave_constant: float = 1.0):
        self.C = exploration_constant
        self.C_rave = rave_constant
        self.rave_table: Dict[Tuple, Tuple[float, int]] = {}
    
    def select_child(self, parent: MCTSNode, tree: MCTSTree) -> MCTSNode:
        best_child = None
        best_score = -float('inf')
        
        for child_id in parent.children_ids:
            child = tree.get_node(child_id)
            
            # UCB1 component
            if child.visit_count == 0:
                ucb = float('inf')
            else:
                ucb = child.mean_reward + self.C * math.sqrt(
                    math.log(parent.visit_count + 1) / (child.visit_count + 1)
                )
            
            # RAVE component
            center = MCTSTree.center_of_bounds(child)
            rave_key = tuple(c // 16 for c in center)
            
            if rave_key in self.rave_table:
                rave_total, rave_count = self.rave_table[rave_key]
                rave_value = rave_total / rave_count if rave_count > 0 else 0.0
            else:
                rave_value = 0.0
            
            # Beta decreases as visits increase
            beta = self.C_rave / math.sqrt(30 + parent.visit_count)
            combined = (1 - beta) * ucb + beta * rave_value
            
            if combined > best_score:
                best_score = combined
                best_child = child
        
        return best_child
    
    def update_rave(self, color: RGBA, reward: float):
        """Update RAVE statistics."""
        key = tuple(c // 16 for c in color)
        if key not in self.rave_table:
            self.rave_table[key] = (0.0, 0)
        total, count = self.rave_table[key]
        self.rave_table[key] = (total + reward, count + 1)

# ============================================================================
# SEARCH MODULE: MCTS ENGINE
# ============================================================================

class HMCTS:
    """Hierarchical Monte Carlo Tree Search."""
    
    def __init__(self, config: dict, evaluator: Callable):
        self.config = config
        self.evaluator = evaluator
        self.mcts_config = config['mcts']
        
        if self.mcts_config.get('rave_enabled', False):
            self.policy = RAVEPolicy(
                exploration_constant=self.mcts_config['exploration_constant'],
                rave_constant=self.mcts_config.get('rave_constant', 1.0)
            )
        else:
            self.policy = UCB1Policy(
                exploration_constant=self.mcts_config['exploration_constant']
            )
        
        self.tree: Optional[MCTSTree] = None
        self.total_evaluations = 0
        self.all_evaluations: List[EvaluationResult] = []
    
    def search(self, root_bounds: ColorBounds = None) -> SearchResult:
        """Run the full MCTS search."""
        if root_bounds is None:
            root_bounds = ((0, 0, 0, 0), (255, 255, 255, 255))
        
        self.tree = MCTSTree(root_bounds)
        total_iters = self.mcts_config['total_iterations']
        max_depth = self.mcts_config['max_depth']
        
        start_time = time.time()
        
        for iteration in tqdm(range(total_iters), desc="MCTS Search"):
            self._run_iteration(max_depth)
        
        best_node = self.tree.get_best_node()
        best_color = MCTSTree.center_of_bounds(best_node)
        
        return SearchResult(
            best_color=best_color,
            best_score=best_node.mean_reward,
            total_evaluations=self.total_evaluations,
            total_time_seconds=time.time() - start_time,
            tree_size=self.tree.size,
            all_evaluations=self.all_evaluations
        )
    
    def _run_iteration(self, max_depth: int):
        """Single MCTS iteration."""
        # Selection
        node = self.tree.root
        while node.children_ids and node.depth < max_depth:
            node = self.policy.select_child(node, self.tree)
        
        # Expansion
        if not node.children_ids and node.depth < max_depth:
            branching = self.mcts_config.get('branching_factor', 2)
            self.tree.partition_space(node.node_id, branching)
            if node.children_ids:
                node = self.tree.get_node(node.children_ids[0])
        
        # Rollout
        rollout_size = self.mcts_config['rollout_size']
        rewards = []
        
        for _ in range(rollout_size):
            color = self._sample_color_from_node(node)
            result = self.evaluator(color)
            rewards.append(result.intelligence_score)
            self.total_evaluations += 1
            self.all_evaluations.append(result)
            
            if isinstance(self.policy, RAVEPolicy):
                self.policy.update_rave(color, result.intelligence_score)
            
            # Early stopping
            if (self.mcts_config.get('early_stopping', False) and
                len(rewards) >= rollout_size // 2):
                best = max(rewards)
                avg = np.mean(rewards)
                if best > self.mcts_config['early_stopping_threshold'] * avg:
                    break
        
        avg_reward = np.mean(rewards) if rewards else 0.0
        
        # Backpropagation
        current = node
        while current is not None:
            self.tree.update_node(current.node_id, avg_reward)
            if current.parent_id is not None:
                current = self.tree.get_node(current.parent_id)
            else:
                break
    
    def _sample_color_from_node(self, node: MCTSNode) -> RGBA:
        """Sample a random color from node's bounding box."""
        min_c, max_c = node.bounds
        color = tuple(
            np.random.randint(min_c[i], max_c[i] + 1)
            for i in range(4)
        )
        return color

# ============================================================================
# EVALUATION MODULE: INTELLIGENCE SCORING
# ============================================================================

class IntelligenceEvaluator:
    """Evaluates network intelligence."""
    
    def __init__(self, config: dict):
        self.config = config
        self.encoder = ColorEncoder(config)
        self.weights = config['evaluation']['intelligence_weights']
        self.batch_size = config['evaluation']['batch_size']
        self.input_dim = config['network']['input_dim']
        self.cache: Dict[RGBA, EvaluationResult] = {}
        self.cache_enabled = config['performance'].get('cache_enabled', True)
    
    def __call__(self, color: RGBA) -> EvaluationResult:
        """Evaluate a single color."""
        # Check cache
        if self.cache_enabled and color in self.cache:
            return self.cache[color]
        
        # Decode color to network config
        net_config = self.encoder.decode(color)
        
        # Create network
        network = UPFNN(net_config, seed=self.config['project']['seed'])
        
        # Generate random inputs
        inputs = self._generate_random_inputs()
        
        # Compute metrics
        expressivity_metrics = self._compute_expressivity(network, inputs)
        rep_capacity = self._compute_representational_capacity(network, inputs)
        stability = self._compute_stability(network, inputs)
        
        # Composite score
        score = (
            self.weights['expressivity'] * expressivity_metrics['composite'] +
            self.weights['representational_capacity'] * rep_capacity +
            self.weights['stability'] * stability
        )
        
        result = EvaluationResult(
            color=color,
            intelligence_score=float(score),
            expressivity=float(expressivity_metrics['composite']),
            representational_capacity=float(rep_capacity),
            stability=float(stability),
            metrics={
                'variance': float(expressivity_metrics['variance']),
                'entropy': float(expressivity_metrics['entropy']),
                'dead_ratio': float(expressivity_metrics['dead_ratio']),
            }
        )
        
        # Cache result
        if self.cache_enabled:
            self.cache[color] = result
        
        return result
    
    def _generate_random_inputs(self) -> np.ndarray:
        """Generate batch of random inputs."""
        return np.random.randn(self.batch_size, self.input_dim).astype(np.float32)
    
    def _compute_expressivity(self, network: UPFNN, 
                             inputs: np.ndarray) -> Dict[str, float]:
        """Compute expressivity metrics."""
        activations = network.forward(inputs)
        
        # Variance
        var_score = np.var(activations, axis=0).mean()
        
        # Entropy
        probs = self._softmax(activations)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1).mean()
        
        # Dead neuron ratio
        dead_ratio = (activations == 0).mean()
        
        # Composite
        composite = (var_score + entropy) / (1 + dead_ratio)
        
        return {
            'variance': var_score,
            'entropy': entropy,
            'dead_ratio': dead_ratio,
            'composite': composite
        }
    
    def _compute_representational_capacity(self, network: UPFNN,
                                          inputs: np.ndarray) -> float:
        """Compute representational capacity via linear probing."""
        if not SKLEARN_AVAILABLE:
            return 0.5  # Default if sklearn not available
        
        # Extract features
        features = network.forward(inputs)
        
        # Create synthetic labels (e.g., based on input statistics)
        labels = (inputs.sum(axis=1) > 0).astype(int)
        
        # Train linear classifier
        clf = LogisticRegression(max_iter=100, random_state=42)
        clf.fit(features, labels)
        
        # Return accuracy
        accuracy = clf.score(features, labels)
        return accuracy
    
    def _compute_stability(self, network: UPFNN,
                          inputs: np.ndarray,
                          noise_std: float = 0.01) -> float:
        """Compute stability metric."""
        outputs_original = network.forward(inputs)
        
        # Add noise
        noise = np.random.randn(*inputs.shape).astype(np.float32) * noise_std
        inputs_perturbed = inputs + noise
        
        outputs_perturbed = network.forward(inputs_perturbed)
        
        # MSE
        mse = np.mean((outputs_original - outputs_perturbed) ** 2)
        
        # Stability (inverse of MSE)
        stability = 1.0 / (1.0 + mse)
        return stability
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# ============================================================================
# DATASETS MODULE
# ============================================================================

class XOR16x16Dataset:
    """XOR dataset with 16x16 images."""
    
    def __init__(self, train_samples_per_class: int = 100,
                 test_samples_per_class: int = 50):
        self.train_samples_per_class = train_samples_per_class
        self.test_samples_per_class = test_samples_per_class
        self.data = self._generate_data()
    
    def _generate_data(self) -> Dict[str, np.ndarray]:
        """Generate XOR patterns."""
        # Canonical patterns (16x16)
        patterns = {
            '00': np.zeros((16, 16), dtype=np.float32),
            '01': np.zeros((16, 16), dtype=np.float32),
            '10': np.zeros((16, 16), dtype=np.float32),
            '11': np.zeros((16, 16), dtype=np.float32)
        }
        
        # Fill quadrants
        patterns['01'][:8, :8] = 1.0  # Top-left
        patterns['10'][:8, 8:] = 1.0  # Top-right
        patterns['11'][:8, :8] = 1.0  # Top-left
        patterns['11'][:8, 8:] = 1.0  # Top-right
        
        labels = {'00': 0, '01': 1, '10': 1, '11': 0}
        
        # Generate training data
        X_train, y_train = [], []
        for pattern_name, pattern in patterns.items():
            for _ in range(self.train_samples_per_class):
                # Add small noise
                augmented = pattern + np.random.randn(16, 16) * 0.1
                augmented = np.clip(augmented, 0, 1)
                X_train.append(augmented.flatten())
                y_train.append(labels[pattern_name])
        
        # Generate test data
        X_test, y_test = [], []
        for pattern_name, pattern in patterns.items():
            for _ in range(self.test_samples_per_class):
                augmented = pattern + np.random.randn(16, 16) * 0.15
                augmented = np.clip(augmented, 0, 1)
                X_test.append(augmented.flatten())
                y_test.append(labels[pattern_name])
        
        return {
            'X_train': np.array(X_train),
            'y_train': np.array(y_train),
            'X_test': np.array(X_test),
            'y_test': np.array(y_test)
        }

# ============================================================================
# OPTIMIZATION MODULE: LOCAL REFINEMENT
# ============================================================================

class LocalRefinement:
    """Local continuous refinement using CMA-ES or Nelder-Mead."""
    
    def __init__(self, evaluator: Callable, budget: int = 100):
        self.evaluator = evaluator
        self.budget = budget
    
    def refine(self, C_initial: RGBA) -> RGBA:
        """Refine the color using local optimization."""
        if not CMA_AVAILABLE:
            logger.warning("CMA-ES not available, returning initial color")
            return C_initial
        
        def objective(c_continuous):
            c_discrete = tuple(int(np.clip(c, 0, 255)) for c in c_continuous)
            result = self.evaluator(c_discrete)
            return -result.intelligence_score  # Minimize negative
        
        # Run CMA-ES
        es = cma.CMAEvolutionStrategy(C_initial, 5.0)
        
        for _ in range(self.budget // 20):
            solutions = es.ask()
            fitnesses = [objective(sol) for sol in solutions]
            es.tell(solutions, fitnesses)
            
            if es.stop():
                break
        
        best = es.result.xbest
        C_refined = tuple(int(np.clip(c, 0, 255)) for c in best)
        
        return C_refined

# ============================================================================
# STATISTICS MODULE
# ============================================================================

class StatisticalAnalyzer:
    """Statistical analysis of results."""
    
    def __init__(self, config: dict):
        self.config = config
    
    def analyze(self, evaluations: List[EvaluationResult]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        if not evaluations:
            return {}
        
        # Convert to arrays
        colors = np.array([e.color for e in evaluations])
        scores = np.array([e.intelligence_score for e in evaluations])
        
        results = {}
        
        # Correlation analysis
        if SCIPY_AVAILABLE:
            correlations = {}
            for i, channel in enumerate(['R', 'G', 'B', 'A']):
                corr, p_value = stats.pearsonr(colors[:, i], scores)
                correlations[channel] = {'correlation': corr, 'p_value': p_value}
            results['correlations'] = correlations
        
        # Top percentile analysis
        top_percentile = self.config['statistics'].get('top_percentile', 5)
        threshold = np.percentile(scores, 100 - top_percentile)
        top_mask = scores >= threshold
        
        results['top_percentile'] = {
            'threshold': threshold,
            'count': top_mask.sum(),
            'mean_score': scores[top_mask].mean(),
            'mean_color': colors[top_mask].mean(axis=0).tolist()
        }
        
        # Basic statistics
        results['basic_stats'] = {
            'total_evaluations': len(evaluations),
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score': scores.max(),
            'best_color': evaluations[np.argmax(scores)].color
        }
        
        return results

# ============================================================================
# VISUALIZATION MODULE
# ============================================================================

class Visualizer:
    """Visualization of results."""
    
    def __init__(self, config: dict, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def visualize(self, evaluations: List[EvaluationResult],
                 tree: Optional[MCTSTree] = None):
        """Generate all visualizations."""
        if not evaluations:
            logger.warning("No evaluations to visualize")
            return
        
        colors = np.array([e.color for e in evaluations])
        scores = np.array([e.intelligence_score for e in evaluations])
        
        # 2D scatter plot
        if MATPLOTLIB_AVAILABLE:
            self._plot_2d_scatter(colors, scores)
        
        # Color distribution
        if MATPLOTLIB_AVAILABLE:
            self._plot_color_distribution(colors, scores)
        
        # Best color visualization
        self._visualize_best_color(evaluations)
        
        logger.info(f"Visualizations saved to {self.output_dir}")
    
    def _plot_2d_scatter(self, colors: np.ndarray, scores: np.ndarray):
        """Plot 2D scatter of R vs G colored by score."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            colors[:, 0], colors[:, 1],
            c=scores, cmap='viridis',
            alpha=0.6, s=10
        )
        
        ax.set_xlabel('Red Channel', fontsize=12)
        ax.set_ylabel('Green Channel', fontsize=12)
        ax.set_title('Intelligence Map: Red vs Green Channels', fontsize=14)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Intelligence Score', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/intelligence_map_2d.png", dpi=150)
        plt.close()
    
    def _plot_color_distribution(self, colors: np.ndarray, scores: np.ndarray):
        """Plot distribution of each channel."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        channel_names = ['Red', 'Green', 'Blue', 'Alpha']
        
        for i, (ax, name) in enumerate(zip(axes.flatten(), channel_names)):
            ax.hist(colors[:, i], bins=50, alpha=0.7, color='steelblue')
            ax.set_xlabel(f'{name} Channel Value', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title(f'{name} Channel Distribution', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/channel_distributions.png", dpi=150)
        plt.close()
    
    def _visualize_best_color(self, evaluations: List[EvaluationResult]):
        """Visualize the best color as an image."""
        best_eval = max(evaluations, key=lambda e: e.intelligence_score)
        best_color = best_eval.color
        
        # Create a 100x100 image of the solid color
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        img[:, :] = best_color
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(
            f'Most Intelligent Image\n'
            f'RGBA: {best_color}\n'
            f'Score: {best_eval.intelligence_score:.4f}',
            fontsize=12
        )
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/best_color.png", dpi=150)
        plt.close()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

class RGBASearchPipeline:
    """Main pipeline orchestrating the RGBA intelligence search."""
    
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = config['project']['output_dir']
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.evaluator = IntelligenceEvaluator(config)
        self.mcts = HMCTS(config, self.evaluator)
        self.refiner = LocalRefinement(self.evaluator, config['refinement']['budget'])
        self.analyzer = StatisticalAnalyzer(config)
        self.visualizer = Visualizer(config, self.output_dir)
        
        # Set random seeds
        np.random.seed(config['project']['seed'])
        random.seed(config['project']['seed'])
        
        logger.info(f"Pipeline initialized. Output directory: {self.output_dir}")

    def run(self):
        """Execute the full search pipeline."""
        logger.info("Starting RGBA Intelligence Search Pipeline")
        start_time = time.time()
        
        results = {}
        
        # Phase 1: Systematic Grid Coverage (Optional but recommended for initialization)
        if self.config['grid'].get('step_size', 0) > 0:
            logger.info("Phase 1: Systematic Grid Coverage")
            grid_results = self._run_grid_search()
            results['grid'] = grid_results
        
        # Phase 2: Hierarchical MCTS Search
        logger.info("Phase 2: Hierarchical MCTS Search")
        mcts_results = self.mcts.search()
        results['mcts'] = {
            'best_color': mcts_results.best_color,
            'best_score': mcts_results.best_score,
            'total_evaluations': mcts_results.total_evaluations,
            'time_seconds': mcts_results.total_time_seconds,
            'tree_size': mcts_results.tree_size
        }
        logger.info(f"MCTS Best Color: {mcts_results.best_color}, Score: {mcts_results.best_score:.4f}")
        
        # Phase 3: Local Continuous Refinement
        if self.config['refinement']['enabled']:
            logger.info("Phase 3: Local Continuous Refinement")
            refined_color = self.refiner.refine(mcts_results.best_color)
            refined_result = self.evaluator(refined_color)
            results['refinement'] = {
                'refined_color': refined_color,
                'refined_score': refined_result.intelligence_score,
                'improvement': refined_result.intelligence_score - mcts_results.best_score
            }
            logger.info(f"Refined Color: {refined_color}, Score: {refined_result.intelligence_score:.4f}")
            final_color = refined_color
            final_score = refined_result.intelligence_score
        else:
            final_color = mcts_results.best_color
            final_score = mcts_results.best_score
        
        # Combine all evaluations for statistics
        all_evaluations = self.mcts.all_evaluations
        
        # Phase 4: Statistical Analysis
        if self.config['statistics']['enabled']:
            logger.info("Phase 4: Statistical Analysis")
            stats_results = self.analyzer.analyze(all_evaluations)
            results['statistics'] = stats_results
        
        # Phase 5: Visualization
        if self.config['visualization']['enabled']:
            logger.info("Phase 5: Visualization")
            self.visualizer.visualize(all_evaluations, self.mcts.tree)
        
        # Save final results
        results['final_best_color'] = final_color
        results['final_best_score'] = final_score
        results['total_pipeline_time_seconds'] = time.time() - start_time
        
        self._save_results(results, all_evaluations)
        
        logger.info("="*60)
        logger.info("SEARCH COMPLETE")
        logger.info(f"Most Intelligent Image (RGBA): {final_color}")
        logger.info(f"Final Intelligence Score: {final_score:.4f}")
        logger.info(f"Total Time: {results['total_pipeline_time_seconds']:.2f} seconds")
        logger.info(f"Total Evaluations: {len(all_evaluations)}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*60)
        
        return results

    def _run_grid_search(self) -> Dict[str, Any]:
        """Run systematic grid search for initial coverage."""
        step = self.config['grid']['step_size']
        colors = []
        for r in range(0, 256, step):
            for g in range(0, 256, step):
                for b in range(0, 256, step):
                    for a in range(0, 256, step):
                        colors.append((r, g, b, a))
        
        logger.info(f"Evaluating {len(colors)} grid colors...")
        grid_results = []
        for color in tqdm(colors, desc="Grid Search"):
            result = self.evaluator(color)
            grid_results.append(result)
            self.mcts.all_evaluations.append(result)
            self.mcts.total_evaluations += 1
        
        best_grid = max(grid_results, key=lambda x: x.intelligence_score)
        return {
            'total_colors': len(colors),
            'best_color': best_grid.color,
            'best_score': best_grid.intelligence_score
        }

    def _save_results(self, results: Dict[str, Any], evaluations: List[EvaluationResult]):
        """Save results to disk."""
        # Save summary JSON
        summary_path = Path(self.output_dir) / "search_summary.json"
        
        # Convert non-serializable types (like numpy types) to native Python types
        def convert_to_native(obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return str(obj)
        
        serializable_results = json.loads(json.dumps(results, default=convert_to_native))
        
        with open(summary_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        # Save all evaluations (sampled if too large to save disk space)
        eval_path = Path(self.output_dir) / "all_evaluations.json"
        max_evals_to_save = 50000
        evals_to_save = evaluations[:max_evals_to_save]
        
        serializable_evals = [
            {
                'color': e.color,
                'intelligence_score': float(e.intelligence_score),
                'expressivity': float(e.expressivity),
                'representational_capacity': float(e.representational_capacity),
                'stability': float(e.stability),
                'metrics': {k: float(v) for k, v in e.metrics.items()}
            }
            for e in evals_to_save
        ]
        
        with open(eval_path, 'w') as f:
            json.dump(serializable_evals, f, indent=2)
            
        logger.info(f"Saved summary to {summary_path}")
        logger.info(f"Saved {len(serializable_evals)} evaluations to {eval_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="RGBA Intelligence Search: Discover the most intelligent solid-color image"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of MCTS iterations")
    parser.add_argument("--grid-step", type=int, default=64, help="Grid search step size (0 to disable)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--refine", action="store_true", help="Enable local continuous refinement")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load base config
    config = DEFAULT_CONFIG.copy()
    
    # Override with user config if provided
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            user_config = yaml.safe_load(f)
            # Deep update config (simplified for this script)
            for k, v in user_config.items():
                if isinstance(v, dict) and k in config:
                    config[k].update(v)
                else:
                    config[k] = v
    
    # Override with CLI args
    config['project']['output_dir'] = args.output_dir
    config['project']['seed'] = args.seed
    config['mcts']['total_iterations'] = args.iterations
    config['grid']['step_size'] = args.grid_step
    config['refinement']['enabled'] = args.refine
    
    # Ensure output directory exists
    Path(config['project']['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Save the effective config for reproducibility
    config_path = Path(config['project']['output_dir']) / "effective_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Loaded configuration. Seed: {config['project']['seed']}")
    
    # Run pipeline
    try:
        pipeline = RGBASearchPipeline(config)
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Search interrupted by user.")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()