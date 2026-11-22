#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
#  ga_image_ffn.py — GA + WFC + CA trainer for IMAGE feed-forward networks
# ═══════════════════════════════════════════════════════════════════════════
#  THE IMAGE IS THE NETWORK:
#    pixel grid (H×W) = neurons × depth · columns = layers · rows = neurons
#    Hue → 1 of 24 activation functions (15° bands; refined ×3 ⇒ 72 classes)
#    Saturation → weight/gain · Value → bias (BiasEngine) ·
#    forward = strict left→right column sweep (acyclic by construction).
#    Two orientations: 'node' (MLP-like) and 'edge' (KAN-like, Liu 2024).
#
#  NO BACKPROP, STRUCTURALLY AND BY POLICY:
#    the network is a sealed oracle L(g) = loss(sweep(g, X), y);
#    GA/WFC/CA never see data; NES & SPSA use loss VALUES only (zeroth-order,
#    Nesterov & Spokoiny 2017); an auditor counts sample presentations and
#    blocks any autograd attempt.
#
#  ANCHORS: Such 2017 · Karth&Smith 2017 · Efros&Leung 1999 · Chua&Yang 1988 ·
#    Kitano 1990 · Wierstra 2014 · Salimans 2017 · Ros&Hansen 2015 ·
#    Hansen&Ostermeier 2001 · Spall 1992 · Jamieson&Talwalkar 2016 ·
#    Mouret&Clune 2015 · Lehman&Stanley 2011 · Cantú-Paz 1998 ·
#    Auger&Hansen 2005 · Eshelman 1991 · Deb 1995/1996/2002 ·
#    Goldberg&Richardson 1987 · Goldberg&Deb 1991 · Fialho 2010 ·
#    Bosman&Thierens 2000 · Rechenberg 1973 · Gauci&Stanley 2010 ·
#    Kolmogorov/Arnold · Liu 2024 (KAN) · Jaeger 2001 · Huang 2006 ·
#    Naeini 2015 (ECE) · Mitchell 2019 (model cards) · Olah 2020 (circuits)
#
#  Usage:
#    python ga_image_ffn.py --test     # full test suite
#    python ga_image_ffn.py --demo     # train on synthetic data, print audit
#    python ga_image_ffn.py --gui      # PySide6 trainer UI
#
#  Library quickstart (no CLI — import the API):
#    from ga_image_ffn import Model
#    m  = Model.fit("data.csv", seconds=180, profile="fast")
#    m.save("runs/v1")                    # model.png + .npz + .json
#    m2 = Model.load("runs/v1")           # fresh process
#    m2.predict(X); m2.predict_proba(X); m2.score(X_test, y_test)
#    m2.recheck(X_lab, y_lab)             # is it STILL intelligent?
#    m2.type_gallery("runs/v1/families"); m2.export_lineage("runs/v1/lineage")
# ═══════════════════════════════════════════════════════════════════════════
__version__ = "6.0.0"

import sys, os, time, math, csv as _csv, json as _json, hashlib, traceback
import unittest
from collections import deque, defaultdict, Counter
import numpy as np

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# 1. UTILITIES
# ═══════════════════════════════════════════════════════════════════════════
def mann_kendall_trend(data):
    n = len(data)
    if n < 4: return 0.0, 1.0
    s = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(data[j] - data[i])
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    z = (s - 1) / math.sqrt(var_s) if s > 0 else \
        ((s + 1) / math.sqrt(var_s) if s < 0 else 0.0)
    from math import erf
    return z, 2 * (1 - 0.5 * (1 + erf(abs(z) / math.sqrt(2))))

def _gumbel(rng, shape):
    u = np.clip(rng.random(shape), 1e-12, 1 - 1e-12)
    return -np.log(-np.log(u))

def _hsv_to_rgb_vec(hsv):
    h = hsv[..., 0] % 1.0
    s = np.clip(hsv[..., 1], 0, 1); v = np.clip(hsv[..., 2], 0, 1)
    i = np.floor(h * 6).astype(int); f = h * 6 - i
    p = v * (1 - s); q = v * (1 - f * s); t = v * (1 - (1 - f) * s)
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v]); g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return (np.clip(np.stack([r, g, b], -1), 0, 1) * 255).astype(np.uint8)

def save_model_png(genome, path):
    rgb = np.ascontiguousarray(_hsv_to_rgb_vec(np.asarray(genome)))
    if not PIL_AVAILABLE:
        np.save(path if path.endswith('.npy') else path + '.npy', genome); return path
    PILImage.fromarray(rgb).save(path); return path

def load_model_png(path):
    if path.endswith('.npy') or not PIL_AVAILABLE:
        return np.load(path if path.endswith('.npy') else path + '.npy')
    hsv = np.array(PILImage.open(path).convert('HSV'))
    return np.stack([hsv[..., 0] / 255., hsv[..., 1] / 255., hsv[..., 2] / 255.], -1)

def make_heatmap(arr):
    a = np.asarray(arr, float)
    fin = np.isfinite(a)
    if fin.any(): a = np.where(fin, a, np.nanmin(a[fin]))
    else: a = np.zeros_like(a)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a)); rng = max(hi - lo, 1e-12)
    n = np.clip((a - lo) / rng, 0, 1)
    r = (255 * n).astype(np.uint8); b = (255 * (1 - n)).astype(np.uint8)
    g = (255 * (1 - np.abs(2 * n - 1))).astype(np.uint8)
    return np.ascontiguousarray(np.stack([r, g, b], -1))

def split(X, y, test_frac=0.25, seed=0):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X))
    n = int(len(X) * (1 - test_frac))
    return X[idx[:n]], y[idx[:n]], X[idx[n:]], y[idx[n:]]

def split3(X, y, test_frac=0.2, val_frac=0.2, seed=0):
    """train / validation / test. Search sees train; selection on val;
    test touched once (Hastie et al., ESL ch.7 discipline)."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X))
    n_te = int(len(X) * test_frac); n_va = int(len(X) * val_frac)
    te, va, tr = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]

def acc01(pred_cols, y01):
    return float((pred_cols.argmax(1) == y01.argmax(1)).mean())

def r2(pred, y):
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    ss = ((y - y.mean(0)) ** 2).sum()
    return float(1.0 - ((pred - y) ** 2).sum() / max(ss, 1e-12))

def count_params(genome):
    g = np.asarray(genome)
    return int((g[..., 1] > 1e-3).sum() + g.shape[0] * g.shape[1])

def sign_test(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.abs(d) > 1e-12]; n = len(d)
    if n == 0: return 1.0
    k = int((d > 0).sum())
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))

def resolve_seed(s):
    if isinstance(s, int): return s
    try: return int(s)
    except (TypeError, ValueError):
        return int(hashlib.sha1(str(s).encode()).hexdigest()[:8], 16)

# ═══════════════════════════════════════════════════════════════════════════
# 2. DECODER SEMANTICS — 24 activations + BiasEngine
# ═══════════════════════════════════════════════════════════════════════════
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
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
def bent_identity(x): return (np.sqrt(x ** 2 + 1.0) - 1.0) / 2.0 + x
def square(x): return np.clip(x, -10, 10) ** 2
def cube(x): return np.clip(x, -10, 10) ** 3
def abs_act(x): return np.abs(x)
def sign_a(x): return np.sign(x)
def step(x): return np.where(x > 0, 1.0, 0.0)
def gaussian_a(x): return np.exp(-np.clip(x, -10, 10) ** 2)
def sinc_a(x): return np.sinc(np.clip(x, -10 * np.pi, 10 * np.pi) / np.pi)
def cosine_a(x): return np.cos(np.clip(x, -10 * np.pi, 10 * np.pi))
def sine_a(x): return np.sin(np.clip(x, -10 * np.pi, 10 * np.pi))

ACTIVATION_MAP = [
    (15, relu, "ReLU"), (30, sigmoid, "Sigmoid"), (45, tanh_act, "Tanh"),
    (60, leaky_relu, "LeakyReLU"), (75, swish, "Swish"), (90, linear, "Linear"),
    (105, gelu, "GELU"), (120, elu, "ELU"), (135, softplus, "Softplus"),
    (150, mish, "Mish"), (165, hard_sigmoid, "HardSigmoid"), (180, softsign, "Softsign"),
    (195, prelu, "PReLU"), (210, hard_swish, "HardSwish"), (225, square, "Square"),
    (240, cube, "Cube"), (255, abs_act, "Abs"), (270, sign_a, "Sign"),
    (285, step, "Step"), (300, gaussian_a, "Gaussian"), (315, bent_identity, "BentIdentity"),
    (330, sinc_a, "Sinc"), (345, cosine_a, "Cosine"), (360, sine_a, "Sine"),
]
ACTS = [f for _, f, _ in ACTIVATION_MAP]
ACTIVATION_NAMES = [n for _, _, n in ACTIVATION_MAP]
N_ACTIVATIONS = 24

# KAN grid-extension: refined decoder re-reads the SAME pixels at finer
# function-argument resolution (Liu et al. 2024, §2.4 — "refining grids").
KAN_FREQS = (0.5, 1.0, 3.0)

def hue_classes(refine=1): return N_ACTIVATIONS * int(refine)

def hue_band_refined(h, refine=1):
    n = hue_classes(refine)
    return np.minimum(n - 1, (np.clip(np.asarray(h, float), 0, 0.999999) * n)).astype(np.int64)

def hue_band(h): return hue_band_refined(h, 1)

def decode_refined(h, refine=1):
    cls = hue_band_refined(h, refine)
    return cls // int(refine), cls % int(refine)

def hue_to_activation(hue_deg):
    i = min(int(hue_deg // 15), N_ACTIVATIONS - 1)
    return ACTS[i], ACTIVATION_NAMES[i], i

_ACT_AWARE_SCALE = np.ones(N_ACTIVATIONS)
_ACT_AWARE_SCALE[[1, 2, 10, 11]] = 0.2      # Sigmoid, Tanh, HardSigmoid, Softsign
_ACT_AWARE_SCALE[[17, 18]] = 2.0            # Sign, Step

class BiasConfig:
    def __init__(self, mapping='centered', spatial=False, depth_scale=True, act_aware=True):
        self.mapping, self.spatial = mapping, spatial
        self.depth_scale, self.act_aware = depth_scale, act_aware

class BiasEngine:
    """Value → bias. mapping ∈ {raw, centered, sigmoidal, gaussian};
    spatial = 3×3 high-pass; depth_scale = 1/√(l+1) (Glorot/He spirit);
    act_aware rescales squashing/threshold functions."""
    @staticmethod
    def compute(V, layer_idx, cfg):
        if cfg.mapping == 'centered':   b = 2.0 * V - 1.0
        elif cfg.mapping == 'sigmoidal':
            b = 2.0 / (1.0 + np.exp(-12 * (V - 0.5))) - 1.0
        elif cfg.mapping == 'gaussian': b = np.exp(-((V - 0.5) ** 2) / 0.05) - 0.5
        else:                           b = V.copy()
        if cfg.spatial:
            P = np.pad(V, 1, mode='edge')
            lm = (P[:-2, :-2] + P[:-2, 1:-1] + P[:-2, 2:] + P[1:-1, :-2] +
                  P[1:-1, 1:-1] + P[1:-1, 2:] + P[2:, :-2] + P[2:, 1:-1] + P[2:, 2:]) / 9.0
            b = V - lm
        if cfg.depth_scale: b = b / np.sqrt(layer_idx + 1)
        return b

# ═══════════════════════════════════════════════════════════════════════════
# 3. THE IMAGE *IS* THE FEED-FORWARD NETWORK
# ═══════════════════════════════════════════════════════════════════════════
class ImageFFN:
    """Genome image (H×W×3 HSV) → strictly acyclic FFN.
    mode='node': z = s_j · (K ∗ [s_{j-1}·a]) + b_j; a = ψ_j(ω·z)   (MLP-like)
    mode='edge': e = ψ_j(ω·a); t = s_j·e; a = s_{j+1}·(K∗t)+b_{j+1} (KAN-like:
                 φ_edge = s_i·K(i−i')·s_i'·ψ(x) ≡ w·spline(x), nodes sum —
                 Liu et al. 2024 Fig. 2.1). Last column's hue chooses the
                 readout nonlinearity (Linear band = KAN's linear Φ_out).
    refine: 1 = legacy 24 bands; 3 = 72 classes (24 fns × 3 argument scales)."""
    def __init__(self, radius=2, coupling='local', kernel='gauss', prune=0.0,
                 bias_cfg=None, mode='node', refine=1):
        self.r, self.coupling, self.prune = int(radius), coupling, float(prune)
        self.bias_cfg = bias_cfg or BiasConfig()
        self.mode, self.refine = mode, int(refine)
        o = np.arange(-self.r, self.r + 1)
        if kernel == 'laplacian':
            g1 = np.exp(-0.5 * (o / max(self.r / 1.5, .5)) ** 2)
            g2 = np.exp(-0.5 * (o / max(self.r, .5)) ** 2)
            self.K = g1 - 0.6 * g2; self.K /= self.K.sum()
        else:
            self.K = np.exp(-0.5 * (o / max(self.r / 1.5, .5)) ** 2); self.K /= self.K.sum()
        self.offsets = list(range(-self.r, self.r + 1))
        self._built = False; self._warned = False

    def build(self, H, S, V):
        H, S, V = (np.asarray(x, float) for x in (H, S, V))
        if H.max() > 1.001: H = H / 360.0
        self.Hn, self.nL = H.shape
        self.w, self.b, self.groups, self.om = [], [], [], []
        for j in range(self.nL):
            wj = S[:, j].astype(np.float64).copy()
            wj[wj < self.prune] = 0.0
            base = BiasEngine.compute(V, j, self.bias_cfg)[:, j]
            fn, fr = decode_refined(H[:, j], self.refine)
            eff = base * (_ACT_AWARE_SCALE[fn] if self.bias_cfg.act_aware else 1.0)
            self.w.append(wj); self.b.append(eff)
            self.groups.append([np.flatnonzero(fn == k) for k in range(N_ACTIVATIONS)])
            self.om.append(KAN_FREQS[fr])
        self.out_act = int(np.bincount(
            hue_band_refined(H[:, -1], self.refine) // self.refine,
            minlength=N_ACTIVATIONS).argmax())
        self._built = True
        return self

    def from_genome(self, g):
        g = np.asarray(g, float)
        return self.build(g[..., 0], g[..., 1], g[..., 2])

    def forward(self, X, return_states=False):
        assert self._built, "call build()/from_genome() first"
        X = np.asarray(X, float)
        if X.ndim == 1: X = X[None, :]
        if X.shape[1] > self.Hn and not self._warned:
            print(f"[ImageFFN] warning: {X.shape[1]} features > {self.Hn} rows — "
                  f"excess features dropped (use auto_config/PCA front-end)")
            self._warned = True
        a = np.zeros((X.shape[0], self.Hn)); d = min(X.shape[1], self.Hn)
        a[:, :d] = X[:, :d]
        states = [a.copy()] if return_states else None
        if self.mode == 'node':
            for j in range(self.nL):
                src = np.ones(self.Hn) if j == 0 else self.w[j - 1]
                t = a * src[None, :]
                acc = np.zeros_like(t)
                for off, kd in zip(self.offsets, self.K):
                    acc += kd * np.roll(t, off, axis=1)
                z = self.w[j][None, :] * acc + self.b[j][None, :]
                out = np.empty_like(z); om = self.om[j]
                for k, idx in enumerate(self.groups[j]):
                    if idx.size: out[:, idx] = ACTS[k](z[:, idx] * om[None, idx])
                a = out
                if return_states: states.append(a.copy())
        else:  # edge (KAN orientation)
            for j in range(self.nL - 1):
                om = self.om[j]
                e = np.empty_like(a)
                for k, idx in enumerate(self.groups[j]):
                    if idx.size: e[:, idx] = ACTS[k](a[:, idx] * om[None, idx])
                t = e * self.w[j][None, :]
                acc = np.zeros_like(t)
                for off, kd in zip(self.offsets, self.K):
                    acc += kd * np.roll(t, off, axis=1)
                a = self.w[j + 1][None, :] * acc + self.b[j + 1][None, :]
                if return_states: states.append(a.copy())
            a = ACTS[self.out_act](a)
        return (a, states) if return_states else a

    def ridge_head_fit(self, X, Y, lam=1e-2):
        """ESN/ELM readout: ONE closed-form solve. No iterations, no chain
        rule (Jaeger 2001; Huang et al. 2006). Still no backprop."""
        Sf = self.forward(X); d = Sf.shape[1]
        self.Wro = np.linalg.solve(Sf.T @ Sf + lam * np.eye(d), Sf.T @ Y)
        return self
    def ridge_head_predict(self, X): return self.forward(X) @ self.Wro

    def summary(self):
        used = sorted({k for j in range(self.nL) for k in range(N_ACTIVATIONS)
                       if self.groups[j][k].size})
        print(f"ImageFFN | {self.nL} layers × {self.Hn} neurons | mode={self.mode} "
              f"refine={self.refine} coupling={self.coupling} "
              f"| bias={self.bias_cfg.mapping}")
        print("activations: " + ", ".join(ACTIVATION_NAMES[k] for k in used))

def _slice_C(pred, C):
    if pred.shape[1] >= C: return pred[:, -C:]
    reps = int(np.ceil(C / max(pred.shape[1], 1)))
    return np.tile(pred, (1, reps))[:, :C]

def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))

def make_task_loss(task, per_sample=False):
    if task == 'classification':
        if per_sample:
            return lambda p, y: -((y * log_softmax(_slice_C(p, y.shape[1]))).sum(1))
        return lambda p, y: float(-((y * log_softmax(_slice_C(p, y.shape[1]))).sum(1)).mean())
    if per_sample:
        return lambda p, y: np.mean((_slice_C(p, y.shape[1]) - y) ** 2, axis=1)
    return lambda p, y: float(np.mean((_slice_C(p, y.shape[1]) - y) ** 2))

# ═══════════════════════════════════════════════════════════════════════════
# 4. KAN INTERPRETABILITY — symbolification + circuit metrics (§2.5/§2.6)
# ═══════════════════════════════════════════════════════════════════════════
def snap_to_bands(g, refine=1):
    n = hue_classes(refine)
    g = np.asarray(g).copy()
    b = np.clip((g[..., 0] * n).astype(int), 0, n - 1)
    g[..., 0] = (b + 0.5) / n
    return np.clip(g, 0, 1)

def symbolify_genome(g, refine=1):
    """KAN §2.6 analog: collapse each column to its dominant (function, freq)
    + mean gain/bias. The symbol circuit — same decoder, by read-off."""
    g = np.asarray(g); H, W, _ = g.shape
    fn, fr = decode_refined(g[..., 0], refine)
    out = np.zeros_like(g); n = hue_classes(refine)
    for j in range(W):
        cnt = np.bincount(fn[:, j], minlength=N_ACTIVATIONS)
        k = int(cnt.argmax()); m = fn[:, j] == k
        f = int(np.bincount(fr[:, j][m], minlength=refine).argmax()) if m.any() else 0
        out[:, j, 0] = ((k * refine + f) + 0.5) / n
        out[:, j, 1] = g[..., 1][:, j].mean()
        out[:, j, 2] = g[..., 2][:, j].mean()
    return out

def symbolify_text(g, refine=1, top=2):
    g = np.asarray(g); fn, _ = decode_refined(g[..., 0], refine)
    lines = []
    for j in range(g.shape[1]):
        cnt = np.bincount(fn[:, j], minlength=N_ACTIVATIONS); tot = cnt.sum()
        order = np.argsort(cnt)[::-1][:top]
        parts = [f"{ACTIVATION_NAMES[k]}×{cnt[k]} ({100*cnt[k]/max(tot,1):.0f}%)"
                 for k in order if cnt[k] > 0]
        live = (g[..., 1][:, j] > 1e-3).mean()
        lines.append(f"col{j:>2d} | {' '.join(parts):<40s} | edge-L1 "
                     f"{g[...,1][:,j].mean():.2f} | live {live*100:.0f}%")
    return "\n".join(lines)

def symbol_fidelity(g, Xva, yva, refine=1, mode='node', radius=2):
    """Agreement between the full image-network and its symbolified circuit."""
    full = ImageFFN(mode=mode, refine=refine, radius=radius).from_genome(g)
    sym = ImageFFN(mode=mode, refine=refine, radius=radius).from_genome(
        symbolify_genome(g, refine))
    pf, ps = full.forward(Xva), sym.forward(Xva)
    if yva.ndim == 2 and yva.shape[1] > 1:
        return float((pf.argmax(1) == ps.argmax(1)).mean())
    a = pf[:, 0] - pf[:, 0].mean(); b = ps[:, 0] - ps[:, 0].mean()
    return float(a @ b / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-12))

def kan_metrics(g, refine=1):
    g = np.asarray(g); fn, _ = decode_refined(g[..., 0], refine)
    live = g[..., 1] > 1e-3
    pur = [np.bincount(fn[:, j], minlength=N_ACTIVATIONS).max() /
           max(fn[:, j].size, 1) for j in range(g.shape[1])]
    hb = np.bincount(fn.ravel(), minlength=N_ACTIVATIONS).astype(float)
    hb /= max(hb.sum(), 1)
    return {'sparsity': float(1 - live.mean()),
            'edge_L1': float(g[..., 1].mean()),
            'col_purity': float(np.mean(pur)),
            'top1_share': float(hb.max()),
            'n_functions_used': int((hb > 1e-3).sum())}

# ═══════════════════════════════════════════════════════════════════════════
# 5. STRUCTURED GENOME GENERATION (semantics-aligned images)
# ═══════════════════════════════════════════════════════════════════════════
def _bilerp(a, H, W):
    hi = np.linspace(0, a.shape[0] - 1, H); wi = np.linspace(0, a.shape[1] - 1, W)
    h0 = np.floor(hi).astype(int); w0 = np.floor(wi).astype(int)
    h1 = np.minimum(h0 + 1, a.shape[0] - 1); w1 = np.minimum(w0 + 1, a.shape[1] - 1)
    fh = (hi - h0)[:, None, None]; fw = (wi - w0)[None, :, None]
    return (a[np.ix_(h0, w0)] * (1 - fh) * (1 - fw) + a[np.ix_(h0, w1)] * (1 - fh) * fw
            + a[np.ix_(h1, w0)] * fh * (1 - fw) + a[np.ix_(h1, w1)] * fh * fw)

def value_noise(rng, H, W, octaves=(1, 2, 4, 8), amps=(.45, .27, .17, .11)):
    acc = np.zeros((H, W))
    for o, aw in zip(octaves, amps):
        g = rng.random((max(2, o), max(2, int(round(o * W / max(H, 1))))))
        acc += aw * _bilerp(g, H, W)
    return np.clip(acc / sum(amps), 0, 1)

def generate_genome(rng, H, W, style='slabs'):
    """slabs = depth-slabs of one activation family; plaids = smooth mixed
    zones; blobs = dense microcircuit islands. Bands align to 15° activation
    boundaries so every region decodes to WHOLE functions."""
    V = value_noise(rng, H, W)
    if style == 'slabs':
        nb = max(2, W // 3)
        edges = np.linspace(0, W, nb + 1).astype(int)
        bands = rng.integers(0, N_ACTIVATIONS, nb)
        colb = np.concatenate([np.full(edges[i + 1] - edges[i], bands[i])
                               for i in range(nb)])
        Hc = np.broadcast_to(((colb + 0.5) / N_ACTIVATIONS)[None, :], (H, W))
        Sc = np.clip(0.30 + 0.65 * value_noise(rng, H, W), 0, 1)
    elif style == 'plaids':
        n1 = value_noise(rng, H, W); n2 = value_noise(rng, H, W)
        Hc = (hue_band(0.5 * (n1 + np.sin(2 * np.pi * n2) ** 2)) + 0.5) / N_ACTIVATIONS % 1.0
        Sc = np.clip(0.25 + 0.7 * (0.5 * n1 + 0.5 * n2), 0, 1)
    else:
        n = value_noise(rng, H, W, octaves=(2, 4), amps=(.6, .4))
        Sc = np.clip(np.where(n > 0.55, 0.55 + 0.45 * n, 0.10 * n), 0, 1)
        Hc = (hue_band(value_noise(rng, H, W)) + 0.5) / N_ACTIVATIONS % 1.0
    return np.clip(np.stack([np.broadcast_to(Hc, (H, W)), Sc, V], -1), 0, 1)

# ═══════════════════════════════════════════════════════════════════════════
# 6. WFC — learned tile grammar + fast collapse
# ═══════════════════════════════════════════════════════════════════════════
class WFCTileVocab:
    """Tile = (hue band, sat band, val band); adjacency learned from elites
    (exponentially decayed — recent elites dominate; Efros&Leung 1999 MRF)."""
    def __init__(self, hb=N_ACTIVATIONS, sb=4, vb=3, smoothing=0.1):
        self.hb, self.sb, self.vb = int(hb), int(sb), int(vb)
        self.T = self.hb * self.sb * self.vb
        self.dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.adj = np.full((4, self.T, self.T), float(smoothing))
        self.learned = False
        self._cache = {}
        self.refresh()

    def tile_of(self, g):
        g = np.asarray(g)
        b = np.ascontiguousarray(g).tobytes()
        t = self._cache.get(b)
        if t is not None: return t
        h = (np.clip(g[..., 0], 0, 0.999) * self.hb).astype(np.int32)
        s = (np.clip(g[..., 1], 0, 0.999) * self.sb).astype(np.int32)
        v = (np.clip(g[..., 2], 0, 0.999) * self.vb).astype(np.int32)
        t = (h * self.sb + s) * self.vb + v
        if len(self._cache) > 256: self._cache.clear()
        self._cache[b] = t
        return t

    def decode(self, tiles):
        tiles = np.asarray(tiles)
        h = tiles // (self.sb * self.vb); s = (tiles // self.vb) % self.sb
        v = tiles % self.vb
        return np.clip(np.stack([(h + .5) / self.hb, (s + .5) / self.sb,
                                 (v + .5) / self.vb], -1), 0, 1)

    def learn(self, genomes, decay=0.85):
        if len(genomes) == 0: return False
        self.adj *= decay; T = self.T
        for g in genomes:
            t = self.tile_of(g); H, W = t.shape
            if H < 2 or W < 2: continue
            for d, (a, b) in enumerate((
                    (t[:, :-1].ravel(), t[:, 1:].ravel()),
                    (t[:, 1:].ravel(), t[:, :-1].ravel()),
                    (t[:-1, :].ravel(), t[1:, :].ravel()),
                    (t[1:, :].ravel(), t[:-1, :].ravel()))):
                self.adj[d] += np.bincount(a * T + b, minlength=T * T).reshape(T, T)
        self.refresh(); self.learned = True
        return True

    def refresh(self):
        f = self.adj.sum(axis=(0, 2))
        self.freq = f / max(f.sum(), 1e-12)
        self.adjN = [a / np.maximum(a.sum(1, keepdims=True), 1e-12) for a in self.adj]

class FastWFC:
    """Phase-parallel (checkerboard) soft collapse: Gumbel-max sampling,
    whole-grid propagation, budgeted rounds, anytime argmax finish
    (Karth & Smith 2017 arc-consistency lineage, softened)."""
    def __init__(self, vocab, temperature=0.85, ps=0.9, sweeps=1, max_rounds=6, rng=None):
        self.v = vocab; self.t_inv = 1.0 / max(float(temperature), 1e-3)
        self.ps = float(ps); self.sweeps = max(0, int(sweeps))
        self.max_rounds = int(max_rounds)
        self.rng = rng or np.random.default_rng()
        self.prior = None; self.wave = None; self.open = None; self.rounds = 0

    def prepare(self, H, W, seed_tiles=None):
        n = 1 if seed_tiles is None else np.asarray(seed_tiles).shape[0]
        wave = np.tile(self.v.freq, (n, H, W, 1)).astype(np.float64)
        if seed_tiles is not None:
            st = np.asarray(seed_tiles); wave *= 0.25
            nn, yy, xx = np.ogrid[:n, :H, :W]
            wave[nn, yy, xx, st] += 0.85
        if self.prior is not None:
            pr = np.asarray(self.prior, float)
            wave = wave * (pr[None, :, :, None] if pr.ndim == 2 else pr)
        wave /= np.maximum(wave.sum(-1, keepdims=True), 1e-12)
        self.wave = wave; self.open = np.ones((n, H, W), bool); self.rounds = 0
        return self

    def _phase_pass(self, phase):
        H, W = self.wave.shape[1:3]
        yy, xx = np.mgrid[0:H, 0:W]
        ph = ((yy % 2) == (phase >> 1)) & ((xx % 2) == (phase & 1))
        sel = self.open & ph
        if not sel.any(): return
        ns, ys, xs = np.nonzero(sel)
        w = self.wave[ns, ys, xs]
        z = np.log(np.maximum(w, 1e-12)) * self.t_inv + _gumbel(self.rng, w.shape)
        t = z.argmax(1)
        self.wave[ns, ys, xs] = 0.0
        self.wave[ns, ys, xs, t] = 1.0
        self.open[ns, ys, xs] = False

    def _propagate(self):
        wave = self.wave; n, H, W, T = wave.shape
        p = wave / np.maximum(wave.sum(-1, keepdims=True), 1e-12)
        flat = p.reshape(-1, T)
        for d, (dy, dx) in enumerate(self.v.dirs):
            sup = (flat @ self.v.adjN[d]).reshape(n, H, W, T)
            sup = np.maximum(sup, 1e-9) ** self.ps
            deliv = np.roll(sup, (dy, dx), axis=(1, 2))
            wave[self.open] *= deliv[self.open]
        s = wave.sum(-1, keepdims=True)
        bad = (s[..., 0] < 1e-8) & self.open
        if bad.any(): wave[bad] = self.v.freq
        wave /= np.maximum(wave.sum(-1, keepdims=True), 1e-12)

    def run_round(self):
        for phase in range(4):
            self._phase_pass(phase)
            for _ in range(self.sweeps):
                self._propagate()
        self.rounds += 1
        return (not self.open.any()) or self.rounds >= self.max_rounds

    def finish(self):
        if self.open.any():
            ns, ys, xs = np.nonzero(self.open)
            w = self.wave[ns, ys, xs]
            t = (np.log(np.maximum(w, 1e-12)) + _gumbel(self.rng, w.shape)).argmax(1)
            self.wave[ns, ys, xs] = 0.0; self.wave[ns, ys, xs, t] = 1.0
            self.open[ns, ys, xs] = False
        return self.genomes()

    def genomes(self, show_open=False):
        g = self.v.decode(self.wave.argmax(-1))
        if show_open and self.open is not None:
            g = g.copy(); g[..., 2] = np.where(self.open, 0.10, g[..., 2])
        return g

def build_geometry_prior(H, W, lam=3.0):
    """HyperNEAT-flavored locality prior: gain mass decays along depth."""
    col = 0.35 + 0.65 * np.exp(-np.arange(W) / max(lam, 1e-9))
    P = np.tile(col[None, :], (H, 1))
    return P / P.mean()

# ═══════════════════════════════════════════════════════════════════════════
# 7. CELLULAR AUTOMATA — developmental operators (Chua 1988; Kitano 1990)
# ═══════════════════════════════════════════════════════════════════════════
class CellularMutator:
    RULES = ('majority', 'life', 'waves', 'reaction', 'erosion')
    def __init__(self, rule='majority', steps=1, prob=0.18, rule_no=90, refine=1):
        self.rule, self.steps = rule, int(steps)
        self.prob, self.rule_no = float(prob), int(rule_no)
        self.refine = int(refine)

    @staticmethod
    def _box(a, k):
        H, W = a.shape; P = np.pad(a, k, mode='wrap'); out = np.zeros_like(a)
        for dy in range(2 * k + 1):
            for dx in range(2 * k + 1):
                out += P[dy:dy + H, dx:dx + W]
        return out

    def develop(self, g, rng=None, steps=None):
        rng = rng or np.random.default_rng()
        g = np.asarray(g, np.float64).copy()
        if self.prob <= 0: return np.clip(g, 0, 1)
        mask = rng.random(g.shape[:2]) < self.prob
        for _ in range(steps or self.steps):
            g = getattr(self, '_' + self.rule)(g, rng, mask)
        return np.clip(g, 0.0, 1.0)

    def _majority(self, g, rng, mask):
        out = g.copy(); s1 = g[..., 1]
        nb = (self._box(s1, 1) - s1) / 8.0
        out[..., 1] = np.where(mask, 0.65 * s1 + 0.35 * nb, s1)
        n = hue_classes(self.refine)
        hb = hue_band_refined(g[..., 0], self.refine)
        up, dn = np.roll(hb, 1, 0), np.roll(hb, -1, 0)
        out[..., 0] = np.where(mask & (up == dn), (up + 0.5) / n, g[..., 0])
        vm = (np.roll(g[..., 2], 1, 0) + np.roll(g[..., 2], -1, 0)) * 0.5
        out[..., 2] = np.where(mask, 0.8 * g[..., 2] + 0.2 * vm, g[..., 2])
        return out

    def _life(self, g, rng, mask):
        alive = (g[..., 1] > 0.5).astype(np.int32)
        nb = self._box(alive, 1) - alive
        nxt = ((alive == 1) & ((nb == 2) | (nb == 3))).astype(np.float64)
        out = g.copy()
        out[..., 1] = np.where(mask, nxt * 0.85 + 0.05, g[..., 1])
        return out

    def _waves(self, g, rng, mask):
        out = g.copy(); s = g[..., 1] > 0.5
        up = np.roll(s, 1, 0)
        bits = ((np.roll(up, 1, 1).astype(np.int32) << 2)
                | (up.astype(np.int32) << 1)
                | np.roll(up, -1, 1).astype(np.int32))
        row = ((self.rule_no >> bits) & 1).astype(bool)
        merged = np.where(mask, row, s)
        jit = 0.55 + 0.3 * rng.random(g.shape[:2])
        out[..., 1] = np.where(merged, np.maximum(g[..., 1], jit),
                               np.minimum(g[..., 1], 0.45))
        return out

    def _reaction(self, g, rng, mask):
        out = g.copy(); a = g[..., 1]
        act = self._box(a, 1) / 9.0; inh = self._box(a, 3) / 49.0
        out[..., 1] = np.where(mask, np.clip(a + 0.35 * act - 0.22 * inh, 0, 1), a)
        out[..., 2] = np.where(mask, np.clip(g[..., 2] + 0.15 * (inh - act), 0, 1), g[..., 2])
        return out

    def _erosion(self, g, rng, mask):
        out = g.copy()
        sm = self._box(g[..., 1], 1) / 9.0
        lo, hi = np.percentile(sm, 30), np.percentile(sm, 70)
        new = np.where(sm < lo, g[..., 1] * 0.4,
                       np.where(sm > hi, np.minimum(g[..., 1] * 1.3 + 0.1, 1), g[..., 1]))
        out[..., 1] = np.where(mask, new, g[..., 1])
        return out

# ═══════════════════════════════════════════════════════════════════════════
# 8. CLASSICAL GA OPERATORS (Deb 1995/1996/2002; Goldberg&Richardson 1987)
# ═══════════════════════════════════════════════════════════════════════════
def sbx(a, b, eta=15, rng=None, prob=0.9):
    """Simulated Binary Crossover (Deb & Agrawal 1995), bounded [0,1].
    Uncrossed genes copy a random parent (BGA discrete recombination,
    Mühlenbein & Schlierkamp-Voosen 1993) — never a silent midpoint."""
    rng = rng or np.random.default_rng()
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = rng.random(a.shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)),
                        (1 / (2 * np.maximum(1 - u, 1e-12))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * a + (1 - beta) * b)
    c2 = 0.5 * ((1 - beta) * a + (1 + beta) * b)
    val = np.where(rng.random(a.shape) < 0.5, c1, c2)
    do = (rng.random(a.shape) < prob) & (np.abs(a - b) > 1e-12)
    src = np.where(rng.random(a.shape) < 0.5, a, b)
    return np.clip(np.where(do, val, src), 0.0, 1.0)

def poly_mut(g, eta=20, rng=None, pm=None):
    """Polynomial mutation (Deb & Goyal 1996). pm=None → 1/(H·W):
    one pixel-gene per genome in expectation (Deb's 1/n rule)."""
    rng = rng or np.random.default_rng()
    g = np.asarray(g, float)
    if pm is None: pm = 1.0 / max(g.shape[0] * g.shape[1], 1)
    do = rng.random(g.shape) < pm
    x = np.clip(g, 0, 1)
    d1, d2 = x, 1.0 - x
    r = rng.random(g.shape); mp = 1.0 / (eta + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        s1 = (2 * r + (1 - 2 * r) * np.maximum(1 - d1, 0) ** mp) ** mp - 1.0
        s2 = 1.0 - (2 * (1 - r) + 2 * (r - 0.5) * np.maximum(1 - d2, 0) ** mp) ** mp
    dq = np.where(r < 0.5, s1, s2)
    return np.clip(np.where(do, x + dq, x), 0.0, 1.0)

def nsga2_sort(objectives):
    """Non-dominated sort + crowding distance (Deb et al. 2002)."""
    objectives = np.asarray(objectives, float)
    n = len(objectives)
    fronts = [[]]; rank = np.zeros(n, int)
    counts = np.zeros(n, int); dominated = [[] for _ in range(n)]
    for p in range(n):
        for q in range(n):
            if p == q: continue
            if np.all(objectives[p] <= objectives[q]) and np.any(objectives[p] < objectives[q]):
                dominated[p].append(q)
            elif np.all(objectives[q] <= objectives[p]) and np.any(objectives[q] < objectives[p]):
                counts[p] += 1
        if counts[p] == 0: rank[p] = 0; fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in dominated[p]:
                counts[q] -= 1
                if counts[q] == 0: rank[q] = i + 1; nxt.append(q)
        i += 1; fronts.append(nxt)
    fronts.pop()
    crowd = np.zeros(n)
    for front in fronts:
        if len(front) <= 2:
            crowd[front] = float('inf'); continue
        for obj in range(objectives.shape[1]):
            si = sorted(front, key=lambda x: objectives[x, obj])
            crowd[si[0]] = float('inf'); crowd[si[-1]] = float('inf')
            vals = objectives[si, obj]; rng_ = vals[-1] - vals[0]
            if rng_ > 1e-12:
                for k in range(1, len(si) - 1):
                    crowd[si[k]] += (vals[k + 1] - vals[k - 1]) / rng_
    return rank, crowd, fronts

def nsga2_select(F, N):
    rank, crowd, _ = nsga2_sort(np.asarray(F, float))
    order = np.lexsort((-crowd, rank))
    return order[:N], rank, crowd

def shared_scores(pop, fits, sigma=None, desc_fn=None):
    """Fitness sharing, minimization form (Goldberg & Richardson 1987):
    f_sh = f · m_i; σ auto = median pairwise distance (Shir & Bäck 2005).
    Niches are BEHAVIOR descriptors, not genotypes — pairs with the archive."""
    desc_fn = desc_fn or BehaviorArchive.describe
    D = np.asarray([desc_fn(g) for g in pop], float)
    pd = np.linalg.norm(D[:, None] - D[None], axis=-1)
    if sigma is None:
        iu = np.triu_indices(len(D), 1)
        sigma = float(np.median(pd[iu])) if len(iu[0]) else 1.0
    m = np.maximum(0.0, 1.0 - pd / max(sigma, 1e-12)).sum(1)
    return np.asarray(fits, float) * np.maximum(m, 1.0)

def speciate(pop, threshold='auto', desc_fn=None):
    """Threshold speciation (v3.4 Species model; 'auto' = 2× median)."""
    desc_fn = desc_fn or BehaviorArchive.describe
    D = [desc_fn(g) for g in pop]
    if threshold == 'auto':
        pd = [np.linalg.norm(D[i] - D[j]) for i in range(len(D))
              for j in range(i + 1, len(D))]
        threshold = 2.0 * float(np.median(pd)) if pd else 1.0
    species = []
    for i in range(len(D)):
        for s in species:
            if np.linalg.norm(D[s[0]] - D[i]) < threshold:
                s.append(i); break
        else:
            species.append([i])
    return species, float(threshold)

# ═══════════════════════════════════════════════════════════════════════════
# 9. ENGINE FACADE
# ═══════════════════════════════════════════════════════════════════════════
class WFCCAEngine:
    """WFC grammar + CA mutator + synthesis, with ms telemetry."""
    def __init__(self, cfg=None, seed=None):
        self.cfg = dict(cfg or {})
        self.vocab = WFCTileVocab(self.cfg.get('wfc_hue_bands', N_ACTIVATIONS),
                                  self.cfg.get('wfc_sat_bands', 4),
                                  self.cfg.get('wfc_val_bands', 3))
        self.ca = CellularMutator(self.cfg.get('ca_rule', 'majority'),
                                  self.cfg.get('ca_steps', 1),
                                  self.cfg.get('ca_prob', 0.18),
                                  self.cfg.get('ca_rule_no', 90))
        self.rng = np.random.default_rng(seed)
        self.ms_spent = 0.0; self.genomes_made = 0

    def set_refine(self, r):
        """Re-quantize hue tiles to the refined grid; CA banding follows.
        Adjacency resets (old tiles ≠ new tiles)."""
        r = max(1, int(r))
        self.cfg['wfc_hue_bands'] = hue_classes(r)
        self.vocab = WFCTileVocab(hue_classes(r), self.cfg.get('wfc_sat_bands', 4),
                                  self.cfg.get('wfc_val_bands', 3))
        self.ca.refine = r

    def learn(self, genomes, decay=0.85):
        t0 = time.perf_counter()
        ch = self.vocab.learn(genomes, decay)
        self.ms_spent += (time.perf_counter() - t0) * 1e3
        return ch

    def _synth(self, vocab, n, shape, seed_tiles=None, temp=None, prior=None):
        t0 = time.perf_counter()
        gen = FastWFC(vocab, temp or self.cfg.get('wfc_temperature', 0.85),
                      self.cfg.get('wfc_prop_strength', 0.9),
                      self.cfg.get('wfc_prop_sweeps', 1),
                      self.cfg.get('wfc_max_rounds', 6), self.rng)
        gen.prior = prior
        gen.prepare(shape[0], shape[1], seed_tiles)
        while not gen.run_round(): pass
        out = gen.finish()
        self.ms_spent += (time.perf_counter() - t0) * 1e3
        self.genomes_made += n
        return out

    def make_genomes(self, n, shape, seed_tiles=None, temp=None, prior=None):
        return self._synth(self.vocab, n, shape, seed_tiles, temp, prior)

    def make_genome(self, H, W, **kw): return self.make_genomes(1, (H, W), **kw)[0]

    def crossover(self, pa, pb, alpha=0.5, temp=None, prior=None):
        ta = self.vocab.tile_of(np.asarray(pa)[None])[0]
        tb = self.vocab.tile_of(np.asarray(pb)[None])[0]
        mix = np.where(self.rng.random(ta.shape) < alpha, ta, tb)
        return self.make_genomes(1, ta.shape, mix[None], temp=temp, prior=prior)[0]

    def mutate(self, g, rng=None):
        t0 = time.perf_counter()
        out = self.ca.develop(g, rng or self.rng)
        self.ms_spent += (time.perf_counter() - t0) * 1e3
        return out

    def save_grammar(self, path): np.savez_compressed(path, adj=self.vocab.adj)
    def load_grammar(self, path):
        self.vocab.adj = np.load(path)['adj']; self.vocab.refresh(); self.vocab.learned = True

# ═══════════════════════════════════════════════════════════════════════════
# 10. NO-BACKPROP INFRA — auditor, ledger, cache, multi-fidelity, surrogate
# ═══════════════════════════════════════════════════════════════════════════
class GradientFreedomAuditor:
    """Counts forward evaluations + sample presentations, blocks autograd if
    present, verifies purity. The report IS the proof."""
    def __init__(self):
        self.forward_calls = 0; self.sample_evals = 0
        self.backward_attempts = 0; self.violations = []; self._patched = False

    def install(self):
        if self._patched: return
        try:
            import torch
            def _blocked(*a, **k):
                self.backward_attempts += 1
                raise RuntimeError("Backpropagation disabled by policy "
                                   "(forward-only training contract).")
            torch.Tensor.backward = _blocked
            torch.autograd.backward = _blocked
            torch.autograd.grad = _blocked
            self._patched = True
        except ImportError:
            pass  # pure-numpy stack: structurally incapable of autograd

    def wrap_loss(self, loss_fn):
        aud = self
        def L(g, X, y):
            aud.forward_calls += 1
            if X is not None: aud.sample_evals += int(len(X))
            f = float(loss_fn(g, X, y))
            if not np.isfinite(f): aud.violations.append("non-finite loss returned")
            return f
        return L

    def purity_check(self, loss_fn, g, X, y):
        f1 = float(loss_fn(g, X, y)); f2 = float(loss_fn(g, X, y))
        ok = abs(f1 - f2) <= 1e-9
        if not ok: self.violations.append("evaluator impure or stateful")
        return ok

    def report(self, ledger=None):
        lines = ["══ NO-BACKPROP AUDIT ══",
                 f"forward evaluations : {self.forward_calls}"
                 + (f"   [{ledger.text()}]" if ledger else ""),
                 f"sample presentations: {self.sample_evals:,}",
                 f"backward attempts   : {self.backward_attempts}",
                 f"violations          : {len(self.violations)}"]
        lines += [f"  ! {v}" for v in self.violations[:5]]
        lines.append("verdict: " + ("PASS — zeroth-order only"
                     if not self.violations and self.backward_attempts == 0 else "FAIL"))
        return "\n".join(lines)

class BudgetLedger:
    def __init__(self):
        self.full = self.slices = self.aux = self.cache_hits = 0
        self.full_saved = self.screened = self.immigrants = 0
    @property
    def used(self): return self.full + self.slices + self.aux
    def text(self):
        saved = self.cache_hits + self.full_saved + self.screened
        eff = 100.0 * saved / max(saved + self.used, 1)
        return (f"full {self.full} · slice {self.slices} · aux {self.aux} · "
                f"cache {self.cache_hits} · SHsaved {self.full_saved} · "
                f"surr {self.screened} · saved {eff:.0f}%")

class FitnessCache:
    """Byte-keyed memo of full-fidelity losses. In image semantics row order
    is spatial identity (CA feeds on it), so keying is exact."""
    def __init__(self, size=4096):
        self.size = size; self.store = {}; self.hits = 0; self.misses = 0
    def lookup(self, g):
        f = self.store.get(np.ascontiguousarray(g).tobytes())
        self.hits += f is not None; self.misses += f is None
        return f
    def insert(self, g, f):
        k = np.ascontiguousarray(g).tobytes()
        if k in self.store: return
        if len(self.store) >= self.size: self.store.pop(next(iter(self.store)))
        self.store[k] = float(f)

class FidelityEvaluator:
    """Class-stratified fixed slices per fidelity level (fair comparisons)."""
    def __init__(self, eval_fn, X, y, fractions, seed=0):
        rng = np.random.default_rng(seed)
        yl = y.argmax(1) if (y.ndim == 2 and y.shape[1] > 1) else np.zeros(len(y), int)
        self.eval_fn = eval_fn; self.X, self.y = X, y; self.slices = []
        for f in fractions:
            parts = []
            for c in np.unique(yl):
                ci = np.flatnonzero(yl == c); rng.shuffle(ci)
                parts.append(ci[:max(int(len(ci) * f), 2)])
            idx = rng.permutation(np.concatenate(parts)) if parts else \
                rng.permutation(len(X))
            self.slices.append(idx[:max(int(len(X) * f), 8)])
    def __call__(self, genome, level):
        s = self.slices[level]
        return self.eval_fn(genome, self.X[s], self.y[s])

class CohortSuccessiveHalving:
    """Whole sibling cohort through fidelity brackets; only survivors get the
    full set (Jamieson & Talwalkar 2016). Returns FULL-set loss for survivors,
    np.inf for eliminated (they are dropped from elitist merge — mixing
    fidelities in the population would corrupt selection). Cache hits count
    as full knowledge."""
    def __init__(self, fidelity, cache, ledger, eta=2, min_keep=2):
        self.fid, self.cache, self.ledger = fidelity, cache, ledger
        self.eta, self.min_keep = eta, min_keep
    def run(self, genomes):
        n = len(genomes)
        if n == 0: return np.array([])
        L = len(self.fid.slices)
        fits = np.full(n, np.inf); at_full = np.zeros(n, bool)
        active = list(range(n)); lvl = 0
        while True:
            for i in active:
                f = self.cache.lookup(genomes[i])
                if f is not None:
                    self.ledger.cache_hits += 1; at_full[i] = True
                else:
                    f = self.fid(genomes[i], min(lvl, L - 1)); self.ledger.slices += 1
                    if lvl >= L - 1:
                        self.cache.insert(genomes[i], f); at_full[i] = True
                fits[i] = f
            if lvl >= L - 1: break
            if len(active) <= self.min_keep:
                for i in active:
                    if not at_full[i]:
                        f = self.fid(genomes[i], L - 1); self.ledger.slices += 1
                        self.cache.insert(genomes[i], f); fits[i] = f; at_full[i] = True
                break
            k = max(self.min_keep, len(active) // self.eta)
            active = sorted(active, key=lambda i: fits[i])[:k]; lvl += 1
        self.ledger.full_saved += int(n - at_full.sum())
        return fits

def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float(ra @ rb / d) if d > 0 else 0.0

class RidgeSurrogate:
    """Closed-form ridge on tile-histogram + behavior features; self-disarms
    while holdout Spearman ρ < gate (Jin 2011, surrogate-assisted EC)."""
    def __init__(self, vocab, window=512, refit_every=5, rho_gate=0.30, lam=1.0, seed=0):
        self.vocab = vocab; self.window = window; self.refit_every = refit_every
        self.rho_gate, self.lam = rho_gate, lam
        self.rng = np.random.default_rng(seed)
        self.F = deque(maxlen=window); self.Y = deque(maxlen=window)
        self.w = self.mu = self.sd = self._ym = None
        self.ok = False; self.rho = 0.0
    def features(self, g):
        g = np.asarray(g)
        t = self.vocab.tile_of(g[None])[0]
        h = np.bincount(t.ravel(), minlength=self.vocab.T).astype(float)
        h /= max(h.sum(), 1.0)
        return np.concatenate([h, BehaviorArchive.describe(g).astype(float),
                               [h.std(), h.max()]])
    def observe(self, g, f):
        self.F.append(self.features(g)); self.Y.append(float(f))
    def refit(self):
        if len(self.Y) < 24: return
        F, Y = np.asarray(self.F), np.asarray(self.Y)
        idx = self.rng.permutation(len(Y)); ntr = max(int(len(Y) * 0.75), 8)
        tr, ho = idx[:ntr], idx[ntr:]
        if len(ho) < 4: return
        self.mu, self.sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        self._ym = float(Y[tr].mean())
        A = (F[tr] - self.mu) / self.sd
        self.w = np.linalg.solve(A.T @ A + self.lam * np.eye(A.shape[1]),
                                 A.T @ (Y[tr] - self._ym))
        self.rho = _spearman(((F[ho] - self.mu) / self.sd) @ self.w, Y[ho] - self._ym)
        self.ok = self.rho >= self.rho_gate
    def predict(self, g):
        return float((((self.features(g) - self.mu) / self.sd) @ self.w) + self._ym)
    def screen(self, genomes, k):
        """Keep k lowest-predicted-loss. Returns (kept, n_screened_away)."""
        if not self.ok or len(genomes) <= k: return list(genomes), 0
        s = np.array([self.predict(g) for g in genomes])
        keep = np.argsort(s)[:k]
        return [genomes[i] for i in keep], len(genomes) - k

# ═══════════════════════════════════════════════════════════════════════════
# 11. ZEROTH-ORDER SEARCHERS — low-rank NES + SPSA
# ═══════════════════════════════════════════════════════════════════════════
class LowRankNES:
    """Diagonal + rank-r basis noise (Ros & Hansen 2015), antithetic pairs
    (Salimans 2017), rank shaping (Wierstra 2014), CSA step size (Hansen 2001).
    tell() expects LOSSES (minimize)."""
    def __init__(self, shape, rank=8, lr=0.03, sigma0=0.08, sigma_min=0.01, rng=None):
        self.shape = tuple(shape); self.D = int(np.prod(shape))
        self.rank = max(1, min(rank, self.D))
        self.lr, self.sigma = float(lr), float(sigma0); self.sigma_min = float(sigma_min)
        self.rng = rng or np.random.default_rng()
        U, _ = np.linalg.qr(self.rng.standard_normal((self.D, self.rank)))
        self.U = U.astype(np.float32)
        self.mu = np.zeros(self.shape, np.float32)
        self.path = np.zeros(self.D, np.float32); self._eps = None
    def _sample(self, n):
        e = self.rng.standard_normal((n, self.D)).astype(np.float32)
        z = self.rng.standard_normal((n, self.rank)).astype(np.float32)
        return (e + (z @ self.U.T) / np.sqrt(self.rank)).reshape((n,) + self.shape)
    def ask(self, anchor, n_pairs):
        a = np.asarray(anchor, np.float32) + self.mu
        self._eps = self._sample(n_pairs) * np.float32(self.sigma)
        return np.clip(a + self._eps, 0, 1), np.clip(a - self._eps, 0, 1)
    def tell(self, f_plus, f_minus):
        f = -np.concatenate([f_plus, f_minus]); n = len(f_plus)
        r = np.argsort(np.argsort(f)).astype(np.float32)
        u = (r + 0.5) / len(f) - 0.5
        u /= np.abs(u).max() + 1e-9
        w = np.clip(u[:n], 0, None); w /= w.sum() + 1e-9
        E = self._eps[:n].reshape(n, self.D)
        g = (w[:, None] * E).sum(0)
        self.mu = np.clip(self.mu + (self.lr / max(self.sigma, 1e-6)) *
                          g.reshape(self.shape), -0.25, 0.25).astype(np.float32)
        G = (E.T @ (w[:, None] * (E @ self.U))).astype(np.float32)
        Un, _ = np.linalg.qr(self.U + 0.1 * G)
        self.U = Un.astype(np.float32)
        tgt = self.sigma * np.sqrt((1 + 1 / self.rank) * float((w ** 2).sum()) * self.D)
        self.path = 0.9 * self.path + np.sqrt(0.99) * g.astype(np.float32)
        self.sigma = float(np.clip(self.sigma * np.exp(
            0.25 * (np.linalg.norm(self.path) / (tgt + 1e-12) - 1)),
            self.sigma_min, 0.5))
    def recenter(self, old, new):
        self.mu = np.clip(self.mu + (np.asarray(old, np.float32)
                                     - np.asarray(new, np.float32)), -0.25, 0.25)
    def apply(self, anchor):
        return np.clip(np.asarray(anchor) + self.mu, 0, 1)

class SPSAPolish:
    """Simultaneous Perturbation Stochastic Approximation (Spall 1992):
    a descent direction on L from TWO loss values per step, size-independent.
    Value-only — zeroth-order, never a chain rule."""
    def __init__(self, L, rng, a=0.05, c=0.06, alpha=0.602, gamma=0.101):
        self.L, self.rng, self.a, self.c = L, rng, a, c
        self.alpha, self.gamma = alpha, gamma
    def improve(self, g, f0, steps=6):
        w = np.asarray(g, np.float64).copy()
        best_g, best_f = w.copy(), float(f0)
        for t in range(1, steps + 1):
            delta = self.rng.choice([-1.0, 1.0], size=w.shape)
            cp = self.c / (t ** self.gamma)
            fp = self.L(np.clip(w + cp * delta, 0, 1))
            fm = self.L(np.clip(w - cp * delta, 0, 1))
            ghat = (fp - fm) / (2 * cp) * delta
            w = np.clip(w - (self.a / (t ** self.alpha)) * ghat, 0, 1)
            f = self.L(w)
            if f < best_f: best_g, best_f = w.copy(), f
        return best_g, best_f

# ═══════════════════════════════════════════════════════════════════════════
# 12. MEMORIES — MAP-Elites, column pool, AOS, islands, restarts
# ═══════════════════════════════════════════════════════════════════════════
class BehaviorArchive:
    """MAP-Elites (Mouret & Clune 2015): best genome per behavior cell.
    reward convention: HIGHER = better."""
    def __init__(self, res=(8, 8, 8), rng=None, descriptor_fn=None):
        self.res = res; self.cells = {}; self.descs = []
        self.rng = rng or np.random.default_rng()
        self.descriptor_fn = descriptor_fn or BehaviorArchive.describe
    @staticmethod
    def describe(g):
        g = np.asarray(g)
        hb = hue_band(g[..., 0]).ravel()
        p = np.bincount(hb, minlength=N_ACTIVATIONS).astype(float)
        p = p / max(p.sum(), 1.0); p = p[p > 0]
        d0 = float(-(p * np.log(p)).sum() / np.log(N_ACTIVATIONS))
        return np.array([d0, float(np.mean(g[..., 1])),
                         float(np.std(g[..., 2]))], np.float32)
    def add(self, genome, reward):
        g = np.asarray(genome)
        d = self.descriptor_fn(g)
        cell = tuple((np.clip(d, 0, 0.9999) * self.res).astype(int))
        cur = self.cells.get(cell)
        if cur is None or reward >= cur[1]:
            if cur is None: self.descs.append(d)
            else:
                try: self.descs[self.descs.index(cur[2])] = d
                except ValueError: self.descs.append(d)
            self.cells[cell] = (g.copy(), float(reward), d)
    def novelty(self, genome, k=5, extra_pop=None):
        d = self.descriptor_fn(np.asarray(genome))
        pool = list(self.descs)
        if extra_pop is not None and len(extra_pop):
            pool = pool + [self.descriptor_fn(np.asarray(x)) for x in extra_pop]
        if len(pool) < k: return 0.0
        dist = np.linalg.norm(np.asarray(pool) - d, axis=1)
        return float(np.mean(np.sort(dist)[:k]))
    def sample(self):
        if not self.cells: return None
        return self.cells[self.rng.choice(list(self.cells.keys()))][0]
    def top(self, n=5):
        items = sorted(self.cells.values(), key=lambda t: -t[1])
        return [g for g, f, d in items[:n]]
    def heatmap(self):
        if not self.cells: return None
        grid = np.full(self.res[:2], np.nan)
        for (a, b, _), (g, f, d) in self.cells.items():
            i, j = a % self.res[0], b % self.res[1]
            grid[i, j] = f if np.isnan(grid[i, j]) else max(grid[i, j], f)
        return grid

class ColumnPool:
    """Fittest hidden-layer COLUMNS (one column = one FFN layer); grafting
    moves co-adapted neuron banks atomically (SANE / gene-pool lineage)."""
    def __init__(self, keep=8):
        self.cols = {}; self.keep = keep
    def add(self, g, reward):
        g = np.asarray(g)
        for j in range(1, g.shape[1] - 1):
            lst = self.cols.setdefault(j, [])
            lst.append((float(reward), g[:, j, :].copy()))
            lst.sort(key=lambda t: -t[0]); del lst[self.keep:]
    def graft(self, g, rng, prob=0.15):
        g = np.asarray(g, np.float64).copy()
        for j, lst in self.cols.items():
            if j < g.shape[1] and lst and rng.random() < prob:
                g[:, j, :] = lst[rng.integers(len(lst))][1]
        return g

class AdaptiveOperatorSelector:
    """Bandit over variation operators (Fialho et al. 2010, credit=avg)."""
    def __init__(self, names, p_min=0.05, seed=0):
        self.names = list(names); self.n = len(names); self.p_min = p_min
        self.q = np.ones(self.n) * 0.5; self.t = 0
        self.recent = {op: deque(maxlen=20) for op in self.names}
        self.rng = np.random.default_rng(seed)
    def get_probabilities(self):
        p = np.maximum(self.q / max(self.q.sum(), 1e-12), 0)
        p = self.p_min + p / max(p.sum(), 1e-12) * (1 - self.n * self.p_min)
        return p / p.sum()
    def select(self):
        self.t += 1
        return int(self.rng.choice(self.n, p=self.get_probabilities()))
    def update(self, op_idx, reward):
        self.recent[self.names[op_idx]].append(reward)
        r = np.mean(self.recent[self.names[op_idx]])
        alpha = 1.0 / max(1, self.t / 10)
        self.q[op_idx] = (1 - alpha) * self.q[op_idx] + alpha * r

class IslandHost:
    """Per-island WFC grammars ('dialects') + ring migration (Cantú-Paz 1998).
    Each island keeps an elite grammar and a DIVERSE grammar (learned from
    distinct archive cells) for exploration-phase synthesis."""
    def __init__(self, engine, n, rng):
        self.e = engine; self.n = max(1, int(n)); self.rng = rng
        c = engine.cfg
        mk = lambda: WFCTileVocab(c.get('wfc_hue_bands', N_ACTIVATIONS),
                                  c.get('wfc_sat_bands', 4), c.get('wfc_val_bands', 3))
        self.vocabs = [mk() for _ in range(self.n)]
        self.diverse = [mk() for _ in range(self.n)]
        self.cur = 0; self.data_prior = None; self._geo = None; self._gk = None
    def set_island(self, i): self.cur = i % self.n
    def learn(self, genomes, decay=0.85): return self.vocabs[self.cur].learn(genomes, decay)
    def learn_diverse(self, genomes): return self.diverse[self.cur].learn(genomes)
    def make_genomes(self, n_out, shape, seed_tiles=None, temp=None, mode='elite'):
        t0 = time.perf_counter()
        v = self.vocabs[self.cur] if mode == 'elite' else self.diverse[self.cur]
        c = self.e.cfg
        gen = FastWFC(v, temp or c.get('wfc_temperature', 0.85),
                      c.get('wfc_prop_strength', 0.9), c.get('wfc_prop_sweeps', 1),
                      c.get('wfc_max_rounds', 6), self.rng)
        pr = np.ones(shape)
        if c.get('geo_prior', 'local') == 'local':
            key = (shape[0], shape[1], c.get('geo_lam', 3.0))
            if self._gk != key:
                self._geo = build_geometry_prior(shape[0], shape[1], c.get('geo_lam', 3.0))
                self._gk = key
            pr *= self._geo
        if self.data_prior is not None and self.data_prior.shape == shape:
            pr *= self.data_prior
        gen.prior = pr
        gen.prepare(shape[0], shape[1], seed_tiles)
        while not gen.run_round(): pass
        out = gen.finish()
        self.e.ms_spent += (time.perf_counter() - t0) * 1e3
        return out
    def crossover(self, pa, pb, island, alpha=0.5, temp=None, mode='elite'):
        self.set_island(island)
        v = self.vocabs[self.cur] if mode == 'elite' else self.diverse[self.cur]
        ta = v.tile_of(np.asarray(pa)[None])[0]
        tb = v.tile_of(np.asarray(pb)[None])[0]
        mix = np.where(self.rng.random(ta.shape) < alpha, ta, tb)
        return self.make_genomes(1, ta.shape, mix[None], temp=temp, mode=mode)[0]

class RestartManager:
    """Mann–Kendall-gated IPOP restarts (Auger & Hansen 2005); the hot-WFC
    refill is CHC's cataclysmic mutation analog (Eshelman 1991)."""
    def __init__(self, hist_len=30, p_gate=0.05, stag_gate=25, max_mult=4):
        self.hist = deque(maxlen=hist_len)
        self.p_gate, self.stag_gate, self.max_mult = p_gate, stag_gate, max_mult
        self.mult = 1; self.restarts = 0
    def observe(self, reward): self.hist.append(float(reward))
    def should_restart(self, stagnation):
        if len(self.hist) < self.hist.maxlen or stagnation < self.stag_gate: return False
        z, p = mann_kendall_trend(list(self.hist))
        return (p > self.p_gate) or (z <= 0.5)
    def trigger(self):
        self.mult = min(self.mult * 2, self.max_mult)
        self.restarts += 1; self.hist.clear()
        return self.mult

# ═══════════════════════════════════════════════════════════════════════════
# 13. SPECTRAL DRIVER — PC1..PCn + loss regime steer GA/WFC/CA
# ═══════════════════════════════════════════════════════════════════════════
class DataSpectra:
    """PCA of training inputs (fitted once). λ-spectrum, participation ratio,
    per-feature contribution to the top-k subspace."""
    def __init__(self, X, n_pc=8):
        self.mu = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        lam = (S ** 2) / max(len(X) - 1, 1)
        k = min(n_pc, len(lam))
        self.V, self.lam = Vt[:k], lam[:k]
        self.total_var = float(lam.sum())
        self.pr = float(lam.sum() ** 2 / ((lam ** 2).sum() + 1e-12))
    def feature_gain(self):
        g = (self.V ** 2).T @ self.lam
        return g / (g.max() + 1e-12)
    def scores(self, X): return (X - self.mu) @ self.V.T
    def error_attribution(self, Xs, per_sample_loss, tau=0.5):
        """Soft-assign each sample's error to dominant input PCs — a linear
        guidance heuristic, NOT causal attribution (that would need gradients)."""
        a = self.scores(Xs) ** 2
        a /= (a.sum(1, keepdims=True) + 1e-12)
        e = a.T @ np.asarray(per_sample_loss, float)
        e = np.clip(e / (e.sum() + 1e-12), 0, None)
        w = np.exp((e - e.max()) / max(tau, 1e-9)); w /= w.sum()
        return e, w, int(np.argmax(e))

class GenomeEDA:
    """Estimation-of-distribution in the elite genomes' principal subspace
    (Bosman & Thierens 2000; PCA-EDA). r and σ set each gen by loss regime."""
    def __init__(self, shape, rng, r_min=2, r_max=32):
        self.shape = tuple(shape); self.rng = rng
        self.r_min, self.r_max = r_min, r_max
        self.mean = self.E = self.s = None
    def fit(self, genomes):
        G = np.asarray([np.asarray(g).ravel() for g in genomes], float)
        self.mean = G.mean(0)
        _, S, Vt = np.linalg.svd(G - self.mean, full_matrices=False)
        self.E = Vt; self.s = S / max(np.sqrt(max(len(G) - 1, 1)), 1.0)
    def around(self, parents, r=None, sigma=0.6):
        if self.mean is None: return None
        m = int(min(r or len(self.s), len(self.s), self.r_max, max(1, self.r_min)))
        p = np.asarray([np.asarray(x).ravel() for x in parents], float).mean(0)
        z = self.rng.standard_normal(m) * (self.s[:m] * sigma)
        return np.clip(p + z @ self.E[:m], 0, 1).reshape(self.shape)
    def sample(self, n, r=None, sigma=1.0):
        if self.mean is None: return None
        m = int(min(r or len(self.s), len(self.s), self.r_max, max(1, self.r_min)))
        z = self.rng.standard_normal((n, m)) * (self.s[:m] * sigma)
        return np.clip(self.mean[None] + z @ self.E[:m], 0, 1).reshape((n,) + self.shape)

class LossRegime:
    """Loss-trajectory state machine (Rechenberg-style parameter control)."""
    EXPLORE = {'init': 0.5, 'refine': 0.15, 'noisy': 0.5, 'plateau': 0.75, 'regress': 0.9}
    def __init__(self, hist_len=30):
        self.hist = deque(maxlen=hist_len)
        self.regime, self.explore = 'init', 0.5
        self.vol = self.slope = self.pct = 0.0; self._scale = None
    def observe(self, loss): self.hist.append(float(loss)); return self
    def update(self):
        h = list(self.hist)
        if len(h) < 6: return self
        a = np.asarray(h)
        self._scale = (abs(a[0]) + 1e-12) if self._scale is None \
            else 0.95 * self._scale + 0.05 * (abs(a[-1]) + 1e-9)
        d = np.diff(a)
        self.slope = float(np.median(d) / self._scale)
        self.vol = float(np.median(np.abs(d - np.median(d))) / self._scale)
        self.pct = float(np.mean(a[-1] >= a))
        z, _ = mann_kendall_trend(h)
        if self.vol > 0.05: r = 'noisy'
        elif z < -0.3 and self.slope < 0: r = 'refine'
        elif z > 0.3 or self.slope > 0.002: r = 'regress'
        else: r = 'plateau'
        self.regime = r
        self.explore += 0.3 * (self.EXPLORE[r] - self.explore)
        return self

class SpectralDriver:
    """Emits a DriveSignal each generation; rewrites CA rule/dosage, WFC
    temperature + grammar mode, EDA r/σ, NES σ multiplier, and schedules KAN
    grid-extension (plateau → refine) and the prune ladder (refine regime,
    while elite edge-L1 is above target). Probe forwards are ledger-counted
    and NEVER touch selection."""
    def __init__(self, engine, islands, archive, cfg=None, rng=None):
        self.e, self.islands, self.archive = engine, islands, archive
        self.cfg = dict(cfg or {})
        self.rng = rng or np.random.default_rng()
        self.data = None; self.eda = None; self.regime = LossRegime()
        self.Xp = self.yp = None; self.signal = {}
        self._diverse_pool = []
        self.refine = 1; self._plateau_run = 0; self._refine_run = 0
        self._prune_i = 0; self._elite_edge_l1 = 1.0
    def fit_data(self, X, y):
        self.data = DataSpectra(X, self.cfg.get('spec_n_pc', 8))
        n = max(int(len(X) * self.cfg.get('spec_probe_frac', 0.25)), 16)
        idx = self.rng.choice(len(X), min(n, len(X)), replace=False)
        self.Xp, self.yp = X[idx], y[idx]
    def attach_shape(self, shape):
        H, W = shape[0], shape[1]
        g = self.data.feature_gain(); row = np.ones(H)
        d = min(len(g), H); row[:d] = 0.25 + 0.75 * g[:d]
        col = 0.35 + 0.65 * np.exp(-np.arange(W) / max(self.cfg.get('geo_lam', 3.0), 1e-9))
        P = row[:, None] * col[None, :]
        self.islands.data_prior = P / P.mean()
        self.eda = GenomeEDA(shape, self.rng, self.cfg.get('eda_r_min', 2),
                             self.cfg.get('eda_r_max', 32))
    def _refresh_diverse_pool(self):
        cells = list(self.archive.cells.values())
        if cells:
            picks = self.rng.choice(len(cells), min(12, len(cells)), replace=False)
            self._diverse_pool = [cells[i][0] for i in picks]
    def activation_probe(self, net, elites, ledger=None):
        """Functional descriptors from forward-pass hidden states only:
        participation ratio + dead-neuron fraction."""
        prs, deads = [], []
        eps = self.cfg.get('pr_dead_eps', 1e-6)
        for g in elites[:self.cfg.get('spec_probe_elites', 6)]:
            _, states = net.from_genome(g).forward(self.Xp, return_states=True)
            S = states[-2] if len(states) > 1 else states[-1]
            v = S.var(0) + 1e-12
            prs.append(float(v.sum() ** 2 / (v ** 2).sum()))
            deads.append(float(np.mean(v < eps)))
        if ledger: ledger.aux += len(prs)
        return float(np.mean(prs)), float(np.mean(deads))
    def _ca_policy(self):
        """THE CA 'learning': loss regime → rule + dosage."""
        r = self.regime.regime
        if r == 'regress':  return 'waves', 2
        if r == 'plateau':  return 'reaction', 2
        if r == 'noisy':    return 'majority', 1
        return ('erosion', 1) if self.regime.pct < 0.30 else ('majority', 1)
    def _refine_schedule(self):
        cap = int(self.cfg.get('img_refine_max', 1))
        after = int(self.cfg.get('refine_plateau_after', 6))
        if self.regime.regime == 'plateau' and self._plateau_run >= after \
                and self.refine < cap:
            self.refine *= 3; self._plateau_run = 0
            return self.refine
        return None
    def _prune_schedule(self):
        ladder = self.cfg.get('prune_ladder', (0.0, .05, .10, .15, .20))
        if self._elite_edge_l1 > self.cfg.get('prune_l1_target', 0.45) \
                and self._prune_i + 1 < len(ladder):
            self._prune_i += 1
            return ladder[self._prune_i]
        return None
    def update(self, gen, pop, fits_loss, net, per_sample_fn, ledger=None):
        if not self.cfg.get('spec_enabled', True): return self.signal
        pop = np.asarray(pop); fits_loss = np.asarray(fits_loss)
        self.regime.observe(float(fits_loss.min())).update()
        ex = self.regime.explore
        self._plateau_run = self._plateau_run + 1 if self.regime.regime == 'plateau' else 0
        self._refine_run = self._refine_run + 1 if self.regime.regime == 'refine' else 0
        kk = max(4, int(len(pop) * 0.25))
        self._elite_edge_l1 = float(pop[np.argsort(fits_loss)[:kk]][..., 1].mean())
        if gen % max(1, self.cfg.get('spec_refit_every', 5)) == 0:
            k = max(4, int(len(pop) * self.cfg.get('eda_pool_frac', 0.25)))
            if self.eda is not None:
                self.eda.fit(pop[np.argsort(fits_loss)[:k]])
            self._refresh_diverse_pool()
            if self.cfg.get('wfc_conditional', True) and self._diverse_pool:
                for i in range(self.islands.n):
                    self.islands.set_island(i)
                    self.islands.learn_diverse(self._diverse_pool)
        e = w = None; fail_pc = 0
        try:
            best = pop[int(np.argmin(fits_loss))]
            per_s = per_sample_fn(best, self.Xp, self.yp)
            if ledger: ledger.aux += 1
            e, w, fail_pc = self.data.error_attribution(self.Xp, per_s)
        except Exception:
            pass
        act_pr, dead = 0.0, 0.0
        try:
            act_pr, dead = self.activation_probe(
                net, pop[np.argsort(fits_loss)[:6]], ledger)
        except Exception:
            pass
        rule, steps = self._ca_policy()
        if self.cfg.get('ca_policy', True):
            self.e.ca.rule, self.e.ca.steps = rule, steps
        r_max = self.cfg.get('eda_r_max', 32); r_min = self.cfg.get('eda_r_min', 2)
        self.signal = {
            'regime': self.regime.regime, 'explore': ex,
            'wfc_temp': self.cfg.get('base_wfc_temp', 0.85) * (1.0 + 1.2 * ex),
            'grammar_mode': 'diverse' if (ex > 0.6 and self.cfg.get('wfc_conditional', True))
                            else 'elite',
            'ca_rule': rule, 'ca_steps': steps,
            'eda_r': int(r_min + (r_max - r_min) * ex), 'eda_sigma': 0.4 + 1.2 * ex,
            'nes_sigma_mult': 1.15 if self.regime.regime == 'plateau'
                              else (0.95 if self.regime.regime == 'refine' else 1.0),
            'fail_pc': fail_pc, 'pc_err': e, 'act_pr': act_pr, 'dead_frac': dead,
            'refine_to': self._refine_schedule(),
            'prune_to': self._prune_schedule() if self.cfg.get('prune_anneal', True)
                        and self._refine_run and
                        self._refine_run % self.cfg.get('prune_every', 5) == 0 else None,
        }
        return self.signal
    def status_text(self):
        s = self.signal
        if not s: return "spectral: —"
        pc = ""
        if s.get('pc_err') is not None:
            top = np.argsort(s['pc_err'])[-3:][::-1]
            pc = " PCerr[" + ",".join(f"P{i+1}:{s['pc_err'][i]:.2f}" for i in top) + "]"
        return (f"regime {s['regime']} | explore {s['explore']:.2f}"
                f" | T_wfc {s['wfc_temp']:.2f} ({s['grammar_mode']})"
                f" | CA {s['ca_rule']}×{s['ca_steps']}"
                f" | EDA r{s['eda_r']} σ{s['eda_sigma']:.2f}"
                f" | actPR {s['act_pr']:.1f} dead {s['dead_frac']*100:.0f}%{pc}")

# ═══════════════════════════════════════════════════════════════════════════
# 14. IMAGE TAXONOMY — named circuit families + curation
# ═══════════════════════════════════════════════════════════════════════════
FUNCTION_FAMILIES = {
    'periodic':  {'Sine', 'Cosine', 'Sinc'},
    'radial':    {'Gaussian'},
    'threshold': {'Step', 'Sign', 'HardSigmoid', 'HardSwish'},
    'squash':    {'Sigmoid', 'Tanh', 'Softsign'},
    'poly':      {'Square', 'Cube'},
    'relu':      {'ReLU', 'LeakyReLU', 'PReLU', 'Swish', 'GELU', 'ELU',
                  'Softplus', 'Mish', 'BentIdentity', 'Abs'},
    'linear':    {'Linear'},
}

def _entropy(p):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def image_features(g, refine=1):
    g = np.asarray(g); H, W, _ = g.shape
    fn, _ = decode_refined(g[..., 0], refine)
    sat, val = g[..., 1], g[..., 2]
    live = sat > 1e-3
    live_col = live.mean(0)
    hb = fn[live]
    cnt = np.bincount(hb, minlength=N_ACTIVATIONS) if hb.size else np.zeros(N_ACTIVATIONS, int)
    share = cnt / max(cnt.sum(), 1)
    fam = {k: float(sum(share[i] for i, nm in enumerate(ACTIVATION_NAMES) if nm in v))
           for k, v in FUNCTION_FAMILIES.items()}
    hue_cnt = np.bincount(fn.ravel(), minlength=N_ACTIVATIONS) / (H * W)
    purity = float(np.mean([np.bincount(fn[:, j], minlength=N_ACTIVATIONS).max()
                            / max(fn[:, j].size, 1) for j in range(W)]))
    return dict(
        live=float(live.mean()), live_col=live_col,
        hollow=float(np.mean(live_col[1:-1] < 0.05)) if W > 2 else 0.0,
        purity=purity, hue_H=float(_entropy(hue_cnt) / np.log(N_ACTIVATIONS)),
        edge_L1=float(sat.mean()), sat_std=float(sat.std()),
        col_var=float(fn.std(0).mean()), row_var=float(fn.std(1).mean()),
        val_mean=float(val.mean()), val_std=float(val.std()),
        **{f'fam_{k}': v for k, v in fam.items()})

def _c01(x): return float(np.clip(x, 0.0, 1.0))

def _type_scores(F):
    sl, st = 1 - F['col_var'] / 1.5, 1 - F['row_var'] / 1.5
    return {
        'Dead Zone':       _c01(max((0.15 - F['live']) / 0.15, F['hollow'] / 0.3)),
        'Sparse Logic':    _c01((0.45 - F['edge_L1']) / 0.45)
                           * _c01(F['fam_threshold'] / 0.30)
                           * _c01(0.4 + 0.6 * F['purity']),
        'Periodic Wave':   _c01(F['fam_periodic'] / 0.40),
        'Radial Blobs':    _c01(F['fam_radial'] / 0.30) * _c01(F['sat_std'] / 0.20),
        'Squash Net':      _c01(F['fam_squash'] / 0.40),
        'Polynomial Core': _c01(F['fam_poly'] / 0.35),
        'Amplifier Bank':  _c01((F['fam_relu'] + F['fam_linear'] - 0.30) / 0.60)
                           * _c01((F['edge_L1'] - 0.35) / 0.40),
        'Slab Circuit':    _c01((sl - 0.5) * 2) * _c01((F['purity'] - 0.5) * 2),
        'Banded Stripes':  _c01((st - 0.5) * 2),
        'Plaid Field':     _c01(F['purity'] - 0.30) * _c01(0.85 - F['purity'])
                           * _c01(F['hue_H'] - 0.40),
        'Noisy Static':    _c01((F['hue_H'] - 0.80) / 0.20)
                           * _c01((F['col_var'] - 1.2) / 1.0)
                           * _c01((F['row_var'] - 1.2) / 1.0),
    }

IMAGE_TYPE_HELP = {
    'Dead Zone': 'almost no live connections — structurally incapable',
    'Sparse Logic': 'low wiring mass, threshold/sign gates — compact logic',
    'Periodic Wave': 'sin/cos/sinc dominated — spectral/oscillatory circuits',
    'Radial Blobs': 'gaussian-kernel dominated — local bump detectors',
    'Squash Net': 'sigmoid/tanh dominated — bounded saturating cells',
    'Polynomial Core': 'square/cube dominated — multiplicative interactions',
    'Amplifier Bank': 'dense relu/linear with strong gains — signal amplifiers',
    'Slab Circuit': 'column-pure activation layers — deep, organized',
    'Banded Stripes': 'row-pure activation bands — feature-type rows',
    'Plaid Field': 'mixed but organized functions — blended filters',
    'Noisy Static': 'no spatial organization — unstructured search material',
    'Hybrid Mosaic': 'no family dominates — mixed circuit',
}

class ImageTyper:
    def __init__(self, refine=1): self.refine = refine
    def type_of(self, g):
        F = image_features(g, self.refine)
        S = _type_scores(F)
        name = max(S, key=S.get); conf = S[name]
        if conf < 0.25: name, conf = 'Hybrid Mosaic', 0.2
        return {'type': name, 'conf': round(min(conf, 1.0), 2), 'features': F}

class TypeLedger:
    """Per-family intelligence accounting across the run."""
    def __init__(self): self.groups = {}
    def record(self, gen, tname, score, passed, genome, conf=1.0, chain=None):
        G = self.groups.setdefault(tname, dict(n=0, cert=0, best=-1.0, scores=[],
                                               last_gen=0, best_g=None,
                                               best_conf=0.0, best_chain=None))
        G['n'] += 1; G['last_gen'] = gen
        G['scores'].append(float(score)); G['scores'] = G['scores'][-20:]
        if passed: G['cert'] += 1
        if score > G['best']:
            G['best'] = float(score)
            G['best_g'] = np.asarray(genome).copy(); G['best_conf'] = conf
            G['best_chain'] = chain
    def trend(self, name):
        s = self.groups[name]['scores']
        if len(s) < 3: return 0.0
        return float(np.polyfit(np.arange(len(s)), np.asarray(s), 1)[0])
    def report(self):
        rows = []
        for name, G in self.groups.items():
            tr = self.trend(name)
            if G['cert'] > 0 and tr >= 0: rec = 'BREED'
            elif G['cert'] > 0: rec = 'KEEP'
            elif name == 'Dead Zone' and G['n'] >= 2 and G['best'] < 40: rec = 'PRUNE NOW'
            elif len(G['scores']) >= 3 and G['best'] < 50 and tr < -3: rec = 'PRUNE'
            elif len(G['scores']) >= 3 and tr > 3: rec = 'WATCH↑'
            else: rec = 'WATCH'
            rows.append((name, G['n'], G['cert'], G['best'], round(tr, 1), rec,
                         G['last_gen']))
        return sorted(rows, key=lambda r: -r[3])
    def text(self):
        lines = ["family                      n cert  best trend rec      last"]
        for name, n, c, b, tr, rec, lg in self.report():
            lines.append(f"{name:<26s}{n:3d} {c:4d} {b:6.1f} {tr:+5.1f} "
                         f"{rec:<9s} {lg:4d}")
        return "\n".join(lines)

def contact_sheet(images, path, cell=64, cols=4):
    if not PIL_AVAILABLE or not images: return None
    from PIL import ImageDraw
    rows = int(np.ceil(len(images) / cols))
    sheet = PILImage.new('RGB', (cols * cell, rows * cell), (18, 18, 28))
    for i, g in enumerate(images[:cols * rows]):
        im = PILImage.fromarray(_hsv_to_rgb_vec(np.asarray(g))).resize(
            (cell, cell), PILImage.NEAREST)
        sheet.paste(im, ((i % cols) * cell, (i // cols) * cell))
    sheet.save(path); return path

def apply_curation(ctl, ledger, typer, max_replace=None):
    """Replace PRUNE-family members with archive/WFC genomes. Conservative
    cap; replacements re-evaluated (ledger-tracked)."""
    recs = {r[0]: r[5] for r in ledger.report()}
    prune_types = {n for n, r in recs.items() if r.startswith('PRUNE')}
    if not prune_types: return 0
    cap = max_replace or max(1, len(ctl.pop) // 8)
    idxs = [i for i, g in enumerate(ctl.pop)
            if typer.type_of(g)['type'] in prune_types][:cap]
    for i in idxs:
        g = ctl.archive.sample()
        if g is None:
            ctl.islands.set_island(i % ctl.islands.n)
            g = ctl.islands.make_genomes(1, ctl.pop.shape[1:3])[0]
        ctl.pop[i] = np.asarray(g).copy()
        ctl.fits[i] = ctl._full(ctl.pop[i])
    return len(idxs)

# ═══════════════════════════════════════════════════════════════════════════
# 15. LINEAGE — genealogy of the run + graph exports
# ═══════════════════════════════════════════════════════════════════════════
class LineageBook:
    """Every genome is a node keyed by its bytes hash. 0 forward passes,
    0 RNG draws: enabling lineage never changes a run."""
    def __init__(self, typer=None, max_records=8192):
        self.typer = typer or ImageTyper()
        self.max_records = max_records
        self.rec = {}; self.order = deque(); self._n = 0
        self.transitions = defaultdict(int)
        self.op_births = defaultdict(int); self.op_survivors = defaultdict(int)
        self.composition = []; self.scores = {}; self._genome_ring = {}
    @staticmethod
    def key(g): return np.ascontiguousarray(g).tobytes()
    def _trim(self):
        while len(self.order) > self.max_records:
            k = self.order.popleft()
            r = self.rec.get(k)
            if r and 'trimmed' not in r:
                self.rec[k] = {'id': r.get('id'), 'trimmed': True,
                               'root': r.get('root'), 'depth': r.get('depth'),
                               'type': r.get('type')}
            self._genome_ring.pop(k, None)
    def birth(self, g, gen, kind='child', parents=(), op=None, ca=None, style=None):
        k = self.key(g)
        ex = self.rec.get(k)
        if ex and 'trimmed' not in ex:
            ex['seen'] = ex.get('seen', 1) + 1
            return ex
        pk = [self.key(p) for p in parents]
        ptypes, proots, pdepth = [], [], []
        for p in pk:
            pr = self.rec.get(p)
            if pr:
                ptypes.append(pr.get('type', '?'))
                proots.append(pr.get('root', p.hex()[:16]))
                pdepth.append(pr.get('depth', 0))
        t = self.typer.type_of(g)
        r = dict(id=self._n, born=int(gen), kind=kind, op=op, ca=ca, style=style,
                 type=t['type'], conf=t['conf'],
                 depth=(max(pdepth) + 1) if pdepth else 0,
                 root=(proots[0] if proots else k.hex()[:16]),
                 pk=pk, parents=[p.hex()[:16] for p in pk], ptypes=ptypes,
                 fit=None, seen=1)
        self._n += 1
        self.rec[k] = r; self.order.append(k); self._trim()
        self._genome_ring[k] = np.asarray(g).copy()
        if len(self._genome_ring) > 1024:
            self._genome_ring.pop(next(iter(self._genome_ring)))
        if kind == 'child' and op:
            self.op_births[op] += 1
            a, b = (ptypes + ['—', '—'])[:2]
            self.transitions[(a, b, t['type'])] += 1
        return r
    def note_survived(self, g, fit):
        r = self.rec.get(self.key(g))
        if r and 'trimmed' not in r:
            r['fit'] = float(fit) if r['fit'] is None else min(r['fit'], float(fit))
            self.op_survivors[r.get('op') or '—'] += 1
    def note_score(self, g, score, certified, gen):
        self.scores[self.key(g)] = (float(score), bool(certified), int(gen))
    def chain(self, g, max_len=12):
        out, k, seen = [], self.key(g), set()
        while len(out) < max_len:
            r = self.rec.get(k)
            if r is None or 'trimmed' in r: break
            tag = f"g{r['born']:>2d} {r['type']}({r.get('conf',0):.2f})"
            if r.get('kind') == 'child' and r.get('op'):
                tag += f" ←{r['op']}" + (f"+{r['ca']}" if r.get('ca') else "")
            elif r.get('kind') != 'child':
                tag += f" [{r['kind']}" + (f":{r['style']}" if r.get('style') else "") + "]"
            out.append(tag)
            if not r.get('pk'): break
            if r['pk'][0] in seen: out.append('↺'); break
            seen.add(r['pk'][0]); k = r['pk'][0]
        return " → ".join(out)
    def founder_dominance(self, pop):
        c = Counter()
        for g in pop:
            r = self.rec.get(self.key(g))
            c[r.get('root', '?') if r else '?'] += 1
        if not c: return 0.0, '?'
        top, n = c.most_common(1)[0]
        return n / max(len(pop), 1), top
    def snapshot(self, gen, pop):
        c = Counter(self.typer.type_of(g)['type'] for g in pop)
        self.composition.append((int(gen), dict(c)))
        return self.composition[-1]
    def op_stats(self):
        return [(op, b, self.op_survivors.get(op, 0),
                 self.op_survivors.get(op, 0) / max(b, 1))
                for op, b in sorted(self.op_births.items())]
    def top_transitions(self, n=5):
        items = sorted(self.transitions.items(), key=lambda kv: -kv[1])[:n]
        return [f"{a} × {b} → {c} ({v})" for (a, b, c), v in items]
    def _attr(self, k, r):
        s = self.scores.get(k)
        return {'type': r.get('type', '?'), 'kind': r.get('kind', '?'),
                'op': r.get('op') or '', 'born': int(r.get('born', 0)),
                'depth': int(r.get('depth', 0)), 'conf': float(r.get('conf', 0)),
                'root': r.get('root', ''),
                'score': s[0] if s else None,
                'certified': bool(s[1]) if s else False}
    def _node_universe(self, max_nodes=4000, focus=None):
        ks = [k for k in self.order if k in self.rec and 'trimmed' not in self.rec[k]]
        if len(ks) <= max_nodes: return set(ks)
        want = set(focus or []); stack = [k for k in want if k in self.rec]
        keep = set(stack)
        while stack:
            k = stack.pop(); r = self.rec.get(k)
            if not r: continue
            for p in r.get('pk', []):
                if p in self.rec and p not in keep: keep.add(p); stack.append(p)
        return keep
    def to_json(self, path):
        recs = []
        for k in self.order:
            r = self.rec[k]
            d = {kk: vv for kk, vv in r.items() if kk != 'pk'}
            d['key_hex'] = k.hex(); d['pk_hex'] = [p.hex() for p in r.get('pk', [])]
            recs.append(d)
        _json.dump({'next_id': self._n, 'records': recs,
                    'transitions': [[a, b, c, v]
                                    for (a, b, c), v in self.transitions.items()],
                    'composition': [[g, d] for g, d in self.composition]},
                   open(path, 'w'))
        return path
    @classmethod
    def from_json(cls, path, typer=None):
        b = cls(typer); d = _json.load(open(path)); b._n = d['next_id']
        for r in d['records']:
            k = bytes.fromhex(r['key_hex'])
            r2 = {kk: vv for kk, vv in r.items() if kk not in ('key_hex', 'pk_hex')}
            r2['pk'] = [bytes.fromhex(p) for p in r.get('pk_hex', [])]
            b.rec[k] = r2; b.order.append(k)
        for a, bb, c, v in d.get('transitions', []): b.transitions[(a, bb, c)] = v
        b.composition = [(g, d) for g, d in d.get('composition', [])]
        return b

def export_graphml(book, path, max_nodes=4000, focus=None):
    import xml.etree.ElementTree as ET
    uni = book._node_universe(max_nodes, focus)
    id_of = {}
    for k in book.order:
        if k in uni and k in book.rec:
            r = book.rec[k]
            id_of[k] = f"n{r['id']}" if 'id' in r else k.hex()[:16]
    NS = "http://graphml.graphdrawing.org/xmlns"
    root = ET.Element(f'{{{NS}}}graphml')
    nkeys = [('type', 'string'), ('kind', 'string'), ('op', 'string'),
             ('born', 'int'), ('depth', 'int'), ('conf', 'double'),
             ('root', 'string'), ('score', 'double'), ('certified', 'boolean')]
    for i, (nm, tp) in enumerate(nkeys):
        ET.SubElement(root, f'{{{NS}}}key', {'id': f'd{i}', 'for': 'node',
                                             'attr.name': nm, 'attr.type': tp})
    ET.SubElement(root, f'{{{NS}}}key', {'id': 'e0', 'for': 'edge',
                                         'attr.name': 'op', 'attr.type': 'string'})
    g = ET.SubElement(root, f'{{{NS}}}graph', {'id': 'lineage',
                                               'edgedefault': 'directed'})
    for k, nid in id_of.items():
        n = ET.SubElement(g, f'{{{NS}}}node', {'id': nid})
        a = book._attr(k, book.rec[k])
        for i, (nm, _tp) in enumerate(nkeys):
            v = a[nm]
            if v is None: continue
            if nm == 'op' and not v: continue
            ET.SubElement(n, f'{{{NS}}}data', {'key': f'd{i}'}).text = str(v)
    ne = 0
    for k, nid in id_of.items():
        for p in book.rec[k].get('pk', []):
            pid = id_of.get(p)
            if pid is None: continue
            e = ET.SubElement(g, f'{{{NS}}}edge', {'source': pid, 'target': nid})
            if book.rec[k].get('op'):
                ET.SubElement(e, f'{{{NS}}}data', {'key': 'e0'}).text = book.rec[k]['op']
            ne += 1
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    return {'path': path, 'nodes': len(id_of), 'edges': ne}

def export_gexf(book, path, max_nodes=4000, focus=None):
    import xml.etree.ElementTree as ET
    uni = book._node_universe(max_nodes, focus)
    id_of = {}
    for k in book.order:
        if k in uni and k in book.rec:
            r = book.rec[k]
            id_of[k] = f"n{r['id']}" if 'id' in r else k.hex()[:16]
    NS = "http://gexf.net/1.3"; VIZ = "http://gexf.net/1.3/viz"
    root = ET.Element(f'{{{NS}}}gexf', {'version': '1.3'})
    ET.SubElement(root, f'{{{NS}}}meta')
    gr = ET.SubElement(root, f'{{{NS}}}graph', {'defaultedgetype': 'directed'})
    ak = ET.SubElement(gr, f'{{{NS}}}attributes', {'class': 'node'})
    for i, (nm, tp) in enumerate([('type', 'string'), ('born', 'int'),
                                  ('score', 'double'), ('certified', 'boolean')]):
        ET.SubElement(ak, f'{{{NS}}}attribute', {'id': f'a{i}', 'title': nm, 'type': tp})
    gg = ET.SubElement(gr, f'{{{NS}}}nodes')
    pal = {"Slab Circuit": (137, 180, 250), "Banded Stripes": (148, 226, 213),
           "Periodic Wave": (166, 227, 161), "Sparse Logic": (249, 226, 175),
           "Radial Blobs": (250, 179, 135), "Squash Net": (203, 166, 247),
           "Polynomial Core": (245, 194, 231), "Amplifier Bank": (148, 187, 208),
           "Plaid Field": (116, 199, 236), "Noisy Static": (108, 112, 134),
           "Dead Zone": (243, 139, 168), "Hybrid Mosaic": (88, 91, 112)}
    for k, nid in id_of.items():
        a = book._attr(k, book.rec[k])
        r, c, b = pal.get(a['type'], (120, 120, 120))
        if a['certified']: c = 255
        sz = 4.0 + (a['score'] or 0) / 12.0
        n = ET.SubElement(gg, f'{{{NS}}}node',
                          {'id': nid, 'label': f"{a['type']} g{a['born']}"})
        ET.SubElement(n, f'{{{VIZ}}}color', {'r': str(r), 'g': str(c), 'b': str(b)})
        ET.SubElement(n, f'{{{VIZ}}}size', {'value': f'{sz:.1f}'})
        att = ET.SubElement(n, f'{{{NS}}}attvalues')
        for i, v in enumerate([a['type'], a['born'], a['score'], a['certified']]):
            if v is None: continue
            ET.SubElement(att, f'{{{NS}}}attvalue', {'for': f'a{i}', 'value': str(v)})
    ge = ET.SubElement(gr, f'{{{NS}}}edges'); ne = 0
    for k, nid in id_of.items():
        for p in book.rec[k].get('pk', []):
            pid = id_of.get(p)
            if pid is None: continue
            ET.SubElement(ge, f'{{{NS}}}edge',
                          {'id': f'e{ne}', 'source': pid, 'target': nid}); ne += 1
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    return {'path': path, 'nodes': len(id_of), 'edges': ne}

def export_timeline_csv(book, path):
    types = sorted({t for _, d in book.composition for t in d})
    with open(path, 'w', newline='') as fh:
        w = _csv.writer(fh); w.writerow(['gen'] + types)
        for gen, d in book.composition:
            w.writerow([gen] + [d.get(t, 0) for t in types])
    return path

def _k_to_genome(book, k):
    g = book._genome_ring.get(k)
    if g is not None: return g
    h = {'Slab Circuit': 0.3, 'Periodic Wave': 0.9, 'Sparse Logic': 0.8}.get(
        book.rec.get(k, {}).get('type', '?'), 0.5)
    return np.full((6, 8, 3), [h, 0.6, 0.5])

def render_fan(book, champions, path, cell=52, pad=6):
    """Ancestor-fan PNG: each champion traces its primary-parent chain."""
    if not PIL_AVAILABLE or not champions: return None
    from PIL import ImageDraw
    chains = []
    for g, s, c in champions:
        ks, k, seen = [], book.key(g), set()
        while len(ks) < 6 and k in book.rec and k not in seen:
            seen.add(k); r = book.rec[k]
            if 'trimmed' in r: break
            ks.append((k, r))
            if not r.get('pk'): break
            k = r['pk'][0]
        chains.append((list(reversed(ks)), s, c))
    W = max(len(c[0]) for c in chains) * (cell + pad) + pad
    H = len(chains) * (cell + 18) + pad
    img = PILImage.new('RGB', (W, H), (18, 18, 28))
    dr = ImageDraw.Draw(img)
    for row, (chain, s, cert) in enumerate(chains):
        y = pad + row * (cell + 18); prev_c = None
        for i, (k, r) in enumerate(chain):
            x = pad + i * (cell + pad)
            th = PILImage.fromarray(_hsv_to_rgb_vec(_k_to_genome(book, k))).resize(
                (cell - 8, cell - 8), PILImage.NEAREST)
            img.paste(th, (x + 4, y + 4))
            dr.rectangle([x, y, x + cell - 1, y + cell - 1],
                         outline=(166, 227, 161) if cert and i == len(chain) - 1
                         else (88, 91, 112), width=2 if i == len(chain) - 1 else 1)
            dr.text((x + 2, y + cell), f"{r.get('type','?')[:10]} g{r.get('born',0)}",
                    fill=(200, 200, 210))
            if prev_c is not None:
                dr.line([prev_c[0], prev_c[1], x, y + cell // 2],
                        fill=(137, 180, 250), width=2)
            prev_c = (x + cell, y + cell // 2)
        dr.text((W - 90, y + 2), f"score {s:.1f}", fill=(166, 227, 161))
    img.save(path); return path

# ═══════════════════════════════════════════════════════════════════════════
# 16. INTELLIGENCE ASSESSMENT — null-calibrated, behavioral, live
# ═══════════════════════════════════════════════════════════════════════════
def perf_gate(test, null, frac=0.2, floor=0.05):
    """Relative performance gate: clear the null by max(floor, frac·headroom).
    On a 90%-majority task, 91% accuracy FAILS — the point of this module."""
    return bool(test >= null + max(floor, frac * max(1.0 - null, 0.0)))

def _ece(probs, labels, bins=10):
    """Expected Calibration Error (Naeini et al. 2015)."""
    conf = probs.max(1); pred = probs.argmax(1)
    corr = (pred == labels).astype(float); e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)

def decode_net(genome, refine=1, mode='node', radius=2):
    return ImageFFN(radius=radius, coupling='local', mode=mode,
                    refine=refine).from_genome(snap_to_bands(genome, refine))

class IntelligenceProbe:
    """Measures intelligence RELATIVE to nulls (majority/mean + best random
    genome) and BEHAVIORALLY (sensitivity, robustness, calibration, diversity)
    plus KAN structure (sparsity, purity, symbol fidelity)."""
    WEIGHTS = {
        'classification': dict(perf=.30, gen=.15, sens=.10, rob=.10, cal=.10,
                               div=.05, sparse=.05, comp=.05, pure=.05, fid=.05),
        'regression':     dict(perf=.35, gen=.20, sens=.10, rob=.10,
                               sparse=.05, comp=.05, pure=.05, fid=.10),
    }
    def __init__(self, task, Xva, yva, Xte, yte, seed=0, n_probe=48):
        self.task, self.Xva, self.yva, self.Xte, self.yte = task, Xva, yva, Xte, yte
        self.rng = np.random.default_rng(seed); self.n_probe = n_probe
        self.probe_evals = 0
        if task == 'classification':
            lab = yva.argmax(1); u, c = np.unique(lab, return_counts=True)
            self.null_majority = float(c.max() / len(lab))
            self.label_H = _entropy(c / c.sum())
        else:
            mu = yva.mean(); ss = ((yva - yva.mean(0)) ** 2).sum()
            self.null_majority = 0.0 if ss <= 1e-12 else \
                1.0 - ((yva - mu) ** 2).sum() / ss
            self.label_H = None
        self.evo_null = None
    def set_evolution_null(self, genomes, refine=1, mode='node', radius=2):
        best = -np.inf
        for g in genomes:
            net = decode_net(g, refine, mode, radius)
            m = self._metric(net, self.Xva, self.yva); self.probe_evals += len(self.Xva)
            best = max(best, m)
        self.evo_null = float(best)
    def _metric(self, net, X, y):
        if self.task == 'classification':
            return acc01(_slice_C(net.forward(X), y.shape[1]), y)
        return r2(_slice_C(net.forward(X), 1), y)
    def behavior(self, net, X, y):
        Xp = X[:min(len(X), self.n_probe)]; yp = y[:len(Xp)]
        self.probe_evals += 2 * len(Xp)
        if self.task == 'classification':
            base = _slice_C(net.forward(Xp), y.shape[1]).argmax(1)
            def flips(eps):
                Xn = np.clip(Xp + self.rng.normal(0, eps, Xp.shape), -6, 6)
                return float(np.mean(
                    _slice_C(net.forward(Xn), y.shape[1]).argmax(1) != base))
            sens = flips(0.30); rob = 1.0 - flips(0.02)
            Z = _slice_C(net.forward(Xp), y.shape[1])
            pr = np.exp(log_softmax(Z))
            calib = 1.0 - min(_ece(pr, yp.argmax(1)), 1.0)
            dist = np.bincount(pr.argmax(1), minlength=y.shape[1]) / len(Xp)
            div = float(np.clip(_entropy(dist) / max(self.label_H, 1e-9), 0, 1))
        else:
            base = _slice_C(net.forward(Xp), 1)[:, 0]
            sy = float(y.std() + 1e-9)
            def resp(eps):
                Xn = np.clip(Xp + self.rng.normal(0, eps, Xp.shape), -6, 6)
                d = np.abs(_slice_C(net.forward(Xn), 1)[:, 0] - base)
                return float(np.mean(d > 0.10 * sy * (1 if eps > 0.1 else 0.1)))
            sens = resp(0.30); rob = 1.0 - resp(0.02)
            calib, div = None, None
        return {'sensitivity': float(np.clip(sens / 0.5, 0, 1)),
                'robustness': float(np.clip(rob, 0, 1)),
                'calibration': calib, 'diversity': div}
    def assess(self, genome, refine=1, mode='node', radius=2):
        net = decode_net(genome, refine, mode, radius)
        test = self._metric(net, self.Xte, self.yte); self.probe_evals += len(self.Xte)
        val = self._metric(net, self.Xva, self.yva); self.probe_evals += len(self.Xva)
        beh = self.behavior(net, self.Xva, self.yva)
        km = kan_metrics(genome, refine)
        fid = symbol_fidelity(genome, self.Xva, self.yva, refine, mode, radius)
        self.probe_evals += 2 * len(self.Xva)
        c = {
            'perf': float(np.clip(test, 0, 1)),
            'gen': float(np.clip(1 - max(0.0, val - test) / max(abs(val), 1e-9), 0, 1)),
            'sens': beh['sensitivity'], 'rob': beh['robustness'],
            'cal': beh['calibration'], 'div': beh['diversity'],
            'sparse': float(np.clip((0.7 - km['edge_L1']) / 0.7, 0, 1)),
            'comp': float(1 - (km['n_functions_used'] - 1) / 23),
            'pure': float(np.clip(km['col_purity'], 0, 1)),
            'fid': float(np.clip(fid, 0, 1)),
        }
        w = self.WEIGHTS[self.task]
        num = sum(v * wt for k, v in c.items()
                  for wt in [w[k]] if v is not None)
        den = sum(wt for k, wt in w.items() if c[k] is not None)
        score = 100.0 * num / max(den, 1e-9)
        gates = {
            'beats_null': perf_gate(test, self.null_majority),
            'evolution_gain': (self.evo_null is None) or (test >= self.evo_null + 0.01),
            'generalization': (val - test) <= 0.15,
            'not_dead': beh['sensitivity'] >= 0.10,
            'robust': beh['robustness'] >= 0.60,
            'sparse': km['edge_L1'] <= 0.70,
            'pure': km['col_purity'] >= 0.45,
            'faithful': fid >= 0.75,
        }
        if self.task == 'classification':
            gates['calibrated'] = beh['calibration'] >= 0.5
            gates['diverse'] = beh['diversity'] >= 0.5
        grade = ('intelligent' if score >= 80 and all(gates.values()) else
                 'promising' if score >= 65 else
                 'marginal' if score >= 50 else 'not intelligent')
        return {'score': round(score, 1), 'grade': grade,
                'pass': bool(all(gates.values())), 'gates': gates,
                'test': float(test), 'val': float(val),
                'null_majority': float(self.null_majority),
                'evo_null': self.evo_null, 'components': c,
                'kan': km, 'symbol_fidelity': float(fid), 'refine': refine}
    def certificate(self, genome, **kw): return self.assess(genome, **kw)

class IntelligenceMonitor:
    """Live 'knowing when': probe the champion every N generations, snapshot
    the genome the moment it certifies (per family too), optionally stop."""
    def __init__(self, probe, every=5, hold=2, stop_when_certified=False,
                 verbose=True, typer=None, ledger=None, book=None):
        self.p = probe; self.every = every; self.hold = hold
        self.stop_when_certified = stop_when_certified
        self.verbose = verbose
        self.typer = typer; self.ledger = ledger; self.book = book
        self.history = []; self._cert_run = 0
        self.best_certified = None; self.best_by_type = {}
    def __call__(self, st):
        g = st['gen']
        if g % self.every and g != 1: return
        a = self.p.assess(st['best_genome'], refine=st.get('refine', 1))
        st['intel'] = a
        self.history.append({'gen': g, 'score': a['score'],
                             'pass': a['pass'], 'grade': a['grade']})
        if self.verbose:
            print(f"  [intel] gen {g:3d} | score {a['score']:5.1f} "
                  f"({a['grade']:15s}) | {'PASS' if a['pass'] else '—'}"
                  f" | test {a['test']:.3f} vs null {a['null_majority']:.3f}")
        if self.typer is not None:
            t = self.typer.type_of(st['best_genome'])
            a['type'], a['type_conf'] = t['type'], t['conf']
            if self.ledger:
                ch = (st.get('lineage') or {}).get('chain')
                self.ledger.record(g, t['type'], a['score'], a['pass'],
                                   st['best_genome'], t['conf'], chain=ch)
                cur = self.best_by_type.get(t['type'])
                if cur is None or a['score'] > cur[2]:
                    self.best_by_type[t['type']] = (
                        np.asarray(st['best_genome']).copy(), g, a['score'])
        if self.book is not None:
            self.book.note_score(st['best_genome'], a['score'], a['pass'], g)
        if a['pass']:
            self._cert_run += 1
            if self.best_certified is None or a['score'] > self.best_certified[2]:
                self.best_certified = (np.asarray(st['best_genome']).copy(),
                                       g, a['score'])
        else:
            self._cert_run = 0
    def should_stop(self):
        return self.stop_when_certified and self._cert_run >= self.hold
    def summary(self):
        return {'history': self.history,
                'best_certified': ({'gen': self.best_certified[1],
                                    'score': self.best_certified[2],
                                    'genome': self.best_certified[0]}
                                   if self.best_certified else None),
                'probe_evals': self.p.probe_evals}

def certify_intelligent_image(genome, Xte, yte, Xva, yva, task='classification',
                              refine=1, mode='node', radius=2, thr=None):
    """Intelligence certificate: generalization · honesty · sparsity ·
    compactness · purity · symbol fidelity — each check falsifiable."""
    thr = {**dict(test=0.85, gap=0.15, edge_l1=0.60, sparsity=0.15,
                  purity=0.50, fidelity=0.80, max_funcs=12), **(thr or {})}
    km = kan_metrics(genome, refine)
    net = ImageFFN(mode=mode, refine=refine, radius=radius).from_genome(genome)
    metric = (lambda n, X, Y: acc01(_slice_C(n.forward(X), Y.shape[1]), Y)) \
        if task == 'classification' else \
        (lambda n, X, Y: r2(_slice_C(n.forward(X), 1), Y))
    test, val = metric(net, Xte, yte), metric(net, Xva, yva)
    fid = symbol_fidelity(genome, Xva, yva, refine, mode, radius)
    checks = {'generalization': test >= thr['test'],
              'val_gap': (val - test) <= thr['gap'],
              'sparsity': km['edge_L1'] <= thr['edge_l1']
                          and km['sparsity'] >= thr['sparsity'],
              'compactness': km['n_functions_used'] <= thr['max_funcs'],
              'purity': km['col_purity'] >= thr['purity'],
              'symbol_fidelity': fid >= thr['fidelity']}
    return {'pass': all(checks.values()), 'checks': checks,
            'metrics': {'test': test, 'val': val, 'fidelity': fid, **km,
                        'refine': refine, 'mode': mode}}

# ═══════════════════════════════════════════════════════════════════════════
# 17. CONFIG
# ═══════════════════════════════════════════════════════════════════════════
CONFIG_DEFAULTS = dict(
    task='classification', test_frac=0.2,
    img_radius=2, img_coupling='local', img_kernel='gauss', img_prune=0.0,
    img_mode='node', img_refine_max=1, refine_plateau_after=6,
    prune_anneal=True, prune_ladder=(0.0, .05, .10, .15, .20), prune_every=5,
    prune_l1_target=0.45,
    bias_mapping='centered', bias_spatial=False, bias_depth_scale=True,
    bias_act_aware=True,
    pop_size=32, n_children=16, genome_H=8, genome_W=16, tournament=3,
    sbx_eta=15, sbx_prob=0.9, pm_eta=20, pm_boost=3.0,
    mo_enabled=False, niching='off', speciate_enabled=True,
    sh_fractions=(0.15, 0.4, 1.0), sh_eta=2,
    wfc_enabled=True, wfc_hue_bands=N_ACTIVATIONS, wfc_sat_bands=4,
    wfc_val_bands=3, wfc_temperature=0.85, wfc_prop_strength=0.9,
    wfc_prop_sweeps=1, wfc_max_rounds=6, wfc_seed_frac=0.25,
    wfc_learn_elite_frac=0.25, wfc_learn_every=1, wfc_budget_ms=60.0,
    ca_enabled=True, ca_rule='majority', ca_rule_no=90, ca_steps=1, ca_prob=0.18,
    nes_enabled=True, nes_pairs=8, nes_lr=0.03, nes_sigma=0.08,
    nes_sigma_min=0.01, nes_rank=8, nes_elite_k=8,
    spsa_enabled=True, spsa_steps=6, spsa_a=0.05, spsa_c=0.06,
    surr_enabled=True, surr_window=512, surr_refit_every=5, surr_rho_gate=0.30,
    cache_size=4096,
    islands=2, migrate_every=5, migrate_n=2,
    colpool=True, col_graft_prob=0.15, col_keep=8,
    qd_enabled=True, qd_res=(8, 8, 8), novelty_k=5, novelty_stagnation=8,
    hy_diversity_low=0.06, hy_stagnation_boost=12, hy_immigrants=4,
    ipop_stag=25, ipop_hist=30, ipop_max_mult=4,
    spec_enabled=True, spec_n_pc=8, spec_refit_every=5, spec_probe_frac=0.25,
    spec_probe_elites=6, eda_enabled=True, eda_pool_frac=0.25, eda_r_min=2,
    eda_r_max=32, base_wfc_temp=0.85, pr_dead_eps=1e-6,
    wfc_conditional=True, ca_policy=True, geo_prior='local', geo_lam=3.0,
    lineage=True, lineage_max=8192,
    max_generations=30, max_seconds=None, auto_pca=0, profile='balanced',
    intel_every=5, intel_hold=2,
)

PROFILES = {
    'fast': dict(nes_enabled=False, spsa_enabled=False, surr_enabled=False,
                 colpool=False, spec_enabled=False, speciate_enabled=False,
                 islands=1, qd_enabled=False, mo_enabled=False, niching='off',
                 wfc_seed_frac=0.15),
    'balanced': dict(),
    'research': dict(islands=2, surr_refit_every=3, spec_refit_every=3,
                     img_refine_max=3, prune_anneal=True),
}

def auto_config(bundle, profile='balanced', seconds=None, gens=None, overrides=None):
    """Sizes the genome FROM the data. Fixes the silent feature-truncation
    bug: H ≥ F always; F > 64 → automatic train-only PCA front-end."""
    F = bundle.X.shape[1]; C = bundle.y.shape[1]
    cfg = {**CONFIG_DEFAULTS}
    cfg['genome_H'] = int(min(max(F, 8), 48))
    cfg['genome_W'] = int(min(26, max(C + 3, 8 + F // 4)))
    cfg['task'] = bundle.task
    cfg['pop_size'], cfg['n_children'] = (16, 8) if profile == 'fast' else (24, 12)
    cfg['max_generations'] = gens if gens else (30 if profile == 'fast' else 40)
    cfg['auto_pca'] = 32 if F > 64 else 0
    cfg['profile'] = profile
    cfg.update(PROFILES.get(profile, {}))
    if seconds: cfg['max_seconds'] = seconds
    cfg.update(overrides or {})
    return cfg

# ═══════════════════════════════════════════════════════════════════════════
# 18. NET POOL — built-decoder LRU
# ═══════════════════════════════════════════════════════════════════════════
class NetPool:
    def __init__(self, net_kw, size=256):
        self.kw = dict(net_kw); self.size = size; self.d = {}; self.q = deque()
    def get(self, g):
        k = np.ascontiguousarray(g).tobytes()
        n = self.d.get(k)
        if n is not None: return n
        n = ImageFFN(**self.kw).from_genome(g)
        if len(self.d) >= self.size: self.d.pop(self.q.popleft(), None)
        self.d[k] = n; self.q.append(k); return n
    def invalidate(self, **kw):
        self.kw.update(kw); self.d.clear(); self.q.clear()

# ═══════════════════════════════════════════════════════════════════════════
# 19. CONTROLLER — orchestrates GA ↔ WFC ↔ CA ↔ NES ↔ SPSA ↔ QD ↔ lineage
# ═══════════════════════════════════════════════════════════════════════════
class NoBackpropController:
    """One forward-pass budget, ten stages per generation. All network contact
    is value-only loss evaluation through the auditor-wrapped oracle."""
    def __init__(self, X, y, cfg=None, seed=0, auditor=None):
        self.cfg = {**CONFIG_DEFAULTS, **(cfg or {})}
        self.rng = np.random.default_rng(seed)
        self.auditor = auditor or GradientFreedomAuditor()
        self.ledger = BudgetLedger()
        self.net = ImageFFN(self.cfg['img_radius'], self.cfg['img_coupling'],
                            self.cfg['img_kernel'], self.cfg['img_prune'],
                            BiasConfig(self.cfg['bias_mapping'], self.cfg['bias_spatial'],
                                       self.cfg['bias_depth_scale'],
                                       self.cfg['bias_act_aware']),
                            mode=self.cfg['img_mode'], refine=1)
        self.nets = NetPool(dict(radius=self.cfg['img_radius'],
                                 coupling=self.cfg['img_coupling'],
                                 kernel=self.cfg['img_kernel'],
                                 prune=self.cfg['img_prune'],
                                 bias_cfg=self.net.bias_cfg,
                                 mode=self.cfg['img_mode'], refine=1))
        self.task = self.cfg['task']
        self.loss_fn = make_task_loss(self.task)
        self._loss_ps = make_task_loss(self.task, per_sample=True)
        self.X, self.y = X, y
        self.refine = 1; self.prune = 0.0; self.caches = {}
        # THE fix: wrap the oracle through the auditor BEFORE anything captures
        # it — every internal evaluation is counted and audited.
        self._loss_direct = self.auditor.wrap_loss(self._loss_direct)
        self._loss_ps_direct = self.auditor.wrap_loss(self._loss_ps_direct)
        self.engine = WFCCAEngine(self.cfg, seed=seed)
        self.islands = IslandHost(self.engine, self.cfg['islands'],
                                  np.random.default_rng(seed + 1))
        self.cache = FitnessCache(self.cfg['cache_size'])
        self.caches = {(1, 0.0): self.cache}
        self.fidelity = FidelityEvaluator(self._loss_direct, X, y,
                                          list(self.cfg['sh_fractions']), seed)
        self.sh = CohortSuccessiveHalving(self.fidelity, self.cache, self.ledger,
                                          self.cfg['sh_eta'])
        self.surr = RidgeSurrogate(self.engine.vocab, self.cfg['surr_window'],
                                   self.cfg['surr_refit_every'],
                                   self.cfg['surr_rho_gate'], seed=seed + 2)
        self.archive = BehaviorArchive(self.cfg['qd_res'], np.random.default_rng(seed + 3))
        self.cpool = ColumnPool(self.cfg['col_keep'])
        self.restart = RestartManager(self.cfg['ipop_hist'],
                                      stag_gate=self.cfg['ipop_stag'],
                                      max_mult=self.cfg['ipop_max_mult'])
        self.driver = SpectralDriver(self.engine, self.islands, self.archive,
                                     self.cfg, np.random.default_rng(seed + 4))
        self.driver.fit_data(X, y)
        self.nes = None
        self.cx = AdaptiveOperatorSelector(['classic', 'wfc', 'pca', 'colpool'],
                                           seed=seed + 5)
        self.lineage = None
        self.pop = None; self.fits = None
        self.best_f = np.inf; self.stag = 0; self.gen = 0; self.throttle = 1.0
        self._anchor = None; self._elite_sig = None
        self.mo = self.cfg.get('mo_enabled', False)
        self._rank = None; self._crowd = None; self.n_species = 1
        self.sbx_eta = self.cfg['sbx_eta']; self.pm_eta = self.cfg['pm_eta']
        self.tournament = self.cfg['tournament']
        self.hist = {'best': [], 'mean': [], 'acc': [], 'regime': []}

    # ── oracle (the ONE door to the network) ──
    def _loss_direct(self, g, X, y):
        return self.loss_fn(self.nets.get(g).forward(X), y)
    def _loss_ps_direct(self, g, Xs, ys):
        return self._loss_ps(self.nets.get(g).forward(Xs), ys)
    def _full(self, g):
        f = self.cache.lookup(g)
        if f is not None:
            self.ledger.cache_hits += 1; return f
        f = self._loss_direct(g, self.X, self.y)
        self.ledger.full += 1; self.cache.insert(g, f)
        return f
    def predict(self, g, X):
        pred = self.nets.get(g).forward(X)
        if self.task == 'classification':
            return _slice_C(pred, self.y.shape[1]).argmax(1)
        return _slice_C(pred, self.y.shape[1])
    def _accuracy(self, g, X=None, y=None):
        if self.task != 'classification': return float('nan')
        X = self.X if X is None else X; y = self.y if y is None else y
        return float(np.mean(self.predict(g, X) == y.argmax(1)))

    # ── lifecycle ──
    def start(self):
        H, W = self.cfg['genome_H'], self.cfg['genome_W']
        N = self.cfg['pop_size']
        pop = []
        styles = ('slabs', 'plaids', 'blobs')
        for i in range(N):
            if i % 2 == 0: pop.append(generate_genome(self.rng, H, W, styles[i % 3]))
            else: pop.append(self.rng.random((H, W, 3)))
        pop = np.asarray(pop)
        nf = max(1, int(N * self.cfg['wfc_seed_frac']))
        if self.cfg['wfc_enabled'] and nf:
            self.engine.learn(pop[:8])
            pop[:nf] = self.engine.make_genomes(nf, (H, W))
        fits = np.array([self._full(g) for g in pop])
        self.pop, self.fits = pop, fits
        self.driver.attach_shape((H, W))
        if self.cfg.get('lineage'):
            self.lineage = LineageBook(ImageTyper(self.refine), self.cfg['lineage_max'])
            for i, g in enumerate(self.pop):
                self.lineage.birth(g, 0, 'founder',
                                   style=('wfc-seed' if i < nf else
                                          'procedural' if i % 2 else 'random'))
        self._absorb_all(); self._learn_islands()
        if self.cfg['nes_enabled']:
            self.nes = LowRankNES((H, W, 3), self.cfg['nes_rank'], self.cfg['nes_lr'],
                                  self.cfg['nes_sigma'], self.cfg['nes_sigma_min'],
                                  np.random.default_rng(self.rng.integers(1 << 31)))
        self.best_f = float(fits.min())

    def _absorb_all(self):
        for g, f in zip(self.pop, self.fits):
            self.archive.add(g, -f)
            if self.cfg['colpool']: self.cpool.add(g, -f)
            if self.cfg['surr_enabled']: self.surr.observe(g, f)

    def _learn_islands(self):
        n = self.islands.n
        for i in range(n):
            idx = np.arange(i, len(self.pop), n)
            self.islands.set_island(i)
            ke = max(2, int(len(idx) * self.cfg['wfc_learn_elite_frac']))
            self.islands.learn(self.pop[idx[np.argsort(self.fits[idx])[:ke]]])

    def _rekey_cache(self):
        key = (self.refine, self.prune)
        if key not in self.caches:
            self.caches[key] = FitnessCache(self.cfg['cache_size'])
        self.cache = self.caches[key]

    def _set_refine(self, r):
        """Refinement changes decode semantics ⇒ every cached loss under the
        old decode is stale. Per-refine caches + population re-eval."""
        r = int(r); self.refine = r; self.net.refine = r
        self.engine.set_refine(r)
        c = self.cfg
        for isl in range(self.islands.n):
            self.islands.set_island(isl)
            for attr in ('vocabs', 'diverse'):
                getattr(self.islands, attr)[isl] = WFCTileVocab(
                    hue_classes(r), c['wfc_sat_bands'], c['wfc_val_bands'])
        self._learn_islands()
        self.islands.learn(list(self.pop[np.argsort(self.fits)[:8]]))
        self._rekey_cache()
        self.nets.invalidate(prune=self.prune, refine=self.refine)
        if self.lineage: self.lineage.typer.refine = r
        self.fits = np.array([self._full(g) for g in self.pop])
        self.ledger.aux += len(self.pop)

    def _set_prune(self, theta):
        self.prune = float(theta); self.net.prune = theta
        self._rekey_cache()
        self.nets.invalidate(prune=self.prune, refine=self.refine)
        self.fits = np.array([self._full(g) for g in self.pop])
        self.ledger.aux += len(self.pop)

    def before_generation(self, gen):
        self.gen = gen
        b = float(self.fits.min())
        if b < self.best_f - 1e-12: self.best_f, self.stag = b, 0
        else: self.stag += 1
        self.restart.observe(-b)
        if gen % max(1, self.cfg['wfc_learn_every']) == 0:
            ke = max(2, int(len(self.pop) * self.cfg['wfc_learn_elite_frac']))
            elites = self.pop[np.argsort(self.fits)[:ke]]
            sig = elites.tobytes()
            if sig != self._elite_sig:
                n = self.islands.n
                for i in range(n):
                    idx = np.arange(i, len(self.pop), n)
                    ii = idx[np.argsort(self.fits[idx])[:max(2, ke // n + 1)]]
                    self.islands.set_island(i)
                    self.islands.learn(self.pop[ii])
                self._elite_sig = sig
        if self.cfg['surr_enabled'] and gen % max(1, self.cfg['surr_refit_every']) == 0:
            self.surr.refit()
        self.driver.update(gen, self.pop, self.fits, self.net,
                           self._loss_ps_direct, self.ledger)
        r = self.driver.signal.get('refine_to')
        if r and r != self.refine: self._set_refine(r)          # refine first…
        p = self.driver.signal.get('prune_to')
        if p is not None and p != self.prune: self._set_prune(p)  # …then prune
        if self.nes is not None:
            self.nes.sigma = float(np.clip(
                self.nes.sigma * self.driver.signal.get('nes_sigma_mult', 1.0),
                self.nes.sigma_min, 0.5))
        # regime-scheduled selection pressure (takeover-time control,
        # Goldberg & Deb 1991): explore ⇒ softer pressure
        ex = self.driver.signal.get('explore', 0.5)
        self.sbx_eta = float(np.clip(self.cfg['sbx_eta'] * (1.6 - 1.2 * ex), 4.0, 40.0))
        self.pm_eta = float(np.clip(self.cfg['pm_eta'] * (1.6 - 1.2 * ex), 4.0, 60.0))
        self.tournament = int(np.clip(2 + round(2 * (1 - ex)), 2, 6))
        if self.cfg.get('speciate_enabled'):
            self._species, _thr = speciate(self.pop)
            self.n_species = len(self._species)
            cents = [self.pop[s[int(np.argmin(self.fits[s]))]] for s in self._species]
            self.engine.learn(cents)     # species centroids teach the grammar
        k = min(self.cfg['nes_elite_k'], len(self.pop))
        eidx = np.argsort(self.fits)[:k]
        r_ = np.argsort(np.argsort(-self.fits[eidx])).astype(float)
        u = (r_ + 0.5) / len(eidx); u /= u.sum()
        anchor = np.clip(np.tensordot(u, self.pop[eidx], axes=1), 0, 1)
        if self.nes is not None and self._anchor is not None:
            self.nes.recenter(self._anchor, anchor)
        self._anchor = anchor

    def nes_step(self):
        if self.nes is None or self._anchor is None: return None
        plus, minus = self.nes.ask(self._anchor, self.cfg['nes_pairs'])
        fp = np.array([self._full(g) for g in plus])
        fm = np.array([self._full(g) for g in minus])
        self.nes.tell(fp, fm)
        center = self.nes.apply(self._anchor)
        for g, f in zip(np.vstack([plus, minus]), np.concatenate([fp, fm])):
            self.archive.add(g, -f)
            if self.cfg['surr_enabled']: self.surr.observe(g, f)
        if self.lineage: self.lineage.birth(center, self.gen, 'nes',
                                            parents=(self._anchor,), op='nes')
        return center

    # ── selection & variation ──
    def _selection_scores(self):
        self._rank = None
        if self.mo:
            F = np.column_stack([self.fits,
                np.asarray([float(np.mean(np.asarray(g)[..., 1])) for g in self.pop])])
            self._rank, self._crowd, _ = nsga2_sort(F)
            return None
        if (self.stag >= self.cfg['novelty_stagnation'] and self.cfg['qd_enabled']
                and len(self.archive.cells) > 3):
            return np.array([-self.archive.novelty(g, self.cfg['novelty_k'], self.pop)
                             for g in self.pop])
        sc = self.fits.copy()
        if self.cfg.get('niching') == 'sharing':
            sc = shared_scores(self.pop, sc)
        return sc

    def _tournament(self, scores, island):
        idx = np.arange(island, len(self.pop), self.islands.n)
        cand = self.rng.choice(idx, min(self.tournament, len(idx)), replace=False)
        if self._rank is not None:
            return int(min(cand, key=lambda i: (self._rank[i], -self._crowd[i])))
        return int(cand[np.argmin(scores[cand])])

    def _classic_cx(self, a, b):
        return sbx(a, b, self.sbx_eta, self.rng, prob=self.cfg['sbx_prob'])
    def _classic_mut(self, g):
        g = np.asarray(g, float)
        return poly_mut(g, self.pm_eta, self.rng,
                        self.cfg['pm_boost'] / max(1, g.shape[0] * g.shape[1]))

    def make_children(self, stop_flag=None):
        K = self.cfg['n_children']; n = self.islands.n
        scores = self._selection_scores()
        sig = self.driver.signal
        temp = sig.get('wfc_temp', self.cfg['wfc_temperature'])
        kids, ops, parent_fits = [], [], []
        for k in range(K):
            if stop_flag is not None and stop_flag(): break
            isl = k % n
            pa = self._tournament(scores, isl); pb = self._tournament(scores, isl)
            parent_fits.append(self.fits[pa])
            op = self.cx.select(); ch = None
            if op == 1 and self.cfg['wfc_enabled'] and self.islands.vocabs[isl].learned:
                ch = self.islands.crossover(self.pop[pa], self.pop[pb], isl,
                                            temp=temp * self.throttle,
                                            mode=sig.get('grammar_mode', 'elite'))
            elif op == 2 and self.cfg['eda_enabled'] and self.driver.eda is not None \
                    and self.driver.eda.mean is not None:
                ch = self.driver.eda.around([self.pop[pa], self.pop[pb]],
                                            r=sig.get('eda_r'),
                                            sigma=sig.get('eda_sigma', 0.6))
            elif op == 3 and self.cfg['colpool'] and len(self.cpool.cols):
                ch = self._classic_cx(self.pop[pa], self.pop[pb])
                ch = self.cpool.graft(ch, self.rng, 0.5)
            if ch is None:
                op = 0
                ch = self._classic_cx(self.pop[pa], self.pop[pb])
            ch = self._classic_mut(ch)
            if self.cfg['ca_enabled'] and self.rng.random() < 0.6:
                ch = self.engine.mutate(ch, self.rng)
            if self.cfg['colpool'] and self.rng.random() < self.cfg['col_graft_prob']:
                ch = self.cpool.graft(ch, self.rng)
            kids.append(np.clip(ch, 0, 1)); ops.append(op)
            if self.lineage:
                self.lineage.birth(kids[-1], self.gen, 'child',
                                   parents=(self.pop[pa], self.pop[pb]),
                                   op=self.cx.names[op],
                                   ca=self.driver.signal.get('ca_rule'))
        self._kid_ops, self._kid_parent_fits = ops, parent_fits
        return kids, ops, parent_fits

    def eval_children(self, kids):
        return self.sh.run(kids)

    def _aos_update(self, kids, kf):
        diffs = np.abs(self.fits[np.isfinite(self.fits)])
        scale = float(np.median(diffs)) + 1e-6
        for op, pf, f in zip(self._kid_ops, self._kid_parent_fits, kf):
            if np.isfinite(f):
                self.cx.update(op, float(np.tanh((pf - f) / scale)))

    def merge(self, kids, kf):
        mask = np.isfinite(kf)
        if not mask.any(): return
        sg = np.asarray([kids[i] for i in np.flatnonzero(mask)]); sf = kf[mask]
        if self.mo:
            Fc = np.column_stack([
                np.concatenate([self.fits, sf]),
                np.asarray([float(np.mean(np.asarray(g)[..., 1])) for g in self.pop]
                           + [float(np.mean(np.asarray(k)[..., 1])) for k in sg])])
            order, _, _ = nsga2_select(Fc, len(self.pop))
            self.pop = np.concatenate([self.pop, sg])[order]
            self.fits = np.concatenate([self.fits, sf])[order]
        else:
            cg = np.vstack([self.pop, sg]); cf = np.concatenate([self.fits, sf])
            idx = np.argsort(cf)[:len(self.pop)]
            self.pop, self.fits = cg[idx], cf[idx]
        merged = set(map(self.lineage.key, sg)) if self.lineage else set()
        if self.lineage:
            for g, f in zip(self.pop, self.fits):
                if self.lineage.key(g) in merged: self.lineage.note_survived(g, f)
        for g, f in zip(sg, sf):
            self.archive.add(g, -f)
            if self.cfg['colpool']: self.cpool.add(g, -f)
            if self.cfg['surr_enabled']: self.surr.observe(g, f)

    def diversify(self):
        dom = None
        if self.lineage: dom, _ = self.lineage.founder_dominance(self.pop)
        idx32 = self.rng.choice(len(self.pop), min(32, len(self.pop)), replace=False)
        sub = self.pop[idx32]
        mad = float(np.mean(np.abs(sub - sub.mean(0))))
        trig = (self.stag >= self.cfg['hy_stagnation_boost']
                or mad < self.cfg['hy_diversity_low']
                or (dom is not None and dom >= 0.85 and self.gen > 10))
        if not trig or not self.cfg['wfc_enabled']: return 0
        sig = self.driver.signal
        T = sig.get('wfc_temp', self.cfg['wfc_temperature']) \
            * (1 + 0.5 * min(self.stag, 50) / 50) * self.throttle
        mode = sig.get('grammar_mode', 'elite')
        n, N = self.islands.n, len(self.pop)
        k = min(self.cfg['hy_immigrants'] * self.restart.mult, max(1, N // 4))
        for i in range(n):
            ii = np.arange(i, N, n)
            worst = ii[np.argsort(self.fits[ii])[-k:]]
            self.islands.set_island(i)
            pool = None
            if self.surr.ok:
                overs = list(self.islands.make_genomes(
                    k * 4, self.pop.shape[1:3], temp=T, mode=mode))
                pool, saved = self.surr.screen(overs, k)
                self.ledger.screened += saved
            for j, w in enumerate(worst):
                r = self.rng.random()
                opn = 'imm:wfc'
                if pool is not None and r < 0.5:
                    self.pop[w] = pool[j % len(pool)]; opn = 'imm:surr'
                elif r < 0.75:
                    g = self.archive.sample(); opn = 'imm:archive'
                    if g is not None: self.pop[w] = g.copy()
                else:
                    self.pop[w] = self.islands.make_genomes(
                        1, self.pop.shape[1:3], temp=T, mode=mode)[0]
                self.fits[w] = self._full(self.pop[w])
                if self.lineage:
                    self.lineage.birth(self.pop[w], self.gen, 'immigrant', op=opn)
                self.ledger.immigrants += 1
        return k * n

    def polish(self):
        ci = int(np.argmin(self.fits))
        g, f = self.pop[ci], float(self.fits[ci])
        if self.cfg['spsa_enabled']:
            sp = SPSAPolish(lambda gg: self._full(gg), self.rng,
                            self.cfg['spsa_a'], self.cfg['spsa_c'])
            g2, f2 = sp.improve(g, f, self.cfg['spsa_steps'])
            if f2 < f:
                if self.lineage:
                    self.lineage.birth(g2, self.gen, 'polish', parents=(g,), op='spsa')
                self.pop[ci], self.fits[ci] = g2, f2
                self.best_f = min(self.best_f, f2)
                return ('spsa', f - f2)
            return ('spsa', 0.0)
        cands = [self.engine.ca.develop(g, self.rng)
                 for _ in range(max(1, self.cfg.get('polish_variants', 4)))]
        if self.surr.ok:
            cands, saved = self.surr.screen(cands, 2)
            self.ledger.screened += saved
        bg, bf = g, f
        for c in cands:
            fc = self._full(c)
            if fc < bf: bg, bf = c, fc
        if bf < f:
            if self.lineage:
                self.lineage.birth(bg, self.gen, 'polish', parents=(g,),
                                   op='ca-variants')
            self.pop[ci], self.fits[ci] = bg, bf
            self.best_f = min(self.best_f, bf)
            return ('ca', f - bf)
        return ('ca', 0.0)

    def after_generation(self):
        n = self.islands.n
        if n > 1 and self.gen % max(1, self.cfg['migrate_every']) == 0:
            for i in range(n):
                j = (i + 1) % n
                ii = np.arange(i, len(self.pop), n); jj = np.arange(j, len(self.pop), n)
                for _ in range(self.cfg['migrate_n']):
                    a = ii[np.argmin(self.fits[ii])]; b = jj[np.argmax(self.fits[jj])]
                    if self.fits[a] < self.fits[b]:
                        self.pop[b] = self.pop[a].copy(); self.fits[b] = self.fits[a]
        if self.restart.should_restart(self.stag):
            self._restart(self.restart.trigger())
            if self.nes is not None:
                self.nes = LowRankNES(self.pop.shape[1:], self.cfg['nes_rank'],
                                      self.cfg['nes_lr'], self.cfg['nes_sigma'],
                                      self.cfg['nes_sigma_min'],
                                      np.random.default_rng(self.rng.integers(1 << 31)))
            self.stag = 0
        ms = self.engine.ms_spent; self.engine.ms_spent = 0.0
        if not np.isinf(self.cfg['wfc_budget_ms']):
            self.throttle = float(np.clip(
                self.throttle * (0.7 if ms > self.cfg['wfc_budget_ms'] else 1.1),
                0.2, 1.0))

    def _restart(self, mult):
        N = len(self.pop); keep = max(2, N // 8)
        order = np.argsort(self.fits); keepers = set(order[:keep].tolist())
        arch = [self.archive.sample() for _ in range(keep)]
        T = self.cfg['wfc_temperature'] * 1.5
        for i in range(N):
            if i in keepers: continue
            self.islands.set_island(i % self.islands.n)
            r = self.rng.random(); opn = 'restart:noise'
            if r < 0.4 and arch[i % keep] is not None:
                self.pop[i] = arch[i % keep].copy(); opn = 'restart:arch'
            elif r < 0.7 and self.cfg['wfc_enabled']:
                self.pop[i] = self.islands.make_genomes(
                    1, self.pop.shape[1:3], temp=T)[0]; opn = 'restart:wfc'
            else:
                self.pop[i] = 0.25 + 0.5 * self.rng.random(self.pop.shape[1:3])
            self.fits[i] = self._full(self.pop[i])
            if self.lineage: self.lineage.birth(self.pop[i], self.gen,
                                                'restart', op=opn)

    # ── one generation ──
    def run_generation(self, stop_flag=None):
        self.before_generation(self.gen)
        center = self.nes_step()
        if center is not None:
            self.pop[-1] = center; self.fits[-1] = self._full(center)
        kids, ops, pf = self.make_children(stop_flag)
        if kids:
            kf = self.eval_children(kids)
            self._aos_update(kids, kf)
            self.merge(kids, kf)
        self.diversify()
        polish_info = self.polish()
        self.after_generation()
        if self.lineage is not None: self.lineage.snapshot(self.gen, self.pop)
        return self.stats(polish_info)

    def lineage_text(self):
        if self.lineage is None: return "lineage: off"
        bi = int(np.argmin(self.fits))
        dom, top = self.lineage.founder_dominance(self.pop)
        comp = self.lineage.composition[-1][1] if self.lineage.composition else {}
        fams = " · ".join(f"{k} {v}" for k, v in
                          sorted(comp.items(), key=lambda kv: -kv[1])[:5]) or "—"
        ops = " · ".join(f"{op} {b}/{s}" for op, b, s, p in
                         self.lineage.op_stats()) or "—"
        trs = " | ".join(self.lineage.top_transitions(3)) or "—"
        return (f"champion: {self.lineage.chain(self.pop[bi])}\n"
                f"roots: {dom*100:.0f}% from {top[:8]} | families: {fams}\n"
                f"ops (birth/surv): {ops}\ntransitions: {trs}")

    def stats(self, polish_info=None):
        bi = int(np.argmin(self.fits))
        acc = self._accuracy(self.pop[bi])
        self.hist['best'].append(float(self.fits[bi]))
        self.hist['mean'].append(float(self.fits.mean()))
        self.hist['acc'].append(acc)
        self.hist['regime'].append(self.driver.regime.regime)
        st = {'gen': self.gen, 'best_loss': float(self.fits[bi]),
              'mean_loss': float(self.fits.mean()), 'acc': acc,
              'best_genome': self.pop[bi].copy(),
              'regime': self.driver.regime.regime,
              'driver': self.driver.status_text(),
              'ledger': self.ledger.text(), 'polish': polish_info,
              'throttle': self.throttle, 'hist': dict(self.hist),
              'archive_grid': self.archive.heatmap()}
        if self.lineage:
            dom, top = self.lineage.founder_dominance(self.pop)
            st['lineage'] = {'chain': self.lineage.chain(self.pop[bi]),
                             'dom': dom, 'top_root': top,
                             'comp': list(self.lineage.composition),
                             'text': self.lineage_text()}
        return st

    def status_text(self):
        sig = f"{self.nes.sigma:.3f}" if self.nes else "—"
        return (f"gen {self.gen} | stag {self.stag} | best {self.best_f:.5f}"
                f" | σ {sig} | surr ρ {self.surr.rho:.2f}"
                f"{'✓' if self.surr.ok else '·'} | isl {self.islands.n}"
                f" | sp {self.n_species} | restarts {self.restart.restarts}"
                f" | thr {self.throttle:.2f} | {self.ledger.text()}")

    # ── checkpoints (pure arrays — no pickle) ──
    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        d = {'meta': np.array([self.gen, self.best_f, self.stag, self.throttle,
                               self.refine, self.prune]),
             'pop': self.pop, 'fits': self.fits,
             'engine_adj': self.engine.vocab.adj}
        if self.nes is not None:
            d['nes_mu'], d['nes_U'] = self.nes.mu, self.nes.U
            d['nes_sigma'] = np.array([self.nes.sigma])
            if self._anchor is not None: d['nes_anchor'] = self._anchor
        for i in range(self.islands.n):
            d[f'isl{i}'] = self.islands.vocabs[i].adj
            d[f'islD{i}'] = self.islands.diverse[i].adj
        if self.archive.cells:
            d['arch_g'] = np.stack([v[0] for v in self.archive.cells.values()])
            d['arch_r'] = np.array([v[1] for v in self.archive.cells.values()])
        np.savez_compressed(path, **d)
        return path

    def load_checkpoint(self, path):
        z = np.load(path); m = z['meta']
        self.gen, self.best_f, self.stag = int(m[0]), float(m[1]), int(m[2])
        self.throttle, self.refine, self.prune = float(m[3]), int(m[4]), float(m[5])
        self.engine.set_refine(max(1, self.refine))
        self.engine.vocab.adj = z['engine_adj']; self.engine.vocab.refresh()
        self.net.refine, self.net.prune = self.refine, self.prune
        self._rekey_cache()
        self.nets.invalidate(prune=self.prune, refine=self.refine)
        self.pop, self.fits = z['pop'], z['fits']
        if self.nes is not None and 'nes_mu' in z:
            self.nes.mu, self.nes.U = z['nes_mu'], z['nes_U']
            self.nes.sigma = float(z['nes_sigma'][0])
            self._anchor = z['nes_anchor'] if 'nes_anchor' in z else None
        for i in range(self.islands.n):
            self.islands.vocabs[i].adj = z[f'isl{i}']; self.islands.vocabs[i].refresh()
            self.islands.diverse[i].adj = z[f'islD{i}']; self.islands.diverse[i].refresh()
        if 'arch_g' in z:
            for g, r in zip(z['arch_g'], z['arch_r']): self.archive.add(g, r)

# ═══════════════════════════════════════════════════════════════════════════
# 20. DATA I/O
# ═══════════════════════════════════════════════════════════════════════════
class DataBundle:
    def __init__(self, X, y, task, feature_names=None, class_names=None,
                 mean=None, sd=None, pca=None, pca_n=None):
        self.X, self.y, self.task = X, y, task
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.class_names = class_names
        self.mean, self.sd = mean, sd
        self.pca, self.pca_n = pca, pca_n

def _clean_numeric(A, name="data"):
    A = np.asarray(A, float)
    keep = ~np.all(np.isnan(A), 0)
    if not keep.all():
        print(f"[data] dropped {int((~keep).sum())} all-NaN column(s) in {name}")
    A = A[:, keep]
    med = np.nanmedian(A, 0)
    if np.isnan(A).any(): print(f"[data] imputed NaNs with column medians in {name}")
    A = np.where(np.isnan(A), med[None, :], A)
    return A

def _onehot(v):
    v = np.asarray(v).ravel()
    u = np.unique(v)
    m = {k: i for i, k in enumerate(u)}
    return np.eye(len(u))[[m[x] for x in v]], [str(k) for k in u]

def _is_continuous(v):
    v = np.asarray(v, float); u = np.unique(v[~np.isnan(v)])
    return len(u) > 25 or not np.allclose(u, np.round(u))

def load_dataset(path, target=None, task='auto', image_size=8, standardize=True):
    """CSV/TSV · .npy/.npz · image folder (subdir-per-class) · sklearn names."""
    if not os.path.exists(path) and path in (
            'iris', 'wine', 'breast_cancer', 'digits', 'diabetes'):
        from sklearn.datasets import load_digits, load_iris, load_wine, \
            load_breast_cancer, load_diabetes
        d = {'digits': load_digits, 'iris': load_iris, 'wine': load_wine,
             'breast_cancer': load_breast_cancer, 'diabetes': load_diabetes}[path]()
        X = _clean_numeric(d.data, path)
        y_raw = np.asarray(d.target).ravel()
        names = list(getattr(d, 'target_names', map(str, np.unique(y_raw))))
        task = 'classification'
    elif os.path.isdir(path):
        if not PIL_AVAILABLE: raise RuntimeError("Pillow required for image folders")
        classes = sorted(d for d in os.listdir(path)
                         if os.path.isdir(os.path.join(path, d)))
        imgs, labs = [], []
        exts = ('.png', '.jpg', '.jpeg', '.bmp')
        if classes:
            for ci, cn in enumerate(classes):
                for f in sorted(os.listdir(os.path.join(path, cn))):
                    if f.lower().endswith(exts):
                        im = PILImage.open(os.path.join(path, cn, f)).convert('L') \
                            .resize((image_size, image_size))
                        imgs.append(np.asarray(im, float).ravel() / 255.0); labs.append(ci)
            names = classes
        else:
            files = sorted(f for f in os.listdir(path) if f.lower().endswith(exts))
            for f in files:
                im = PILImage.open(os.path.join(path, f)).convert('L') \
                    .resize((image_size, image_size))
                imgs.append(np.asarray(im, float).ravel() / 255.0); labs.append(0)
            names, labs = ['data'], np.zeros(len(imgs), int)
        X = np.asarray(imgs); y_raw = np.asarray(labs); task = 'classification'
    elif path.endswith(('.npy', '.npz')):
        z = np.load(path, allow_pickle=True) if path.endswith('.npz') \
            else {'X': np.load(path)}
        X = _clean_numeric(np.asarray(z['X'] if 'X' in z else list(z.values())[0],
                                      float), "X")
        y_raw = np.asarray(z['y'] if 'y' in z else list(z.values())[1]).ravel()
        names = None
    else:  # tabular CSV
        with open(path) as f:
            rows = [r for r in _csv.reader(f) if r]
        header, data = (rows[0], rows[1:]) if rows and not _all_num(rows[0]) \
            else ([f"f{i}" for i in range(len(rows[0]))], rows)
        tcol = len(header) - 1
        if isinstance(target, str) and target in header: tcol = header.index(target)
        elif isinstance(target, int): tcol = target % len(header)
        feats = [i for i in range(len(header)) if i != tcol]
        cols = [[r[i] for r in data] for i in range(len(header))]
        Xc = []
        for i in feats:
            try:
                Xc.append([float(x) for x in cols[i]])
            except ValueError:
                print(f"[data] factorizing categorical feature '{header[i]}'")
                u = sorted(set(cols[i])); mmp = {k: j for j, k in enumerate(u)}
                Xc.append([mmp[x] for x in cols[i]])
        X = np.asarray(Xc, float).T
        try:
            y_raw = np.asarray([float(x) for x in cols[tcol]])
        except ValueError:
            u = sorted(set(cols[tcol])); mmp = {k: j for j, k in enumerate(u)}
            y_raw = np.asarray([mmp[x] for x in cols[tcol]])
        names = [header[i] for i in feats]
        if task == 'auto':
            task = 'regression' if _is_continuous(y_raw) else 'classification'
    if standardize:
        mu = np.nanmedian(X, 0); sd = np.nanstd(X, 0) + 1e-9; X = (X - mu) / sd
    else: mu, sd = None, None
    if task == 'classification':
        y, class_names = _onehot(y_raw)
    else:
        y, class_names = np.asarray(y_raw, float).reshape(-1, 1), None
    return DataBundle(X, y, task, feature_names=names, class_names=class_names,
                      mean=mu, sd=sd)

def _all_num(row):
    try:
        [float(x) for x in row]; return True
    except ValueError:
        return False

def make_bundle(X, y, task='auto'):
    X = _clean_numeric(np.asarray(X, float), "X")
    mu = np.nanmedian(X, 0); sd = np.nanstd(X, 0) + 1e-9; X = (X - mu) / sd
    y = np.asarray(y)
    if task == 'auto':
        task = 'classification' if (y.ndim == 2 and y.shape[1] > 1)
        or (y.ndim == 1 and not _is_continuous(y)) else 'regression'
    if task == 'classification':
        if y.ndim == 1: y, cn = _onehot(y)
        else: y, cn = y, [str(i) for i in range(y.shape[1])]
    else:
        y, cn = np.asarray(y, float).reshape(-1, 1), None
    return DataBundle(X, y, task, mean=mu, sd=sd, class_names=cn)

# ═══════════════════════════════════════════════════════════════════════════
# 21. TRAIN — deadlines, checkpoints, resume, intelligence monitor
# ═══════════════════════════════════════════════════════════════════════════
def train(X, y, cfg=None, seed=0, generations=None, callback=None, stop_flag=None,
          target_loss=None, patience=None, max_seconds=None,
          checkpoint_every=0, checkpoint_dir='checkpoints', resume_from=None,
          intel_monitor=None):
    cfg = {**CONFIG_DEFAULTS, **(cfg or {})}
    auditor = GradientFreedomAuditor(); auditor.install()
    ctl = NoBackpropController(X, y, cfg, seed=seed, auditor=auditor)
    if resume_from:
        ctl.load_checkpoint(resume_from)
    else:
        ctl.start()
        auditor.purity_check(ctl._loss_direct, ctl.pop[0], X[:16], y[:16])
        if intel_monitor is not None and ctl.lineage is not None:
            intel_monitor.book = ctl.lineage
    t0 = time.time()
    deadline = t0 + max_seconds if max_seconds else None
    G = generations if generations is not None else cfg.get('max_generations', 30)
    ck = os.path.join(checkpoint_dir, f'ckpt_seed{seed}.npz')
    for gen in range(ctl.gen, G):
        if stop_flag is not None and stop_flag(): break
        if deadline is not None and time.time() > deadline: break
        st = ctl.run_generation(stop_flag=stop_flag)
        st['refine'] = ctl.refine
        if callback: callback(st)
        if intel_monitor:
            intel_monitor(st)
            if intel_monitor.should_stop(): break
        if checkpoint_every and (gen + 1) % checkpoint_every == 0:
            ctl.save_checkpoint(ck)
        if target_loss is not None and st['best_loss'] <= target_loss: break
        if patience is not None and ctl.stag >= patience: break
    final = ctl.stats()
    final.update(elapsed=time.time() - t0,
                 audit=auditor.report(ctl.ledger),
                 evals=auditor.forward_calls,
                 sample_evals=auditor.sample_evals,
                 deadline_hit=bool(deadline and time.time() > deadline),
                 generations_done=ctl.gen, controller=ctl)
    if intel_monitor: final['intelligence'] = intel_monitor.summary()
    return final

# ═══════════════════════════════════════════════════════════════════════════
# 22. MODEL — the deployable artifact (sklearn-shaped)
# ═══════════════════════════════════════════════════════════════════════════
def _metrics_for(model, X, y):
    if model.task == 'classification':
        yl = y.argmax(1); yp = model.predict(X)
        k = len(model.class_names)
        cm = np.zeros((k, k), int)
        for a, b in zip(yl, yp): cm[a, b] += 1
        per = [cm[i, i] / max(cm[i].sum(), 1) for i in range(k)]
        prec = [cm[i, i] / max(cm[:, i].sum(), 1) for i in range(k)]
        f1 = float(np.mean([2 * p * r_ / max(p + r_, 1e-12)
                            for p, r_ in zip(prec, per)])) if k > 2 else None
        return {'accuracy': float((yp == yl).mean()),
                'per_class': [float(p) for p in per],
                'confusion': cm.tolist(), 'macro_f1': f1}
    yp = np.asarray(model.predict(X), float).reshape(-1, 1)
    yv = np.asarray(y, float).reshape(-1, 1)
    return {'r2': r2(yp, yv),
            'mae': float(np.mean(np.abs(yp - yv)))}

class Model:
    """sklearn-shaped wrapper around an evolved image-network.
    fit → predict / predict_proba / score / recheck;
    save → model.png (human artifact) + model.npz (bit-exact) + model.json."""
    def __init__(self, task, class_names, feature_names, mean, sd, pca, pca_n,
                 cfg, genome, metrics, audit, meta):
        self.task, self.class_names = task, class_names
        self.feature_names, self.mean, self.sd = feature_names, mean, sd
        self.pca, self.pca_n = pca, pca_n
        self.cfg, self.genome, self.metrics = cfg, genome, metrics
        self.audit, self.meta = audit, meta
        self.net = ImageFFN(radius=cfg['img_radius'], coupling=cfg['img_coupling'],
                            kernel=cfg['img_kernel'], prune=cfg.get('img_prune', 0),
                            bias_cfg=BiasConfig(cfg['bias_mapping'], cfg['bias_spatial'],
                                                cfg['bias_depth_scale'],
                                                cfg['bias_act_aware']),
                            mode=cfg.get('img_mode', 'node'),
                            refine=cfg.get('img_refine', 1)).from_genome(genome)

    def _prep(self, X):
        X = (np.asarray(X, float) - self.mean) / self.sd
        if self.pca is not None:
            V, mu2 = self.pca
            X = (X - mu2) @ V.T
            X = X / (X.std(0) + 1e-9)
        return X

    def _out(self, X):
        C = len(self.class_names) if self.task == 'classification' else 1
        return _slice_C(self.net.forward(self._prep(X)), C)

    def predict(self, X):
        o = self._out(X)
        return o.argmax(1) if self.task == 'classification' else o[:, 0]

    def predict_proba(self, X):
        assert self.task == 'classification'
        return np.exp(log_softmax(self._out(X)))

    def score(self, X, y):
        if self.task == 'classification':
            yl = y.argmax(1) if y.ndim == 2 else np.asarray(y).ravel()
            return float((self.predict(X) == yl).mean())
        return r2(np.asarray(self.predict(X), float).reshape(-1, 1),
                  np.asarray(y, float).reshape(-1, 1))

    def recheck(self, X, y):
        """Re-verify the intelligence gates on NEW labeled data — offline
        successor of the removed drift-watch; same null-calibrated logic."""
        it = self.metrics.get('intelligence') or {}
        out = {'n': int(len(X))}
        if self.task == 'classification':
            yl = y.argmax(1) if y.ndim == 2 else np.asarray(y).ravel().astype(int)
            yp = self.predict(X)
            u, c = np.unique(yl, return_counts=True)
            null = float(c.max() / len(yl))
            acc = float((yp == yl).mean())
            out.update(accuracy=acc, null_majority=null,
                       beats_null=bool(perf_gate(acc, null)),
                       n_per_class={self.class_names[int(k_)] if self.class_names
                                    else str(k_): int(v_) for k_, v_ in zip(u, c)})
            k = len(self.class_names)
            cm = np.zeros((k, k), int)
            for a_, b_ in zip(yl, yp): cm[a_, b_] += 1
            rec = [cm[i, i] / max(cm[i].sum(), 1) for i in range(k)]
            tr = self.metrics.get('per_class') or rec
            out['recall_drift'] = [float(t - r_) for t, r_ in zip(tr, rec)]
        else:
            yv = np.asarray(y, float).ravel()
            yp = np.asarray(self.predict(X), float).ravel()
            ss = ((yv - yv.mean()) ** 2).sum()
            null = 0.0 if ss <= 1e-12 else 1.0 - ((yv - yv.mean()) ** 2).sum() / ss
            m = r2(yp[:, None], yv[:, None])
            out.update(r2=float(m), null_mean_r2=float(null),
                       beats_null=bool(m >= null + 0.05))
        out['certificate_still_holds'] = bool(it.get('pass', False)) and out['beats_null']
        return out

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        g = snap_to_bands(self.genome, self.cfg.get('img_refine', 1))
        save_model_png(g, os.path.join(directory, 'model.png'))
        np.savez_compressed(os.path.join(directory, 'model.npz'),
                            genome=g.astype(np.float32),
                            mean=self.mean, sd=self.sd,
                            pcaV=self.pca[0] if self.pca else np.zeros(0),
                            pcaMu=self.pca[1] if self.pca else np.zeros(0))
        meta = {k: (v if isinstance(v, (int, float, str, bool, list, type(None)))
                    else str(v)) for k, v in self.cfg.items()}
        _json.dump(dict(version=__version__, task=self.task,
                        class_names=self.class_names,
                        feature_names=self.feature_names, cfg=meta,
                        metrics=self.metrics,
                        audit_verdict=self.audit.strip().splitlines()[-1],
                        **self.meta),
                   open(os.path.join(directory, 'model.json'), 'w'), indent=2)
        return directory

    @classmethod
    def load(cls, directory):
        z = np.load(os.path.join(directory, 'model.npz'))
        meta = _json.load(open(os.path.join(directory, 'model.json')))
        cfg = meta['cfg']
        for k in ('img_radius', 'genome_H', 'genome_W', 'pop_size', 'img_refine'):
            if k in cfg: cfg[k] = int(cfg[k])
        pca = (z['pcaV'], z['pcaMu']) if z['pcaV'].size else None
        return cls(task=meta['task'], class_names=meta['class_names'],
                   feature_names=meta['feature_names'],
                   mean=z['mean'], sd=z['sd'], pca=pca,
                   pca_n=cfg.get('auto_pca', 0), cfg=cfg,
                   genome=z['genome'].astype(np.float64),
                   metrics=meta.get('metrics', {}),
                   audit=meta.get('audit_verdict', ''),
                   meta={k: v for k, v in meta.items()
                         if k in ('params', 'generations_done', 'elapsed',
                                  'profile', 'source', 'seed')})

    @classmethod
    def fit(cls, data, y=None, target=None, seconds=120, profile='balanced',
            seed=0, overrides=None, verbose=True, source=None, intel_stop=False):
        """One-call trainer: auto-config → GA/WFC/CA search → intelligence
        certification → artifact. data = path or (X[, y]) arrays."""
        bundle = load_dataset(data, target=target) if isinstance(data, str) \
            else make_bundle(data, y)
        s = resolve_seed(seed)
        Xtr, ytr, Xva, yva, Xte, yte = split3(bundle.X, bundle.y, seed=s)
        if bundle.X.shape[1] > 64:                     # leak-free PCA front-end
            sp = DataSpectra(Xtr, 32)
            mu2 = Xtr.mean(0); V = sp.V
            pr = lambda A: (A - mu2) @ V.T
            sd2 = pr(Xtr).std(0) + 1e-9
            pr2 = lambda A: pr(A) / sd2
            bundle = DataBundle(pr2(Xtr), ytr, bundle.task, bundle.feature_names,
                                bundle.class_names, bundle.mean, bundle.sd,
                                pca=(V, mu2), pca_n=32)
            Xte = pr2(Xte); Xva = pr2(Xva)
        cfg = auto_config(bundle, profile, seconds=seconds, overrides=overrides)
        cfg['img_refine'] = 1
        t0 = time.time()
        def cb(st):
            if verbose and (st['gen'] % 10 == 0 or st['gen'] == 1):
                a = f"{st['acc']*100:5.1f}%" if st['acc'] == st['acc'] else "  n/a"
                print(f"  gen {st['gen']:3d} | loss {st['best_loss']:.5f} | acc {a}"
                      f" | {st['regime']:8s}")
        intel = IntelligenceProbe(bundle.task, Xva, yva, Xte, yte, seed=s)
        rng0 = np.random.default_rng(s + 77)
        intel.set_evolution_null([rng0.random((cfg['genome_H'], cfg['genome_W'], 3))
                                  for _ in range(6)])
        typer = ImageTyper(refine=1); ledger = TypeLedger()
        mon = IntelligenceMonitor(intel, every=cfg.get('intel_every', 5),
                                  hold=cfg.get('intel_hold', 2),
                                  stop_when_certified=bool(intel_stop),
                                  verbose=verbose, typer=typer, ledger=ledger)
        res = train(Xtr, ytr, cfg=cfg, seed=s, max_seconds=seconds,
                    callback=cb, intel_monitor=mon)
        ctl = res['controller']
        cand_genome = (mon.best_certified[0] if mon.best_certified is not None
                       else res['best_genome'])
        intel_final = intel.assess(cand_genome, refine=ctl.refine,
                                   mode=cfg.get('img_mode', 'node'))
        genome = snap_to_bands(cand_genome, ctl.refine)
        cfg['img_refine'] = ctl.refine
        probe = Model(bundle.task, bundle.class_names, bundle.feature_names,
                      bundle.mean, bundle.sd, bundle.pca, bundle.pca_n,
                      cfg, genome, {}, res['audit'], {})
        metrics = _metrics_for(probe, Xte, yte)
        metrics.update(params=count_params(genome),
                       generations_done=res['generations_done'],
                       elapsed=round(time.time() - t0, 1),
                       sample_evals=res['sample_evals'],
                       deadline_hit=res['deadline_hit'], profile=profile,
                       intel_gen=(mon.best_certified or (None, None))[1],
                       intelligence={'score': intel_final['score'],
                                     'grade': intel_final['grade'],
                                     'pass': intel_final['pass'],
                                     'gates': intel_final['gates'],
                                     'null_majority': intel_final['null_majority'],
                                     'evo_null': intel_final['evo_null']})
        if ledger.groups: metrics['types'] = ledger.text()
        metrics['type_of_model'] = typer.type_of(genome)['type']
        if ctl.lineage is not None:
            metrics['champion_lineage'] = ctl.lineage.chain(genome)
            metrics['root_dominance'] = ctl.lineage.founder_dominance(ctl.pop)[0]
        m = cls(bundle.task, bundle.class_names, bundle.feature_names,
                bundle.mean, bundle.sd, bundle.pca, bundle.pca_n,
                cfg, genome, metrics, res['audit'],
                dict(profile=profile, seed=str(seed), source=str(source or data)))
        m._bundle = bundle; m._ledger = ledger; m._typer = typer
        m._lineage = ctl.lineage
        return m

    def type_gallery(self, directory):
        out = []
        if not getattr(self, '_ledger', None): return out
        os.makedirs(directory, exist_ok=True)
        idx = {}
        for name, G in self._ledger.groups.items():
            if G.get('best_g') is not None:
                p = contact_sheet([G['best_g']], os.path.join(
                    directory, f"{name.replace(' ', '_').lower()}.png"))
                if p: out.append((name, p))
                idx[name] = {'score': G['best'], 'gen': G['last_gen'],
                             'chain': G.get('best_chain')}
        if idx: _json.dump(idx, open(os.path.join(directory, 'index.json'), 'w'),
                           indent=2)
        return out

    def export_lineage(self, directory):
        lb = getattr(self, '_lineage', None)
        if lb is None: return []
        os.makedirs(directory, exist_ok=True)
        out = [export_graphml(lb, os.path.join(directory, 'lineage.graphml')),
               export_gexf(lb, os.path.join(directory, 'lineage.gexf')),
               export_timeline_csv(lb, os.path.join(directory,
                                                    'families_timeline.csv'))]
        champs = []
        it = self.metrics.get('intelligence') or {}
        champs.append((self.genome, it.get('score', 50.0), it.get('pass', False)))
        if getattr(self, '_ledger', None):
            for name, G in self._ledger.groups.items():
                if G.get('best_g') is not None and G['cert'] > 0:
                    champs.append((G['best_g'], G['best'], True))
        p = render_fan(lb, champs, os.path.join(directory, 'lineage_fan.png'))
        if p: out.append(p)
        return out

def write_report(model, path):
    m = model.metrics
    lines = [f"Image-FFN (no backprop) — {model.task}",
             f"source {model.meta.get('source')} | profile {model.meta.get('profile')}"
             f" | seed {model.meta.get('seed')}",
             f"image family: {m.get('type_of_model', '?')}",
             f"genome {model.genome.shape[0]}×{model.genome.shape[1]} | "
             f"live weights {m.get('params')}",
             f"trained {m.get('generations_done')} gens in {m.get('elapsed')}s "
             f"(deadline_hit={m.get('deadline_hit')}) using "
             f"{m.get('sample_evals', 0):,} sample presentations"]
    it = m.get('intelligence')
    if it:
        lines.append(f"INTELLIGENCE: {it['score']}/100 — {it['grade'].upper()} "
                     f"({'CERTIFIED' if it['pass'] else 'not certified'})"
                     f" | nulls: majority {it['null_majority']:.3f}, "
                     f"random-genome "
                     f"{'n/a' if it['evo_null'] is None else f'{it[chr(101)+chr(118)+chr(111)+chr(95)+chr(110)+chr(117)+chr(108)+chr(108)]:.3f}'}"
                     f" | certified at gen {m.get('intel_gen')}")
        bad = [k for k, v in it['gates'].items() if not v]
        lines.append("  gates failed: " + (", ".join(bad) if bad else "none"))
    t = m.get('types')
    if t: lines += ["image families:"] + ["  " + ln for ln in t.splitlines()[1:]]
    if model.task == 'classification':
        lines.append(f"TEST accuracy {m['accuracy']*100:.2f}%  macro-F1 "
                     f"{'—' if m.get('macro_f1') is None else f'{m[chr(109)+chr(97)+chr(99)+chr(114)+chr(111)+chr(95)+chr(102)+chr(49)]:.3f}'}")
        for n, p in zip(model.class_names, m['per_class']):
            lines.append(f"  class {n:<12s} recall {p*100:5.1f}%")
        lines.append("confusion (rows=true):")
        lines += ["  " + " ".join(f"{v:5d}" for v in row) for row in m['confusion']]
    else:
        lines.append(f"TEST R² {m['r2']:+.3f} | MAE {m['mae']:.4f}")
    lc = m.get('champion_lineage')
    if lc: lines.append("ancestry: " + lc)
    dom = m.get('root_dominance')
    if dom is not None:
        lines.append(f"root dominance: {dom*100:.0f}%"
                     + ("  ⚠ one ancestor dominates — diversity alarm"
                        if dom > 0.85 else ""))
    lines.append("audit: " + model.audit.strip().splitlines()[-1])
    open(path, 'w').write("\n".join(lines))
    return path

# ═══════════════════════════════════════════════════════════════════════════
# 23. DATASETS (synthetic, for demo/tests/examples)
# ═══════════════════════════════════════════════════════════════════════════
def make_blobs_task(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-1, .5, (n // 2, 2)), rng.normal(1, .5, (n // 2, 2))])
    y = np.eye(2)[np.hstack([np.zeros(n // 2, int), np.ones(n // 2, int)])]
    return X, y

def make_threshold_task(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 1))
    return x, np.eye(2)[(x[:, 0] > 0.5).astype(int)]

def make_noisy_xor(n=240, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2))
    y = ((X[:, 0] > .5) ^ (X[:, 1] > .5)).astype(int)
    return X, np.eye(2)[y]

def make_spiral(n=120, seed=0):
    X = np.zeros((2 * n, 2)); y = np.zeros(2 * n, int)
    for i in range(n):
        r, t = i / n * 5, 1.75 * i / n * 2 * np.pi
        X[i] = [r * np.cos(t), r * np.sin(t)]
        X[n + i] = [r * np.cos(t + np.pi), r * np.sin(t + np.pi)]; y[n + i] = 1
    return X, np.eye(2)[y]

def make_sine(n=240, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2 * np.pi, (n, 1))
    return x, np.sin(x) + rng.normal(0, 0.03, (n, 1))

def make_kof4(n=240, seed=0, k=2):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, (n, 4)).astype(float)
    return X, np.eye(2)[(X.sum(1) >= k).astype(int)]

def make_shapes(n=240, seed=0, grid=8, noise=0.15):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, grid * grid)); y = np.zeros(n, int); c0 = grid // 2
    for i in range(n):
        img = np.zeros((grid, grid)); c = i % 4
        if c == 0: img[c0, :] = 1
        elif c == 1: img[:, c0] = 1
        elif c == 2: img[np.arange(grid), np.arange(grid)] = 1
        else: img[np.arange(grid), np.arange(grid)[::-1]] = 1
        X[i] = np.clip(img + rng.normal(0, noise, img.shape), 0, 1).ravel()
    return X, np.eye(4)[y]

def make_composition_task(n=400, seed=0):
    """KA-form task: outer ⊕/⊗ over inner univariates (KAN's strong regime)."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 3))
    y = (np.sin(2 * np.pi * X[:, 0]) * X[:, 1] + X[:, 2] ** 2).reshape(-1, 1)
    return X, y

# ═══════════════════════════════════════════════════════════════════════════
# 24. TESTS
# ═══════════════════════════════════════════════════════════════════════════
class TestActivations(unittest.TestCase):
    def test_all_finite(self):
        z = np.linspace(-3, 3, 50)
        for f in ACTS: self.assertTrue(np.all(np.isfinite(f(z))))
    def test_map_24(self):
        self.assertEqual(len(ACTIVATION_MAP), 24)
        self.assertEqual(ACTIVATION_MAP[0][2], "ReLU")
        self.assertEqual(hue_to_activation(16)[1], "Sigmoid")
        self.assertEqual(hue_to_activation(200)[1], "HardSwish")
    def test_hue_band_roundtrip(self):
        b = np.array([0, 5, 23]); h = (b + 0.5) / N_ACTIVATIONS
        self.assertTrue(np.array_equal(hue_band(h), b))

class TestBiasEngine(unittest.TestCase):
    def test_ranges(self):
        V = np.random.default_rng(0).random((6, 8))
        for m in ('raw', 'centered', 'sigmoidal', 'gaussian'):
            self.assertTrue(np.all(np.isfinite(
                BiasEngine.compute(V, 0, BiasConfig(mapping=m)))))
    def test_depth_scale(self):
        V = np.full((4, 4), 0.9)
        b0 = BiasEngine.compute(V, 0, BiasConfig(depth_scale=True))
        b3 = BiasEngine.compute(V, 3, BiasConfig(depth_scale=True))
        self.assertLess(np.abs(b3).mean(), np.abs(b0).mean())

class TestImageFFN(unittest.TestCase):
    def _net(self, H=6, W=10, **kw):
        g = np.random.default_rng(1).random((H, W, 3))
        return ImageFFN(**kw).from_genome(g), g
    def test_forward_shape_and_states(self):
        net, _ = self._net()
        X = np.random.default_rng(2).random((7, 4))
        out, states = net.forward(X, return_states=True)
        self.assertEqual(out.shape, (7, 6))
        self.assertEqual(len(states), net.nL + 1)   # clamp + one per column
    def test_acyclic(self):
        """Perturbing column j leaves states 0..j identical — info flows only
        left→right ⇒ strictly feedforward."""
        g = np.random.default_rng(3).random((6, 10, 3))
        X = np.random.default_rng(4).random((5, 3))
        s1 = ImageFFN().from_genome(g).forward(X, return_states=True)[1]
        g2 = g.copy(); g2[:, 4, :] += 0.5; g2 = np.clip(g2, 0, 1)
        s2 = ImageFFN().from_genome(g2).forward(X, return_states=True)[1]
        for j in range(5):
            self.assertTrue(np.allclose(s1[j], s2[j], atol=1e-12),
                            f"state {j} affected by later column")
        self.assertFalse(np.allclose(s1[-1], s2[-1]))
    def test_determinism_and_couplings(self):
        net, g = self._net(coupling='global_sum')
        X = np.random.default_rng(5).random((4, 3))
        self.assertTrue(np.allclose(net.forward(X), net.forward(X)))
        self.assertEqual(ImageFFN(coupling='local').from_genome(g).forward(X).shape,
                         (4, 6))
    def test_ridge_head(self):
        rng = np.random.default_rng(6)
        X = rng.random((60, 3)); y = (X.sum(1, keepdims=True) > 1.5).astype(float)
        net, _ = self._net(H=6, W=12)
        net.ridge_head_fit(X, y)
        self.assertGreaterEqual(
            np.mean((net.ridge_head_predict(X) > 0.5) == (y > 0.5)), 0.5)
    def test_summary_runs(self):
        self._net()[0].summary()

class TestGeneration(unittest.TestCase):
    def test_styles(self):
        rng = np.random.default_rng(0)
        for s in ('slabs', 'plaids', 'blobs'):
            g = generate_genome(rng, 8, 16, s)
            self.assertEqual(g.shape, (8, 16, 3))
            self.assertTrue(g.min() >= 0 and g.max() <= 1)
    def test_determinism(self):
        a = generate_genome(np.random.default_rng(7), 6, 12, 'slabs')
        b = generate_genome(np.random.default_rng(7), 6, 12, 'slabs')
        self.assertTrue(np.array_equal(a, b))

class TestWFC(unittest.TestCase):
    def test_vocab_roundtrip(self):
        v = WFCTileVocab()
        t = np.random.default_rng(0).integers(0, v.T, (6, 10))
        self.assertTrue(np.array_equal(v.tile_of(v.decode(t)), t))
    def test_learn_and_collapse(self):
        v = WFCTileVocab()
        rng = np.random.default_rng(1)
        pop = np.stack([generate_genome(rng, 6, 12, 'slabs') for _ in range(6)])
        v.learn(pop)
        self.assertTrue(v.learned)
        gen = FastWFC(v, rng=np.random.default_rng(2))
        gen.prepare(6, 12)
        while not gen.run_round(): pass
        out = gen.finish()
        self.assertEqual(out.shape, (1, 6, 12, 3))
        self.assertTrue(np.all((out >= 0) & (out <= 1)))
        self.assertTrue(np.array_equal(hue_band(out[..., 0]),
                                       v.tile_of(out) // (v.sb * v.vb)))
    def test_seeded_crossover_and_prior(self):
        v = WFCTileVocab(); rng = np.random.default_rng(3)
        a = generate_genome(rng, 6, 10, 'slabs'); b = generate_genome(rng, 6, 10, 'blobs')
        v.learn(np.stack([a, b]))
        gen = FastWFC(v, rng=np.random.default_rng(4))
        gen.prior = np.ones((6, 10))
        ta = v.tile_of(a[None])[0]; tb = v.tile_of(b[None])[0]
        mix = np.where(rng.random(ta.shape) < 0.5, ta, tb)
        gen.prepare(6, 10, mix[None])
        while not gen.run_round(): pass
        self.assertEqual(gen.finish()[0].shape, (6, 10, 3))

class TestCA(unittest.TestCase):
    def test_rules_bounds_and_mask(self):
        g = np.random.default_rng(0).random((8, 12, 3))
        for rule in CellularMutator.RULES:
            self.assertTrue(np.allclose(
                CellularMutator(rule, prob=0.0).develop(g), np.clip(g, 0, 1)))
            out = CellularMutator(rule, prob=0.5).develop(g, np.random.default_rng(1))
            self.assertEqual(out.shape, g.shape)
            self.assertTrue(out.min() >= 0 and out.max() <= 1)
    def test_seed_determinism(self):
        g = np.random.default_rng(2).random((6, 8, 3))
        a = CellularMutator('reaction', prob=0.5).develop(g, np.random.default_rng(5))
        b = CellularMutator('reaction', prob=0.5).develop(g, np.random.default_rng(5))
        self.assertTrue(np.array_equal(a, b))

class TestEngine(unittest.TestCase):
    def test_seed_and_ops(self):
        eng = WFCCAEngine({'wfc_temperature': 0.9}, seed=0)
        pop = np.stack([generate_genome(np.random.default_rng(1), 6, 10, s)
                        for s in ('slabs', 'plaids', 'blobs', 'slabs')])
        eng.learn(pop)
        self.assertEqual(eng.crossover(pop[0], pop[1]).shape, pop[0].shape)
        m = eng.mutate(pop[2])
        self.assertTrue(m.min() >= 0 and m.max() <= 1)
    def test_grammar_persistence(self):
        eng = WFCCAEngine(seed=0)
        pop = np.stack([generate_genome(np.random.default_rng(2), 6, 10, 'slabs')
                        for _ in range(4)])
        eng.learn(pop); eng.save_grammar('_tmp_grammar.npz')
        eng2 = WFCCAEngine(seed=1); eng2.load_grammar('_tmp_grammar.npz')
        self.assertTrue(np.allclose(eng.vocab.adj, eng2.vocab.adj))
        os.remove('_tmp_grammar.npz')

class TestGAOperators(unittest.TestCase):
    def test_sbx_bounds_and_interior(self):
        a = np.full((4, 6, 3), 0.2); b = np.full_like(a, 0.8)
        c = sbx(a, b, eta=15, rng=np.random.default_rng(0), prob=1.0)
        self.assertTrue(c.min() >= 0 and c.max() <= 1)
        self.assertTrue(((c > 0.2) & (c < 0.8)).any())
    def test_sbx_eta_contracts(self):
        a = np.full((64, 4, 3), 0.3); b = np.full_like(a, 0.7)
        hi = sbx(a, b, eta=50, rng=np.random.default_rng(1), prob=1.0)
        lo = sbx(a, b, eta=1, rng=np.random.default_rng(2), prob=1.0)
        self.assertLess(np.abs(hi - 0.5).std(), np.abs(lo - 0.5).std())
    def test_poly_mut_bounds_and_rate(self):
        g = np.random.default_rng(3).random((6, 8, 3))
        self.assertTrue(np.allclose(poly_mut(g, 20, np.random.default_rng(4), pm=0.0), g))
        m = poly_mut(g, 20, np.random.default_rng(4), pm=1.0)
        self.assertFalse(np.allclose(m, g))
        self.assertTrue(m.min() >= 0 and m.max() <= 1)
    def test_nsga2_fronts(self):
        F = np.array([[0., 0.], [1., 1.], [.5, .5], [1., 0.], [0., 1.]])
        rank, crowd, _ = nsga2_sort(F)
        self.assertEqual(int(rank[0]), 0)
        self.assertEqual(int(rank[2]), 2)
        idx, _, _ = nsga2_select(F, 3)
        self.assertIn(0, idx.tolist())
    def test_sharing_penalizes_clones(self):
        pop = [np.zeros((4, 4, 3))] * 4 + [np.ones((4, 4, 3))]
        fits = np.array([0.1] * 4 + [0.5])
        sh = shared_scores(pop, fits, sigma=0.1)
        self.assertGreater(sh[0], 0.1)
        self.assertAlmostEqual(sh[4], 0.5)
    def test_speciate_two_clusters(self):
        rng = np.random.default_rng(0)
        pop = np.concatenate([rng.random((6, 4, 4, 3)) * 0.05 + 0.1,
                              rng.random((6, 4, 4, 3)) * 0.05 + 0.9])
        sp, _thr = speciate(pop)
        self.assertEqual(len(sp), 2)

class TestNoBackpropInfra(unittest.TestCase):
    def test_auditor_counts_and_purity(self):
        aud = GradientFreedomAuditor()
        L = aud.wrap_loss(lambda g, X, y: float(np.mean(g)))
        g = np.zeros((4, 4, 3))
        L(g, None, None); L(g, np.zeros((10, 2)), None)
        self.assertEqual(aud.forward_calls, 2)
        self.assertEqual(aud.sample_evals, 10)
        self.assertTrue(aud.purity_check(
            lambda g, X, y: float(np.mean(g)), g, None, None))
        self.assertEqual(len(aud.violations), 0)
    def test_cache(self):
        c = FitnessCache()
        g = np.random.default_rng(0).random((4, 4, 3))
        c.insert(g, 1.5)
        self.assertEqual(c.lookup(g), 1.5); self.assertEqual(c.hits, 1)
        self.assertIsNone(c.lookup(g + 0.1)); self.assertEqual(c.misses, 1)
    def test_cohort_sh_saves_evals(self):
        calls = {'n': 0}
        def ev(g, X, y):
            calls['n'] += 1
            return float(np.mean(g[..., 1]))
        fid = FidelityEvaluator(ev, np.zeros((32, 2)), np.zeros((32, 2)),
                                (0.25, 0.5, 1.0), 0)
        led = BudgetLedger(); sh = CohortSuccessiveHalving(fid, FitnessCache(), led)
        rng = np.random.default_rng(1)
        kids = [rng.random((6, 10, 3)) for _ in range(8)]
        fits = sh.run(kids)
        self.assertEqual(int(np.isfinite(fits).sum()), 2)   # survivors only
        self.assertLess(calls['n'], 8 * 3)                  # triage happened
        self.assertGreaterEqual(led.full_saved, 6)
    def test_surrogate_learns_and_screens(self):
        v = WFCTileVocab(); s = RidgeSurrogate(v, seed=0)
        rng = np.random.default_rng(0)
        gens = [rng.random((6, 10, 3)) for _ in range(60)]
        w = rng.normal(0, 1, v.T + 5)
        for g in gens: s.observe(g, float(s.features(g) @ w))
        s.refit()
        self.assertTrue(s.ok); self.assertGreater(s.rho, 0.8)
        kept, screened = s.screen(gens, 10)
        self.assertEqual(len(kept), 10); self.assertEqual(screened, 50)

class TestSearchers(unittest.TestCase):
    def test_nes_minimizes_quadratic(self):
        nes = LowRankNES((12,), rank=4, lr=0.1, sigma0=0.15,
                         rng=np.random.default_rng(0))
        target = np.full(12, 0.3)
        f = lambda w: float(np.mean((w - target) ** 2))
        anchor = np.full(12, 0.6); f0 = f(anchor)
        for _ in range(40):
            p, m = nes.ask(anchor, 6)
            nes.tell(np.array([f(x.reshape(12)) for x in p]),
                     np.array([f(x.reshape(12)) for x in m]))
        self.assertLess(f(nes.apply(anchor).reshape(12)), f0)
    def test_spsa_improves(self):
        f = lambda g: float(np.mean((g - 0.3) ** 2))
        g0 = np.random.default_rng(1).random((4, 4, 3))
        sp = SPSAPolish(f, np.random.default_rng(2), a=0.1, c=0.05)
        g1, f1 = sp.improve(g0, f(g0), steps=25)
        self.assertLess(f1, f(g0))

class TestMemories(unittest.TestCase):
    def test_archive(self):
        a = BehaviorArchive(rng=np.random.default_rng(0))
        rng = np.random.default_rng(1)
        for i in range(40): a.add(rng.random((6, 8, 3)), reward=float(i % 7))
        self.assertGreater(len(a.cells), 1)
        self.assertGreaterEqual(a.novelty(rng.random((6, 8, 3))), 0)
        self.assertLessEqual(len(a.top(3)), 3)
    def test_column_pool(self):
        p = ColumnPool(keep=4)
        g = np.random.default_rng(0).random((6, 10, 3))
        p.add(g, 1.0); p.add(g + 0.1, 2.0)
        self.assertEqual(p.graft(g * 0.5, np.random.default_rng(1), prob=1.0).shape,
                         g.shape)
        self.assertTrue(np.allclose(
            ColumnPool().graft(g, np.random.default_rng(2), 1.0), g))
    def test_aos(self):
        aos = AdaptiveOperatorSelector(['a', 'b'], seed=0)
        for _ in range(50):
            op = aos.select(); aos.update(op, 1.0 if op == 1 else 0.0)
        self.assertGreater(aos.get_probabilities()[1], aos.get_probabilities()[0])
    def test_islands(self):
        eng = WFCCAEngine(seed=0)
        isl = IslandHost(eng, 2, np.random.default_rng(0))
        g = generate_genome(np.random.default_rng(1), 6, 10, 'slabs')
        isl.set_island(0); isl.learn([g, g + 0.01])
        isl.set_island(1); isl.learn([g])
        self.assertEqual(isl.crossover(g, g + 0.01, 0).shape, g.shape)
        self.assertEqual(isl.make_genomes(2, (6, 10), mode='diverse').shape,
                         (2, 6, 10, 3))
    def test_restart_manager(self):
        rm = RestartManager(hist_len=12, stag_gate=5)
        for v in np.linspace(1, 0.1, 12): rm.observe(v)
        self.assertFalse(rm.should_restart(20))
        rm2 = RestartManager(hist_len=12, stag_gate=5)
        for v in [0.5] * 12: rm2.observe(v)
        self.assertTrue(rm2.should_restart(20))

class TestSpectral(unittest.TestCase):
    def test_spectra(self):
        sp = DataSpectra(np.random.default_rng(0).normal(size=(100, 5)), 3)
        self.assertEqual(sp.V.shape[0], 3)
        self.assertTrue(0 < sp.pr <= 3 + 1e-9)
    def test_regime(self):
        r = LossRegime()
        for v in np.linspace(1.0, 0.1, 12): r.observe(v)
        r.update()
        self.assertEqual(r.regime, 'refine'); self.assertLess(r.explore, 0.5)
    def test_eda(self):
        rng = np.random.default_rng(0)
        eda = GenomeEDA((6, 10, 3), rng)
        pop = np.stack([generate_genome(rng, 6, 10, 'slabs') for _ in range(10)])
        eda.fit(pop)
        self.assertEqual(eda.sample(3).shape, (3, 6, 10, 3))
        self.assertEqual(eda.around([pop[0], pop[1]]).shape, (6, 10, 3))

class TestKAN(unittest.TestCase):
    def test_refine1_is_legacy(self):
        g = np.random.default_rng(0).random((6, 10, 3))
        X = np.random.default_rng(1).random((5, 3))
        self.assertTrue(np.allclose(ImageFFN(refine=1).from_genome(g).forward(X),
                                    ImageFFN().from_genome(g).forward(X)))
    def test_refine3_reinterprets_same_pixels(self):
        h = np.full((4, 4), (20 + 0.5) / 72)
        g = np.stack([h, np.full((4, 4), 0.8), np.full((4, 4), 0.5)], -1)
        X = np.random.default_rng(2).random((6, 3))
        self.assertFalse(np.allclose(ImageFFN(refine=3).from_genome(g).forward(X),
                                     ImageFFN(refine=1).from_genome(g).forward(X)))
    def test_edge_mode_shapes_and_difference(self):
        g = np.random.default_rng(3).random((6, 10, 3))
        X = np.random.default_rng(4).random((5, 3))
        n = ImageFFN(mode='edge').from_genome(g).forward(X)
        d = ImageFFN(mode='node').from_genome(g).forward(X)
        self.assertEqual(n.shape, d.shape)
        self.assertFalse(np.allclose(n, d))
    def test_edge_acyclic(self):
        g = np.random.default_rng(5).random((6, 10, 3))
        X = np.random.default_rng(6).random((4, 3))
        for mode in ('node', 'edge'):
            s1 = ImageFFN(mode=mode).from_genome(g).forward(X, return_states=True)[1]
            g2 = g.copy(); g2[:, 4, 0] = (g2[:, 4, 0] + 0.13) % 1.0
            s2 = ImageFFN(mode=mode).from_genome(g2).forward(X, return_states=True)[1]
            for k in range(5):
                self.assertTrue(np.allclose(s1[k], s2[k], atol=1e-12),
                                f"{mode} state {k}")
            self.assertFalse(np.allclose(s1[5], s2[5]))
    def test_symbolify_purity_and_fidelity(self):
        g = np.random.default_rng(7).random((6, 8, 3))
        g[..., 0] = np.broadcast_to((np.arange(8) % 24 + 0.5) / 24, (6, 8))
        self.assertAlmostEqual(kan_metrics(g)['col_purity'], 1.0, places=5)
        X, y = make_threshold_task(80, 0)
        self.assertTrue(0.0 <= symbol_fidelity(g, X, y) <= 1.0)
    def test_refine_schedule(self):
        d = SpectralDriver(WFCCAEngine(seed=0),
                           IslandHost(WFCCAEngine(seed=0), 1, np.random.default_rng(0)),
                           BehaviorArchive(),
                           {'img_refine_max': 3, 'refine_plateau_after': 3})
        d.regime.regime = 'plateau'; d._plateau_run = 3
        self.assertEqual(d._refine_schedule(), 3)
        self.assertIsNone(d._refine_schedule())
    def test_cache_isolation_across_refine(self):
        c1, c2 = FitnessCache(), FitnessCache()
        g = np.random.default_rng(8).random((4, 4, 3))
        c1.insert(g, 1.0)
        self.assertIsNone(c2.lookup(g))

class TestKAN2(unittest.TestCase):
    def test_engine_set_refine(self):
        e = WFCCAEngine(seed=0); e.set_refine(3)
        self.assertEqual(e.vocab.T, 24 * 3 * 4 * 3)
        g = np.full((6, 8, 3), 0.5); g[..., 0] = 0.3
        out = e.ca.develop(g, np.random.default_rng(0))
        self.assertTrue(out.min() >= 0 and out.max() <= 1)
    def test_edge_output_activation(self):
        rng = np.random.default_rng(0)
        g = rng.random((6, 10, 3)); g2 = g.copy()
        g2[:, -1, 0] = (1 + 0.5) / 24                      # Sigmoid output
        o = ImageFFN(mode='edge').from_genome(g2).forward(rng.random((8, 3)))
        self.assertTrue(np.all(o > 0) and np.all(o < 1))
    def test_prune_changes_decode(self):
        rng = np.random.default_rng(1)
        g = rng.random((6, 10, 3)); g[..., 1] *= 0.3
        X = rng.random((5, 3))
        self.assertFalse(np.allclose(
            ImageFFN(prune=0.0).from_genome(g).forward(X),
            ImageFFN(prune=0.2).from_genome(g).forward(X)))
    def test_prune_schedule_monotone(self):
        d = SpectralDriver(WFCCAEngine(seed=0),
                           IslandHost(WFCCAEngine(seed=0), 1, np.random.default_rng(0)),
                           BehaviorArchive(), {'prune_ladder': (0.0, .05, .10)})
        d._elite_edge_l1 = 0.9
        self.assertEqual(d._prune_schedule(), 0.05)
        self.assertEqual(d._prune_schedule(), 0.10)
        d._elite_edge_l1 = 0.10
        self.assertIsNone(d._prune_schedule())
    def test_certificate_panel(self):
        g = np.random.default_rng(2).random((6, 10, 3))
        X, y = make_threshold_task(80, 0)
        Xa, ya, Xb, yb, Xc, yc = split3(X, y, seed=0)
        c = certify_intelligent_image(g, Xc, yc, Xb, yb)
        self.assertEqual(set(c['checks']),
                         {'generalization', 'val_gap', 'sparsity', 'compactness',
                          'purity', 'symbol_fidelity'})
        self.assertIsInstance(c['pass'], bool)

class TestIntelligence(unittest.TestCase):
    def _probe(self, task='classification', seed=0):
        X, y = make_blobs_task(160, seed=1)
        Xa, ya, Xb, yb, Xc, yc = split3(X, y, seed=seed)
        return IntelligenceProbe(task, Xb, yb, Xc, yc, seed=seed)
    def test_relative_gate_rejects_majority_copier(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (200, 4))
        y = np.eye(2)[(rng.random(200) < 0.9).astype(int)]
        Xa, ya, Xb, yb, Xc, yc = split3(X, y, seed=0)
        pr = IntelligenceProbe('classification', Xb, yb, Xc, yc)
        self.assertGreater(pr.null_majority, 0.85)
        self.assertFalse(perf_gate(pr.null_majority + 0.01, pr.null_majority))
        self.assertTrue(perf_gate(pr.null_majority + 0.08, pr.null_majority))
    def test_dead_network_detected(self):
        pr = self._probe()
        g = np.random.default_rng(3).random((8, 12, 3))
        g[:, 0, 1] = 0.0; g[..., 1] *= 0.2; g[..., 2] = 0.85
        a = pr.assess(g)
        self.assertLess(a['components']['sens'], 0.25)
        self.assertFalse(a['gates']['not_dead'])
    def test_score_bounds_and_grade(self):
        a = self._probe().assess(np.random.default_rng(4).random((8, 12, 3)))
        self.assertTrue(0.0 <= a['score'] <= 100.0)
        self.assertIn(a['grade'], ('intelligent', 'promising', 'marginal',
                                   'not intelligent'))
        self.assertIsInstance(a['pass'], bool)
    def test_ece_sanity(self):
        n = 200
        p = np.full((n, 2), 0.5); lab = np.array([0, 1] * (n // 2))
        self.assertLess(_ece(p, lab), 0.05)
        p = np.zeros((n, 2)); p[:, 0] = 1.0
        self.assertGreater(_ece(p, np.zeros(n, int)), 0.9)
    def _cfg(self):
        return dict(profile='fast', pop_size=10, n_children=6, genome_H=8,
                    genome_W=12, islands=1, nes_enabled=False, spsa_enabled=False,
                    surr_enabled=False, wfc_budget_ms=np.inf)
    def test_monitor_history_and_capture(self):
        X, y = make_blobs_task(120, seed=2)
        Xa, ya, Xb, yb, Xc, yc = split3(X, y, seed=0)
        mon = IntelligenceMonitor(
            IntelligenceProbe('classification', Xb, yb, Xc, yc),
            every=2, verbose=False, typer=ImageTyper(), ledger=TypeLedger())
        res = train(X, y, cfg=self._cfg(), seed=0, generations=6,
                    intel_monitor=mon)
        self.assertGreaterEqual(len(mon.history), 3)
        self.assertIn('intelligence', res)
        if mon.best_certified is not None:
            self.assertEqual(mon.best_certified[0].shape, (8, 12, 3))
    def test_early_stop_when_certified(self):
        X, y = make_blobs_task(120, seed=3)
        Xa, ya, Xb, yb, Xc, yc = split3(X, y, seed=0)
        pr = IntelligenceProbe('classification', Xb, yb, Xc, yc)
        calls = {'n': 0}; orig = pr.assess
        def fake(g, **k):
            calls['n'] += 1
            if calls['n'] >= 3:
                return {'score': 90.0, 'grade': 'intelligent', 'pass': True,
                        'gates': {}, 'test': 0.9, 'val': 0.9,
                        'null_majority': 0.0, 'evo_null': None, 'components': {},
                        'kan': kan_metrics(g), 'symbol_fidelity': 1.0, 'refine': 1}
            return orig(g, **k)
        pr.assess = fake
        mon = IntelligenceMonitor(pr, every=1, hold=1, stop_when_certified=True,
                                  verbose=False)
        res = train(X, y, cfg=self._cfg(), seed=0, generations=50,
                    intel_monitor=mon)
        self.assertLess(res['generations_done'], 50)
        self.assertIsNotNone(mon.best_certified)
    def test_model_fit_carries_intelligence(self):
        X, y = make_blobs_task(120, seed=4)
        m = Model.fit(X, y, seconds=2.0, profile='fast', seed=0, verbose=False,
                      overrides={'max_generations': 3})
        self.assertIn('intelligence', m.metrics)
        m.save('_tmp_intel')
        j = _json.load(open('_tmp_intel/model.json'))
        self.assertIn('intelligence', j['metrics'])
        import shutil; shutil.rmtree('_tmp_intel')

class TestTaxonomy(unittest.TestCase):
    def _g(self, hue2d, sat=0.8, H=8, W=12):
        h = np.broadcast_to(np.asarray(hue2d, float), (H, W))
        return np.clip(np.stack([h, np.full((H, W), sat),
                                 np.full((H, W), 0.5)], -1), 0, 1)
    def test_slab_vs_stripes(self):
        ty = ImageTyper()
        colh = np.tile(((np.arange(12) % 24) + 0.5) / 24, (8, 1))
        self.assertEqual(ty.type_of(self._g(colh))['type'], 'Slab Circuit')
        rowh = np.tile((((np.arange(8) % 24) + 0.5) / 24)[:, None], (1, 12))
        self.assertEqual(ty.type_of(self._g(rowh))['type'], 'Banded Stripes')
    def test_dead_and_periodic(self):
        ty = ImageTyper()
        self.assertEqual(ty.type_of(self._g(np.full((8, 12), 0.5), sat=0.02))['type'],
                         'Dead Zone')
        per = np.tile(np.array([(21 + .5) / 24, (22 + .5) / 24, (23 + .5) / 24]), (8, 4))
        self.assertEqual(ty.type_of(self._g(per))['type'], 'Periodic Wave')
    def test_sparse_logic_and_random(self):
        ty = ImageTyper()
        thr = np.array([285 / 360, 270 / 360, 280 / 360])
        cb = thr[(np.add.outer(np.arange(8), np.arange(12))) % 3]
        self.assertEqual(ty.type_of(self._g(cb, sat=0.15))['type'], 'Sparse Logic')
        self.assertIn(ty.type_of(np.random.default_rng(0).random((8, 12, 3)))['type'],
                      ('Noisy Static', 'Hybrid Mosaic', 'Plaid Field'))
    def test_ledger_and_curation(self):
        led = TypeLedger()
        g = np.zeros((6, 8, 3)); g[..., 1] = 0.02
        for gen in range(4): led.record(gen, 'Dead Zone', 10 + gen, False, g)
        self.assertEqual(led.report()[0][5], 'PRUNE NOW')
        X, y = make_blobs_task(80, seed=8)
        cfg = dict(profile='fast', pop_size=12, n_children=6, genome_H=6,
                   genome_W=10, islands=1, nes_enabled=False, spsa_enabled=False,
                   surr_enabled=False, wfc_budget_ms=np.inf)
        ctl = NoBackpropController(X, y, cfg, seed=0); ctl.start()
        replaced = apply_curation(ctl, led, ImageTyper())
        self.assertLessEqual(replaced, max(1, len(ctl.pop) // 8))
        self.assertTrue(np.isfinite(ctl.fits).all())
    def test_monitor_groups(self):
        X, y = make_blobs_task(120, seed=9)
        Xa, ya, Xb, yb, Xc, yc = split3(X, y, seed=0)
        mon = IntelligenceMonitor(
            IntelligenceProbe('classification', Xb, yb, Xc, yc),
            every=2, verbose=False, typer=ImageTyper(), ledger=TypeLedger())
        train(X, y, cfg=TestIntelligence._cfg(self), seed=0, generations=6,
              intel_monitor=mon)
        self.assertGreater(len(mon.ledger.groups), 0)
        self.assertTrue(mon.ledger.text().startswith('family'))
        if PIL_AVAILABLE and mon.best_by_type:
            name, img = next(iter(mon.best_by_type.items()))
            p = contact_sheet([img[0]], '_tmp_fam.png')
            self.assertTrue(os.path.exists(p)); os.remove(p)

class TestLineage(unittest.TestCase):
    def test_birth_links_depth_root(self):
        lb = LineageBook(ImageTyper())
        a = np.random.default_rng(0).random((6, 8, 3))
        b = np.random.default_rng(1).random((6, 8, 3))
        lb.birth(a, 0, 'founder'); lb.birth(b, 0, 'founder')
        c = sbx(a, b, 15, np.random.default_rng(2))
        r = lb.birth(c, 1, 'child', parents=(a, b), op='classic')
        self.assertEqual(r['depth'], 1); self.assertEqual(len(r['ptypes']), 2)
        d = poly_mut(c, 20, np.random.default_rng(3))
        r2 = lb.birth(d, 2, 'child', parents=(c,), op='classic')
        self.assertEqual(r2['depth'], 2)
        self.assertEqual(r2['root'], lb.rec[lb.key(a)]['root'])
    def test_transitions_and_operator_survival(self):
        lb = LineageBook(ImageTyper())
        a = np.random.default_rng(4).random((6, 8, 3))
        b = np.random.default_rng(5).random((6, 8, 3))
        for i in range(5):
            c = sbx(a, b, 15, np.random.default_rng(10 + i))
            lb.birth(c, 1, 'child', parents=(a, b), op='wfc')
            lb.note_survived(c, 0.1 + i)
        op, births, surv, _rate = lb.op_stats()[0]
        self.assertEqual((op, births, surv), ('wfc', 5, 5))
        self.assertGreater(len(lb.top_transitions(3)), 0)
    def test_chain_walk_and_dominance(self):
        lb = LineageBook(ImageTyper())
        a = np.random.default_rng(6).random((6, 8, 3)); lb.birth(a, 0, 'founder')
        ch = np.clip(a + 0.01, 0, 1)
        lb.birth(ch, 1, 'child', parents=(a,), op='wfc')
        self.assertIn('founder', lb.chain(ch)); self.assertIn('wfc', lb.chain(ch))
        self.assertAlmostEqual(lb.founder_dominance([a, ch, a])[0], 1.0)
    def test_composition_and_json(self):
        lb = LineageBook(ImageTyper())
        pop = np.random.default_rng(7).random((6, 8, 10, 3))
        lb.snapshot(0, pop)
        self.assertEqual(sum(lb.composition[0][1].values()), 6)
        a = np.random.default_rng(8).random((6, 8, 3)); lb.birth(a, 0, 'founder')
        lb.birth(a + 0.01, 1, 'child', parents=(a,), op='wfc')
        lb.to_json('_tmp_lin.json')
        self.assertEqual(LineageBook.from_json('_tmp_lin.json').chain(a + 0.01),
                         lb.chain(a + 0.01))
        os.remove('_tmp_lin.json')
    def test_controller_e2e_records_everything(self):
        X, y = make_blobs_task(80, seed=10)
        cfg = dict(profile='fast', pop_size=10, n_children=6, genome_H=6,
                   genome_W=10, islands=1, nes_enabled=False, spsa_enabled=False,
                   surr_enabled=False, wfc_budget_ms=np.inf)
        ctl = NoBackpropController(X, y, cfg, seed=0); ctl.start()
        for _ in range(3): ctl.run_generation()
        for g in ctl.pop:
            self.assertIn(ctl.lineage.key(g), ctl.lineage.rec)
        txt = ctl.lineage_text()
        self.assertIn('champion:', txt); self.assertIn('roots:', txt)
        self.assertGreaterEqual(len(ctl.lineage.composition), 3)
        self.assertIn('chain', ctl.stats()['lineage'])

class TestLineageExport(unittest.TestCase):
    def _book(self):
        lb = LineageBook(ImageTyper())
        a = np.random.default_rng(0).random((6, 8, 3))
        b = np.random.default_rng(1).random((6, 8, 3))
        lb.birth(a, 0, 'founder'); lb.birth(b, 0, 'founder')
        c = sbx(a, b, 15, np.random.default_rng(2))
        lb.birth(c, 1, 'child', parents=(a, b), op='classic')
        d = poly_mut(c, 20, np.random.default_rng(3))
        lb.birth(d, 2, 'child', parents=(c,), op='wfc')
        lb.note_score(d, 84.7, True, 2)
        return lb, a, d
    def test_graphml_wellformed(self):
        import xml.etree.ElementTree as ET
        lb, _a, _d = self._book()
        info = export_graphml(lb, '_tmp_lin.graphml')
        root = ET.parse('_tmp_lin.graphml').getroot()
        ns = '{http://graphml.graphdrawing.org/xmlns}'
        self.assertEqual(len(root.findall(f'.//{ns}node')), info['nodes'])
        self.assertEqual(len(root.findall(f'.//{ns}edge')), 3)
        self.assertIn('certified', ET.tostring(root).decode())
        os.remove('_tmp_lin.graphml')
    def test_gexf_colors(self):
        import xml.etree.ElementTree as ET
        lb, _a, _d = self._book()
        export_gexf(lb, '_tmp_lin.gexf')
        self.assertIn('viz', ET.tostring(ET.parse('_tmp_lin.gexf').getroot()).decode())
        os.remove('_tmp_lin.gexf')
    def test_timeline_and_fan(self):
        lb, a, d = self._book()
        pop = np.stack([a, d]); lb.snapshot(0, pop); lb.snapshot(1, pop)
        p = export_timeline_csv(lb, '_tmp_tl.csv')
        rows = list(_csv.reader(open(p)))
        self.assertEqual(rows[0][0], 'gen'); self.assertEqual(len(rows), 3)
        os.remove(p)
        if PIL_AVAILABLE:
            q = render_fan(lb, [(d, 84.7, True)], '_tmp_fan.png')
            self.assertTrue(os.path.exists(q) and os.path.getsize(q) > 0)
            os.remove(q)
    def test_monitor_tags_book_and_e2e_export(self):
        X, y = make_blobs_task(120, seed=11)
        cfg = dict(profile='fast', pop_size=10, n_children=6, genome_H=8,
                   genome_W=12, islands=1, nes_enabled=False, spsa_enabled=False,
                   surr_enabled=False, wfc_budget_ms=np.inf)
        Xa, ya, Xb, yb, Xc, yc = split3(X, y, seed=0)
        ctl = NoBackpropController(X, y, cfg, seed=0)
        mon = IntelligenceMonitor(
            IntelligenceProbe('classification', Xb, yb, Xc, yc), every=1,
            verbose=False)
        mon.book = ctl.lineage = LineageBook(ImageTyper())
        ctl.start()
        for _ in range(4): ctl.run_generation()
        self.assertGreater(len(ctl.lineage.scores), 0)
        info = export_graphml(ctl.lineage, '_tmp_e2e.graphml')
        self.assertGreaterEqual(info['nodes'], len(ctl.pop))
        os.remove('_tmp_e2e.graphml')

class TestMOController(unittest.TestCase):
    def test_mo_smoke_and_front(self):
        X, y = make_blobs_task(120, seed=0)
        cfg = dict(pop_size=12, n_children=8, genome_H=6, genome_W=10, islands=1,
                   nes_pairs=2, mo_enabled=True, wfc_budget_ms=np.inf)
        res = train(X, y, cfg=cfg, seed=0, generations=4)
        self.assertGreaterEqual(int((res['controller']._rank == 0).sum()), 1)
        self.assertIn('PASS', res['audit'])

class TestControllerE2E(unittest.TestCase):
    def _run(self, seed):
        X, y = make_threshold_task(120, seed=1)
        cfg = dict(profile='fast', pop_size=16, n_children=8, genome_H=6,
                   genome_W=14, wfc_budget_ms=np.inf)
        auditor = GradientFreedomAuditor(); auditor.install()
        ctl = NoBackpropController(X, y, cfg, seed=seed, auditor=auditor)
        ctl.start()
        first = float(ctl.fits.min())
        for _ in range(6): ctl.run_generation()
        return ctl, first, auditor
    def test_learns_and_audit_passes(self):
        ctl, first, auditor = self._run(0)
        self.assertLess(float(ctl.fits.min()), first)
        self.assertGreater(ctl.auditor.sample_evals, 0)
        self.assertGreater(ctl.ledger.full_saved + ctl.ledger.cache_hits, 0)
        self.assertEqual(auditor.backward_attempts, 0)
        self.assertEqual(len(auditor.violations), 0)
        self.assertIn('PASS', auditor.report(ctl.ledger))
    def test_determinism(self):
        _, f1, _ = self._run(7)
        _, f2, _ = self._run(7)
        self.assertEqual(f1, f2)

class TestTrainAPI(unittest.TestCase):
    def test_train_classification(self):
        X, y = make_blobs_task(160, seed=0)
        res = train(X, y, cfg=dict(profile='fast', pop_size=20, n_children=10,
                                   genome_H=6, genome_W=14, wfc_budget_ms=np.inf),
                    seed=0, generations=10)
        self.assertGreater(res['hist']['acc'][-1], res['hist']['acc'][0])
        self.assertIn('PASS', res['audit'])
        save_model_png(snap_to_bands(res['best_genome']), '_tmp_model.png')
        self.assertTrue(np.allclose(res['best_genome'],
                                    load_model_png('_tmp_model.png'), atol=1e-4))
        os.remove('_tmp_model.png')

class TestPractical(unittest.TestCase):
    def test_loader_nan_and_task(self):
        p = '_tmp_prac.csv'
        with open(p, 'w') as f:
            f.write("a,b,label\n1,2,0\n,4,1\n3,,0\n5,6,1\n")
        b = load_dataset(p)
        self.assertFalse(np.isnan(b.X).any())
        self.assertEqual(b.task, 'classification')
        self.assertEqual(b.y.shape, (4, 2))
        os.remove(p)
    def test_auto_config_no_truncation(self):
        b = make_bundle(np.random.default_rng(0).random((40, 6)),
                        (np.random.default_rng(1).random(40) > .5).astype(int))
        self.assertGreaterEqual(auto_config(b, 'fast')['genome_H'], 6)
        b2 = make_bundle(np.random.default_rng(2).random((40, 100)),
                         (np.random.default_rng(3).random(40) > .5).astype(int))
        self.assertEqual(auto_config(b2)['auto_pca'], 32)
    def test_model_roundtrip_and_exactness(self):
        X, y = make_blobs_task(120, seed=3)
        m = Model.fit(X, y, seconds=2.0, profile='fast', seed=0, verbose=False,
                      overrides={'max_generations': 3})
        m.save('_tmp_model_dir')
        m2 = Model.load('_tmp_model_dir')
        Xs = np.random.default_rng(4).random((9, 2))
        self.assertTrue(np.array_equal(m.predict(Xs), m2.predict(Xs)))
        self.assertTrue(np.allclose(m2.predict_proba(Xs).sum(1), 1.0))
        self.assertTrue(os.path.exists('_tmp_model_dir/model.png'))
        import shutil; shutil.rmtree('_tmp_model_dir')
    def test_recheck_verdict_gate(self):
        X, y = make_blobs_task(120, seed=5)
        m = Model.fit(X, y, seconds=1.5, profile='fast', seed=0, verbose=False,
                      overrides={'max_generations': 2})
        X2, y2 = make_blobs_task(60, seed=6)
        r = m.recheck(X2, y2)
        for k in ('accuracy', 'null_majority', 'beats_null', 'recall_drift',
                  'certificate_still_holds'):
            self.assertIn(k, r)
        r2 = m.recheck(X2[:40], np.eye(2)[np.zeros(40, int)])
        self.assertFalse(r2['beats_null'])
        self.assertFalse(r2['certificate_still_holds'])
    def test_deadline_and_partial_generation(self):
        X, y = make_blobs_task(80, seed=7)
        res = train(X, y, cfg=dict(profile='fast', pop_size=12, n_children=6,
                                   genome_H=6, genome_W=10, islands=1,
                                   nes_enabled=False, spsa_enabled=False,
                                   surr_enabled=False, wfc_budget_ms=np.inf),
                    seed=0, generations=10 ** 6, max_seconds=1.0)
        self.assertGreaterEqual(res['generations_done'], 1)
        self.assertLess(res['elapsed'], 30)
        self.assertTrue(res['deadline_hit'])
    def test_checkpoint_resume_state(self):
        X, y = make_blobs_task(80, seed=8)
        cfg = dict(profile='fast', pop_size=12, n_children=6, genome_H=6,
                   genome_W=10, islands=1, nes_enabled=False, spsa_enabled=False,
                   surr_enabled=False, wfc_budget_ms=np.inf)
        r1 = train(X, y, cfg=cfg, seed=0, generations=2, checkpoint_every=2,
                   checkpoint_dir='_tmp_ck')
        f2 = float(r1['controller'].fits.min())
        ctl2 = NoBackpropController(X, y, cfg, seed=0)
        ctl2.load_checkpoint('_tmp_ck/ckpt_seed0.npz')
        self.assertAlmostEqual(float(ctl2.fits.min()), f2, places=9)
        self.assertEqual(ctl2.gen, 2)
        ctl2.run_generation(); self.assertEqual(ctl2.gen, 3)
        import shutil; shutil.rmtree('_tmp_ck')
    def test_netpool_caches(self):
        pool = NetPool(dict(radius=2, coupling='local'), size=2)
        g = np.random.default_rng(0).random((5, 6, 3))
        n1 = pool.get(g); self.assertIs(n1, pool.get(g))
        pool.get(np.random.default_rng(1).random((5, 6, 3)))
        pool.get(np.random.default_rng(2).random((5, 6, 3)))
        self.assertNotIn(np.ascontiguousarray(g).tobytes(), pool.d)

def run_tests():
    print(f"ga_image_ffn v{__version__} — running test suite…\n")
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules['__main__'])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1

# ═══════════════════════════════════════════════════════════════════════════
# 25. GUI (PySide6, optional)
# ═══════════════════════════════════════════════════════════════════════════
def gui():
    try:
        from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPointF, QRectF
        from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
        from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
            QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout, QSpinBox,
            QDoubleSpinBox, QComboBox, QGroupBox, QTabWidget, QPlainTextEdit,
            QCheckBox, QFileDialog)
    except ImportError:
        print("PySide6 not installed — GUI unavailable. Use --demo or --test.")
        return 1

    def hsv_pixmap(g, us=10):
        rgb = np.ascontiguousarray(_hsv_to_rgb_vec(np.asarray(g)))
        h, w, _ = rgb.shape
        qi = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qi.copy()).scaled(
            w * us, h * us, Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

    class ConvergencePlot(QWidget):
        def __init__(self):
            super().__init__(); self.hist = {'best': [], 'mean': [], 'acc': []}
            self.setMinimumHeight(250)
        def set_hist(self, hist): self.hist = hist; self.update()
        def paintEvent(self, _):
            p = QPainter(self); p.fillRect(self.rect(), QColor("#1e1e2e"))
            w, h = self.width(), self.height() - 20
            for key, col in (('best', '#89b4fa'), ('mean', '#6c7086'),
                             ('acc', '#a6e3a1')):
                v = self.hist.get(key, [])
                if len(v) < 2: continue
                a = np.asarray(v, float)
                lo, hi = a.min(), a.max(); rng_ = max(hi - lo, 1e-12)
                p.setPen(QPen(QColor(col), 1.6)); prev = None
                for i, x in enumerate(a):
                    pt = QPointF(40 + i / max(len(a) - 1, 1) * (w - 60),
                                 10 + (1 - (x - lo) / rng_) * (h - 20))
                    if prev: p.drawLine(prev, pt)
                    prev = pt
            p.setPen(QColor("#cdd6f4")); p.setFont(QFont("Arial", 8))
            if self.hist['best']:
                p.drawText(6, 18, f"best {self.hist['best'][-1]:.5f}")
            if self.hist.get('acc') and self.hist['acc'][-1] == self.hist['acc'][-1]:
                p.drawText(6, 34, f"acc {self.hist['acc'][-1]*100:.1f}%")
            p.end()

    class IntelPanel(QWidget):
        def __init__(self):
            super().__init__(); self.a = None; self.hist = []
            self.setMinimumHeight(300)
        def update_(self, intel, history):
            if intel: self.a = intel
            if history is not None:
                self.hist = [h['score'] for h in history]
            self.update()
        def paintEvent(self, _):
            p = QPainter(self); p.fillRect(self.rect(), QColor("#1e1e2e"))
            if not self.a: p.end(); return
            p.setPen(QColor("#cdd6f4"))
            p.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            p.drawText(10, 20, f"Intelligence: {self.a['score']}/100 — "
                        f"{self.a['grade']}  "
                        f"{'✓ CERTIFIED' if self.a['pass'] else ''}")
            comps = self.a['components']
            keys = [k for k in comps if comps[k] is not None]
            for i, k in enumerate(keys):
                y = 40 + i * 22
                p.setPen(QColor("#6c7086")); p.drawText(10, y + 14, k)
                p.setBrush(QColor("#313244")); p.setPen(Qt.PenStyle.NoPen)
                p.drawRect(150, y, 240, 15)
                p.setBrush(QColor("#a6e3a1" if comps[k] >= .6 else
                                  "#f9e2af" if comps[k] >= .35 else "#f38ba8"))
                p.drawRect(150, y, int(240 * comps[k]), 15)
            if len(self.hist) >= 2:
                p.setPen(QPen(QColor("#89b4fa"), 1.6)); prev = None
                base = self.height() - 20
                for i, v in enumerate(self.hist):
                    pt = QPointF(150 + i / max(len(self.hist) - 1, 1) * 240,
                                 base - v / 100.0 * 60)
                    if prev: p.drawLine(prev, pt)
                    prev = pt
            p.end()

    class CompBars(QWidget):
        def __init__(self):
            super().__init__(); self.data = []
            self.setToolTip("Population composition by image family")
        def set_data(self, comp): self.data = comp or []; self.update()
        def paintEvent(self, _):
            p = QPainter(self); p.fillRect(self.rect(), QColor("#1e1e2e"))
            if not self.data: p.end(); return
            types = sorted({t for _, d in self.data for t in d})
            pal = ["#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8", "#cba6f7",
                   "#94e2d5", "#fab387", "#f5c2e7", "#585b70", "#6c7086",
                   "#74c7ec", "#b4befe"]
            col = {t: pal[i % len(pal)] for i, t in enumerate(types)}
            n, W, H = len(self.data), self.width() - 8, self.height() - 16
            for i, (_gen, d) in enumerate(self.data):
                x = 4 + i / max(n, 1) * W; y = 8.0
                tot = max(sum(d.values()), 1); bw = max(W / max(n, 1) - 1, 2.0)
                for t in types:
                    if t in d:
                        hh = d[t] / tot * H
                        p.fillRect(QRectF(x, y, bw, hh), QColor(col[t])); y += hh
            p.end()

    class Worker(QThread):
        progress = Signal(dict); finished_run = Signal(dict)
        def __init__(self, X, y, cfg, gens, seed):
            super().__init__(); self.X, self.y, self.cfg = X, y, cfg
            self.gens, self.seed = gens, seed; self._stop = False
        def stop(self): self._stop = True
        def run(self):
            try:
                res = train(self.X, self.y, self.cfg, self.seed, self.gens,
                            callback=self.progress.emit,
                            stop_flag=lambda: self._stop)
                self.finished_run.emit(res)
            except Exception:
                traceback.print_exc()
                self.finished_run.emit({'audit': 'ERROR (see console)',
                                        'controller': None})

    class WFCStudio(QWidget):
        def __init__(self):
            super().__init__()
            self.eng = WFCCAEngine({'wfc_temperature': 0.85}, seed=0)
            self.gen = None; self.cur = None
            lay = QHBoxLayout(self); btns = QVBoxLayout()
            for label, fn, tip in [
                    ("🧩 Collapse new", self._new,
                     "Collapse a genome from the current grammar"),
                    ("📚 Learn from shown", self._learn,
                     "Add the shown genome to the grammar"),
                    ("⏸/▶ Animate", self._toggle,
                     "Pause/resume collapse animation"),
                    ("🦠 CA step", self._ca,
                     "One CA developmental step on shown genome"),
                    ("💾 Export PNG", self._export,
                     "Save shown genome as model PNG")]:
                b = QPushButton(label); b.setToolTip(tip); b.clicked.connect(fn)
                btns.addWidget(b)
            btns.addWidget(QLabel("temp:"))
            self.temp = QDoubleSpinBox(); self.temp.setRange(0.05, 3.0)
            self.temp.setSingleStep(0.05); self.temp.setValue(0.85)
            btns.addWidget(self.temp); btns.addStretch()
            lw = QWidget(); lw.setLayout(lw); lw.setMaximumWidth(220)
            lay.addWidget(lw)
            self.img = QLabel("Collapse a genome to begin")
            self.img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.img.setMinimumSize(320, 240)
            lay.addWidget(self.img, 1)
            self.timer = QTimer(self); self.timer.setInterval(40)
            self.timer.timeout.connect(self._tick)
        def _render(self):
            if self.cur is not None: self.img.setPixmap(hsv_pixmap(self.cur, 12))
        def _new(self):
            H, W = (self.cur.shape[:2] if self.cur is not None else (12, 20))
            self.gen = FastWFC(self.eng.vocab, self.temp.value(),
                               rng=np.random.default_rng())
            self.gen.prepare(H, W); self.timer.start()
        def _tick(self):
            if self.gen is None: self.timer.stop(); return
            done = self.gen.run_round()
            self.cur = self.gen.genomes(show_open=not done)[0]
            self._render()
            if done: self.timer.stop()
        def _toggle(self):
            self.timer.stop() if self.timer.isActive() else self.timer.start()
        def _learn(self):
            if self.cur is not None: self.eng.learn([self.cur])
        def _ca(self):
            if self.cur is not None:
                self.cur = self.eng.mutate(self.cur); self._render()
        def _export(self):
            if self.cur is None: return
            path, _ = QFileDialog.getSaveFileName(self, "Save model", "model.png",
                                                  "PNG (*.png)")
            if path: save_model_png(self.cur, path)
        def show_genome(self, g):
            self.cur = np.asarray(g).copy(); self._render()

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"GA·WFC·CA Image-FFN Trainer v{__version__} "
                                f"(no backprop)")
            self.resize(1280, 800)
            root = QHBoxLayout(); central = QWidget(); central.setLayout(root)
            self.setCentralWidget(central)
            box = QGroupBox("Configuration"); form = QFormLayout(box)
            self.task = QComboBox(); self.task.addItems(['classification',
                                                         'regression'])
            self.dataset = QComboBox(); self.dataset.addItems(['blobs', 'threshold'])
            self.gens = QSpinBox(); self.gens.setRange(1, 5000); self.gens.setValue(40)
            self.pop = QSpinBox(); self.pop.setRange(8, 512); self.pop.setValue(32)
            self.kids = QSpinBox(); self.kids.setRange(2, 256); self.kids.setValue(16)
            self.gh = QSpinBox(); self.gh.setRange(4, 64); self.gh.setValue(8)
            self.gw = QSpinBox(); self.gw.setRange(4, 128); self.gw.setValue(16)
            self.radius = QSpinBox(); self.radius.setRange(1, 6); self.radius.setValue(2)
            self.coupling = QComboBox(); self.coupling.addItems(['local',
                                                                 'global_sum'])
            self.mode = QComboBox(); self.mode.addItems(['node', 'edge'])
            self.isl = QSpinBox(); self.isl.setRange(1, 8); self.isl.setValue(2)
            self.spsa = QCheckBox("SPSA polish"); self.spsa.setChecked(True)
            self.surr = QCheckBox("Surrogate screen"); self.surr.setChecked(True)
            self.intel_stop = QCheckBox("Stop when certified"); self.intel_stop.setChecked(False)
            self.seed = QSpinBox(); self.seed.setRange(0, 9999); self.seed.setValue(0)
            for lbl, w in [("Task", self.task), ("Dataset", self.dataset),
                           ("Generations", self.gens), ("Population", self.pop),
                           ("Children", self.kids), ("Genome H (neurons)", self.gh),
                           ("Genome W (depth)", self.gw), ("Kernel radius", self.radius),
                           ("Coupling", self.coupling), ("Mode", self.mode),
                           ("Islands", self.isl), ("", self.spsa), ("", self.surr),
                           ("", self.intel_stop), ("Seed", self.seed)]:
                if lbl: form.addRow(lbl, w)
                else: form.addRow(w)
            self.b_train = QPushButton("▶ Train")
            self.b_train.clicked.connect(self.start_train)
            self.b_stop = QPushButton("⏹ Stop"); self.b_stop.setEnabled(False)
            self.b_stop.clicked.connect(lambda: self.worker and self.worker.stop())
            self.b_save = QPushButton("💾 Save best model")
            self.b_save.clicked.connect(self.save_model); self.b_save.setEnabled(False)
            for w in (self.b_train, self.b_stop, self.b_save): form.addRow(w)
            left = QWidget(); left.setLayout(QVBoxLayout())
            left.layout().addWidget(box); left.layout().addStretch()
            left.setMaximumWidth(280); root.addWidget(left)
            self.tabs = QTabWidget(); root.addWidget(self.tabs, 1)
            self.plot = ConvergencePlot(); self.tabs.addTab(self.plot, "Convergence")
            self.genome_lbl = QLabel("No trained genome yet")
            self.genome_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tabs.addTab(self.genome_lbl, "Genome = Network")
            self.arch_lbl = QLabel("Archive heatmap appears after training")
            self.arch_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tabs.addTab(self.arch_lbl, "MAP-Elites")
            self.intel = IntelPanel(); self.tabs.addTab(self.intel, "Intelligence")
            tab_l = QWidget(); v = QVBoxLayout(tab_l)
            self.comp = CompBars(); self.lin = QPlainTextEdit()
            self.lin.setReadOnly(True)
            v.addWidget(self.comp, 1); v.addWidget(self.lin, 1)
            self.tabs.addTab(tab_l, "Lineage")
            self.studio = WFCStudio(); self.tabs.addTab(self.studio, "WFC/CA Studio")
            self.dash = QPlainTextEdit(); self.dash.setReadOnly(True)
            self.tabs.addTab(self.dash, "Dashboard")
            self.audit = QPlainTextEdit(); self.audit.setReadOnly(True)
            self.tabs.addTab(self.audit, "Audit")
            self.statusBar().showMessage(
                "Ready. The image IS the feed-forward network.")
            self.worker = None; self.best_g = None; self.mon = None
        def start_train(self):
            if self.worker and self.worker.isRunning(): return
            X, y = (make_blobs_task(240, self.seed.value())
                    if self.dataset.currentText() == 'blobs'
                    else make_threshold_task(240, self.seed.value()))
            if self.task.currentText() == 'regression':
                y = X.sum(1, keepdims=True)
            cfg = dict(task=self.task.currentText(), pop_size=self.pop.value(),
                       n_children=self.kids.value(), genome_H=self.gh.value(),
                       genome_W=self.gw.value(), img_radius=self.radius.value(),
                       img_coupling=self.coupling.currentText(),
                       img_mode=self.mode.currentText(),
                       islands=self.isl.value(), spsa_enabled=self.spsa.isChecked(),
                       surr_enabled=self.surr.isChecked())
            self.worker = Worker(X, y, cfg, self.gens.value(), self.seed.value())
            self.worker.progress.connect(self.on_progress)
            self.worker.finished_run.connect(self.on_done)
            self.b_train.setEnabled(False); self.b_stop.setEnabled(True)
            self.dash.clear(); self.audit.clear()
            self.worker.start()
        def on_progress(self, st):
            self.plot.set_hist(st['hist'])
            self.best_g = st['best_genome']
            self.genome_lbl.setPixmap(hsv_pixmap(
                st['best_genome'], max(4, 320 // st['best_genome'].shape[1])))
            self.studio.show_genome(st['best_genome'])
            ag = st.get('archive_grid')
            if ag is not None:
                rgb = make_heatmap(ag); h, w, _ = rgb.shape
                qi = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
                self.arch_lbl.setPixmap(QPixmap.fromImage(qi.copy()).scaled(
                    280, 280, Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.FastTransformation))
            self.intel.update_(st.get('intel'),
                               self.mon.history if self.mon else None)
            L = st.get('lineage') or {}
            self.comp.set_data(L.get('comp'))
            self.lin.setPlainText(L.get('text', ''))
            pol = st.get('polish')
            acc = st['acc']
            accs = f"{acc*100:.1f}%" if acc == acc else "n/a"
            self.statusBar().showMessage(
                f"gen {st['gen']} | loss {st['best_loss']:.5f} | acc {accs}"
                f" | {st['regime']} | polish {pol} | thr {st['throttle']:.2f}")
            self.dash.setPlainText(f"{st['driver']}\n\n{st['ledger']}")
        def on_done(self, res):
            self.b_train.setEnabled(True); self.b_stop.setEnabled(False)
            self.b_save.setEnabled(self.best_g is not None)
            self.audit.setPlainText(res.get('audit', ''))
            ctl = res.get('controller')
            if ctl: self.dash.appendPlainText("\n" + ctl.status_text())
            self.statusBar().showMessage(
                f"Done in {res.get('elapsed', 0):.1f}s — audit: see Audit tab.")
        def save_model(self):
            if self.best_g is None: return
            path, _ = QFileDialog.getSaveFileName(self, "Save model", "model.png",
                                                  "PNG (*.png)")
            if path: save_model_png(self.best_g, path)

    app = QApplication(sys.argv)
    win = MainWindow(); win.show()
    return app.exec()

# ═══════════════════════════════════════════════════════════════════════════
# 26. DEMO + MAIN
# ═══════════════════════════════════════════════════════════════════════════
def demo():
    print(f"═ ga_image_ffn v{__version__} — demo ═")
    print("Task: 2-D Gaussian blobs → 2 classes. The genome IMAGE is the "
          "network. No backprop anywhere.\n")
    m = Model.fit(make_blobs_task(240, seed=0)[0], make_blobs_task(240, seed=0)[1],
                  seconds=60, profile='balanced', seed=0, source='blobs-demo')
    it = m.metrics.get('intelligence', {})
    print(f"\nTEST accuracy {m.metrics['accuracy']*100:.1f}%"
          f" | intelligence {it.get('score', '—')}/100 ({it.get('grade', '—')})"
          f" | family: {m.metrics.get('type_of_model')}")
    print(f"ancestry: {m.metrics.get('champion_lineage', '—')[:120]}")
    m.save('runs/demo')
    write_report(m, 'runs/demo/report.txt')
    m.type_gallery('runs/demo/families')
    m.export_lineage('runs/demo/lineage')
    print("\n" + m.audit)
    print("\nartifacts → runs/demo/ (model.png IS the trained network)")
    return 0

def main():
    """Library + GUI entry points. For scripted use, import the API:
    Model.fit → Model.predict → Model.recheck (see module docstring)."""
    flags = set(sys.argv[1:])
    if '--test' in flags or 'test' in flags: return run_tests()
    if '--demo' in flags or 'demo' in flags: return demo()
    if '--gui' in flags or 'gui' in flags:   return gui()
    rc = run_tests()
    print("\n" + "═" * 70)
    return demo() if rc == 0 else rc

if __name__ == "__main__":
    sys.exit(main())