#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
#  ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ██████╗ ██╗██╗  ██╗
#  ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔══██╗██║╚██╗██╔╝
#  ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██████╔╝██║ ╚███╔╝
#  ██║╚██╗██║██╔══╝  ██║   ██║██╔═══╝ ██║   ██║██╔══██╗██║ ██╔██╗
#  ██║ ╚████║███████╗╚██████╔╝██║     ╚██████╔╝██║  ██║██║██╔╝ ██╗
#  ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
#  NeuroPix — an RGBA image IS a neural network (NNIMG v2).
#
#  Codec    : header + weights(4/px; optional 16-bit hi/lo bands) + bias/gain
#             + skips.  Architectures: mlp, local, cnn1d, rnn, lstm, res,
#             sparse, v1chain (any RGBA photo falls back to a pixel-chain net).
#  Intelligence:
#     CII5 = ( I/log2S · I/H · determinism · (1-confusion) · stability )^(1/5)
#     I(X;Y)  multi-positive InfoNCE bound (Oord'18, von Kügelgen'21),
#             maximin stimulus design
#     H(Y)    Kozachenko–Leonenko k-NN entropy (effective information)
#     stability CCA alignment of codes across independent stimulus sets
#     + TSE neural complexity (Tononi–Sporns–Edelman'94), MDL efficiency
#  Search   : successive halving → linear max / quickselect / merge sort /
#             binary + interpolation lookup → hill-climb(+restarts) /
#             annealing / GA(μ+λ) / MAP-Elites — all on batched forwards.
#  Task mode: --task xor  → fitness = XOR accuracy; evolution TRAINS the
#             network through its image bytes.
#  GUI      : PySide6 — live progress, cancel, pool table, embedded charts,
#             drag & drop PNGs, report dashboards.  Console mode: --cli.
#
#  Run:  pip install numpy pillow matplotlib PySide6
#        python neuropix.py                (GUI)
#        python neuropix.py --cli          (console report + charts)
#        python neuropix.py --cli --task xor --regen --seed 42
# ═══════════════════════════════════════════════════════════════════════════
import os, sys, glob, math, time, zlib, copy, hashlib, itertools, argparse, threading
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────── 0. config ───────────────────────────
MAGIC, VERSION = b"NNIM", 2
ACTS  = ["linear", "relu", "tanh", "sigmoid", "softmax"]
ARCHS = ["mlp", "local", "cnn1d", "rnn", "lstm", "res", "sparse"]
REPORT = "report"
GOLD, STEEL, RED = "#e6a817", "#4c72b0", "#c44e52"
LN2, EPS = math.log(2), 1e-12
FIT  = dict(S=8,  R=3)          # search fitness budget
FULL = dict(S=16, R=4)          # final scoring budget
TASK = "none"                   # "none" | "xor" (set by CLI / GUI)
POOL_DIR, POOL_N, POOL_SEED = "image_pool", 12, 7
SCORE_STATS = {"evals": 0, "hits": 0}
_CACHE = {}

U16 = lambda v: int(v).to_bytes(2, "little")
def F16(v): return np.asarray(v, "<f2").tobytes()
def quant(M, s):
    return np.clip(np.rint(np.asarray(M, np.float32) / s * 127.5 + 127.5), 0, 255).astype(np.uint8)
def dequant(Q, s): return (np.asarray(Q, np.float32) - 127.5) / 127.5 * s
def quant16(M, s):
    q = np.clip(np.rint(np.asarray(M, np.float32) / s * 32767.5 + 32767.5), 0, 65535).astype(np.uint16)
    return (q >> 8).astype(np.uint8), (q & 255).astype(np.uint8)
def dequant16(Qh, Ql, s):
    q = (Qh.astype(np.uint16) << 8) | Ql
    return (q.astype(np.float32) - 32767.5) / 32767.5 * s
def log(m): print(m, flush=True)
def clip8(v): return np.uint8(min(255, max(0, int(v))))

# ─────────────────────────── 1. NNIMG v2 codec ───────────────────────────
def extra_bits(arch, ly):
    if arch == "cnn1d":
        return ((ly["K"] & 255) << 8) | ((ly.get("stride", 1) & 15) << 4) | (ly.get("pad", 0) & 15)
    if arch == "local":          return int(ly.get("K", 3)) & 255
    if arch in ("rnn", "lstm"):  return int(ly.get("T", 8)) & 255
    return int(ly.get("extra", 0)) & 65535

def fan_in(arch, ly):
    if arch == "cnn1d":          return ly["n_in"] * ly["K"]
    if arch == "local":          return int(ly.get("K", 3))
    if arch in ("rnn", "lstm"):  return ly["n_in"] + ly["n_out"]
    return ly["n_in"]

def param_rows(arch, ly): return 4 * ly["n_out"] if arch == "lstm" else ly["n_out"]

def header_bytes(spec):
    b = bytearray(MAGIC); b.append(VERSION); b.append(ARCHS.index(spec["arch"]))
    b += U16(len(spec["layers"]))
    for ly in spec["layers"]:
        b += U16(ly["n_in"]) + U16(ly["n_out"])
        b.append(ACTS.index(ly.get("act", "tanh"))); b.append(int(ly.get("flags", 0)))
        b += F16(ly.get("scale", 1.0)) + U16(extra_bits(spec["arch"], ly))
    sk = spec.get("skips", []); b += U16(len(sk))
    for s in sk: b += bytes(int(v) & 255 for v in s)
    return bytes(b)

def encode(spec):
    arch = spec["arch"]; hdr = header_bytes(spec)
    W = 2
    for ly in spec["layers"]:
        W = max(W, -(-fan_in(arch, ly) // 4))
        if arch == "sparse": W = max(W, -(-ly["n_in"] // 32))
    adj = sum(ly["n_out"] for ly in spec["layers"]) if arch == "sparse" else 0
    wr  = sum(param_rows(arch, ly) * (2 if int(ly.get("flags", 0)) & 1 else 1) for ly in spec["layers"])
    br  = sum(param_rows(arch, ly) for ly in spec["layers"])
    ns  = len(spec.get("skips", []))
    hr  = -(-len(hdr) // (4 * W))
    img = np.zeros((hr + adj + wr + br + ns, W, 4), np.uint8)
    hb = np.frombuffer(hdr, np.uint8); idx = np.arange(len(hb))
    img.reshape(-1, 4)[idx // 4, idx % 4] = hb
    r = hr
    if arch == "sparse":
        for ly in spec["layers"]:
            A = np.asarray(ly.get("A", np.ones((ly["n_out"], ly["n_in"]))), np.uint8)
            pk = np.packbits(A, axis=1); need = -(-pk.shape[1] // 4) * 4
            pk = np.pad(pk, ((0, 0), (0, need - pk.shape[1])))
            img[r:r + ly["n_out"], :need // 4] = pk.reshape(ly["n_out"], need // 4, 4); r += ly["n_out"]
    for ly in spec["layers"]:
        fan = fan_in(arch, ly); need = -(-fan // 4) * 4; rows = param_rows(arch, ly)
        Wm = np.asarray(ly["W"], np.float32).reshape(rows, fan)
        if int(ly.get("flags", 0)) & 1:
            for q in quant16(Wm, ly.get("scale", 1.0)):
                q = np.pad(q, ((0, 0), (0, need - fan)))
                img[r:r + rows, :need // 4] = q.reshape(rows, need // 4, 4); r += rows
        else:
            q = np.pad(quant(Wm, ly.get("scale", 1.0)), ((0, 0), (0, need - fan)))
            img[r:r + rows, :need // 4] = q.reshape(rows, need // 4, 4); r += rows
    for ly in spec["layers"]:
        n = param_rows(arch, ly); s = ly.get("scale", 1.0)
        bb = np.asarray(ly.get("b", np.zeros(n)), np.float32).reshape(-1)
        gg = np.asarray(ly.get("g", np.ones(n)),  np.float32).reshape(-1)
        if bb.size != n: bb = np.zeros(n, np.float32)
        if gg.size != n: gg = np.ones(n, np.float32)
        col = np.stack([quant(bb, s), quant(gg, s), np.full(n, 127, np.uint8),
                        np.full(n, 255, np.uint8)], axis=1)
        img[r:r + n, :1] = col.reshape(n, 1, 4); r += n
    for fl, fr, tl, tr in spec.get("skips", []):
        img[r, 0] = [fl & 255, fr & 255, tl & 255, tr & 255]; r += 1
    return img

def decode(img):
    img = np.ascontiguousarray(np.asarray(img, np.uint8))
    H, Wd = img.shape[:2]; raw = img.reshape(-1)
    if raw.size < 10 or raw[:4].tobytes() != MAGIC: raise ValueError("not NNIMG")
    if int(raw[4]) != VERSION:                      raise ValueError("bad version")
    def u16(o): return int.from_bytes(raw[o:o + 2].tobytes(), "little")
    arch = ARCHS[int(raw[5])]; L = u16(6); off = 8
    if raw.size < 8 + 10 * L + 2: raise ValueError("truncated header")
    layers = []
    for _ in range(L):
        n_in, n_out = u16(off), u16(off + 2)
        act, flags = ACTS[int(raw[off + 4])], int(raw[off + 5])
        scale = float(np.frombuffer(raw[off + 6:off + 8].tobytes(), "<f2")[0])
        if not np.isfinite(scale) or scale <= 0: scale = 1.0
        ex = u16(off + 8); off += 10
        ly = dict(n_in=n_in, n_out=n_out, act=act, flags=flags, scale=scale)
        if arch == "cnn1d":           ly.update(K=max(1, (ex >> 8) & 255), stride=max(1, (ex >> 4) & 15), pad=ex & 15)
        elif arch == "local":         ly["K"] = max(1, ex & 255)
        elif arch in ("rnn", "lstm"): ly["T"] = max(1, ex & 255)
        layers.append(ly)
    ns = u16(off); off += 2; skips = []
    for _ in range(ns):
        skips.append((int(raw[off]), int(raw[off + 1]), int(raw[off + 2]), int(raw[off + 3]))); off += 4
    hr = -(-off // (4 * Wd))
    if hr >= H: raise ValueError("truncated")
    r = hr
    if arch == "sparse":
        for ly in layers:
            nb = -(-ly["n_in"] // 32)
            if r + ly["n_out"] > H: raise ValueError("truncated adjacency")
            pk = img[r:r + ly["n_out"], :nb].reshape(ly["n_out"], -1)
            ly["A"] = np.unpackbits(pk, axis=1)[:, :ly["n_in"]].astype(np.float32); r += ly["n_out"]
    for ly in layers:
        fan = fan_in(arch, ly); nb = -(-fan // 4); rows = param_rows(arch, ly)
        mult = 2 if ly["flags"] & 1 else 1
        if r + rows * mult > H: raise ValueError("truncated weights")
        if ly["flags"] & 1:
            Qh = img[r:r + rows, :nb].reshape(rows, -1)[:, :fan]; r += rows
            Ql = img[r:r + rows, :nb].reshape(rows, -1)[:, :fan]; r += rows
            Wm = dequant16(Qh, Ql, ly["scale"])
        else:
            Q = img[r:r + rows, :nb].reshape(rows, -1)[:, :fan]; r += rows
            Wm = dequant(Q, ly["scale"])
        if arch == "cnn1d":  Wm = Wm.reshape(ly["n_out"], ly["n_in"], ly["K"])
        if arch == "sparse": Wm = Wm * ly["A"]
        ly["W"] = Wm
    for ly in layers:
        n = param_rows(arch, ly)
        if r + n > H: raise ValueError("truncated bias")
        ch = img[r:r + n, :1, :].reshape(n, 4)
        ly["b"] = dequant(ch[:, 0], ly["scale"]); ly["g"] = dequant(ch[:, 1], ly["scale"]); r += n
    return dict(arch=arch, layers=layers, skips=skips, meta=dict(hdr_rows=hr, H=H, W=Wd))

def decode_any(img):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 4:
        try: return decode(img)
        except ValueError: pass
        g = img.astype(np.float32); h, w = g.shape[:2]
        if w >= 2 and h >= 1:
            up = np.clip(np.arange(h) - 1, 0, h - 1); dn = np.clip(np.arange(h) + 1, 0, h - 1)
            layers = []
            for x in range(1, w):
                Wm = np.zeros((h, h), np.float32); b = np.zeros(h, np.float32)
                for y in range(h):
                    Wm[y, up[y]] = g[y, x, 0] / 127.5 - 1
                    Wm[y, y]     = g[y, x, 1] / 127.5 - 1
                    Wm[y, dn[y]] = g[y, x, 2] / 127.5 - 1
                    b[y] = g[y, x, 3] / 127.5 - 1
                layers.append(dict(n_in=h, n_out=h, act="tanh", scale=1.0,
                                   flags=0, W=Wm, b=b, g=np.ones(h, np.float32)))
            return dict(arch="v1chain", layers=layers, skips=[], meta=dict(hdr_rows=0, H=h, W=w))
    raise ValueError("undecodable image")

# ─────────────────────────── 2. forward (single + batched) ───────────────────────────
def _act(a, c):
    if c == 1: return np.maximum(a, 0.0)
    if c == 2: return np.tanh(a)
    if c == 3: return 1.0 / (1.0 + np.exp(-np.clip(a, -60, 60)))
    if c == 4:
        e = np.exp(a - a.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
    return a

def forward(net, x, trace=False):
    arch = net["arch"]; a = np.asarray(x, np.float32); states = []
    if arch in ("mlp", "res", "sparse", "v1chain"):
        outs = []
        for i, ly in enumerate(net["layers"]):
            pre = a @ ly["W"].T + ly["b"] * ly["g"]
            if arch == "res":
                for fl, fr, tl, tr in net["skips"]:
                    if tl == i and 0 <= fl < i and fl < len(outs):
                        s_ = outs[fl]; m = min(len(s_), len(pre)); pre[:m] = pre[:m] + s_[:m]
            a = _act(pre, ACTS.index(ly["act"])); outs.append(a)
            if trace: states.append(a.copy())
        return (a, states) if trace else a
    if arch == "local":
        for ly in net["layers"]:
            K = ly["K"]
            c = np.rint(np.arange(ly["n_out"]) * (ly["n_in"] - 1) / max(1, ly["n_out"] - 1)).astype(int)
            idx = np.clip(c[:, None] + np.arange(K)[None, :] - K // 2, 0, ly["n_in"] - 1)
            a = _act((a[idx] * ly["W"]).sum(1) + ly["b"] * ly["g"], ACTS.index(ly["act"]))
            if trace: states.append(a.copy())
        return (a, states) if trace else a
    if arch == "cnn1d":
        if a.ndim == 1: a = a[:, None]
        for ly in net["layers"]:
            K, st, pd = ly["K"], ly["stride"], ly["pad"]
            xp = np.pad(a, ((0, 0), (pd, pd)))
            To = 1 + (xp.shape[1] - K) // st
            Wf = ly["W"].reshape(ly["n_out"], -1)
            cols = np.stack([xp[:, t * st:t * st + K].reshape(-1) for t in range(To)], 1)
            a = _act(Wf @ cols + (ly["b"] * ly["g"])[:, None], ACTS.index(ly["act"]))
            if trace: states.append(a.ravel().copy())
        return (a, states) if trace else a
    if arch in ("rnn", "lstm"):
        for li, ly in enumerate(net["layers"]):
            T = ly["T"]
            if a.ndim == 1:   seq = np.tile(a, (T, 1))
            elif len(a) >= T: seq = a[:T]
            else:             seq = np.tile(a, (-(-T // len(a)), 1))[:T]
            if arch == "rnn":
                Wx, Wh = ly["W"][:, :ly["n_in"]], ly["W"][:, ly["n_in"]:]
                h = np.zeros(ly["n_out"], np.float32); hs = []
                for t in range(T):
                    h = _act(Wx @ seq[t] + Wh @ h + ly["b"] * ly["g"], ACTS.index(ly["act"]))
                    hs.append(h.copy())
            else:
                W3 = ly["W"].reshape(4, ly["n_out"], -1)
                Wx, Wh = W3[:, :, :ly["n_in"]], W3[:, :, ly["n_in"]:]
                bg = (ly["b"] * ly["g"]).reshape(4, ly["n_out"])
                h = np.zeros(ly["n_out"], np.float32); c = np.zeros(ly["n_out"], np.float32); hs = []
                for t in range(T):
                    z = np.einsum("koi,i->ko", Wx, seq[t]) + np.einsum("koi,i->ko", Wh, h)
                    i_ = _act(z[0] + bg[0], 3); f_ = _act(z[1] + bg[1] + 1.0, 3)
                    g_ = _act(z[2] + bg[2], 2); o_ = _act(z[3] + bg[3], 3)
                    c = f_ * c + g_; h = o_ * _act(c, 2); hs.append(h.copy())
            a = np.stack(hs)
            if li == len(net["layers"]) - 1 and trace: states = hs
        out = a[-1]
        return (out, states) if trace else out
    raise ValueError(arch)

def forward_batch(net, X):
    """One vectorized pass over the whole ensemble."""
    arch = net["arch"]
    Xs = [np.asarray(x, np.float32) for x in X]
    if arch in ("rnn", "lstm"):
        if len(net["layers"]) > 1 and len({l["T"] for l in net["layers"]}) > 1:
            return [forward(net, x) for x in Xs]
        A = np.stack([x if x.ndim == 2 else np.tile(x, (net["layers"][0]["T"], 1)) for x in Xs])
        B = A.shape[0]
        for ly in net["layers"]:
            T = ly["T"]
            if A.shape[1] < T: A = np.tile(A, (1, -(-T // A.shape[1]), 1))
            S_ = A[:, :T, :]
            if arch == "rnn":
                Wx, Wh = ly["W"][:, :ly["n_in"]], ly["W"][:, ly["n_in"]:]
                Zx = np.einsum("bti,oi->bot", S_, Wx); ai = ACTS.index(ly["act"])
                h = np.zeros((B, ly["n_out"]), np.float32); hs = []
                for t in range(T):
                    h = _act(Zx[:, t] + h @ Wh.T + (ly["b"] * ly["g"])[None, :], ai)
                    hs.append(h.copy())
            else:
                nsg = ly["n_out"]
                W3 = ly["W"].reshape(4, nsg, -1)
                Wx, Wh = W3[:, :, :ly["n_in"]], W3[:, :, ly["n_in"]:]
                bg = (ly["b"] * ly["g"]).reshape(4, nsg)
                Zx = np.einsum("bti,koi->bkot", S_, Wx)
                h = np.zeros((B, nsg), np.float32); c = np.zeros((B, nsg), np.float32); hs = []
                for t in range(T):
                    Z = Zx[:, :, t] + np.einsum("bi,koi->bko", h, Wh)
                    i_ = _act(Z[:, 0] + bg[0], 3); f_ = _act(Z[:, 1] + bg[1] + 1.0, 3)
                    g_ = _act(Z[:, 2] + bg[2], 2); o_ = _act(Z[:, 3] + bg[3], 3)
                    c = f_ * c + g_; h = o_ * _act(c, 2); hs.append(h.copy())
            A = np.stack(hs, 1)
        return [A[b, -1] for b in range(B)]
    if arch == "cnn1d":
        A = np.stack([x if x.ndim == 2 else x[None, :] for x in Xs]); B = A.shape[0]
        for ly in net["layers"]:
            K, st, pd = ly["K"], ly["stride"], ly["pad"]
            xp = np.pad(A, ((0, 0), (0, 0), (pd, pd)))
            To = 1 + (xp.shape[2] - K) // st
            Wf = np.asarray(ly["W"], np.float32).reshape(ly["n_out"], -1)
            cols = np.stack([xp[:, :, t * st:t * st + K].reshape(B, -1) for t in range(To)], 2)
            A = _act(np.einsum("bkt,ok->bot", cols, Wf) + (ly["b"] * ly["g"])[None, :, None],
                     ACTS.index(ly["act"]))
        return [A[b].ravel() for b in range(B)]
    A = np.stack([x.reshape(-1) for x in Xs]); outs = []
    for i, ly in enumerate(net["layers"]):
        if arch == "local":
            K = ly["K"]
            cidx = np.clip(np.rint(np.arange(ly["n_out"]) * (ly["n_in"] - 1)
                           / max(1, ly["n_out"] - 1)).astype(int)[:, None]
                           + np.arange(K)[None, :] - K // 2, 0, ly["n_in"] - 1)
            pre = (A[:, cidx] * np.asarray(ly["W"], np.float32)[None]).sum(2) + (ly["b"] * ly["g"])[None, :]
        else:
            pre = A @ np.asarray(ly["W"], np.float32).T + (ly["b"] * ly["g"])[None, :]
        if arch == "res":
            for fl, fr, tl, tr in net["skips"]:
                if tl == i and 0 <= fl < len(outs):
                    s_ = outs[fl]; m = min(s_.shape[1], pre.shape[1]); pre[:, :m] += s_[:, :m]
        A = _act(pre, ACTS.index(ly["act"])); outs.append(A)
    return [A[b] for b in range(A.shape[0])]

def make_stimulus(net, rng):
    l0 = net["layers"][0]; a = net["arch"]
    if a == "cnn1d":         return rng.uniform(-1, 1, (l0["n_in"], 8)).astype(np.float32)
    if a in ("rnn", "lstm"): return rng.uniform(-1, 1, (l0["T"], l0["n_in"])).astype(np.float32)
    return rng.uniform(-1, 1, l0["n_in"]).astype(np.float32)

def describe(net):
    a = net["arch"]
    if a == "v1chain":
        return f"v1chain {net['layers'][0]['n_in']}u x{len(net['layers'])}L"
    dims = [net["layers"][0]["n_in"]] + [l["n_out"] for l in net["layers"]]
    s = f"{a} " + "->".join(map(str, dims))
    if a == "local":         s += f" K{net['layers'][0]['K']}"
    if a == "cnn1d":
        l = net["layers"][0]; s += f" K{l['K']} s{l['stride']} p{l['pad']}"
    if a in ("rnn", "lstm"): s += f" T{net['layers'][0]['T']}"
    if a == "res" and net.get("skips"): s += f" +{len(net['skips'])}skip"
    hi = sum(1 for l in net["layers"] if l.get("flags", 0) & 1)
    return s + (f" [{hi}x16b]" if hi else "")

def param_count(net):
    return sum(fan_in(net["arch"], l) * param_rows(net["arch"], l) + param_rows(net["arch"], l)
               for l in net["layers"])

# ─────────────────────────── 3. information theory ───────────────────────────
def digamma(x):
    r = 0.0
    while x < 6: r -= 1.0 / x; x += 1.0
    v = 1.0 / x
    return r + math.log(x) - 0.5 * v - v ** 2 / 12.0 + v ** 4 / 120.0 - v ** 6 / 252.0

def logball(d): return 0.5 * d * math.log(math.pi) - math.lgamma(0.5 * d + 1.0)

def _z(Y):
    Y = np.asarray(Y, float)
    return (Y - Y.mean(0)) / np.maximum(Y.std(0), 1e-9)

def _d2(Z):
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return d2

def kl_entropy(Y, k=3):
    """Kozachenko–Leonenko k-NN entropy, bits."""
    Y = np.asarray(Y, float)
    if Y.ndim == 1: Y = Y[:, None]
    N, d = Y.shape
    if N < k + 2 or d == 0: return 0.0
    Z = _z(Y) + np.random.default_rng(0).normal(0, 1e-6, Y.shape)
    rk = np.maximum(np.sqrt(np.sort(_d2(Z), 1)[:, k - 1]), 1e-12)
    return (digamma(N) - digamma(k) + logball(d) + d * float(np.mean(np.log(rk)))) / LN2

def _lse(a):
    m = a.max(); return m + math.log(np.exp(a - m).sum() + 1e-300)

def infonce_mi(Y, cls, S):
    """Multi-positive InfoNCE lower bound on I(X;Y), bits (max log2 S)."""
    Z = _z(Y); d2 = _d2(Z)
    finite = d2[np.isfinite(d2)]
    med = float(np.median(finite)) if finite.size else 0.0
    if med <= EPS: return 0.0
    Sm = -d2 / (2 * med); cls = np.asarray(cls)
    lse_all = np.array([_lse(Sm[i]) for i in range(len(Z))])
    tot = 0.0
    for i in range(len(Z)):
        P = np.where(cls == cls[i])[0]
        tot += _lse(Sm[i, P]) - lse_all[i]
    return float(np.clip((math.log(S) - tot / len(Z)) / LN2, 0.0, math.log2(S)))

def build_ensemble(net, S, R, seed=0):
    """Maximin (farthest-point) stimulus design + jittered replicas."""
    rng = np.random.default_rng(seed)
    shape = np.shape(make_stimulus(net, rng))
    cand = rng.uniform(-1, 1, (6 * S,) + shape)
    flat = cand.reshape(len(cand), -1)
    sel = [int(rng.integers(len(cand)))]; dmin = np.full(len(cand), np.inf)
    for _ in range(S - 1):
        dmin = np.minimum(dmin, ((flat - flat[sel[-1]]) ** 2).sum(1))
        sel.append(int(np.argmax(dmin)))
    base = [cand[i] for i in sel]
    X = []
    for x in base:
        for _ in range(R):
            X.append(np.asarray(x, float) + rng.normal(0, 0.05, shape))
    return base, X, np.repeat(np.arange(S), R)

def channel_metrics_from_X(net, X, cls, S):
    Y = np.array([np.asarray(o, float).ravel() for o in forward_batch(net, X)])
    I = infonce_mi(Y, cls, S)
    H = max(kl_entropy(Y), 0.0)
    Rn = max(1, len(X) // S)
    Yb = Y.reshape(S, Rn, -1).mean(1)
    dY = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    same = cls[:, None] == cls[None, :]; np.fill_diagonal(same, False)
    d_intra = float(dY[same].mean()) if same.any() else 0.0
    dC = np.sqrt(((Yb[:, None, :] - Yb[None, :, :]) ** 2).sum(-1))
    off = ~np.eye(S, dtype=bool)
    d_inter = float(dC[off].mean()) if S > 1 else 0.0
    det = float(np.clip(1 - d_intra / max(d_inter, EPS), 0, 1)) if d_inter > 1e-9 else 0.0
    conf = 0.0
    if S > 1 and d_intra > 1e-9:
        dC2 = dC + np.eye(S) * 1e9
        conf = float(np.mean(dC2.min(1) < d_intra))
    compl = float(np.clip(I / H, 0, 1)) if H > 1e-9 else 0.0
    return dict(I=I, H=H, I_frac=I / max(math.log2(S), EPS), completeness=compl, det=det,
                nonconf=1 - conf, confusion=conf, d_intra=d_intra, d_inter=d_inter,
                Y=Y, cls=np.asarray(cls), Yb=Yb, S=S, R=Rn)

def channel_metrics(net, S=16, R=4, seed=0):
    _, X, cls = build_ensemble(net, S, R, seed)
    return channel_metrics_from_X(net, X, cls, S)

def code_stability(net, seed=0):
    """CCA alignment of response codes across two INDEPENDENT stimulus sets."""
    S2 = max(4, FULL["S"] // 2)
    ca = channel_metrics(net, S2, FULL["R"], seed)["Yb"]
    cb = channel_metrics(net, S2, FULL["R"], seed + 9876)["Yb"]
    M = _z(ca).T @ _z(cb) / len(ca)
    s = np.linalg.svd(M, compute_uv=False)
    return float(np.clip(s[0] if s.size else 0.0, 0, 1))

def cii_factors(c):
    f = [("I_frac", np.clip(c["I_frac"], 0, 1)), ("completeness", c["completeness"]),
         ("det", c["det"]), ("nonconf", np.clip(c["nonconf"], 0, 1))]
    if "stab" in c: f.append(("stab", np.clip(c["stab"], 0, 1)))
    return f

def cii(c):
    f = [v for _, v in cii_factors(c)]
    return float(np.prod(np.maximum(f, 0.0)) ** (1.0 / len(f)))

def tse(Y, max_sub=20, seed=1):
    """Tononi–Sporns–Edelman neural complexity + bipartition integration."""
    Y = np.asarray(Y, float); N, n = Y.shape
    if n < 2: return 0.0, np.zeros(0), 0.0, 0.0
    rng = np.random.default_rng(seed); Hfull = kl_entropy(Y); curve = []
    for k in range(1, n):
        if math.comb(n, k) <= max_sub: subs = list(itertools.combinations(range(n), k))
        else: subs = [tuple(rng.permutation(n)[:k]) for _ in range(max_sub)]
        curve.append(float(np.mean([kl_entropy(Y[:, list(s)]) for s in subs])) / k - Hfull / n)
    integ = 0.0
    for _ in range(16):
        p = rng.permutation(n); a, b = p[:n // 2], p[n // 2:]
        if len(a) and len(b): integ += kl_entropy(Y[:, a]) + kl_entropy(Y[:, b]) - Hfull
    seg = float(np.mean([kl_entropy(Y[:, [j]]) for j in range(n)]))
    return float(np.sum(curve)), np.array(curve), max(integ / 16, 0.0), seg

def layer_profile(net, S=16, R=4, seed=0):
    _, X, cls = build_ensemble(net, S, R, seed)
    traces = [forward(net, x, trace=True)[1] for x in X]
    prof = []
    if net["arch"] in ("rnn", "lstm"):
        T = min(len(t) for t in traces)
        for t in range(T):
            Yt = np.array([tr[t].ravel() for tr in traces])
            prof.append((infonce_mi(Yt, cls, S), kl_entropy(Yt)))
        return prof, True
    for l in range(len(traces[0])):
        Yl = np.array([tr[l].ravel() for tr in traces])
        prof.append((infonce_mi(Yl, cls, S), kl_entropy(Yl)))
    return prof, False

def genome_ctx(img):
    try: net = decode_any(img)
    except Exception: return None
    H, Wd = img.shape[:2]
    if net["arch"] == "v1chain":
        return dict(hdr_h=0, bands=[("all", 0, H, Wd)], W=Wd, H=H, L=0, arch="v1chain")
    r = net["meta"]["hdr_rows"]; bands = []
    if net["arch"] == "sparse":
        for ly in net["layers"]:
            nb = -(-ly["n_in"] // 32)
            bands.append(("adj", r, r + ly["n_out"], nb)); r += ly["n_out"]
    for ly in net["layers"]:
        nb = -(-fan_in(net["arch"], ly) // 4)
        rows = param_rows(net["arch"], ly) * (2 if ly.get("flags", 0) & 1 else 1)
        bands.append(("w", r, min(r + rows, H), nb)); r += rows
    for ly in net["layers"]:
        n = param_rows(net["arch"], ly)
        bands.append(("b", r, min(r + n, H), 1)); r += n
    if net["skips"]: bands.append(("skip", r, min(r + len(net["skips"]), H), 1))
    return dict(hdr_h=net["meta"]["hdr_rows"], bands=[b for b in bands if b[1] < b[2]],
                W=Wd, H=H, L=len(net["layers"]), arch=net["arch"])

def image_it(img):
    raw = np.ascontiguousarray(img, np.uint8).tobytes()
    zb = len(zlib.compress(raw, 9))
    _, cnt = np.unique(raw, return_counts=True); p = cnt / raw.size
    ctx = genome_ctx(img); bands, tot = [], 0.0
    if ctx:
        for kind, a, b, cols in ctx["bands"]:
            band = img[a:b, :cols].ravel()
            if band.size == 0: continue
            _, c2 = np.unique(band, return_counts=True); pb = c2 / band.size
            bits = float(-(pb * np.log2(pb)).sum()) * band.size
            bands.append((kind, b - a, bits)); tot += bits
    return dict(H_img=float(-(p * np.log2(p)).sum()), compress=raw.size / max(zb, 1),
                mdl_bits=8 * zb, bands=bands, self_bits=tot)

# ─────────────────────────── 4. classic search / sort ───────────────────────────
def linear_search_max(items):
    best = items[0]
    for it in items[1:]:
        if it[0] > best[0]: best = it
    return best

def merge_sort(items):
    if len(items) <= 1: return items[:]
    m = len(items) // 2
    L, R = merge_sort(items[:m]), merge_sort(items[m:])
    out, i, j = [], 0, 0
    while i < len(L) and j < len(R):
        if L[i][0] <= R[j][0]: out.append(L[i]); i += 1
        else:                  out.append(R[j]); j += 1
    return out + L[i:] + R[j:]

def quickselect_topk(items, k, seed=0):
    a = list(items); k = max(0, min(k, len(a)))
    if k == 0: return []
    rng = np.random.default_rng(seed); lo, hi = 0, len(a)
    while hi - lo > 1:
        piv = a[lo + int(rng.integers(hi - lo))][0]; seg = a[lo:hi]
        gt = [x for x in seg if x[0] > piv]; eq = [x for x in seg if x[0] == piv]
        lt = [x for x in seg if x[0] < piv]
        a[lo:hi] = gt + eq + lt; ngt = lo + len(gt)
        if ngt >= k:             hi = ngt
        elif ngt + len(eq) >= k: break
        else:                    lo = ngt + len(eq)
    return a[:k]

def binary_search_nearest(asc, t):
    if len(asc) == 1: return 0
    lo, hi = 0, len(asc) - 1
    if t <= asc[lo][0]: return lo
    if t >= asc[hi][0]: return hi
    while lo <= hi:
        m = (lo + hi) // 2
        if asc[m][0] < t: lo = m + 1
        else:             hi = m - 1
    lo = min(lo, len(asc) - 1)
    if lo == 0: return 0
    return lo if abs(asc[lo][0] - t) <= abs(asc[lo - 1][0] - t) else lo - 1

def interp_search_nearest(asc, t):
    if len(asc) <= 2:
        return int(np.argmin([abs(x[0] - t) for x in asc]))
    lo, hi = 0, len(asc) - 1
    if t <= asc[lo][0]: return lo
    if t >= asc[hi][0]: return hi
    while hi - lo > 1 and asc[lo][0] < t < asc[hi][0]:
        pos = lo + max(1, min(hi - lo - 1,
              int(round((t - asc[lo][0]) * (hi - lo) / max(asc[hi][0] - asc[lo][0], EPS)))))
        if asc[pos][0] < t: lo = pos
        else:               hi = pos
    return lo if abs(t - asc[lo][0]) <= abs(t - asc[hi][0]) else hi

# ─────────────────────────── 5. mutations + searchers ───────────────────────────
def _rc(b, W): return b // (4 * W), (b % (4 * W)) // 4, b % 4

def _generic_point(img, rng, sigma):
    y, x, c = int(rng.integers(img.shape[0])), int(rng.integers(img.shape[1])), int(rng.integers(4))
    img[y, x, c] = clip8(int(img[y, x, c]) + rng.normal(0, sigma))

def mutate_point(img, ctx, rng, sigma):
    wb = [b for b in ctx["bands"] if b[0] == "w"] if ctx else []
    if wb:
        k, a, b, cols = wb[int(rng.integers(len(wb)))]
        y = int(rng.integers(a, min(b, img.shape[0]))); x = int(rng.integers(cols)); c = int(rng.integers(4))
        img[y, x, c] = clip8(int(img[y, x, c]) + rng.normal(0, sigma)); return True
    _generic_point(img, rng, sigma); return True

def mutate_bias(img, ctx, rng, sigma):
    bb = [b for b in ctx["bands"] if b[0] == "b"] if ctx else []
    if bb:
        k, a, b, cols = bb[int(rng.integers(len(bb)))]
        y = int(rng.integers(a, min(b, img.shape[0]))); c = int(rng.integers(2))
        img[y, 0, c] = clip8(int(img[y, 0, c]) + rng.normal(0, sigma)); return True
    return mutate_point(img, ctx, rng, sigma)

def mutate_adj(img, ctx, rng):
    ab = [b for b in ctx["bands"] if b[0] == "adj"] if ctx else []
    if not ab: return mutate_point(img, ctx, rng, 12.0)
    k, a, b, cols = ab[int(rng.integers(len(ab)))]
    y = int(rng.integers(a, min(b, img.shape[0]))); x = int(rng.integers(cols)); c = int(rng.integers(4))
    img[y, x, c] = np.uint8(int(img[y, x, c]) ^ (1 << int(rng.integers(8)))); return True

def mutate_scale(img, ctx, rng):
    if not ctx or ctx["L"] <= 0: return False
    b0 = 8 + int(rng.integers(ctx["L"])) * 10 + 6
    r1, c1, k1 = _rc(b0, ctx["W"]); r2, c2, k2 = _rc(b0 + 1, ctx["W"])
    v = float(np.frombuffer(bytes([int(img[r1, c1, k1]), int(img[r2, c2, k2])]), "<f2")[0])
    if not np.isfinite(v) or v <= 0: return False
    nb = np.asarray(float(np.clip(v * math.exp(rng.normal(0, 0.35)), 1e-3, 60.0)), "<f2").tobytes()
    img[r1, c1, k1] = nb[0]; img[r2, c2, k2] = nb[1]; return True

def mutate_row(img, ctx, rng, sigma, donors=()):
    wb = [b for b in ctx["bands"] if b[0] == "w"] if ctx else []
    if not wb: return mutate_point(img, ctx, rng, sigma)
    k, a, b, cols = wb[int(rng.integers(len(wb)))]
    y = int(rng.integers(a, min(b, img.shape[0])))
    if donors and rng.random() < 0.7:
        d = donors[int(rng.integers(len(donors)))]
        if d.shape == img.shape: img[y] = d[y]; return True
    img[y, :cols] = np.clip(np.rint(img[y, :cols].astype(np.float32)
                            + rng.normal(0, sigma, img[y, :cols].shape)), 0, 255).astype(np.uint8)
    return True

def mutate(img, ctx, rng, sigma, donors=()):
    img = img.copy()
    if ctx is None:
        _generic_point(img, rng, sigma); return img
    u = rng.random()
    if   u < 0.40: mutate_point(img, ctx, rng, sigma)
    elif u < 0.58: mutate_bias(img, ctx, rng, sigma)
    elif u < 0.68: mutate_adj(img, ctx, rng)
    elif u < 0.78:
        if not mutate_scale(img, ctx, rng): mutate_point(img, ctx, rng, sigma)
    elif u < 0.92: mutate_row(img, ctx, rng, sigma, donors)
    else:                                   # generic data-area point mutation
        y = int(rng.integers(ctx["hdr_h"], img.shape[0])); x = int(rng.integers(img.shape[1]))
        c = int(rng.integers(4))
        img[y, x, c] = clip8(int(img[y, x, c]) + rng.normal(0, sigma))
    return img

def crossover(a, b, hdr_h, rng):
    if a.shape != b.shape: return a.copy()
    H = a.shape[0]; lo = min(hdr_h + 1, H - 1)
    cut = int(rng.integers(lo, H)) if H > lo else H
    return np.ascontiguousarray(np.concatenate([a[:cut], b[cut:]], axis=0))

# ---- XOR task (fitness = 1 - MSE; evolution TRAINS through the image) ----
XOR_X = [np.array([-0.9, -0.9], np.float32), np.array([-0.9, 0.9], np.float32),
         np.array([0.9, -0.9], np.float32), np.array([0.9, 0.9], np.float32)]
XOR_Y = np.array([0.0, 1.0, 1.0, 0.0], np.float32)

def task_fitness_net(net):
    Y = np.array([float(np.asarray(o).ravel()[0]) for o in forward_batch(net, XOR_X)])
    mse = float(np.mean((Y - XOR_Y) ** 2))
    acc = float(np.mean((Y > 0.5) == (XOR_Y > 0.5)))
    return float(np.clip(1 - mse, 0, 1)), mse, acc, Y

def task_fitness(img): return task_fitness_net(decode_any(img))

def score_img(img, S=None, R=None):
    if TASK == "xor":
        key = (hashlib.blake2b(np.ascontiguousarray(img, np.uint8).tobytes(),
                               digest_size=16).digest(), "xor")
    else:
        S = FIT["S"] if S is None else S; R = FIT["R"] if R is None else R
        key = (hashlib.blake2b(np.ascontiguousarray(img, np.uint8).tobytes(),
                               digest_size=16).digest(), S, R)
    if key in _CACHE:
        SCORE_STATS["hits"] += 1; return _CACHE[key]
    try:
        v = task_fitness(img)[0] if TASK == "xor" else cii(channel_metrics(decode_any(img), S=S, R=R))
    except Exception:
        v = 0.0
    _CACHE[key] = v; SCORE_STATS["evals"] += 1
    return v

def hill_climb(seed_img, evals, ctx, seed=0, sigma=12.0):
    rng = np.random.default_rng(seed)
    cur = seed_img.copy(); f = score_img(cur); best, bf = cur.copy(), f
    hist = [(0, f, f)]; since = 0
    for i in range(1, evals + 1):
        if since > max(60, evals // 6):                     # plateau restart
            cur = mutate(best, ctx, rng, 30.0); f = score_img(cur); since = 0
        cand = mutate(cur, ctx, rng, sigma); fc = score_img(cand)
        if fc > f: cur, f, since = cand, fc, 0
        else: since += 1
        if f > bf: best, bf = cur.copy(), f
        hist.append((i, f, bf))
    return best, bf, hist, evals + 1

def anneal(seed_img, evals, ctx, seed=0, T0=0.010, T1=1e-5):
    rng = np.random.default_rng(seed)
    cur = seed_img.copy(); f = score_img(cur); best, bf = cur.copy(), f
    hist = [(0, f, f)]
    for i in range(1, evals + 1):
        T = T0 * (T1 / T0) ** (i / max(1, evals - 1))
        cand = mutate(cur, ctx, rng, max(2.0, 30.0 * T / T0)); fc = score_img(cand)
        if fc > f or rng.random() < math.exp(min(0.0, (fc - f) / max(T, 1e-12))):
            cur, f = cand, fc
            if f > bf: best, bf = cur.copy(), f
        hist.append((i, f, bf))
    return best, bf, hist, evals + 1

def _tour(rng, F, pop):
    i, j = rng.integers(0, pop, 2)
    return int(i) if F[i] >= F[j] else int(j)

def evolve(seed_img, gens, pop, elite, ctx, seed=0, sigma0=24.0, patience=8):
    rng = np.random.default_rng(seed)
    P = [seed_img.copy() for _ in range(pop)]
    for i in range(1, pop): P[i] = mutate(P[i], ctx, rng, sigma0 * 2)
    F = [score_img(p) for p in P]; used = pop
    b = int(np.argmax(F)); gbest, gbf, noimp, sigma = P[b].copy(), F[b], 0, sigma0
    hdr_h = ctx["hdr_h"] if ctx else 0
    hist = [(0, used, F[b], float(np.mean(F)), float(np.std(F)), F[b])]
    for g in range(1, gens + 1):
        order = sorted(range(pop), key=lambda i: -F[i])
        donors = [P[i] for i in order[:max(2, pop // 4)]]
        newP = [P[i].copy() for i in order[:elite]]; newF = list(F[:elite])
        while len(newP) < pop:
            child = crossover(P[_tour(rng, F, pop)], P[_tour(rng, F, pop)], hdr_h, rng)
            child = mutate(child, ctx, rng, sigma, donors)
            if rng.random() < 0.15: child = mutate(child, ctx, rng, sigma, donors)
            newP.append(child); newF.append(score_img(child)); used += 1
        newP[-1] = mutate(P[order[0]], ctx, rng, sigma * 3, donors)
        newF[-1] = score_img(newP[-1]); used += 1
        P, F = newP, newF
        bi = int(np.argmax(F)); bsf = max(hist[-1][5], F[bi])
        hist.append((g, used, F[bi], float(np.mean(F)), float(np.std(F)), bsf))
        if F[0] > gbf + 1e-6: gbf, gbest, noimp = F[0], P[0].copy(), 0
        else: noimp += 1
        sigma = max(2.0, sigma * 0.97)
        if noimp >= patience:
            for i in range(elite, pop):
                P[i] = mutate(gbest, ctx, rng, 40.0); F[i] = score_img(P[i]); used += 1
            noimp, sigma = 0, sigma0
    return gbest, gbf, hist, used

def map_elites(seed_img, ctx, evals=350, seed=0, res=10, pop0=14):
    """Quality-Diversity: archive over (identifiability, determinism), quality = CII4."""
    rng = np.random.default_rng(seed)
    archive = {}; curve = []; used = [0]
    def ev(img):
        used[0] += 1
        try:
            c = channel_metrics(decode_any(img), S=FIT["S"], R=FIT["R"]); f = cii(c)
            bx = min(res - 1, max(0, int(c["I_frac"] * res)))
            by = min(res - 1, max(0, int(c["det"] * res)))
        except Exception:
            return None
        return (bx, by), f
    P = [seed_img] + [mutate(seed_img, ctx, rng, 20.0) for _ in range(pop0 - 1)]
    for img in P:
        r = ev(img)
        if r:
            cell, f = r
            if cell not in archive or f > archive[cell][0]: archive[cell] = (f, img)
    for _ in range(max(1, evals // pop0)):
        if not archive: break
        keys = list(archive)
        for _ in range(pop0):
            parent = archive[keys[int(rng.integers(len(keys)))]][1]
            child = mutate(parent, ctx, rng, 16.0)
            r = ev(child)
            if r:
                cell, f = r
                if cell not in archive or f > archive[cell][0]: archive[cell] = (f, child)
        curve.append((used[0], max(v[0] for v in archive.values())))
    if archive: bf, bimg = max(archive.values(), key=lambda t: t[0])
    else:       bf, bimg = score_img(seed_img), seed_img
    return bimg, bf, archive, curve, used[0]

def successive_halving(items, budgets=(4, 8, 16), keep=0.5):
    pool = list(items); ranked = []
    for si, S in enumerate(budgets):
        R = 3 if S < 16 else 4
        scored = sorted(((score_img(load_img(p), S, R), n, p) for n, p in pool), reverse=True)
        if si == len(budgets) - 1:
            ranked = scored; break
        k = max(2, int(round(len(scored) * keep)))
        pool = [(n, p) for _, n, p in scored[:k]]
        log(f"  halving S={S:2d}: scored {len(scored)}, kept top {k}")
    return ranked

# ─────────────────────────── 6. statistics ───────────────────────────
def proxy_score(net, S=16, seed=0):
    """OLD metric: pure response diversity (cannot tell chaos from information)."""
    rng = np.random.default_rng(seed)
    xs = [make_stimulus(net, rng) for _ in range(S)]
    O = np.array([np.asarray(o, float).ravel() for o in forward_batch(net, xs)])
    D = O.shape[1]
    if D == 0: return 0.0
    d2 = ((O[:, None, :] - O[None, :, :]) ** 2).sum(-1); iu = np.triu_indices(S, 1)
    return float(np.sqrt(d2[iu]).mean() / (2 * np.sqrt(D)))

def spread_per_stimulus(Yb):
    d = np.sqrt(((Yb[:, None, :] - Yb[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(d, axis=1)

def bootstrap_beats(Yb_w, Yb_r, B=400, seed=0):
    sw, sr = spread_per_stimulus(Yb_w), spread_per_stimulus(Yb_r)
    rng = np.random.default_rng(seed); S = len(sw); d = []
    for _ in range(B):
        idx = rng.integers(0, S, S)
        d.append(float(sw[idx].mean() - sr[idx].mean()))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.mean(d)), float(lo), float(hi), np.array(d)

def signflip_p(diffs, B=5000, seed=1):
    d = np.asarray(diffs, float); obs = d.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(B, len(d)))
    null = (signs * d).mean(1)
    return float((np.sum(null >= obs) + 1) / (B + 1))

def saliency(net, xs, eps=0.08):
    base = np.array([np.asarray(o, float).ravel() for o in forward_batch(net, xs)])
    n_in = net["layers"][0]["n_in"]; arch = net["arch"]; sal = np.zeros(n_in)
    for d in range(n_in):
        xs2 = []
        for x in xs:
            y = np.array(x, float, copy=True)
            if y.ndim == 1:       y[d] += eps
            elif arch == "cnn1d": y[d, :] += eps
            else:                 y[:, d] += eps
            xs2.append(y)
        Y2 = np.array([np.asarray(o, float).ravel() for o in forward_batch(net, xs2)])
        sal[d] = np.abs(Y2 - base).mean()
    return sal

def ablation(net, X, cls, S, mode="cii"):
    def zero(n2, i):
        n2["layers"][i]["W"] = np.zeros_like(n2["layers"][i]["W"])
        n2["layers"][i]["b"] = np.zeros_like(n2["layers"][i]["b"])
        return n2
    drops = []
    if mode == "task": base = task_fitness_net(net)[0]
    else:              base = cii(channel_metrics_from_X(net, X, cls, S))
    for i in range(len(net["layers"])):
        n2 = zero(copy.deepcopy(net), i)
        try:
            v = task_fitness_net(n2)[0] if mode == "task" \
                else cii(channel_metrics_from_X(n2, X, cls, S))
        except Exception: v = 0.0
        drops.append(1 - v / max(base, 1e-9))
    return drops, base

def seed_robustness(net, seeds=(0, 1, 2, 3, 4)):
    rows = []
    for sd in seeds:
        ch = channel_metrics(net, **FULL, seed=sd)
        ch["stab"] = code_stability(net, seed=sd)
        rows.append((ch["I"], ch["det"], ch["stab"], cii(ch)))
    arr = np.array(rows)
    return arr.mean(0), arr.std(0)

def factor_contributions(ch, med):
    base = cii(ch); out = {}
    for k, _ in cii_factors(ch):
        c2 = dict(ch); c2[k] = med.get(k, ch[k])
        out[k] = base - cii(c2)
    return out

def analyze(img, name="", want_profile=False):
    net = decode_any(img)
    ch = channel_metrics(net, **FULL, seed=0)
    ch["stab"] = code_stability(net)
    C, curve, integ, seg = tse(ch["Y"])
    it_ = image_it(img)
    prof_rec = layer_profile(net, **FULL, seed=0) if want_profile else (None, False)
    return dict(name=name, img=np.ascontiguousarray(img, np.uint8), net=net, ch=ch,
                cii=cii(ch), tse=C, tse_curve=curve, integ=integ, seg=seg,
                eff=ch["I"] / max(it_["mdl_bits"] / 8192.0, 1e-9), img_it=it_,
                profile=prof_rec, proxy=proxy_score(net, S=FULL["S"]))

# ─────────────────────────── 7. pool generation ───────────────────────────
def random_spec(arch, rng):
    acts = ["tanh", "relu", "sigmoid"]
    def mk(a, b, hi=False):
        s = float(rng.uniform(0.6, 1.4))
        return dict(n_in=a, n_out=b, act=acts[int(rng.integers(3))], scale=s,
                    flags=1 if hi else 0,
                    W=(rng.normal(0, 1.0 / math.sqrt(a), (b, a)) * s).astype(np.float32),
                    b=rng.normal(0, 0.1, b).astype(np.float32),
                    g=(1.0 + 0.05 * rng.normal(0, 1, b)).astype(np.float32))
    if arch in ("mlp", "res", "sparse"):
        dims = [6, 10, 8, 4]
        layers = [mk(a, b, hi=bool(rng.random() < 0.34)) for a, b in zip(dims, dims[1:])]
        spec = dict(arch=arch, layers=layers)
        if arch == "sparse":
            for ly in layers:
                ly["A"] = (rng.random((ly["n_out"], ly["n_in"])) < 0.6).astype(np.uint8)
        if arch == "res": spec["skips"] = [(0, 0, len(layers) - 1, 0)]
        return spec
    if arch == "local":
        dims = [8, 8, 8, 4]; layers = []
        for a, b in zip(dims, dims[1:]):
            ly = mk(a, b); ly["K"] = 3
            ly["W"] = (rng.normal(0, 1.0 / math.sqrt(3), (b, 3)) * ly["scale"]).astype(np.float32)
            layers.append(ly)
        return dict(arch="local", layers=layers)
    if arch == "cnn1d":
        layers = []
        for a, b in ((3, 4), (4, 2)):
            ly = mk(a, b); ly["K"], ly["stride"], ly["pad"] = 3, 1, 1
            ly["W"] = (rng.normal(0, 1.0 / math.sqrt(a * 3), (b, a, 3)) * ly["scale"]).astype(np.float32)
            layers.append(ly)
        return dict(arch="cnn1d", layers=layers)
    if arch == "rnn":
        ly = mk(4, 6); ly["T"] = 8
        ly["W"] = (np.concatenate([rng.normal(0, 1 / 2.0, (6, 4)),
                                   rng.normal(0, 1 / math.sqrt(6), (6, 6))], 1)
                   * ly["scale"]).astype(np.float32)
        return dict(arch="rnn", layers=[ly])
    if arch == "lstm":
        ni, no = 4, 5
        ly = mk(ni, no); ly["T"] = 8
        Wg = np.concatenate([rng.normal(0, 1 / 2.0, (no, ni)),
                             rng.normal(0, 1 / math.sqrt(no), (no, no))], 1) * ly["scale"]
        ly["W"] = np.tile(Wg, (4, 1)).astype(np.float32)
        ly["b"] = np.zeros(4 * no, np.float32); ly["g"] = np.ones(4 * no, np.float32)
        return dict(arch="lstm", layers=[ly])
    raise ValueError(arch)

def make_xor_spec(rng):
    layers = []
    for a, b, act in ((2, 4, "tanh"), (4, 1, "sigmoid")):
        s = float(rng.uniform(0.9, 1.6))
        layers.append(dict(n_in=a, n_out=b, act=act, scale=s, flags=0,
                           W=(rng.normal(0, 1.6 / math.sqrt(a), (b, a)) * s).astype(np.float32),
                           b=rng.normal(0, 0.4, b).astype(np.float32),
                           g=np.ones(b, np.float32)))
    return dict(arch="mlp", layers=layers)

def load_img(path):
    return np.ascontiguousarray(np.asarray(Image.open(path).convert("RGBA"), np.uint8))

def make_pool_if_needed(force=False):
    os.makedirs(POOL_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(POOL_DIR, "*.png"))
    if not force and len(existing) >= POOL_N: return
    if force:
        for p in existing: os.remove(p)
    rng = np.random.default_rng(POOL_SEED)
    log(f"generating {POOL_N} pool images in {POOL_DIR}/ ...")
    for i in range(POOL_N):
        if TASK == "xor":
            spec, name = make_xor_spec(rng), f"xor_{i:02d}.png"
        else:
            spec = random_spec(["mlp", "sparse", "local", "cnn1d", "rnn", "lstm", "res"][i % 7], rng)
            name = f"{spec['arch']}_{i:02d}.png"
        Image.fromarray(encode(spec), "RGBA").save(os.path.join(POOL_DIR, name))

def dedup_pool():
    seen, out = set(), []
    for p in sorted(glob.glob(os.path.join(POOL_DIR, "*.png"))):
        h = hashlib.blake2b(open(p, "rb").read(), digest_size=12).hexdigest()
        if h in seen: continue
        seen.add(h); out.append((os.path.basename(p), p))
    return out

def craft(kind, seed=5):
    rng = np.random.default_rng(seed); dims = [6, 10, 8, 4]; layers = []
    for a, b in zip(dims, dims[1:]):
        s = {"chaotic": 30.0, "constant": 1e-3}.get(kind)
        s = float(rng.uniform(0.6, 1.4)) if s is None else s
        layers.append(dict(n_in=a, n_out=b, act="tanh", scale=s, flags=0,
                           W=(rng.normal(0, 1.0 / math.sqrt(a), (b, a)) * s).astype(np.float32),
                           b=np.zeros(b, np.float32), g=np.ones(b, np.float32)))
    return encode(dict(arch="mlp", layers=layers))

# ─────────────────────────── 8. charts ───────────────────────────
def chart_dashboard(pool, med, champ, cname, path):
    fig = plt.figure(figsize=(17, 9), constrained_layout=True); gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    top = pool[:10]
    names = [A["name"] for A in top][::-1]; vals = [A["cii"] for A in top][::-1]
    ax.barh(names, vals, color=[GOLD if A is champ else STEEL for A in top][::-1])
    ax.axvline(float(np.median([A["cii"] for A in pool])), color=RED, ls="--", lw=1, label="pool median")
    ax.set_title("pool intelligence (CII5, S=16)"); ax.set_xlabel("CII")
    ax.legend(fontsize=7); ax.tick_params(labelsize=7); ax.grid(axis="x", alpha=.3)
    ax = fig.add_subplot(gs[0, 1])
    fl = ["I_frac", "completeness", "det", "nonconf", "stab"]
    labels = ["I/log$_2$S", "I/H", "determ.", "1-conf.", "stability"]
    v = [champ["ch"][k] for k in fl] + [champ["cii"]]
    mv = [float(np.median([A["ch"][k] for A in pool])) for k in fl] + \
         [float(np.median([A["cii"] for A in pool]))]
    ax.bar(labels + ["CII5"], v, color=[STEEL] * 5 + [GOLD])
    ax.scatter(range(6), mv, color=RED, marker="D", s=30, zorder=3, label="pool median")
    for i, vv in enumerate(v): ax.text(i, vv + .01, f"{vv:.2f}", ha="center", fontsize=7)
    ax.set_ylim(0, 1.15); ax.legend(fontsize=7); ax.grid(axis="y", alpha=.3)
    ax.set_title("champion factors (CII5 = geometric mean of first five)")
    ax = fig.add_subplot(gs[0, 2])
    Y = champ["ch"]["Y"]; Z = _z(Y)
    U, Sv, Vt = np.linalg.svd(Z, full_matrices=False)
    P = Z @ Vt[:2].T
    if P.shape[1] == 1: P = np.hstack([P, np.zeros((len(P), 1))])
    var = Sv[:2] ** 2 / max((Sv ** 2).sum(), 1e-9)
    cls = champ["ch"]["cls"]
    for m_ in range(champ["ch"]["S"]):
        p = P[cls == m_]
        ax.scatter(p[:, 0], p[:, 1], s=14, color=plt.cm.tab20(m_ % 20))
        ax.scatter(p[:, 0].mean(), p[:, 1].mean(), marker="X", s=70,
                   color=plt.cm.tab20(m_ % 20), edgecolor="k", linewidths=.5)
    ax.set_title(f"response space PCA ({var.sum()*100:.0f}% var)\n"
                 f"{champ['ch']['S']} tight clusters = high I(X;Y)", fontsize=9)
    ax.grid(alpha=.3); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax = fig.add_subplot(gs[1, 0])
    wI, wH = champ["ch"]["I"], champ["ch"]["H"]
    mI = float(np.median([A["ch"]["I"] for A in pool]))
    mH = float(np.median([A["ch"]["H"] for A in pool]))
    ax.bar(["champion", "pool median"], [wI, mI], color=GOLD, label="I(X;Y) about input")
    ax.bar(["champion", "pool median"], [max(wH - wI, 0), max(mH - mI, 0)],
           bottom=[wI, mI], color="#bbbbbb", label="residual (noise/idiosyncratic)")
    for i, (hh_, ii_) in enumerate([(wH, wI), (mH, mI)]):
        ax.text(i, hh_, f"H={hh_:.2f}b", ha="center", va="bottom", fontsize=8)
        ax.text(i, ii_ / 2, f"I={ii_:.2f}b", ha="center", fontsize=8)
    ax.set_ylabel("bits"); ax.legend(fontsize=7); ax.grid(axis="y", alpha=.3)
    ax.set_title("output entropy decomposition")
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter([math.log2(max(A["img_it"]["mdl_bits"], 2)) for A in pool],
               [A["ch"]["I"] for A in pool], color=STEEL, s=30, label="pool")
    ax.scatter([math.log2(max(champ["img_it"]["mdl_bits"], 2))], [wI], color=GOLD,
               s=200, marker="*", edgecolor="k", zorder=3, label="champion")
    ax.set_xlabel("log$_2$ MDL bits (zlib of image)"); ax.set_ylabel("transmitted I(X;Y) bits")
    ax.set_title(f"intelligence per byte: {champ['eff']:.2f} bits/KB")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax = fig.add_subplot(gs[1, 2], polar=True)
    src = dict(champ["ch"]); src["eff"] = champ["eff"]; src["tse"] = max(champ["tse"], 0)
    keys = ["I_frac", "completeness", "det", "nonconf", "stab", "eff"]
    labs = ["I(X;Y)\nidentifiability", "completeness\nI/H", "determinism",
            "non-\nconfusion", "code\nstability", "bits per KB\n(MDL)"]
    e = 0.05
    rat = [float(np.clip((src[k] + e) / (med[k] + e), 0, 2.5)) for k in keys]
    ang = np.linspace(0, 2 * np.pi, len(labs), endpoint=False).tolist(); ang += ang[:1]
    rr = rat + rat[:1]
    ax.plot(ang, rr, color=GOLD, lw=2); ax.fill(ang, rr, color=GOLD, alpha=.25)
    ax.plot(ang, [1] * len(ang), color=STEEL, ls="--", lw=1.5, label="pool median = 1x")
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylim(0, 2.5); ax.set_yticklabels([])
    for a_, vv in zip(ang[:-1], rat): ax.annotate(f"{vv:.1f}x", (a_, vv), fontsize=6, ha="center")
    ax.legend(fontsize=6, loc="lower right", framealpha=.3)
    ax.set_title("champion / pool median", pad=16)
    fig.suptitle(f"Why '{cname}'?  information-theoretic dashboard (CII5={champ['cii']:.3f})"
                 + ("  — task: XOR (diagnostic)" if TASK == "xor" else ""),
                 fontsize=14, weight="bold")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)

def chart_progress(curves, finals, gh, abl, mode_label, path):
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6), constrained_layout=True)
    for lbl, x, y in curves:
        if len(x): ax[0].plot(x, y, lw=1.8, label=lbl)
    ax[0].set_xlabel("evaluations"); ax[0].set_ylabel("best fitness so far")
    ax[0].set_title("search progress"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    names = [n for _, n, _ in finals]; vals = [s for s, _, _ in finals]
    ax[1].bar(names, vals, color=[GOLD if v_ == max(vals) else STEEL for v_ in vals])
    for i, v_ in enumerate(vals): ax[1].text(i, v_, f"{v_:.4f}", ha="center", va="bottom", fontsize=8)
    ax[1].set_title("final score"); ax[1].tick_params(axis="x", labelsize=7); ax[1].grid(axis="y", alpha=.3)
    g = [h[0] for h in gh]; m = np.array([h[3] for h in gh]); sd = np.array([h[4] for h in gh])
    ax[2].fill_between(g, m - sd, m + sd, alpha=.2, color=STEEL, label="mean ± σ")
    ax[2].plot(g, m, color=STEEL, lw=1.6, label="GA mean")
    ax[2].plot(g, [h[2] for h in gh], color=GOLD, lw=1.8, label="GA best gen")
    ax[2].set_xlabel("generation"); ax[2].set_title("GA population statistics")
    ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
    ax[3].bar([f"L{i}" for i in range(len(abl))], abl, color=STEEL)
    for i, v_ in enumerate(abl): ax[3].text(i, v_, f"-{v_*100:.0f}%", ha="center", va="bottom", fontsize=8)
    ax[3].set_title(f"layer reliance ({mode_label})"); ax[3].set_ylabel("drop")
    ax[3].grid(axis="y", alpha=.3)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)

def chart_plane(champ, pool, advs, path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    prof, rec = champ["profile"]
    xs = np.arange(1, len(prof) + 1)
    ax[0].plot(xs, [p[0] for p in prof], "-o", color=GOLD, lw=2, label="I(X;T)")
    ax[0].plot(xs, [max(p[1], 0) for p in prof], "--s", color=STEEL, lw=1.4, label="H(T)")
    ax[0].set_xlabel("timestep t" if rec else "layer l")
    ax[0].set_title("information plane" + (" - recurrent memory" if rec else ""))
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    px = [A["proxy"] for A in pool] + [A["proxy"] for A in advs]
    py = [A["cii"] for A in pool] + [A["cii"] for A in advs]
    ax.scatter(px, py, c=[STEEL] * len(pool) + [RED, RED, GOLD], s=40)
    for A in advs:
        ax.annotate(A["name"], (A["proxy"], A["cii"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.scatter([champ["proxy"]], [champ["cii"]], color=GOLD, marker="*", s=220,
               edgecolor="k", zorder=3, label="champion")
    ax.set_xlabel("OLD proxy: response diversity"); ax.set_ylabel("CII")
    ax.set_title("why the metric changed:\ndiversity cannot tell chaos from information")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    bands = champ["img_it"]["bands"]; tot = max(champ["img_it"]["self_bits"], 1e-9)
    names = [f"{k} ({r}r)" for k, r, _ in bands][:7]
    vals = [b / tot for _, _, b in bands][:7]
    ax[2].barh(names[::-1], vals[::-1], color=STEEL)
    ax[2].set_xlabel("share of image self-information")
    ax[2].set_title(f"image as code: H={champ['img_it']['H_img']:.2f} b/byte, "
                    f"zlib x{champ['img_it']['compress']:.2f}")
    ax[2].grid(axis="x", alpha=.3)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)

def chart_archive(archive, res, contrib, robust, champ_ch, path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    grid = np.zeros((res, res))
    for (bx, by), (f, _) in archive.items(): grid[by, bx] = f
    im = ax[0].imshow(grid, origin="lower", cmap="viridis", vmin=0, vmax=max(1e-9, grid.max()),
                      extent=[0, 1, 0, 1], aspect="auto")
    bx = min(res - 1, int(champ_ch["I_frac"] * res)); by = min(res - 1, int(champ_ch["det"] * res))
    ax[0].scatter([(bx + .5) / res], [(by + .5) / res], marker="*", s=220, color=GOLD, edgecolor="k")
    fig.colorbar(im, ax=ax[0], fraction=.046)
    ax[0].set_xlabel("behavior 1: I/log$_2$S"); ax[0].set_ylabel("behavior 2: determinism")
    ax[0].set_title(f"MAP-Elites archive: {len(archive)}/{res*res} cells "
                    f"({len(archive)/res/res*100:.0f}% coverage)")
    names = ["I/log$_2$S", "I/H", "determ.", "1-conf.", "stability"]
    vals = [contrib.get(k, 0.0) for k in ["I_frac", "completeness", "det", "nonconf", "stab"]]
    ax[1].bar(names, vals, color=STEEL)
    for i, v_ in enumerate(vals): ax[1].text(i, v_, f"{v_:.3f}", ha="center", va="bottom", fontsize=8)
    ax[1].set_title("factor contribution to the win\n(CII drop if factor = pool median)")
    ax[1].tick_params(axis="x", labelsize=7); ax[1].grid(axis="y", alpha=.3)
    mu, sd = robust
    rel_m = mu / np.maximum(np.abs(mu), 1e-9); rel_s = sd / np.maximum(np.abs(mu), 1e-9)
    ax[2].errorbar(range(4), rel_m, yerr=rel_s, fmt="o", color=STEEL, capsize=4)
    ax[2].axhline(1.0, color=RED, ls="--", lw=1)
    ax[2].set_xticks(range(4)); ax[2].set_xticklabels(["I", "det", "stab", "CII"], fontsize=8)
    ax[2].set_title("robustness across 5 stimulus seeds\n(relative, mean = 1)")
    ax[2].grid(alpha=.3)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)

# ─────────────────────────── 9. console pipeline (--cli) ───────────────────────────
def cli_main():
    global TASK, POOL_DIR, POOL_N, POOL_SEED
    p = argparse.ArgumentParser(description="NeuroPix console mode")
    p.add_argument("--regen", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--pool", type=int, default=12)
    p.add_argument("--evals", type=int, default=350)
    p.add_argument("--gens", type=int, default=22)
    p.add_argument("--pop", type=int, default=16)
    p.add_argument("--task", choices=["none", "xor"], default="none")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    TASK = args.task; POOL_N = max(4, args.pool); POOL_SEED = args.seed
    POOL_DIR = "image_pool_xor" if TASK == "xor" else "image_pool"
    evals, gens, pop = args.evals, args.gens, args.pop
    if args.quick: evals, gens, pop = evals // 2, max(6, gens // 2), max(8, pop // 2)
    t0 = time.time(); os.makedirs(REPORT, exist_ok=True)
    make_pool_if_needed(force=args.regen)
    items = dedup_pool()
    log(f"pool: {len(items)} unique images | task={TASK}\n")

    log("successive halving screening (budgets 4/8/16)...")
    ranked = successive_halving(items)
    pool = [analyze(load_img(p), n) for _, n, p in ranked]
    if TASK == "xor":
        for A in pool: A["fit"] = task_fitness(A["img"])[0]
        pool.sort(key=lambda A: -A["fit"])
    else:
        pool.sort(key=lambda A: -A["cii"])

    hdr = (f"{'CII5':>6} {'I':>6} {'H':>6} {'I/H':>5} {'det':>5} {'cnf':>5} {'stab':>5} "
           f"{'TSE':>6} {'b/KB':>7} {'old':>6}" + (f" {'fit':>6}" if TASK == "xor" else "") + "  image")
    log("\n" + hdr)
    for A in pool:
        c = A["ch"]
        row = (f"{A['cii']:6.3f} {c['I']:6.3f} {c['H']:6.3f} {c['completeness']:5.2f} "
               f"{c['det']:5.2f} {c['confusion']:5.2f} {c['stab']:5.2f} {A['tse']:6.2f} "
               f"{A['eff']:7.2f} {A['proxy']:6.3f}")
        if TASK == "xor": row += f" {A['fit']:6.3f}"
        log(row + f"  {A['name']} [{describe(A['net'])}]")
    cis = [A["cii"] for A in pool]
    log(f"pool CII5: mean={np.mean(cis):.3f} median={np.median(cis):.3f} σ={np.std(cis):.3f}")

    keyf = (lambda A: A["fit"]) if TASK == "xor" else (lambda A: A["cii"])
    best_lin = linear_search_max([(keyf(A), A["name"]) for A in pool])
    asc = merge_sort([(keyf(A), A["name"], i) for i, A in enumerate(pool)])
    top3 = merge_sort(quickselect_topk(asc, 3))[::-1]
    tgt = float(np.median([keyf(A) for A in pool]))
    i1 = interp_search_nearest(asc, tgt); i2 = binary_search_nearest(asc, tgt)
    log(f"\nlinear search max          : {best_lin[1]} ({best_lin[0]:.3f})")
    log(f"quickselect top-3          : {', '.join(n for _, n, _ in top3)}")
    log(f"target {tgt:.3f}: interpolation -> {asc[i1][1]} ({asc[i1][0]:.3f}) | "
        f"binary -> {asc[i2][1]} ({asc[i2][0]:.3f})")

    w = pool[0]; seed_img = w["img"]; ctx = genome_ctx(seed_img)
    log(f"\nimprovement search from '{w['name']}' (fitness budget {evals} evals each)...")
    tA = time.time(); hb, _, hh, he = hill_climb(seed_img, evals, ctx, seed=args.seed * 10 + 1); tA = time.time() - tA
    tB = time.time(); ab_, _, ah, ae = anneal(seed_img, evals, ctx, seed=args.seed * 10 + 2); tB = time.time() - tB
    tC = time.time(); gb, _, gh, ge = evolve(seed_img, gens, pop, 4, ctx, seed=args.seed * 10 + 3); tC = time.time() - tC
    results = [("hill-climb", hb, he, tA), ("annealing", ab_, ae, tB), ("GA mu+lambda", gb, ge, tC)]
    mc = []
    if TASK != "xor":
        tb = time.time(); mb, _, archive, mc, me = map_elites(seed_img, ctx, evals, seed=args.seed * 10 + 4)
        results.append(("MAP-Elites", mb, me, time.time() - tb))
    finals = sorted([(score_img(img, **FULL), nm, img) for nm, img, _, _ in results],
                    key=lambda t: -t[0])
    for nm, img, used, tt in results:
        log(f"  {nm:<13} {score_img(img, **FULL):.4f}  ({used} evals, {tt:.1f}s)")
    fs, cname, cimg = finals[0]

    champ = analyze(cimg, cname, want_profile=True)
    cimg = np.ascontiguousarray(cimg, np.uint8)
    rt = bool(np.array_equal(encode(decode(cimg)), cimg))
    Image.fromarray(cimg, "RGBA").save(f"{REPORT}/smartest.png")

    med = dict(I_frac=float(np.median([A["ch"]["I_frac"] for A in pool])),
               completeness=float(np.median([A["ch"]["completeness"] for A in pool])),
               det=float(np.median([A["ch"]["det"] for A in pool])),
               nonconf=float(np.median([A["ch"]["nonconf"] for A in pool])),
               stab=float(np.median([A["ch"]["stab"] for A in pool])),
               eff=float(np.median([A["eff"] for A in pool])))
    contrib = factor_contributions(champ["ch"], med)
    robust = seed_robustness(champ["net"])
    mode = "task" if TASK == "xor" else "cii"
    if len(pool) > 1:
        m_b, lo_b, hi_b, diffs = bootstrap_beats(pool[0]["ch"]["Yb"], pool[1]["ch"]["Yb"])
        pval = signflip_p(diffs)
        d_eff = float(diffs.mean() / max(diffs.std(ddof=1), 1e-12))
    else:
        m_b, lo_b, hi_b, pval, d_eff = 0.0, 0.0, 0.0, 1.0, 0.0
    base_stim, X, cls = build_ensemble(champ["net"], **FULL, seed=0)
    abl, base_m = ablation(champ["net"], X, cls, FULL["S"], mode=mode)
    sal = saliency(champ["net"], base_stim)
    sal_top = np.argsort(sal)[::-1][:2]
    advs = [analyze(craft(k), k) for k in ("constant", "chaotic", "healthy")]

    curves = [("hill-climb", [h[0] for h in hh], np.maximum.accumulate([h[2] for h in hh])),
              ("annealing",  [h[0] for h in ah], np.maximum.accumulate([h[2] for h in ah])),
              ("GA",         [h[1] for h in gh], [h[5] for h in gh])]
    if TASK != "xor": curves.append(("MAP-Elites", [u for u, _ in mc], [b for _, b in mc]))
    chart_dashboard(pool, med, champ, cname, f"{REPORT}/01_dashboard.png")
    chart_progress(curves, finals, gh, abl,
                   "fitness drop when layer zeroed" if mode == "task"
                   else "CII drop when layer zeroed",
                   f"{REPORT}/02_search_progress.png")
    chart_plane(champ, pool, advs, f"{REPORT}/03_information_plane.png")
    if TASK != "xor":
        chart_archive(archive, 10, contrib, robust, champ["ch"], f"{REPORT}/04_archive_map.png")

    c = champ["ch"]
    fl = [("identifiability I/log2(S)", "I_frac"), ("completeness I/H", "completeness"),
          ("determinism", "det"), ("non-confusion", "nonconf"), ("code stability", "stab")]
    ratios = sorted(((lab, c[k] / max(med[k], 1e-9)) for lab, k in fl), key=lambda t: -t[1])
    topc = max(contrib.items(), key=lambda kv: kv[1])
    clab = {"I_frac": "identifiability", "completeness": "completeness", "det": "determinism",
            "nonconf": "non-confusion", "stab": "stability"}[topc[0]]
    log("\n" + "=" * 70)
    log(f"WHY '{cname}' WAS CHOSEN (champion after search; score {fs:.3f})")
    log("=" * 70)
    log(f"I(X;Y) = {c['I']:.3f} of {math.log2(FULL['S']):.0f} possible bits "
        f"(InfoNCE lower bound, S={FULL['S']} x R={FULL['R']} replicas, maximin stimuli)")
    log(f"H(Y) = {c['H']:.3f} bits (KL k-NN entropy) -> I/H = {c['completeness']:.2f} | "
        f"det = {c['det']:.2f} | confusion = {c['confusion']:.2f} | stability = {c['stab']:.2f}")
    log(f"TSE = {champ['tse']:.2f} bits | integration = {champ['integ']:.2f} | "
        f"segregation = {champ['seg']:.2f} | MDL eff = {champ['eff']:.2f} bits/KB")
    log(f"vs runner-up '{pool[1]['name']}': {m_b:+.4f}, 95% CI [{lo_b:+.4f}, {hi_b:+.4f}], "
        f"sign-flip p = {pval:.4f}, Cohen's d = {d_eff:.2f}")
    log(f"factor contribution: {clab} +{topc[1]:.3f} CII (largest)")
    log(f"robustness (5 stimulus seeds): I {robust[0][0]:.3f}±{robust[1][0]:.3f}, "
        f"det {robust[0][1]:.3f}±{robust[1][1]:.3f}, stab {robust[0][2]:.3f}±{robust[1][2]:.3f}, "
        f"CII {robust[0][3]:.3f}±{robust[1][3]:.3f}")
    log(f"layer reliance: " + ", ".join(f"L{i} -{d*100:.0f}%" for i, d in enumerate(abl)) +
        f" | top inputs: in{sal_top[0]}, in{sal_top[1]} | round-trip bit-exact: {rt}")
    if TASK == "xor":
        _, mse, acc, Y = task_fitness(cimg)
        log("XOR truth table (trained through the image by evolution):")
        for x, yv, yp in zip(XOR_X, XOR_Y, Y):
            log(f"  {x[0]:+.1f},{x[1]:+.1f} -> target {yv:.0f}  pred {yp:.3f}")
        log(f"  MSE = {mse:.4f}  accuracy = {acc*100:.0f}%")
    log("\nadversarial check (old proxy vs CII):")
    for A in advs:
        log(f"  {A['name']:<9} old-proxy {A['proxy']:.3f}  CII {A['cii']:.3f}  "
            f"I {A['ch']['I']:.2f}  H {A['ch']['H']:.2f}  det {A['ch']['det']:.2f}")
    log("\nVERDICT: chosen because its " +
        " and ".join(f"{lab} is {r_:.2f}x the pool median" for lab, r_ in ratios[:2]) +
        f"; the win is carried most by {clab} (+{topc[1]:.3f} CII), it generalizes its code to\n"
        f"fresh stimuli (stability {c['stab']:.2f}±{robust[1][2]:.2f}), and no other image "
        f"transmits more information ABOUT ITS INPUTS.")
    log("Formulas: KL k-NN entropy; InfoNCE (Oord'18, von Kugelgen'21); EI decomposition "
        "(Hoel'13);\nTSE (Tononi-Sporns-Edelman'94); MDL (Rissanen); CCA stability. "
        "CII5 = (I_frac*I/H*det*(1-conf)*stab)^(1/5).")
    log("=" * 70)
    log(f"\nevals={SCORE_STATS['evals']} cache_hits={SCORE_STATS['hits']} "
        f"wall={time.time() - t0:.1f}s")
    saved = ["01_dashboard.png", "02_search_progress.png", "03_information_plane.png"]
    if TASK != "xor": saved.append("04_archive_map.png")
    log("saved:", *[f"{REPORT}/{s}" for s in saved], f"{REPORT}/smartest.png")

# ─────────────────────────── 10. GUI (PySide6) ───────────────────────────
CANCEL = threading.Event()
STATE = {"champ_img": None, "champ_src": None}
CANCELLED_MSG = "cancelled by user"

class Cancelled(Exception):
    pass

class Patch:
    """Wrap engine scoring/log globals with cancel checks, eval counting, Qt emits."""
    def __init__(self, jobs):
        self.jobs = jobs; self.evals = 0; self.best = -1e9; self.total = 1
        self._orig = {}
    def set_total(self, n):
        self.total = max(1, int(n))
        try: self.jobs.total.emit(self.total)
        except Exception: pass
    def __enter__(self):
        G = globals()
        self._orig = {"log": log, "score": score_img, "cm": channel_metrics}
        G["log"] = lambda m: self.jobs.logline.emit(str(m))
        def score(img, S=None, R=None):
            self._check(); self.evals += 1
            v = self._orig["score"](img, S, R)
            self.best = max(self.best, v)
            if self.evals % 5 == 0: self.jobs.tick.emit(self.evals, self.best)
            return v
        def cm(net, S=16, R=4, seed=0):
            self._check(); self.evals += 1
            return self._orig["cm"](net, S, R, seed)
        G["score_img"] = score
        G["channel_metrics"] = cm
        return self
    def _check(self):
        if CANCEL.is_set(): raise Cancelled()
    def __exit__(self, *exc):
        G = globals()
        G["score_img"] = self._orig["score"]
        G["channel_metrics"] = self._orig["cm"]
        G["log"] = self._orig["log"]
        return False

def _row_of(A):
    c = A["ch"]
    return dict(name=A["name"], cii=A["cii"], I=c["I"], H=c["H"], IH=c["completeness"],
                det=c["det"], stab=c.get("stab", 0.0), tse=A["tse"], eff=A["eff"],
                proxy=A["proxy"], arch=describe(A["net"]),
                fit=A.get("fit", None), img=A["img"])

def _details_text(A):
    c = A["ch"]
    s = [f"image : {A['name']}",
         f"arch  : {describe(A['net'])}   params={param_count(A['net'])}",
         f"CII5  : {A['cii']:.4f}   (I/log2S={c['I_frac']:.2f}, I/H={c['completeness']:.2f},",
         f"        det={c['det']:.2f}, 1-conf={c['nonconf']:.2f}, stab={c['stab']:.2f})",
         f"I(X;Y)={c['I']:.3f} bits   H(Y)={c['H']:.2f} bits   TSE={A['tse']:.2f} bits",
         f"MDL   : zlib x{A['img_it']['compress']:.2f} -> {A['eff']:.2f} bits/KB",
         f"old diversity proxy = {A['proxy']:.3f}  (cannot tell chaos from information)"]
    is_xorish = (TASK == "xor") or (
        A["net"]["layers"] and A["net"]["layers"][0]["n_in"] == 2
        and A["net"]["layers"][-1]["n_out"] == 1)
    if is_xorish:
        try:
            _, mse, acc, Y = task_fitness(A["img"])
            s.append(f"XOR: MSE={mse:.4f} acc={acc*100:.0f}%  preds="
                     + " ".join(f"{y:.2f}" for y in Y))
        except Exception:
            pass
    return "\n".join(s)

def _quick_text(label, img):
    A = analyze(img, label, want_profile=False)
    return f"{label}: CII {A['cii']:.4f}  I {A['ch']['I']:.3f}b  det {A['ch']['det']:.2f}"

def _screen_pool(jobs, task):
    items = dedup_pool()
    if not items: raise RuntimeError("pool is empty — generate it first")
    ranked = successive_halving(items)
    pool = [analyze(load_img(pth), nm) for _, nm, pth in ranked]
    if task == "xor":
        for A in pool: A["fit"] = task_fitness(A["img"])[0]
        pool.sort(key=lambda A: -A["fit"])
    else:
        pool.sort(key=lambda A: -A["cii"])
    return items, pool

def _job_generate(jobs, n, seed, task):
    global TASK, POOL_DIR, POOL_N, POOL_SEED
    jobs.phase.emit("generate pool")
    TASK, POOL_N, POOL_SEED = task, max(4, n), seed
    POOL_DIR = "image_pool_xor" if task == "xor" else "image_pool"
    with Patch(jobs) as p:
        p.set_total(max(n, 1))
        make_pool_if_needed(force=True)
    jobs.done.emit("generate", {"dir": POOL_DIR, "task": task})

def _job_open(jobs, path):
    jobs.phase.emit("open image")
    img = load_img(path); shown = img
    net = decode_any(img)                       # raises -> failed signal
    img2 = img
    if net["arch"] == "v1chain" and (img.shape[0] > 48 or img.shape[1] > 48):
        h2 = max(2, min(48, img.shape[0])); w2 = max(2, min(48, img.shape[1]))
        nearest = getattr(Image, "Resampling", Image).NEAREST
        img2 = np.ascontiguousarray(
            Image.fromarray(np.ascontiguousarray(img, "RGBA")).resize((w2, h2), nearest))
        jobs.logline.emit(f"photo -> v1chain too big; scoring a {w2}x{h2} "
                          f"downscale (display keeps original size)")
    with Patch(jobs) as p:
        p.set_total(1)
        A = analyze(img2, os.path.basename(path), want_profile=False)
    A["img"] = shown
    jobs.champion.emit(shown)
    jobs.rows.emit([_row_of(A)])
    jobs.text.emit(_details_text(A))
    jobs.done.emit("open", {"name": A["name"]})

def _job_score_pool(jobs, task):
    global TASK
    jobs.phase.emit("score pool")
    TASK = task
    with Patch(jobs) as p:
        p.set_total(max(4 * len(dedup_pool()), 8))
        _, pool = _screen_pool(jobs, task)
    jobs.rows.emit([_row_of(A) for A in pool])
    jobs.champion.emit(pool[0]["img"])
    jobs.text.emit(_details_text(pool[0]))
    jobs.done.emit("score", {})

def _job_find(jobs, params):
    global TASK
    t0 = time.time()
    evals, gens, pop, task, seed = (params["evals"], params["gens"],
                                    params["pop"], params["task"], params["seed"])
    TASK = task
    with Patch(jobs) as p:
        p.set_total(len(dedup_pool()) * 4 + 4 * evals + pop * gens + 200)
        jobs.phase.emit("screening (successive halving)")
        _, pool = _screen_pool(jobs, task)
        jobs.rows.emit([_row_of(A) for A in pool])

        w = pool[0]; seed_img = w["img"]; ctx = genome_ctx(seed_img)
        jobs.phase.emit("hill-climb")
        hb, _, hh, _ = hill_climb(seed_img, evals, ctx, seed=seed * 10 + 1)
        jobs.champion.emit(hb)
        jobs.phase.emit("annealing")
        ab, _, ah, _ = anneal(seed_img, evals, ctx, seed=seed * 10 + 2)
        jobs.champion.emit(ab)
        jobs.phase.emit("GA mu+lambda")
        gb, _, gh, _ = evolve(seed_img, gens, pop, 4, ctx, seed=seed * 10 + 3)
        jobs.champion.emit(gb)
        results = [("hill-climb", hb, hh), ("annealing", ab, ah), ("GA", gb, gh)]
        archive, mc = None, []
        if task != "xor":
            jobs.phase.emit("MAP-Elites")
            mb, _, archive, mc, _ = map_elites(seed_img, ctx, evals=evals, seed=seed * 10 + 4)
            jobs.champion.emit(mb)
            results.append(("MAP-Elites", mb, [(u, b) for u, b in mc]))

        finals = sorted([(score_img(img, **FULL), nm, img) for nm, img, _ in results],
                        key=lambda t: -t[0])
        for s_, nm, _ in finals:
            jobs.logline.emit(f"  {nm:<12} final score {s_:.4f}")
        fs, cname, cimg = finals[0]

        jobs.phase.emit("champion analysis")
        champ = analyze(cimg, cname, want_profile=True)
        cimg = np.ascontiguousarray(cimg, np.uint8)
        rt = bool(np.array_equal(encode(decode(cimg)), cimg))
        os.makedirs(REPORT, exist_ok=True)
        Image.fromarray(cimg, "RGBA").save(f"{REPORT}/smartest.png")

        med = dict(I_frac=float(np.median([A["ch"]["I_frac"] for A in pool])),
                   completeness=float(np.median([A["ch"]["completeness"] for A in pool])),
                   det=float(np.median([A["ch"]["det"] for A in pool])),
                   nonconf=float(np.median([A["ch"]["nonconf"] for A in pool])),
                   stab=float(np.median([A["ch"]["stab"] for A in pool])),
                   eff=float(np.median([A["eff"] for A in pool])))
        contrib = factor_contributions(champ["ch"], med)
        robust = seed_robustness(champ["net"])
        mode = "task" if task == "xor" else "cii"
        base_stim, X, cls = build_ensemble(champ["net"], **FULL, seed=0)
        abl, _ = ablation(champ["net"], X, cls, FULL["S"], mode=mode)
        sal = saliency(champ["net"], base_stim)
        sal_top = np.argsort(sal)[::-1][:2]
        advs = [analyze(craft(k), k) for k in ("constant", "chaotic", "healthy")]

        jobs.phase.emit("rendering report charts")
        curves = [("hill-climb", [h[0] for h in hh], np.maximum.accumulate([h[2] for h in hh])),
                  ("annealing",  [h[0] for h in ah], np.maximum.accumulate([h[2] for h in ah])),
                  ("GA",         [h[1] for h in gh], [h[5] for h in gh])]
        if task != "xor":
            curves.append(("MAP-Elites", [u for u, _ in mc], [b for _, b in mc]))
        chart_dashboard(pool, med, champ, cname, f"{REPORT}/01_dashboard.png")
        chart_progress(curves, finals, gh, abl,
                       "fitness drop when layer zeroed" if mode == "task"
                       else "CII drop when layer zeroed",
                       f"{REPORT}/02_search_progress.png")
        chart_plane(champ, pool, advs, f"{REPORT}/03_information_plane.png")
        if task != "xor":
            chart_archive(archive, 10, contrib, robust, champ["ch"],
                          f"{REPORT}/04_archive_map.png")

    c = champ["ch"]
    fl = [("identifiability I/log2(S)", "I_frac"), ("completeness I/H", "completeness"),
          ("determinism", "det"), ("non-confusion", "nonconf"), ("code stability", "stab")]
    ratios = sorted(((lab, c[k] / max(med[k], 1e-9)) for lab, k in fl), key=lambda t: -t[1])
    topc = max(contrib.items(), key=lambda kv: kv[1])
    clab = {"I_frac": "identifiability", "completeness": "completeness", "det": "determinism",
            "nonconf": "non-confusion", "stab": "stability"}[topc[0]]
    lines = [
        f"CHAMPION: {cname}  score {fs:.4f}  [{describe(champ['net'])}]",
        f"I(X;Y)={c['I']:.3f}/{math.log2(FULL['S']):.0f} bits  H={c['H']:.2f}  "
        f"I/H={c['completeness']:.2f}  det={c['det']:.2f}  stab={c['stab']:.2f}",
        f"TSE={champ['tse']:.2f}b  MDL eff={champ['eff']:.2f} bits/KB  round-trip={rt}",
        "vs median: " + ", ".join(f"{lab} {r_:.2f}x" for lab, r_ in ratios[:3]),
        f"win carried by {clab} (+{topc[1]:.3f}); relies most on layer {int(np.argmax(abl))}",
        f"robustness (5 seeds): CII {robust[0][3]:.3f}±{robust[1][3]:.3f}, "
        f"stab {robust[0][2]:.3f}±{robust[1][2]:.3f}",
        f"top inputs: in{sal_top[0]}, in{sal_top[1]}",
    ]
    if task == "xor":
        _, mse, acc, Y = task_fitness(cimg)
        lines.append("XOR truth table (trained through the image):")
        for x, yv, yp in zip(XOR_X, XOR_Y, Y):
            lines.append(f"  {x[0]:+.1f},{x[1]:+.1f} -> {yv:.0f}  pred {yp:.3f}")
        lines.append(f"  MSE={mse:.4f}  accuracy={acc*100:.0f}%")
    lines.append(f"[wall {time.time()-t0:.1f}s | engine evals={SCORE_STATS['evals']} "
                 f"cache_hits={SCORE_STATS['hits']}]")
    jobs.text.emit("\n".join(lines))
    payload = dict(champ=cimg, name=cname, finals=finals,
                   report=["01_dashboard.png", "02_search_progress.png",
                           "03_information_plane.png"]
                          + ([] if task == "xor" else ["04_archive_map.png"]))
    jobs.done.emit("find", payload)

def _job_single_search(jobs, algo, params):
    global TASK
    evals, gens, pop, task, seed = (params["evals"], params["gens"],
                                    params["pop"], params["task"], params["seed"])
    TASK = task
    start = STATE.get("champ_img")
    if start is None:
        items = dedup_pool()
        if not items: raise RuntimeError("no image to start from")
        start = load_img(items[0][1])
    with Patch(jobs) as p:
        p.set_total(evals if algo != "GA" else gens * pop)
        jobs.phase.emit(algo)
        ctx = genome_ctx(start)
        if algo == "hill-climb":
            img, f, hh, _ = hill_climb(start, evals, ctx, seed=seed + 1)
            curve = ([h[0] for h in hh], np.maximum.accumulate([h[2] for h in hh]))
        elif algo == "annealing":
            img, f, ah, _ = anneal(start, evals, ctx, seed=seed + 2)
            curve = ([h[0] for h in ah], np.maximum.accumulate([h[2] for h in ah]))
        elif algo == "GA":
            img, f, gh, _ = evolve(start, gens, pop, 4, ctx, seed=seed + 3)
            curve = ([h[1] for h in gh], [h[5] for h in gh])
        else:
            img, f, archive, mc, _ = map_elites(start, ctx, evals=evals, seed=seed + 4)
            curve = ([u for u, _ in mc], [b for _, b in mc])
    final = score_img(img, **FULL)
    A = analyze(img, algo, want_profile=False)
    jobs.champion.emit(img)
    jobs.rows.emit([_row_of(A)])
    jobs.text.emit(_quick_text(algo, img) +
                   f"\nfinal score (S=16): {final:.4f}  [{describe(A['net'])}]")
    jobs.done.emit("search", {"curve": (algo,) + curve, "img": img})

def run_gui():
    from PySide6.QtCore import Qt, QObject, Signal, QRunnable, QThreadPool
    from PySide6.QtGui import QImage, QPixmap, QAction
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
        QHBoxLayout, QFormLayout, QSpinBox, QComboBox, QGroupBox,
        QTableWidget, QTableWidgetItem, QPlainTextEdit, QProgressBar, QFileDialog,
        QMessageBox, QTabWidget, QSplitter, QHeaderView)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    def np_to_qimage(arr):
        arr = np.ascontiguousarray(arr, np.uint8)
        h, w, _ = arr.shape
        return QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()

    def np_to_pixmap(arr, box=224):
        pm = QPixmap.fromImage(np_to_qimage(arr))
        return pm.scaled(box, box, Qt.KeepAspectRatio, Qt.FastTransformation)

    class Jobs(QObject):
        tick     = Signal(int, float)
        phase    = Signal(str)
        total    = Signal(int)
        logline  = Signal(str)
        rows     = Signal(object)
        champion = Signal(object)
        text     = Signal(str)
        done     = Signal(str, object)
        failed   = Signal(str)

    class Task(QRunnable):
        def __init__(self, jobs, fn, *a):
            super().__init__()
            self.jobs, self.fn, self.a = jobs, fn, a
        def run(self):
            try:
                self.fn(self.jobs, *self.a)
            except Cancelled:
                self.jobs.failed.emit(CANCELLED_MSG)
            except Exception as e:
                import traceback
                self.jobs.failed.emit(f"{e}\n{traceback.format_exc(limit=6)}")

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("NeuroPix — RGBA image ⇄ neural network "
                                "(information-theoretic intelligence search)")
            self.resize(1380, 860)
            self.jobs = Jobs()
            self.jobs.tick.connect(self.on_tick)
            self.jobs.phase.connect(self.on_phase)
            self.jobs.total.connect(lambda v: setattr(self, "_total_v", max(1, v)))
            self.jobs.logline.connect(self.on_log)
            self.jobs.rows.connect(self.on_rows)
            self.jobs.champion.connect(self.on_champion)
            self.jobs.text.connect(self.on_text)
            self.jobs.done.connect(self.on_done)
            self.jobs.failed.connect(self.on_failed)
            self.series = {}; self._byname = {}; self._total_v = 100
            self._build_menu(); self._build_ui()

        def _build_menu(self):
            m = self.menuBar().addMenu("&File")
            act_open = QAction("Open PNG…", self); act_open.triggered.connect(self.open_png)
            act_save = QAction("Save Champion…", self); act_save.triggered.connect(self.save_champ)
            act_quit = QAction("Quit", self); act_quit.triggered.connect(self.close)
            m.addAction(act_open); m.addAction(act_save); m.addSeparator(); m.addAction(act_quit)
            h = self.menuBar().addMenu("&Help")
            about = QAction("About", self)
            about.triggered.connect(lambda: QMessageBox.information(
                self, "About NeuroPix",
                "NeuroPix — an RGBA image IS a neural network (NNIMG v2).\n"
                "Intelligence = CII5 information composite:\n"
                "  (I/log2S · I/H · determinism · (1-confusion) · stability)^(1/5)\n"
                "Search: successive halving, hill-climb, annealing, GA μ+λ, MAP-Elites.\n"
                "Task mode xor: evolution trains the network through its image bytes.\n"
                "Full statistical report + dashboards are saved to report/."))
            h.addAction(about)

        def _build_ui(self):
            left = QWidget(); lv = QVBoxLayout(left); lv.setSpacing(8)
            gp = QGroupBox("Parameters"); pf = QFormLayout(gp)
            self.sp_pool = QSpinBox(); self.sp_pool.setRange(4, 64); self.sp_pool.setValue(12)
            self.sp_seed = QSpinBox(); self.sp_seed.setRange(0, 9999); self.sp_seed.setValue(7)
            self.sp_evals = QSpinBox(); self.sp_evals.setRange(50, 5000); self.sp_evals.setValue(350)
            self.sp_gens = QSpinBox(); self.sp_gens.setRange(4, 100); self.sp_gens.setValue(22)
            self.sp_pop = QSpinBox(); self.sp_pop.setRange(6, 64); self.sp_pop.setValue(16)
            self.cb_task = QComboBox(); self.cb_task.addItems(["none (CII)", "xor (train)"])
            for w, lab in ((self.sp_pool, "pool size"), (self.sp_seed, "seed"),
                           (self.sp_evals, "evals / searcher"), (self.sp_gens, "GA generations"),
                           (self.sp_pop, "GA population"), (self.cb_task, "task")):
                pf.addRow(lab, w)
            lv.addWidget(gp)

            gb = QGroupBox("Actions"); bv = QVBoxLayout(gb)
            def btn(txt, fn):
                b = QPushButton(txt); b.clicked.connect(fn); bv.addWidget(b); return b
            self.b_gen   = btn("Generate Pool", self.generate)
            self.b_open  = btn("Open PNG…", self.open_png)
            self.b_score = btn("Score Pool", self.score_pool)
            self.b_find  = btn("▶ Find Smartest", self.find_smartest)
            hb = QHBoxLayout()
            self.cb_algo = QComboBox()
            self.cb_algo.addItems(["hill-climb", "annealing", "GA", "MAP-Elites"])
            self.b_run1 = QPushButton("Run Searcher")
            self.b_run1.clicked.connect(self.run_one)
            hb.addWidget(self.cb_algo); hb.addWidget(self.b_run1); bv.addLayout(hb)
            self.b_stop = btn("Stop", self.stop); self.b_stop.setEnabled(False)
            lv.addWidget(gb); lv.addStretch(1)

            self.tabs = QTabWidget()
            wimg = QWidget(); iv = QHBoxLayout(wimg)
            self.lbl_champ = self._img_label("champion")
            self.lbl_src   = self._img_label("source")
            iv.addWidget(self.lbl_champ); iv.addWidget(self.lbl_src)
            info = QVBoxLayout()
            self.lbl_arch = QLabel("architecture: —"); self.lbl_arch.setWordWrap(True)
            self.txt_details = QPlainTextEdit(); self.txt_details.setReadOnly(True)
            self.txt_details.setPlaceholderText("metrics / verdict appear here")
            info.addWidget(self.lbl_arch); info.addWidget(self.txt_details)
            iw = QWidget(); iw.setLayout(info); iw.setMinimumWidth(330)
            iv.addWidget(iw, 1)
            self.tabs.addTab(wimg, "Images")

            self.fig = Figure(figsize=(6, 3.4))
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.tabs.addTab(self.canvas, "Live search")

            self.report_lbl = QLabel("run “Find Smartest” to render report charts")
            self.report_lbl.setAlignment(Qt.AlignCenter)
            self.tabs.addTab(self.report_lbl, "Report PNG")

            self.table = QTableWidget(0, 12)
            self.table.setHorizontalHeaderLabels(
                ["image", "CII", "fit", "I", "H", "I/H", "det", "stab", "TSE",
                 "b/KB", "old proxy", "architecture"])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.table.setSelectionBehavior(QTableWidget.SelectRows)
            self.table.itemSelectionChanged.connect(self.on_row)
            self.tabs.addTab(self.table, "Pool table")

            self.logv = QPlainTextEdit(); self.logv.setReadOnly(True)
            self.tabs.addTab(self.logv, "Log")

            split = QSplitter(Qt.Vertical)
            split.addWidget(self.tabs)
            bot = QWidget(); bv2 = QVBoxLayout(bot); bv2.setContentsMargins(0, 0, 0, 0)
            self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setValue(0)
            bv2.addWidget(self.bar); split.addWidget(bot)
            split.setSizes([640, 60])

            central = QWidget(); lay = QHBoxLayout(central)
            lay.addWidget(left, 0); lay.addWidget(split, 1)
            self.setCentralWidget(central)
            self.statusBar().showMessage("NeuroPix ready — generate a pool or open a PNG")
            self.setAcceptDrops(True)

        def _img_label(self, cap):
            lab = QLabel(cap); lab.setAlignment(Qt.AlignCenter)
            lab.setMinimumSize(240, 240); lab.setStyleSheet(
                "QLabel{background:#202020;color:#888;border:1px solid #444;}"
                "QLabel:hover{border-color:#e6a817;}")
            return lab

        def dragEnterEvent(self, e):
            if e.mimeData().hasUrls(): e.acceptProposedAction()
        def dropEvent(self, e):
            for u in e.mimeData().urls():
                p = u.toLocalFile()
                if p.lower().endswith(".png"):
                    self._start(_job_open, p); break

        def _params(self):
            return dict(pool=self.sp_pool.value(), seed=self.sp_seed.value(),
                        evals=self.sp_evals.value(), gens=self.sp_gens.value(),
                        pop=self.sp_pop.value(),
                        task="xor" if self.cb_task.currentIndex() == 1 else "none")

        def _start(self, fn, *a):
            if not self.b_stop.isEnabled():
                CANCEL.clear(); self.b_stop.setEnabled(True); self._disable(True)
                QThreadPool.globalInstance().start(Task(self.jobs, fn, *a))
            else:
                self.on_log("a job is already running — press Stop first")

        def _disable(self, off):
            for b in (self.b_gen, self.b_open, self.b_score, self.b_find, self.b_run1):
                b.setEnabled(not off)

        def generate(self):
            p = self._params(); self._start(_job_generate, p["pool"], p["seed"], p["task"])
        def open_png(self):
            path, _ = QFileDialog.getOpenFileName(self, "Open RGBA image", "", "PNG (*.png)")
            if path: self._start(_job_open, path)
        def score_pool(self):
            self._start(_job_score_pool, self._params()["task"])
        def find_smartest(self):
            self.series.clear(); self._redraw()
            self._start(_job_find, self._params())
        def run_one(self):
            self.series.clear(); self._redraw()
            self._start(_job_single_search, self.cb_algo.currentText(), self._params())
        def stop(self):
            CANCEL.set(); self.statusBar().showMessage("stopping…")

        def save_champ(self):
            img = STATE.get("champ_img")
            if img is None:
                QMessageBox.information(self, "Save", "no champion yet"); return
            path, _ = QFileDialog.getSaveFileName(self, "Save champion image",
                                                  "smartest.png", "PNG (*.png)")
            if path:
                Image.fromarray(np.ascontiguousarray(img, np.uint8), "RGBA").save(path)
                self.on_log(f"saved {path}")

        def on_phase(self, name):
            self._phase = name
            self.statusBar().showMessage(f"running: {name}")
            self.on_log(f"--- phase: {name}")

        def on_tick(self, evals, best):
            self.bar.setValue(min(100, int(100 * evals / self._total_v)))
            self.statusBar().showMessage(f"evals={evals}  best-so-far={best:.4f}")
            ph = getattr(self, "_phase", "?")
            xs, ys = self.series.get(ph, ([], []))
            xs.append(evals); ys.append(best); self.series[ph] = (xs, ys)
            self._redraw()

        def _redraw(self):
            self.fig.clear(); ax = self.fig.add_subplot(111)
            for i, (lab, (xs, ys)) in enumerate(self.series.items()):
                ax.plot(xs, ys, lw=1.8,
                        color=(GOLD if i == len(self.series) - 1 else STEEL), label=lab)
            ax.set_xlabel("evaluations"); ax.set_ylabel("best fitness so far")
            ax.set_title("NeuroPix live search progress"); ax.grid(alpha=.3)
            if self.series: ax.legend(fontsize=7)
            self.canvas.draw_idle()

        def on_log(self, msg): self.logv.appendPlainText(str(msg))

        def on_rows(self, rows):
            self.table.setRowCount(0); self._byname = {}
            for d in rows:
                r = self.table.rowCount(); self.table.insertRow(r)
                vals = [d["name"], f"{d['cii']:.4f}",
                        "—" if d["fit"] is None else f"{d['fit']:.3f}",
                        f"{d['I']:.3f}", f"{d['H']:.2f}", f"{d['IH']:.2f}",
                        f"{d['det']:.2f}", f"{d['stab']:.2f}", f"{d['tse']:.2f}",
                        f"{d['eff']:.1f}", f"{d['proxy']:.3f}", d["arch"]]
                for cix, v in enumerate(vals):
                    it = QTableWidgetItem(str(v))
                    if cix == 0: it.setData(Qt.UserRole, d["name"])
                    self.table.setItem(r, cix, it)
                self._byname[d["name"]] = d
            if rows:
                self.on_champion(rows[0]["img"])
                STATE["champ_src"] = rows[0]["img"]
                self.lbl_src.setPixmap(np_to_pixmap(rows[0]["img"]))
            self.on_log(f"table updated ({len(rows)} images)")

        def on_row(self):
            items = self.table.selectedItems()
            if not items: return
            name = self.table.item(items[0].row(), 0).data(Qt.UserRole)
            d = self._byname.get(name)
            if d is not None:
                img = d["img"]
                self.lbl_src.setPixmap(np_to_pixmap(img))
                STATE["champ_src"] = img
                self.lbl_arch.setText(f"selected: {name}  [{d['arch']}]")

        def on_champion(self, img):
            STATE["champ_img"] = np.ascontiguousarray(img, np.uint8)
            self.lbl_champ.setPixmap(np_to_pixmap(img))

        def on_text(self, txt):
            self.txt_details.setPlainText(txt); self.on_log(txt)

        def on_done(self, job, payload):
            self.b_stop.setEnabled(False); self._disable(False); self.bar.setValue(100)
            if job == "find":
                rep = payload.get("report", [])
                if rep:
                    path = os.path.join(REPORT, rep[0])
                    if os.path.exists(path):
                        pm = QPixmap(path)
                        self.report_lbl.setPixmap(pm.scaled(
                            self.report_lbl.width() or 1200, 900,
                            Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.statusBar().showMessage(
                    f"done — champion '{payload['name']}' (report PNGs in {REPORT}/)")
            elif job == "search":
                lab, xs, ys = payload["curve"]
                self.series[lab] = (list(xs), list(ys)); self._redraw()
                self.statusBar().showMessage(f"{lab} finished")
            elif job == "score":
                self.statusBar().showMessage("pool scored")
            elif job == "generate":
                self.statusBar().showMessage(f"pool ready in {payload['dir']}/ "
                                             f"(task={payload['task']})")
                self.on_log(f"pool ready: {payload['dir']}/")
            elif job == "open":
                self.statusBar().showMessage(f"opened {payload['name']}")

        def on_failed(self, msg):
            self.b_stop.setEnabled(False); self._disable(False)
            if msg.strip() == CANCELLED_MSG:
                self.statusBar().showMessage("cancelled"); self.on_log("cancelled by user")
            else:
                self.statusBar().showMessage("error — see log")
                self.on_log("ERROR:\n" + msg)
                QMessageBox.critical(self, "Job failed", msg.splitlines()[0])

        def closeEvent(self, e):
            CANCEL.set()
            QThreadPool.globalInstance().waitForDone(3000)
            e.accept()

    app = QApplication(sys.argv)
    app.setApplicationName("NeuroPix")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

# ─────────────────────────── entry point ───────────────────────────
def main():
    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        return cli_main()
    try:
        import PySide6  # noqa: F401
    except ImportError:
        print("[NeuroPix] PySide6 not installed — running console mode. "
              "pip install PySide6 for the GUI.")
        return cli_main()
    run_gui()

if __name__ == "__main__":
    main()