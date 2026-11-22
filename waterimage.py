import sys
import random
import math
import time
import json
import os
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QSlider, QGroupBox, 
                               QPushButton, QScrollArea, QFileDialog, QTabWidget,
                               QComboBox)
from PySide6.QtCore import Qt, QTimer, QPointF, Signal
from PySide6.QtGui import (QPainter, QColor, QRadialGradient, QPen, QBrush, 
                           QPixmap, QImage, QKeySequence, QShortcut)

# ============================================================
# BACKEND DETECTION
# ============================================================
CUPY_AVAILABLE = False
NUMBA_AVAILABLE = False
CUDA_AVAILABLE = False
BACKEND_INFO = {"cupy": "❌ Not installed", "numba": "❌ Not installed", "cuda": "❌ Not available"}

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    BACKEND_INFO["cupy"] = f"✅ v{cp.__version__}"
except ImportError:
    cp = None

try:
    from numba import cuda, jit, prange
    import numpy as np
    NUMBA_AVAILABLE = True
    BACKEND_INFO["numba"] = f"✅ v{__import__('numba').__version__}"
    try:
        if cuda.is_available():
            CUDA_AVAILABLE = True
            device = cuda.get_current_device()
            BACKEND_INFO["cuda"] = f"✅ {device.name}"
        else:
            BACKEND_INFO["cuda"] = "❌ No CUDA device"
    except Exception as e:
        BACKEND_INFO["cuda"] = f"❌ {str(e)[:30]}"
except ImportError:
    cuda = None
    jit = None

# ============================================================
# NUMBA CUDA KERNELS
# ============================================================
if NUMBA_AVAILABLE and CUDA_AVAILABLE:
    @cuda.jit
    def _cuda_density_kernel(x, y, mass, density, n, h2, rest_density, pressure_mult):
        i = cuda.grid(1)
        if i < n:
            d = 0.0
            xi = x[i]
            yi = y[i]
            for j in range(n):
                dx = x[j] - xi
                dy = y[j] - yi
                r2 = dx*dx + dy*dy
                if r2 < h2:
                    diff = h2 - r2
                    d += mass[j] * diff * diff * diff
            density[i] = max(d, 0.001)

    @cuda.jit
    def _cuda_pressure_kernel(x, y, mass, density, pressure, n, h2, pressure_mult, rest_density):
        i = cuda.grid(1)
        if i < n:
            pressure[i] = pressure_mult * (density[i] - rest_density)

    @cuda.jit
    def _cuda_force_kernel(x, y, vx, vy, mass, density, pressure, fx, fy, n, h, viscosity, surface_tension):
        i = cuda.grid(1)
        if i < n:
            fxi = 0.0
            fyi = 0.0
            xi = x[i]
            yi = y[i]
            pi = pressure[i]
            di = density[i]
            h2 = h * h
            for j in range(n):
                if i == j:
                    continue
                dx = x[j] - xi
                dy = y[j] - yi
                r2 = dx*dx + dy*dy
                if r2 < h2 and r2 > 0.01:
                    r = math.sqrt(r2)
                    diff = h - r
                    # Pressure
                    pf = -mass[j] * (pi + pressure[j]) / (2.0 * density[j]) * diff * diff / r
                    fxi += pf * dx
                    fyi += pf * dy
                    # Viscosity
                    vf = viscosity * mass[j] * (h - r) / density[j]
                    fxi += vf * (vx[j] - vx[i])
                    fyi += vf * (vy[j] - vy[i])
                    # Surface tension
                    if r > h * 0.5:
                        coh = surface_tension * mass[j] * diff * diff / r
                        fxi += coh * dx
                        fyi += coh * dy
            fx[i] = fxi / di
            fy[i] = fyi / di

# ============================================================
# NUMBA CPU (JIT) KERNELS
# ============================================================
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def _numba_cpu_density(x, y, mass, density, n, h2):
        for i in prange(n):
            d = 0.0
            xi = x[i]
            yi = y[i]
            for j in range(n):
                dx = x[j] - xi
                dy = y[j] - yi
                r2 = dx*dx + dy*dy
                if r2 < h2:
                    diff = h2 - r2
                    d += mass[j] * diff * diff * diff
            density[i] = max(d, 0.001)

    @jit(nopython=True, parallel=True, cache=True)
    def _numba_cpu_forces(x, y, vx, vy, mass, density, pressure, fx, fy, n, h, viscosity, surface_tension):
        h2 = h * h
        for i in prange(n):
            fxi = 0.0
            fyi = 0.0
            xi = x[i]
            yi = y[i]
            pi = pressure[i]
            di = density[i]
            for j in range(n):
                if i == j:
                    continue
                dx = x[j] - xi
                dy = y[j] - yi
                r2 = dx*dx + dy*dy
                if r2 < h2 and r2 > 0.01:
                    r = math.sqrt(r2)
                    diff = h - r
                    pf = -mass[j] * (pi + pressure[j]) / (2.0 * density[j]) * diff * diff / r
                    fxi += pf * dx
                    fyi += pf * dy
                    vf = viscosity * mass[j] * (h - r) / density[j]
                    fxi += vf * (vx[j] - vx[i])
                    fyi += vf * (vy[j] - vy[i])
                    if r > h * 0.5:
                        coh = surface_tension * mass[j] * diff * diff / r
                        fxi += coh * dx
                        fyi += coh * dy
            fx[i] = fxi / di
            fy[i] = fyi / di

# ============================================================
# CUPY RAW KERNELS
# ============================================================
if CUPY_AVAILABLE:
    _cupy_density_src = r'''
    extern "C" __global__
    void cupy_density(const float* x, const float* y, const float* mass,
                      float* density, int n, float h2) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            float d = 0.0f;
            float xi = x[i];
            float yi = y[i];
            for (int j = 0; j < n; j++) {
                float dx = x[j] - xi;
                float dy = y[j] - yi;
                float r2 = dx*dx + dy*dy;
                if (r2 < h2) {
                    float diff = h2 - r2;
                    d += mass[j] * diff * diff * diff;
                }
            }
            density[i] = fmaxf(d, 0.001f);
        }
    }
    '''
    _cupy_forces_src = r'''
    extern "C" __global__
    void cupy_forces(const float* x, const float* y, const float* vx, const float* vy,
                     const float* mass, const float* density, const float* pressure,
                     float* fx, float* fy, int n, float h, float viscosity, float surface_tension) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            float fxi = 0.0f, fyi = 0.0f;
            float xi = x[i], yi = y[i];
            float pi = pressure[i], di = density[i];
            float h2 = h * h;
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                float dx = x[j] - xi;
                float dy = y[j] - yi;
                float r2 = dx*dx + dy*dy;
                if (r2 < h2 && r2 > 0.01f) {
                    float r = sqrtf(r2);
                    float diff = h - r;
                    float pf = -mass[j] * (pi + pressure[j]) / (2.0f * density[j]) * diff * diff / r;
                    fxi += pf * dx;
                    fyi += pf * dy;
                    float vf = viscosity * mass[j] * (h - r) / density[j];
                    fxi += vf * (vx[j] - vx[i]);
                    fyi += vf * (vy[j] - vy[i]);
                    if (r > h * 0.5f) {
                        float coh = surface_tension * mass[j] * diff * diff / r;
                        fxi += coh * dx;
                        fyi += coh * dy;
                    }
                }
            }
            fx[i] = fxi / di;
            fy[i] = fyi / di;
        }
    }
    '''
    try:
        _cupy_density_kernel = cp.RawKernel(_cupy_density_src, 'cupy_density')
        _cupy_forces_kernel = cp.RawKernel(_cupy_forces_src, 'cupy_forces')
    except Exception as e:
        print(f"CuPy kernel compile failed: {e}")
        CUPY_AVAILABLE = False

# ============================================================
# UI COMPONENTS
# ============================================================
class ValueSlider(QWidget):
    valueChanged = Signal(object)
    
    def __init__(self, label, min_val, max_val, default_val, step=1.0, is_int=False):
        super().__init__()
        self.step = step
        self.is_int = is_int
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(label)
        self.label.setFixedWidth(120)
        self.label.setStyleSheet("color: #8d99ae; border: none;")
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(min_val / step), int(max_val / step))
        self.slider.setValue(int(default_val / step))
        
        self.value_label = QLabel(str(default_val))
        self.value_label.setFixedWidth(50)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setStyleSheet("color: #5bc0be; font-weight: bold; border: none;")
        
        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        
        self.slider.valueChanged.connect(self.on_change)
        
    def on_change(self, val):
        actual_val = val * self.step
        if self.is_int:
            actual_val = int(actual_val)
            self.value_label.setText(str(actual_val))
        else:
            self.value_label.setText(f"{actual_val:.3f}")
        self.valueChanged.emit(actual_val)
        
    def set_value(self, val):
        self.slider.setValue(int(val / self.step))

# ============================================================
# PARTICLE & RIPPLE
# ============================================================
class Particle:
    def __init__(self, x, y, vx, vy, radius):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.radius = radius
        self.density = 0.0
        self.pressure = 0.0
        self.life = 1000
        self.is_splash = False
        self.mass = radius * radius
        self.color = (100, 200, 255)
        self.edge_color = (20, 80, 180)

class WaveRipple:
    def __init__(self, x, y, strength=1.0):
        self.x, self.y = x, y
        self.radius = 0.0
        self.max_radius = 100.0 * strength
        self.life = 1.0
        self.speed = 80.0 * strength
        self.decay = 0.015

# ============================================================
# PHYSICS CANVAS
# ============================================================
class PhysicsCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        
        self.params = {
            'gravity': 0.5, 'spawn_rate': 10.0, 'drop_size': 8.0,
            'max_particles': 800, 'wind': 0.0, 'wind_turbulence': 0.3,
            'time_scale': 1.0, 'surface_tension': 0.5, 'viscosity': 0.02,
            'pressure_multiplier': 50.0, 'rest_density': 1.0,
            'smoothing_radius': 30.0, 'collision_damping': 0.7,
            'backend': 'auto'
        }
        
        self.particles = []
        self.ripples = []
        self.spawn_accumulator = 0.0
        
        # GPU arrays (lazy init)
        self._gpu_x = None
        self._gpu_y = None
        self._gpu_vx = None
        self._gpu_vy = None
        self._gpu_mass = None
        self._gpu_density = None
        self._gpu_pressure = None
        self._gpu_fx = None
        self._gpu_fy = None
        self._gpu_capacity = 0
        
        # Active backend tracking
        self.active_backend = "python"
        self.backend_status = "Pure Python"
        
        # Background
        self.background_image = None
        
        # Performance
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        self.physics_time = 0.0
        self.compute_time = 0.0
        
        # Animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)
        
        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(self.calculate_fps)
        self.fps_timer.start(500)

    def update_params(self, key, value):
        self.params[key] = value

    def set_background(self, path):
        if path:
            self.background_image = QPixmap(path)
        else:
            self.background_image = None
        self.update()

    def clear_background(self):
        self.background_image = None
        self.update()

    def calculate_fps(self):
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        self.frame_count = 0
        self.last_fps_time = current_time
        try:
            self.parent().parent().parent().update_stats(
                self.fps, len(self.particles), len(self.ripples), 
                self.physics_time * 1000, self.compute_time * 1000,
                self.backend_status
            )
        except:
            pass

    def spawn_droplet(self, x=None, y=None, size_mult=1.0):
        if len(self.particles) >= self.params['max_particles']:
            return
        if x is None: 
            x = random.uniform(50, self.width() - 50)
        if y is None: 
            y = -20
        
        vx = 0
        vy = random.uniform(1, 3)
        radius = self.params['drop_size'] * size_mult * random.uniform(0.8, 1.2)
        self.particles.append(Particle(x, y, vx, vy, radius))

    def create_splash(self, x, y, impact_velocity):
        count = int(8 + impact_velocity * 2)
        vel = min(15, 5 + impact_velocity)
        for _ in range(count):
            angle = random.uniform(-math.pi, 0)
            speed = random.uniform(vel * 0.5, vel * 1.5)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            radius = random.uniform(1.5, 3.5)
            p = Particle(x, y, vx, vy, radius)
            p.is_splash = True
            p.life = random.randint(20, 40)
            self.particles.append(p)
        strength = min(1.0, impact_velocity / 10.0)
        self.ripples.append(WaveRipple(x, y, strength))

    # ============================================================
    # GPU ARRAY MANAGEMENT
    # ============================================================
    def _ensure_gpu_arrays(self, n):
        """Allocate/resize GPU arrays to fit n particles."""
        if n > self._gpu_capacity:
            new_cap = max(n, self._gpu_capacity * 2, 512)
            if CUPY_AVAILABLE and self.params['backend'] in ('auto', 'cupy'):
                self._gpu_x = cp.zeros(new_cap, dtype=cp.float32)
                self._gpu_y = cp.zeros(new_cap, dtype=cp.float32)
                self._gpu_vx = cp.zeros(new_cap, dtype=cp.float32)
                self._gpu_vy = cp.zeros(new_cap, dtype=cp.float32)
                self._gpu_mass = cp.zeros(new_cap, dtype=cp.float32)
                self._gpu_density = cp.zeros(new_cap, dtype=cp.float32)
                self._gpu_pressure = cp.zeros(new_cap, dtype=cp.float32)
                self._gpu_fx = cp.zeros(new_cap, dtype=cp.float32)
                self._gpu_fy = cp.zeros(new_cap, dtype=cp.float32)
            else:
                self._gpu_x = np.zeros(new_cap, dtype=np.float32)
                self._gpu_y = np.zeros(new_cap, dtype=np.float32)
                self._gpu_vx = np.zeros(new_cap, dtype=np.float32)
                self._gpu_vy = np.zeros(new_cap, dtype=np.float32)
                self._gpu_mass = np.zeros(new_cap, dtype=np.float32)
                self._gpu_density = np.zeros(new_cap, dtype=np.float32)
                self._gpu_pressure = np.zeros(new_cap, dtype=np.float32)
                self._gpu_fx = np.zeros(new_cap, dtype=np.float32)
                self._gpu_fy = np.zeros(new_cap, dtype=np.float32)
            self._gpu_capacity = new_cap

    def _sync_to_arrays(self):
        """Copy particle data into contiguous arrays."""
        n = len(self.particles)
        self._ensure_gpu_arrays(n)
        for i, p in enumerate(self.particles):
            self._gpu_x[i] = p.x
            self._gpu_y[i] = p.y
            self._gpu_vx[i] = p.vx
            self._gpu_vy[i] = p.vy
            self._gpu_mass[i] = p.mass
        return n

    def _sync_from_arrays(self, n):
        """Copy computed data back to particles."""
        for i in range(n):
            p = self.particles[i]
            p.density = float(self._gpu_density[i])
            p.pressure = float(self._gpu_pressure[i])
            p.vx += float(self._gpu_fx[i])
            p.vy += float(self._gpu_fy[i])

    # ============================================================
    # COMPUTE BACKENDS
    # ============================================================
    def _compute_python(self, n):
        """Pure Python SPH - baseline."""
        h = self.params['smoothing_radius']
        h2 = h * h
        rest_density = self.params['rest_density']
        pressure_mult = self.params['pressure_multiplier']
        viscosity = self.params['viscosity']
        surface_tension = self.params['surface_tension']
        
        # Density
        for i in range(n):
            p = self.particles[i]
            d = 0.0
            for j in range(n):
                n2 = self.particles[j]
                dx = n2.x - p.x
                dy = n2.y - p.y
                r2 = dx*dx + dy*dy
                if r2 < h2:
                    diff = h2 - r2
                    d += n2.mass * diff * diff * diff
            p.density = max(d, 0.001)
            p.pressure = pressure_mult * (p.density - rest_density)
        
        # Forces
        for i in range(n):
            p = self.particles[i]
            if p.is_splash:
                continue
            fx, fy = 0.0, 0.0
            for j in range(n):
                if i == j:
                    continue
                n2 = self.particles[j]
                dx = n2.x - p.x
                dy = n2.y - p.y
                r2 = dx*dx + dy*dy
                if r2 < h2 and r2 > 0.01:
                    r = math.sqrt(r2)
                    diff = h - r
                    pf = -n2.mass * (p.pressure + n2.pressure) / (2 * n2.density) * diff * diff / r
                    fx += pf * dx
                    fy += pf * dy
                    vf = viscosity * n2.mass * (h - r) / n2.density
                    fx += vf * (n2.vx - p.vx)
                    fy += vf * (n2.vy - p.vy)
                    if r > h * 0.5:
                        coh = surface_tension * n2.mass * diff * diff / r
                        fx += coh * dx
                        fy += coh * dy
            p.vx += fx / p.density
            p.vy += fy / p.density

    def _compute_numba_cpu(self, n):
        """Numba JIT-compiled CPU computation."""
        self._sync_to_arrays()
        h = self.params['smoothing_radius']
        h2 = h * h
        rest_density = self.params['rest_density']
        pressure_mult = self.params['pressure_multiplier']
        viscosity = self.params['viscosity']
        surface_tension = self.params['surface_tension']
        
        _numba_cpu_density(
            self._gpu_x[:n], self._gpu_y[:n], self._gpu_mass[:n],
            self._gpu_density[:n], n, h2
        )
        
        # Pressure (simple, vectorized)
        for i in range(n):
            self._gpu_pressure[i] = pressure_mult * (self._gpu_density[i] - rest_density)
        
        _numba_cpu_forces(
            self._gpu_x[:n], self._gpu_y[:n], self._gpu_vx[:n], self._gpu_vy[:n],
            self._gpu_mass[:n], self._gpu_density[:n], self._gpu_pressure[:n],
            self._gpu_fx[:n], self._gpu_fy[:n], n, h, viscosity, surface_tension
        )
        
        self._sync_from_arrays(n)

    def _compute_numba_cuda(self, n):
        """Numba CUDA GPU computation."""
        self._sync_to_arrays()
        
        # Copy to GPU
        d_x = cuda.to_device(self._gpu_x[:n])
        d_y = cuda.to_device(self._gpu_y[:n])
        d_vx = cuda.to_device(self._gpu_vx[:n])
        d_vy = cuda.to_device(self._gpu_vy[:n])
        d_mass = cuda.to_device(self._gpu_mass[:n])
        d_density = cuda.to_device(self._gpu_density[:n])
        d_pressure = cuda.to_device(self._gpu_pressure[:n])
        d_fx = cuda.to_device(self._gpu_fx[:n])
        d_fy = cuda.to_device(self._gpu_fy[:n])
        
        h = self.params['smoothing_radius']
        h2 = h * h
        rest_density = self.params['rest_density']
        pressure_mult = self.params['pressure_multiplier']
        viscosity = self.params['viscosity']
        surface_tension = self.params['surface_tension']
        
        threads = 128
        blocks = (n + threads - 1) // threads
        
        _cuda_density_kernel[blocks, threads](
            d_x, d_y, d_mass, d_density, n, h2, rest_density, pressure_mult
        )
        _cuda_pressure_kernel[blocks, threads](
            d_x, d_y, d_mass, d_density, d_pressure, n, h2, pressure_mult, rest_density
        )
        _cuda_force_kernel[blocks, threads](
            d_x, d_y, d_vx, d_vy, d_mass, d_density, d_pressure,
            d_fx, d_fy, n, h, viscosity, surface_tension
        )
        
        # Copy back
        self._gpu_density[:n] = d_density.copy_to_host()
        self._gpu_pressure[:n] = d_pressure.copy_to_host()
        self._gpu_fx[:n] = d_fx.copy_to_host()
        self._gpu_fy[:n] = d_fy.copy_to_host()
        
        self._sync_from_arrays(n)

    def _compute_cupy(self, n):
        """CuPy GPU computation."""
        self._sync_to_arrays()
        
        # Convert to cupy arrays
        g_x = cp.asarray(self._gpu_x[:n])
        g_y = cp.asarray(self._gpu_y[:n])
        g_vx = cp.asarray(self._gpu_vx[:n])
        g_vy = cp.asarray(self._gpu_vy[:n])
        g_mass = cp.asarray(self._gpu_mass[:n])
        g_density = cp.asarray(self._gpu_density[:n])
        g_pressure = cp.asarray(self._gpu_pressure[:n])
        g_fx = cp.asarray(self._gpu_fx[:n])
        g_fy = cp.asarray(self._gpu_fy[:n])
        
        h = float(self.params['smoothing_radius'])
        h2 = h * h
        rest_density = float(self.params['rest_density'])
        pressure_mult = float(self.params['pressure_multiplier'])
        viscosity = float(self.params['viscosity'])
        surface_tension = float(self.params['surface_tension'])
        
        threads = 128
        blocks = (n + threads - 1) // threads
        
        _cupy_density_kernel((blocks,), (threads,),
            (g_x, g_y, g_mass, g_density, n, h2))
        
        # Pressure (elementwise)
        cp.ElementwiseKernel(
            'float32 density, float32 rest_density, float32 pressure_mult',
            'float32 pressure',
            'pressure = pressure_mult * (density - rest_density)',
            'pressure_kernel'
        )(g_density, rest_density, pressure_mult, g_pressure)
        
        _cupy_forces_kernel((blocks,), (threads,),
            (g_x, g_y, g_vx, g_vy, g_mass, g_density, g_pressure,
             g_fx, g_fy, n, h, viscosity, surface_tension))
        
        # Copy back
        self._gpu_density[:n] = cp.asnumpy(g_density)
        self._gpu_pressure[:n] = cp.asnumpy(g_pressure)
        self._gpu_fx[:n] = cp.asnumpy(g_fx)
        self._gpu_fy[:n] = cp.asnumpy(g_fy)
        
        self._sync_from_arrays(n)

    def _compute_physics(self, n):
        """Dispatch to the selected backend."""
        backend = self.params['backend']
        
        # Auto-select best backend
        if backend == 'auto':
            if CUDA_AVAILABLE and NUMBA_AVAILABLE:
                backend = 'numba_cuda'
            elif CUPY_AVAILABLE:
                backend = 'cupy'
            elif NUMBA_AVAILABLE:
                backend = 'numba_cpu'
            else:
                backend = 'python'
        
        try:
            if backend == 'python':
                self.active_backend = 'python'
                self.backend_status = "🐌 Pure Python"
                self._compute_python(n)
            elif backend == 'numba_cpu' and NUMBA_AVAILABLE:
                self.active_backend = 'numba_cpu'
                self.backend_status = "⚡ Numba CPU (JIT)"
                self._compute_numba_cpu(n)
            elif backend == 'numba_cuda' and CUDA_AVAILABLE:
                self.active_backend = 'numba_cuda'
                self.backend_status = "🚀 Numba CUDA (GPU)"
                self._compute_numba_cuda(n)
            elif backend == 'cupy' and CUPY_AVAILABLE:
                self.active_backend = 'cupy'
                self.backend_status = "🔥 CuPy (GPU)"
                self._compute_cupy(n)
            else:
                # Fallback
                self.active_backend = 'python'
                self.backend_status = "🐌 Pure Python (fallback)"
                self._compute_python(n)
        except Exception as e:
            print(f"Backend {backend} failed: {e}, falling back to Python")
            self.active_backend = 'python'
            self.backend_status = f"🐌 Python (error: {str(e)[:20]})"
            self._compute_python(n)

    # ============================================================
    # COLLISION HANDLING (CPU - kept simple)
    # ============================================================
    def handle_collisions(self):
        for p in self.particles:
            for n in self.particles:
                if n is p or n.is_splash:
                    continue
                dx = n.x - p.x
                dy = n.y - p.y
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = p.radius + n.radius
                if dist < min_dist and dist > 0.01:
                    overlap = min_dist - dist
                    nx, ny = dx / dist, dy / dist
                    total_mass = p.mass + n.mass
                    p.x -= nx * overlap * (n.mass / total_mass)
                    p.y -= ny * overlap * (n.mass / total_mass)
                    n.x += nx * overlap * (p.mass / total_mass)
                    n.y += ny * overlap * (p.mass / total_mass)
                    dvx = p.vx - n.vx
                    dvy = p.vy - n.vy
                    dv_dot_n = dvx * nx + dvy * ny
                    if dv_dot_n > 0:
                        impulse = (2 * dv_dot_n) / total_mass
                        d = self.params['collision_damping']
                        p.vx -= impulse * n.mass * nx * d
                        p.vy -= impulse * n.mass * ny * d
                        n.vx += impulse * p.mass * nx * d
                        n.vy += impulse * p.mass * ny * d

    # ============================================================
    # MAIN UPDATE LOOP
    # ============================================================
    def update_frame(self):
        start_time = time.time()
        self.frame_count += 1
        
        time_scale = self.params['time_scale']
        
        # Spawn
        self.spawn_accumulator += self.params['spawn_rate'] * time_scale / 60.0
        while self.spawn_accumulator >= 1.0:
            self.spawn_accumulator -= 1.0
            self.spawn_droplet()
        
        gravity = self.params['gravity'] * time_scale
        wind = self.params['wind'] * time_scale
        wind_turb = self.params['wind_turbulence']
        
        # SPH physics (GPU/CPU accelerated)
        n = len(self.particles)
        if n > 1:
            compute_start = time.time()
            self._compute_physics(n)
            self.compute_time = time.time() - compute_start
            self.handle_collisions()
        else:
            self.compute_time = 0.0
        
        # Update positions
        for p in self.particles:
            p.vy += gravity
            wind_force = wind + random.uniform(-wind_turb, wind_turb) * abs(wind)
            p.vx += wind_force * time_scale
            p.x += p.vx * time_scale
            p.y += p.vy * time_scale
            p.life -= 1
            
            if p.y >= self.height() and not p.is_splash:
                impact_vel = abs(p.vy)
                self.create_splash(p.x, self.height(), impact_vel)
                p.life = 0
            elif p.y >= self.height() and p.is_splash:
                p.life = 0
            elif p.is_splash and p.life <= 0:
                p.life = 0
            elif p.x < -50 or p.x > self.width() + 50:
                p.life = 0
        
        for r in self.ripples:
            r.radius += r.speed * time_scale / 60.0
            r.life -= r.decay * time_scale
        
        self.particles = [p for p in self.particles if p.life > 0]
        self.ripples = [r for r in self.ripples if r.life > 0]
        
        self.physics_time = time.time() - start_time
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        if self.background_image:
            scaled = self.background_image.scaled(w, h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x = (w - scaled.width()) // 2
            y = (h - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.fillRect(self.rect(), QColor(11, 19, 43))
        
        for r in self.ripples:
            alpha = int(150 * r.life)
            color = QColor(150, 220, 255, alpha)
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(r.x, r.y), r.radius, r.radius * 0.3)
        
        for p in self.particles:
            if p.is_splash:
                alpha = int(255 * (p.life / 40))
                color = QColor(*p.color, alpha)
                painter.setBrush(color)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(p.x, p.y), p.radius, p.radius)
            else:
                gradient = QRadialGradient(p.x - p.radius * 0.3, p.y - p.radius * 0.3, p.radius * 1.3)
                gradient.setColorAt(0.0, QColor(*p.color, 220))
                gradient.setColorAt(0.6, QColor(*p.edge_color, 180))
                gradient.setColorAt(1.0, QColor(*p.edge_color, 100))
                painter.setBrush(QBrush(gradient))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(p.x, p.y), p.radius, p.radius)
                hl_r = p.radius * 0.3
                hl_x = p.x - p.radius * 0.3
                hl_y = p.y - p.radius * 0.3
                hl_gradient = QRadialGradient(hl_x, hl_y, hl_r)
                hl_gradient.setColorAt(0.0, QColor(255, 255, 255, 200))
                hl_gradient.setColorAt(1.0, QColor(255, 255, 255, 0))
                painter.setBrush(QBrush(hl_gradient))
                painter.drawEllipse(QPointF(hl_x, hl_y), hl_r, hl_r)
        
        painter.end()

# ============================================================
# SETTINGS PANEL
# ============================================================
class SettingsPanel(QWidget):
    param_changed = Signal(str, object)
    load_background = Signal()
    clear_background = Signal()
    save_config = Signal()
    load_config = Signal()
    reset_sim = Signal()

    def __init__(self):
        super().__init__()
        self.setFixedWidth(360)
        self.setStyleSheet("""
            QWidget { background-color: #1c2541; color: #e0e1dd; font-family: 'Segoe UI', Arial; }
            QGroupBox { font-weight: bold; border: 1px solid #3a506b; border-radius: 5px; margin-top: 10px; padding-top: 15px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background-color: #3a506b; border: none; padding: 8px; border-radius: 4px; color: #e0e1dd; font-weight: bold; }
            QPushButton:hover { background-color: #5bc0be; color: #0b132b; }
            QComboBox { background-color: #3a506b; border: none; padding: 5px; border-radius: 4px; color: #e0e1dd; }
            QLabel { border: none; }
            QTabWidget::pane { border: 1px solid #3a506b; border-radius: 3px; }
            QTabBar::tab { background-color: #3a506b; padding: 8px 15px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background-color: #5bc0be; color: #0b132b; }
        """)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("QScrollArea { background-color: #1c2541; border: none; }")
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(15, 15, 15, 15)
        
        title = QLabel("💧 GPU-Accelerated Water Sim")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #5bc0be;")
        layout.addWidget(title)
        layout.addSpacing(10)
        
        # Actions
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()
        self.reset_btn = QPushButton("🔄 Reset (R)")
        self.reset_btn.clicked.connect(self.reset_sim.emit)
        action_layout.addWidget(self.reset_btn)
        self.save_btn = QPushButton("💾 Save Config")
        self.save_btn.clicked.connect(self.save_config.emit)
        action_layout.addWidget(self.save_btn)
        self.load_btn = QPushButton("📂 Load Config")
        self.load_btn.clicked.connect(self.load_config.emit)
        action_layout.addWidget(self.load_btn)
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)
        
        # Background
        bg_group = QGroupBox("Background")
        bg_layout = QHBoxLayout()
        self.bg_load_btn = QPushButton("Load Image")
        self.bg_load_btn.clicked.connect(self.load_background.emit)
        self.bg_clear_btn = QPushButton("Clear")
        self.bg_clear_btn.clicked.connect(self.clear_background.emit)
        bg_layout.addWidget(self.bg_load_btn)
        bg_layout.addWidget(self.bg_clear_btn)
        bg_group.setLayout(bg_layout)
        layout.addWidget(bg_group)
        
        # Tabs
        tabs = QTabWidget()
        
        # Compute tab (NEW!)
        compute_tab = QWidget()
        compute_layout = QVBoxLayout(compute_tab)
        
        backend_group = QGroupBox("Compute Backend")
        backend_inner = QVBoxLayout()
        
        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["auto", "python", "numba_cpu", "numba_cuda", "cupy"])
        self.backend_combo.currentTextChanged.connect(lambda v: self.param_changed.emit('backend', v))
        backend_row.addWidget(self.backend_combo)
        backend_inner.addLayout(backend_row)
        
        # Status display
        self.status_label = QLabel("Active: detecting...")
        self.status_label.setStyleSheet("color: #ffd700; font-weight: bold; padding: 8px; background-color: #0b132b; border-radius: 4px;")
        self.status_label.setWordWrap(True)
        backend_inner.addWidget(self.status_label)
        
        backend_group.setLayout(backend_inner)
        compute_layout.addWidget(backend_group)
        
        # Availability info
        avail_group = QGroupBox("Library Availability")
        avail_inner = QVBoxLayout()
        for name, status in BACKEND_INFO.items():
            lbl = QLabel(f"<b>{name.upper()}</b>: {status}")
            lbl.setStyleSheet("color: #8d99ae; padding: 3px;")
            avail_inner.addWidget(lbl)
        avail_group.setLayout(avail_inner)
        compute_layout.addWidget(avail_group)
        
        # Install instructions
        install_group = QGroupBox("Install Instructions")
        install_inner = QVBoxLayout()
        install_text = QLabel(
            "<b>For GPU acceleration:</b><br>"
            "<code>pip install cupy-cuda12x</code><br>"
            "<code>pip install numba</code><br><br>"
            "<b>Note:</b> CuPy requires CUDA toolkit and NVIDIA GPU. "
            "Use <code>cupy-cuda12x</code> for CUDA 12.x (matches your cupy126 request)."
        )
        install_text.setStyleSheet("color: #8d99ae; font-size: 11px;")
        install_text.setWordWrap(True)
        install_inner.addWidget(install_text)
        install_group.setLayout(install_inner)
        compute_layout.addWidget(install_group)
        
        compute_layout.addStretch()
        tabs.addTab(compute_tab, "🚀 Compute")
        
        # Physics tab
        physics_tab = QWidget()
        physics_layout = QVBoxLayout(physics_tab)
        
        physics_group = QGroupBox("Physics")
        physics_inner = QVBoxLayout()
        
        self.gravity_slider = self.create_slider("Gravity", 0, 2.0, 0.5, 0.05)
        physics_inner.addWidget(self.gravity_slider)
        self.gravity_slider.valueChanged.connect(lambda v: self.param_changed.emit('gravity', v))
        
        self.wind_slider = self.create_slider("Wind", -2.0, 2.0, 0.0, 0.1)
        physics_inner.addWidget(self.wind_slider)
        self.wind_slider.valueChanged.connect(lambda v: self.param_changed.emit('wind', v))
        
        self.wind_turb_slider = self.create_slider("Wind Turbulence", 0, 1.0, 0.3, 0.05)
        physics_inner.addWidget(self.wind_turb_slider)
        self.wind_turb_slider.valueChanged.connect(lambda v: self.param_changed.emit('wind_turbulence', v))
        
        self.time_scale_slider = self.create_slider("Time Scale", 0.1, 3.0, 1.0, 0.1)
        physics_inner.addWidget(self.time_scale_slider)
        self.time_scale_slider.valueChanged.connect(lambda v: self.param_changed.emit('time_scale', v))
        
        physics_group.setLayout(physics_inner)
        physics_layout.addWidget(physics_group)
        
        fluid_group = QGroupBox("Water Properties")
        fluid_inner = QVBoxLayout()
        
        self.viscosity_slider = self.create_slider("Viscosity", 0, 0.5, 0.02, 0.01)
        fluid_inner.addWidget(self.viscosity_slider)
        self.viscosity_slider.valueChanged.connect(lambda v: self.param_changed.emit('viscosity', v))
        
        self.surface_tension_slider = self.create_slider("Surface Tension", 0, 1.0, 0.5, 0.05)
        fluid_inner.addWidget(self.surface_tension_slider)
        self.surface_tension_slider.valueChanged.connect(lambda v: self.param_changed.emit('surface_tension', v))
        
        self.pressure_slider = self.create_slider("Pressure", 0, 200, 50, 5, True)
        fluid_inner.addWidget(self.pressure_slider)
        self.pressure_slider.valueChanged.connect(lambda v: self.param_changed.emit('pressure_multiplier', v))
        
        fluid_group.setLayout(fluid_inner)
        physics_layout.addWidget(fluid_group)
        
        physics_layout.addStretch()
        tabs.addTab(physics_tab, "Physics")
        
        # Scene tab
        scene_tab = QWidget()
        scene_layout = QVBoxLayout(scene_tab)
        
        scene_group = QGroupBox("Scene")
        scene_inner = QVBoxLayout()
        
        self.spawn_slider = self.create_slider("Spawn Rate", 0, 50, 10, 1, True)
        scene_inner.addWidget(self.spawn_slider)
        self.spawn_slider.valueChanged.connect(lambda v: self.param_changed.emit('spawn_rate', v))
        
        self.size_slider = self.create_slider("Drop Size", 2, 20, 8, 1, True)
        scene_inner.addWidget(self.size_slider)
        self.size_slider.valueChanged.connect(lambda v: self.param_changed.emit('drop_size', v))
        
        self.max_part_slider = self.create_slider("Max Particles", 100, 2000, 800, 50, True)
        scene_inner.addWidget(self.max_part_slider)
        self.max_part_slider.valueChanged.connect(lambda v: self.param_changed.emit('max_particles', v))
        
        scene_group.setLayout(scene_inner)
        scene_layout.addWidget(scene_group)
        
        scene_layout.addStretch()
        tabs.addTab(scene_tab, "Scene")
        
        layout.addWidget(tabs)
        
        scroll.setWidget(container)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        
        self.sliders = {
            'gravity': self.gravity_slider, 'wind': self.wind_slider,
            'wind_turbulence': self.wind_turb_slider, 'time_scale': self.time_scale_slider,
            'viscosity': self.viscosity_slider, 'surface_tension': self.surface_tension_slider,
            'pressure_multiplier': self.pressure_slider, 'spawn_rate': self.spawn_slider,
            'drop_size': self.size_slider, 'max_particles': self.max_part_slider
        }

    def create_slider(self, label, min_val, max_val, default_val, step=1.0, is_int=False):
        return ValueSlider(label, min_val, max_val, default_val, step, is_int)

    def apply_config(self, config):
        for key, val in config.items():
            if key in self.sliders:
                self.sliders[key].set_value(val)
            if key == 'backend':
                self.backend_combo.setCurrentText(val)

    def update_backend_status(self, status):
        self.status_label.setText(f"Active: {status}")

# ============================================================
# MAIN WINDOW
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("💧 GPU-Accelerated Water Simulation")
        self.resize(1300, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        canvas_wrapper = QWidget()
        canvas_wrapper.setStyleSheet("background-color: #0b132b;")
        canvas_layout = QVBoxLayout(canvas_wrapper)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        self.canvas = PhysicsCanvas()
        canvas_layout.addWidget(self.canvas, 1)
        
        # Stats bar
        stats_bar = QWidget()
        stats_bar.setStyleSheet("background-color: rgba(0,0,0,0.7); padding: 5px;")
        stats_layout = QHBoxLayout(stats_bar)
        stats_layout.setContentsMargins(10, 3, 10, 3)
        self.fps_label = QLabel("FPS: 0")
        self.particles_label = QLabel("Particles: 0")
        self.ripples_label = QLabel("Ripples: 0")
        self.compute_label = QLabel("Compute: 0ms")
        self.backend_label = QLabel("Backend: -")
        for lbl in (self.fps_label, self.particles_label, self.ripples_label, self.compute_label, self.backend_label):
            lbl.setStyleSheet("color: #5bc0be; font-weight: bold; border: none;")
            stats_layout.addWidget(lbl)
        stats_layout.addStretch()
        canvas_layout.addWidget(stats_bar)
        
        main_layout.addWidget(canvas_wrapper, 1)
        
        self.settings = SettingsPanel()
        
        self.settings.param_changed.connect(self.canvas.update_params)
        self.settings.load_background.connect(self.load_background)
        self.settings.clear_background.connect(self.canvas.clear_background)
        self.settings.save_config.connect(self.save_config)
        self.settings.load_config.connect(self.load_config)
        self.settings.reset_sim.connect(self.reset_sim)
        
        main_layout.addWidget(self.settings)
        
        QShortcut(QKeySequence("R"), self, self.reset_sim)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_config)
        QShortcut(QKeySequence("Ctrl+O"), self, self.load_background)

    def update_stats(self, fps, particles, ripples, physics_time, compute_time, backend_status):
        self.fps_label.setText(f"FPS: {fps:.0f}")
        self.particles_label.setText(f"Particles: {particles}")
        self.ripples_label.setText(f"Ripples: {ripples}")
        self.compute_label.setText(f"Compute: {compute_time:.1f}ms")
        self.backend_label.setText(f"Backend: {backend_status}")
        self.settings.update_backend_status(backend_status)

    def reset_sim(self):
        self.canvas.particles = []
        self.canvas.ripples = []
        self.canvas.spawn_accumulator = 0.0

    def load_background(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Background", "", 
                                              "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.canvas.set_background(path)

    def save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Config", "", "JSON (*.json)")
        if path:
            with open(path, 'w') as f:
                json.dump(self.canvas.params, f)

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON (*.json)")
        if path:
            try:
                with open(path, 'r') as f:
                    config = json.load(f)
                    for k, v in config.items():
                        self.canvas.update_params(k, v)
                    self.settings.apply_config(config)
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("QMainWindow { background-color: #0b132b; }")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())