import sys
import math
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel,
                               QSpinBox, QDoubleSpinBox, QFileDialog, QGroupBox,
                               QFormLayout, QProgressBar, QTabWidget, QComboBox,
                               QRadioButton, QButtonGroup)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QThread, Signal, QSize


# ---------------------------------------------------------------------------
# Realistic terrain generator (ridges + valleys)
# ---------------------------------------------------------------------------
def make_terrain(seed: int, size: int, octaves: int = 8, persistence: float = 0.5,
                 ridge_factor: float = 0.6) -> np.ndarray:
    """Generate a realistic heightmap with ridges and valleys."""
    rng = np.random.default_rng(seed)
    hmap = np.zeros((size, size), dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    
    for o in range(octaves):
        grid_size = max(2, size // (2 ** (octaves - o - 1)))
        noise = rng.random((grid_size, grid_size)).astype(np.float32)
        xs = np.linspace(0, grid_size - 1, size)
        ys = np.linspace(0, grid_size - 1, size)
        xi, yi = np.meshgrid(xs, ys)
        x0 = np.floor(xi).astype(int); y0 = np.floor(yi).astype(int)
        x1 = np.minimum(x0 + 1, grid_size - 1)
        y1 = np.minimum(y0 + 1, grid_size - 1)
        xf = xi - x0; yf = yi - y0
        top    = noise[y0, x0] * (1 - xf) + noise[y0, x1] * xf
        bottom = noise[y1, x0] * (1 - xf) + noise[y1, x1] * xf
        interp = top * (1 - yf) + bottom * yf
        
        # Ridge noise: invert valleys to create sharp ridges
        ridge = 1.0 - 2.0 * np.abs(interp - 0.5)
        ridge = np.power(ridge, 1.5)  # sharpen ridges
        
        # Blend regular noise with ridge noise
        blended = interp * (1 - ridge_factor) + ridge * ridge_factor
        hmap += blended * amplitude
        total_amp += amplitude
        amplitude *= persistence
    
    hmap /= total_amp
    
    # Add some large-scale features (continental shelves)
    large_scale = rng.random((8, 8)).astype(np.float32)
    xs = np.linspace(0, 7, size)
    ys = np.linspace(0, 7, size)
    xi, yi = np.meshgrid(xs, ys)
    x0 = np.floor(xi).astype(int); y0 = np.floor(yi).astype(int)
    x1 = np.minimum(x0 + 1, 7); y1 = np.minimum(y0 + 1, 7)
    xf = xi - x0; yf = yi - y0
    top = large_scale[y0, x0] * (1 - xf) + large_scale[y0, x1] * xf
    bottom = large_scale[y1, x0] * (1 - xf) + large_scale[y1, x1] * xf
    large = top * (1 - yf) + bottom * yf
    hmap = hmap * 0.7 + large * 0.3
    
    return hmap


# ---------------------------------------------------------------------------
# Colormap
# ---------------------------------------------------------------------------
def colormap(hmap: np.ndarray) -> np.ndarray:
    h = hmap.copy()
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    stops = [
        (0.00,  20,  40,  90),
        (0.35,  40,  90, 160),
        (0.40, 210, 200, 150),
        (0.45,  80, 140,  60),
        (0.60,  60, 110,  40),
        (0.75, 120, 100,  80),
        (0.88, 160, 150, 140),
        (1.00, 255, 255, 255),
    ]
    heights = np.array([s[0] for s in stops], dtype=np.float32)
    colors  = np.array([s[1:] for s in stops], dtype=np.float32)
    img = np.zeros((h.shape[0], h.shape[1], 3), dtype=np.uint8)
    for c in range(3):
        img[..., c] = np.interp(h, heights, colors[:, c]).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Hydraulic erosion (particle-based)
# ---------------------------------------------------------------------------
def erode_particles(heightmap, num_droplets, inertia=0.05, sediment_capacity=4.0,
                    min_slope=0.01, erode_speed=0.3, deposit_speed=0.3,
                    evaporate_speed=0.01, gravity=4.0, max_lifetime=64,
                    start_radius=4, rng=None, progress_cb=None):
    H, W = heightmap.shape
    if rng is None:
        rng = np.random.default_rng()
    hmap = heightmap.astype(np.float32).copy()

    for i in range(num_droplets):
        x = rng.uniform(start_radius, W - start_radius)
        y = rng.uniform(start_radius, H - start_radius)
        dir_x = dir_y = 0.0
        speed = 1.0; water = 1.0; sediment = 0.0

        for _ in range(max_lifetime):
            ix, iy = int(x), int(y)
            if ix < 1 or iy < 1 or ix >= W - 1 or iy >= H - 1:
                break
            u, v = x - ix, y - iy
            h00, h10 = hmap[iy, ix], hmap[iy, ix+1]
            h01, h11 = hmap[iy+1, ix], hmap[iy+1, ix+1]
            gx = (h00-h10)*(1-v) + (h01-h11)*v
            gy = (h00-h01)*(1-u) + (h10-h11)*u
            dir_x = dir_x*inertia - gx*(1-inertia)
            dir_y = dir_y*inertia - gy*(1-inertia)
            n = math.hypot(dir_x, dir_y)
            if n < 1e-8:
                a = rng.uniform(0, 2*math.pi)
                dir_x, dir_y = math.cos(a), math.sin(a)
            else:
                dir_x /= n; dir_y /= n
            nx, ny = x + dir_x, y + dir_y
            nix, niy = int(nx), int(ny)
            if nix < 0 or niy < 0 or nix >= W-1 or niy >= H-1:
                break
            nu, nv = nx - nix, ny - niy
            old_h = h00*(1-u)*(1-v) + h10*u*(1-v) + h01*(1-u)*v + h11*u*v
            new_h = (hmap[niy,nix]*(1-nu)*(1-nv) + hmap[niy,nix+1]*nu*(1-nv) +
                     hmap[niy+1,nix]*(1-nu)*nv + hmap[niy+1,nix+1]*nu*nv)
            dh = new_h - old_h
            cap = max(-dh, min_slope) * speed * water * sediment_capacity
            if sediment > cap or dh > 0:
                amt = min(sediment, dh) if dh > 0 else (sediment - cap) * deposit_speed
                sediment -= amt
                hmap[iy,ix]   += amt*(1-u)*(1-v)
                hmap[iy,ix+1] += amt*u*(1-v)
                hmap[iy+1,ix] += amt*(1-u)*v
                hmap[iy+1,ix+1]+= amt*u*v
            else:
                amt = min((cap - sediment)*erode_speed, -dh)
                sediment += amt
                hmap[iy,ix]   -= amt*(1-u)*(1-v)
                hmap[iy,ix+1] -= amt*u*(1-v)
                hmap[iy+1,ix] -= amt*(1-u)*v
                hmap[iy+1,ix+1]-= amt*u*v
            speed = math.sqrt(max(1e-8, speed*speed + dh*gravity))
            water *= (1 - evaporate_speed)
            x, y = nx, ny
        if progress_cb and i % 500 == 0:
            progress_cb(i, num_droplets)
    if progress_cb:
        progress_cb(num_droplets, num_droplets)
    return hmap


# ---------------------------------------------------------------------------
# Dust storm
# ---------------------------------------------------------------------------
def dust_storm(heightmap, num_particles=30000, wind_angle_deg=45.0,
               wind_speed=2.0, turbulence=0.4, settling_rate=0.02,
               dust_load_init=3.0, deflation_rate=0.1, max_steps=200,
               rng=None, progress_cb=None):
    H, W = heightmap.shape
    if rng is None:
        rng = np.random.default_rng()
    hmap = heightmap.astype(np.float32).copy()

    rad = math.radians(wind_angle_deg)
    wx, wy = math.cos(rad), math.sin(rad)

    padded = np.pad(hmap, 1, mode='edge')
    laplacian = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                 padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * hmap)
    lap_max = max(abs(laplacian.min()), abs(laplacian.max()), 1e-6)
    curvature = laplacian / lap_max

    grad_y, grad_x = np.gradient(hmap)
    slope_wind = grad_x * wx + grad_y * wy

    for i in range(num_particles):
        if abs(wx) > abs(wy):
            if wx > 0:
                x = rng.uniform(0, 3)
            else:
                x = rng.uniform(W - 3, W)
            y = rng.uniform(0, H)
        else:
            if wy > 0:
                y = rng.uniform(0, 3)
            else:
                y = rng.uniform(H - 3, H)
            x = rng.uniform(0, W)

        dust = dust_load_init * rng.uniform(0.5, 1.5)
        local_speed = wind_speed

        for _ in range(max_steps):
            ix, iy = int(x), int(y)
            if ix < 1 or iy < 1 or ix >= W - 1 or iy >= H - 1:
                break

            u, v = x - ix, y - iy
            sw = (slope_wind[iy, ix]*(1-u)*(1-v) + slope_wind[iy, ix+1]*u*(1-v) +
                  slope_wind[iy+1, ix]*(1-u)*v + slope_wind[iy+1, ix+1]*u*v)
            cv = (curvature[iy, ix]*(1-u)*(1-v) + curvature[iy, ix+1]*u*(1-v) +
                  curvature[iy+1, ix]*(1-u)*v + curvature[iy+1, ix+1]*u*v)

            local_speed = max(0.1, wind_speed - sw * 3.0)

            deposit_factor = settling_rate * (
                1.0 + max(0.0, sw * 4.0) + max(0.0, cv * 2.0)
            )
            deposit_amt = dust * deposit_factor
            deposit_amt = min(deposit_amt, dust)

            deflation_factor = 0.0
            if cv < -0.1 and sw < -0.01:
                deflation_factor = deflation_rate * abs(cv) * local_speed * 0.5

            net = deposit_amt - deflation_factor
            hmap[iy,   ix]   += net * (1-u)*(1-v)
            hmap[iy,   ix+1] += net * u*(1-v)
            hmap[iy+1, ix]   += net * (1-u)*v
            hmap[iy+1, ix+1] += net * u*v

            dust -= deposit_amt
            if dust < 0.01:
                break

            turb_x = rng.normal(0, turbulence)
            turb_y = rng.normal(0, turbulence)
            step = local_speed * 1.5
            x += (wx + turb_x) * step
            y += (wy + turb_y) * step

        if progress_cb and i % 500 == 0:
            progress_cb(i, num_particles)
    if progress_cb:
        progress_cb(num_particles, num_particles)
    return hmap


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------
class SimWorker(QThread):
    progress = Signal(int, int)
    finished = Signal(object)

    def __init__(self, func, heightmap, params):
        super().__init__()
        self.func = func
        self.heightmap = heightmap
        self.params = params

    def run(self):
        result = self.func(self.heightmap, progress_cb=self._prog, **self.params)
        self.finished.emit(result)

    def _prog(self, done, total):
        self.progress.emit(done, total)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Terrain Erosion & Dust Storm Simulator")
        self.resize(1200, 750)

        self.size = 512
        self.heightmap = make_terrain(seed=42, size=self.size)
        self.original = self.heightmap.copy()
        self.worker = None
        self.view_mode = "colored"  # "colored" or "heightmap"

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(QSize(512, 512))
        self.image_label.setStyleSheet("background:#222; border-radius:6px;")
        root.addWidget(self.image_label, 1)

        # Panel
        panel = QWidget()
        panel.setFixedWidth(340)
        playout = QVBoxLayout(panel)
        root.addWidget(panel)

        # --- View mode ---
        vg = QGroupBox("View Mode")
        vl = QVBoxLayout()
        self.view_group = QButtonGroup()
        self.radio_colored = QRadioButton("Colored Terrain")
        self.radio_colored.setChecked(True)
        self.radio_heightmap = QRadioButton("Heightmap Image (Grayscale)")
        self.view_group.addButton(self.radio_colored)
        self.view_group.addButton(self.radio_heightmap)
        self.radio_colored.toggled.connect(self._view_changed)
        vl.addWidget(self.radio_colored)
        vl.addWidget(self.radio_heightmap)
        self.stats_label = QLabel("Height: 0.00 - 1.00 | Size: 512×512")
        self.stats_label.setStyleSheet("color: #888; font-size: 10px;")
        vl.addWidget(self.stats_label)
        vg.setLayout(vl)
        playout.addWidget(vg)

        # --- Terrain ---
        tg = QGroupBox("Terrain")
        tf = QFormLayout()
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 99999); self.seed_spin.setValue(42)
        self.size_spin = QSpinBox(); self.size_spin.setRange(128, 1024); self.size_spin.setValue(512)
        self.size_spin.setSingleStep(64)
        tf.addRow("Seed:", self.seed_spin)
        tf.addRow("Size:", self.size_spin)
        btn_gen = QPushButton("Generate New Terrain")
        btn_gen.clicked.connect(self.generate_terrain)
        tf.addRow(btn_gen)
        btn_load = QPushButton("Load Heightmap Image…")
        btn_load.clicked.connect(self.load_heightmap)
        tf.addRow(btn_load)
        tg.setLayout(tf)
        playout.addWidget(tg)

        # --- Image adjustment ---
        adj = QGroupBox("Image Adjustment")
        af = QFormLayout()
        self.spn_contrast = QDoubleSpinBox()
        self.spn_contrast.setRange(0.1, 5.0)
        self.spn_contrast.setValue(1.0)
        self.spn_contrast.setSingleStep(0.1)
        self.spn_contrast.valueChanged.connect(self._adjust_image)
        
        self.spn_brightness = QDoubleSpinBox()
        self.spn_brightness.setRange(-1.0, 1.0)
        self.spn_brightness.setValue(0.0)
        self.spn_brightness.setSingleStep(0.05)
        self.spn_brightness.valueChanged.connect(self._adjust_image)
        
        af.addRow("Contrast:", self.spn_contrast)
        af.addRow("Brightness:", self.spn_brightness)
        btn_reset_adj = QPushButton("Reset Adjustment")
        btn_reset_adj.clicked.connect(self._reset_adjustment)
        af.addRow(btn_reset_adj)
        adj.setLayout(af)
        playout.addWidget(adj)

        # --- Tabs for Erosion / Dust ---
        tabs = QTabWidget()
        playout.addWidget(tabs)

        # == Erosion tab ==
        ew = QWidget()
        ef = QFormLayout(ew)
        self.spn_droplets = QSpinBox(); self.spn_droplets.setRange(1000,500000); self.spn_droplets.setValue(50000); self.spn_droplets.setSingleStep(5000)
        self.spn_inertia  = QDoubleSpinBox(); self.spn_inertia.setRange(0,1); self.spn_inertia.setValue(0.05); self.spn_inertia.setSingleStep(0.01)
        self.spn_capacity = QDoubleSpinBox(); self.spn_capacity.setRange(0,20); self.spn_capacity.setValue(4.0); self.spn_capacity.setSingleStep(0.1)
        self.spn_erode    = QDoubleSpinBox(); self.spn_erode.setRange(0,1); self.spn_erode.setValue(0.3); self.spn_erode.setSingleStep(0.05)
        self.spn_deposit  = QDoubleSpinBox(); self.spn_deposit.setRange(0,1); self.spn_deposit.setValue(0.3); self.spn_deposit.setSingleStep(0.05)
        self.spn_evap     = QDoubleSpinBox(); self.spn_evap.setRange(0.001,0.2); self.spn_evap.setValue(0.01); self.spn_evap.setSingleStep(0.001); self.spn_evap.setDecimals(3)
        self.spn_gravity  = QDoubleSpinBox(); self.spn_gravity.setRange(0.1,20); self.spn_gravity.setValue(4.0)
        self.spn_lifetime = QSpinBox(); self.spn_lifetime.setRange(8,256); self.spn_lifetime.setValue(64)
        ef.addRow("Droplets:", self.spn_droplets)
        ef.addRow("Inertia:", self.spn_inertia)
        ef.addRow("Sediment cap:", self.spn_capacity)
        ef.addRow("Erode speed:", self.spn_erode)
        ef.addRow("Deposit speed:", self.spn_deposit)
        ef.addRow("Evaporation:", self.spn_evap)
        ef.addRow("Gravity:", self.spn_gravity)
        ef.addRow("Max lifetime:", self.spn_lifetime)
        tabs.addTab(ew, "💧 Hydraulic Erosion")

        # == Dust Storm tab ==
        dw = QWidget()
        df = QFormLayout(dw)

        self.spn_dust_particles = QSpinBox()
        self.spn_dust_particles.setRange(1000, 500000)
        self.spn_dust_particles.setValue(30000)
        self.spn_dust_particles.setSingleStep(5000)

        self.spn_wind_angle = QDoubleSpinBox()
        self.spn_wind_angle.setRange(0, 360)
        self.spn_wind_angle.setValue(225)
        self.spn_wind_angle.setSingleStep(5)
        self.spn_wind_angle.setSuffix("°")

        self.spn_wind_speed = QDoubleSpinBox()
        self.spn_wind_speed.setRange(0.2, 10)
        self.spn_wind_speed.setValue(2.0)
        self.spn_wind_speed.setSingleStep(0.2)

        self.spn_turbulence = QDoubleSpinBox()
        self.spn_turbulence.setRange(0, 2)
        self.spn_turbulence.setValue(0.4)
        self.spn_turbulence.setSingleStep(0.05)

        self.spn_settling = QDoubleSpinBox()
        self.spn_settling.setRange(0.001, 0.2)
        self.spn_settling.setValue(0.02)
        self.spn_settling.setSingleStep(0.005)
        self.spn_settling.setDecimals(3)

        self.spn_dust_load = QDoubleSpinBox()
        self.spn_dust_load.setRange(0.1, 20)
        self.spn_dust_load.setValue(3.0)
        self.spn_dust_load.setSingleStep(0.5)

        self.spn_deflation = QDoubleSpinBox()
        self.spn_deflation.setRange(0, 1)
        self.spn_deflation.setValue(0.1)
        self.spn_deflation.setSingleStep(0.02)

        self.spn_dust_steps = QSpinBox()
        self.spn_dust_steps.setRange(20, 1000)
        self.spn_dust_steps.setValue(200)

        df.addRow("Particles:", self.spn_dust_particles)
        df.addRow("Wind angle:", self.spn_wind_angle)
        df.addRow("Wind speed:", self.spn_wind_speed)
        df.addRow("Turbulence:", self.spn_turbulence)
        df.addRow("Settling rate:", self.spn_settling)
        df.addRow("Dust load:", self.spn_dust_load)
        df.addRow("Deflation:", self.spn_deflation)
        df.addRow("Max steps:", self.spn_dust_steps)

        self.wind_preset = QComboBox()
        self.wind_preset.addItems(["Custom", "N → S (180°)", "S → N (0°)",
                                    "W → E (90°)", "E → W (270°)",
                                    "SW → NE (45°)", "NW → SE (135°)",
                                    "NE → SW (225°)", "SE → NW (315°)"])
        self.wind_preset.currentIndexChanged.connect(self._wind_preset_changed)
        df.addRow("Preset:", self.wind_preset)

        tabs.addTab(dw, "🌪️ Dust Storm")

        # --- Run buttons ---
        rg = QGroupBox("Run")
        rl = QVBoxLayout()
        self.btn_erode = QPushButton("💧 Run Hydraulic Erosion")
        self.btn_erode.clicked.connect(self.run_erosion)
        self.btn_dust = QPushButton("🌪️ Run Dust Storm")
        self.btn_dust.clicked.connect(self.run_dust)
        self.btn_reset = QPushButton("↺ Reset to Original")
        self.btn_reset.clicked.connect(self.reset_terrain)
        self.btn_save = QPushButton("💾 Save Image…")
        self.btn_save.clicked.connect(self.save_image)
        self.progress = QProgressBar()
        self.status = QLabel("Ready.")
        rl.addWidget(self.btn_erode)
        rl.addWidget(self.btn_dust)
        rl.addWidget(self.btn_reset)
        rl.addWidget(self.btn_save)
        rl.addWidget(self.progress)
        rl.addWidget(self.status)
        rg.setLayout(rl)
        playout.addWidget(rg)
        playout.addStretch()

        self.update_display()
        self._update_stats()

    # --- View mode ---
    def _view_changed(self):
        self.view_mode = "heightmap" if self.radio_heightmap.isChecked() else "colored"
        self.update_display()

    def _update_stats(self):
        hmin, hmax = self.heightmap.min(), self.heightmap.max()
        H, W = self.heightmap.shape
        self.stats_label.setText(f"Height: {hmin:.3f} - {hmax:.3f} | Size: {W}×{H}")

    def _adjust_image(self):
        # Apply contrast/brightness to the working heightmap for display
        self.update_display()

    def _reset_adjustment(self):
        self.spn_contrast.setValue(1.0)
        self.spn_brightness.setValue(0.0)

    # --- Wind preset ---
    def _wind_preset_changed(self, idx):
        angles = [None, 180, 0, 90, 270, 45, 135, 225, 315]
        if idx > 0 and angles[idx] is not None:
            self.spn_wind_angle.setValue(angles[idx])

    # --- Helpers ---
    def update_display(self):
        hmap = self.heightmap.copy()
        
        # Apply contrast/brightness
        contrast = self.spn_contrast.value()
        brightness = self.spn_brightness.value()
        hmap = (hmap - hmap.mean()) * contrast + hmap.mean() + brightness
        
        if self.view_mode == "colored":
            rgb = colormap(hmap)
        else:
            # Grayscale heightmap
            h = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
            gray = (h * 255).astype(np.uint8)
            rgb = np.stack([gray, gray, gray], axis=-1)
        
        H, W, _ = rgb.shape
        img = QImage(rgb.data, W, H, 3*W, QImage.Format_RGB888).copy()
        self.image_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def generate_terrain(self):
        self.size = self.size_spin.value()
        self.heightmap = make_terrain(seed=self.seed_spin.value(), size=self.size)
        self.original = self.heightmap.copy()
        self._update_stats()
        self.update_display()
        self.status.setText(f"Generated {self.size}×{self.size} terrain.")

    def load_heightmap(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Heightmap Image", "",
                                              "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path: return
        img = QImage(path)
        if img.isNull():
            self.status.setText("Failed to load image.")
            return
        
        w, h = img.width(), img.height()
        gray = img.convertToFormat(QImage.Format_Grayscale8)
        arr = np.frombuffer(gray.constBits(), dtype=np.uint8).reshape(h, w).astype(np.float32) / 255.0
        
        # Auto-normalize to [0, 1]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        
        self.heightmap = arr
        self.original = arr.copy()
        self.size = max(h, w)
        self.size_spin.setValue(self.size)
        self._update_stats()
        self.update_display()
        self.status.setText(f"Loaded {w}×{h} heightmap from {path.split('/')[-1]}")

    def reset_terrain(self):
        self.heightmap = self.original.copy()
        self._update_stats()
        self.update_display()
        self.status.setText("Reset to original.")

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save", "terrain.png", "PNG (*.png);;JPEG (*.jpg)")
        if path:
            hmap = self.heightmap.copy()
            contrast = self.spn_contrast.value()
            brightness = self.spn_brightness.value()
            hmap = (hmap - hmap.mean()) * contrast + hmap.mean() + brightness
            
            if self.view_mode == "colored":
                rgb = colormap(hmap)
            else:
                h = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
                gray = (h * 255).astype(np.uint8)
                rgb = np.stack([gray, gray, gray], axis=-1)
            
            H, W, _ = rgb.shape
            QImage(rgb.data, W, H, 3*W, QImage.Format_RGB888).copy().save(path)
            self.status.setText(f"Saved → {path}")

    def _lock(self, locked):
        self.btn_erode.setEnabled(not locked)
        self.btn_dust.setEnabled(not locked)

    def _start_worker(self, func, params, label):
        if self.worker and self.worker.isRunning(): return
        self._lock(True)
        self.progress.setValue(0)
        self.status.setText(f"{label}…")
        self.worker = SimWorker(func, self.heightmap.copy(), params)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, done, total):
        self.progress.setMaximum(total)
        self.progress.setValue(done)

    def _on_finished(self, result):
        self.heightmap = result
        self._update_stats()
        self.update_display()
        self._lock(False)
        self.status.setText("Done.")

    def run_erosion(self):
        self._start_worker(erode_particles, dict(
            num_droplets=self.spn_droplets.value(),
            inertia=self.spn_inertia.value(),
            sediment_capacity=self.spn_capacity.value(),
            erode_speed=self.spn_erode.value(),
            deposit_speed=self.spn_deposit.value(),
            evaporate_speed=self.spn_evap.value(),
            gravity=self.spn_gravity.value(),
            max_lifetime=self.spn_lifetime.value(),
            rng=np.random.default_rng(),
        ), "Eroding")

    def run_dust(self):
        self._start_worker(dust_storm, dict(
            num_particles=self.spn_dust_particles.value(),
            wind_angle_deg=self.spn_wind_angle.value(),
            wind_speed=self.spn_wind_speed.value(),
            turbulence=self.spn_turbulence.value(),
            settling_rate=self.spn_settling.value(),
            dust_load_init=self.spn_dust_load.value(),
            deflation_rate=self.spn_deflation.value(),
            max_steps=self.spn_dust_steps.value(),
            rng=np.random.default_rng(),
        ), "Dust storm blowing")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()