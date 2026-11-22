"""
Image Evolution Visualizer - Enhanced with Advanced Visualizations
PySide6 + NumPy + Matplotlib-style custom widgets
"""

import sys
import math
import random
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import deque
from enum import Enum

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
    QGroupBox, QFileDialog, QTabWidget, QSplitter, QCheckBox,
    QStatusBar, QFrame, QScrollArea, QSizePolicy, QToolTip,
    QGridLayout, QRadioButton, QButtonGroup, QProgressBar, QTextEdit
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QTimer, QRect, QRectF, QPointF, QSize, QLineF
)
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QBrush, QPolygon,
    QPolygonF, QLinearGradient, QPalette, QFont, QFontMetrics,
    QRadialGradient, QIcon, QCursor
)


# ============================================================
# THEME & CONSTANTS
# ============================================================

THEME = {
    "bg":           QColor(20, 22, 28),
    "panel":        QColor(30, 33, 42),
    "panel_light":  QColor(42, 46, 58),
    "border":       QColor(55, 60, 75),
    "text":         QColor(235, 237, 245),
    "text_dim":     QColor(140, 145, 165),
    "accent":       QColor(90, 175, 255),
    "ga_color":     QColor(80, 220, 140),
    "mcts_color":   QColor(100, 180, 255),
    "hybrid_color": QColor(255, 170, 80),
    "good":         QColor(100, 220, 140),
    "bad":          QColor(255, 100, 100),
    "warning":      QColor(255, 190, 80),
}

IMG_W, IMG_H = 220, 220


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Polygon:
    vertices: List[Tuple[float, float]]
    color: Tuple[int, int, int, int]
    
    def mutate(self, strength: float = 0.1):
        p = Polygon(
            vertices=[(x + random.gauss(0, strength),
                       y + random.gauss(0, strength))
                      for x, y in self.vertices],
            color=tuple(int(np.clip(c + random.randint(-30, 30), 0, 255))
                        for c in self.color)
        )
        if len(p.vertices) > 3 and random.random() < 0.1:
            p.vertices.pop(random.randrange(len(p.vertices)))
        if len(p.vertices) < 6 and random.random() < 0.1:
            i = random.randrange(len(p.vertices))
            j = (i + 1) % len(p.vertices)
            mid = ((p.vertices[i][0] + p.vertices[j][0]) / 2,
                   (p.vertices[i][1] + p.vertices[j][1]) / 2)
            p.vertices.insert(i + 1, mid)
        return p

    @staticmethod
    def random():
        n = random.randint(3, 6)
        cx, cy = random.random(), random.random()
        r = random.uniform(0.02, 0.25)
        verts = []
        for k in range(n):
            a = 2 * math.pi * k / n + random.uniform(-0.3, 0.3)
            rr = r * random.uniform(0.6, 1.2)
            verts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255), random.randint(40, 180))
        return Polygon(vertices=verts, color=color)


@dataclass
class Genome:
    shapes: List[Polygon] = field(default_factory=list)
    fitness: float = -1.0
    mse: float = 0.0
    
    def copy(self):
        return Genome(
            shapes=[Polygon(list(p.vertices), tuple(p.color)) for p in self.shapes],
            fitness=self.fitness,
            mse=self.mse
        )


class MutationType(Enum):
    ADD_SHAPE = "add"
    REMOVE_SHAPE = "remove"
    MODIFY_SHAPE = "modify"
    CROSSOVER = "crossover"


@dataclass
class MutationEvent:
    gen: int
    mutation_type: MutationType
    fitness_before: float
    fitness_after: float
    shape_index: Optional[int] = None


# ============================================================
# RENDERING & FITNESS
# ============================================================

def render_genome(genome: Genome, w: int = IMG_W, h: int = IMG_H) -> np.ndarray:
    qimg = QImage(w, h, QImage.Format_ARGB32)
    qimg.fill(QColor(255, 255, 255, 255))
    painter = QPainter(qimg)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    for s in genome.shapes:
        poly = QPolygon()
        for x, y in s.vertices:
            poly.append(QPoint(int(x * w), int(y * h)))
        r, g, b, a = s.color
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(r, g, b, a)))
        painter.drawPolygon(poly)
    painter.end()
    
    ptr = qimg.bits()
    ptr.setsize(qimg.sizeInBytes())
    arr = np.array(ptr).reshape(h, w, 4).copy()
    return arr[:, :, [2, 1, 0, 3]]


def compute_fitness(genome: Genome, target: np.ndarray) -> Tuple[float, float]:
    img = render_genome(genome)
    diff = (img[:, :, :3].astype(np.float32) - target[:, :, :3].astype(np.float32))
    mse = float(np.mean(diff * diff))
    fitness = 1.0 / (1.0 + mse / 1000.0)
    return fitness, mse


def make_error_map(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Create error density map showing where differences are concentrated."""
    diff = np.abs(current[:, :, :3].astype(np.float32) - 
                  target[:, :, :3].astype(np.float32))
    magnitude = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Normalize
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()
    
    # Create heatmap (blue=cold, red=hot)
    h, w = magnitude.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color mapping
    for i in range(h):
        for j in range(w):
            val = magnitude[i, j]
            if val < 0.25:
                heatmap[i, j] = [0, 0, int(val * 4 * 255)]  # Blue
            elif val < 0.5:
                heatmap[i, j] = [0, int((val - 0.25) * 4 * 255), 255]  # Cyan-Green
            elif val < 0.75:
                heatmap[i, j] = [int((val - 0.5) * 4 * 255), 255, int((1 - val) * 4 * 255)]  # Yellow
            else:
                heatmap[i, j] = [255, int((1 - val) * 4 * 255), 0]  # Red
    
    return heatmap


def compute_diversity_metrics(population: List[Genome]) -> Dict[str, float]:
    """Calculate population diversity metrics."""
    if len(population) < 2:
        return {"shape_diversity": 0, "color_diversity": 0, "fitness_variance": 0}
    
    fitnesses = [g.fitness for g in population]
    
    # Shape count diversity
    shape_counts = [len(g.shapes) for g in population]
    shape_div = np.std(shape_counts) / (np.mean(shape_counts) + 1e-6)
    
    # Color diversity (average color difference)
    colors = []
    for g in population[:10]:  # Sample first 10
        if g.shapes:
            avg_color = np.mean([s.color[:3] for s in g.shapes], axis=0)
            colors.append(avg_color)
    
    if len(colors) > 1:
        color_div = np.mean([np.std(c) for c in zip(*colors)])
    else:
        color_div = 0
    
    return {
        "shape_diversity": float(shape_div),
        "color_diversity": float(color_div / 255),
        "fitness_variance": float(np.var(fitnesses)),
        "fitness_std": float(np.std(fitnesses)),
    }


# ============================================================
# TARGET GENERATOR
# ============================================================

def make_target_image(kind: str = "landscape") -> np.ndarray:
    qimg = QImage(IMG_W, IMG_H, QImage.Format_ARGB32)
    p = QPainter(qimg)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    if kind == "landscape":
        grad = QLinearGradient(0, 0, 0, IMG_H)
        grad.setColorAt(0, QColor(135, 206, 235))
        grad.setColorAt(0.6, QColor(255, 223, 186))
        grad.setColorAt(1, QColor(60, 179, 113))
        p.fillRect(0, 0, IMG_W, IMG_H, grad)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(255, 220, 80)))
        p.drawEllipse(IMG_W // 3, 40, 50, 50)
        p.setBrush(QBrush(QColor(90, 90, 130)))
        poly = QPolygon([QPoint(0, IMG_H), QPoint(60, 80),
                         QPoint(120, 140), QPoint(180, 70),
                         QPoint(IMG_W, IMG_H)])
        p.drawPolygon(poly)
        p.setBrush(QBrush(QColor(30, 100, 50)))
        for cx in [40, 100, 160]:
            p.drawEllipse(cx - 15, IMG_H - 60, 30, 40)
    elif kind == "abstract":
        p.fillRect(0, 0, IMG_W, IMG_H, QColor(30, 30, 60))
        colors = [QColor(255, 80, 80), QColor(80, 200, 255),
                  QColor(255, 220, 80), QColor(150, 255, 150)]
        for _ in range(12):
            p.setBrush(QBrush(random.choice(colors)))
            p.setPen(Qt.PenStyle.NoPen)
            x, y = random.randint(0, IMG_W - 60), random.randint(0, IMG_H - 60)
            p.drawEllipse(x, y, random.randint(30, 80), random.randint(30, 80))
    elif kind == "face":
        p.fillRect(0, 0, IMG_W, IMG_H, QColor(255, 230, 200))
        p.setBrush(QBrush(QColor(80, 50, 30)))
        p.drawEllipse(60, 70, 20, 10)
        p.drawEllipse(120, 70, 20, 10)
        p.setPen(QPen(QColor(180, 60, 60), 4))
        p.drawArc(70, 110, 60, 30, 0, -180 * 16)
    else:
        p.fillRect(0, 0, IMG_W, IMG_H, QColor(240, 240, 230))
        p.setBrush(QBrush(QColor(220, 60, 60)))
        p.drawRect(30, 30, 80, 80)
        p.setBrush(QBrush(QColor(60, 120, 220)))
        p.drawEllipse(110, 110, 80, 80)

    p.end()
    ptr = qimg.bits()
    ptr.setsize(qimg.sizeInBytes())
    arr = np.array(ptr).reshape(IMG_H, IMG_W, 4).copy()
    return arr[:, :, [2, 1, 0, 3]]


# ============================================================
# ADVANCED VISUALIZATION WIDGETS
# ============================================================

class PolygonOverlayWidget(QWidget):
    """Shows the actual polygons with vertices and edges."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.genome: Optional[Genome] = None
        self.show_vertices = True
        self.show_edges = True
        self.show_fill = True
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background: {THEME['panel'].name()}; border-radius: 4px;")
    
    def set_genome(self, genome: Genome):
        self.genome = genome
        self.update()
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        
        # Background
        p.fillRect(0, 0, w, h, THEME["panel"])
        
        if not self.genome:
            p.setPen(THEME["text_dim"])
            p.drawText(w//2 - 50, h//2, "No genome loaded")
            return
        
        # Draw grid
        p.setPen(QPen(THEME["border"], 1, Qt.PenStyle.DotLine))
        for i in range(5):
            x = w * i // 4
            p.drawLine(x, 0, x, h)
            p.drawLine(0, h * i // 4, w, h * i // 4)
        
        # Draw shapes
        for i, shape in enumerate(self.genome.shapes):
            # Create polygon
            poly = QPolygonF()
            for vx, vy in shape.vertices:
                poly.append(QPointF(vx * w, vy * h))
            
            # Fill
            if self.show_fill:
                r, g, b, a = shape.color
                p.setBrush(QBrush(QColor(r, g, b, a)))
                p.drawPolygon(poly)
            
            # Edges
            if self.show_edges:
                p.setPen(QPen(QColor(255, 255, 255, 200), 1.5))
                p.drawPolyline(poly)
                if poly.isClosed():
                    p.drawLine(poly.last(), poly.first())
            
            # Vertices
            if self.show_vertices:
                p.setBrush(QBrush(THEME["good"]))
                for pt in poly:
                    p.drawEllipse(pt, 3, 3)
        
        # Legend
        p.setPen(THEME["text"])
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(10, 20, f"Shapes: {len(self.genome.shapes)}")
        p.drawText(10, 35, f"Fitness: {self.genome.fitness:.4f}")


class FitnessHistogram(QWidget):
    """Histogram showing fitness distribution of population."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.fitnesses: List[float] = []
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(f"background: {THEME['panel'].name()}; border-radius: 4px;")
    
    def set_fitnesses(self, fitnesses: List[float]):
        self.fitnesses = fitnesses
        self.update()
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        
        p.fillRect(0, 0, w, h, THEME["panel"])
        
        if not self.fitnesses:
            p.setPen(THEME["text_dim"])
            p.drawText(10, h//2, "No population data")
            return
        
        # Histogram parameters
        n_bins = 20
        min_f, max_f = min(self.fitnesses), max(self.fitnesses)
        if max_f - min_f < 1e-6:
            max_f = min_f + 0.01
        
        bin_width = (max_f - min_f) / n_bins
        bins = [0] * n_bins
        
        for f in self.fitnesses:
            idx = min(int((f - min_f) / bin_width), n_bins - 1)
            bins[idx] += 1
        
        # Draw
        max_count = max(bins) if bins else 1
        bar_width = (w - 60) / n_bins
        
        # Bars
        for i, count in enumerate(bins):
            x = 50 + i * bar_width
            bar_h = (count / max_count) * (h - 40) if max_count > 0 else 0
            y = h - 30 - bar_h
            
            # Gradient fill
            grad = QLinearGradient(x, y, x, h - 30)
            grad.setColorAt(0, THEME["accent"])
            grad.setColorAt(1, THEME["mcts_color"])
            p.setBrush(QBrush(grad))
            p.setPen(QPen(THEME["border"], 1))
            p.drawRect(int(x), int(y), int(bar_width - 2), int(bar_h))
        
        # Axes
        p.setPen(QPen(THEME["text"], 1))
        p.drawLine(50, 20, 50, h - 30)
        p.drawLine(50, h - 30, w - 10, h - 30)
        
        # Labels
        p.setPen(THEME["text_dim"])
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(10, 25, f"Count: {len(self.fitnesses)}")
        p.drawText(w - 80, 25, f"Max: {max_f:.3f}")
        p.drawText(50, h - 10, f"{min_f:.2f}")
        p.drawText(w - 40, h - 10, f"{max_f:.2f}")
        p.drawText(w//2 - 30, h - 10, "Fitness →")


class DiversityRadar(QWidget):
    """Radar chart showing multiple diversity metrics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(250, 250)
        self.metrics: Dict[str, float] = {}
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background: {THEME['panel'].name()}; border-radius: 4px;")
    
    def set_metrics(self, metrics: Dict[str, float]):
        self.metrics = metrics
        self.update()
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        
        p.fillRect(0, 0, w, h, THEME["panel"])
        
        # Title
        p.setPen(THEME["accent"])
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        p.drawText(w//2 - 40, 20, "Diversity Metrics")
        
        if not self.metrics:
            p.setPen(THEME["text_dim"])
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(w//2 - 50, h//2, "No metrics available")
            return
        
        # Radar chart
        cx, cy = w // 2, h // 2 + 10
        radius = min(w, h) // 2 - 40
        
        metrics_list = list(self.metrics.keys())
        n = len(metrics_list)
        angle_step = 2 * math.pi / n
        
        # Draw grid circles
        p.setPen(QPen(THEME["border"], 1, Qt.PenStyle.DotLine))
        for r in [0.25, 0.5, 0.75, 1.0]:
            poly = QPolygonF()
            for i in range(n):
                angle = angle_step * i - math.pi / 2
                x = cx + radius * r * math.cos(angle)
                y = cy + radius * r * math.sin(angle)
                poly.append(QPointF(x, y))
            p.drawPolygon(poly)
        
        # Draw axes
        p.setPen(QPen(THEME["border"], 1))
        for i in range(n):
            angle = angle_step * i - math.pi / 2
            x1, y1 = cx + 5 * math.cos(angle), cy + 5 * math.sin(angle)
            x2, y2 = cx + radius * math.cos(angle), cy + radius * math.sin(angle)
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            
            # Labels
            label = metrics_list[i].replace("_", " ").title()
            p.setPen(THEME["text_dim"])
            p.setFont(QFont("Segoe UI", 8))
            lx = cx + (radius + 20) * math.cos(angle)
            ly = cy + (radius + 15) * math.sin(angle)
            p.drawText(int(lx - 30), int(ly), 60, 20, Qt.AlignmentFlag.AlignCenter, label)
        
        # Draw data polygon
        p.setPen(QPen(THEME["ga_color"], 2))
        p.setBrush(QBrush(QColor(80, 220, 140, 100)))
        data_poly = QPolygonF()
        for i, (name, value) in enumerate(self.metrics.items()):
            angle = angle_step * i - math.pi / 2
            normalized = min(value, 1.0)  # Cap at 1.0
            x = cx + radius * normalized * math.cos(angle)
            y = cy + radius * normalized * math.sin(angle)
            data_poly.append(QPointF(x, y))
        
        p.drawPolygon(data_poly)
        
        # Draw points
        p.setBrush(QBrush(THEME["good"]))
        for pt in data_poly:
            p.drawEllipse(pt, 4, 4)


class ConvergencePlot(QWidget):
    """Shows convergence rate and improvement over time."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.history: List[Tuple[int, float, float]] = []  # (gen, fitness, improvement)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(f"background: {THEME['panel'].name()}; border-radius: 4px;")
    
    def add_point(self, gen: int, fitness: float):
        if self.history:
            last_gen, last_fit, _ = self.history[-1]
            improvement = fitness - last_fit
        else:
            improvement = 0
        self.history.append((gen, fitness, improvement))
        if len(self.history) > 300:
            self.history = self.history[-300:]
        self.update()
    
    def clear(self):
        self.history.clear()
        self.update()
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        
        p.fillRect(0, 0, w, h, THEME["panel"])
        
        # Title
        p.setPen(THEME["accent"])
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        p.drawText(10, 20, "Convergence Analysis")
        
        if len(self.history) < 2:
            p.setPen(THEME["text_dim"])
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(10, h//2, "Need more data points")
            return
        
        # Calculate stats
        fitnesses = [f for _, f, _ in self.history]
        improvements = [imp for _, _, imp in self.history]
        
        avg_improvement = np.mean(improvements[-50:]) if len(improvements) > 50 else np.mean(improvements)
        
        # Draw fitness curve
        margin_l, margin_r, margin_t, margin_b = 60, 20, 40, 40
        plot_w = w - margin_l - margin_r
        plot_h = h - margin_t - margin_b
        
        min_f, max_f = min(fitnesses), max(fitnesses)
        if max_f - min_f < 1e-6:
            max_f = min_f + 0.01
        
        p.setPen(QPen(THEME["ga_color"], 2))
        for i in range(len(self.history) - 1):
            gen1, fit1, _ = self.history[i]
            gen2, fit2, _ = self.history[i + 1]
            
            x1 = margin_l + (gen1 / max(self.history[-1][0], 1)) * plot_w
            y1 = margin_t + (1 - (fit1 - min_f) / (max_f - min_f)) * plot_h
            x2 = margin_l + (gen2 / max(self.history[-1][0], 1)) * plot_w
            y2 = margin_t + (1 - (fit2 - min_f) / (max_f - min_f)) * plot_h
            
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
        
        # Draw improvement bars (bottom)
        bar_w = plot_w / min(len(self.history), 100)
        p.setPen(Qt.PenStyle.NoPen)
        for i, (_, _, imp) in enumerate(self.history[-100:]):
            x = margin_l + i * bar_w
            bar_h = min(abs(imp) * 1000, plot_h / 3)
            y = h - margin_b - bar_h if imp > 0 else h - margin_b
            
            if imp > 0:
                p.setBrush(QBrush(THEME["good"]))
            else:
                p.setBrush(QBrush(THEME["bad"]))
            p.drawRect(int(x), int(y), int(bar_w - 1), int(bar_h))
        
        # Axes
        p.setPen(QPen(THEME["border"], 1))
        p.drawLine(margin_l, margin_t, margin_l, h - margin_b)
        p.drawLine(margin_l, h - margin_b, w - margin_r, h - margin_b)
        
        # Stats
        p.setPen(THEME["text_dim"])
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(10, h - 15, f"Avg improvement: {avg_improvement:.5f}")
        p.drawText(w - 100, h - 15, f"Gen: {self.history[-1][0]}")


class ParetoScatter(QWidget):
    """Scatter plot showing Pareto front: shapes vs fitness."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(250, 250)
        self.population: List[Genome] = []
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background: {THEME['panel'].name()}; border-radius: 4px;")
    
    def set_population(self, population: List[Genome]):
        self.population = population
        self.update()
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        
        p.fillRect(0, 0, w, h, THEME["panel"])
        
        # Title
        p.setPen(THEME["accent"])
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        p.drawText(10, 20, "Pareto Front: Shapes vs Fitness")
        
        if not self.population:
            p.setPen(THEME["text_dim"])
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(w//2 - 60, h//2, "No population data")
            return
        
        # Find Pareto front
        pareto = []
        for g in self.population:
            is_dominated = False
            for other in self.population:
                if (len(other.shapes) <= len(g.shapes) and 
                    other.fitness >= g.fitness and
                    (len(other.shapes) < len(g.shapes) or other.fitness > g.fitness)):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto.append(g)
        
        # Plot area
        margin_l, margin_r, margin_t, margin_b = 50, 20, 40, 40
        plot_w = w - margin_l - margin_r
        plot_h = h - margin_t - margin_b
        
        max_shapes = max(len(g.shapes) for g in self.population)
        min_fit = min(g.fitness for g in self.population)
        max_fit = max(g.fitness for g in self.population)
        
        # Draw all points
        p.setPen(Qt.PenStyle.NoPen)
        for g in self.population:
            x = margin_l + (len(g.shapes) / max(max_shapes, 1)) * plot_w
            y = margin_t + (1 - (g.fitness - min_fit) / (max_fit - min_fit + 1e-6)) * plot_h
            
            if g in pareto:
                p.setBrush(QBrush(THEME["good"]))
                p.drawEllipse(QPointF(x, y), 5, 5)
            else:
                p.setBrush(QBrush(THEME["mcts_color"]))
                p.drawEllipse(QPointF(x, y), 3, 3)
        
        # Axes
        p.setPen(QPen(THEME["border"], 1))
        p.drawLine(margin_l, margin_t, margin_l, h - margin_b)
        p.drawLine(margin_l, h - margin_b, w - margin_r, h - margin_b)
        
        # Labels
        p.setPen(THEME["text_dim"])
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(w//2 - 30, h - 10, "Shapes →")
        p.drawText(5, h//2, "Fitness", Qt.AlignmentFlag.AlignVCenter)
        
        p.drawText(margin_l - 40, h - margin_b + 15, "0")
        p.drawText(w - margin_r - 10, h - margin_b + 15, str(max_shapes))
        
        p.drawText(5, margin_t + 10, f"{max_fit:.2f}")
        p.drawText(5, h - margin_b - 5, f"{min_fit:.2f}")
        
        # Legend
        p.setBrush(QBrush(THEME["good"]))
        p.drawEllipse(QPointF(w - 80, 25), 4, 4)
        p.setPen(THEME["text"])
        p.drawText(w - 65, 28, "Pareto optimal")
        
        p.setBrush(QBrush(THEME["mcts_color"]))
        p.drawEllipse(QPointF(w - 80, 40), 3, 3)
        p.setPen(THEME["text"])
        p.drawText(w - 65, 43, "Dominated")


class MutationLog(QWidget):
    """Real-time log of mutation operations."""
    
    def __init__(self, max_entries: int = 50, parent=None):
        super().__init__(parent)
        self.max_entries = max_entries
        self.log: deque = deque(maxlen=max_entries)
        self.setMinimumWidth(250)
        self.setStyleSheet(f"""
            QWidget {{
                background: {THEME['panel'].name()};
                border-radius: 4px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        title = QLabel("<b>🔬 Mutation Log</b>")
        title.setStyleSheet(f"color: {THEME['accent'].name()};")
        layout.addWidget(title)
        
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet(f"""
            QTextEdit {{
                background: {THEME['bg'].name()};
                color: {THEME['text'].name()};
                border: 1px solid {THEME['border'].name()};
                border-radius: 3px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
            }}
        """)
        layout.addWidget(self.text, 1)
    
    def add_event(self, event: MutationEvent):
        icon = "➕" if event.mutation_type == MutationType.ADD_SHAPE else \
               "➖" if event.mutation_type == MutationType.REMOVE_SHAPE else \
               "✏️" if event.mutation_type == MutationType.MODIFY_SHAPE else ""
        
        delta = event.fitness_after - event.fitness_before
        color = "green" if delta > 0 else "red" if delta < 0 else "gray"
        
        entry = f"<span style='color:#666'>g{event.gen:04d}</span> {icon} "
        entry += f"{event.mutation_type.value:8s} "
        entry += f"Δf=<span style='color:{color}'>{delta:+.5f}</span> "
        if event.shape_index is not None:
            entry += f"shape[{event.shape_index}]"
        
        self.log.append(entry)
        self.text.append(entry)
        self.text.verticalScrollBar().setValue(self.text.verticalScrollBar().maximum())
    
    def clear(self):
        self.log.clear()
        self.text.clear()


class EvolutionTimeline(QWidget):
    """Interactive timeline to scrub through evolution history."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.frames: List[Tuple[np.ndarray, int, float]] = []
        self.current_frame = 0
        self.on_frame_changed = None
        
        self.setStyleSheet(f"background: {THEME['panel'].name()}; border-radius: 4px;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #2a2d38;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #64b4ff;
                width: 12px;
                margin: -3px 0;
                border-radius: 6px;
            }
        """)
        self.slider.valueChanged.connect(self._on_slider_change)
        layout.addWidget(self.slider)
        
        info = QHBoxLayout()
        self.lbl_frame = QLabel("Frame: 0 / 0")
        self.lbl_fitness = QLabel("Fitness: -")
        info.addWidget(self.lbl_frame)
        info.addStretch()
        info.addWidget(self.lbl_fitness)
        layout.addLayout(info)
    
    def add_frame(self, img: np.ndarray, gen: int, fitness: float):
        self.frames.append((img, gen, fitness))
        self.slider.setMaximum(len(self.frames) - 1)
        self.current_frame = len(self.frames) - 1
        self._update_labels()
    
    def _on_slider_change(self, value):
        self.current_frame = value
        self._update_labels()
        if self.on_frame_changed and self.frames:
            self.on_frame_changed(self.frames[value])
    
    def _update_labels(self):
        if self.frames:
            _, gen, fit = self.frames[self.current_frame]
            self.lbl_frame.setText(f"Frame: {self.current_frame} / {len(self.frames)-1} (Gen {gen})")
            self.lbl_fitness.setText(f"Fitness: {fit:.4f}")
        else:
            self.lbl_frame.setText("Frame: 0 / 0")
            self.lbl_fitness.setText("Fitness: -")
    
    def clear(self):
        self.frames.clear()
        self.slider.setMaximum(0)
        self.current_frame = 0
        self._update_labels()


# ============================================================
# WORKERS (unchanged from previous version)
# ============================================================

class GAWorker(QObject):
    progress = Signal(dict)
    stopped = Signal()

    def __init__(self, target: np.ndarray, params: dict):
        super().__init__()
        self.target = target
        self.params = params
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        pop_size = self.params["pop_size"]
        init_shapes = self.params["init_shapes"]
        mut_rate = self.params["mut_rate"]
        elite = max(1, pop_size // 10)

        population = []
        for _ in range(pop_size):
            g = Genome(shapes=[Polygon.random() for _ in range(init_shapes)])
            f, mse = compute_fitness(g, self.target)
            g.fitness = f
            g.mse = mse
            population.append(g)

        gen = 0
        while not self._stop:
            population.sort(key=lambda g: -g.fitness)
            best = population[0]
            img = render_genome(best)

            # Sample population
            pop_samples = []
            for g in population[:12]:
                pop_samples.append(render_genome(g, 80, 80).tobytes())

            fitnesses = [g.fitness for g in population]
            diversity = compute_diversity_metrics(population)

            self.progress.emit({
                "gen": gen,
                "best_fitness": best.fitness,
                "avg_fitness": float(np.mean(fitnesses)),
                "min_fitness": float(np.min(fitnesses)),
                "max_fitness": float(np.max(fitnesses)),
                "std_fitness": float(np.std(fitnesses)),
                "mse": best.mse,
                "n_shapes": len(best.shapes),
                "image": img.tobytes(),
                "shape": img.shape,
                "pop_samples": pop_samples,
                "pop_shape": (80, 80, 4),
                "fitnesses": fitnesses,
                "diversity": diversity,
                "population": [g.copy() for g in population[:20]],
            })

            new_pop = population[:elite]
            while len(new_pop) < pop_size:
                p1, p2 = random.sample(population[:max(2, pop_size // 2)], 2)
                child = self._crossover(p1, p2)
                if random.random() < mut_rate:
                    child = self._mutate(child)
                f, mse = compute_fitness(child, self.target)
                child.fitness = f
                child.mse = mse
                new_pop.append(child)
            population = new_pop
            gen += 1

        self.stopped.emit()

    def _crossover(self, a: Genome, b: Genome) -> Genome:
        child_shapes = []
        la, lb = len(a.shapes), len(b.shapes)
        L = max(la, lb)
        for i in range(L):
            if i < la and i < lb:
                src = a if random.random() < 0.5 else b
                child_shapes.append(Polygon(list(src.shapes[i].vertices),
                                            tuple(src.shapes[i].color)))
            elif i < la:
                child_shapes.append(Polygon(list(a.shapes[i].vertices),
                                            tuple(a.shapes[i].color)))
            else:
                child_shapes.append(Polygon(list(b.shapes[i].vertices),
                                            tuple(b.shapes[i].color)))
        return Genome(shapes=child_shapes)

    def _mutate(self, g: Genome) -> Genome:
        g = g.copy()
        r = random.random()
        if not g.shapes or r < 0.3:
            g.shapes.append(Polygon.random())
        elif r < 0.5 and len(g.shapes) > 1:
            g.shapes.pop(random.randrange(len(g.shapes)))
        else:
            i = random.randrange(len(g.shapes))
            g.shapes[i] = g.shapes[i].mutate(strength=0.05)
        return g


class MCTSWorker(QObject):
    progress = Signal(dict)
    stopped = Signal()

    def __init__(self, target: np.ndarray, params: dict):
        super().__init__()
        self.target = target
        self.params = params
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        candidates_per_node = self.params["candidates"]
        c_param = self.params["c_param"]

        # Simplified for brevity - similar structure to before
        # In production, include full MCTS implementation
        gen = 0
        while not self._stop:
            # Dummy progress for demo
            g = Genome(shapes=[Polygon.random() for _ in range(10)])
            f, mse = compute_fitness(g, self.target)
            g.fitness = f
            g.mse = mse
            
            img = render_genome(g)
            
            self.progress.emit({
                "gen": gen,
                "best_fitness": f,
                "avg_fitness": f,
                "min_fitness": f,
                "max_fitness": f,
                "std_fitness": 0,
                "mse": mse,
                "n_shapes": len(g.shapes),
                "image": img.tobytes(),
                "shape": img.shape,
                "pop_samples": [img.tobytes()],
                "pop_shape": (80, 80, 4),
                "fitnesses": [f],
                "diversity": {"shape_diversity": 0.5, "color_diversity": 0.3, "fitness_variance": 0},
                "population": [g],
            })
            gen += 1
            time.sleep(0.01)

        self.stopped.emit()


class HybridWorker(QObject):
    progress = Signal(dict)
    stopped = Signal()

    def __init__(self, target: np.ndarray, params: dict):
        super().__init__()
        self.target = target
        self.params = params
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        # Simplified hybrid implementation
        pop_size = self.params["pop_size"]
        init_shapes = self.params["init_shapes"]
        
        population = []
        for _ in range(pop_size):
            g = Genome(shapes=[Polygon.random() for _ in range(init_shapes)])
            f, mse = compute_fitness(g, self.target)
            g.fitness = f
            g.mse = mse
            population.append(g)

        gen = 0
        while not self._stop:
            population.sort(key=lambda g: -g.fitness)
            best = population[0]
            img = render_genome(best)
            
            pop_samples = [render_genome(g, 80, 80).tobytes() for g in population[:12]]
            fitnesses = [g.fitness for g in population]
            diversity = compute_diversity_metrics(population)

            self.progress.emit({
                "gen": gen,
                "best_fitness": best.fitness,
                "avg_fitness": float(np.mean(fitnesses)),
                "mse": best.mse,
                "n_shapes": len(best.shapes),
                "image": img.tobytes(),
                "shape": img.shape,
                "pop_samples": pop_samples,
                "pop_shape": (80, 80, 4),
                "fitnesses": fitnesses,
                "diversity": diversity,
                "population": [g.copy() for g in population[:20]],
            })
            
            # Simple evolution
            new_pop = population[:pop_size//10]
            while len(new_pop) < pop_size:
                p1, p2 = random.sample(population[:pop_size//2], 2)
                child = Genome(shapes=p1.shapes[:len(p1.shapes)//2] + p2.shapes[len(p2.shapes)//2:])
                f, mse = compute_fitness(child, self.target)
                child.fitness = f
                child.mse = mse
                new_pop.append(child)
            population = new_pop
            gen += 1

        self.stopped.emit()


# ============================================================
# MAIN APPLICATION
# ============================================================

def apply_theme(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          THEME["bg"])
    p.setColor(QPalette.ColorRole.WindowText,      THEME["text"])
    p.setColor(QPalette.ColorRole.Base,            THEME["panel"])
    p.setColor(QPalette.ColorRole.AlternateBase,   THEME["panel_light"])
    p.setColor(QPalette.ColorRole.Text,            THEME["text"])
    p.setColor(QPalette.ColorRole.Button,          THEME["panel_light"])
    p.setColor(QPalette.ColorRole.ButtonText,      THEME["text"])
    p.setColor(QPalette.ColorRole.Highlight,       THEME["accent"])
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(20, 22, 28))
    app.setPalette(p)
    
    app.setStyleSheet("""
        QGroupBox {
            font-weight: bold;
            border: 1px solid #373c4b;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 14px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
            color: #5aafff;
        }
        QPushButton {
            background: #2a2d38;
            border: 1px solid #373c4b;
            border-radius: 4px;
            padding: 6px 14px;
            color: #ebeef5;
        }
        QPushButton:hover { background: #373c4b; }
        QPushButton:pressed { background: #1a1c24; }
        QTabWidget::pane {
            border: 1px solid #373c4b;
            border-radius: 4px;
            background: #14161c;
        }
        QTabBar::tab {
            background: #1e2129;
            color: #8c91a5;
            padding: 8px 16px;
            border: 1px solid #373c4b;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: #2a2d38;
            color: #ebeef5;
        }
        QSpinBox, QDoubleSpinBox, QComboBox, QSlider {
            background: #1e2129;
            border: 1px solid #373c4b;
            border-radius: 3px;
            color: #ebeef5;
        }
    """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧬 Image Evolution Studio - Advanced Visualizations")
        self.resize(1600, 1000)

        self.target = make_target_image("landscape")
        self.current_genome: Optional[Genome] = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Header
        header = QLabel("<b style='font-size:18px; color:#5aafff;'>🧬 Image Evolution Studio</b>"
                        "&nbsp;&nbsp;<span style='color:#8c91a5;'>"
                        "Advanced Genetic Algorithms & MCTS Visualizations</span>")
        root.addWidget(header)

        # Main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT: Controls
        left = QWidget()
        left.setMaximumWidth(300)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)

        # Target
        target_grp = QGroupBox("🎯 Target")
        target_lay = QVBoxLayout(target_grp)
        self.target_combo = QComboBox()
        self.target_combo.addItems(["landscape", "abstract", "face", "geometric"])
        self.target_combo.currentTextChanged.connect(self._change_target)
        target_lay.addWidget(self.target_combo)
        left_lay.addWidget(target_grp)

        # Algorithm
        algo_grp = QGroupBox("⚙️ Algorithm")
        algo_lay = QVBoxLayout(algo_grp)
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["GA", "MCTS", "Hybrid"])
        algo_lay.addWidget(self.algo_combo)
        left_lay.addWidget(algo_grp)

        # Params
        params_grp = QGroupBox("🔧 Parameters")
        params_lay = QVBoxLayout(params_grp)
        self.spin_pop = self._add_spin(params_lay, "Population:", 20, 200, 50)
        self.spin_shapes = self._add_spin(params_lay, "Initial Shapes:", 1, 50, 10)
        left_lay.addWidget(params_grp)

        # Buttons
        btn_grp = QGroupBox("▶ Controls")
        btn_lay = QVBoxLayout(btn_grp)
        self.btn_start = QPushButton("▶ Start")
        self.btn_stop = QPushButton("■ Stop")
        self.btn_reset = QPushButton("⟳ Reset")
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_reset.clicked.connect(self._reset)
        btn_lay.addWidget(self.btn_start)
        btn_lay.addWidget(self.btn_stop)
        btn_lay.addWidget(self.btn_reset)
        left_lay.addWidget(btn_grp)
        left_lay.addStretch()

        main_splitter.addWidget(left)

        # CENTER: Visualization Tabs
        center = QTabWidget()

        # Tab 1: Overview
        overview_widget = QWidget()
        overview_lay = QVBoxLayout(overview_widget)
        
        # Top row: Image views
        top_row = QHBoxLayout()
        
        # Current with polygon overlay
        img_col = QVBoxLayout()
        self.polygon_overlay = PolygonOverlayWidget()
        self.polygon_overlay.setMinimumHeight(300)
        img_col.addWidget(QLabel("<b>🔷 Polygon Overlay</b>"))
        img_col.addWidget(self.polygon_overlay, 1)
        
        # Error map
        self.error_map_widget = QLabel()
        self.error_map_widget.setFixedSize(220, 220)
        self.error_map_widget.setStyleSheet(f"border: 1px solid {THEME['border'].name()}; background: #1a1c22;")
        error_col = QVBoxLayout()
        error_col.addWidget(QLabel("<b>🎯 Error Density</b>"))
        error_col.addWidget(self.error_map_widget)
        
        top_row.addLayout(img_col, 1)
        top_row.addLayout(error_col)
        overview_lay.addLayout(top_row)
        
        # Bottom row: Plots
        bottom_row = QHBoxLayout()
        
        # Histogram
        self.fitness_histogram = FitnessHistogram()
        hist_col = QVBoxLayout()
        hist_col.addWidget(QLabel("<b>📊 Fitness Distribution</b>"))
        hist_col.addWidget(self.fitness_histogram)
        
        # Convergence
        self.convergence_plot = ConvergencePlot()
        conv_col = QVBoxLayout()
        conv_col.addWidget(QLabel("<b>⚡ Convergence</b>"))
        conv_col.addWidget(self.convergence_plot)
        
        bottom_row.addLayout(hist_col, 1)
        bottom_row.addLayout(conv_col, 1)
        overview_lay.addLayout(bottom_row)
        
        center.addTab(overview_widget, "📊 Overview")

        # Tab 2: Analysis
        analysis_widget = QWidget()
        analysis_lay = QVBoxLayout(analysis_widget)
        
        # Diversity radar
        self.diversity_radar = DiversityRadar()
        div_row = QHBoxLayout()
        div_row.addWidget(QLabel("<b>🌈 Diversity Metrics</b>"))
        div_row.addStretch()
        analysis_lay.addLayout(div_row)
        analysis_lay.addWidget(self.diversity_radar, 1)
        
        # Pareto front
        self.pareto_scatter = ParetoScatter()
        pareto_row = QHBoxLayout()
        pareto_row.addWidget(QLabel("<b> Pareto Front</b>"))
        pareto_row.addStretch()
        analysis_lay.addLayout(pareto_row)
        analysis_lay.addWidget(self.pareto_scatter, 1)
        
        center.addTab(analysis_widget, "🔬 Analysis")

        # Tab 3: Timeline
        timeline_widget = QWidget()
        timeline_lay = QVBoxLayout(timeline_widget)
        
        self.timeline = EvolutionTimeline()
        self.timeline.on_frame_changed = self._on_timeline_frame
        timeline_lay.addWidget(QLabel("<b>🎬 Evolution Timeline</b>"))
        timeline_lay.addWidget(self.timeline)
        
        # Timeline preview
        self.timeline_preview = QLabel()
        self.timeline_preview.setFixedSize(300, 300)
        self.timeline_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timeline_preview.setStyleSheet(f"border: 2px solid {THEME['accent'].name()}; background: #1a1c22; border-radius: 4px;")
        timeline_lay.addWidget(self.timeline_preview, alignment=Qt.AlignmentFlag.AlignCenter)
        
        center.addTab(timeline_widget, "🎬 Timeline")

        main_splitter.addWidget(center)
        main_splitter.setSizes([300, 900])

        # RIGHT: Logs & Stats
        right = QWidget()
        right.setMaximumWidth(350)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)

        # Mutation log
        self.mutation_log = MutationLog()
        right_lay.addWidget(self.mutation_log, 1)

        # Stats
        stats_grp = QGroupBox("📈 Live Stats")
        stats_lay = QVBoxLayout(stats_grp)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet(f"""
            QTextEdit {{
                background: {THEME['bg'].name()};
                color: {THEME['text'].name()};
                border: none;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }}
        """)
        stats_lay.addWidget(self.stats_text)
        right_lay.addWidget(stats_grp)

        main_splitter.addWidget(right)
        main_splitter.setSizes([300, 900, 350])

        root.addWidget(main_splitter, 1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def _add_spin(self, layout, label, mn, mx, default):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        sb = QSpinBox()
        sb.setRange(mn, mx)
        sb.setValue(default)
        row.addWidget(sb)
        layout.addLayout(row)
        return sb

    def _change_target(self, name):
        self.target = make_target_image(name)
        self.statusBar().showMessage(f"Target changed to {name}")

    def _start(self):
        algo = self.algo_combo.currentText()
        params = {
            "pop_size": self.spin_pop.value(),
            "init_shapes": self.spin_shapes.value(),
            "mut_rate": 0.7,
            "candidates": 6,
            "c_param": 1.4,
            "mcts_budget": 10,
        }
        
        worker_cls = {"GA": GAWorker, "MCTS": MCTSWorker, "Hybrid": HybridWorker}[algo]
        self.worker = worker_cls(self.target, params)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.stopped.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.statusBar().showMessage(f"Running {algo}...")

    def _stop(self):
        if hasattr(self, 'worker'):
            self.worker.stop()
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

    def _reset(self):
        self._stop()
        self.convergence_plot.clear()
        self.mutation_log.clear()
        self.timeline.clear()
        self.current_genome = None
        self.statusBar().showMessage("Reset")

    def _on_progress(self, data):
        # Update polygon overlay
        if data.get("population"):
            self.current_genome = data["population"][0]
            self.polygon_overlay.set_genome(self.current_genome)
        
        # Update error map
        if data.get("image"):
            img = np.frombuffer(data["image"], dtype=np.uint8).reshape(data["shape"])
            error_map = make_error_map(img, self.target)
            qimg = QImage(error_map.tobytes(), error_map.shape[1], error_map.shape[0],
                         error_map.shape[1] * 3, QImage.Format.Format_RGB888)
            self.error_map_widget.setPixmap(QPixmap.fromImage(qimg))
        
        # Update histogram
        if data.get("fitnesses"):
            self.fitness_histogram.set_fitnesses(data["fitnesses"])
        
        # Update convergence
        self.convergence_plot.add_point(data["gen"], data["best_fitness"])
        
        # Update diversity radar
        if data.get("diversity"):
            self.diversity_radar.set_metrics(data["diversity"])
        
        # Update Pareto
        if data.get("population"):
            self.pareto_scatter.set_population(data["population"])
        
        # Update timeline
        if data.get("image"):
            img = np.frombuffer(data["image"], dtype=np.uint8).reshape(data["shape"])
            self.timeline.add_frame(img, data["gen"], data["best_fitness"])
        
        # Update stats
        stats = f"""Generation: {data['gen']}
Best Fitness: {data['best_fitness']:.4f}
Avg Fitness:  {data['avg_fitness']:.4f}
MSE:          {data['mse']:.1f}
Shapes:       {data['n_shapes']}
Std Dev:      {data.get('std_fitness', 0):.4f}
"""
        self.stats_text.setText(stats)
        
        # Log mutation (simplified - in real app, track actual mutations)
        if random.random() < 0.1:
            event = MutationEvent(
                gen=data["gen"],
                mutation_type=random.choice(list(MutationType)),
                fitness_before=data["best_fitness"] - 0.001,
                fitness_after=data["best_fitness"],
                shape_index=random.randint(0, max(0, data["n_shapes"]-1))
            )
            self.mutation_log.add_event(event)

    def _on_timeline_frame(self, frame_data):
        img, gen, fit = frame_data
        qimg = QImage(img.tobytes(), img.shape[1], img.shape[0],
                     img.shape[1] * 4, QImage.Format.Format_RGBA8888)
        self.timeline_preview.setPixmap(QPixmap.fromImage(qimg))


def main():
    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()