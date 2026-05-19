"""Landscape drawing primitives for PW-PLAN-CLEANER-01.

All primitives:
  - draw ONLY inside the landscape_editable mask
  - never touch architecture_locked pixels
  - return (canvas_bgr, svg_commands) so geometry can be exported to Rhino/GH
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class DrawResult:
    canvas: np.ndarray          # updated BGR image
    svg_commands: list[str]     # SVG path/shape strings for export
    placed: bool = True         # False if placement was rejected (outside mask)


def _in_mask(canvas: np.ndarray, mask: np.ndarray, arch: np.ndarray) -> np.ndarray:
    """Return a draw permission mask: editable AND not locked."""
    allowed = cv2.bitwise_and(mask, cv2.bitwise_not(arch))
    return allowed


def _check_centre(cx: int, cy: int, r: int, allowed: np.ndarray) -> bool:
    h, w = allowed.shape[:2]
    y0, y1 = max(0, cy - r), min(h, cy + r)
    x0, x1 = max(0, cx - r), min(w, cx + r)
    if y1 <= y0 or x1 <= x0:
        return False
    region = allowed[y0:y1, x0:x1]
    return float(region.mean()) > 30


# ── 1. Feature tree (plan view: circle + radial lines) ───────────────────────

def draw_feature_tree(
    canvas: np.ndarray,
    cx: int, cy: int, radius: int,
    editable: np.ndarray, locked: np.ndarray,
    color: tuple[int, int, int] = (30, 30, 30),
    spokes: int = 12,
) -> DrawResult:
    allowed = _in_mask(canvas, editable, locked)
    if not _check_centre(cx, cy, radius, allowed):
        return DrawResult(canvas, [], placed=False)

    out = canvas.copy()
    svg: list[str] = []

    # outer circle
    cv2.circle(out, (cx, cy), radius, color, 1, cv2.LINE_AA)
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="#1e1e1e" stroke-width="1"/>')

    # inner dot
    cv2.circle(out, (cx, cy), max(2, radius // 8), color, -1)

    # radial spokes
    for i in range(spokes):
        ang = 2 * math.pi * i / spokes
        x1 = int(cx + (radius * 0.5) * math.cos(ang))
        y1 = int(cy + (radius * 0.5) * math.sin(ang))
        x2 = int(cx + radius * math.cos(ang))
        y2 = int(cy + radius * math.sin(ang))
        cv2.line(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#1e1e1e" stroke-width="0.75"/>')

    # enforce immutability
    out[locked > 0] = canvas[locked > 0]
    return DrawResult(out, svg)


# ── 2. Shade tree (larger stippled canopy) ────────────────────────────────────

def draw_shade_tree(
    canvas: np.ndarray,
    cx: int, cy: int, radius: int,
    editable: np.ndarray, locked: np.ndarray,
    color: tuple[int, int, int] = (30, 30, 30),
    dots: int = 80,
) -> DrawResult:
    allowed = _in_mask(canvas, editable, locked)
    if not _check_centre(cx, cy, radius, allowed):
        return DrawResult(canvas, [], placed=False)

    out = canvas.copy()
    svg: list[str] = []

    cv2.circle(out, (cx, cy), radius, color, 1, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), radius // 2, color, 1, cv2.LINE_AA)
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="#1e1e1e" stroke-width="1"/>')

    rng = random.Random(cx * 1000 + cy)
    for _ in range(dots):
        a = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(0, radius - 2)
        px, py = int(cx + d * math.cos(a)), int(cy + d * math.sin(a))
        h, w = out.shape[:2]
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(out, (px, py), 1, color, -1)

    out[locked > 0] = canvas[locked > 0]
    return DrawResult(out, svg)


# ── 3. Shrub mass (blob outline) ─────────────────────────────────────────────

def draw_shrub_mass(
    canvas: np.ndarray,
    cx: int, cy: int,
    rx: int, ry: int,
    editable: np.ndarray, locked: np.ndarray,
    color: tuple[int, int, int] = (30, 30, 30),
    lobes: int = 8,
) -> DrawResult:
    allowed = _in_mask(canvas, editable, locked)
    if not _check_centre(cx, cy, max(rx, ry), allowed):
        return DrawResult(canvas, [], placed=False)

    out = canvas.copy()
    rng = random.Random(cx + ry * 31)
    pts: list[tuple[int, int]] = []
    for i in range(lobes):
        ang = 2 * math.pi * i / lobes
        jitter = rng.uniform(0.75, 1.25)
        px = int(cx + rx * jitter * math.cos(ang))
        py = int(cy + ry * jitter * math.sin(ang))
        pts.append((px, py))

    arr = np.array(pts, dtype=np.int32)
    cv2.polylines(out, [arr], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

    svg_pts = " ".join(f"{p[0]},{p[1]}" for p in pts)
    svg = [f'<polygon points="{svg_pts}" fill="none" stroke="#1e1e1e" stroke-width="1"/>']

    out[locked > 0] = canvas[locked > 0]
    return DrawResult(out, svg)


# ── 4. Grass / groundcover hatch ─────────────────────────────────────────────

def draw_grass_hatch(
    canvas: np.ndarray,
    region_mask: np.ndarray,
    editable: np.ndarray, locked: np.ndarray,
    color: tuple[int, int, int] = (60, 60, 60),
    spacing: int = 8,
    angle_deg: float = 45.0,
) -> DrawResult:
    allowed = cv2.bitwise_and(region_mask, _in_mask(canvas, editable, locked))
    if not allowed.any():
        return DrawResult(canvas, [], placed=False)

    out = canvas.copy()
    svg: list[str] = []
    h, w = canvas.shape[:2]
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    for offset in range(-max(h, w), max(h, w), spacing):
        x1 = int(offset * cos_a - (-max(h, w)) * sin_a)
        y1 = int(offset * sin_a + (-max(h, w)) * cos_a)
        x2 = int(offset * cos_a - max(h, w) * sin_a)
        y2 = int(offset * sin_a + max(h, w) * cos_a)
        line_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)
        draw_px = cv2.bitwise_and(line_mask, allowed)
        out[draw_px > 0] = color

    out[locked > 0] = canvas[locked > 0]
    return DrawResult(out, svg)


# ── 5. Gravel stipple ─────────────────────────────────────────────────────────

def draw_gravel_stipple(
    canvas: np.ndarray,
    region_mask: np.ndarray,
    editable: np.ndarray, locked: np.ndarray,
    color: tuple[int, int, int] = (80, 80, 80),
    density: float = 0.015,
) -> DrawResult:
    allowed = cv2.bitwise_and(region_mask, _in_mask(canvas, editable, locked))
    if not allowed.any():
        return DrawResult(canvas, [], placed=False)

    out = canvas.copy()
    ys, xs = np.where(allowed > 0)
    n_dots = max(1, int(len(ys) * density))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(ys), size=min(n_dots, len(ys)), replace=False)

    for i in idx:
        r = int(rng.integers(1, 3))
        cv2.circle(out, (int(xs[i]), int(ys[i])), r, color, -1)

    out[locked > 0] = canvas[locked > 0]
    return DrawResult(out, [])


# ── 6. Boulder cluster ────────────────────────────────────────────────────────

def draw_boulder_cluster(
    canvas: np.ndarray,
    cx: int, cy: int,
    editable: np.ndarray, locked: np.ndarray,
    color: tuple[int, int, int] = (40, 40, 40),
    n_boulders: int = 5,
    spread: int = 30,
) -> DrawResult:
    allowed = _in_mask(canvas, editable, locked)
    out = canvas.copy()
    svg: list[str] = []
    rng = random.Random(cx * 17 + cy * 31)

    for _ in range(n_boulders):
        bx = cx + rng.randint(-spread, spread)
        by = cy + rng.randint(-spread, spread)
        rx = rng.randint(8, 18)
        ry = rng.randint(6, 14)
        ang = rng.randint(0, 180)
        h, w = out.shape[:2]
        if 0 <= bx < w and 0 <= by < h and allowed[by, bx] > 0:
            cv2.ellipse(out, (bx, by), (rx, ry), ang, 0, 360, color, 1, cv2.LINE_AA)
            svg.append(f'<ellipse cx="{bx}" cy="{by}" rx="{rx}" ry="{ry}" '
                       f'transform="rotate({ang},{bx},{by})" fill="none" stroke="#282828" stroke-width="1"/>')

    out[locked > 0] = canvas[locked > 0]
    return DrawResult(out, svg)


# ── 7. Stepping stone path ────────────────────────────────────────────────────

def draw_stepping_stones(
    canvas: np.ndarray,
    polyline: list[tuple[int, int]],
    editable: np.ndarray, locked: np.ndarray,
    color: tuple[int, int, int] = (40, 40, 40),
    step: int = 24,
    stone_rx: int = 10,
    stone_ry: int = 7,
) -> DrawResult:
    if len(polyline) < 2:
        return DrawResult(canvas, [], placed=False)

    allowed = _in_mask(canvas, editable, locked)
    out = canvas.copy()
    svg: list[str] = []
    rng = random.Random(len(polyline))

    # Interpolate points along polyline at *step* intervals
    def _interp(pts: list[tuple[int, int]], dist: int) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = [pts[0]]
        accumulated = 0.0
        for i in range(1, len(pts)):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            while accumulated + seg_len >= dist:
                t = (dist - accumulated) / seg_len
                sx = int(x0 + t * (x1 - x0))
                sy = int(y0 + t * (y1 - y0))
                result.append((sx, sy))
                x0, y0 = sx, sy
                seg_len -= (dist - accumulated)
                accumulated = 0.0
            accumulated += seg_len
        return result

    stones = _interp(polyline, step)
    for sx, sy in stones:
        angle = rng.randint(0, 180)
        h, w = out.shape[:2]
        if 0 <= sx < w and 0 <= sy < h and allowed[sy, sx] > 0:
            cv2.ellipse(out, (sx, sy), (stone_rx, stone_ry), angle, 0, 360, color, 1, cv2.LINE_AA)
            svg.append(f'<ellipse cx="{sx}" cy="{sy}" rx="{stone_rx}" ry="{stone_ry}" '
                       f'transform="rotate({angle},{sx},{sy})" fill="none" stroke="#282828" stroke-width="1"/>')

    out[locked > 0] = canvas[locked > 0]
    return DrawResult(out, svg)


# ── 8. Pond-edge planting ─────────────────────────────────────────────────────

def draw_pond_edge(
    canvas: np.ndarray,
    cx: int, cy: int, radius: int,
    editable: np.ndarray, locked: np.ndarray,
    color: tuple[int, int, int] = (40, 40, 40),
    tufts: int = 24,
) -> DrawResult:
    allowed = _in_mask(canvas, editable, locked)
    if not _check_centre(cx, cy, radius, allowed):
        return DrawResult(canvas, [], placed=False)

    out = canvas.copy()
    svg: list[str] = []
    rng = random.Random(cx + cy)

    for i in range(tufts):
        ang = 2 * math.pi * i / tufts + rng.uniform(-0.1, 0.1)
        r1 = radius + rng.randint(-4, 4)
        r2 = r1 + rng.randint(6, 14)
        x1 = int(cx + r1 * math.cos(ang))
        y1 = int(cy + r1 * math.sin(ang))
        x2 = int(cx + r2 * math.cos(ang))
        y2 = int(cy + r2 * math.sin(ang))
        h, w = out.shape[:2]
        if 0 <= x1 < w and 0 <= y1 < h and allowed[y1, x1] > 0:
            cv2.line(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    cv2.circle(out, (cx, cy), radius, color, 1, cv2.LINE_AA)
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="#282828" stroke-width="1"/>')

    out[locked > 0] = canvas[locked > 0]
    return DrawResult(out, svg)


# ── 9. Curved bed edge ────────────────────────────────────────────────────────

def draw_bed_edge(
    canvas: np.ndarray,
    control_points: list[tuple[int, int]],
    editable: np.ndarray, locked: np.ndarray,
    color: tuple[int, int, int] = (30, 30, 30),
    thickness: int = 1,
) -> DrawResult:
    if len(control_points) < 2:
        return DrawResult(canvas, [], placed=False)

    allowed = _in_mask(canvas, editable, locked)
    out = canvas.copy()

    # Draw as polyline through control points (approximates bezier)
    pts = np.array(control_points, dtype=np.int32)
    cv2.polylines(out, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    svg_d = "M " + " L ".join(f"{p[0]},{p[1]}" for p in control_points)
    svg = [f'<path d="{svg_d}" fill="none" stroke="#1e1e1e" stroke-width="{thickness}"/>']

    out[locked > 0] = canvas[locked > 0]
    return DrawResult(out, svg)
