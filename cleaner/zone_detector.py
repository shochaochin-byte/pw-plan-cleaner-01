"""Hybrid landscape zone detection — three input modes.

All modes produce / update the ``landscape_editable`` mask in a MaskPackage.
Modes are additive: auto-detect first, then refine with manual lasso or flood fill.
"""
from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np

from cleaner.mask_package import MaskPackage


# ── Mode 1: AI auto-detect ────────────────────────────────────────────────────

def auto_detect_zones(
    image_bgr: np.ndarray,
    architecture_locked: np.ndarray,
    min_area: int = 2000,
    brightness_thresh: int = 210,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (filled_mask, polygon_list) for detected outdoor/landscape zones.

    Detects bright open regions not occupied by architecture — a proxy for
    outdoor / landscape areas.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bright = cv2.threshold(gray, brightness_thresh, 255, cv2.THRESH_BINARY)[1]
    inv_arch = cv2.bitwise_not(architecture_locked)
    candidate = cv2.bitwise_and(bright, inv_arch)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(candidate)
    polygons: list[np.ndarray] = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(mask, [c], -1, 255, -1)
            epsilon = 0.005 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            polygons.append(approx.reshape(-1, 2))

    # Never bleed into locked architecture
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(architecture_locked))
    return mask, polygons


def assign_tiers(
    landscape_mask: np.ndarray,
    architecture_locked: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split landscape_mask into foreground / midground / background tiers.

    Tiers are computed via distance transform from the architecture boundary:
      - background  = closest to architecture (inner)
      - midground   = transition zone
      - foreground  = furthest from architecture (outer)
    """
    if not landscape_mask.any():
        empty = np.zeros_like(landscape_mask)
        return empty, empty, empty

    dist = cv2.distanceTransform(
        cv2.bitwise_not(architecture_locked), cv2.DIST_L2, 5
    )
    # Normalise within landscape region only
    d_in_zone = dist.copy()
    d_in_zone[landscape_mask == 0] = 0
    max_d = d_in_zone.max()
    if max_d == 0:
        empty = np.zeros_like(landscape_mask)
        return empty, empty, empty

    norm = d_in_zone / max_d

    background  = ((norm > 0) & (norm <= 0.33)).astype(np.uint8) * 255
    midground   = ((norm > 0.33) & (norm <= 0.66)).astype(np.uint8) * 255
    foreground  = (norm > 0.66).astype(np.uint8) * 255

    background  = cv2.bitwise_and(background,  landscape_mask)
    midground   = cv2.bitwise_and(midground,   landscape_mask)
    foreground  = cv2.bitwise_and(foreground,  landscape_mask)
    return foreground, midground, background


# ── Mode 2: Manual polygon from streamlit-drawable-canvas JSON ────────────────

def canvas_json_to_mask(
    canvas_json: dict[str, Any],
    shape: tuple[int, int],
    scale: float = 1.0,
) -> np.ndarray:
    """Convert ``st_canvas`` JSON result to a uint8 mask.

    Parameters
    ----------
    canvas_json: result.json_data from st_canvas
    shape:       (height, width) of the target mask
    scale:       scale factor if canvas display size ≠ mask size
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    objects = (canvas_json or {}).get("objects", [])
    for obj in objects:
        obj_type = obj.get("type", "")
        if obj_type == "path":
            pts = _parse_svg_path(obj.get("path", []), scale)
            if pts is not None and len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 255)
        elif obj_type == "rect":
            l = int(obj.get("left", 0) * scale)
            t = int(obj.get("top",  0) * scale)
            rw = int(obj.get("width",  0) * scale)
            rh = int(obj.get("height", 0) * scale)
            cv2.rectangle(mask, (l, t), (l + rw, t + rh), 255, -1)
        elif obj_type == "circle":
            cx = int((obj.get("left", 0) + obj.get("radius", 0)) * scale)
            cy = int((obj.get("top",  0) + obj.get("radius", 0)) * scale)
            r  = int(obj.get("radius", 0) * scale)
            cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def _parse_svg_path(path_cmds: list, scale: float) -> np.ndarray | None:
    """Extract polygon points from fabric.js path command list."""
    pts: list[list[int]] = []
    for cmd in path_cmds:
        if not cmd:
            continue
        letter = cmd[0]
        if letter in ("M", "L") and len(cmd) >= 3:
            pts.append([int(cmd[1] * scale), int(cmd[2] * scale)])
        elif letter == "Q" and len(cmd) >= 5:
            pts.append([int(cmd[3] * scale), int(cmd[4] * scale)])
    if len(pts) < 3:
        return None
    return np.array(pts, dtype=np.int32)


# ── Mode 3: Flood fill from click point ───────────────────────────────────────

def flood_fill_zone(
    image_bgr: np.ndarray,
    seed_xy: tuple[int, int],
    architecture_locked: np.ndarray,
    tolerance: int = 15,
) -> np.ndarray:
    """Flood-fill from *seed_xy* and return the filled uint8 mask.

    Fill is constrained: pixels inside ``architecture_locked`` act as barriers.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Blocked pixels become black barriers
    barrier = gray.copy()
    barrier[architecture_locked > 0] = 0

    # floodFill needs a (h+2)×(w+2) mask
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    seed = (int(seed_xy[0]), int(seed_xy[1]))
    seed = (min(max(seed[0], 0), w - 1), min(max(seed[1], 0), h - 1))

    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
    cv2.floodFill(
        barrier, ff_mask, seed, 255,
        loDiff=(tolerance,), upDiff=(tolerance,), flags=flags,
    )
    filled = ff_mask[1:-1, 1:-1]
    filled = cv2.bitwise_and(filled, cv2.bitwise_not(architecture_locked))
    return filled


# ── Merge helper ──────────────────────────────────────────────────────────────

def merge_zone_masks(*masks: np.ndarray) -> np.ndarray:
    """Combine multiple zone masks (additive OR)."""
    if not masks:
        raise ValueError("At least one mask required")
    out = masks[0].copy()
    for m in masks[1:]:
        if m.shape == out.shape and m.size:
            out = cv2.bitwise_or(out, m)
    return out
