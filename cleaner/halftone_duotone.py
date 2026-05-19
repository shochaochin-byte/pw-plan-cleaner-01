"""Halftone duotone rendering for architectural plan output.

Produces a two-colour halftone suitable for print/risograph output:
  - ink_dark  (default: deep navy) for structure / preserved elements
  - ink_light (default: coral red) for detected hatch / landscape zones

The halftone is built by converting luminance to dot-size AM screening
(amplitude-modulated), rotated to the classic 45° print angle.
"""
from __future__ import annotations

import cv2
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return b, g, r


def _make_halftone_channel(
    gray: np.ndarray,
    cell: int = 12,
    angle_deg: float = 45.0,
    invert: bool = True,
) -> np.ndarray:
    """Return a uint8 mask where halftone dots are 255, background 0."""
    h, w = gray.shape
    # rotate source so we can tile on a grid then rotate back
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # build output canvas
    canvas = np.zeros((h, w), dtype=np.uint8)

    half = cell // 2
    for cy in range(-cell, h + cell, cell):
        for cx in range(-cell, w + cell, cell):
            # sample source luminance at cell centre
            sy = int(round(cy))
            sx = int(round(cx))
            sy_c = max(0, min(h - 1, sy))
            sx_c = max(0, min(w - 1, sx))
            lum = int(gray[sy_c, sx_c])
            if invert:
                lum = 255 - lum

            # dot radius proportional to luminance
            radius = int(round(half * (lum / 255.0) ** 0.75))
            if radius < 1:
                continue

            # rotate cell centre
            rx = int(round(cx * cos_a - cy * sin_a))
            ry = int(round(cx * sin_a + cy * cos_a))

            cv2.circle(canvas, (rx % w, ry % h), radius, 255, -1)

    return canvas


def _fast_halftone(
    gray: np.ndarray,
    cell: int = 12,
    angle_deg: float = 45.0,
    invert: bool = True,
) -> np.ndarray:
    """Vectorised halftone — faster than the loop version for large images."""
    h, w = gray.shape
    cell = max(4, cell)

    # work in a padded grid
    pad = cell
    ph, pw = h + 2 * pad, w + 2 * pad
    canvas = np.zeros((ph, pw), dtype=np.uint8)

    ys = np.arange(pad, ph, cell)
    xs = np.arange(pad, pw, cell)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx.ravel(); gy = gy.ravel()

    # sample from source
    sy = np.clip(gy - pad, 0, h - 1)
    sx = np.clip(gx - pad, 0, w - 1)
    lum = gray[sy, sx].astype(np.float32)
    if invert:
        lum = 255.0 - lum

    radii = (cell // 2 * (lum / 255.0) ** 0.75).astype(np.int32)

    # rotate centres
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rx = (gx * cos_a - gy * sin_a).astype(np.int32) % pw
    ry = (gx * sin_a + gy * cos_a).astype(np.int32) % ph

    for i in range(len(rx)):
        r = int(radii[i])
        if r >= 1:
            cv2.circle(canvas, (int(rx[i]), int(ry[i])), r, 255, -1)

    return canvas[pad: pad + h, pad: pad + w]


# ── public API ────────────────────────────────────────────────────────────────

def halftone_duotone(
    image_bgr: np.ndarray,
    hatch_mask: np.ndarray | None = None,
    color_dark: str = "#1B1F5E",   # navy — architecture / ink
    color_light: str = "#E8382A",  # coral — hatch / landscape
    cell: int = 10,
    paper_color: str = "#F5F0E8",
) -> np.ndarray:
    """Render *image_bgr* as a two-colour halftone duotone.

    Parameters
    ----------
    image_bgr:   source image (BGR uint8)
    hatch_mask:  uint8 mask of detected hatch regions (255 = hatch).
                 If None, the full image is rendered in the dark ink only.
    color_dark:  hex colour for structural ink (dark layer)
    color_light: hex colour for hatch / accent ink (light layer)
    cell:        halftone cell size in pixels (8–20 looks good for print)
    paper_color: background paper colour hex

    Returns
    -------
    BGR uint8 image suitable for display or saving as PNG/PDF.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # ── paper base ──
    bg_b, bg_g, bg_r = _hex_to_bgr(paper_color)
    out = np.full((h, w, 3), [bg_b, bg_g, bg_r], dtype=np.uint8)

    # ── dark layer: architectural structure (45°) ──
    dark_dots = _fast_halftone(gray, cell=cell, angle_deg=45.0, invert=True)
    if hatch_mask is not None:
        # suppress dark dots where hatch will live
        dark_dots = cv2.bitwise_and(dark_dots, cv2.bitwise_not(hatch_mask))
    db, dg, dr = _hex_to_bgr(color_dark)
    out[dark_dots > 0] = [db, dg, dr]

    # ── light layer: hatch / accent (75°) ──
    if hatch_mask is not None and hatch_mask.any():
        # build a luminance channel from the hatch region only
        hatch_gray = gray.copy()
        hatch_gray[hatch_mask == 0] = 0          # zero outside hatch
        light_dots = _fast_halftone(hatch_gray, cell=cell - 2, angle_deg=75.0, invert=True)
        light_dots = cv2.bitwise_and(light_dots, hatch_mask)
        lb, lg, lr = _hex_to_bgr(color_light)
        out[light_dots > 0] = [lb, lg, lr]

    return out


def duotone_flat(
    image_bgr: np.ndarray,
    hatch_mask: np.ndarray | None = None,
    color_dark: str = "#1B1F5E",
    color_light: str = "#E8382A",
    paper_color: str = "#F5F0E8",
    threshold: int = 128,
) -> np.ndarray:
    """Flat (no halftone) duotone — bitmap-style, faster for preview."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    bg_b, bg_g, bg_r = _hex_to_bgr(paper_color)
    out = np.full((h, w, 3), [bg_b, bg_g, bg_r], dtype=np.uint8)

    ink_mask = (gray < threshold).astype(np.uint8) * 255
    if hatch_mask is not None:
        struct_mask = cv2.bitwise_and(ink_mask, cv2.bitwise_not(hatch_mask))
        hatch_ink = cv2.bitwise_and(ink_mask, hatch_mask)
    else:
        struct_mask = ink_mask
        hatch_ink = np.zeros_like(ink_mask)

    db, dg, dr = _hex_to_bgr(color_dark)
    lb, lg, lr = _hex_to_bgr(color_light)
    out[struct_mask > 0] = [db, dg, dr]
    out[hatch_ink > 0] = [lb, lg, lr]
    return out
