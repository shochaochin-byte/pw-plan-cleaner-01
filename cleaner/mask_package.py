"""Canonical 5-mask taxonomy for PW-PLAN-CLEANER-01.

Architecture immutability contract
-----------------------------------
architecture_locked is the authoritative hard boundary.
Every downstream stage MUST call ``apply_immutability(canvas, pkg)``
before writing any output, ensuring no pixel inside the locked mask
is ever modified by AI, manual tools, or export pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class MaskPackage:
    """Five canonical masks covering the full semantic layer set."""

    architecture_locked: np.ndarray   # walls, dims, text, doors — IMMUTABLE
    landscape_editable: np.ndarray    # AI / tools may draw only here
    hatch_removed: np.ndarray         # cleaned hatch regions
    intervention_red: np.ndarray      # AI edits / changed areas  (RED colour semantics)
    analysis_blue: np.ndarray         # preserved / intelligence layer (BLUE colour semantics)

    # Derived tier masks (set by zone_detector)
    tier_foreground: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))
    tier_midground:  np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))
    tier_background: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))

    @property
    def shape(self) -> tuple[int, int]:
        return self.architecture_locked.shape[:2]

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "architecture_locked": self.architecture_locked,
            "landscape_editable":  self.landscape_editable,
            "hatch_removed":       self.hatch_removed,
            "intervention_red":    self.intervention_red,
            "analysis_blue":       self.analysis_blue,
        }


def build_mask_package(
    image_bgr: np.ndarray,
    hatch_mask: np.ndarray | None = None,
) -> MaskPackage:
    """Build canonical masks from *image_bgr*.

    Parameters
    ----------
    image_bgr:   source plan image (BGR uint8)
    hatch_mask:  pre-computed hatch mask from cleaner pipeline (uint8 0/255)
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # ── architecture_locked ──────────────────────────────────────────────────
    # Strong edges (walls, lines, text, dimensions)
    arch = cv2.Canny(gray, 60, 180)
    arch = cv2.dilate(arch, np.ones((3, 3), np.uint8), iterations=2)
    # Very dark ink (vector drawings often print at near-black)
    dark = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)[1]
    architecture_locked = cv2.bitwise_or(arch, dark)

    # ── hatch_removed ────────────────────────────────────────────────────────
    if hatch_mask is None:
        hatch_mask = np.zeros((h, w), dtype=np.uint8)
    hatch_removed = hatch_mask.copy()

    # ── landscape_editable ───────────────────────────────────────────────────
    # Bright open regions NOT in architecture
    bright = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)[1]
    inv_arch = cv2.bitwise_not(architecture_locked)
    landscape = cv2.bitwise_and(bright, inv_arch)
    landscape = cv2.morphologyEx(landscape, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    landscape_editable = landscape

    # ── analysis_blue (BLUE: preserved / intelligence) ───────────────────────
    # Architecture after hatch removal — "what's been confirmed preserved"
    analysis_blue = cv2.bitwise_and(architecture_locked, cv2.bitwise_not(hatch_removed))

    # ── intervention_red (RED: changed / affected / AI) ─────────────────────
    # Starts as hatch region; grows as AI edits are applied
    intervention_red = hatch_removed.copy()

    return MaskPackage(
        architecture_locked=architecture_locked,
        landscape_editable=landscape_editable,
        hatch_removed=hatch_removed,
        intervention_red=intervention_red,
        analysis_blue=analysis_blue,
    )


def apply_immutability(
    canvas_bgr: np.ndarray,
    pkg: MaskPackage,
    paper_color: tuple[int, int, int] = (232, 240, 245),
) -> np.ndarray:
    """Zero-out any pixels that fall inside ``pkg.architecture_locked``.

    Call this as the final step after any AI or tool overlay to enforce
    the architecture immutability contract.
    """
    out = canvas_bgr.copy()
    # Restore locked pixels to the original (architecture colour)
    out[pkg.architecture_locked > 0] = (0, 0, 0)
    return out


def composite_masks_preview(
    base_bgr: np.ndarray,
    pkg: MaskPackage,
    alpha: float = 0.45,
    visible: dict[str, bool] | None = None,
) -> np.ndarray:
    """Overlay masks onto *base_bgr* for preview purposes.

    Colour semantics:
      architecture_locked → black (not tinted — always opaque)
      landscape_editable  → blue (analysis)
      hatch_removed       → red (intervention)
      intervention_red    → red
      analysis_blue       → blue
    """
    if visible is None:
        visible = {k: True for k in pkg.as_dict()}

    out = base_bgr.copy().astype(np.float32)

    def _tint(mask: np.ndarray, bgr: tuple[int, int, int]) -> None:
        layer = np.zeros_like(out)
        layer[mask > 0] = bgr
        m3 = (mask[:, :, None] > 0).astype(np.float32)
        np.copyto(out, out * (1 - alpha * m3) + layer * (alpha * m3))

    if visible.get("landscape_editable", True):
        _tint(pkg.landscape_editable, (250, 66, 53))   # blue
    if visible.get("hatch_removed", True):
        _tint(pkg.hatch_removed, (42, 56, 232))         # red
    if visible.get("intervention_red", True):
        _tint(pkg.intervention_red, (42, 56, 232))      # red
    if visible.get("analysis_blue", True):
        _tint(pkg.analysis_blue, (250, 66, 53))         # blue

    return np.clip(out, 0, 255).astype(np.uint8)
