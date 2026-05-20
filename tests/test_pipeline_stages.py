"""Tests for individual processing stages of the PW-PLAN-CLEANER-01 pipeline."""
from __future__ import annotations

import numpy as np
import pytest
import cv2


# ── helpers ───────────────────────────────────────────────────────────────────

def _gray_gradient(h: int = 100, w: int = 100) -> np.ndarray:
    """Uint8 grayscale gradient image (0 → 255 left to right)."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(row, (h, 1))


def _bgr_gradient(h: int = 100, w: int = 100) -> np.ndarray:
    gray = _gray_gradient(h, w)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _white_with_diagonals(h: int = 120, w: int = 120) -> np.ndarray:
    """White BGR image with black diagonal lines drawn on it."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    for i in range(0, w, 15):
        cv2.line(img, (i, 0), (i + h, h), (0, 0, 0), 1)
    return img


def _binary_mask(h: int = 100, w: int = 100, fill_ratio: float = 0.3) -> np.ndarray:
    """Uint8 mask with a rectangular filled region (value 255)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    r = int(h * fill_ratio)
    c = int(w * fill_ratio)
    mask[r: h - r, c: w - c] = 255
    return mask


# ── Stage 1: halftone_duotone ─────────────────────────────────────────────────

class TestHalftoneDuotone:
    def setup_method(self):
        from cleaner.halftone_duotone import halftone_duotone, duotone_flat
        self.halftone_duotone = halftone_duotone
        self.duotone_flat = duotone_flat
        self.img = _bgr_gradient()

    def test_halftone_duotone_output_shape(self):
        result = self.halftone_duotone(self.img)
        assert result.shape == self.img.shape, "Output shape must match input shape"

    def test_halftone_duotone_dtype_uint8(self):
        result = self.halftone_duotone(self.img)
        assert result.dtype == np.uint8, "Output dtype must be uint8"

    def test_halftone_duotone_not_all_paper_color(self):
        result = self.halftone_duotone(self.img)
        # paper color is #F5F0E8 → BGR (232, 240, 245)
        paper_bgr = np.array([232, 240, 245], dtype=np.uint8)
        all_paper = np.all(result == paper_bgr, axis=2).all()
        assert not all_paper, "Output must not be entirely the paper color"

    def test_halftone_duotone_at_least_2_distinct_colors(self):
        result = self.halftone_duotone(self.img)
        # Count unique BGR triplets
        flat = result.reshape(-1, 3)
        unique_colors = np.unique(flat, axis=0)
        assert len(unique_colors) >= 2, f"Expected >= 2 distinct colors, got {len(unique_colors)}"

    def test_halftone_duotone_with_hatch_mask(self):
        mask = _binary_mask()
        result = self.halftone_duotone(self.img, hatch_mask=mask)
        assert result.shape == self.img.shape
        assert result.dtype == np.uint8

    def test_duotone_flat_output_shape(self):
        result = self.duotone_flat(self.img)
        assert result.shape == self.img.shape

    def test_duotone_flat_dtype_uint8(self):
        result = self.duotone_flat(self.img)
        assert result.dtype == np.uint8

    def test_duotone_flat_not_all_paper_color(self):
        # Use a dark image so ink pixels are produced
        dark_img = _bgr_gradient()
        dark_img[:, :50] = 0  # force left half dark
        result = self.duotone_flat(dark_img)
        paper_bgr = np.array([232, 240, 245], dtype=np.uint8)
        all_paper = np.all(result == paper_bgr, axis=2).all()
        assert not all_paper, "Flat duotone must not be entirely the paper color"

    def test_duotone_flat_at_least_2_distinct_colors(self):
        dark_img = _bgr_gradient()
        dark_img[:, :50] = 0
        result = self.duotone_flat(dark_img)
        flat = result.reshape(-1, 3)
        unique_colors = np.unique(flat, axis=0)
        assert len(unique_colors) >= 2

    def test_duotone_flat_with_hatch_mask(self):
        mask = _binary_mask()
        dark_img = np.zeros_like(self.img)  # all dark → ink everywhere
        result = self.duotone_flat(dark_img, hatch_mask=mask)
        assert result.shape == self.img.shape
        assert result.dtype == np.uint8


# ── Stage 2: raster_cleaner ───────────────────────────────────────────────────

class TestRasterCleaner:
    def setup_method(self):
        from cleaner.raster_cleaner import clean_raster_image
        self.clean_raster_image = clean_raster_image
        self.img = _white_with_diagonals()

    def test_returns_two_outputs(self):
        result = self.clean_raster_image(self.img)
        assert len(result) == 2, "clean_raster_image must return (cleaned, overlay)"

    def test_cleaned_shape_matches_input(self):
        cleaned, _ = self.clean_raster_image(self.img)
        assert cleaned.shape == self.img.shape

    def test_overlay_shape_matches_input(self):
        _, overlay = self.clean_raster_image(self.img)
        assert overlay.shape == self.img.shape

    def test_overlay_is_bgr(self):
        _, overlay = self.clean_raster_image(self.img)
        assert overlay.ndim == 3, "Overlay must be a 3-channel image"
        assert overlay.shape[2] == 3, "Overlay must have exactly 3 channels (BGR)"

    def test_cleaned_dtype_uint8(self):
        cleaned, _ = self.clean_raster_image(self.img)
        assert cleaned.dtype == np.uint8

    def test_overlay_dtype_uint8(self):
        _, overlay = self.clean_raster_image(self.img)
        assert overlay.dtype == np.uint8

    def test_sensitivity_range(self):
        for sens in [0.0, 0.5, 1.0]:
            cleaned, overlay = self.clean_raster_image(self.img, sensitivity=sens)
            assert cleaned.shape == self.img.shape
            assert overlay.shape == self.img.shape


# ── Stage 3: masking ──────────────────────────────────────────────────────────

class TestMasking:
    def setup_method(self):
        from cleaner.masking import colorize_overlay, mask_to_transparent_png
        self.colorize_overlay = colorize_overlay
        self.mask_to_transparent_png = mask_to_transparent_png
        self.base = _bgr_gradient()
        self.mask = _binary_mask()

    def test_colorize_overlay_shape(self):
        zeros = np.zeros((100, 100), dtype=np.uint8)
        result = self.colorize_overlay(self.base, self.mask, zeros, zeros)
        assert result.shape == self.base.shape

    def test_colorize_overlay_dtype(self):
        zeros = np.zeros((100, 100), dtype=np.uint8)
        result = self.colorize_overlay(self.base, self.mask, zeros, zeros)
        assert result.dtype == np.uint8

    def test_colorize_overlay_red_pixels_applied(self):
        zeros = np.zeros((100, 100), dtype=np.uint8)
        result = self.colorize_overlay(self.base, self.mask, zeros, zeros)
        # Where mask is 255, pixel should be [0, 0, 255] (red in BGR)
        mask_coords = np.where(self.mask > 0)
        if len(mask_coords[0]):
            r, c = mask_coords[0][0], mask_coords[1][0]
            assert list(result[r, c]) == [0, 0, 255], "Red mask pixels should be [0, 0, 255] BGR"

    def test_colorize_overlay_blue_pixels_applied(self):
        zeros = np.zeros((100, 100), dtype=np.uint8)
        result = self.colorize_overlay(self.base, zeros, self.mask, zeros)
        mask_coords = np.where(self.mask > 0)
        if len(mask_coords[0]):
            r, c = mask_coords[0][0], mask_coords[1][0]
            assert list(result[r, c]) == [255, 0, 0], "Blue mask pixels should be [255, 0, 0] BGR"

    def test_colorize_overlay_green_pixels_applied(self):
        zeros = np.zeros((100, 100), dtype=np.uint8)
        result = self.colorize_overlay(self.base, zeros, zeros, self.mask)
        mask_coords = np.where(self.mask > 0)
        if len(mask_coords[0]):
            r, c = mask_coords[0][0], mask_coords[1][0]
            assert list(result[r, c]) == [0, 180, 0], "Green mask pixels should be [0, 180, 0] BGR"

    def test_mask_to_transparent_png_shape(self):
        result = self.mask_to_transparent_png(self.base, self.mask)
        assert result.shape == (self.base.shape[0], self.base.shape[1], 4), \
            "transparent PNG must have 4 channels (BGRA)"

    def test_mask_to_transparent_png_dtype(self):
        result = self.mask_to_transparent_png(self.base, self.mask)
        assert result.dtype == np.uint8

    def test_mask_to_transparent_png_alpha_channel(self):
        result = self.mask_to_transparent_png(self.base, self.mask)
        # Alpha channel should match the input mask
        np.testing.assert_array_equal(result[:, :, 3], self.mask)

    def test_mask_to_transparent_png_zero_alpha_outside_mask(self):
        result = self.mask_to_transparent_png(self.base, self.mask)
        outside = self.mask == 0
        assert np.all(result[:, :, 3][outside] == 0), \
            "Alpha outside mask must be 0 (fully transparent)"


# ── Stage 4: segmentation ─────────────────────────────────────────────────────

class TestSegmentation:
    def setup_method(self):
        from cleaner.segmentation import build_segmentation_masks, SegmentationMasks
        self.build_segmentation_masks = build_segmentation_masks
        self.SegmentationMasks = SegmentationMasks
        self.img = _bgr_gradient(120, 120)

    def test_returns_segmentation_masks_dataclass(self):
        result = self.build_segmentation_masks(self.img)
        assert isinstance(result, self.SegmentationMasks)

    def test_architecture_mask_shape(self):
        result = self.build_segmentation_masks(self.img)
        assert result.architecture_mask.shape == self.img.shape[:2]

    def test_landscape_mask_shape(self):
        result = self.build_segmentation_masks(self.img)
        assert result.landscape_mask.shape == self.img.shape[:2]

    def test_hatch_mask_shape(self):
        result = self.build_segmentation_masks(self.img)
        assert result.hatch_mask.shape == self.img.shape[:2]

    def test_circulation_mask_shape(self):
        result = self.build_segmentation_masks(self.img)
        assert result.circulation_mask.shape == self.img.shape[:2]

    def test_all_masks_dtype_uint8(self):
        result = self.build_segmentation_masks(self.img)
        for name, mask in [
            ("architecture_mask", result.architecture_mask),
            ("landscape_mask", result.landscape_mask),
            ("hatch_mask", result.hatch_mask),
            ("circulation_mask", result.circulation_mask),
        ]:
            assert mask.dtype == np.uint8, f"{name} must be uint8"

    def test_hatch_mask_defaults_to_zeros_when_none(self):
        result = self.build_segmentation_masks(self.img, hatch_mask=None)
        assert np.all(result.hatch_mask == 0), \
            "hatch_mask should default to all-zeros when None is passed"

    def test_explicit_hatch_mask_preserved(self):
        hatch = _binary_mask(120, 120)
        result = self.build_segmentation_masks(self.img, hatch_mask=hatch)
        np.testing.assert_array_equal(result.hatch_mask, hatch)

    def test_masks_are_binary(self):
        result = self.build_segmentation_masks(self.img)
        for name, mask in [
            ("architecture_mask", result.architecture_mask),
            ("landscape_mask", result.landscape_mask),
            ("circulation_mask", result.circulation_mask),
        ]:
            unique_vals = np.unique(mask)
            assert set(unique_vals).issubset({0, 255}), \
                f"{name} must be binary (0 or 255), got values: {unique_vals}"


# ── Vector debug bbox clamping/validation ─────────────────────────────────────

class TestVectorDebugMasks:
    def setup_method(self):
        from cleaner.vector_debug import build_vector_debug_masks
        self.build_vector_debug_masks = build_vector_debug_masks

    def test_clamps_out_of_bounds_and_negative_bboxes(self):
        decisions = [
            {"bbox": [-10, -5, 20, 10], "remove": True},
            {"bbox": [90, 90, 120, 140], "remove": False},
        ]
        red, blue, red_boxes, skipped = self.build_vector_debug_masks(decisions, (100, 100, 3), scale=1.0)

        assert skipped == 0
        assert red_boxes == [(0, 0, 20, 10)]
        assert np.all(red[0:10, 0:20] == 255)
        assert np.all(blue[90:100, 90:100] == 255)

    def test_rejects_degenerate_bboxes_and_increments_metric(self):
        decisions = [
            {"bbox": [10, 10, 10, 30], "remove": True},
            {"bbox": [5, 8, 12, 8], "remove": False},
            {"bbox": [20, 20, 25, 25], "remove": True},
        ]
        red, blue, red_boxes, skipped = self.build_vector_debug_masks(decisions, (100, 100), scale=1.0)

        assert skipped == 2
        assert red_boxes == [(20, 20, 25, 25)]
        assert int(np.count_nonzero(red)) == 25
        assert int(np.count_nonzero(blue)) == 0
