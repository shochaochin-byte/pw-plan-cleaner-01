"""Tests for the AI landscape drawing engine — all 6 new modules."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def plan_img():
    """Synthetic 300×400 plan: white background, black wall lines."""
    img = np.full((300, 400, 3), 240, dtype=np.uint8)
    # Draw walls
    cv2.rectangle(img, (50, 50), (350, 250), (0, 0, 0), 3)
    cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 0), 2)
    return img


@pytest.fixture
def hatch_mask(plan_img):
    h, w = plan_img.shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    m[60:90, 60:150] = 255
    return m


@pytest.fixture
def pkg(plan_img, hatch_mask):
    from cleaner.mask_package import build_mask_package
    return build_mask_package(plan_img, hatch_mask)


# ── layer_reader ──────────────────────────────────────────────────────────────

class TestLayerReader:
    def test_no_layers_pdf_returns_empty(self, tmp_path):
        import fitz
        from cleaner.layer_reader import read_pdf_layers
        doc = fitz.open()
        doc.new_page()
        pdf_bytes = doc.tobytes()
        doc.close()
        result = read_pdf_layers(pdf_bytes)
        assert result.has_layers is False
        assert result.layers == []

    def test_visibility_dict_empty_when_no_layers(self, tmp_path):
        import fitz
        from cleaner.layer_reader import read_pdf_layers
        doc = fitz.open()
        doc.new_page()
        result = read_pdf_layers(doc.tobytes())
        doc.close()
        assert result.visibility == {}

    def test_render_with_no_visibility_returns_ndarray(self, tmp_path):
        import fitz
        from cleaner.layer_reader import render_with_visibility
        doc = fitz.open()
        doc.new_page()
        pdf_bytes = doc.tobytes()
        doc.close()
        img = render_with_visibility(pdf_bytes, {}, 0, 1.0)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3


# ── mask_package ──────────────────────────────────────────────────────────────

class TestMaskPackage:
    def test_shape_matches_input(self, pkg, plan_img):
        assert pkg.shape == plan_img.shape[:2]

    def test_all_masks_uint8(self, pkg):
        for name, m in pkg.as_dict().items():
            assert m.dtype == np.uint8, f"{name} dtype should be uint8"

    def test_architecture_locked_has_ink(self, pkg):
        assert pkg.architecture_locked.any(), "architecture_locked should detect wall edges"

    def test_hatch_removed_matches_input(self, pkg, hatch_mask):
        assert np.array_equal(pkg.hatch_removed, hatch_mask)

    def test_landscape_editable_not_in_locked(self, pkg):
        # No landscape pixel should be in architecture_locked
        overlap = cv2.bitwise_and(pkg.landscape_editable, pkg.architecture_locked)
        # Small overlap is allowed due to morphological ops — but should be < 5% of editable
        if pkg.landscape_editable.any():
            ratio = overlap.sum() / pkg.landscape_editable.sum()
            assert ratio < 0.05

    def test_apply_immutability_restores_locked(self, pkg, plan_img):
        from cleaner.mask_package import apply_immutability
        # Paint everything red
        red_canvas = np.zeros_like(plan_img)
        red_canvas[:] = (0, 0, 255)
        out = apply_immutability(red_canvas, pkg)
        # Locked pixels must not be red (must be black)
        locked_px = out[pkg.architecture_locked > 0]
        assert not (locked_px == [0, 0, 255]).all(axis=1).any()

    def test_composite_preview_returns_same_shape(self, pkg, plan_img):
        from cleaner.mask_package import composite_masks_preview
        out = composite_masks_preview(plan_img, pkg)
        assert out.shape == plan_img.shape

    def test_as_dict_has_five_keys(self, pkg):
        assert set(pkg.as_dict().keys()) == {
            "architecture_locked", "landscape_editable",
            "hatch_removed", "intervention_red", "analysis_blue",
        }


# ── zone_detector ─────────────────────────────────────────────────────────────

class TestZoneDetector:
    def test_auto_detect_returns_mask_and_polygons(self, plan_img, pkg):
        from cleaner.zone_detector import auto_detect_zones
        mask, polys = auto_detect_zones(plan_img, pkg.architecture_locked)
        assert mask.shape == plan_img.shape[:2]
        assert mask.dtype == np.uint8
        assert isinstance(polys, list)

    def test_auto_zone_not_in_locked(self, plan_img, pkg):
        from cleaner.zone_detector import auto_detect_zones
        mask, _ = auto_detect_zones(plan_img, pkg.architecture_locked)
        overlap = cv2.bitwise_and(mask, pkg.architecture_locked)
        assert overlap.sum() == 0

    def test_assign_tiers_three_masks(self, pkg):
        from cleaner.zone_detector import assign_tiers
        fg, mg, bg = assign_tiers(pkg.landscape_editable, pkg.architecture_locked)
        for m in (fg, mg, bg):
            assert m.shape == pkg.shape
            assert m.dtype == np.uint8

    def test_assign_tiers_no_overlap(self, pkg):
        from cleaner.zone_detector import assign_tiers
        fg, mg, bg = assign_tiers(pkg.landscape_editable, pkg.architecture_locked)
        ab = cv2.bitwise_and(fg, bg)
        assert ab.sum() == 0

    def test_canvas_json_to_mask_empty(self, plan_img):
        from cleaner.zone_detector import canvas_json_to_mask
        mask = canvas_json_to_mask({}, plan_img.shape[:2])
        assert mask.shape == plan_img.shape[:2]
        assert not mask.any()

    def test_canvas_json_rect(self, plan_img):
        from cleaner.zone_detector import canvas_json_to_mask
        jdata = {"objects": [{"type": "rect", "left": 10, "top": 10, "width": 50, "height": 40}]}
        mask = canvas_json_to_mask(jdata, plan_img.shape[:2])
        assert mask[20, 20] == 255

    def test_flood_fill_returns_mask(self, plan_img, pkg):
        from cleaner.zone_detector import flood_fill_zone
        mask = flood_fill_zone(plan_img, (200, 150), pkg.architecture_locked)
        assert mask.shape == plan_img.shape[:2]
        assert mask.dtype == np.uint8

    def test_merge_zone_masks(self, plan_img):
        from cleaner.zone_detector import merge_zone_masks
        a = np.zeros(plan_img.shape[:2], dtype=np.uint8)
        a[10:20, 10:20] = 255
        b = np.zeros_like(a)
        b[50:60, 50:60] = 255
        merged = merge_zone_masks(a, b)
        assert merged[15, 15] == 255
        assert merged[55, 55] == 255


# ── landscape_overlay ─────────────────────────────────────────────────────────

@pytest.fixture
def big_editable():
    """Large editable mask with a clear open area."""
    m = np.zeros((300, 400), dtype=np.uint8)
    m[50:250, 100:350] = 255
    return m


@pytest.fixture
def no_locked():
    return np.zeros((300, 400), dtype=np.uint8)


class TestLandscapeOverlay:
    def test_feature_tree_placed(self, plan_img, big_editable, no_locked):
        from cleaner.landscape_overlay import draw_feature_tree
        res = draw_feature_tree(plan_img, 200, 150, 20, big_editable, no_locked)
        assert res.canvas.shape == plan_img.shape
        assert res.placed

    def test_tree_outside_mask_rejected(self, plan_img, no_locked):
        from cleaner.landscape_overlay import draw_feature_tree
        # Editable mask is all zeros
        empty = np.zeros((300, 400), dtype=np.uint8)
        res = draw_feature_tree(plan_img, 200, 150, 20, empty, no_locked)
        assert not res.placed

    def test_shade_tree(self, plan_img, big_editable, no_locked):
        from cleaner.landscape_overlay import draw_shade_tree
        res = draw_shade_tree(plan_img, 200, 150, 25, big_editable, no_locked)
        assert res.placed and res.canvas.shape == plan_img.shape

    def test_shrub_mass(self, plan_img, big_editable, no_locked):
        from cleaner.landscape_overlay import draw_shrub_mass
        res = draw_shrub_mass(plan_img, 200, 150, 20, 15, big_editable, no_locked)
        assert res.canvas.shape == plan_img.shape

    def test_gravel_stipple(self, plan_img, big_editable, no_locked):
        from cleaner.landscape_overlay import draw_gravel_stipple
        res = draw_gravel_stipple(plan_img, big_editable, big_editable, no_locked)
        assert res.canvas.shape == plan_img.shape

    def test_boulder_cluster(self, plan_img, big_editable, no_locked):
        from cleaner.landscape_overlay import draw_boulder_cluster
        res = draw_boulder_cluster(plan_img, 200, 150, big_editable, no_locked)
        assert res.canvas.shape == plan_img.shape

    def test_stepping_stones(self, plan_img, big_editable, no_locked):
        from cleaner.landscape_overlay import draw_stepping_stones
        pts = [(120, 150), (200, 150), (280, 150)]
        res = draw_stepping_stones(plan_img, pts, big_editable, no_locked)
        assert res.canvas.shape == plan_img.shape

    def test_bed_edge(self, plan_img, big_editable, no_locked):
        from cleaner.landscape_overlay import draw_bed_edge
        pts = [(100, 200), (200, 180), (300, 200)]
        res = draw_bed_edge(plan_img, pts, big_editable, no_locked)
        assert res.canvas.shape == plan_img.shape

    def test_pond_edge(self, plan_img, big_editable, no_locked):
        from cleaner.landscape_overlay import draw_pond_edge
        res = draw_pond_edge(plan_img, 200, 150, 30, big_editable, no_locked)
        assert res.canvas.shape == plan_img.shape

    def test_immutability_respected(self, plan_img, big_editable):
        from cleaner.landscape_overlay import draw_feature_tree
        locked = np.zeros((300, 400), dtype=np.uint8)
        locked[130:170, 180:220] = 255
        original_locked_region = plan_img[130:170, 180:220].copy()
        res = draw_feature_tree(plan_img, 200, 150, 40, big_editable, locked)
        # Locked region must be unchanged (restored to original)
        restored = res.canvas[130:170, 180:220]
        assert np.array_equal(restored, original_locked_region)


# ── ai_proposal ───────────────────────────────────────────────────────────────

class TestAIProposal:
    def test_procedural_backend_returns_result(self, plan_img, pkg):
        from cleaner.ai_proposal import ProceduralBackend
        backend = ProceduralBackend()
        result = backend.propose(plan_img, pkg, "naturalistic")
        assert result.overlay_png is not None
        assert result.overlay_png.ndim == 3
        assert result.backend_used == "Procedural (no API)"

    def test_procedural_no_error(self, plan_img, pkg):
        from cleaner.ai_proposal import ProceduralBackend
        result = ProceduralBackend().propose(plan_img, pkg)
        assert result.error is None

    def test_stubs_return_error_message(self, plan_img, pkg):
        from cleaner.ai_proposal import GeminiVisionBackend, OpenAIBackend
        for BackendCls in (GeminiVisionBackend, OpenAIBackend):
            result = BackendCls().propose(plan_img, pkg)
            assert result.error is not None

    def test_backends_dict_has_expected_keys(self):
        from cleaner.ai_proposal import BACKENDS
        assert "Claude Vision" in BACKENDS
        assert "Procedural (no API)" in BACKENDS


# ── export ────────────────────────────────────────────────────────────────────

class TestExport:
    def test_save_landscape_bundle_creates_files(self, plan_img, pkg, tmp_path):
        from cleaner.export import save_landscape_bundle
        masks = pkg.as_dict()
        bundle = save_landscape_bundle(
            base_dir=tmp_path,
            stem="test",
            cleaned_pdf=None,
            original_bgr=plan_img,
            cleaned_bgr=plan_img,
            proposal_bgr=None,
            masks=masks,
            svg_elements=["<circle cx='10' cy='10' r='5'/>"],
            primitives=[{"type": "tree", "cx": 100, "cy": 100, "radius": 15}],
            debug_data={"test": True},
        )
        assert "preview" in bundle
        assert bundle["preview"].exists()
        assert "svg" in bundle
        assert bundle["svg"].exists()
        assert "grasshopper_json" in bundle
        assert bundle["grasshopper_json"].exists()
        assert "metadata" in bundle

    def test_grasshopper_json_valid(self, plan_img, pkg, tmp_path):
        from cleaner.export import save_landscape_bundle
        bundle = save_landscape_bundle(
            tmp_path, "test", None, plan_img, plan_img, None,
            pkg.as_dict(), [], [{"type": "tree", "cx": 50, "cy": 50, "radius": 10}], {}
        )
        data = json.loads(bundle["grasshopper_json"].read_text())
        assert "crs" in data
        assert "landscape_zones" in data
        assert "primitives" in data
        assert data["primitives"][0]["type"] == "tree"

    def test_svg_is_valid_xml(self, plan_img, pkg, tmp_path):
        import xml.etree.ElementTree as ET
        from cleaner.export import save_landscape_bundle
        bundle = save_landscape_bundle(
            tmp_path, "test", None, plan_img, plan_img, None,
            {}, ["<circle cx='5' cy='5' r='3'/>"], [], {}
        )
        # Should parse without error
        ET.parse(str(bundle["svg"]))

    def test_mask_files_saved(self, plan_img, pkg, tmp_path):
        from cleaner.export import save_landscape_bundle
        bundle = save_landscape_bundle(
            tmp_path, "test", None, plan_img, plan_img, None,
            pkg.as_dict(), [], [], {}
        )
        for key in ("mask_architecture_locked", "mask_landscape_editable"):
            assert key in bundle, f"Missing {key}"
            assert bundle[key].exists()
