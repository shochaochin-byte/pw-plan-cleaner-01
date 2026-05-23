from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import cv2
import fitz
import numpy as np

from cleaner.ai_proposal import BACKENDS, ProposalResult
from cleaner.geometry_parser import parse_pdf_geometry
from cleaner.mask_package import build_mask_package
from cleaner.masking import mask_to_transparent_png
from cleaner.raster_cleaner import clean_raster_image
from cleaner.vector_cleaner import clean_vector_pdf_bytes
from cleaner.zone_detector import auto_detect_zones, assign_tiers, merge_zone_masks


@dataclass(frozen=True)
class PipelineContext:
    pdf_bytes: bytes
    before_bgr: np.ndarray
    sensitivity: float
    do_vector: bool
    zone_mode: str
    manual_zone_mask: np.ndarray | None = None
    ai_backend_name: str = "Disabled"
    ai_api_key: str = ""
    ai_prompt: str = ""
    ai_style: str = "balanced"
    ai_tier: str = "mid"

    # stage outputs
    geometry_meta: dict[str, Any] | None = None
    cleaned_pdf: bytes | None = None
    cleaned_bgr: np.ndarray | None = None
    debug_data: list[dict[str, Any]] | None = None
    red_mask: np.ndarray | None = None
    blue_mask: np.ndarray | None = None
    red_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    pkg: Any | None = None
    transparent_png: np.ndarray | None = None
    ai_result: ProposalResult | None = None
    stage_results: dict[str, dict[str, Any]] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_stage(ctx: PipelineContext, stage_key: str, fn, optional: bool = False) -> PipelineContext:
    start = perf_counter()
    started_at = _now_iso()
    try:
        out = fn(ctx)
        elapsed_ms = round((perf_counter() - start) * 1000, 2)
        sr = dict(out.stage_results)
        sr[stage_key] = {
            "name": stage_key,
            "status": "ok",
            "optional": optional,
            "started_at": started_at,
            "ended_at": _now_iso(),
            "duration_ms": elapsed_ms,
        }
        return replace(out, stage_results=sr)
    except Exception as exc:
        elapsed_ms = round((perf_counter() - start) * 1000, 2)
        sr = dict(ctx.stage_results)
        sr[stage_key] = {
            "name": stage_key,
            "status": "failed" if not optional else "skipped",
            "optional": optional,
            "started_at": started_at,
            "ended_at": _now_iso(),
            "duration_ms": elapsed_ms,
            "error": str(exc),
        }
        if optional:
            return replace(ctx, stage_results=sr)
        raise RuntimeError(f"Pipeline stage failed: {stage_key}: {exc}")


def run_pipeline(context: PipelineContext, options: dict[str, Any] | None = None) -> PipelineContext:
    options = options or {}

    def stage1_hatch_clean(ctx: PipelineContext) -> PipelineContext:
        if ctx.do_vector:
            cleaned_pdf, debug_data, decisions_by_page = clean_vector_pdf_bytes(ctx.pdf_bytes, ctx.sensitivity)
            cleaned = ctx.before_bgr
            try:
                import cleaner.preview as _preview
                cleaned = _preview.render_pdf_page(cleaned_pdf, 0, 1.5)
            except Exception:
                pass
            red = np.zeros(ctx.before_bgr.shape[:2], dtype=np.uint8)
            blue = np.zeros(ctx.before_bgr.shape[:2], dtype=np.uint8)
            red_boxes = []
            decision_page = decisions_by_page[0] if decisions_by_page else []
            for d in decision_page:
                x0, y0, x1, y1 = [int(v * 1.5) for v in d["bbox"]]
                if d["remove"]:
                    red[y0:y1, x0:x1] = 255
                    red_boxes.append((x0, y0, x1, y1))
                else:
                    blue[y0:y1, x0:x1] = 255
            return replace(ctx, cleaned_pdf=cleaned_pdf, cleaned_bgr=cleaned, debug_data=debug_data, red_mask=red, blue_mask=blue, red_boxes=red_boxes)

        cleaned, overlay_rb = clean_raster_image(ctx.before_bgr, ctx.sensitivity)
        cleaned_pdf_doc = fitz.open()
        p = cleaned_pdf_doc.new_page(width=ctx.before_bgr.shape[1], height=ctx.before_bgr.shape[0])
        ok, buf = cv2.imencode('.png', cleaned)
        if not ok:
            raise RuntimeError("PNG encoding failed")
        p.insert_image(p.rect, stream=buf.tobytes())
        red = cv2.inRange(overlay_rb, (0, 0, 200), (20, 20, 255))
        blue = cv2.inRange(overlay_rb, (200, 0, 0), (255, 30, 30))
        return replace(ctx, cleaned_pdf=cleaned_pdf_doc.tobytes(), cleaned_bgr=cleaned, debug_data=[{"page": 0, "mode": "raster", "note": "fallback"}], red_mask=red, blue_mask=blue)

    def stage2_geometry_masks(ctx: PipelineContext) -> PipelineContext:
        gmeta = parse_pdf_geometry(ctx.pdf_bytes)
        pkg = build_mask_package(ctx.before_bgr, hatch_mask=ctx.red_mask)
        return replace(ctx, geometry_meta=gmeta, pkg=pkg)

    def stage3_manual_overlay(ctx: PipelineContext) -> PipelineContext:
        pkg = ctx.pkg
        if pkg is None:
            return ctx
        if ctx.zone_mode == "Auto-detect" or ctx.manual_zone_mask is None:
            auto_zone, _ = auto_detect_zones(ctx.before_bgr, pkg.architecture_locked)
            manual = ctx.manual_zone_mask
            zone_final = merge_zone_masks(auto_zone, manual) if manual is not None and manual.shape == auto_zone.shape else auto_zone
        else:
            zone_final = ctx.manual_zone_mask
        merged_land = cv2.bitwise_or(pkg.landscape_editable, zone_final)
        merged_land = cv2.bitwise_and(merged_land, cv2.bitwise_not(pkg.architecture_locked))
        pkg.landscape_editable[:] = merged_land
        fg, mg, bg = assign_tiers(pkg.landscape_editable, pkg.architecture_locked)
        pkg.tier_foreground, pkg.tier_midground, pkg.tier_background = fg, mg, bg
        transparent = mask_to_transparent_png(ctx.cleaned_bgr, cv2.bitwise_not(ctx.red_mask))
        return replace(ctx, pkg=pkg, transparent_png=transparent)

    def stage4_ai(ctx: PipelineContext) -> PipelineContext:
        if options.get("run_ai") and ctx.pkg is not None and ctx.ai_backend_name != "Disabled":
            from cleaner.ai_proposal import ClaudeVisionBackend
            backend = ClaudeVisionBackend(api_key=ctx.ai_api_key) if ctx.ai_backend_name == "Claude Vision" else BACKENDS[ctx.ai_backend_name]
            result = backend.propose(ctx.before_bgr, ctx.pkg, ctx.ai_prompt, ctx.ai_style, ctx.ai_tier)
            return replace(ctx, ai_result=result)
        return ctx

    def stage5_optional_cad_mcp(ctx: PipelineContext) -> PipelineContext:
        # Placeholder non-blocking stage: keep pipeline moving if unavailable.
        return ctx

    def stage6_render_export(ctx: PipelineContext) -> PipelineContext:
        return ctx

    stages = [
        ("1_hatch_clean", stage1_hatch_clean, False),
        ("2_geometry_mask_detection", stage2_geometry_masks, False),
        ("3_manual_overlay_merge", stage3_manual_overlay, False),
        ("4_ai_proposal", stage4_ai, False),
        ("5_optional_cad_mcp_handoff", stage5_optional_cad_mcp, True),
        ("6_render_export", stage6_render_export, False),
    ]

    ctx = context
    for key, fn, optional in stages:
        ctx = _run_stage(ctx, key, fn, optional=optional)
    return ctx
