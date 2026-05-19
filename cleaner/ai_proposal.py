"""Pluggable AI backend for landscape proposal generation.

Primary: ClaudeVisionBackend — uses claude-sonnet-4-6 with tool use.
Stubs: GeminiVisionBackend, OpenAIBackend (need API keys).
Fallback: ProceduralBackend — deterministic, no API.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from cleaner.landscape_overlay import (
    draw_feature_tree,
    draw_shade_tree,
    draw_shrub_mass,
    draw_boulder_cluster,
    draw_gravel_stipple,
    draw_stepping_stones,
    draw_grass_hatch,
    draw_bed_edge,
)
from cleaner.mask_package import MaskPackage, apply_immutability


@dataclass
class ProposalResult:
    overlay_png: np.ndarray          # RGBA overlay to composite on plan
    svg_paths: list[str]             # SVG elements for export
    zone_labels: dict[str, list]     # {"foreground": [...], "midground": [...], "background": [...]}
    primitives: list[dict]           # placed primitive records for GH JSON
    confidence: float = 1.0
    backend_used: str = "unknown"
    error: str | None = None


@runtime_checkable
class LandscapeAIBackend(Protocol):
    def propose(
        self,
        plan_image: np.ndarray,
        pkg: MaskPackage,
        prompt: str,
        style: str,
        tier: str,
    ) -> ProposalResult: ...

    @property
    def name(self) -> str: ...


# ── helpers ───────────────────────────────────────────────────────────────────

def _bgr_to_b64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.standard_b64encode(buf.tobytes()).decode()


def _blank_overlay(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.zeros((h, w, 4), dtype=np.uint8)


def _apply_primitives(
    plan_bgr: np.ndarray,
    pkg: MaskPackage,
    records: list[dict],
) -> tuple[np.ndarray, list[str]]:
    """Render a list of primitive dicts onto *plan_bgr*."""
    canvas = plan_bgr.copy()
    all_svg: list[str] = []
    tier_mask = _tier_to_mask(pkg, "all")

    for rec in records:
        t = rec.get("type", "")
        try:
            if t == "tree":
                res = draw_feature_tree(canvas, rec["cx"], rec["cy"], rec["radius"],
                                        tier_mask, pkg.architecture_locked)
            elif t == "shade_tree":
                res = draw_shade_tree(canvas, rec["cx"], rec["cy"], rec["radius"],
                                      tier_mask, pkg.architecture_locked)
            elif t == "shrub":
                res = draw_shrub_mass(canvas, rec["cx"], rec["cy"], rec["rx"], rec["ry"],
                                      tier_mask, pkg.architecture_locked)
            elif t == "boulder":
                res = draw_boulder_cluster(canvas, rec["cx"], rec["cy"],
                                           tier_mask, pkg.architecture_locked,
                                           n_boulders=rec.get("count", 4),
                                           spread=rec.get("spread", 25))
            elif t == "gravel":
                region = tier_mask
                res = draw_gravel_stipple(canvas, region, tier_mask, pkg.architecture_locked)
            elif t == "stepping_stones":
                pts = [tuple(p) for p in rec.get("points", [])]
                res = draw_stepping_stones(canvas, pts, tier_mask, pkg.architecture_locked)
            elif t == "grass":
                res = draw_grass_hatch(canvas, tier_mask, tier_mask, pkg.architecture_locked)
            elif t == "bed_edge":
                pts = [tuple(p) for p in rec.get("points", [])]
                res = draw_bed_edge(canvas, pts, tier_mask, pkg.architecture_locked)
            else:
                continue
            canvas = res.canvas
            all_svg.extend(res.svg_commands)
        except Exception:
            continue

    return canvas, all_svg


def _tier_to_mask(pkg: MaskPackage, tier: str) -> np.ndarray:
    if tier == "foreground" and pkg.tier_foreground.size:
        return pkg.tier_foreground
    if tier == "midground" and pkg.tier_midground.size:
        return pkg.tier_midground
    if tier == "background" and pkg.tier_background.size:
        return pkg.tier_background
    return pkg.landscape_editable


# ── ClaudeVisionBackend ───────────────────────────────────────────────────────

_CLAUDE_TOOLS = [
    {
        "name": "place_tree",
        "description": "Place a feature tree symbol at the given plan coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cx": {"type": "integer", "description": "Centre x in pixels"},
                "cy": {"type": "integer", "description": "Centre y in pixels"},
                "radius": {"type": "integer", "description": "Tree canopy radius in pixels"},
                "tier": {"type": "string", "enum": ["foreground", "midground", "background"]},
            },
            "required": ["cx", "cy", "radius"],
        },
    },
    {
        "name": "place_shade_tree",
        "description": "Place a larger shade tree symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cx": {"type": "integer"}, "cy": {"type": "integer"},
                "radius": {"type": "integer"},
                "tier": {"type": "string", "enum": ["foreground", "midground", "background"]},
            },
            "required": ["cx", "cy", "radius"],
        },
    },
    {
        "name": "place_shrub_mass",
        "description": "Place a shrub mass blob.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cx": {"type": "integer"}, "cy": {"type": "integer"},
                "rx": {"type": "integer", "description": "X radius"},
                "ry": {"type": "integer", "description": "Y radius"},
                "tier": {"type": "string", "enum": ["foreground", "midground", "background"]},
            },
            "required": ["cx", "cy", "rx", "ry"],
        },
    },
    {
        "name": "place_boulder_cluster",
        "description": "Place a cluster of boulders.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cx": {"type": "integer"}, "cy": {"type": "integer"},
                "count": {"type": "integer"}, "spread": {"type": "integer"},
            },
            "required": ["cx", "cy"],
        },
    },
    {
        "name": "draw_bed_edge",
        "description": "Draw a curved planting bed edge through control points.",
        "input_schema": {
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                    "description": "List of [x,y] control points",
                },
            },
            "required": ["points"],
        },
    },
]

_STYLE_PROMPTS = {
    "Tropical":     "Use lush canopy trees, dense shrub masses, and pond-edge planting. Organic curved bed edges.",
    "Formal":       "Use symmetrical tree placement, clipped shrub masses, stepping stone paths on axis.",
    "Naturalistic": "Use informal tree clusters, scattered boulders, gravel stipple, and flowing bed edges.",
    "Minimalist":   "Use sparse feature trees, large gravel zones, and simple stepping stone paths.",
}


class ClaudeVisionBackend:
    name = "Claude Vision"

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6"):
        self._model = model
        self._api_key = api_key

    def propose(
        self,
        plan_image: np.ndarray,
        pkg: MaskPackage,
        prompt: str,
        style: str = "Naturalistic",
        tier: str = "all",
    ) -> ProposalResult:
        try:
            import anthropic
        except ImportError:
            return ProposalResult(
                _blank_overlay(pkg.shape), [], {}, [], 0.0,
                self.name, "anthropic package not installed"
            )

        client = anthropic.Anthropic(api_key=self._api_key)

        # Composite plan + editable zones as visual context
        preview = plan_image.copy()
        preview[pkg.landscape_editable > 0] = [
            int(preview[pkg.landscape_editable > 0][:, 0].mean() * 0.8 + 200 * 0.2) if pkg.landscape_editable.any() else 200,
            int(preview[pkg.landscape_editable > 0][:, 1].mean() * 0.8 + 220 * 0.2) if pkg.landscape_editable.any() else 220,
            int(preview[pkg.landscape_editable > 0][:, 2].mean() * 0.8 + 200 * 0.2) if pkg.landscape_editable.any() else 200,
        ] if pkg.landscape_editable.any() else preview[pkg.landscape_editable > 0]

        b64 = _bgr_to_b64(preview)
        h, w = pkg.shape
        style_hint = _STYLE_PROMPTS.get(style, "")

        system = (
            "You are an expert landscape architect. You are given an architectural floor plan "
            "with the landscape/outdoor zones highlighted in light blue. "
            "Your task: place landscape elements ONLY inside the highlighted zones. "
            "Never place elements on walls, rooms, or architectural elements. "
            f"Image size: {w}×{h} pixels. "
            "Use the provided tools to place landscape elements. "
            "Place 5-15 elements appropriate to the scale and style."
        )

        user_content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            },
            {
                "type": "text",
                "text": (
                    f"Style: {style}. {style_hint}\n"
                    f"Tier focus: {tier}.\n"
                    f"Additional instruction: {prompt}\n\n"
                    "Place landscape elements now using the available tools."
                ),
            },
        ]

        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=system,
                tools=_CLAUDE_TOOLS,
                messages=[{"role": "user", "content": user_content}],
            )
        except Exception as e:
            return ProposalResult(_blank_overlay(pkg.shape), [], {}, [], 0.0, self.name, str(e))

        # Parse tool use calls into primitive records
        records: list[dict] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            inp = block.input
            if block.name == "place_tree":
                records.append({"type": "tree", "cx": inp["cx"], "cy": inp["cy"], "radius": inp["radius"]})
            elif block.name == "place_shade_tree":
                records.append({"type": "shade_tree", "cx": inp["cx"], "cy": inp["cy"], "radius": inp["radius"]})
            elif block.name == "place_shrub_mass":
                records.append({"type": "shrub", "cx": inp["cx"], "cy": inp["cy"],
                                 "rx": inp["rx"], "ry": inp["ry"]})
            elif block.name == "place_boulder_cluster":
                records.append({"type": "boulder", "cx": inp["cx"], "cy": inp["cy"],
                                 "count": inp.get("count", 4), "spread": inp.get("spread", 25)})
            elif block.name == "draw_bed_edge":
                records.append({"type": "bed_edge", "points": inp["points"]})

        canvas, svg = _apply_primitives(plan_image, pkg, records)
        canvas = apply_immutability(canvas, pkg)

        overlay = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
        return ProposalResult(overlay, svg, {tier: records}, records, 0.9, self.name)


# ── ProceduralBackend (no API) ────────────────────────────────────────────────

class ProceduralBackend:
    name = "Procedural (no API)"

    def propose(
        self,
        plan_image: np.ndarray,
        pkg: MaskPackage,
        prompt: str = "",
        style: str = "Naturalistic",
        tier: str = "all",
    ) -> ProposalResult:
        tier_mask = _tier_to_mask(pkg, tier)
        if not tier_mask.any():
            return ProposalResult(_blank_overlay(pkg.shape), [], {}, [], 0.0, self.name)

        # Find centroid clusters from the editable mask
        contours, _ = cv2.findContours(tier_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        records: list[dict] = []
        rng = np.random.default_rng(0)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 500:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            r = int(math.sqrt(area / math.pi) * 0.3)
            r = min(max(r, 12), 60)

            if style in ("Formal", "Minimalist"):
                records.append({"type": "tree", "cx": cx, "cy": cy, "radius": r})
            elif style == "Tropical":
                records.append({"type": "shade_tree", "cx": cx, "cy": cy, "radius": r + 10})
                records.append({"type": "shrub", "cx": cx + r, "cy": cy, "rx": r // 2, "ry": r // 3})
            else:
                records.append({"type": "tree", "cx": cx, "cy": cy, "radius": r})
                if area > 3000:
                    records.append({"type": "boulder", "cx": cx - r, "cy": cy + r,
                                    "count": 3, "spread": 20})

        canvas, svg = _apply_primitives(plan_image, pkg, records)
        canvas = apply_immutability(canvas, pkg)
        overlay = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
        return ProposalResult(overlay, svg, {tier: records}, records, 0.75, self.name)


# ── Stubs ─────────────────────────────────────────────────────────────────────

class GeminiVisionBackend:
    name = "Gemini Vision"

    def propose(self, plan_image, pkg, prompt="", style="Naturalistic", tier="all"):
        return ProposalResult(
            _blank_overlay(pkg.shape), [], {}, [], 0.0, self.name,
            "Gemini backend not yet implemented — add GOOGLE_API_KEY and install google-generativeai"
        )


class OpenAIBackend:
    name = "GPT-4o Vision"

    def propose(self, plan_image, pkg, prompt="", style="Naturalistic", tier="all"):
        return ProposalResult(
            _blank_overlay(pkg.shape), [], {}, [], 0.0, self.name,
            "OpenAI backend not yet implemented — add OPENAI_API_KEY and install openai"
        )


import math  # noqa: E402 (needed inside functions above)

BACKENDS: dict[str, LandscapeAIBackend] = {
    "Claude Vision":        ClaudeVisionBackend(),
    "Procedural (no API)":  ProceduralBackend(),
    "Gemini Vision":        GeminiVisionBackend(),
    "GPT-4o Vision":        OpenAIBackend(),
}
