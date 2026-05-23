"""Optional CAD/MCP bridge for URL-driven geometry payloads."""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class CADBridgePayload:
    units: str
    layers: list[dict[str, Any]]
    curves: list[dict[str, Any]]
    polylines: list[dict[str, Any]]
    source: str = "url"


_ALLOWED_UNITS = {"px", "mm", "cm", "m", "in", "ft"}


def parse_query_bridge_payload(query_params: Any, key: str = "cad_payload") -> dict[str, Any] | None:
    """Parse JSON payload from Streamlit query params.

    Supports either plain JSON text or URL-safe base64 encoded JSON.
    """
    if query_params is None:
        return None

    raw = query_params.get(key)
    if raw is None or raw == "":
        return None

    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if raw is None:
        return None

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")

    # Try plain JSON first.
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Fallback: URL-safe base64 JSON.
    try:
        padded = raw + "=" * (-len(raw) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8")
        return json.loads(decoded)
    except Exception:
        return None


def validate_bridge_payload(payload: dict[str, Any]) -> CADBridgePayload:
    if not isinstance(payload, dict):
        raise ValueError("CAD payload must be a JSON object")

    units = str(payload.get("units", "px")).lower()
    if units not in _ALLOWED_UNITS:
        raise ValueError(f"Unsupported units '{units}'")

    layers = payload.get("layers", [])
    curves = payload.get("curves", [])
    polylines = payload.get("polylines", [])
    for field_name, field_value in (("layers", layers), ("curves", curves), ("polylines", polylines)):
        if not isinstance(field_value, list):
            raise ValueError(f"{field_name} must be an array")

    return CADBridgePayload(units=units, layers=layers, curves=curves, polylines=polylines)


def sanitize_coordinate(value: Any, *, min_value: float = 0.0, max_value: float = 100000.0) -> float:
    try:
        v = float(value)
    except Exception:
        return min_value
    if not np.isfinite(v):
        return min_value
    return float(np.clip(v, min_value, max_value))


def payload_to_guidance_masks(payload: CADBridgePayload, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Convert bridge geometry into binary guidance masks used by downstream stages."""
    h, w = shape
    landscape_hint = np.zeros((h, w), dtype=np.uint8)
    architecture_hint = np.zeros((h, w), dtype=np.uint8)

    layer_by_id = {str(l.get("id", l.get("name", ""))): str(l.get("name", "")).lower() for l in payload.layers}

    def _draw_polyline(points: list[Any], layer_key: str, closed: bool = False):
        if len(points) < 2:
            return
        pts = []
        for p in points:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            x = int(round(sanitize_coordinate(p[0], max_value=w - 1)))
            y = int(round(sanitize_coordinate(p[1], max_value=h - 1)))
            pts.append([x, y])
        if len(pts) < 2:
            return
        arr = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))

        lname = layer_by_id.get(str(layer_key), str(layer_key).lower())
        target = landscape_hint if any(k in lname for k in ("land", "plant", "soft", "zone")) else architecture_hint
        cv2.polylines(target, [arr], isClosed=closed, color=255, thickness=3)
        if closed:
            cv2.fillPoly(target, [arr], color=255)

    for pl in payload.polylines:
        if not isinstance(pl, dict):
            continue
        _draw_polyline(pl.get("points", []), pl.get("layer", ""), bool(pl.get("closed", False)))

    for cv in payload.curves:
        if not isinstance(cv, dict):
            continue
        _draw_polyline(cv.get("control_points", cv.get("points", [])), cv.get("layer", ""), False)

    return {
        "landscape_hint": landscape_hint,
        "architecture_hint": architecture_hint,
    }


def serialize_for_export(payload: CADBridgePayload, guidance_masks: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "units": payload.units,
        "layers": payload.layers,
        "curves": payload.curves,
        "polylines": payload.polylines,
        "guidance": {
            "landscape_hint_pixels": int(np.count_nonzero(guidance_masks.get("landscape_hint", 0))),
            "architecture_hint_pixels": int(np.count_nonzero(guidance_masks.get("architecture_hint", 0))),
        },
    }
