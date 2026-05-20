from __future__ import annotations

import json
from typing import Any


def parse_cad_geometry_payload(raw_payload: str) -> dict[str, Any]:
    payload = json.loads(raw_payload)
    curves = payload.get("curves")
    if not isinstance(curves, list):
        raise ValueError("Invalid payload: expected 'curves' array")
    return payload


def curves_to_segments(payload: dict[str, Any]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for curve in payload.get("curves", []):
        start = curve.get("start", {})
        end = curve.get("end", {})
        segments.append(((float(start["x"]), float(start["y"])), (float(end["x"]), float(end["y"]))))
    return segments
