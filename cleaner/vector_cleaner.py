from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
from typing import Any

import fitz


@dataclass
class DrawDecision:
    index: int
    remove: bool
    reason: str
    bbox: tuple[float, float, float, float]


SAFE_TEXT_MARGIN = 6.0


def _line_angle_deg(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return 0.0
    return round((degrees(atan2(dy, dx)) + 180.0) % 180.0, 1)


def _line_len(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return hypot(p2[0] - p1[0], p2[1] - p1[1])


def _bbox_from_items(items: list[tuple[Any, ...]]) -> fitz.Rect:
    pts: list[tuple[float, float]] = []
    for item in items:
        if not item:
            continue
        t = item[0]
        if t == "l":
            _, p1, p2 = item
            pts.extend([p1, p2])
    if not pts:
        return fitz.Rect(0, 0, 0, 0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return fitz.Rect(min(xs), min(ys), max(xs), max(ys))


def _touches_any(rect: fitz.Rect, blockers: list[fitz.Rect], pad: float) -> bool:
    ex = fitz.Rect(rect.x0 - pad, rect.y0 - pad, rect.x1 + pad, rect.y1 + pad)
    return any(ex.intersects(b) for b in blockers)


def _collect_text_rects(page: fitz.Page) -> list[fitz.Rect]:
    text = page.get_text("rawdict")
    rects: list[fitz.Rect] = []
    for block in text.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if "bbox" in span:
                    rects.append(fitz.Rect(span["bbox"]))
    return rects


def detect_hatch_candidates(page: fitz.Page, sensitivity: float = 0.35) -> tuple[list[DrawDecision], list[dict[str, Any]]]:
    drawings = page.get_drawings()
    text_rects = _collect_text_rects(page)

    max_line_width = max(0.06, 0.22 - sensitivity * 0.14)
    max_avg_line_len = max(6.0, 20.0 - sensitivity * 9.0)
    min_line_count = max(5, int(10 - sensitivity * 5))
    max_long_seg_len = 80.0  # protect likely walls/major boundaries

    decisions: list[DrawDecision] = []
    removed: list[dict[str, Any]] = []

    # conservative title block band (bottom ~14%)
    title_block_zone = fitz.Rect(0, page.rect.height * 0.86, page.rect.width, page.rect.height)

    for i, d in enumerate(drawings):
        items = d.get("items", [])
        rect = fitz.Rect(d.get("rect") or _bbox_from_items(items))
        width = float(d.get("width") or 0.0)
        fill = d.get("fill")

        lines = [it for it in items if it and it[0] == "l"]
        if not lines:
            decisions.append(DrawDecision(i, False, "non-line-vector", (rect.x0, rect.y0, rect.x1, rect.y1)))
            continue

        lengths = [_line_len(p1, p2) for _, p1, p2 in lines]
        angles = [_line_angle_deg(p1, p2) for _, p1, p2 in lines]

        avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0
        unique_angles = len(set(angles))
        long_seg_count = sum(1 for ln in lengths if ln > max_long_seg_len)

        if _touches_any(rect, text_rects, SAFE_TEXT_MARGIN) or rect.intersects(title_block_zone):
            decisions.append(DrawDecision(i, False, "touches-text-or-title", (rect.x0, rect.y0, rect.x1, rect.y1)))
            continue

        dense_short_repeated = len(lines) >= min_line_count and avg_len <= max_avg_line_len and unique_angles <= 4
        light_stroke = width <= max_line_width
        likely_wall_or_opening = long_seg_count > 0 or width > (max_line_width * 1.9)

        remove = dense_short_repeated and light_stroke and not likely_wall_or_opening
        if fill is not None and dense_short_repeated and not likely_wall_or_opening:
            remove = True

        reason = "hatch-candidate" if remove else "preserve-architectural"
        decisions.append(DrawDecision(i, remove, reason, (rect.x0, rect.y0, rect.x1, rect.y1)))

        if remove:
            removed.append(
                {
                    "index": i,
                    "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                    "line_count": len(lines),
                    "avg_length": round(avg_len, 2),
                    "unique_angles": unique_angles,
                    "width": width,
                    "reason": reason,
                }
            )

    return decisions, removed


def clean_vector_pdf_bytes(pdf_bytes: bytes, sensitivity: float = 0.35) -> tuple[bytes, list[dict[str, Any]], list[list[dict[str, Any]]]]:
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = fitz.open()
    debug: list[dict[str, Any]] = []
    all_decisions: list[list[dict[str, Any]]] = []

    for pno in range(len(src)):
        page = src[pno]
        decisions, removed = detect_hatch_candidates(page, sensitivity=sensitivity)

        new_page = out.new_page(width=page.rect.width, height=page.rect.height)
        new_page.show_pdf_page(page.rect, src, pno)

        for r in removed:
            new_page.add_redact_annot(fitz.Rect(r["bbox"]), fill=(1, 1, 1))
        if removed:
            new_page.apply_redactions(images=0, graphics=1, text=0)

        debug.append({"page": pno, "removed": removed, "count": len(removed)})
        all_decisions.append([
            {"bbox": list(d.bbox), "remove": d.remove, "reason": d.reason, "index": d.index} for d in decisions
        ])

    return out.tobytes(), debug, all_decisions
