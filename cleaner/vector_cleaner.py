from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import fitz


@dataclass
class DrawDecision:
    index: int
    remove: bool
    reason: str
    bbox: tuple[float, float, float, float]


SAFE_TEXT_MARGIN = 4.0


def _line_angle_deg(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    import math

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return 0.0
    return round((math.degrees(math.atan2(dy, dx)) + 180) % 180, 1)


def _bbox_from_items(items: list[tuple[Any, ...]]) -> fitz.Rect:
    pts: list[fitz.Point] = []
    for item in items:
        if item[0] == "l":
            _, p1, p2 = item
            pts.extend([fitz.Point(p1), fitz.Point(p2)])
    if not pts:
        return fitz.Rect(0, 0, 0, 0)
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return fitz.Rect(min(xs), min(ys), max(xs), max(ys))


def _touches_any(rect: fitz.Rect, blockers: list[fitz.Rect], pad: float = SAFE_TEXT_MARGIN) -> bool:
    expanded = fitz.Rect(rect.x0 - pad, rect.y0 - pad, rect.x1 + pad, rect.y1 + pad)
    return any(expanded.intersects(b) for b in blockers)


def detect_hatch_candidates(page: fitz.Page, sensitivity: float = 0.35) -> tuple[list[DrawDecision], list[dict[str, Any]]]:
    drawings = page.get_drawings()
    text_blocks = [fitz.Rect(b[:4]) for b in page.get_text("blocks") if len(b) >= 4]

    # Conservative defaults
    max_line_width = max(0.07, 0.25 - sensitivity * 0.15)
    max_avg_line_len = max(8.0, 24.0 - sensitivity * 10.0)
    min_line_count = int(8 - sensitivity * 4)

    decisions: list[DrawDecision] = []
    removed: list[dict[str, Any]] = []

    for i, d in enumerate(drawings):
        items = d.get("items", [])
        width = float(d.get("width") or 0.0)
        fill = d.get("fill")

        lines = [it for it in items if it and it[0] == "l"]
        if not lines:
            decisions.append(DrawDecision(i, False, "no-line-items", tuple(d.get("rect", (0, 0, 0, 0)))))
            continue

        lengths = []
        angles = []
        for _, p1, p2 in lines:
            p1p = fitz.Point(p1)
            p2p = fitz.Point(p2)
            lengths.append((p2p - p1p).unit.length * (p2p - p1p).norm())
            angles.append(_line_angle_deg(p1, p2))

        avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0
        unique_angles = len(set(angles))

        bbox = d.get("rect") or _bbox_from_items(items)
        rect = fitz.Rect(bbox)

        # Conservative safety: never remove if touching text block or title area lower strip
        title_block_zone = fitz.Rect(0, page.rect.height * 0.85, page.rect.width, page.rect.height)
        if _touches_any(rect, text_blocks) or rect.intersects(title_block_zone):
            decisions.append(DrawDecision(i, False, "touches-text-or-title", (rect.x0, rect.y0, rect.x1, rect.y1)))
            continue

        dense_short_repeated = len(lines) >= min_line_count and avg_len <= max_avg_line_len and unique_angles <= 3
        light_stroke = width <= max_line_width
        has_fill = fill is not None

        remove = dense_short_repeated and (light_stroke or has_fill)
        reason = "hatch-candidate" if remove else "preserve-architectural"

        decisions.append(DrawDecision(i, remove, reason, (rect.x0, rect.y0, rect.x1, rect.y1)))
        if remove:
            removed.append({
                "index": i,
                "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                "line_count": len(lines),
                "avg_length": round(avg_len, 2),
                "angles": sorted(list(set(angles)))[:8],
                "width": width,
                "reason": reason,
            })

    return decisions, removed


def build_clean_page_overlay(page: fitz.Page, decisions: list[DrawDecision]) -> fitz.Pixmap:
    shape = page.new_shape()
    for d in decisions:
        rect = fitz.Rect(d.bbox)
        color = (1, 0, 0) if d.remove else (0, 0, 1)
        shape.draw_rect(rect)
        shape.finish(color=color, fill=None, width=0.5)
    shape.commit()
    return page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)


def clean_vector_pdf_bytes(pdf_bytes: bytes, sensitivity: float = 0.35) -> tuple[bytes, list[dict[str, Any]], list[bytes]]:
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = fitz.open()
    debug: list[dict[str, Any]] = []
    previews: list[bytes] = []

    for pno in range(len(src)):
        page = src[pno]
        decisions, removed = detect_hatch_candidates(page, sensitivity=sensitivity)
        debug.append({"page": pno, "removed": removed, "count": len(removed)})

        # Rebuild page by raster fallback for visual parity while omitting redaction regions
        new_page = out.new_page(width=page.rect.width, height=page.rect.height)
        new_page.show_pdf_page(page.rect, src, pno)

        redacts = [fitz.Rect(r["bbox"]) for r in removed]
        for r in redacts:
            new_page.add_redact_annot(r, fill=(1, 1, 1))
        if redacts:
            new_page.apply_redactions(images=0, graphics=1, text=0)

        overlay_doc = fitz.open()
        ov_page = overlay_doc.new_page(width=page.rect.width, height=page.rect.height)
        shape = ov_page.new_shape()
        for d in decisions:
            rect = fitz.Rect(d.bbox)
            shape.draw_rect(rect)
            shape.finish(color=(1, 0, 0) if d.remove else (0, 0, 1), fill=None, width=0.5)
        shape.commit()
        previews.append(overlay_doc.tobytes())
        overlay_doc.close()

    return out.tobytes(), debug, previews
