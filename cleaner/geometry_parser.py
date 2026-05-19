from __future__ import annotations

from dataclasses import dataclass, asdict
from math import atan2, degrees, hypot
from typing import Any
import io

import fitz
import pdfplumber


@dataclass
class VectorPrimitive:
    kind: str
    bbox: tuple[float, float, float, float]
    width: float
    angle: float | None
    length: float | None
    page: int


def _line_angle(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return (degrees(atan2(p2[1] - p1[1], p2[0] - p1[0])) + 180.0) % 180.0


def _line_len(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return hypot(p2[0] - p1[0], p2[1] - p1[1])


def parse_pdf_geometry(pdf_bytes: bytes) -> dict[str, Any]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out: dict[str, Any] = {"pages": []}

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as plumber_doc:
        plumber_pages = plumber_doc.pages

        for pno, page in enumerate(doc):
            drawings = page.get_drawings()
            text_blocks = [fitz.Rect(b[:4]) for b in page.get_text("blocks") if len(b) >= 4]
            plumber_page = plumber_pages[pno]
            dims = plumber_page.extract_words()
            primitives: list[dict[str, Any]] = []

            for d in drawings:
                width = float(d.get("width") or 0.0)
                for it in d.get("items", []):
                    if it and it[0] == "l":
                        _, p1, p2 = it
                        l = _line_len(p1, p2)
                        a = round(_line_angle(p1, p2), 1)
                        bbox = (min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1]))
                        primitives.append(asdict(VectorPrimitive("line", bbox, width, a, l, pno)))

            out["pages"].append(
                {
                    "page": pno,
                    "drawing_count": len(drawings),
                    "text_count": len(text_blocks),
                    "dimension_token_count": len(dims),
                    "primitives": primitives,
                }
            )
    return out
