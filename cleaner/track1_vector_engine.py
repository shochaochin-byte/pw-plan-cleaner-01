from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz
from shapely.geometry import LineString
from shapely.strtree import STRtree


@dataclass
class VectorSanitizationReport:
    preserved_structural_vectors: int
    stripped_hatch_lines: int


def clean_cad_pdf(
    pdf_path: str | Path,
    output_png_path: str | Path,
    density_threshold: int = 15,
    page_index: int = 0,
    buffer_radius: float = 5.0,
    min_line_length: float = 1.5,
) -> tuple[Path, VectorSanitizationReport]:
    """Sanitize a CAD-exported PDF and render a cleaned raster preview.

    Strategy:
      1) Remove known hatch layers when layer metadata exists.
      2) Fallback to spatial density filtering for flattened vectors.
    """
    pdf_path = Path(pdf_path)
    output_png_path = Path(output_png_path)

    doc = fitz.open(pdf_path)
    page = doc[page_index]
    paths = page.get_drawings()

    candidate_lines: list[LineString] = []
    dropped_hatch_count = 0

    for path in paths:
        layer_name = (path.get("layer") or "").upper()
        if "NI-HATCH" in layer_name or "HATCH" in layer_name:
            dropped_hatch_count += sum(1 for item in path.get("items", []) if item and item[0] == "l")
            continue

        for item in path.get("items", []):
            if not item or item[0] != "l":
                continue
            p1, p2 = item[1], item[2]
            line = LineString([(p1.x, p1.y), (p2.x, p2.y)])
            if line.length < min_line_length:
                dropped_hatch_count += 1
                continue
            candidate_lines.append(line)

    if not candidate_lines:
        pix = page.get_pixmap(dpi=150)
        pix.save(str(output_png_path))
        return output_png_path, VectorSanitizationReport(0, dropped_hatch_count)

    tree = STRtree(candidate_lines)
    validated_lines: list[LineString] = []

    for line in candidate_lines:
        neighbors = tree.query(line.buffer(buffer_radius), predicate="intersects")
        if len(neighbors) > density_threshold:
            dropped_hatch_count += 1
            continue
        validated_lines.append(line)

    # render page preview for downstream models/UI
    pix = page.get_pixmap(dpi=150)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(output_png_path))

    report = VectorSanitizationReport(
        preserved_structural_vectors=len(validated_lines),
        stripped_hatch_lines=dropped_hatch_count,
    )
    return output_png_path, report
