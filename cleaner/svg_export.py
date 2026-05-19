from __future__ import annotations

from pathlib import Path


def export_debug_svg(width: int, height: int, red_boxes: list[tuple[float, float, float, float]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for x0, y0, x1, y1 in red_boxes:
        lines.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="red" stroke-width="1"/>')
    lines.append('</svg>')
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
