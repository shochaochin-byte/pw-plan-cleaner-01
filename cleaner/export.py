from __future__ import annotations

from pathlib import Path
import json
import cv2
import numpy as np


def save_outputs(
    base_dir: Path,
    stem: str,
    cleaned_pdf: bytes | None,
    cleaned_png_bgr: np.ndarray,
    debug_data,
    transparent_png: np.ndarray | None = None,
    svg_path: Path | None = None,
    landscape_mask: np.ndarray | None = None,
):
    base_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    if cleaned_pdf is not None:
        pdf_path = base_dir / f"{stem}_cleaned.pdf"
        pdf_path.write_bytes(cleaned_pdf)
        out["pdf"] = pdf_path

    png_path = base_dir / f"{stem}_cleaned.png"
    cv2.imwrite(str(png_path), cleaned_png_bgr)
    out["png"] = png_path

    if transparent_png is not None:
        tpng_path = base_dir / f"{stem}_transparent.png"
        cv2.imwrite(str(tpng_path), transparent_png)
        out["transparent_png"] = tpng_path

    if landscape_mask is not None:
        mask_path = base_dir / f"{stem}_landscape_mask.png"
        cv2.imwrite(str(mask_path), landscape_mask)
        out["landscape_mask"] = mask_path

    if svg_path is not None and svg_path.exists():
        out["svg"] = svg_path

    dbg_path = base_dir / f"{stem}_debug.json"
    dbg_path.write_text(json.dumps(debug_data, indent=2), encoding="utf-8")
    out["debug"] = dbg_path
    return out
