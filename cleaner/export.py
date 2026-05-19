from __future__ import annotations

from pathlib import Path
import json
import cv2


def save_outputs(base_dir: Path, stem: str, cleaned_pdf: bytes | None, cleaned_png_bgr, debug_data):
    base_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    if cleaned_pdf is not None:
        pdf_path = base_dir / f"{stem}_cleaned.pdf"
        pdf_path.write_bytes(cleaned_pdf)
        out["pdf"] = pdf_path

    png_path = base_dir / f"{stem}_cleaned.png"
    cv2.imwrite(str(png_path), cleaned_png_bgr)
    out["png"] = png_path

    dbg_path = base_dir / f"{stem}_debug.json"
    dbg_path.write_text(json.dumps(debug_data, indent=2), encoding="utf-8")
    out["debug"] = dbg_path
    return out
