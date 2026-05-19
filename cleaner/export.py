"""Export pipeline — full versioned bundle for PW-PLAN-CLEANER-01."""
from __future__ import annotations

import json
from pathlib import Path

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
    """Legacy single-output save. Kept for backwards compat."""
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


def save_landscape_bundle(
    base_dir: Path,
    stem: str,
    cleaned_pdf: bytes | None,
    original_bgr: np.ndarray,
    cleaned_bgr: np.ndarray,
    proposal_bgr: np.ndarray | None,
    masks: dict[str, np.ndarray],
    svg_elements: list[str],
    primitives: list[dict],
    debug_data: dict,
    dpi: int = 300,
) -> dict[str, Path]:
    """Save the full versioned export bundle.

    Bundle structure
    ----------------
    {base_dir}/{stem}_export/
      cleaned.pdf
      landscape_proposal.pdf   (if proposal available)
      overlay.svg
      preview.png
      masks/
        architecture_locked.png
        landscape_editable.png
        hatch_removed.png
        intervention_red.png
        analysis_blue.png
      grasshopper_exchange.json
      metadata.json
    """
    export_dir = base_dir / f"{stem}_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = export_dir / "masks"
    mask_dir.mkdir(exist_ok=True)

    out: dict[str, Path] = {}

    # cleaned PDF
    if cleaned_pdf is not None:
        p = export_dir / "cleaned.pdf"
        p.write_bytes(cleaned_pdf)
        out["pdf"] = p

    # preview PNG
    preview = proposal_bgr if proposal_bgr is not None else cleaned_bgr
    preview_path = export_dir / "preview.png"
    cv2.imwrite(str(preview_path), preview)
    out["preview"] = preview_path

    # masks
    mask_names = [
        "architecture_locked",
        "landscape_editable",
        "hatch_removed",
        "intervention_red",
        "analysis_blue",
    ]
    for name in mask_names:
        if name in masks and masks[name] is not None:
            mp = mask_dir / f"{name}.png"
            cv2.imwrite(str(mp), masks[name])
            out[f"mask_{name}"] = mp

    # SVG overlay
    h, w = original_bgr.shape[:2]
    svg_content = (
        f'<?xml version="1.0" encoding="utf-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">\n'
        + "\n".join(svg_elements)
        + "\n</svg>\n"
    )
    svg_path = export_dir / "overlay.svg"
    svg_path.write_text(svg_content, encoding="utf-8")
    out["svg"] = svg_path

    # Grasshopper exchange JSON
    h, w = original_bgr.shape[:2]
    gh_data = {
        "crs": {"unit": "px", "dpi": dpi, "width": w, "height": h},
        "landscape_zones": _mask_to_zones(masks.get("landscape_editable")),
        "primitives": primitives,
        "locked_regions": _mask_to_zones(masks.get("architecture_locked")),
    }
    gh_path = export_dir / "grasshopper_exchange.json"
    gh_path.write_text(json.dumps(gh_data, indent=2), encoding="utf-8")
    out["grasshopper_json"] = gh_path

    # metadata
    meta_path = export_dir / "metadata.json"
    meta_path.write_text(json.dumps(debug_data, indent=2), encoding="utf-8")
    out["metadata"] = meta_path

    return out


def _mask_to_zones(mask: np.ndarray | None) -> list[dict]:
    """Convert a binary mask to a list of polygon zone dicts."""
    if mask is None or not mask.any():
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zones = []
    for i, c in enumerate(contours):
        if cv2.contourArea(c) < 100:
            continue
        zones.append({
            "id": f"zone_{i}",
            "polygon": c.reshape(-1, 2).tolist(),
            "area_px": int(cv2.contourArea(c)),
        })
    return zones
