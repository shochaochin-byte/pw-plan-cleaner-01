"""Read OCG (Optional Content Group) layers from a PDF via PyMuPDF.

Falls back gracefully when a PDF has no layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import fitz
import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class OCGLayer:
    xref: int
    name: str
    on: bool
    locked: bool = False
    depth: int = 0


@dataclass
class LayerReadResult:
    layers: list[OCGLayer]
    has_layers: bool

    @property
    def visibility(self) -> dict[int, bool]:
        return {l.xref: l.on for l in self.layers}


def read_pdf_layers(pdf_bytes: bytes) -> LayerReadResult:
    """Return OCG layer list from *pdf_bytes*. Empty list if no OCG layers."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    raw = doc.get_ocgs()          # {xref: {"name", "on", ...}}
    ui  = doc.layer_ui_configs()  # [{"number", "text", "on", "locked", "depth", "type"}]
    doc.close()

    if not raw:
        return LayerReadResult(layers=[], has_layers=False)

    # Build xref → ui info map
    ui_map: dict[int, dict] = {}
    for item in ui:
        num = item.get("number", -1)
        if num in raw:
            ui_map[num] = item

    layers: list[OCGLayer] = []
    for xref, info in raw.items():
        ui_info = ui_map.get(xref, {})
        layers.append(
            OCGLayer(
                xref=xref,
                name=info.get("name", f"Layer {xref}"),
                on=info.get("on", True),
                locked=ui_info.get("locked", False),
                depth=ui_info.get("depth", 0),
            )
        )
    layers.sort(key=lambda l: l.name)
    return LayerReadResult(layers=layers, has_layers=True)


def render_with_visibility(
    pdf_bytes: bytes,
    visibility: dict[int, bool],
    page_index: int = 0,
    scale: float = 2.0,
) -> np.ndarray:
    """Render *page_index* of *pdf_bytes* with the given layer visibility.

    Returns BGR uint8 ndarray.
    """
    import cv2

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    on_xrefs  = [x for x, v in visibility.items() if v]
    off_xrefs = [x for x, v in visibility.items() if not v]
    if on_xrefs or off_xrefs:
        doc.set_layer(-1, on=on_xrefs, off=off_xrefs)

    page = doc[page_index]
    mat  = fitz.Matrix(scale, scale)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    doc.close()

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
