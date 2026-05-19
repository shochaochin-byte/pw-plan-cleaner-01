from __future__ import annotations

import fitz
import numpy as np
import cv2


def render_pdf_page(pdf_bytes: bytes, page_index: int = 0, zoom: float = 2.0) -> np.ndarray:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
