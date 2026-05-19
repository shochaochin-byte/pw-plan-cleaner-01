from __future__ import annotations

import cv2
import numpy as np


def clean_raster_image(image_bgr: np.ndarray, sensitivity: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray

    k = max(1, int(1 + sensitivity * 3))
    thin = cv2.morphologyEx(inv, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    strong = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    hatch_mask = cv2.threshold(thin, int(80 - sensitivity * 25), 255, cv2.THRESH_BINARY)[1]
    strong_mask = cv2.threshold(strong, 140, 255, cv2.THRESH_BINARY)[1]
    hatch_mask = cv2.bitwise_and(hatch_mask, cv2.bitwise_not(strong_mask))

    cleaned = image_bgr.copy()
    cleaned[hatch_mask > 0] = [255, 255, 255]

    overlay = image_bgr.copy()
    overlay[hatch_mask > 0] = [0, 0, 255]
    overlay[strong_mask > 0] = [255, 0, 0]
    return cleaned, overlay
