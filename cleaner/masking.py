from __future__ import annotations

import cv2
import numpy as np


def mask_to_transparent_png(base_bgr: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
    rgba = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha_mask
    return rgba


def colorize_overlay(base_bgr: np.ndarray, red_mask: np.ndarray, blue_mask: np.ndarray, green_mask: np.ndarray) -> np.ndarray:
    out = base_bgr.copy()
    out[red_mask > 0] = [0, 0, 255]
    out[blue_mask > 0] = [255, 0, 0]
    out[green_mask > 0] = [0, 180, 0]
    return out
