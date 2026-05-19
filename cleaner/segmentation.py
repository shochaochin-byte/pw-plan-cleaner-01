from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class SegmentationMasks:
    architecture_mask: np.ndarray
    landscape_mask: np.ndarray
    hatch_mask: np.ndarray
    circulation_mask: np.ndarray


def build_segmentation_masks(image_bgr: np.ndarray, hatch_mask: np.ndarray | None = None) -> SegmentationMasks:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    arch = cv2.Canny(gray, 70, 180)
    arch = cv2.dilate(arch, np.ones((2, 2), np.uint8), iterations=1)

    # outdoors proxy: bright large open regions away from strong edges
    inv_edges = cv2.bitwise_not(arch)
    bright = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]
    landscape = cv2.bitwise_and(inv_edges, bright)
    landscape = cv2.morphologyEx(landscape, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # circulation proxy: medium-width connected pathways
    circ = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)[1]
    circ = cv2.morphologyEx(circ, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    if hatch_mask is None:
        hatch_mask = np.zeros_like(gray)

    return SegmentationMasks(arch, landscape, hatch_mask, circ)
