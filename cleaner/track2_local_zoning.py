from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import SAM


@dataclass
class ZoneSegmentationResult:
    binary_mask: np.ndarray
    total_area_pixels: int
    contours: list[np.ndarray]


class LocalZoningEngine:
    def __init__(self, model_checkpoint: str = "sam2.1_b.pt", device: str = "cpu"):
        self.model = SAM(model_checkpoint)
        self.device = device

    def segment_zone(self, image_path: str, click_x: int, click_y: int) -> ZoneSegmentationResult:
        results = self.model.predict(
            source=image_path,
            points=[[click_x, click_y]],
            labels=[1],
            device=self.device,
        )

        result_instance = results[0]
        binary_mask = result_instance.masks.data[0].cpu().numpy().astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area_pixels = int(np.sum(binary_mask == 255))
        return ZoneSegmentationResult(binary_mask, total_area_pixels, contours)
