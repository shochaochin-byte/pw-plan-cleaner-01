from __future__ import annotations

import numpy as np


def build_vector_debug_masks(decision_page: list[dict], image_shape: tuple[int, int, int] | tuple[int, int], scale: float = 1.5):
    h, w = image_shape[:2]
    red = np.zeros((h, w), dtype=np.uint8)
    blue = np.zeros((h, w), dtype=np.uint8)
    red_boxes: list[tuple[int, int, int, int]] = []
    skipped_invalid_bboxes = 0

    for d in decision_page:
        x0, y0, x1, y1 = [int(v * scale) for v in d["bbox"]]
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))

        if x1 <= x0 or y1 <= y0:
            skipped_invalid_bboxes += 1
            continue

        if d["remove"]:
            red[y0:y1, x0:x1] = 255
            red_boxes.append((x0, y0, x1, y1))
        else:
            blue[y0:y1, x0:x1] = 255

    return red, blue, red_boxes, skipped_invalid_bboxes
