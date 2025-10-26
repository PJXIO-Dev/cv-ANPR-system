"""Visualization helpers."""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from yolo_ocr.pipeline.postprocess import PlatePrediction

# Geometry constants copied from the legacy visualization routine to preserve
# the bounding-box proportions, label layout, and styling.
CAR_W_FACTOR = 1.8
CAR_H_FACTOR = 1.4
CAR_W_CAP = 0.22
CAR_H_CAP = 0.28
CAR_UP_BIAS = 0.45


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _draw_vehicle_box_and_label(image: np.ndarray, plate_box: tuple[int, int, int, int], plate_text: str, confidence: float) -> None:
    """Render the legacy-styled vehicle box and label for a detected plate."""

    height, width = image.shape[:2]
    x1p, y1p, x2p, y2p = plate_box
    cx = int((x1p + x2p) * 0.5)
    cy = int((y1p + y2p) * 0.5)
    pw = max(1, x2p - x1p)
    ph = max(1, y2p - y1p)

    car_w = int(pw * CAR_W_FACTOR)
    car_h = int(ph * CAR_H_FACTOR)
    car_w = min(car_w, int(width * CAR_W_CAP))
    car_h = min(car_h, int(height * CAR_H_CAP))

    car_x1 = _clamp(cx - car_w // 2, 0, width - 1)
    car_y1 = _clamp(cy - int(car_h * CAR_UP_BIAS), 0, height - 1)
    car_x2 = _clamp(car_x1 + car_w, 0, width - 1)
    car_y2 = _clamp(car_y1 + car_h, 0, height - 1)

    cv2.rectangle(image, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 4)

    show_text = plate_text if plate_text else "None"
    label = f"{show_text} ({confidence:.2f})"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(1.2, (car_x2 - car_x1) / 700.0))
    text_size, _ = cv2.getTextSize(label, font, font_scale, 2)
    text_w, text_h = text_size
    pad_x, pad_y = 8, 6

    lbl_x1 = car_x1 + 4
    lbl_y1 = max(0, car_y1 - text_h - 2 * pad_y)
    lbl_y2 = lbl_y1 + text_h + 2 * pad_y
    lbl_x2 = lbl_x1 + text_w + 2 * pad_x

    cv2.rectangle(image, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), (255, 255, 255), -1)
    cv2.rectangle(image, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), (0, 0, 0), 2)

    text_x = lbl_x1 + pad_x
    text_y = lbl_y1 + pad_y + text_h
    cv2.putText(image, label, (text_x, text_y), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)


def draw_detections(image: np.ndarray, detections: Iterable[PlatePrediction]) -> np.ndarray:
    """Render detections and OCR results using the legacy visualization layout."""

    output = image.copy()
    for det in detections:
        if not det.text:
            continue
        x1, y1, x2, y2 = map(int, det.bbox)
        _draw_vehicle_box_and_label(output, (x1, y1, x2, y2), det.text, det.confidence)
    return output


__all__ = ["draw_detections"]
