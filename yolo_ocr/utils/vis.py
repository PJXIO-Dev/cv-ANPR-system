"""Visualization helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np

from yolo_ocr.pipeline.postprocess import PlatePrediction


@dataclass
class BoxStyle:
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 1


def draw_detections(image: np.ndarray, detections: Iterable[PlatePrediction], style: BoxStyle | None = None) -> np.ndarray:
    """Render detections and OCR results on top of an image."""

    style = style or BoxStyle()
    output = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), style.color, style.thickness)
        label = f"{det.text or 'UNK'} ({det.confidence:.2f})"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, style.font_scale, style.font_thickness)
        cv2.rectangle(output, (x1, y1 - text_h - baseline), (x1 + text_w, y1), style.color, -1)
        cv2.putText(
            output,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            style.font_scale,
            (0, 0, 0),
            style.font_thickness,
            lineType=cv2.LINE_AA,
        )
    return output


__all__ = ["draw_detections", "BoxStyle"]
