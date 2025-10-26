"""Post-processing helpers for license plate predictions."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from yolo_ocr.config import PostprocessConfig
from yolo_ocr.detectors.base import Detection
from yolo_ocr.ocr.base import OcrResult


@dataclass
class PlatePrediction:
    """Unified representation of a detected plate and OCR text."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_name: str
    text: str


def sanitize_text(text: str) -> str:
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def is_valid_plate(text: str, config: PostprocessConfig) -> bool:
    if not text:
        return False
    if not config.plate_regex:
        return True
    pattern = re.compile(config.plate_regex)
    return bool(pattern.match(text))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def deduplicate(detections: List[Detection], config: PostprocessConfig) -> List[Detection]:
    filtered: List[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if det.confidence < config.min_confidence:
            continue
        if any(iou(det.bbox, kept.bbox) > config.dedup_iou_threshold for kept in filtered):
            continue
        filtered.append(det)
    if config.keep_top_k is not None:
        filtered = filtered[: config.keep_top_k]
    return filtered


def merge(detections: Sequence[Detection], ocr_results: Sequence[OcrResult], config: PostprocessConfig) -> List[PlatePrediction]:
    predictions: List[PlatePrediction] = []
    for det, ocr in zip(detections, ocr_results):
        text = sanitize_text(ocr.text)
        if not is_valid_plate(text, config) and config.replace_invalid_with_empty:
            text = ""
        predictions.append(
            PlatePrediction(
                bbox=det.bbox,
                confidence=det.confidence,
                class_name=det.class_name,
                text=text,
            )
        )
    return predictions


__all__ = ["PlatePrediction", "deduplicate", "merge", "sanitize_text", "is_valid_plate"]
