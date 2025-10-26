"""End-to-end detection + OCR pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np

from yolo_ocr.config import PipelineConfig
from yolo_ocr.detectors.base import Detection, Detector
from yolo_ocr.ocr.base import OCR
from yolo_ocr.pipeline import postprocess
from yolo_ocr.pipeline.postprocess import PlatePrediction
from yolo_ocr.utils.vis import draw_detections


@dataclass
class PipelineResult:
    """Result for a processed frame/image."""

    predictions: List[PlatePrediction]
    annotated: np.ndarray | None = None


class YoloOcrPipeline:
    """Coordinates detector, OCR backend and post-processing."""

    def __init__(self, detector: Detector, ocr: OCR, config: PipelineConfig) -> None:
        self.detector = detector
        self.ocr = ocr
        self.config = config

    def load(self) -> None:
        self.detector.load()
        self.ocr.load()
        if self.config.detector.warmup_iterations > 0:
            dummy_shape = (self.config.ocr.resize_height, self.config.ocr.resize_width, 3)
            self.detector.warmup(
                batch_size=self.config.detector.batch_size,
                iterations=self.config.detector.warmup_iterations,
                image_shape=dummy_shape,
            )

    def _resize_for_detection(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        target_width = self.config.detector.resize_target_width
        if target_width is None or image.shape[1] <= target_width:
            return image, 1.0
        scale = target_width / image.shape[1]
        new_size = (target_width, int(image.shape[0] * scale))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        return resized, scale

    def _get_roi(self, image_small: np.ndarray, scale: float, orig_height: int) -> tuple[np.ndarray, int]:
        if not self.config.detector.roi_from_center:
            return image_small, 0
        start = int(orig_height * self.config.detector.roi_start_fraction * scale)
        start = max(0, min(start, image_small.shape[0] - 1))
        return image_small[start:, :], start

    def _map_detection(self, det: Detection, scale: float, roi_offset: int) -> Detection:
        x1, y1, x2, y2 = det.bbox
        y1 += roi_offset
        y2 += roi_offset
        if scale != 1.0:
            x1 /= scale
            x2 /= scale
            y1 /= scale
            y2 /= scale
        return Detection(bbox=(x1, y1, x2, y2), confidence=det.confidence, class_id=det.class_id, class_name=det.class_name)

    def _crop_plate(self, image: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
        pad = self.config.ocr.padding
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return image[max(0, y1 - 1):min(h, y1 + 1), max(0, x1 - 1):min(w, x1 + 1)].copy()
        return image[y1:y2, x1:x2].copy()

    def process_image(self, image: np.ndarray) -> PipelineResult:
        resized, scale = self._resize_for_detection(image)
        roi_img, roi_offset = self._get_roi(resized, scale, image.shape[0])
        detections_batch = self.detector.infer([roi_img])
        detections_roi = detections_batch[0] if detections_batch else []
        mapped: List[Detection] = [self._map_detection(det, scale, roi_offset) for det in detections_roi]
        filtered = postprocess.deduplicate(mapped, self.config.postprocess)
        crops = [self._crop_plate(image, det.bbox) for det in filtered]
        ocr_results = self.ocr.recognize(crops)
        predictions = postprocess.merge(filtered, ocr_results, self.config.postprocess) if ocr_results else []
        annotated = None
        if self.config.visualize and predictions:
            annotated = draw_detections(image, predictions)
        return PipelineResult(predictions=predictions, annotated=annotated)

    def process_batch(self, images: Sequence[np.ndarray]) -> List[PipelineResult]:
        results: List[PipelineResult] = []
        if not images:
            return results

        resized_images: List[np.ndarray] = []
        scales: List[float] = []
        offsets: List[int] = []
        orig_images: List[np.ndarray] = []
        for image in images:
            resized, scale = self._resize_for_detection(image)
            roi_img, offset = self._get_roi(resized, scale, image.shape[0])
            resized_images.append(roi_img)
            scales.append(scale)
            offsets.append(offset)
            orig_images.append(image)

        detections_batches = self.detector.infer(resized_images)
        for dets, image, scale, offset in zip(detections_batches, orig_images, scales, offsets):
            mapped = [self._map_detection(det, scale, offset) for det in dets]
            filtered = postprocess.deduplicate(mapped, self.config.postprocess)
            crops = [self._crop_plate(image, det.bbox) for det in filtered]
            ocr_results = self.ocr.recognize(crops)
            predictions = postprocess.merge(filtered, ocr_results, self.config.postprocess) if ocr_results else []
            annotated = draw_detections(image, predictions) if self.config.visualize and predictions else None
            results.append(PipelineResult(predictions=predictions, annotated=annotated))
        return results


__all__ = ["YoloOcrPipeline", "PipelineResult"]
