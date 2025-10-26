"""Ultralytics YOLO detector backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from yolo_ocr.config import DetectorConfig
from yolo_ocr.detectors.base import Detection, Detector


@dataclass
class YoloUltralyticsDetector(Detector):
    """Wrapper around ``ultralytics`` models with batching and FP16 support."""

    config: DetectorConfig

    def __post_init__(self) -> None:
        self._model = None
        self._names: dict[int, str] | None = None

    def load(self) -> None:
        from ultralytics import YOLO

        device = None if self.config.device == "auto" else self.config.device
        self._model = YOLO(self.config.model_path)
        if device is not None:
            self._model.to(device)
        self._names = self._model.names

    def infer(self, images: Sequence[np.ndarray]) -> List[List[Detection]]:
        if self._model is None:
            raise RuntimeError("Detector has not been loaded. Call load() first.")

        if not images:
            return []

        results = self._model.predict(
            images,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            half=self.config.fp16,
            device=self.config.device,
            batch=self.config.batch_size,
            verbose=False,
        )

        detections_batch: List[List[Detection]] = []
        for res in results:
            res_detections: List[Detection] = []
            if res.boxes is None:
                detections_batch.append(res_detections)
                continue
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            for bbox, score, cls_idx in zip(boxes, scores, classes):
                name = self._names.get(cls_idx, str(cls_idx)) if self._names else str(cls_idx)
                res_detections.append(
                    Detection(
                        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        confidence=float(score),
                        class_id=int(cls_idx),
                        class_name=name,
                    )
                )
            detections_batch.append(res_detections)
        return detections_batch

    def warmup(self, *, batch_size: int, iterations: int = 1, image_shape: tuple[int, int, int] | None = None) -> None:
        if self._model is None:
            return
        h, w = (image_shape or (640, 640, 3))[:2]
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(iterations):
            self._model.predict(
                [dummy] * batch_size,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                half=self.config.fp16,
                device=self.config.device,
                batch=batch_size,
                verbose=False,
            )


__all__ = ["YoloUltralyticsDetector"]
