"""TensorRT backend placeholder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from yolo_ocr.config import DetectorConfig
from yolo_ocr.detectors.base import Detection, Detector


@dataclass
class YoloTensorRTDetector(Detector):
    """Stub implementation illustrating the expected interface."""

    config: DetectorConfig

    def load(self) -> None:
        raise NotImplementedError(
            "TensorRT backend is not implemented in this template. "
            "Use scripts/export_onnx.py to generate an engine and implement inference."
        )

    def infer(self, images: Sequence[np.ndarray]) -> List[List[Detection]]:
        raise NotImplementedError("TensorRT backend is not implemented yet.")


__all__ = ["YoloTensorRTDetector"]
