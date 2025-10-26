"""ONNX Runtime backend for YOLO models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from yolo_ocr.config import DetectorConfig
from yolo_ocr.detectors.base import Detection, Detector


@dataclass
class YoloOnnxDetector(Detector):
    """Lightweight ONNX Runtime detector."""

    config: DetectorConfig

    def __post_init__(self) -> None:
        self._session = None
        self._class_names: dict[int, str] = {}

    def load(self) -> None:
        import onnxruntime as ort

        providers = None
        if self.config.device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(self.config.model_path, providers=providers)

    def infer(self, images: Sequence[np.ndarray]) -> List[List[Detection]]:
        if self._session is None:
            raise RuntimeError("Detector has not been loaded. Call load() first.")
        if not images:
            return []
        # Placeholder implementation: rely on ultralytics export format.
        raise NotImplementedError(
            "ONNX detector is provided as a stub. Export a model with scripts/export_onnx.py "
            "and implement post-processing for your target graph."
        )


__all__ = ["YoloOnnxDetector"]
