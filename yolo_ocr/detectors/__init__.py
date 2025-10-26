"""Detector backends."""
from .base import Detection, Detector
from .yolo_ultralytics import YoloUltralyticsDetector
from .yolo_onnx import YoloOnnxDetector
from .yolo_tensorrt import YoloTensorRTDetector

__all__ = [
    "Detection",
    "Detector",
    "YoloUltralyticsDetector",
    "YoloOnnxDetector",
    "YoloTensorRTDetector",
]
