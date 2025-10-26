"""Public API for the YOLO + OCR pipeline."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np

from yolo_ocr.config import DetectorConfig, PipelineConfig, load_config
from yolo_ocr.detectors.base import Detector
from yolo_ocr.detectors.yolo_ultralytics import YoloUltralyticsDetector
from yolo_ocr.detectors.yolo_onnx import YoloOnnxDetector
from yolo_ocr.detectors.yolo_tensorrt import YoloTensorRTDetector
from yolo_ocr.ocr.base import OCR
from yolo_ocr.ocr.paddle_ocr import PaddleOCRBackend
from yolo_ocr.ocr.tesseract_ocr import TesseractOCR
from yolo_ocr.ocr.trocr_hf import TrOcrHF
from yolo_ocr.pipeline.pipeline import PipelineResult, YoloOcrPipeline
from yolo_ocr.utils.io import VideoReader


def _resolve_detector_device(config: DetectorConfig) -> None:
    """Ensure the detector device is compatible with the runtime environment."""

    requested = (config.device or "").lower()
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except Exception:  # pragma: no cover - torch availability depends on install
        cuda_available = False

    if requested in {"", "auto"}:
        config.device = "cuda:0" if cuda_available else "cpu"
    elif requested.startswith("cuda") and not cuda_available:
        warnings.warn(
            "CUDA device requested but not available. Falling back to CPU.",
            RuntimeWarning,
        )
        config.device = "cpu"

    if not config.device or config.device.startswith("cpu"):
        config.device = "cpu"
        config.fp16 = False


def _prepare_config(config: PipelineConfig) -> PipelineConfig:
    _resolve_detector_device(config.detector)
    return config


def _build_detector(config: PipelineConfig) -> Detector:
    backend = config.detector.backend.lower()
    if backend == "yolo_ultralytics":
        return YoloUltralyticsDetector(config.detector)
    if backend == "yolo_onnx":
        return YoloOnnxDetector(config.detector)
    if backend == "yolo_tensorrt":
        return YoloTensorRTDetector(config.detector)
    raise ValueError(f"Unsupported detector backend: {backend}")


def _build_ocr(config: PipelineConfig) -> OCR:
    backend = config.ocr.backend.lower()
    if backend == "tesseract":
        return TesseractOCR(config.ocr)
    if backend == "paddleocr":
        return PaddleOCRBackend(config.ocr)
    if backend == "trocr":
        return TrOcrHF(config.ocr)
    raise ValueError(f"Unsupported OCR backend: {backend}")


def create_pipeline(config_path: str | Path | None = None, overrides: dict | None = None) -> YoloOcrPipeline:
    """Instantiate and load a pipeline from config."""

    config = _prepare_config(load_config(config_path, overrides))
    detector = _build_detector(config)
    ocr = _build_ocr(config)
    pipeline = YoloOcrPipeline(detector=detector, ocr=ocr, config=config)
    pipeline.load()
    return pipeline


def run_on_image(image: np.ndarray, config: PipelineConfig | None = None) -> PipelineResult:
    """Run the pipeline on a single image array."""

    cfg = _prepare_config(config or load_config())
    pipeline = YoloOcrPipeline(detector=_build_detector(cfg), ocr=_build_ocr(cfg), config=cfg)
    pipeline.load()
    return pipeline.process_image(image)


def run_on_video(source: str | int, config_path: str | Path | None = None, *, stride: int = 1) -> Iterable[PipelineResult]:
    """Generator that yields ``PipelineResult`` for every processed frame."""

    pipeline = create_pipeline(config_path)
    with VideoReader(source, stride=stride) as reader:
        for frame in reader.frames():
            yield pipeline.process_image(frame.image)


__all__ = ["create_pipeline", "run_on_image", "run_on_video", "PipelineResult", "YoloOcrPipeline"]
