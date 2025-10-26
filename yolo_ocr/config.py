"""Configuration utilities for the YOLO + OCR pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DetectorConfig:
    """Configuration options for the detection backend."""

    backend: str = "yolo_ultralytics"
    model_path: str = "yolov8n.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "auto"
    fp16: bool = True
    batch_size: int = 1
    warmup_iterations: int = 1
    roi_from_center: bool = True
    roi_start_fraction: float = 0.5
    resize_target_width: Optional[int] = 1280


@dataclass
class OCRConfig:
    """Configuration options for OCR backends."""

    backend: str = "tesseract"
    language: str = "eng"
    padding: int = 8
    resize_width: int = 256
    resize_height: int = 128
    batch_size: int = 8
    fp16: bool = False


@dataclass
class PostprocessConfig:
    """Post-processing configuration values."""

    dedup_iou_threshold: float = 0.2
    min_confidence: float = 0.25
    keep_top_k: Optional[int] = None
    plate_regex: Optional[str] = None
    replace_invalid_with_empty: bool = True


@dataclass
class PipelineConfig:
    """High level pipeline configuration."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    num_workers: int = 0
    visualize: bool = True


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            target[key] = _deep_update(dict(target[key]), value)
        else:
            target[key] = value
    return target


def load_config(path: str | Path | None = None, overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    """Load configuration from a YAML file and optional overrides."""

    data: Dict[str, Any] = {}
    if path:
        with open(Path(path), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    if overrides:
        data = _deep_update(data, overrides)

    detector = DetectorConfig(**data.get("detector", {}))
    ocr = OCRConfig(**data.get("ocr", {}))
    postprocess = PostprocessConfig(**data.get("postprocess", {}))
    pipeline = PipelineConfig(
        detector=detector,
        ocr=ocr,
        postprocess=postprocess,
        num_workers=data.get("num_workers", 0),
        visualize=data.get("visualize", True),
    )
    return pipeline


__all__ = [
    "DetectorConfig",
    "OCRConfig",
    "PipelineConfig",
    "PostprocessConfig",
    "load_config",
]
