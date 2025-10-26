"""Detector interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class Detection:
    """Detection result returned by the detector."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class Detector(ABC):
    """Abstract base class for detection backends."""

    @abstractmethod
    def load(self) -> None:
        """Load underlying weights and allocate resources."""

    @abstractmethod
    def infer(self, images: Sequence[np.ndarray]) -> List[List[Detection]]:
        """Run inference on a batch of images and return detections."""

    def warmup(self, *, batch_size: int, iterations: int = 1, image_shape: tuple[int, int, int] | None = None) -> None:
        """Optional warmup hook for accelerators."""

        _ = batch_size
        _ = iterations
        _ = image_shape


__all__ = ["Detection", "Detector"]
