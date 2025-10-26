"""OCR interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class OcrResult:
    """Container for OCR output."""

    text: str
    confidence: float = 1.0


class OCR(ABC):
    """Abstract OCR backend interface."""

    @abstractmethod
    def load(self) -> None:
        """Allocate resources (models, sessions, etc.)."""

    @abstractmethod
    def recognize(self, crops: Sequence[np.ndarray]) -> List[OcrResult]:
        """Return text predictions for the provided crops."""

    def warmup(self, *, batch_size: int, iterations: int = 1, image_shape: tuple[int, int, int] | None = None) -> None:
        _ = batch_size
        _ = iterations
        _ = image_shape


__all__ = ["OCR", "OcrResult"]
