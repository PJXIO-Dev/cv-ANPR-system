"""PaddleOCR backend stub."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from yolo_ocr.config import OCRConfig
from yolo_ocr.ocr.base import OCR, OcrResult


@dataclass
class PaddleOCRBackend(OCR):
    """Placeholder for PaddleOCR integration."""

    config: OCRConfig

    def load(self) -> None:
        raise NotImplementedError("Install paddleocr and implement load().")

    def recognize(self, crops: Sequence[np.ndarray]) -> List[OcrResult]:
        raise NotImplementedError("Install paddleocr and implement recognize().")


__all__ = ["PaddleOCRBackend"]
