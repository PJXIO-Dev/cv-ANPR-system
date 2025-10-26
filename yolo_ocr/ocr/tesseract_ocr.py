"""Tesseract OCR backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import pytesseract

from yolo_ocr.config import OCRConfig
from yolo_ocr.ocr.base import OCR, OcrResult


@dataclass
class TesseractOCR(OCR):
    """Simple wrapper around pytesseract."""

    config: OCRConfig

    def load(self) -> None:  # pragma: no cover - nothing to do
        pass

    def recognize(self, crops: Sequence[np.ndarray]) -> List[OcrResult]:
        results: List[OcrResult] = []
        if not crops:
            return results
        for crop in crops:
            resized = cv2.resize(crop, (self.config.resize_width, self.config.resize_height), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh, lang=self.config.language)
            text = text.strip().upper()
            results.append(OcrResult(text=text, confidence=1.0))
        return results


__all__ = ["TesseractOCR"]
