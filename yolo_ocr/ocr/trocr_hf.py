"""Transformer OCR backend using Hugging Face TrOCR."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from yolo_ocr.config import OCRConfig
from yolo_ocr.ocr.base import OCR, OcrResult


@dataclass
class TrOcrHF(OCR):
    """Lazy-loading wrapper around the Hugging Face TrOCR model."""

    config: OCRConfig

    def __post_init__(self) -> None:
        self._processor = None
        self._model = None
        self._device = "cpu"

    def load(self) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch

        self._processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        self._model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def recognize(self, crops: Sequence[np.ndarray]) -> List[OcrResult]:
        if self._processor is None or self._model is None:
            raise RuntimeError("Call load() before running OCR.")
        if not crops:
            return []

        import torch

        images = [self._processor(images=crop, return_tensors="pt").pixel_values for crop in crops]
        pixel_values = torch.cat(images, dim=0).to(self._device)
        with torch.inference_mode():
            generated_ids = self._model.generate(pixel_values)
        texts = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [OcrResult(text=text.strip().upper(), confidence=1.0) for text in texts]


__all__ = ["TrOcrHF"]
