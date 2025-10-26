"""GPT-based OCR backend (optional)."""
from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np

from yolo_ocr.config import OCRConfig
from yolo_ocr.ocr.base import OCR, OcrResult


LOGGER = logging.getLogger(__name__)


def _encode_image(image: np.ndarray) -> str:
    """Encode an image as a base64 PNG payload for GPT APIs."""

    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode crop for GPT OCR request.")
    return base64.b64encode(buffer).decode("utf-8")


@dataclass
class GptOCR(OCR):
    """Experimental GPT-based OCR backend.

    The backend requires the ``openai`` package and a valid ``OPENAI_API_KEY``.
    If either dependency is missing, callers should catch the raised
    ``RuntimeError`` and fall back to a different OCR engine.
    """

    config: OCRConfig

    def __post_init__(self) -> None:
        self._client = None
        self._model = os.getenv("GPT_OCR_MODEL", "gpt-4o-mini")

    def load(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is required for GPT OCR") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        self._client = OpenAI(api_key=api_key)

    def recognize(self, crops: Sequence[np.ndarray]) -> List[OcrResult]:
        if self._client is None:
            raise RuntimeError("GPT OCR backend has not been loaded.")

        results: List[OcrResult] = []
        for crop in crops:
            if crop.size == 0:
                results.append(OcrResult(text="", confidence=0.0))
                continue

            payload = _encode_image(crop)
            try:
                response = self._client.responses.create(  # type: ignore[attr-defined]
                    model=self._model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        "Extract the license plate text from this image. "
                                        "Return only the plate characters without additional commentary."
                                    ),
                                },
                                {
                                    "type": "input_image",
                                    "image_base64": payload,
                                },
                            ],
                        }
                    ],
                )
            except Exception as exc:  # pragma: no cover - network interaction
                LOGGER.warning("GPT OCR request failed: %s", exc)
                results.append(OcrResult(text="", confidence=0.0))
                continue

            text = ""
            try:
                text = response.output_text.strip()
            except Exception:  # pragma: no cover - SDK variance
                LOGGER.debug("Unexpected GPT OCR response format: %s", response)

            results.append(OcrResult(text=text.upper(), confidence=1.0 if text else 0.0))
        return results


__all__ = ["GptOCR"]

