"""OCR backends."""
from .base import OCR, OcrResult
from .tesseract_ocr import TesseractOCR
from .paddle_ocr import PaddleOCRBackend
from .trocr_hf import TrOcrHF

__all__ = ["OCR", "OcrResult", "TesseractOCR", "PaddleOCRBackend", "TrOcrHF"]
