"""OCR backends."""
from .base import OCR, OcrResult
from .tesseract_ocr import TesseractOCR
from .paddle_ocr import PaddleOCRBackend
from .trocr_hf import TrOcrHF
from .gpt_ocr import GptOCR

__all__ = ["OCR", "OcrResult", "TesseractOCR", "PaddleOCRBackend", "TrOcrHF", "GptOCR"]
