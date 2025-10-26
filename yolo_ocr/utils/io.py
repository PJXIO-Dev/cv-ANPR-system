"""I/O helpers for images and videos."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional

import cv2
import numpy as np


@dataclass
class VideoFrame:
    """Represents a video frame along with metadata."""

    image: np.ndarray
    index: int
    timestamp: float


class VideoReader:
    """Iterate over frames from a video file or camera stream."""

    def __init__(self, source: str | int, *, stride: int = 1, warmup: int = 0) -> None:
        self.source = source
        self.stride = max(1, stride)
        self.warmup = max(0, warmup)
        self._cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 0.0
        self.frame_size: tuple[int, int] | None = None

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                raise RuntimeError(f"Unable to open video source: {self.source}")
            self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self.frame_size = (width, height) if width > 0 and height > 0 else None
            for _ in range(self.warmup):
                self._cap.read()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self.fps = 0.0
            self.frame_size = None

    def frames(self) -> Generator[VideoFrame, None, None]:
        if self._cap is None:
            self.open()
        assert self._cap is not None
        index = 0
        fps = self.fps
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if index % self.stride == 0:
                ts = index / fps if fps > 0 else 0.0
                yield VideoFrame(image=frame, index=index, timestamp=ts)
            index += 1


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as a numpy array in BGR order."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Write an image to disk."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), image):
        raise IOError(f"Failed to save image to {out_path}")


def batchify(items: Iterable[np.ndarray], batch_size: int) -> Generator[list[np.ndarray], None, None]:
    """Yield fixed size batches from an iterable of arrays."""

    batch: list[np.ndarray] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


__all__ = ["VideoReader", "VideoFrame", "load_image", "save_image", "batchify"]
