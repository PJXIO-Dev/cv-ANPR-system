"""Timing helpers used for benchmarking."""
from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Deque, Generator


@dataclass
class MovingAverage:
    """Compute a moving average for the last ``window`` measurements."""

    window: int = 50

    def __post_init__(self) -> None:
        self._values: Deque[float] = deque(maxlen=self.window)

    def update(self, value: float) -> float:
        self._values.append(value)
        return self.value

    @property
    def value(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)


@contextmanager
def time_block(meter: MovingAverage | None = None) -> Generator[None, None, None]:
    """Context manager that measures the elapsed time for a code block."""

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if meter is not None:
            meter.update(elapsed)


__all__ = ["MovingAverage", "time_block"]
