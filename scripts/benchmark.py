"""Benchmark baseline vs optimized inference."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from yolo_ocr.api import create_pipeline
from yolo_ocr.utils.io import VideoReader
from yolo_ocr.utils.timing import MovingAverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO + OCR pipeline")
    parser.add_argument("source", help="Video file or camera index")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML configuration")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames to process")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = create_pipeline(args.config)
    meter = MovingAverage(window=50)

    with VideoReader(args.source, stride=args.stride) as reader:
        for idx, frame in enumerate(reader.frames()):
            start = time.perf_counter()
            pipeline.process_image(frame.image)
            meter.update(time.perf_counter() - start)
            if (idx + 1) % 20 == 0:
                print(f"Processed {idx + 1} frames | avg latency {meter.value * 1000:.2f} ms | FPS {1.0 / meter.value if meter.value else 0:.2f}")
            if idx + 1 >= args.frames:
                break

    if meter.value:
        print(f"Final average latency: {meter.value * 1000:.2f} ms | FPS: {1.0 / meter.value:.2f}")


if __name__ == "__main__":
    main()
