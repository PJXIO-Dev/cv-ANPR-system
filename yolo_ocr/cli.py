"""Command line interface for the YOLO + OCR pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import cv2

from yolo_ocr.api import create_pipeline
from yolo_ocr.pipeline.pipeline import PipelineResult
from yolo_ocr.utils.io import VideoReader, load_image, save_image


def _print_predictions(results: Iterable[tuple[int | str, PipelineResult]]) -> None:
    for tag, result in results:
        if not result.predictions:
            print(f"[{tag}] No plates detected")
            continue
        for plate in result.predictions:
            conf_pct = plate.confidence * 100.0
            print(f"[{tag}] {plate.text} ({conf_pct:.1f}% conf)")


def _run_image(args: argparse.Namespace) -> None:
    pipeline = create_pipeline(args.config)
    image = load_image(args.source)
    result = pipeline.process_image(image)
    _print_predictions([(args.source, result)])
    if args.output:
        annotated = result.annotated if result.annotated is not None else image
        save_image(args.output, annotated)
        print(f"Annotated image written to {args.output}")


def _run_video(args: argparse.Namespace) -> None:
    pipeline = create_pipeline(args.config)
    try:
        source: str | int
        source = int(args.source)
    except (ValueError, TypeError):
        source = args.source
    reader = VideoReader(source, stride=args.stride)
    writer: cv2.VideoWriter | None = None
    try:
        with reader:
            for frame in reader.frames():
                result = pipeline.process_image(frame.image)
                _print_predictions([(frame.index, result)])
                if args.output:
                    annotated = result.annotated if result.annotated is not None else frame.image
                    if writer is None:
                        fps = reader.fps if reader.fps > 0 else args.fps
                        if fps <= 0:
                            fps = 30.0
                        height, width = annotated.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
                    writer.write(annotated)
    finally:
        if writer is not None:
            writer.release()
        if args.output and writer is not None:
            print(f"Annotated video written to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the YOLO + OCR pipeline from the command line.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Path to the pipeline YAML config.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("image", help="Run inference on a single image file.")
    image_parser.add_argument("source", type=Path, help="Path to the input image.")
    image_parser.add_argument("--output", type=Path, help="Optional path to save an annotated copy of the image.")
    image_parser.set_defaults(func=_run_image)

    video_parser = subparsers.add_parser("video", help="Run inference on a video file or camera index.")
    video_parser.add_argument("source", help="Video path or camera index (e.g. 0).")
    video_parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame (default: 1).")
    video_parser.add_argument("--output", type=Path, help="Optional path to save an annotated video.")
    video_parser.add_argument("--fps", type=float, default=0.0, help="Override FPS when writing output video.")
    video_parser.set_defaults(func=_run_video)

    return parser


def main(argv: list[str] | None = None) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
