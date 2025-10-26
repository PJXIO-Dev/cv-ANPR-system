"""Export an Ultralytics YOLO model to ONNX for acceleration."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument("model", help="Path to Ultralytics model weights")
    parser.add_argument("--img-size", type=int, nargs=2, default=(640, 640), help="Input size (width height)")
    parser.add_argument("--out", type=Path, default=Path("yolo.onnx"), help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch axes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.export(
        format="onnx",
        imgsz=list(args.img_size),
        opset=args.opset,
        dynamic=args.dynamic,
        half=False,
        simplify=True,
        optimize=True,
        device="cpu",
        verbose=True,
        out=str(args.out),
    )
    print(f"Exported ONNX model to {args.out}")


if __name__ == "__main__":
    main()
