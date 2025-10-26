# YOLO + OCR License Plate Pipeline

A modular, production-ready pipeline that detects vehicles/license plates with YOLO and reads them using OCR backends. The project refactors the original monolithic prototype into reusable modules, adds optional acceleration paths (FP16, batching, ONNX/TensorRT hooks), and provides tooling for benchmarking and deployment.

## Features

- Ultralytics YOLO detector with ROI cropping, batching, and FP16 inference
- Plug-and-play OCR backends (Tesseract by default, TrOCR/Paddle/OpenAI hooks ready)
- Unified post-processing with regex validation and deduplication
- Optional visualization utilities for debugging and demos
- Benchmark/export scripts for measuring throughput and converting to ONNX

## Installation

1. Install Python 3.10 or later.
2. Clone the repository and install dependencies:

   ```bash
   pip install -r requirements.txt
   # or
   pip install -e .
   ```

3. Download a YOLO checkpoint (e.g. `yolov8n.pt`) into `models/` (create the folder if it does not exist). The default config expects `models/yolov8n.pt`, so adjust `detector.model_path` if you choose a different location or filename.
4. (Optional) Install OCR extras:
   - `pytesseract` requires a local Tesseract binary (see <https://tesseract-ocr.github.io/tessdoc/Installation.html>)
   - `pip install paddleocr` for PaddleOCR
   - `pip install transformers accelerate sentencepiece` for TrOCR

## Project Structure

```
.
├─ pyproject.toml / requirements.txt
├─ configs/default.yaml
├─ scripts/
│  ├─ benchmark.py
│  └─ export_onnx.py
└─ yolo_ocr/
   ├─ api.py
   ├─ config.py
   ├─ utils/
   ├─ detectors/
   ├─ ocr/
   └─ pipeline/
```

Each module focuses on a single responsibility and can be swapped or extended independently.

### Default data directories

- `models/` – store YOLO checkpoints or exported ONNX/TensorRT engines here; update `configs/default.yaml` if you relocate them.
- `input/` – drop ad-hoc test images or videos here before invoking the CLI (e.g., `python3 -m yolo_ocr.cli image input/car.jpg`).
- `output/` – the CLI examples below save annotated artifacts to this folder so results stay organized and tracked by git.

## Main Pipeline Usage (`yolo_ocr`)

The refactored package exposes a first-class CLI and Python API so you can run the modular pipeline directly—no legacy scripts required.

### CLI quick start (installed as a package)

```bash
yolo-ocr --config configs/default.yaml image samples/car.jpg --output artifacts/car_annotated.jpg
```

### Running the CLI module directly

```bash
python3 -m yolo_ocr.cli --config configs/default.yaml image samples/car.jpg --output output/car_annotated.jpg
```

### Image mode

```bash
python3 -m yolo_ocr.cli --config configs/default.yaml image input/frame.jpg --output output/frame_annotated.jpg
```

- Prints plate predictions (text + confidence) to stdout.
- Writes an annotated copy when `--output` is provided.

### Video/camera mode

```bash
python3 -m yolo_ocr.cli --config configs/default.yaml video input/traffic.mp4 --output output/traffic.mp4 --stride 2
```

- Accepts file paths or camera indices (e.g. `video 0`).
- Prints per-frame predictions.
- Saves an annotated video if `--output` is specified (defaults to `mp4v`, override FPS with `--fps`).
- Skips frames using `--stride` for faster throughput without altering detection/recognition settings.

Use this entry point for production integrations, CLI demos, or rapid experiments. The CLI internally instantiates the same `create_pipeline` factory used in the Python API so all configuration and optimization features remain available.

### Python API example

```python
import cv2
from yolo_ocr.api import create_pipeline

pipeline = create_pipeline("configs/default.yaml")
frame = cv2.imread("samples/car.jpg")
result = pipeline.process_image(frame)

for plate in result.predictions:
    print(plate.text, plate.confidence)

if result.annotated is not None:
    cv2.imwrite("annotated.jpg", result.annotated)
```

For video streams:

```python
from yolo_ocr.api import run_on_video

for output in run_on_video("video.mp4", "configs/default.yaml", stride=2):
    ...  # consume PipelineResult objects
```

## Support Utilities (`scripts/`)

Two optional helper scripts live in the `scripts/` directory to support deployment and optimization workflows. They are not required for day-to-day inference but make it easier to profile and accelerate the detector when needed.

### `scripts/benchmark.py`

Profile end-to-end latency or frames-per-second for a particular video source and configuration. It loads the pipeline described in your YAML config, runs it over the specified number of frames, and prints moving-average timing statistics so you can compare baseline versus accelerated settings.

```bash
python scripts/benchmark.py <video_or_camera> --config configs/default.yaml --frames 500 --stride 2
```

- `<video_or_camera>` accepts a path to a file (e.g., `data/traffic.mp4`) or a camera index like `0` for a webcam.
- `--frames` controls how many frames to profile (default `200`).
- `--stride` lets you skip frames (e.g., `--stride 2` processes every other frame).
- Adjust the YAML file to toggle batching, FP16, or alternate backends and rerun the benchmark to observe the impact.

### `scripts/export_onnx.py`

Run this script when you want to deploy the YOLO detector with ONNX Runtime or TensorRT. It wraps Ultralytics' built-in export and exposes the most common knobs via CLI flags.

```bash
python scripts/export_onnx.py yolov8n.pt --img-size 640 640 --out models/yolov8n.onnx --opset 12 --dynamic
```

- `--img-size` sets the input resolution (width height) you plan to serve.
- `--out` specifies where the exported ONNX file will be written.
- `--opset` controls the ONNX opset version (use a value supported by your runtime).
- Pass `--dynamic` when you need dynamic batch dimensions for batched inference.

After exporting, point `detector.backend` to `yolo_onnx` and set `detector.model_path` to the generated `.onnx` file in your YAML configuration.

## Configuration Reference (`configs/default.yaml`)

| Key | Description |
| --- | --- |
| `detector.backend` | Detection backend (`yolo_ultralytics`, `yolo_onnx`, `yolo_tensorrt`). |
| `detector.model_path` | Path to YOLO weights (`.pt` or exported engine). Defaults to `models/yolov8n.pt`; update if you store the model elsewhere. |
| `detector.conf_threshold` / `iou_threshold` | Confidence and NMS thresholds. |
| `detector.device` | `auto`, `cpu`, `cuda:0`, etc. The pipeline automatically falls back to CPU when CUDA isn't available. |
| `detector.fp16` | Enable half precision for compatible GPUs (automatically disabled on CPU fallback). |
| `detector.batch_size` | Batch size for inference. |
| `detector.warmup_iterations` | Number of warm-up passes to stabilize latency. |
| `detector.roi_from_center` / `roi_start_fraction` | Process only lower portion of the frame to save compute. |
| `detector.resize_target_width` | Downscale wide frames before detection. |
| `ocr.backend` | OCR backend (`tesseract`, `paddleocr`, `trocr`). |
| `ocr.language` | Language code for OCR model. |
| `ocr.padding` | Extra pixels around detections before OCR. |
| `ocr.resize_width` / `resize_height` | Target crop size passed to OCR. |
| `postprocess.dedup_iou_threshold` | IOU threshold for deduping overlapping boxes. |
| `postprocess.min_confidence` | Minimum detector confidence. |
| `postprocess.keep_top_k` | Optional cap on predictions per frame. |
| `postprocess.plate_regex` | Optional regex to validate license plates. Leave blank to accept all sanitized OCR text. |
| `visualize` | Save annotated frames when true. |

Override any setting programmatically by calling `create_pipeline(..., overrides={...})`.

## Optimization Tips

- **Batching:** Increase `detector.batch_size` when processing many frames at once.
- **FP16:** Keep `detector.fp16: true` on GPUs supporting half precision for ~1.5× faster inference.
- **ONNX Runtime / TensorRT:** Use `scripts/export_onnx.py` to export a model, then implement the ONNX/TensorRT stubs in `yolo_ocr/detectors/` for even lower latency.
- **ROI Cropping:** `roi_from_center` processes the lower half of the frame first, focusing on road areas and reducing background detections.
- **Warm-up:** Increase `warmup_iterations` for stable latency measurements on GPUs.
- **Threaded Capture:** Integrate the `VideoReader` helper with your own threaded capture if ingest is a bottleneck.

## Benchmarking

Measure throughput on a video file or RTSP stream:

```bash
python scripts/benchmark.py data/traffic.mp4 --frames 500 --stride 2
```

The script reports moving-average latency and FPS so you can compare baseline vs optimized settings.

## Troubleshooting

- **CUDA not found:** Ensure the right CUDA toolkit is installed and that `torch.cuda.is_available()` returns `True`.
- **Tesseract missing:** Install the Tesseract binary and verify `pytesseract.pytesseract.tesseract_cmd` points to it.
- **Empty OCR text:** Increase `ocr.padding`, switch to `trocr`, or supply a permissive `postprocess.plate_regex` only when you need additional filtering.
- **ONNX/TensorRT errors:** Confirm opset compatibility and implement the backend-specific post-processing logic.

## Credits & License

- Detection powered by [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OCR backends from [Tesseract OCR](https://github.com/tesseract-ocr/tesseract), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), and [Hugging Face](https://huggingface.co/)
- Licensed under the MIT License. Refer to `LICENSE` if provided or include one that suits your deployment.
