# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Architectural floor plan inventory tool that converts hand-drawn floor plans (photos/scans) into DXF (CAD) files. University project (Jagiellonian University, Python course).

Pipeline: **image → YOLOv8 detection (local Ultralytics) → NMS/merge → OCR (custom ResNet18) → wall graph construction → DXF export**

## Commands

```bash
# Setup
pip install -r requirements.txt

# Train YOLO model on local dataset (requires yolo_dataset/ exported from Roboflow)
python train_yolo.py
cp runs/detect/yolo_floorplan/weights/best.pt yolo_floorplan.pt

# Run main pipeline (processes test.jpg → rzut.dxf)
python main.py

# Prepare training dataset from sketch cards in sketches/
python dataset_creator.py

# Train the digit recognition model (outputs digit_ocr_resnet18.pth)
python trainer.py
```

Python 3.11, venv in `venv/`. No test suite exists.

## Architecture

### Two versions of the code coexist

**`main.py`** is the monolithic entry point that contains the full pipeline inline — it does NOT import from the extracted modules. It handles detection, OCR, wall graph building, and DXF export all in one file (~627 lines).

**Extracted modules** (`detection.py`, `model.py`, `ocr.py`, `dxf_export.py`) contain the same logic refactored into functions, but `main.py` has not been updated to use them. These modules are the intended clean architecture:

- **`model.py`** — `get_trained_model()`: loads the ResNet18 model from `digit_ocr_resnet18.pth`
- **`detection.py`** — `nms_dimension_values()`, `merge_adjacent_dimension_values()`: post-processing of YOLO detections
- **`train_yolo.py`** — trains YOLOv8n on local dataset, outputs `runs/detect/yolo_floorplan/weights/best.pt`
- **`ocr.py`** — `segment_digits()`, `_pad_and_resize()`: digit segmentation via column projection analysis
- **`dxf_export.py`** — `export_to_dxf(final_results, image_shape, output_path)`: wall graph → DXF conversion

### Standalone training tools

- **`dataset_creator.py`** — `create_dataset_from_sketches()`: cuts sketch cards (red-channel extraction, Otsu binarization) into 64x64 digit samples in `my_dataset/`
- **`trainer.py`** — `train_professional_model()`: two-phase training (Phase 1: MNIST+user data pretrain, Phase 2: user-data-only fine-tune) of modified ResNet18 (1-channel input, Dropout(0.5), 10 classes)

### Key design decisions

- Dimensions are matched directly to walls (skipping dimension_line intermediary) — this produced better results than the three-way correlation approach
- OCR extracts the red channel specifically (dimensions are drawn in red on the sketches)
- Digit confidence threshold is 70% — below that, digits are flagged as uncertain
- Wall connectivity uses a snap distance (SNAP_DIST) of 15% of image size or 200px minimum
- Pixel-to-real-unit scale is estimated from known dimension-gap pairs, then applied to fill unknowns

## Environment

- No API keys required — YOLO detection runs locally via Ultralytics
- YOLO model file: `yolo_floorplan.pt` (trained locally from `yolo_dataset/`, gitignored)
- Dataset originally annotated in Roboflow (workspace `python-hegfw`, project `my-first-project-gpcqz`, version 5), exported in YOLOv8 format to `yolo_dataset/`
- Trained OCR model file: `digit_ocr_resnet18.pth` (~43MB, gitignored)

## Language

Project comments, print statements, and README are in Polish.
