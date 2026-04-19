# Solution 1 — YOLO + YOLO + EasyOCR

Sudoku solver by computer vision. Takes a screenshot of [sudoku.com](https://sudoku.com/), detects the grid, reads digits, solves, and auto-fills the answer.

## Pipeline

```
Screenshot → YOLO gamezone detection → Crop
           → YOLO cells detection (81 cells) → K-means 9×9 mapping
           → EasyOCR digit recognition → Backtracking solver
           → Keyboard auto-fill on sudoku.com
```

| Step | Method | Details |
|------|--------|---------|
| Game zone detection | YOLOv8 (fine-tuned) | Detects the grid area in full screenshot |
| Cell detection | YOLOv8 (fine-tuned) | Detects all 81 individual cells (filled + empty) |
| Cell mapping | 1D K-means | Maps detected bboxes to 9×9 grid using center clustering |
| Digit recognition | EasyOCR | Reads digits 1-9 with preprocessing (crop, threshold, equalize) |
| Solving | Backtracking | Recursive constraint-based solver |
| Interaction | pyautogui | Zig-zag keyboard navigation to fill cells |

## Project structure

```
computer-vision-solution-1/
├── models/
│   ├── gamezone/best.pt      # YOLO weights — game zone detection
│   └── cells/best.pt         # YOLO weights — cell detection
├── src/
│   ├── main.py               # Entry point (argparse)
│   ├── gui_app.py            # Tkinter GUI + screenshot + auto-fill
│   ├── pipeline.py           # Detection pipeline (YOLO → YOLO → OCR)
│   ├── config.py             # All config in one dataclass
│   ├── yolo_utils.py         # best_box, clamp_box helpers
│   ├── grid_mapping.py       # K-means 9×9 mapping from cell detections
│   ├── ocr_easy.py           # EasyOCR wrapper with preprocessing
│   ├── resolver.py           # Sudoku backtracking solver
│   └── image_io.py           # Image loading utility
├── Dockerfile
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- CUDA GPU recommended (EasyOCR + YOLO)
- Model weights in `models/gamezone/best.pt` and `models/cells/best.pt`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run the GUI
python -m src.main

# With debug output (saves intermediate images)
python -m src.main --debug_dir debug/
```

**Steps:**
1. Open [sudoku.com](https://sudoku.com/) in your browser
2. Launch the app
3. Click **Détecter** — takes a screenshot and detects the grid
4. Verify the detected grid is correct
5. Click **Résoudre** — solves and auto-fills
6. You have ~2s to click the first cell on sudoku.com before typing starts

## Docker

The Dockerfile containerizes the application only (not training).

```bash
# Build
docker build -t sudoku-solver-v1 .

# Run (requires display forwarding for GUI)
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  sudoku-solver-v1
```

> Note: GUI + pyautogui require X11 display access. On Windows, use VcXsrv or run natively instead.

Training has its own Docker — see `../training/README.md`.

## Config

All parameters in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `imgsz` | 640 | YOLO input resolution |
| `conf` | 0.75 | YOLO confidence threshold |
| `iou` | 0.55 | YOLO NMS IoU threshold |
| `max_det` | 300 | Max detections per image |
| `gamezone_class_name` | `"gameZone"` | Class name for grid detection |
| `empty_class_name` | `"emptycells"` | Class name for empty cells |

## Training

Model training is handled separately in the `../training/` folder with its own Dockerfile. See `../training/README.md` for full instructions.
