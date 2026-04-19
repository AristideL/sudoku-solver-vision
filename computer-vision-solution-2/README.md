# Solution 2 — YOLO + Classical CV + Template Matching

Sudoku solver by computer vision. Takes a screenshot of [sudoku.com](https://sudoku.com/), detects the grid, reads digits, solves, and auto-fills the answer.

## Pipeline

```
Screenshot → YOLO gamezone detection → Crop
           → Border re-crop (contour/projection) → Uniform 9×9 division
           → Template matching digit recognition → Backtracking solver
           → Keyboard auto-fill on sudoku.com
```

| Step | Method | Details |
|------|--------|---------|
| Game zone detection | YOLOv8 (fine-tuned) | Detects the grid area in full screenshot |
| Border re-crop | Classical CV | Contour detection + projection fallback to snap crop to exact grid border |
| Cell division | Uniform split | Divides re-cropped grid into 81 equal cells |
| Digit recognition | Template matching | OpenCV `TM_CCOEFF_NORMED` against reference templates |
| Solving | Backtracking | Recursive constraint-based solver |
| Interaction | pyautogui | Zig-zag keyboard navigation to fill cells |

### Key difference from Solution 1

Solution 1 uses **two YOLO models** (gamezone + cells detection) and **EasyOCR** for digit reading.

Solution 2 replaces the second YOLO model with **classical CV** (contour detection, morphological operations, projection analysis) for grid alignment, and replaces EasyOCR with **template matching** for digit recognition. Only one YOLO model is needed.

## Project structure

```
computer-vision-solution-2/
├── models/
│   └── gamezone/best.pt      # YOLO weights — game zone detection
├── numbers/
│   └── 1.jpg..9.jpg          # Digit templates for matching
├── src/
│   ├── main.py               # Entry point (argparse)
│   ├── gui_app.py            # Tkinter GUI + screenshot + auto-fill
│   ├── pipeline.py           # Detection pipeline (YOLO → CV → Template)
│   ├── config.py             # All config in one dataclass
│   ├── yolo_utils.py         # best_box, clamp_box helpers
│   ├── grid_detector.py      # Border re-crop + uniform 9×9 division
│   ├── template_matcher.py   # Template matching digit recognition
│   ├── resolver.py           # Sudoku backtracking solver
│   └── image_io.py           # Image loading utility
├── Dockerfile
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- CUDA GPU recommended (YOLO)
- Model weights in `models/gamezone/best.pt`
- Digit templates in `numbers/` (optional — generated programmatically if missing)

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
3. Click **Detect** — takes a screenshot and detects the grid
4. Verify the detected grid is correct
5. Click **Solve** — solves and auto-fills
6. You have ~2s to click the first cell on sudoku.com before typing starts

## Docker

The Dockerfile containerizes the application only (not training).

```bash
# Build
docker build -t sudoku-solver-v2 .

# Run (requires display forwarding for GUI)
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  sudoku-solver-v2
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
| `gamezone_class_name` | `"gameZone"` | Class name for grid detection |
| `match_threshold` | 0.35 | Minimum template match score (0-1) |
| `cell_padding_ratio` | 0.18 | Cell border crop ratio before matching |

## Training

Model training is handled separately in the `../training/` folder with its own Dockerfile. See `../training/README.md` for full instructions.
