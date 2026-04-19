from __future__ import annotations
import cv2
import numpy as np
import os
from ultralytics import YOLO

from .config import AppConfig
from .yolo_utils import clamp_box, best_box
from .grid_detector import detect_grid
from .template_matcher import TemplateMatcher


def run_detection_pipeline(cfg: AppConfig, bgr: np.ndarray, debug_dir: str | None = None) -> str:
    """
    Main detection pipeline.
    
    1. Load YOLO, detect gamezone, crop
    2. Detect grid borders and perform uniform division
    3. Read digits using template matching
    4. Return 9-line string of the grid
    """
    h, w = bgr.shape[:2]
    
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "00_input.jpg"), bgr)
    
    # Step 1: Load YOLO and detect gamezone
    model = YOLO(cfg.gamezone_weights)
    results = model.predict(bgr, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou, verbose=False)
    result = results[0]
    
    best = best_box(result, target_class_name=cfg.gamezone_class_name)
    if best is None:
        raise ValueError("No game zone detected")
    
    conf, xyxy, cls_id, cls_name = best
    x1, y1, x2, y2 = xyxy
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
    
    crop_bgr = bgr[int(y1):int(y2), int(x1):int(x2)]
    
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "01_gamezone_crop.jpg"), crop_bgr)
    
    # Step 2: Detect grid and get cell boxes
    recropped_bgr, cell_boxes = detect_grid(crop_bgr, debug_dir=debug_dir)
    
    # Step 3: Create template matcher and read digits
    matcher = TemplateMatcher(
        templates_dir="numbers",
        cell_padding_ratio=cfg.cell_padding_ratio,
        match_threshold=cfg.match_threshold
    )
    
    grid = []
    for r in range(9):
        row = []
        for c in range(9):
            x1, y1, x2, y2 = cell_boxes[r, c]
            cell_bgr = recropped_bgr[y1:y2, x1:x2]
            
            digit = matcher.read_digit(cell_bgr, debug_name=f"cell_{r}_{c}", debug_dir=debug_dir)
            row.append(str(digit))
        
        grid.append(" ".join(row))
    
    result_str = "\n".join(grid)
    
    if debug_dir:
        with open(os.path.join(debug_dir, "grid_output.txt"), "w") as f:
            f.write(result_str)
    
    return result_str
