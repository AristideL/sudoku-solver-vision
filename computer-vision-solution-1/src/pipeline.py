# src/pipeline.py
from __future__ import annotations

import numpy as np
import torch
import os
import cv2
from ultralytics import YOLO

from .config import AppConfig
from .yolo_utils import best_box, clamp_box
from .grid_mapping import build_occupancy_grid
from .ocr_easy import EasyOCRReader


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_detection_pipeline(cfg: AppConfig, bgr: np.ndarray, ocr: EasyOCRReader, debug_dir: str | None = None) -> str:
    """
    Pipeline: screenshot/image -> gameZone -> crop -> cells -> mapping 9x9 -> OCR digits.
    Returns: multiline string 9 lines, each "d d d d d d d d d" with 0 for empty.
    """

    h, w = bgr.shape[:2]

    # 0) Init models
    zone_model = YOLO(cfg.gamezone_weights)
    cells_model = YOLO(cfg.cells_weights)

    # 1) Detect game zone
    _sync_cuda()
    rz = zone_model.predict(
        bgr, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou, verbose=False
    )[0]
    _sync_cuda()

    bz = best_box(rz, target_class_name=cfg.gamezone_class_name)
    if bz is None:
        bz = best_box(rz, target_class_name=None)
        if bz is None:
            raise RuntimeError("No gameZone bbox detected.")
        # keep your behavior (print warning)
        print(
            f"[WARN] gameZone class '{cfg.gamezone_class_name}' not found. "
            f"Using best bbox of class '{bz[3]}'."
        )

    z_conf, z_xyxy, _, z_name = bz
    x1, y1, x2, y2 = clamp_box(*z_xyxy, w, h)

    # Debug saves
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "00_input.jpg"), bgr)
        ov = bgr.copy()
        cv2.rectangle(ov, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.imwrite(os.path.join(debug_dir, "01_gamezone_overlay.jpg"), ov)

    # 2) Crop
    crop = bgr[y1:y2, x1:x2].copy()
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "01_crop.jpg"), crop)

    print(
        f"GameZone: class={z_name}, conf={z_conf:.3f}, "
        f"box=({x1},{y1})-({x2},{y2}), crop={crop.shape[1]}x{crop.shape[0]}"
    )

    # 3) Detect cells on crop
    _sync_cuda()
    rc = cells_model.predict(
        crop,
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=cfg.iou,
        max_det=cfg.max_det,
        verbose=False,
    )[0]
    _sync_cuda()

    boxes = rc.boxes
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("No cells detected in crop.")

    xyxy = boxes.xyxy.cpu().numpy()
    cconf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    names = rc.names

    # 4) Map to 9x9
    mapping = build_occupancy_grid(
        xyxy=xyxy,
        conf=cconf,
        cls=cls,
        names=names,
        empty_class_name=cfg.empty_class_name,
    )

    digits = np.zeros((9, 9), dtype=int)

    # 5) OCR each cell
    for r in range(9):
        for c in range(9):
            if mapping.class_grid[r, c] == "" or mapping.class_grid[r, c] == cfg.empty_class_name:
                continue

            cx1, cy1, cx2, cy2 = mapping.box_grid[r, c]
            if cx1 < 0:
                continue

            cell_img = crop[cy1:cy2, cx1:cx2]
            digits[r, c] = ocr.read_digit(cell_img, debug_name=(f"r{r}_c{c}" if debug_dir else None), debug_dir=(os.path.join(debug_dir, "ocr") if debug_dir else None))

    # Save cells overlay if debug_dir
    if debug_dir:
        cells_dir = os.path.join(debug_dir, "cells")
        os.makedirs(cells_dir, exist_ok=True)

    # 6) Format result
    result = "\n".join(
        " ".join(str(d) if d != 0 else "0" for d in row) for row in digits
    )

    if debug_dir:
        # Render digits overlay
        digits_img = crop.copy()
        for r in range(9):
            for c in range(9):
                if digits[r, c] > 0:
                    x1c, y1c, x2c, y2c = mapping.box_grid[r, c]
                    if x1c >= 0:
                        cx = int(0.5*(x1c+x2c))
                        cy = int(0.5*(y1c+y2c))
                        cv2.putText(digits_img, str(int(digits[r,c])), (cx-6, cy+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        cv2.imwrite(os.path.join(debug_dir, "03_digits_overlay.jpg"), digits_img)

    return result
