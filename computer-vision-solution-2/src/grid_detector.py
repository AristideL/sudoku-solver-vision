from __future__ import annotations
import cv2
import numpy as np
import os


def _find_outer_border(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Detect outer grid border using contours.
    
    Returns (x, y, w, h) of the bounding box, or None if not found.
    """
    # Adaptive threshold to handle lighting variations
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Morphological close to connect nearby lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    h, w = gray.shape
    image_area = h * w
    threshold_area = 0.3 * image_area
    
    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        
        # Must cover at least 30% of image
        if area < threshold_area:
            continue
        
        # Must be roughly square (sudoku grid)
        aspect_ratio = float(cw) / ch if ch > 0 else 0
        if 0.7 <= aspect_ratio <= 1.3:
            candidates.append((x, y, cw, ch, area))
    
    if not candidates:
        return None
    
    # Return largest candidate by area
    candidates.sort(key=lambda t: t[4], reverse=True)
    x, y, cw, ch, _ = candidates[0]
    return (x, y, cw, ch)


def _refine_border_with_projections(gray: np.ndarray) -> tuple[int, int, int, int]:
    """
    Fallback border detection using morphological projections.
    Returns (x1, y1, x2, y2) - image coordinates of border.
    """
    h, w = gray.shape
    
    # Isolate horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 2, 1))
    h_morph = cv2.morphologyEx(gray, cv2.MORPH_OPEN, h_kernel)
    h_proj = np.sum(h_morph, axis=1)
    
    # Isolate vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 2))
    v_morph = cv2.morphologyEx(gray, cv2.MORPH_OPEN, v_kernel)
    v_proj = np.sum(v_morph, axis=0)
    
    # Find strong lines
    h_threshold = np.max(h_proj) * 0.2
    v_threshold = np.max(v_proj) * 0.2
    
    h_indices = np.where(h_proj > h_threshold)[0]
    v_indices = np.where(v_proj > v_threshold)[0]
    
    if len(h_indices) == 0 or len(v_indices) == 0:
        return (0, 0, w, h)
    
    y1 = int(h_indices[0])
    y2 = int(h_indices[-1])
    x1 = int(v_indices[0])
    x2 = int(v_indices[-1])
    
    return (x1, y1, x2, y2)


def recrop_to_grid_border(crop_bgr: np.ndarray, debug_dir: str | None = None) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Re-crop image to focus on the sudoku grid border.
    
    Returns (recropped_image, (x1, y1, x2, y2)) where coordinates are within crop_bgr.
    """
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "01_gray.jpg"), gray)
    
    # Try contour-based detection first
    border_box = _find_outer_border(gray)
    
    if border_box is not None:
        x, y, bw, bh = border_box
        x1, y1, x2, y2 = x, y, x + bw, y + bh
    else:
        # Fallback to projection-based
        x1, y1, x2, y2 = _refine_border_with_projections(gray)
    
    # Clamp to image bounds
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    
    # Sanity check: if result is too small (< 50% of original), use margin trim
    result_w = x2 - x1
    result_h = y2 - y1
    result_area = result_w * result_h
    original_area = w * h
    
    if result_area < 0.5 * original_area:
        # Use small margin trim
        margin = int(min(w, h) * 0.05)
        x1, y1, x2, y2 = margin, margin, w - margin, h - margin
    
    recropped = crop_bgr[y1:y2, x1:x2]
    
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "02_recropped.jpg"), recropped)
    
    return recropped, (x1, y1, x2, y2)


def detect_grid(crop_bgr: np.ndarray, debug_dir: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect the sudoku grid and divide into 9x9 cells using uniform division.
    
    Returns (recropped_image, cell_boxes) where cell_boxes is shape (9, 9, 4)
    with each cell containing [x1, y1, x2, y2] in recropped image coordinates.
    """
    recropped, _ = recrop_to_grid_border(crop_bgr, debug_dir=debug_dir)
    
    h, w = recropped.shape[:2]
    
    # Uniform 9x9 division using linspace
    x_coords = np.linspace(0, w, 10).astype(int)
    y_coords = np.linspace(0, h, 10).astype(int)
    
    cell_boxes = np.zeros((9, 9, 4), dtype=int)
    
    for r in range(9):
        for c in range(9):
            x1 = x_coords[c]
            x2 = x_coords[c + 1]
            y1 = y_coords[r]
            y2 = y_coords[r + 1]
            cell_boxes[r, c] = [x1, y1, x2, y2]
    
    if debug_dir:
        debug_img = recropped.copy()
        for r in range(9):
            for c in range(9):
                x1, y1, x2, y2 = cell_boxes[r, c]
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imwrite(os.path.join(debug_dir, "03_grid_overlay.jpg"), debug_img)
    
    return recropped, cell_boxes
