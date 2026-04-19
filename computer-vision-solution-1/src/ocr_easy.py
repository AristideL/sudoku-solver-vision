from __future__ import annotations

import os

import cv2
import numpy as np
import easyocr

class EasyOCRReader:
    """OCR digits 1..9 via EasyOCR.

    Notes:
    - Allowlist to avoid unwanted characters.
    - Preprocessing to stabilize on sudoku.com.
    """

    def __init__(self, gpu: bool = True, min_conf: float = 0.4):
        self.reader = easyocr.Reader(["en"], gpu=gpu)
        self.min_conf = float(min_conf)

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """Preprocess Sudoku cell for OCR.
        Goal: remove background (blue/white) + reduce grid + isolate digit.
        """
        h, w = bgr.shape[:2]

        # Crop borders (grid perturbs a lot)
        pad = int(min(h, w) * 0.18)  # 18% works better than 10% on sudoku.com
        if pad > 0:
            bgr = bgr[pad:h - pad, pad:w - pad]

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # OTSU
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Normalize: black background, white digit (more stable)
        if th.mean() > 127:
            th = 255 - th

        return th

    def read_digit(self, cell_bgr: np.ndarray, debug_name: str | None = None, debug_dir: str | None = None) -> int:
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

        th = self.preprocess(cell_bgr)  # binary

        rgb = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
        rgb_inv = cv2.cvtColor(255 - th, cv2.COLOR_GRAY2RGB)

        if debug_name and debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"{debug_name}_th.png"), th)
            cv2.imwrite(os.path.join(debug_dir, f"{debug_name}_th_inv.png"), 255 - th)

        def run_ocr(img_rgb):
            return self.reader.readtext(
                img_rgb,
                allowlist="123456789",
                detail=1,
                paragraph=False,
                decoder="greedy",
                text_threshold=0.4,
                low_text=0.3,
                link_threshold=0.3,
                mag_ratio=1.5,        # Enlarges digit (important)
                canvas_size=256,      # Helps the detector
            )

        res1 = run_ocr(rgb)
        res = res1 if res1 else []

        if not res:
            if debug_name:
                print(f"OCR: no text detected for cell '{debug_name}'")
            return 0

        best = max(res, key=lambda x: float(x[2]))
        text = str(best[1]).strip()
        conf = float(best[2])

        if conf < self.min_conf:
            return 0

        # Common corrections
        text = text.replace("I", "1").replace("l", "1").replace("|", "1").replace("g", "9")

        if not text.isdigit():
            return 0

        v = int(text)
        out = v if 1 <= v <= 9 else 0

        return out

