from __future__ import annotations
import cv2
import numpy as np
import os
from pathlib import Path


class TemplateMatcher:
    """Template-based digit recognition for sudoku cells."""
    
    def __init__(self, templates_dir: str = "numbers", cell_padding_ratio: float = 0.18, 
                 match_threshold: float = 0.35):
        """
        Initialize template matcher.
        
        Args:
            templates_dir: Directory containing template images (1.jpg..9.jpg)
            cell_padding_ratio: Ratio of cell to crop from borders
            match_threshold: Minimum match score (0-1) to recognize a digit
        """
        self.templates_dir = templates_dir
        self.cell_padding_ratio = cell_padding_ratio
        self.match_threshold = match_threshold
        self.templates = {}
        
        # Try to load templates from directory
        self._load_templates()
        
        # If no templates loaded, generate them programmatically
        if not self.templates:
            self._generate_templates()
    
    def _load_templates(self):
        """Load template images from directory."""
        if not os.path.exists(self.templates_dir):
            return
        
        for digit in range(1, 10):
            template_path = os.path.join(self.templates_dir, f"{digit}.jpg")
            if os.path.exists(template_path):
                img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    processed = self._preprocess_template(img)
                    self.templates[digit] = [processed]
    
    def _preprocess_template(self, template_gray: np.ndarray) -> np.ndarray:
        """Preprocess a template image."""
        # Apply OTSU threshold
        _, binary = cv2.threshold(template_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Normalize: ensure white-on-black
        if np.sum(binary > 128) < np.sum(binary < 128):
            binary = cv2.bitwise_not(binary)
        
        # Resize to standard size
        template = cv2.resize(binary, (48, 48))
        
        # Normalize intensities
        template = template.astype(np.float32) / 255.0
        
        return template
    
    def _generate_templates(self):
        """Generate template images programmatically."""
        fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX
        ]
        scales = [1.0, 1.3, 1.6, 1.9]
        
        for digit in range(1, 10):
            digit_str = str(digit)
            self.templates[digit] = []
            
            for font in fonts:
                for scale in scales:
                    # Create image
                    img = np.ones((64, 64), dtype=np.uint8) * 255
                    
                    # Get text size
                    size, baseline = cv2.getTextSize(digit_str, font, scale, 2)
                    
                    # Center text
                    x = (64 - size[0]) // 2
                    y = (64 + size[1]) // 2
                    
                    # Draw text
                    cv2.putText(img, digit_str, (x, y), font, scale, 0, 2)
                    
                    # Preprocess
                    processed = self._preprocess_template(img)
                    self.templates[digit].append(processed)
    
    def _preprocess_cell(self, cell_bgr: np.ndarray) -> np.ndarray:
        """Preprocess a cell image for matching."""
        h, w = cell_bgr.shape[:2]
        
        # Crop borders by padding ratio
        pad = int(min(h, w) * self.cell_padding_ratio)
        cropped = cell_bgr[pad:h-pad, pad:w-pad]
        
        # Grayscale
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        # Histogram equalization
        gray = cv2.equalizeHist(gray)
        
        # Gaussian blur
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # OTSU threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Normalize: white-on-black
        if np.sum(binary > 128) < np.sum(binary < 128):
            binary = cv2.bitwise_not(binary)
        
        # Resize to standard size
        processed = cv2.resize(binary, (48, 48))
        
        # Normalize intensities
        processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def _is_empty_cell(self, processed: np.ndarray) -> bool:
        """Check if cell is empty based on white pixel ratio."""
        white_ratio = np.sum(processed > 0.5) / processed.size
        
        # Empty if mostly white (< 3%) or mostly black (> 85%)
        return white_ratio < 0.03 or white_ratio > 0.85
    
    def read_digit(self, cell_bgr: np.ndarray, debug_name: str | None = None, 
                   debug_dir: str | None = None) -> int:
        """
        Read digit from cell image.
        
        Returns digit (1-9) if recognized, else 0 (empty).
        """
        processed = self._preprocess_cell(cell_bgr)
        
        # Check if empty
        if self._is_empty_cell(processed):
            return 0
        
        # Match against all templates
        best_digit = 0
        best_score = self.match_threshold
        
        for digit in range(1, 10):
            for template in self.templates.get(digit, []):
                # Template matching using normalized cross-correlation
                result = cv2.matchTemplate(processed, template, cv2.TM_CCOEFF_NORMED)
                score = float(result.max())
                
                if score > best_score:
                    best_score = score
                    best_digit = digit
        
        return best_digit
