import cv2
import numpy as np

def load_bgr(path: str) -> np.ndarray:
    """Charge une image depuis disque en format BGR (OpenCV).
    Particularité: lève une erreur claire si l'image n'existe pas.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img