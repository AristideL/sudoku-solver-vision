from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

def clamp_box(x1, y1, x2, y2, w: int, h: int) -> tuple[int, int, int, int]:
    """Clamp une bbox dans les limites image.
    Particularité: protège contre les bboxes légèrement hors cadre (fréquent en CV).
    """
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bbox after clamping.")
    return x1, y1, x2, y2

def best_box(result, target_class_name: Optional[str] = None):
    """Retourne la meilleure bbox (confiance max) d'un résultat YOLO.
    Particularité: permet de filtrer par nom de classe (utile pour gameZone).
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    names = result.names  # dict: id -> name

    candidates = []
    for i in range(len(xyxy)):
        cname = names.get(int(cls[i]), str(cls[i]))
        if target_class_name is None or cname == target_class_name:
            candidates.append((float(conf[i]), xyxy[i], int(cls[i]), cname))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0]  # (conf, xyxy, cls_id, cls_name)