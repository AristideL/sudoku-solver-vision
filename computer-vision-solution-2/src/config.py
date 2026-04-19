from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    gamezone_weights: str = "models/gamezone/best.pt"
    gamezone_class_name: str = "gameZone"
    imgsz: int = 640
    conf: float = 0.75
    iou: float = 0.55
    match_threshold: float = 0.35
    cell_padding_ratio: float = 0.18
