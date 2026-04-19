from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    """Configuration centrale du projet.
    Particularité: regroupe tous les chemins et paramètres d'inférence au même endroit.
    """
    img_path: str = "training/gamezone-models/trainer/test.png"  # change si besoin
    gamezone_weights: str = "models/gamezone/best.pt"
    cells_weights: str = "models/cells/best.pt"

    gamezone_class_name: str = "gameZone"     # classe du modèle 1
    empty_class_name: str = "emptycells"      # classe 'vide' du modèle 2

    imgsz: int = 640
    conf: float = 0.75
    iou: float = 0.55
    max_det: int = 300