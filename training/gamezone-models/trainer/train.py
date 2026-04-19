import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from ultralytics import YOLO


# ----------------------------
# Utils
# ----------------------------

def _mkdir(p: Path) -> None:
    """Create directory if missing."""
    p.mkdir(parents=True, exist_ok=True)

def _copy(src: Path, dst: Path) -> None:
    """Copy file with overwrite."""
    _mkdir(dst.parent)
    shutil.copy2(src, dst)

def _find_coco_json(dataset_dir: Path) -> Path:
    """Find a COCO annotation json file inside dataset_dir."""
    candidates = list(dataset_dir.rglob("*.json"))
    # Prefer typical names
    for name in ["coco.json", "_coco.json", "annotations.json", "instances.json"]:
        for c in candidates:
            if c.name.lower() == name or c.name.lower().endswith(name):
                return c
    if not candidates:
        raise FileNotFoundError(f"No .json found in {dataset_dir}")
    # fallback: first json
    return candidates[0]

def _find_image(path_hint: Path, filename: str) -> Path:
    """Locate an image by filename (COCO images[].file_name can be relative)."""
    # Try direct
    p = path_hint / filename
    if p.exists():
        return p
    # Try basename search
    base = Path(filename).name
    hits = list(path_hint.rglob(base))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Image not found for file_name={filename}")

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ----------------------------
# COCO -> YOLO conversion
# ----------------------------

def coco_to_yolo_labels(
    coco_json: Path,
    images_root: Path,
) -> Tuple[Dict[int, str], Dict[int, List[Tuple[int, float, float, float, float]]]]:
    """
    Convert COCO annotations to YOLO format per image.

    Returns:
      - cat_id_to_name
      - img_id_to_yolo: img_id -> list of (cls_idx, xc, yc, w, h) normalized [0..1]
    """
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Categories: map COCO category_id -> contiguous class index [0..C-1]
    categories = coco.get("categories", [])
    categories_sorted = sorted(categories, key=lambda x: int(x["id"]))
    cat_id_to_idx = {int(c["id"]): i for i, c in enumerate(categories_sorted)}
    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in categories_sorted}

    # Images info
    images = coco.get("images", [])
    img_info = {int(im["id"]): im for im in images}

    # Prepare output mapping
    img_id_to_yolo: Dict[int, List[Tuple[int, float, float, float, float]]] = {int(im["id"]): [] for im in images}

    # Convert each annotation bbox [x,y,w,h] (pixels) to YOLO [xc,yc,w,h] normalized
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        img_id = int(ann["image_id"])
        cat_id = int(ann["category_id"])
        bbox = ann.get("bbox", None)
        if bbox is None or img_id not in img_info or cat_id not in cat_id_to_idx:
            continue

        x, y, bw, bh = map(float, bbox)
        w_img = float(img_info[img_id]["width"])
        h_img = float(img_info[img_id]["height"])
        if w_img <= 0 or h_img <= 0:
            continue

        xc = (x + bw / 2.0) / w_img
        yc = (y + bh / 2.0) / h_img
        wn = bw / w_img
        hn = bh / h_img

        # Clamp to [0..1] just in case (dirty annotations)
        xc = _clamp(xc, 0.0, 1.0)
        yc = _clamp(yc, 0.0, 1.0)
        wn = _clamp(wn, 0.0, 1.0)
        hn = _clamp(hn, 0.0, 1.0)

        cls_idx = cat_id_to_idx[cat_id]
        img_id_to_yolo[img_id].append((cls_idx, xc, yc, wn, hn))

    return cat_id_to_name, img_id_to_yolo


def export_yolov8_dataset(
    dataset_dir: Path,
    output_dir: Path,
    val_split: float,
    seed: int,
) -> Path:
    """
    Build a YOLOv8 dataset folder:
      output_dir/
        images/train, images/val
        labels/train, labels/val
        data.yaml

    Returns:
      path to data.yaml
    """
    random.seed(seed)

    coco_json = _find_coco_json(dataset_dir)
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_root = dataset_dir  # we search recursively from dataset_dir
    cat_id_to_name, img_id_to_yolo = coco_to_yolo_labels(coco_json, images_root)

    images = coco.get("images", [])
    if not images:
        raise ValueError("COCO json has no images[]")

    # Split train/val by image ids
    img_ids = [int(im["id"]) for im in images]
    random.shuffle(img_ids)
    n_val = int(len(img_ids) * val_split)
    val_ids = set(img_ids[:n_val])
    train_ids = set(img_ids[n_val:])

    # Output folders
    img_train = output_dir / "images" / "train"
    img_val = output_dir / "images" / "val"
    lbl_train = output_dir / "labels" / "train"
    lbl_val = output_dir / "labels" / "val"
    _mkdir(img_train); _mkdir(img_val); _mkdir(lbl_train); _mkdir(lbl_val)

    # For each image: copy image + write label file
    for im in images:
        img_id = int(im["id"])
        file_name = str(im["file_name"])
        src_img = _find_image(images_root, file_name)
        stem = src_img.stem

        if img_id in val_ids:
            dst_img = img_val / src_img.name
            dst_lbl = lbl_val / f"{stem}.txt"
        else:
            dst_img = img_train / src_img.name
            dst_lbl = lbl_train / f"{stem}.txt"

        _copy(src_img, dst_img)

        yolo_lines = []
        for cls_idx, xc, yc, wn, hn in img_id_to_yolo.get(img_id, []):
            yolo_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        _mkdir(dst_lbl.parent)
        dst_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

    # Build YOLO data.yaml
    # IMPORTANT: Ultralytics wants "path" to dataset root and relative train/val paths.
    names = []
    # cat_id_to_name is keyed by COCO category_id, but we created contiguous indices by sorting category_id.
    # Rebuild names array in that exact order:
    sorted_cat_ids = sorted(cat_id_to_name.keys())
    for _cid in sorted_cat_ids:
        names.append(cat_id_to_name[_cid])

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join([
            f"path: {output_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "",
            "names:",
            *[f"  {i}: {n}" for i, n in enumerate(names)],
            ""
        ]),
        encoding="utf-8"
    )

    return data_yaml


# ----------------------------
# Train YOLOv8
# ----------------------------

def train_yolov8(
    data_yaml: Path,
    model: str,
    project_dir: Path,
    name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    lr0: float,
) -> None:
    """
    Train YOLOv8 using Ultralytics.
    Particularité: expose les hyperparams clés pour la comparaison (epochs/imgsz/batch/lr0/model).
    """
    y = YOLO(model)
    y.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        project=str(project_dir),
        name=name,
        device=0,   # GPU
        plots=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="Folder containing images + COCO json")
    ap.add_argument("--out_yolo_dir", default="yolo_dataset", help="Output YOLOv8 dataset folder")
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt / yolov8s.pt / ...")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.01)

    ap.add_argument("--project", default="runs/detect", help="Ultralytics output project folder")
    ap.add_argument("--name", default="sudoku_cells_y8")

    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_yolo_dir = Path(args.out_yolo_dir).resolve()
    project_dir = Path(args.project).resolve()

    print(f"[1/2] Converting COCO -> YOLOv8 dataset: {out_yolo_dir}")
    data_yaml = export_yolov8_dataset(
        dataset_dir=dataset_dir,
        output_dir=out_yolo_dir,
        val_split=args.val_split,
        seed=args.seed,
    )
    print(f"Created: {data_yaml}")

    print(f"[2/2] Training YOLOv8: model={args.model}, epochs={args.epochs}, imgsz={args.imgsz}")
    train_yolov8(
        data_yaml=data_yaml,
        model=args.model,
        project_dir=project_dir,
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
    )

    print("\nDone.")
    print(f"Best weights will be in: {project_dir / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()