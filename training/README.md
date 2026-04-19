# Training — YOLO Model Training

Contains datasets, training scripts, and trained weights for the two YOLO models used by both solutions.

## Models

| Model | Classes | Dataset | Weights |
|-------|---------|---------|---------|
| **gamezone** | `sudoku`, `gameZone` | 16 annotated screenshots | `gamezone-models/trainer/runs/detect/y8n_cells_960/weights/best.pt` |
| **cells** | `cells`, `emptycells`, `numbercells` | 17 annotated gamezone crops | `cells-models/trainer/runs/detect/y8n_cells_960/weights/best.pt` |

## Folder structure

```
training/
├── gamezone-models/
│   ├── datasets/                        # COCO-format annotations + images
│   │   ├── _annotations.coco.json
│   │   └── *.png / *.jpg
│   └── trainer/
│       ├── train_from_coco_to_yolov8.py # Training script (COCO → YOLO + train)
│       ├── yolov8n.pt                   # Base YOLOv8n pretrained weights
│       ├── data_yolo8_cells/            # Generated YOLO dataset (images + labels)
│       │   └── data.yaml
│       └── runs/detect/y8n_cells_960/   # Training output
│           ├── weights/best.pt          # ← Best trained weights
│           ├── results.csv
│           ├── confusion_matrix.png
│           └── *.jpg                    # Training batch previews
├── cells-models/
│   └── (same structure as gamezone-models)
├── Dockerfile
└── README.md
```

## Training script

Both models use the same script: `train_from_coco_to_yolov8.py`

The script does two things:
1. **Converts** COCO annotations to YOLOv8 format (images/labels split into train/val)
2. **Trains** a YOLOv8 model on the converted dataset

## How to train

### Without Docker

```bash
# Train gamezone model
python gamezone-models/trainer/train_from_coco_to_yolov8.py \
  --dataset_dir gamezone-models/datasets \
  --out_yolo_dir gamezone-models/trainer/data_yolo8_cells \
  --model yolov8n.pt \
  --epochs 150 \
  --imgsz 640 \
  --batch 32 \
  --project gamezone-models/trainer/runs/detect \
  --name y8n_cells_960

# Train cells model
python cells-models/trainer/train_from_coco_to_yolov8.py \
  --dataset_dir cells-models/datasets \
  --out_yolo_dir cells-models/trainer/data_yolo8_cells \
  --model yolov8n.pt \
  --epochs 150 \
  --imgsz 960 \
  --batch 16 \
  --project cells-models/trainer/runs/detect \
  --name y8n_cells_960
```

### With Docker

```bash
# Build training image
docker build -t sudoku-training .

# Train gamezone model (GPU)
docker run --rm --gpus all --shm-size=8g -v "${PWD}:/training" sudoku-training python gamezone-models/trainer/train.py --dataset_dir gamezone-models/datasets --out_yolo_dir gamezone-models/trainer/data_yolo8_cells --model yolov8n.pt --epochs 150 --imgsz 640 --batch 32 --project gamezone-models/trainer/runs/detect --name y8n_cells_960

# Train cells model (GPU)
docker run --rm --gpus all --shm-size=8g -v "${PWD}:/training" sudoku-training python cells-models/trainer/train.py --dataset_dir cells-models/datasets --out_yolo_dir cells-models/trainer/data_yolo8_cells --model yolov8n.pt --epochs 150 --imgsz 960 --batch 16 --project cells-models/trainer/runs/detect --name y8n_cells_960

# Interactive session
docker run -it --rm --gpus all -v $(pwd):/training sudoku-training bash
```

## Script parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset_dir` | required | Folder with images + COCO JSON |
| `--out_yolo_dir` | `yolo_dataset` | Output YOLOv8 dataset folder |
| `--val_split` | 0.15 | Validation split ratio |
| `--seed` | 42 | Random seed for reproducibility |
| `--model` | `yolov8n.pt` | Base model (`yolov8n.pt`, `yolov8s.pt`, etc.) |
| `--epochs` | 50 | Training epochs |
| `--imgsz` | 960 | Training image size |
| `--batch` | 16 | Batch size |
| `--lr0` | 0.01 | Initial learning rate |
| `--project` | `runs/detect` | Output project directory |
| `--name` | `sudoku_cells_y8` | Run name |

## Deploying trained weights

After training, copy `best.pt` to the solution folders:

```bash
# For Solution 1
cp gamezone-models/trainer/runs/detect/y8n_cells_960/weights/best.pt \
   ../computer-vision-solution-1/models/gamezone/best.pt

cp cells-models/trainer/runs/detect/y8n_cells_960/weights/best.pt \
   ../computer-vision-solution-1/models/cells/best.pt

# For Solution 2 (gamezone only)
cp gamezone-models/trainer/runs/detect/y8n_cells_960/weights/best.pt \
   ../computer-vision-solution-2/models/gamezone/best.pt
```

## Dataset annotation

Datasets were annotated using [Roboflow](https://roboflow.com/) and exported in COCO format. The training script handles conversion to YOLO format automatically.
