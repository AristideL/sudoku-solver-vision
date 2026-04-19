from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set, Tuple

import numpy as np


def kmeans_1d(values: np.ndarray, k: int, iters: int = 50, tol: float = 1e-3) -> np.ndarray:
    """Cluster 1D values into k centroids (Lloyd k-means).

    Notes:
    - Deterministic init via quantiles -> stable mapping for grid rows/cols.
    - Returns sorted centroids.
    """
    v = np.asarray(values, dtype=np.float32)
    if v.size < k:
        raise ValueError(f"kmeans_1d: need >= {k} points, got {v.size}")

    v = np.sort(v)
    centroids = np.quantile(v, np.linspace(0, 1, k), method="linear").astype(np.float32)

    for _ in range(iters):
        labels = np.abs(v[:, None] - centroids[None, :]).argmin(axis=1)

        new_centroids = centroids.copy()
        for j in range(k):
            pts = v[labels == j]
            if pts.size:
                new_centroids[j] = float(pts.mean())

        if np.allclose(new_centroids, centroids, atol=tol):
            break
        centroids = new_centroids

    return np.sort(centroids)


@dataclass(frozen=True)
class GridMapping:
    """Result of mapping detections to a 9x9 grid.

    grid_state: '.' empty, '#' filled
    box_grid:  (9,9,4) int xyxy bbox per cell, or -1 when missing
    class_grid: (9,9) class name per cell ('' if missing)
    classes_seen: set of class names present in detections
    """
    grid_state: np.ndarray
    box_grid: np.ndarray
    class_grid: np.ndarray
    classes_seen: Set[str]


def build_occupancy_grid(
    xyxy: np.ndarray,
    conf: np.ndarray,
    cls: np.ndarray,
    names: dict,
    empty_class_name: str,
) -> GridMapping:
    """Map cell detections (xyxy/conf/cls) to a 9x9 grid.

    Notes:
    - Uses 1D k-means on bbox centers to estimate 9 row centers + 9 col centers.
    - Handles collisions (multiple detections to same cell) by keeping max confidence.
    - Returns bbox/class per cell -> needed for OCR later.
    """
    xyxy = np.asarray(xyxy, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)
    cls = np.asarray(cls, dtype=np.int32)

    if xyxy.ndim != 2 or xyxy.shape[1] != 4:
        raise ValueError(f"xyxy must be (N,4), got {xyxy.shape}")
    if conf.shape[0] != xyxy.shape[0] or cls.shape[0] != xyxy.shape[0]:
        raise ValueError("conf/cls must have same length as xyxy")

    # Centers
    cx = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
    cy = (xyxy[:, 1] + xyxy[:, 3]) * 0.5

    row_centroids = kmeans_1d(cy, k=9)
    col_centroids = kmeans_1d(cx, k=9)

    grid_state = np.full((9, 9), ".", dtype="<U1")
    best_score = np.full((9, 9), -1.0, dtype=np.float32)

    # Store per-cell bbox + class (for OCR later)
    box_grid = np.full((9, 9, 4), -1, dtype=np.int32)  # xyxy
    class_grid = np.full((9, 9), "", dtype="<U32")

    classes_seen: Set[str] = set()

    for i in range(xyxy.shape[0]):
        cname = names.get(int(cls[i]), str(int(cls[i])))
        classes_seen.add(cname)

        r = int(np.argmin(np.abs(row_centroids - cy[i])))
        c = int(np.argmin(np.abs(col_centroids - cx[i])))

        is_empty = (cname == empty_class_name)
        state = "." if is_empty else "#"

        if float(conf[i]) > float(best_score[r, c]):
            best_score[r, c] = float(conf[i])
            grid_state[r, c] = state
            box_grid[r, c] = xyxy[i].astype(np.int32)
            class_grid[r, c] = cname

    return GridMapping(
        grid_state=grid_state,
        box_grid=box_grid,
        class_grid=class_grid,
        classes_seen=classes_seen,
    )