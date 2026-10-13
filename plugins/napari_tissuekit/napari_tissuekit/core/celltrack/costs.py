"""Overlap matrices and assignment costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cells import FrameCells
from .motion import MotionField


@dataclass
class PairCosts:
    """Cost matrix and supporting arrays for one frame pair.

    Indices are 0-based into FrameCells (compact cell indices).
    ``cost[i, j]`` is finite only for candidate pairs; others are np.inf.
    """

    cost: np.ndarray  # (n_prev, n_cur) float64
    n_prev: int
    n_cur: int
    # Supporting matrices (n_prev, n_cur) and the weights used for ``cost``;
    # filled by build_pair_costs, used to score division candidates.
    iou: np.ndarray | None = None
    inter: np.ndarray | None = None
    dist: np.ndarray | None = None
    weights: dict | None = None


def warp_labels(labels: np.ndarray, field: MotionField) -> np.ndarray:
    """Nearest-neighbour warp of a label image by the motion field.

    For each destination pixel (y,x) in the *next* frame, look up the source
    pixel in the previous frame as (y - dy, x - dx). This places previous
    labels into the coordinate system of the current frame.
    """
    height, width = labels.shape
    dense = field.dense()
    yy, xx = np.mgrid[0:height, 0:width]
    src_y = np.rint(yy - dense[..., 0]).astype(np.int32)
    src_x = np.rint(xx - dense[..., 1]).astype(np.int32)
    valid = (
        (src_y >= 0) & (src_y < height) & (src_x >= 0) & (src_x < width)
    )
    out = np.zeros_like(labels)
    out[valid] = labels[src_y[valid], src_x[valid]]
    return out


def overlap_matrix(
    prev: FrameCells,
    cur: FrameCells,
    field: MotionField,
) -> tuple[np.ndarray, np.ndarray]:
    """IoU matrix between previous and current cells after warping prev.

    Returns (iou, intersection) both shaped (n_prev, n_cur).
    """
    warped = warp_labels(prev.labels_image, field)
    cur_idx = cur.labels_image
    n_prev, n_cur = prev.n_cells, cur.n_cells
    if n_prev == 0 or n_cur == 0:
        return (
            np.zeros((n_prev, n_cur), dtype=np.float64),
            np.zeros((n_prev, n_cur), dtype=np.float64),
        )

    pair = warped.astype(np.int64) * (n_cur + 1) + cur_idx.astype(np.int64)
    counts = np.bincount(
        pair.ravel(), minlength=(n_prev + 1) * (n_cur + 1)
    )
    inter = counts.reshape(n_prev + 1, n_cur + 1)[1:, 1:].astype(np.float64)
    area_prev = prev.areas[:, None]
    area_cur = cur.areas[None, :]
    union = area_prev + area_cur - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou, inter


def centroid_distance_matrix(
    prev: FrameCells, cur: FrameCells, field: MotionField
) -> np.ndarray:
    """Euclidean distance between motion-corrected prev centroids and cur."""
    n_prev, n_cur = prev.n_cells, cur.n_cells
    if n_prev == 0 or n_cur == 0:
        return np.zeros((n_prev, n_cur), dtype=np.float64)

    cy = prev.centroids[:, 0]
    cx = prev.centroids[:, 1]
    dy, dx = field.sample(cy, cx)
    pred_y = cy + dy
    pred_x = cx + dx
    # (n_prev, 1) vs (1, n_cur)
    dy2 = pred_y[:, None] - cur.centroids[None, :, 0]
    dx2 = pred_x[:, None] - cur.centroids[None, :, 1]
    return np.sqrt(dy2 * dy2 + dx2 * dx2)


def build_pair_costs(
    prev: FrameCells,
    cur: FrameCells,
    field: MotionField,
    *,
    w_iou: float = 1.0,
    w_dist: float = 0.5,
    w_area: float = 0.25,
    max_distance: float = 50.0,
    min_iou: float = 0.05,
) -> PairCosts:
    """Assemble a sparse cost matrix for LAP.

    Candidates require iou >= min_iou OR dist <= max_distance.
    Non-candidates get cost = inf.
    """
    n_prev, n_cur = prev.n_cells, cur.n_cells
    iou, inter = overlap_matrix(prev, cur, field)
    dist = centroid_distance_matrix(prev, cur, field)
    weights = {
        "w_iou": w_iou,
        "w_dist": w_dist,
        "w_area": w_area,
        "max_distance": max_distance,
    }

    cost = np.full((n_prev, n_cur), np.inf, dtype=np.float64)
    if n_prev == 0 or n_cur == 0:
        return PairCosts(
            cost=cost, n_prev=n_prev, n_cur=n_cur,
            iou=iou, inter=inter, dist=dist, weights=weights,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        area_ratio = np.abs(
            np.log(
                np.maximum(prev.areas[:, None], 1.0)
                / np.maximum(cur.areas[None, :], 1.0)
            )
        )

    candidate = (iou >= min_iou) | (dist <= max_distance)
    # Soft distance gate: still allow very high IoU beyond max_distance.
    candidate |= iou >= 0.3

    raw = (
        w_iou * (1.0 - iou)
        + w_dist * (dist / max(max_distance, 1e-6))
        + w_area * area_ratio
    )
    cost[candidate] = raw[candidate]
    return PairCosts(
        cost=cost, n_prev=n_prev, n_cur=n_cur,
        iou=iou, inter=inter, dist=dist, weights=weights,
    )
