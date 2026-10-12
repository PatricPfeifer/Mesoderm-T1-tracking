"""Per-frame cell tables and adjacency graphs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import expand_labels


@dataclass
class FrameCells:
    """Compact cell table for one frame.

    Labels are remapped to contiguous indices 1..n for fast matrix ops.
    ``labels`` maps local index (0 = background) -> original Cellpose label.
    """

    frame: int  # index into Movie (0..T-1 after blank drop)
    original_frame: int  # index in the on-disk stack
    labels_image: np.ndarray  # remapped 0..n compact labels, same Y,X
    original_labels: np.ndarray  # length n+1, index i -> original label (0 at 0)
    areas: np.ndarray  # length n, float64
    centroids: np.ndarray  # (n, 2) float64, (y, x)
    adjacency: dict[int, set[int]] = field(default_factory=dict)
    # adjacency keys/values use original Cellpose labels

    @property
    def n_cells(self) -> int:
        return int(self.areas.shape[0])

    def index_of(self, original_label: int) -> int | None:
        """0-based cell index for an original label, or None."""
        hits = np.where(self.original_labels[1:] == original_label)[0]
        if hits.size == 0:
            return None
        return int(hits[0])


def compact_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remap arbitrary positive labels to 1..n contiguous.

    Returns (compact_image, original_labels) where original_labels[i] is the
    source label for compact index i (original_labels[0] = 0).
    """
    uniq = np.unique(labels)
    uniq = uniq[uniq != 0]
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    original = np.zeros(len(uniq) + 1, dtype=np.int64)
    for i, lab in enumerate(uniq, start=1):
        lut[lab] = i
        original[i] = int(lab)
    return lut[labels], original


def build_adjacency(
    labels: np.ndarray, expand: int = 3
) -> dict[int, set[int]]:
    """label -> set of touching labels (original label values).

    Cellpose labels are shrunk and separated by background, so we grow them by
    ``expand`` px before reading adjacency.
    """
    work = expand_labels(labels, distance=expand) if expand > 0 else labels
    adj: dict[int, set[int]] = {
        int(v): set() for v in np.unique(labels) if v != 0
    }
    valid = set(adj)
    height, width = work.shape
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = work[: height - dy, max(0, -dx) : width - max(0, dx)]
        b = work[dy:, max(0, dx) : width + min(0, dx)]
        mask = (a > 0) & (b > 0) & (a != b)
        if not mask.any():
            continue
        for u, v in zip(a[mask].tolist(), b[mask].tolist()):
            if u in valid and v in valid:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def build_frame_cells(
    labels: np.ndarray,
    *,
    frame: int,
    original_frame: int,
    expand: int = 3,
) -> FrameCells:
    """Build a FrameCells table from one label plane."""
    compact, original = compact_labels(labels)
    n = int(original.shape[0] - 1)
    areas = np.zeros(n, dtype=np.float64)
    centroids = np.zeros((n, 2), dtype=np.float64)

    for region in regionprops(compact):
        i = int(region.label) - 1
        areas[i] = float(region.area)
        centroids[i] = region.centroid

    # Adjacency on original labels (after expand of original, not compact).
    adj = build_adjacency(labels, expand=expand)

    return FrameCells(
        frame=frame,
        original_frame=original_frame,
        labels_image=compact,
        original_labels=original,
        areas=areas,
        centroids=centroids,
        adjacency=adj,
    )


def build_all_frames(
    masks: np.ndarray,
    frame_indices: np.ndarray,
    *,
    expand: int = 3,
) -> list[FrameCells]:
    """Build FrameCells for every plane in ``masks``."""
    frames: list[FrameCells] = []
    for t in range(masks.shape[0]):
        fc = build_frame_cells(
            masks[t],
            frame=t,
            original_frame=int(frame_indices[t]),
            expand=expand,
        )
        frames.append(fc)
        if (t + 1) % 20 == 0 or t == 0 or t == masks.shape[0] - 1:
            print(
                f"  cells frame {t}/{masks.shape[0] - 1}: "
                f"{fc.n_cells} cells, avg_degree="
                f"{_avg_degree(fc.adjacency):.2f}",
                flush=True,
            )
    return frames


def _avg_degree(adj: dict[int, set[int]]) -> float:
    if not adj:
        return 0.0
    return sum(len(v) for v in adj.values()) / len(adj)


def track_neighbourhoods(
    adj: dict[int, set[int]], label_to_track: dict[int, int]
) -> dict[int, set[int]]:
    """Adjacency expressed in track-ID space."""
    return {
        label_to_track[lab]: {
            label_to_track[n] for n in neighbours if n in label_to_track
        }
        for lab, neighbours in adj.items()
        if lab in label_to_track
    }
