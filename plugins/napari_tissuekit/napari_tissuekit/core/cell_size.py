"""One minimal cell size for every step (segmentation, outline edits, manual drawing, tracking)."""
from __future__ import annotations

import numpy as np

MIN_CELL_AREA = 15  # px: a labelled region with fewer pixels is not a cell


def remove_small_labels(labels: np.ndarray, min_area: int = MIN_CELL_AREA) -> tuple[np.ndarray, int]:
    """Copy of a (Y,X) or (T,Y,X) label image with every label of fewer than ``min_area`` pixels in a frame set to 0.
    Returns the copy and the number of removed (frame, label) regions."""
    out = np.array(labels, copy=True)
    frames = out[np.newaxis] if out.ndim == 2 else out
    removed = 0
    for frame in frames:
        counts = np.bincount(frame.ravel())
        small = np.flatnonzero((counts > 0) & (counts < min_area))
        small = small[small > 0]
        if small.size:
            frame[np.isin(frame, small)] = 0
            removed += int(small.size)
    return out, removed
