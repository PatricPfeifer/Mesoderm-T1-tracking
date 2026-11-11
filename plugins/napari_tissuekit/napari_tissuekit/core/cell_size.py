"""One minimal cell size for every step (segmentation, outline edits, manual drawing, tracking)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label

MIN_CELL_AREA = 50  # px: a labelled region with fewer pixels is not a cell


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


ABSORB_REACH = 8.0  # px: a removed region is only filled from cells this close (an isolated speck is just dropped)


def absorb_small_labels(
    labels: np.ndarray, min_area: int = MIN_CELL_AREA, reach: float = ABSORB_REACH
) -> tuple[np.ndarray, int]:
    """Copy of a (Y,X) or (T,Y,X) label image without cells under ``min_area`` px and without tiny holes.

    A label of fewer than ``min_area`` pixels in a frame is not a cell, and neither is an empty gap enclosed by
    cells (not touching the image edge) that is smaller than ``min_area``. Their pixels are not left empty:
    each is given to the nearest neighbouring cell (within ``reach`` px), so the neighbours meet there and the
    membranes run across the spot. Returns the copy and the number of removed labels plus filled gaps.
    """
    out = np.array(labels, copy=True)
    frames = out[np.newaxis] if out.ndim == 2 else out
    changed = 0
    for frame in frames:
        counts = np.bincount(frame.ravel())
        small = np.flatnonzero((counts > 0) & (counts < min_area))
        small = small[small > 0]
        fill = np.isin(frame, small) if small.size else np.zeros(frame.shape, dtype=bool)
        if small.size:
            frame[fill] = 0
        background = frame == 0
        regions = label(background, connectivity=1)
        n = int(regions.max())
        if n:
            area = np.bincount(regions.ravel(), minlength=n + 1)
            edge = np.unique(np.concatenate([regions[0], regions[-1], regions[:, 0], regions[:, -1]]))
            tiny = area < min_area
            tiny[edge] = False
            tiny[0] = False
            tiny[np.unique(regions[fill])] = False  # the hole a removed cell leaves is counted as that cell
            fill |= tiny[regions]
            changed += int(tiny.sum())
        changed += int(small.size)
        if not fill.any():
            continue
        dist, (iy, ix) = ndi.distance_transform_edt(background, return_indices=True)
        sel = fill & (dist <= reach)
        frame[sel] = frame[iy[sel], ix[sel]]
    return out, changed
