"""One minimal cell size for every step (segmentation, outline edits, manual drawing, tracking).

The minimum is an AREA IN MICROMETRES (``MIN_CELL_AREA_UM2``), so it means the same cell at every magnification; with the movie's pixel
size it becomes a number of pixels (``min_cell_area``). Without a pixel size the old fixed ``MIN_CELL_AREA`` px is used."""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label

MIN_CELL_AREA_UM2 = 4.0  # um^2: the default; a labelled region with a smaller area is not a cell
MIN_CELL_AREA = 50  # px: the minimum when the pixel size is not known

_um_per_px: float | None = None  # pixel size of the movie being worked on (set by the plugin when it knows it)
_min_area_um2: float = MIN_CELL_AREA_UM2  # the minimum in use (an option: plugin spin box, --min-area-um2)


def set_min_area_um2(value: float | None) -> None:
    """Minimum cell area in um^2 for everything that follows (None: back to the default of 4 um^2)."""
    global _min_area_um2
    _min_area_um2 = float(value) if value and value > 0 else MIN_CELL_AREA_UM2


def get_min_area_um2() -> float:
    return _min_area_um2


def set_pixel_size(um_per_px: float | None) -> None:
    """Pixel size (um) the default minimum area is worked out with; None goes back to the fixed pixel minimum."""
    global _um_per_px
    _um_per_px = float(um_per_px) if um_per_px and um_per_px > 0 else None


MIN_CELL_RADIUS_FRACTION = 0.58  # of the radius of a round cell of the minimum area (4 um^2: 0.65 um)


def min_cell_area(um_per_px: float | None = None) -> int:
    """Minimum cell area in pixels: the minimum in um^2 (default 4) at this pixel size (else the one set with set_pixel_size, else MIN_CELL_AREA px)."""
    um = um_per_px if um_per_px and um_per_px > 0 else _um_per_px
    return MIN_CELL_AREA if not um else max(1, math.ceil(_min_area_um2 / (um * um)))


def min_cell_radius(um_per_px: float | None = None) -> float:
    """Smallest allowed 'half width' of a cell in pixels: the biggest circle that fits inside a hole-filled cell must have at least this
    radius (a fraction of the radius of a round cell of the minimum area), so thin ribbons of empty space are no cells but a misshapen cell
    with a solid core is."""
    return MIN_CELL_RADIUS_FRACTION * math.sqrt(min_cell_area(um_per_px) / math.pi)


def remove_small_labels(labels: np.ndarray, min_area: int | None = None) -> tuple[np.ndarray, int]:
    """Copy of a (Y,X) or (T,Y,X) label image with every label of fewer than ``min_area`` pixels in a frame set to 0.
    Returns the copy and the number of removed (frame, label) regions. ``min_area`` None: ``min_cell_area()``."""
    min_area = min_cell_area() if min_area is None else min_area
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
    labels: np.ndarray, min_area: int | None = None, reach: float = ABSORB_REACH
) -> tuple[np.ndarray, int]:
    """Copy of a (Y,X) or (T,Y,X) label image without cells under ``min_area`` px and without tiny holes.

    A label of fewer than ``min_area`` pixels in a frame is not a cell, and neither is an empty gap enclosed by
    cells (not touching the image edge) that is smaller than ``min_area``. Their pixels are not left empty:
    each is given to the nearest neighbouring cell (within ``reach`` px), so the neighbours meet there and the
    membranes run across the spot. Returns the copy and the number of removed labels plus filled gaps.
    """
    min_area = min_cell_area() if min_area is None else min_area
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
