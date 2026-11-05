"""Optional step after segmentation: give empty space that is enclosed by cells a cell id of its own."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label

from .cell_size import MIN_CELL_AREA

MIN_HOLE_HALFWIDTH = 2.5  # px: thinner gaps are membrane gaps between neighbours, not missing cells


def fill_enclosed_holes(
    labels: np.ndarray,
    min_area: int = MIN_CELL_AREA,
    min_halfwidth: float = MIN_HOLE_HALFWIDTH,
    max_area: int | None = None,
) -> tuple[np.ndarray, int]:
    """Copy of a (Y,X) or (T,Y,X) label image in which every enclosed hole becomes a new cell.

    A hole is a connected empty region (4-neighbour) that is closed by cells. A region that touches the
    image edge anywhere is NOT closed (a cell edge that would reach the image border does not close a
    cell), so the tissue's outside and every gap running off the frame stay empty. A closed region only
    becomes a cell if it has at least ``min_area`` px and is at least ``min_halfwidth`` px thick
    (distance to the nearest cell), so thin membrane gaps are left to the outline step, and, when
    given, at most ``max_area`` px. New ids follow the highest id of each frame.
    Returns the copy and the number of new cells.
    """
    out = np.array(labels, copy=True)
    frames = out[np.newaxis] if out.ndim == 2 else out
    added = 0
    for frame in frames:
        background = frame == 0
        regions = label(background, connectivity=1)
        n = int(regions.max())
        if n == 0:
            continue
        ids = np.arange(1, n + 1)
        area = np.asarray(ndi.sum(background, regions, ids))
        widest = np.asarray(ndi.maximum(ndi.distance_transform_edt(background), regions, ids))
        touches_edge = np.zeros(n + 1, dtype=bool)
        touches_edge[np.unique(np.concatenate([regions[0], regions[-1], regions[:, 0], regions[:, -1]]))] = True
        keep = (area >= min_area) & (widest >= min_halfwidth) & ~touches_edge[1:]
        if max_area is not None:
            keep &= area <= max_area
        todo = ids[keep]
        if todo.size == 0:
            continue
        top = int(frame.max())
        if top + todo.size > np.iinfo(frame.dtype).max:
            raise ValueError(f"label dtype {frame.dtype} cannot hold {top + todo.size} labels")
        new_id = np.zeros(n + 1, dtype=frame.dtype)
        new_id[todo] = np.arange(top + 1, top + 1 + todo.size, dtype=frame.dtype)
        sel = new_id[regions] > 0
        frame[sel] = new_id[regions][sel]
        added += int(todo.size)
    return out, added
