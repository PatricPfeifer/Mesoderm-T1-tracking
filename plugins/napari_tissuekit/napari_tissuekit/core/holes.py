"""Optional step after segmentation: give empty space that is enclosed by cells a cell id of its own."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label

from .cell_size import min_cell_area, min_cell_radius

MIN_HOLE_HALFWIDTH = 2.5  # px: thinner gaps are membrane gaps between neighbours, not missing cells
MIN_HOLE_SOLIDITY = 0.6  # area / convex area: below this a hole is a tangle of membrane gaps, not a cell
MIN_HOLE_FRAMES = 3  # a hole that fails only ONE of the shape rules is a cell only if it lasts this many frames in a row
LINK_OVERLAP = 0.25  # holes of consecutive frames are the same one when this much of the smaller overlaps (after growing 3 px)


def fill_enclosed_holes(
    labels: np.ndarray,
    min_area: int | None = None,
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
    Returns the copy and the number of new cells. ``min_area`` None: ``cell_size.min_cell_area()``.
    """
    min_area = min_cell_area() if min_area is None else min_area
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


def _hole_nodes(frame: np.ndarray, min_area: int, min_radius: float, min_solidity: float, min_halfwidth: float) -> list[dict]:
    """The enclosed empty areas of one frame that are big enough to be cells, each with its shape measures and the rule it fails.

    rule: 'pass', 'both' (radius and solidity), 'radius' (the biggest circle inside has a radius under ``min_radius`` px) or 'solidity'
    (area / convex area under ``min_solidity``)."""
    from skimage.morphology import convex_hull_image

    background = frame == 0
    regions = label(background, connectivity=1)
    n = int(regions.max())
    if n == 0:
        return []
    ids = np.arange(1, n + 1)
    area = np.asarray(ndi.sum(background, regions, ids))
    widest = np.asarray(ndi.maximum(ndi.distance_transform_edt(background), regions, ids))
    edge = np.zeros(n + 1, dtype=bool)
    edge[np.unique(np.concatenate([regions[0], regions[-1], regions[:, 0], regions[:, -1]]))] = True
    keep = (area >= min_area) & (widest >= min_halfwidth) & ~edge[1:]
    objects = ndi.find_objects(regions)
    nodes = []
    for i in ids[keep]:
        sl = objects[i - 1]
        crop = regions[sl] == i
        hull = int(convex_hull_image(crop).sum())
        solidity = float(area[i - 1]) / max(hull, 1)
        bad_r, bad_s = widest[i - 1] < min_radius, solidity < min_solidity
        nodes.append(dict(slice=sl, mask=crop, area=int(area[i - 1]), radius=float(widest[i - 1]), solidity=solidity,
                          rule="both" if bad_r and bad_s else "radius" if bad_r else "solidity" if bad_s else "pass"))
    return nodes


def _same_hole(a: dict, b: dict, grow: int = 3) -> bool:
    """Two holes of consecutive frames are the same one when they overlap (the first grown by ``grow`` px, so a hole that moved a little still counts)."""
    ys = slice(max(min(a["slice"][0].start, b["slice"][0].start) - grow, 0), max(a["slice"][0].stop, b["slice"][0].stop) + grow)
    xs = slice(max(min(a["slice"][1].start, b["slice"][1].start) - grow, 0), max(a["slice"][1].stop, b["slice"][1].stop) + grow)
    shape = (ys.stop - ys.start, xs.stop - xs.start)

    def place(node):
        canvas = np.zeros(shape, dtype=bool)
        sy, sx = node["slice"]
        canvas[sy.start - ys.start:sy.stop - ys.start, sx.start - xs.start:sx.stop - xs.start] = node["mask"]
        return canvas

    first = ndi.binary_dilation(place(a), iterations=grow)
    return (first & place(b)).sum() >= LINK_OVERLAP * min(a["area"], b["area"])


def fill_holes_stack(
    masks: np.ndarray,
    min_area: int | None = None,
    min_radius: float | None = None,
    min_solidity: float = MIN_HOLE_SOLIDITY,
    min_frames: int = MIN_HOLE_FRAMES,
    min_halfwidth: float = MIN_HOLE_HALFWIDTH,
) -> tuple[np.ndarray, dict]:
    """Fill the empty areas enclosed by cells of a (T,Y,X) label stack, but only those that can be cells.

    An enclosed empty area of at least ``min_area`` px is checked on two shapes: the biggest circle inside it must have a radius of at least
    ``min_radius`` px (``cell_size.min_cell_radius``), and area / convex area must be at least ``min_solidity``.
      - passes both: a new cell (ids follow the highest id of its frame);
      - fails both: never a cell, its space is taken over by the nearest cells;
      - fails one: a cell only if the same area is found in ``min_frames`` frames in a row (areas of consecutive frames that overlap are the
        same one; areas that fail both do not count), otherwise taken over like the first case.
    Areas touching the image edge (the outside of the tissue) are left as they are. A 2D input has no neighbouring frames, so an area that
    fails one rule is taken over. Returns the copy and a report with the number of areas per outcome."""
    min_area = min_cell_area() if min_area is None else min_area
    min_radius = min_cell_radius() if min_radius is None else min_radius
    out = np.array(masks, copy=True)
    stack = out[np.newaxis] if out.ndim == 2 else out
    nodes = [_hole_nodes(f, min_area, min_radius, min_solidity, min_halfwidth) for f in stack]

    # union-find over the areas that fail at most one rule: chains through consecutive frames
    parent = {}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for t, ns in enumerate(nodes):
        for i, n in enumerate(ns):
            if n["rule"] != "both":
                parent[(t, i)] = (t, i)
    for t in range(len(nodes) - 1):
        for i, a in enumerate(nodes[t]):
            if a["rule"] == "both":
                continue
            for j, b in enumerate(nodes[t + 1]):
                if b["rule"] != "both" and _same_hole(a, b):
                    parent[find((t, i))] = find((t + 1, j))
    frames_of: dict = {}
    for k in parent:
        frames_of.setdefault(find(k), set()).add(k[0])

    report = dict(candidates=0, cells=0, persistent=0, absorbed_both=0, absorbed_short=0)
    for t, ns in enumerate(nodes):
        frame = stack[t]
        take, give = [], []  # to become cells / to be taken over
        for i, n in enumerate(ns):
            report["candidates"] += 1
            if n["rule"] == "pass":
                take.append(n)
                report["cells"] += 1
            elif n["rule"] != "both" and len(frames_of[find((t, i))]) >= min_frames:
                take.append(n)
                report["persistent"] += 1
            else:
                give.append(n)
                report["absorbed_both" if n["rule"] == "both" else "absorbed_short"] += 1
        if give:
            rejected = np.zeros(frame.shape, dtype=bool)
            for n in give:
                rejected[n["slice"]] |= n["mask"]
        top = int(frame.max())
        if take and top + len(take) > np.iinfo(frame.dtype).max:
            raise ValueError(f"label dtype {frame.dtype} cannot hold {top + len(take)} labels")
        for k, n in enumerate(take):
            view = frame[n["slice"]]
            view[n["mask"]] = top + 1 + k
        if give:
            _, (iy, ix) = ndi.distance_transform_edt(frame == 0, return_indices=True)
            frame[rejected] = frame[iy[rejected], ix[rejected]]
    return out, report


def repair_gaps(labels: np.ndarray, same_share: float = 0.6, edge_share: float = 0.8) -> tuple[np.ndarray, dict]:
    """Fill every empty area enclosed by cells in a TRACKED (T,Y,X) stack, deciding with the neighbouring frames.

    For a gap (any size or shape, not touching the image edge) the cell that covers its pixels in the frame before and in the frame after is
    looked up. If it is the SAME cell (track id) in both and covers at least ``same_share`` of the gap in each, the cell was missed in this
    frame: the gap gets that id and the track is not broken. In the first or last frame only one neighbour exists and it must cover at least
    ``edge_share``. In every other case (several cells cover the spot, the cell is gone or only appears, different cells before and after) the
    tissue closed up there: the nearest cells take the gap over. Evidence is read from the unchanged stack, so repairs do not feed each other.
    Returns the copy and a report: counts per outcome."""
    out = np.array(labels, copy=True)
    stack = out[np.newaxis] if out.ndim == 2 else out
    src = stack.copy() if stack.shape[0] > 1 else stack
    n = stack.shape[0]
    report = dict(gaps=0, restored=0, absorbed=0)
    for f in range(n):
        frame = stack[f]
        bg, k = ndi.label(frame == 0)
        if k == 0:
            continue
        edge = set(np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]])).tolist())
        objects = ndi.find_objects(bg)
        take = []
        rejected = np.zeros(frame.shape, dtype=bool)
        for i in range(1, k + 1):
            if i in edge:
                continue
            sl = objects[i - 1]
            m = bg[sl] == i
            report["gaps"] += 1
            tops = []
            for g in (f - 1, f + 1):
                if 0 <= g < n:
                    ids = src[g][sl][m]
                    ids = ids[ids > 0]
                    if ids.size:
                        c = np.bincount(ids)
                        tops.append((int(c.argmax()), float(c.max()) / m.sum()))
                    else:
                        tops.append((0, 0.0))
            if len(tops) == 2:
                same = tops[0][0] == tops[1][0] and tops[0][0] > 0 and min(tops[0][1], tops[1][1]) >= same_share
            else:
                same = len(tops) == 1 and tops[0][0] > 0 and tops[0][1] >= edge_share
            if same:
                take.append((sl, m, tops[0][0]))
                report["restored"] += 1
            else:
                rejected[sl] |= m
                report["absorbed"] += 1
        for sl, m, tid in take:
            frame[sl][m] = tid
        if rejected.any():
            _, (iy, ix) = ndi.distance_transform_edt(frame == 0, return_indices=True)
            frame[rejected] = frame[iy[rejected], ix[rejected]]
    return out, report
