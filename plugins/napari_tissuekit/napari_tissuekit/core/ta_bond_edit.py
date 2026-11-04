"""Headless Tissue Analyzer bond correction (EPySeg ``paint2`` algorithms).

Ports ``Createpaintwidget.apply_drawing`` to
numpy arrays, calling the ported EPySeg ``wshed`` (``_wshed.py``).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.segmentation import expand_labels, find_boundaries, watershed

from ._wshed import wshed
from .cell_size import MIN_CELL_AREA


def labels_to_hand(labels: np.ndarray) -> np.ndarray:
    """Instance labels (0=bg) → TA handCorrection (255=bond/bg, 0=cell)."""
    lab = np.asarray(labels)
    hand = np.full(lab.shape, 255, dtype=np.uint8)
    hand[lab > 0] = 0
    hand[find_boundaries(lab, mode="outer")] = 255
    return hand


def exterior_mask(hand: np.ndarray) -> np.ndarray:
    """Large background areas of a handCorrection (not 1-2 px bonds).

    TA's ``wshed(seeds='mask')`` floods every 255 pixel, including the empty
    area outside the tissue; these pixels are restored to 255 afterwards.
    """
    return ndi.binary_opening(
        np.asarray(hand) == 255, structure=np.ones((3, 3), bool), border_value=1
    )


def enclosed_background(
    exterior: np.ndarray, drawn: np.ndarray, min_area: int = MIN_CELL_AREA
) -> np.ndarray:
    """Background pieces cut off by ``drawn`` strokes (new cells).

    Each background region split by the strokes keeps its largest piece as
    background; the other pieces (≥ ``min_area`` px) are returned.
    """
    before = label(exterior, connectivity=1)
    after = label(exterior & ~drawn, connectivity=1)
    out = np.zeros(exterior.shape, dtype=bool)
    if after.max() == 0:
        return out
    areas = np.bincount(after.ravel())
    pieces = ndi.find_objects(after)
    # Parent background region of each piece (pieces lie inside one region).
    by_region: dict[int, list[int]] = {}
    for pid, sl in enumerate(pieces, start=1):
        if sl is None:
            continue
        sub = after[sl] == pid
        parent = int(before[sl][sub][0])
        by_region.setdefault(parent, []).append(pid)
    for pids in by_region.values():
        if len(pids) < 2:
            continue
        keep = max(pids, key=lambda p: areas[p])
        for p in pids:
            if p != keep and areas[p] >= min_area:
                out[pieces[p - 1]] |= after[pieces[p - 1]] == p
    return out


def erased_into_background(
    hand: np.ndarray,
    exterior: np.ndarray,
    erased: np.ndarray,
    merged_hand: np.ndarray,
) -> np.ndarray:
    """Cells to delete because their border with the background was erased.

    A contact is an erased pixel touching the background and exactly one
    cell (so erasing the end of a bond between two edge cells only merges
    them). Returns the cells (in ``merged_hand``) that contain a contact.
    """
    cross = ndi.generate_binary_structure(2, 1)
    contact = erased & ndi.binary_dilation(exterior, structure=cross)
    if not np.any(contact):
        return np.zeros(erased.shape, dtype=bool)
    cells_before = label(np.asarray(hand) != 255, connectivity=1)
    # Distinct cells in the 4-neighbourhood: compare max and min nonzero ids.
    hi = ndi.maximum_filter(cells_before, footprint=cross)
    lo_src = np.where(cells_before > 0, cells_before, np.iinfo(cells_before.dtype).max)
    lo = ndi.minimum_filter(lo_src, footprint=cross)
    contact &= (hi > 0) & (hi == lo)
    # A stroke that also erases a bond between two cells is a merge: the
    # brush clipping the background edge must not delete the cells.
    strokes = label(erased, connectivity=2)
    merging = np.unique(strokes[erased & (hi > 0) & (lo < hi)])
    contact &= ~np.isin(strokes, merging[merging > 0])
    if not np.any(contact):
        return np.zeros(erased.shape, dtype=bool)
    cells_after = label(merged_hand != 255, connectivity=1)
    ids = np.unique(cells_after[contact])
    ids = ids[ids > 0]
    return np.isin(cells_after, ids)


MAX_ENCLOSED_GAP = 5.0  # px: half-width (distance to the nearest cell) up to which an enclosed empty gap is closed


def enclosed_gaps(labels: np.ndarray, max_gap: float = MAX_ENCLOSED_GAP) -> np.ndarray:
    """Pixels of the empty gaps in ``labels`` that are enclosed by cells (not reaching the image edge) and have a
    half-width of at most ``max_gap`` px."""
    background = np.asarray(labels) == 0
    regions = label(background, connectivity=1)
    n = int(regions.max())
    if n == 0:
        return np.zeros(background.shape, dtype=bool)
    widest = np.asarray(ndi.maximum(ndi.distance_transform_edt(background), regions, np.arange(1, n + 1)))
    edge = np.unique(np.concatenate([regions[0], regions[-1], regions[:, 0], regions[:, -1]]))
    closed = np.zeros(n + 1, dtype=bool)
    closed[1:] = widest <= max_gap
    closed[edge] = False
    closed[0] = False
    return closed[regions]


def outside_background(ext: np.ndarray, labels: np.ndarray, max_gap: float = MAX_ENCLOSED_GAP) -> np.ndarray:
    """The part of ``ext`` (see :func:`exterior_mask`) that really is outside the tissue.

    Segmentation leaves small empty gaps between cells. A gap that is enclosed by cells (does not reach the image
    edge) and whose half-width is at most ``max_gap`` px is not outside: the neighbouring cells meet in it, so it
    gets one membrane instead of one outline line on each side. Larger holes and the background outside the tissue
    stay empty.
    """
    if not ext.any():
        return ext
    gaps = enclosed_gaps(labels, max_gap)
    if not gaps.any():
        return ext
    ext_regions = label(ext, connectivity=1)
    touched = np.zeros(int(ext_regions.max()) + 1, dtype=bool)
    touched[np.unique(ext_regions[gaps & ext])] = True  # an ext area that holds a closed gap is the gap (plus its rim)
    touched[0] = False
    return ext & ~touched[ext_regions]


def thin_hand(labels_t: np.ndarray) -> np.ndarray:
    """Labels (one frame) → TA handCorrection with 1 px bonds."""
    hand = labels_to_hand(labels_t)
    ext = outside_background(exterior_mask(hand), labels_t)
    thin = np.asarray(wshed(hand, seeds="mask"), dtype=np.uint8)
    thin[ext] = 255
    return thin


def thin_hand_stack(masks: np.ndarray, progress=None) -> np.ndarray:
    """T,Y,X labels → T,Y,X handCorrection with 1 px bonds."""
    stack = np.asarray(masks)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    out = np.empty(stack.shape, dtype=np.uint8)
    n = stack.shape[0]
    # skimage watershed releases the GIL, so frames thin in parallel.
    workers = max(1, min(8, os.cpu_count() or 1))
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(thin_hand, stack[t]): t for t in range(n)}
        for done, fut in enumerate(as_completed(futures), start=1):
            out[futures[fut]] = fut.result()
            if progress is not None:
                progress(int(100 * done / n), f"thinning bonds, frame {done}/{n}")
    return out


OUTLINE_MATCH_TOLERANCE = 20  # px per frame: a saved outline this close to its reconstruction is taken as it is


def hand_from_outlines(labels_t: np.ndarray, outlines_t: np.ndarray, tolerance: int = OUTLINE_MATCH_TOLERANCE) -> np.ndarray | None:
    """handCorrection of one frame rebuilt from saved 1 px outlines (no re-thinning), or None.

    The regions between the outlines that hold a labelled pixel are the cells; everything else (the outlines and
    the empty background outside the tissue) is 255. The result is only returned when it reproduces the saved
    outlines to within ``tolerance`` px and has one cell per label, so a file that is not the plugin's own 1 px bonds (e.g.
    Cellpose-style outlines) falls back to :func:`thin_hand`.
    """
    from .outline_io import hand_to_outline

    lab = np.asarray(labels_t)
    out = np.asarray(outlines_t) > 0
    if lab.shape != out.shape:
        return None
    regions = label(~out, connectivity=1, background=0)
    labelled = (lab > 0) & ~out
    size = np.bincount(regions.ravel(), minlength=int(regions.max()) + 1)
    n_labelled = np.bincount(regions[labelled], minlength=size.size)
    # a cell is a region that is mostly labelled (the empty background outside the tissue may hold a few stray labelled pixels)
    has_cell = n_labelled * 2 >= size
    has_cell[0] = False
    # a label keeps only the region that holds most of its pixels (small pockets of a label stay bond / background)
    sel = labelled & has_cell[regions]
    base = int(lab.max()) + 1
    keys, counts = np.unique(regions[sel].astype(np.int64) * base + lab[sel], return_counts=True)
    key_region, key_label = keys // base, keys % base
    order = np.lexsort((counts, key_label))
    last = np.ones(order.size, dtype=bool)
    last[:-1] = key_label[order][1:] != key_label[order][:-1]
    keep = np.zeros_like(has_cell)
    keep[key_region[order][last]] = True
    has_cell &= keep
    hand = np.where(has_cell[regions], 0, 255).astype(np.uint8)
    if int(has_cell.sum()) != int(np.unique(lab[labelled & has_cell[regions]]).size):
        return None
    if int(np.count_nonzero((hand_to_outline(hand) > 0) != out)) > tolerance:
        return None
    return hand


def hand_stack_from_outlines(masks: np.ndarray, outlines: np.ndarray, progress=None):
    """Starting bonds for editing masks that come with saved ``outlines``: ``(hand, outlines, fits)``.

    ``fits[t]`` says the saved frame is the plugin's own 1 px bonds of these masks (as opposed to some other kind of
    outline file). Those frames keep the saved lines, so loading a saved edit and editing again moves no line - except
    when the file still draws two lines around an empty gap that is closed now (a file from an older plugin version):
    that frame, and every frame that does not fit, gets 1 px bonds rebuilt from the masks (:func:`thin_hand`).
    ``outlines`` are the bonds as shown (the saved lines or the rebuilt ones).
    """
    from .outline_io import hand_to_outline

    stack = np.asarray(masks)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    saved = np.asarray(outlines)
    if saved.ndim == 2:
        saved = saved[np.newaxis, ...]
    fits = saved.shape == stack.shape
    hand_out = np.empty(stack.shape, dtype=np.uint8)
    outline_out = np.empty(stack.shape, dtype=np.uint8)
    fit = np.zeros(stack.shape[0], dtype=bool)
    n = stack.shape[0]

    def one(t: int):
        hand = hand_from_outlines(stack[t], saved[t]) if fits else None
        fitted = hand is not None
        if fitted:
            gaps = enclosed_gaps(stack[t])
            if gaps.any() and float((hand[gaps] == 255).mean()) > 0.5:
                hand = None  # the file still draws two lines around a gap: out of date
        if hand is not None:
            return t, hand, (saved[t] > 0).astype(np.uint8), True
        hand = thin_hand(stack[t])
        return t, hand, hand_to_outline(hand), fitted

    workers = max(1, min(8, os.cpu_count() or 1))
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, t) for t in range(n)]
        for done, fut in enumerate(as_completed(futures), start=1):
            t, hand, outline, ok = fut.result()
            hand_out[t], outline_out[t], fit[t] = hand, outline, ok
            if progress is not None:
                progress(int(100 * done / n), f"preparing bonds, frame {done}/{n}")
    return hand_out, outline_out, fit


def hand_to_labels(hand: np.ndarray) -> np.ndarray:
    """handCorrection → instance labels (connectivity-1, bg=255)."""
    lab = label(np.asarray(hand), connectivity=1, background=255)
    if lab.max() < 65536:
        return lab.astype(np.uint16)
    return lab.astype(np.uint32)


def _get_white_bounds(img: np.ndarray):
    coords = np.where(img != 0)
    if coords[0].size == 0:
        return None
    return (
        int(np.min(coords[0])),
        int(np.max(coords[0])),
        int(np.min(coords[1])),
        int(np.max(coords[1])),
    )


def get_white_bounds(imgs) -> list[int] | None:
    """Same contract as ``epyseg.img.get_white_bounds`` (min_y,max_y,min_x,max_x)."""
    if isinstance(imgs, list):
        bounds = [10000000, 0, 10000000, 0]
        any_hit = False
        for img in imgs:
            cur = _get_white_bounds(img)
            if cur is None:
                continue
            any_hit = True
            bounds[0] = min(bounds[0], cur[0])
            bounds[1] = max(bounds[1], cur[1])
            bounds[2] = min(bounds[2], cur[2])
            bounds[3] = max(bounds[3], cur[3])
        return bounds if any_hit else None
    cur = _get_white_bounds(imgs)
    if cur is None:
        return None
    return [cur[0], cur[1], cur[2], cur[3]]


def stroke_diff(base_outlines: np.ndarray, edited_outlines: np.ndarray):
    """Compare 0/1 outline layers → (erased, drawn) stroke masks (uint8 0/255)."""
    base = (np.asarray(base_outlines) > 0)
    edit = (np.asarray(edited_outlines) > 0)
    erased = np.zeros(base.shape, dtype=np.uint8)
    drawn = np.zeros(base.shape, dtype=np.uint8)
    erased[base & ~edit] = 255
    drawn[edit & ~base] = 255
    return erased, drawn


def _remap_roi_labels(
    base_frame: np.ndarray, roi_slice: tuple, new_hand_roi: np.ndarray
) -> np.ndarray:
    """Paste ROI hand → labels, remapping IDs by overlap with base."""
    out = base_frame.copy()
    ys, xs = roi_slice
    new_lab = label(new_hand_roi, connectivity=1, background=255)
    if not np.any(new_lab):
        out[ys, xs] = 0
        return out

    base_roi = base_frame[ys, xs]
    next_id = int(base_frame.max()) + 1
    remapped = np.zeros_like(new_lab, dtype=np.uint32)
    claimed: set[int] = set()

    comps = []
    for prop in regionprops(new_lab):
        comps.append((prop.area, int(prop.label), prop.coords))
    comps.sort(key=lambda row: row[0], reverse=True)

    for _area, cid, coords in comps:
        vals = base_roi[coords[:, 0], coords[:, 1]]
        vals = vals[vals > 0]
        maj = 0
        if vals.size:
            u, c = np.unique(vals, return_counts=True)
            for idx in np.argsort(-c):
                v = int(u[idx])
                if v not in claimed:
                    maj = v
                    break
        if maj > 0:
            remapped[coords[:, 0], coords[:, 1]] = maj
            claimed.add(maj)
        else:
            remapped[coords[:, 0], coords[:, 1]] = next_id
            next_id += 1

    out[ys, xs] = remapped
    if out.max() < 65536:
        return out.astype(np.uint16)
    return out.astype(np.uint32)


MAX_SLIVER_WIDTH = 5.0  # px (minor axis): a "cell" this narrow and long is two lines drawn beside each other
MIN_SLIVER_ELONGATION = 2.5  # length / width
MAX_SLIVER_AREA = 200


def merge_slivers(bonds: np.ndarray, before: np.ndarray) -> np.ndarray:
    """Collapse the very narrow cells an edit created (two bond lines 1-4 px apart) into one line.

    A stroke painted beside an existing membrane leaves a sliver between the two lines. Cells narrower than
    ``MAX_SLIVER_WIDTH``, long and thin, smaller than ``MAX_SLIVER_AREA`` px (or smaller than the minimal cell size),
    and cut out of a clearly larger area of
    ``before`` (the bonds before the edit) are filled and the bonds around them thinned again. Cells that were
    already that small before the edit, and cells touching the border of ``bonds`` (cut off by the crop), are left
    alone.
    """
    h, w = bonds.shape
    lab = label(bonds, connectivity=1, background=255)
    old = label(before, connectivity=1, background=255)
    old_area = np.bincount(old.ravel())
    drop = []
    for r in regionprops(lab):
        y0, x0, y1, x1 = r.bbox
        if y0 == 0 or x0 == 0 or y1 == h or x1 == w:
            continue
        too_small = r.area < MIN_CELL_AREA  # not a cell at all (see cell_size)
        thin = (
            r.area < MAX_SLIVER_AREA
            and r.minor_axis_length < MAX_SLIVER_WIDTH
            and r.major_axis_length >= MIN_SLIVER_ELONGATION * max(r.minor_axis_length, 1.0)
        )
        if not (too_small or thin):
            continue
        was = np.bincount(old[r.coords[:, 0], r.coords[:, 1]]).argmax()
        if was == 0 or old_area[was] >= 2 * r.area:  # it was bond or part of a bigger cell: this edit made it
            drop.append(r.label)
    if not drop:
        return bonds
    sliver = np.isin(lab, drop)
    filled = bonds.copy()
    filled[sliver] = 255
    thinned = np.zeros_like(filled)
    thinned[watershed(filled, markers=None, watershed_line=True) == 0] = 255
    near = ndi.binary_dilation(sliver, iterations=4)  # only the neighbourhood of the slivers changes
    out = bonds.copy()
    out[near] = thinned[near]
    return out


def apply_drawing_hand(
    hand: np.ndarray,
    erased: np.ndarray,
    drawn: np.ndarray,
    *,
    min_seed_area: int = 0,
    pad: int = 128,
) -> np.ndarray:
    """Port of ``Createpaintwidget.apply_drawing`` on a handCorrection array.

    Merges strokes into ``hand``, then local ``wshed(..., seeds='mask')``.
    ``pad`` is the context margin around the edit (TA uses 128).
    """
    pad = int(pad)
    drawn_mask = np.asarray(hand, dtype=np.uint8).copy()
    erased = np.asarray(erased)
    drawn = np.asarray(drawn)
    # Outside-the-tissue background, from the bonds before this edit (thick
    # drawn strokes must still be thinned by the watershed).
    exterior = exterior_mask(drawn_mask) & (erased == 0)

    # Merge strokes into mask (edit_drawing + apply_drawing convention).
    drawn_mask[erased != 0] = 0
    drawn_mask[drawn != 0] = 255

    bounds = get_white_bounds([drawn, erased])
    if np.any(erased):
        # Erasing a cell's border with the background merges the cell into
        # the background, i.e. deletes it.
        gone = erased_into_background(hand, exterior, erased != 0, drawn_mask)
        if np.any(gone):
            drawn_mask[gone] = 255
            exterior |= gone
            cb = get_white_bounds(gone)
            bounds = [min(bounds[0], cb[0]), max(bounds[1], cb[1]),
                      min(bounds[2], cb[2]), max(bounds[3], cb[3])]
    if np.any(drawn):
        # Strokes in the background that cut off part of it make new cells
        # (background is 255 like a bond, so drawing alone changes nothing).
        new_cells = enclosed_background(exterior, drawn != 0)
        # Keep a 1 px bond to existing cells so the new cell stays separate.
        cells = drawn_mask == 0
        new_cells &= ~ndi.binary_dilation(cells, structure=ndi.generate_binary_structure(2, 1))
        if np.any(new_cells):
            drawn_mask[new_cells] = 0
            exterior &= ~new_cells
            cb = get_white_bounds(new_cells)
            bounds = [min(bounds[0], cb[0]), max(bounds[1], cb[1]),
                      min(bounds[2], cb[2]), max(bounds[3], cb[3])]
    if bounds is None:
        if min_seed_area:
            out = np.asarray(
                wshed(drawn_mask, seeds="mask", min_seed_area=int(min_seed_area)),
                dtype=np.uint8,
            )
            out[exterior] = 255
            return out
        return drawn_mask

    if np.count_nonzero(erased) != 0:
        lab_erased = label(erased, connectivity=2, background=0)
        rps_lab_erased = regionprops(lab_erased)
        lab_drawn_mask = label(drawn_mask, connectivity=1, background=255)
        rps_lab_drawn_mask = regionprops(lab_drawn_mask)
        for erased_bond in rps_lab_erased:
            ids = np.unique(
                lab_drawn_mask[erased_bond.coords[:, 0], erased_bond.coords[:, 1]]
            )
            for id_ in ids:
                if id_ != 0:
                    cur_bounds = rps_lab_drawn_mask[id_ - 1].bbox
                    bounds[0] = min(bounds[0], cur_bounds[0])
                    bounds[1] = max(bounds[1], cur_bounds[2])
                    bounds[2] = min(bounds[2], cur_bounds[1])
                    bounds[3] = max(bounds[3], cur_bounds[3])

    min_y, max_y, min_x, max_x = bounds
    first_x = pad
    first_y = pad
    min_x = min_x - pad
    max_x = max_x + pad
    min_y = min_y - pad
    max_y = max_y + pad

    if min_x < 0:
        first_x = pad + min_x
        min_x = 0
    if min_y < 0:
        first_y = pad + min_y
        min_y = 0
    if max_y > drawn_mask.shape[0]:
        max_y = drawn_mask.shape[0]
    if max_x > drawn_mask.shape[1]:
        max_x = drawn_mask.shape[1]

    crop = drawn_mask[min_y:max_y, min_x:max_x]
    minished = np.asarray(
        wshed(crop, seeds="mask", min_seed_area=int(min_seed_area or 0)),
        dtype=np.uint8,
    )
    minished[exterior[min_y:max_y, min_x:max_x]] = 255
    minished = merge_slivers(minished, np.asarray(hand, dtype=np.uint8)[min_y:max_y, min_x:max_x])
    minished[exterior[min_y:max_y, min_x:max_x]] = 255
    y1 = first_y
    y2 = min(first_y + (bounds[1] - bounds[0]) + 1, minished.shape[0])
    x1 = first_x
    x2 = min(first_x + (bounds[3] - bounds[2]) + 1, minished.shape[1])
    drawn_mask[min_y:max_y, min_x:max_x][y1:y2, x1:x2] = minished[y1:y2, x1:x2]
    return drawn_mask


def changed_frames(base_outlines: np.ndarray, edited_outlines: np.ndarray) -> list[int]:
    """Indices of frames whose outline layer differs from the base (T,Y,X)."""
    base = np.asarray(base_outlines)
    edit = np.asarray(edited_outlines)
    if base.ndim == 2:
        base, edit = base[np.newaxis, ...], edit[np.newaxis, ...]
    # Per frame: whole-stack temporaries are slow on large movies.
    return [
        int(t)
        for t in range(base.shape[0])
        if not np.array_equal(base[t], edit[t])
        and not np.array_equal(base[t] > 0, edit[t] > 0)
    ]


def _write_frame(out: np.ndarray, t: int, frame: np.ndarray) -> np.ndarray:
    """Store ``frame`` in ``out[t]``; upcast ``out`` if the labels overflow."""
    if frame.max(initial=0) > np.iinfo(out.dtype).max:
        out = out.astype(np.uint32)
    out[t] = frame
    return out


def _fill_bonds(
    frame: np.ndarray,
    hand: np.ndarray,
    ys: slice,
    xs: slice,
    base: np.ndarray | None = None,
) -> None:
    """Give 1 px bond pixels in ``ys, xs`` to a neighbouring cell (in place).

    Keeps masks touching like cellpose output; the outside background stays 0.
    A pixel keeps its ``base`` label when that cell still borders it, so
    untouched cell edges do not move.
    """
    h, w = frame.shape
    y0, y1, _ = ys.indices(h)
    x0, x1, _ = xs.indices(w)
    cy0, cy1 = max(0, y0 - 2), min(h, y1 + 2)
    cx0, cx1 = max(0, x0 - 2), min(w, x1 + 2)
    crop = frame[cy0:cy1, cx0:cx1]
    hand_crop = hand[cy0:cy1, cx0:cx1]
    bond = (crop == 0) & (hand_crop == 255) & ~exterior_mask(hand_crop)
    if not np.any(bond):
        return
    grown = expand_labels(crop, distance=2)
    if base is not None:
        old = base[cy0:cy1, cx0:cx1]
        padded = np.pad(crop, 1)
        bordered = np.zeros(crop.shape, dtype=bool)
        ch, cw = crop.shape
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                bordered |= padded[dy : dy + ch, dx : dx + cw] == old
        keep = bond & (old > 0) & bordered
        grown = np.where(keep, old, grown)
    inner = (slice(y0 - cy0, y1 - cy0), slice(x0 - cx0, x1 - cx0))
    sub = crop[inner]
    sub[bond[inner]] = grown[inner][bond[inner]]


def relabel_changed(base_t: np.ndarray, old_hand: np.ndarray, new_hand: np.ndarray):
    """New labels for one frame after its handCorrection changed.

    Only cells touching the changed pixels (old or new) are relabelled, with
    IDs kept by overlap. Returns ``(labels, (ys, xs))`` or None if unchanged.
    """
    diff = old_hand != new_hand
    if not np.any(diff):
        return None
    y0, y1, x0, x1 = get_white_bounds(diff)
    y1, x1 = y1 + 1, x1 + 1
    new_lab = label(new_hand, connectivity=1, background=255)
    boxes = [(y0, y1, x0, x1)]
    for lab_img in (new_lab, base_t):
        ids = np.unique(lab_img[y0:y1, x0:x1])
        objs = ndi.find_objects(lab_img)
        for i in ids[ids > 0]:
            sl = objs[int(i) - 1]
            if sl is not None:
                boxes.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    h, w = new_hand.shape
    ys = slice(max(0, min(b[0] for b in boxes) - 2), min(h, max(b[1] for b in boxes) + 2))
    xs = slice(max(0, min(b[2] for b in boxes) - 2), min(w, max(b[3] for b in boxes) + 2))
    frame = _remap_roi_labels(base_t, (ys, xs), new_hand[ys, xs])
    _fill_bonds(frame, new_hand, ys, xs, base=base_t)
    return frame, (ys, xs)


def apply_bond_strokes(
    masks: np.ndarray,
    hand: np.ndarray,
    base_outlines: np.ndarray,
    edited_outlines: np.ndarray,
    *,
    frames,
    min_seed_area: int = 0,
    pad: int = 128,
):
    """Apply painted bond strokes (TA ``apply_drawing``) to T,Y,X stacks.

    ``hand`` (T,Y,X handCorrection with 1 px bonds) is updated in place and
    ``masks`` too, unless labels outgrow its dtype (then a uint32 copy is
    returned). Returns ``(masks, rois)`` with ``rois`` = ``(t, ys, xs)``.
    """
    rois: list[tuple[int, slice, slice]] = []
    for t in frames:
        t = int(t)
        erased, drawn = stroke_diff(base_outlines[t], edited_outlines[t])
        new_hand = apply_drawing_hand(
            hand[t], erased, drawn, min_seed_area=min_seed_area, pad=pad
        )
        res = relabel_changed(masks[t], hand[t], new_hand)
        hand[t] = new_hand
        if res is not None:
            masks = _write_frame(masks, t, res[0])
            rois.append((t, *res[1]))
    return masks, rois
