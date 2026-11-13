"""Headless bond (outline) correction on Tissue Analyzer style handCorrection arrays (255 = bond / background).

The 1 px bonds are built once from the masks with the ported EPySeg ``wshed`` (``_wshed.py``, :func:`thin_hand`).
Edits are then applied exactly as drawn, without a watershed (:func:`apply_strokes`): lines that were not painted or
erased never move.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.segmentation import expand_labels, find_boundaries, watershed

from ._wshed import wshed
from .cell_size import min_cell_area


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
    exterior: np.ndarray, drawn: np.ndarray, min_area: int | None = None
) -> np.ndarray:
    """Background pieces cut off by ``drawn`` strokes (new cells).

    Each background region split by the strokes keeps its largest piece as
    background; the other pieces (≥ ``min_area`` px) are returned.
    """
    min_area = min_cell_area() if min_area is None else min_area
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
    # An ext area that reaches the image edge is the background outside the tissue, never a gap: a single closed gap
    # that happens to touch it (e.g. a 1 px hole beside the tissue border) must only give up its own rim, not the
    # whole outside (which would then be flooded by the watershed and leave outline lines running to the image edge).
    edge = np.unique(np.concatenate([ext_regions[0], ext_regions[-1], ext_regions[:, 0], ext_regions[:, -1]]))
    touched[edge] = False
    near_gap = ndi.binary_dilation(gaps, structure=np.ones((3, 3), bool), iterations=2)
    return ext & ~touched[ext_regions] & ~(np.isin(ext_regions, edge) & near_gap)


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


def make_bonds(masks: np.ndarray, progress=None) -> tuple[np.ndarray, np.ndarray]:
    """The outline step of segmentation: T,Y,X masks → ``(hand, outlines)``.

    ``hand`` is the handCorrection stack with 1 px bonds (:func:`thin_hand_stack`), ``outlines`` the 0/1 stack of the
    bond pixels that touch a cell (what the Outlines layer shows and ``*_cell_outlines.tif`` holds, as 0/255). Every
    segmentation (napari, command line, cluster) runs this right after the masks, so outlines always exist.
    """
    from .outline_io import hand_stack_to_outline

    hand = thin_hand_stack(masks, progress)
    return hand, hand_stack_to_outline(hand)


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
        if not fits:
            hand = thin_hand(stack[t])
            return t, hand, hand_to_outline(hand), False
        hand, rebuilt, fitted = frame_bonds(stack[t], saved[t])
        return t, hand, ((saved[t] > 0).astype(np.uint8) if rebuilt is None else rebuilt), fitted

    workers = max(1, min(8, os.cpu_count() or 1))
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, t) for t in range(n)]
        for done, fut in enumerate(as_completed(futures), start=1):
            t, hand, outline, ok = fut.result()
            hand_out[t], outline_out[t], fit[t] = hand, outline, ok
            if progress is not None:
                progress(int(100 * done / n), f"preparing bonds, frame {done}/{n}")
    return hand_out, outline_out, fit


def frame_bonds(mask_t: np.ndarray, outline_t: np.ndarray):
    """Bonds of one frame that comes with saved outlines: ``(hand, rebuilt, fitted)``.

    ``rebuilt`` is None when the saved lines are the plugin's own 1 px bonds of ``mask_t`` (they are kept, no line
    moves); otherwise (another kind of outline file, or an older plugin version's file that still draws two lines
    around a closed gap) the bonds are rebuilt from the masks and ``rebuilt`` holds their outlines. ``fitted``: the
    saved lines matched the masks (even if out of date).
    """
    from .outline_io import hand_to_outline

    hand = hand_from_outlines(mask_t, outline_t)
    fitted = hand is not None
    if fitted:
        gaps = enclosed_gaps(mask_t)
        if gaps.any() and float((hand[gaps] == 255).mean()) > 0.5:
            hand = None  # the file still draws two lines around a gap: out of date
    if hand is not None:
        return hand, None, fitted
    hand = thin_hand(mask_t)
    return hand, hand_to_outline(hand), fitted


class FrameBonds:
    """The handCorrection stack (T,Y,X) built one frame at a time, the first time a frame is used.

    Loading saved outlines then needs no pass over the whole movie: ``bonds[t]`` builds frame ``t`` from the current
    masks and outlines (``source(t) -> (mask_t, outline_t)``, :func:`frame_bonds`) when it is first read, and edits
    store their result with ``bonds[t] = ...``. Only frames that were used take memory. ``on_rebuilt(t, outline)`` is
    called when a frame's saved outlines did not fit and its bonds were rebuilt from the masks (the caller shows the
    new lines). Indexing is by frame only, like ``stack[t]``.
    """

    def __init__(self, shape, source, on_rebuilt=None):
        self.shape = tuple(int(s) for s in shape)
        self.ndim = 3
        self.dtype = np.dtype(np.uint8)
        self._source = source
        self._on_rebuilt = on_rebuilt
        self._frames: dict[int, np.ndarray] = {}
        self.rebuilt: set[int] = set()  # frames whose saved outlines were replaced by bonds rebuilt from the masks

    def __len__(self) -> int:
        return self.shape[0]

    def is_built(self, t) -> bool:
        return int(t) in self._frames

    def install(self, t, hand: np.ndarray, rebuilt: np.ndarray | None = None) -> None:
        """Store frame ``t`` computed elsewhere (e.g. ahead of time in a thread): ``frame_bonds``' hand and rebuilt."""
        t = int(t)
        self._frames[t] = np.asarray(hand, dtype=np.uint8)
        if rebuilt is not None:
            self.rebuilt.add(t)
            if self._on_rebuilt is not None:
                self._on_rebuilt(t, rebuilt)

    def __getitem__(self, t) -> np.ndarray:
        if not isinstance(t, (int, np.integer)):
            raise TypeError("FrameBonds is indexed by frame only")
        t = int(t)
        if t not in self._frames:
            if not 0 <= t < self.shape[0]:
                raise IndexError(t)
            hand, rebuilt, _ = frame_bonds(*self._source(t))
            self.install(t, hand, rebuilt)
        return self._frames[t]

    def __setitem__(self, t, value) -> None:
        # a copy, as assigning into a numpy stack would make: edits change frames in place, and the value may be an
        # undo snapshot that must stay as it is
        self._frames[int(t)] = np.array(value, dtype=np.uint8, copy=True)


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


LOOP_MIN_INSIDE = 9  # px: a hole this small inside a stroke is a curl of the brush, not a drawn loop
LOOP_GAP = 3  # px: a stroke whose two ends are at most this far apart counts as a closed loop
LOOP_MAX_GAP_ANGLE = 2 * np.pi / 3  # a loop's painted pixels leave no gap wider than this, seen from its inside
_CROSS = ndi.generate_binary_structure(2, 1)


def _stroke_loop(stroke: np.ndarray, painted: np.ndarray | None = None) -> np.ndarray:
    """Inside of the loop that one stroke draws by itself (empty when it is not closed).

    A stroke closes when it encloses an area on its own, or when its two ends are at most ``LOOP_GAP`` px apart (they
    are then joined). Enclosed areas smaller than ``LOOP_MIN_INSIDE`` px are curls of the brush and do not count.
    ``painted`` is the part of ``stroke`` that was really painted (the rest are outline pixels joining its pieces
    across the lines it crosses): joining pixels inside the loop belong to its inside, not to the loop.
    """
    from skimage.draw import line as draw_line
    from skimage.morphology import skeletonize

    if painted is None:
        painted = stroke

    def holes(s):
        filled = ndi.binary_fill_holes(s)
        outside = ~filled
        # joining pixels touching the outside are where the loop crosses a line: they are the loop itself
        joint = s & ~painted & ~ndi.binary_dilation(outside, structure=np.ones((3, 3), bool))
        inside = filled & ~(s & ~joint)
        lab = label(inside, connectivity=1)
        keep = []
        for r in regionprops(lab):
            if r.area < LOOP_MIN_INSIDE:
                continue
            # the painted pixels must go round the inside: seen from its centre, no gap wider than LOOP_MAX_GAP_ANGLE
            # (a straight stroke passing a junction encloses an area together with the lines, but from one side only)
            ring = painted & ndi.binary_dilation(lab == r.label, structure=np.ones((3, 3), bool), iterations=2)
            ys, xs = np.nonzero(ring)
            if ys.size < 3:
                continue
            ang = np.sort(np.arctan2(ys - r.centroid[0], xs - r.centroid[1]))
            gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi]))
            if gaps.max() <= LOOP_MAX_GAP_ANGLE:
                keep.append(r.label)
        return np.isin(lab, keep)

    nothing = np.zeros(stroke.shape, dtype=bool)
    inside = holes(stroke)
    if inside.any():
        return inside, nothing
    skel = skeletonize(painted)
    nb = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - skel
    ends = np.argwhere(skel & (nb == 1))
    gap = nothing.copy()
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            if np.hypot(*(ends[i] - ends[j])) <= LOOP_GAP:
                rr, cc = draw_line(*ends[i], *ends[j])
                gap[rr, cc] = True
    gap &= ~stroke
    if not gap.any():
        return inside, nothing
    painted = painted | gap
    inside = holes(stroke | gap)
    return inside, (gap if inside.any() else nothing)


def apply_strokes(hand: np.ndarray, erased: np.ndarray, drawn: np.ndarray, report: list | None = None) -> np.ndarray:
    """Apply painted / erased outline pixels to one frame's handCorrection exactly as drawn (no watershed).

    What you draw is what you get; lines you did not touch never move:

    * Erased bond pixels join the cells on both sides. Erasing a cell's border with the background deletes the cell
      (:func:`erased_into_background`). The leftover bits of an erased membrane that now separate nothing go too.
    * Drawn pixels become membrane. The parts of a stroke that separate nothing (the loose ends sticking out past the
      last line it crosses, a stroke drawn along an existing line) are taken off again; thick strokes are thinned to
      the 1 px line between the cells they separate. A stroke that separates nothing at all (an open line) stays as
      drawn, so it can be closed later.
    * A stroke that closes on itself (:func:`_stroke_loop`) makes its inside one new cell: the old lines inside the
      loop go, the cells around lose the part inside. A stroke that cuts off part of the background outside the
      tissue makes that part a cell.
    * Pieces are cells whatever their size.

    ``report`` (a list) gets ``("open", n_px)`` for every stroke kept as an open line and ``("loop", n_px)`` for
    every new cell made by a loop or cut off the background.
    """
    hand = np.asarray(hand, dtype=np.uint8)
    erased = np.asarray(erased) != 0
    drawn = np.asarray(drawn) != 0
    new = hand.copy()
    ext = exterior_mask(hand) & ~erased  # background outside the tissue (before this edit)

    if erased.any():
        new[erased] = 0
        gone = erased_into_background(hand, ext, erased, new)
        if gone.any():
            new[gone] = 255
            ext |= gone

    # One stroke = drawn pixels joined across the membranes they cross: where a stroke crosses a line its pixels
    # there were outline already, so they are not in ``drawn`` and the stroke falls apart into pieces.
    near = ndi.binary_dilation(drawn, structure=np.ones((3, 3), bool), iterations=2)
    from .outline_io import hand_to_outline

    crossed = near & (hand_to_outline(hand) > 0) & ~drawn  # outline pixels (also the tissue's outer border)
    groups = label(drawn | crossed, connectivity=2)  # also where a stroke runs along a line for a few px
    strokes = np.where(drawn, groups, 0)
    loops = np.zeros(hand.shape, dtype=bool)
    made = []  # strokes that made a new cell by themselves (they are never "open")
    for k, sl in enumerate(ndi.find_objects(groups), start=1):
        if sl is None:
            continue
        pad = LOOP_GAP + 2
        ys = slice(max(0, sl[0].start - pad), sl[0].stop + pad)
        xs = slice(max(0, sl[1].start - pad), sl[1].stop + pad)
        inside, gap = _stroke_loop(groups[ys, xs] == k, drawn[ys, xs])
        if inside.any():
            loops[ys, xs] |= inside
            made.append(k)
            if gap.any():  # the small gap between the loop's two ends is drawn in
                drawn[ys, xs] |= gap
                strokes[ys, xs][gap] = k
    new[drawn] = 255
    if loops.any():
        new[loops] = 0  # the loop's inside is one cell: the old lines inside it go
        ext &= ~loops
        if report is not None:
            report.extend(("loop", int(n)) for n in np.bincount(label(loops, connectivity=1).ravel())[1:])
    if drawn.any():
        # background cut off by a stroke (closed against the tissue edge) becomes a cell, 1 px away from the cells
        cut = enclosed_background(ext & ~loops, groups > 0, min_area=LOOP_MIN_INSIDE)
        cut &= ~ndi.binary_dilation(new == 0, structure=_CROSS)
        if cut.any():
            new[cut] = 0
            ext &= ~cut
            made.extend(int(k) for k in np.unique(strokes[ndi.binary_dilation(cut, structure=np.ones((3, 3), bool))]) if k)
            if report is not None:
                report.extend(("loop", int(n)) for n in np.bincount(label(cut, connectivity=1).ravel())[1:])

    # Which drawn (and erased-membrane) pixels really separate two areas: flood every cell and every background area
    # into those pixels only; where two different areas meet the pixel stays membrane, the rest joins its area.
    cells = label(new == 0, connectivity=1)
    outside = label(ext & (new == 255) & ~drawn, connectivity=1)
    markers = cells.astype(np.int64)
    markers[outside > 0] = outside[outside > 0] + int(cells.max())
    stroke_px = drawn & (new == 255) & (markers == 0)
    flood = stroke_px.copy()
    if erased.any():
        # leftover bits of an erased membrane: bond pixels next to one cell only (the background around the tissue
        # does not count: a bit sticking out of the tissue's outer border is a leftover too), joined to the erased
        # pixels. A bit that touches one cell in all joins it; others are flooded like strokes.
        p = np.pad(np.where(markers <= cells.max(), markers, 0), 1)
        nbs = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        hi = nbs.max(axis=0)
        lo = np.where(nbs > 0, nbs, np.iinfo(np.int64).max).min(axis=0)
        lonely = (new == 255) & (markers == 0) & ~drawn & (hi > 0) & (hi == lo)
        bits = label(lonely, connectivity=2)
        touch = np.unique(bits[ndi.binary_dilation(erased, structure=np.ones((3, 3), bool))])
        for k in touch[touch > 0]:
            bit = bits == k
            if np.unique(hi[bit]).size == 1:
                new[bit] = 0
            else:
                flood |= bit
    if flood.any():
        ws = watershed(np.zeros(hand.shape, np.uint8), markers, mask=(markers > 0) | flood, watershed_line=True)
        joined = flood & (ws > 0)
        # pixels that join a cell become cell; those that join the background stay 255 (background)
        new[joined & (ws <= cells.max())] = 0
        separating = stroke_px & (ws == 0)
        for k in np.unique(strokes[stroke_px]):
            if k == 0 or k in made:
                continue
            comp = strokes == k
            if not (separating & comp).any():
                new[comp] = 255  # an open line: it separates nothing yet, keep it as drawn
                if report is not None:
                    report.append(("open", int(comp.sum())))
    return new


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
    report: list | None = None,
):
    """Apply painted / erased outline pixels to T,Y,X stacks exactly as drawn (:func:`apply_strokes`).

    ``hand`` (T,Y,X handCorrection with 1 px bonds) is updated in place and
    ``masks`` too, unless labels outgrow its dtype (then a uint32 copy is
    returned). Returns ``(masks, rois)`` with ``rois`` = ``(t, ys, xs)``.
    ``report`` (a list) gets ``(t, kind, n_px)`` per open line / new cell (see :func:`apply_strokes`).
    """
    rois: list[tuple[int, slice, slice]] = []
    for t in frames:
        t = int(t)
        erased, drawn = stroke_diff(base_outlines[t], edited_outlines[t])
        notes: list = []
        new_hand = apply_strokes(hand[t], erased, drawn, report=notes)
        if report is not None:
            report.extend((t, kind, n) for kind, n in notes)
        res = relabel_changed(masks[t], hand[t], new_hand)
        hand[t] = new_hand
        if res is not None:
            masks = _write_frame(masks, t, res[0])
            rois.append((t, *res[1]))
    return masks, rois
