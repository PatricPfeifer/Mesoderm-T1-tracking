"""Yellow outline TIFF helpers and base-anchored local mask updates."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as cc_label
from skimage.segmentation import find_boundaries

YELLOW = (255, 255, 0)


def as_label_stack(masks: np.ndarray) -> np.ndarray:
    arr = np.asarray(masks)
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"expected T,Y,X labels, got {arr.shape}")
    return arr


def as_outline_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr[np.newaxis, ...]
    if arr.ndim == 4 and arr.shape[-1] == 3:
        return arr
    raise ValueError(f"expected T,Y,X,3 outline RGB, got {arr.shape}")


def _outline_bool_from_masks(frame: np.ndarray) -> np.ndarray:
    """Boolean outline for one label frame."""
    try:
        from cellpose.utils import masks_to_outlines

        return np.asarray(masks_to_outlines(frame), dtype=bool)
    except Exception:
        if not np.any(frame):
            return np.zeros(frame.shape, dtype=bool)
        return find_boundaries(frame, mode="outer")


def masks_to_outline_rgb(masks: np.ndarray) -> np.ndarray:
    """Yellow (255,255,0) outlines on black — matches classic *_cp_outlines.tif."""
    stack = as_label_stack(masks)
    out = np.zeros((*stack.shape, 3), dtype=np.uint8)
    for t in range(stack.shape[0]):
        border = _outline_bool_from_masks(stack[t])
        out[t][border] = YELLOW
    return out


def masks_to_outline_labels(masks: np.ndarray) -> np.ndarray:
    """0/1 label stack for napari paint (1 = outline)."""
    rgb = masks_to_outline_rgb(masks)
    return (rgb.max(axis=-1) > 0).astype(np.uint8)


_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def hand_to_outline(hand: np.ndarray) -> np.ndarray:
    """TA handCorrection (255 = bond/background) → 0/1 outline, 1 px wide.

    Bond pixels that touch a cell (4-neighbourhood); the empty background
    outside the tissue is not drawn.
    """
    hand = np.asarray(hand)
    cells = hand != 255
    near_cell = ndi.binary_dilation(cells, structure=_CROSS)
    return (near_cell & ~cells).astype(np.uint8)


def hand_stack_to_outline(hand: np.ndarray) -> np.ndarray:
    """T,Y,X handCorrection → T,Y,X 0/1 outlines (see :func:`hand_to_outline`)."""
    stack = as_label_stack(hand)
    out = np.empty(stack.shape, dtype=np.uint8)
    for t in range(stack.shape[0]):
        out[t] = hand_to_outline(stack[t])
    return out


def update_outline_roi(
    outlines_t: np.ndarray,
    hand_t: np.ndarray,
    ys: slice,
    xs: slice,
    *,
    grow: int = 1,
) -> None:
    """Recompute 0/1 outlines of one frame in place around ``ys, xs``.

    Outlines depend on the 4-neighbourhood, so the ROI is grown by ``grow``
    px and computed on one more pixel of context.
    """
    h, w = hand_t.shape
    y0, y1, _ = ys.indices(h)
    x0, x1, _ = xs.indices(w)
    y0, y1 = max(0, y0 - grow), min(h, y1 + grow)
    x0, x1 = max(0, x0 - grow), min(w, x1 + grow)
    cy0, cy1 = max(0, y0 - 1), min(h, y1 + 1)
    cx0, cx1 = max(0, x0 - 1), min(w, x1 + 1)
    border = hand_to_outline(hand_t[cy0:cy1, cx0:cx1])
    outlines_t[y0:y1, x0:x1] = border[y0 - cy0 : y1 - cy0, x0 - cx0 : x1 - cx0]


def outline_labels_to_rgb(outline_labels: np.ndarray) -> np.ndarray:
    """Convert 0/1 outline labels to yellow-on-black RGB."""
    lab = as_label_stack(outline_labels)
    rgb = np.zeros((*lab.shape, 3), dtype=np.uint8)
    rgb[lab > 0] = YELLOW
    return rgb


def rgb_or_labels_to_outline_labels(data: np.ndarray) -> np.ndarray:
    """Normalize imported outlines (RGB Image or Labels) → T,Y,X uint8 0/1.

    RGB: any nonzero channel counts as outline (yellow-on-black TIFs).
    Labels / binary: nonzero pixels count as outline.
    """
    arr = np.asarray(data)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        return (arr.max(axis=-1) > 0).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return (arr.max(axis=-1) > 0).astype(np.uint8)[np.newaxis, ...]
    # Channel-first RGB: T,3,Y,X or 3,Y,X
    if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
        return (arr.max(axis=1) > 0).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] == 3:
        return (arr.max(axis=0) > 0).astype(np.uint8)[np.newaxis, ...]
    return (as_label_stack(arr) > 0).astype(np.uint8)


def outlines_rgb_to_masks(rgb: np.ndarray) -> np.ndarray:
    """Full-frame outlines → instance labels (interiors = cells; largest = background)."""
    stack = as_outline_rgb(rgb)
    out = np.zeros(stack.shape[:3], dtype=np.uint32)
    for t in range(stack.shape[0]):
        interior = stack[t].max(axis=-1) == 0
        cc = cc_label(interior, connectivity=1)
        counts = np.bincount(cc.ravel())
        if counts.size <= 1:
            continue
        counts[0] = 0
        bg = int(counts.argmax())
        cc[cc == bg] = 0
        uniq = np.unique(cc)
        uniq = uniq[uniq != 0]
        remapped = np.zeros_like(cc, dtype=np.uint32)
        for i, u in enumerate(uniq, start=1):
            remapped[cc == u] = i
        out[t] = remapped
    if out.max() < 65536:
        return out.astype(np.uint16)
    return out


def outlines_rgb_to_masks_local(
    base_masks: np.ndarray,
    edited_outlines: np.ndarray,
    *,
    dilate: int = 3,
) -> np.ndarray:
    """Update base masks only near outline edits; preserve IDs elsewhere.

    1. Diff edited outlines vs outlines from ``base_masks``.
    2. Expand to all base cells that touch the changed neighborhood.
    3. Re-fill that ROI from edited outlines; map components to base IDs
       (largest CC keeps a claimed ID; splits get new IDs; merges take majority).
    """
    base = as_label_stack(base_masks)
    edited = as_outline_rgb(edited_outlines)
    if edited.shape[:3] != base.shape:
        raise ValueError(
            f"outline shape {edited.shape[:3]} != masks shape {base.shape}"
        )

    base_ol = masks_to_outline_rgb(base)
    out = base.astype(np.uint32, copy=True)
    next_id = int(out.max()) + 1
    dilate = max(0, int(dilate))

    for t in range(base.shape[0]):
        base_b = base_ol[t].max(axis=-1) > 0
        edit_b = edited[t].max(axis=-1) > 0
        changed = base_b != edit_b
        if not np.any(changed):
            continue

        roi0 = (
            ndi.binary_dilation(changed, iterations=dilate) if dilate else changed
        )
        affected = np.unique(base[t][roi0])
        affected = affected[affected > 0]
        roi = roi0.copy()
        for lab in affected:
            roi |= base[t] == int(lab)

        interior = ~edit_b
        cc = cc_label(interior, connectivity=1)
        counts = np.bincount(cc.ravel())
        bg = 0
        if counts.size > 1:
            counts = counts.copy()
            counts[0] = 0
            bg = int(counts.argmax())

        affected_set = {int(x) for x in affected}
        comps: list[tuple[int, int, int, np.ndarray]] = []
        for c in np.unique(cc[roi]):
            c = int(c)
            if c == 0:
                continue
            pix = cc == c
            write = pix & roi
            if not np.any(write):
                continue
            # Skip true exterior background unless it still overlaps old cells.
            overlaps_cells = bool(affected_set) and np.any(
                np.isin(base[t][write], list(affected_set))
            )
            if c == bg and not overlaps_cells:
                continue
            base_vals = base[t][write]
            vals, cnts = np.unique(base_vals, return_counts=True)
            maj = 0
            for idx in np.argsort(-cnts):
                v = int(vals[idx])
                if v != 0:
                    maj = v
                    break
            if maj == 0 and overlaps_cells:
                # Open outline merged a cell into exterior — keep majority
                # among the affected labels that still sit under this component.
                sub = base_vals[np.isin(base_vals, list(affected_set))]
                if sub.size:
                    vals, cnts = np.unique(sub, return_counts=True)
                    maj = int(vals[np.argmax(cnts)])
            comps.append((int(write.sum()), c, maj, write))

        comps.sort(key=lambda row: row[0], reverse=True)
        claimed: set[int] = set()
        new_frame = out[t].copy()
        new_frame[roi] = 0

        for _size, _c, maj, write in comps:
            if maj > 0 and maj not in claimed:
                new_id = maj
                claimed.add(maj)
            else:
                new_id = next_id
                next_id += 1
            new_frame[write] = new_id

        out[t] = new_frame

    if out.max() < 65536:
        return out.astype(np.uint16)
    return out
