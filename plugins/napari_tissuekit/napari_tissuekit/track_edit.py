"""Edit track IDs on a T,Y,X label stack. The pixel value is the track ID.

Edits rewrite only the pixels of the chosen IDs. Unchanged frames and the
other cells on a frame are left as they are.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

_WORKERS = min(8, os.cpu_count() or 1)


@dataclass
class CellUndo:
    """Pixels of one frame, saved before an edit."""

    frame: int
    ys: np.ndarray
    xs: np.ndarray
    values: np.ndarray


class ProtectedCellsError(ValueError):
    """An edit would also change a cell the user corrected earlier."""

    def __init__(self, message: str, frame: int, ys: np.ndarray, xs: np.ndarray):
        super().__init__(message)
        self.frame = int(frame)
        self.ys = ys
        self.xs = xs


def changed_pixels(undos: list[CellUndo], width: int) -> dict[int, np.ndarray]:
    """{frame: sorted flat pixel indices} of everything an edit rewrote."""
    out: dict[int, list[np.ndarray]] = {}
    for u in undos:
        out.setdefault(int(u.frame), []).append(
            u.ys.astype(np.int64) * int(width) + u.xs.astype(np.int64)
        )
    return {t: np.unique(np.concatenate(v)) for t, v in out.items()}


def merge_protected(records: list[dict[int, np.ndarray]]) -> dict[int, np.ndarray]:
    """Union of several ``changed_pixels`` records."""
    merged: dict[int, list[np.ndarray]] = {}
    for rec in records:
        for t, idx in rec.items():
            merged.setdefault(t, []).append(idx)
    return {t: np.unique(np.concatenate(v)) for t, v in merged.items()}


def _pixels_by_id(
    stack: np.ndarray, frames, ids: tuple[int, ...]
) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """{frame: {id: (ys, xs)}} for the ids found on each frame.

    One scan per id and frame, spread over threads (numpy releases the GIL), so a
    whole-movie edit stays fast. Frames without any of the ids map to {}.
    """

    def one(t: int):
        plane = stack[t]
        found = {}
        for i in ids:
            ys, xs = np.nonzero(plane == i)
            if len(ys):
                found[i] = (ys, xs)
        return t, found

    with ThreadPoolExecutor(_WORKERS) as pool:
        return dict(pool.map(one, frames))


def _relabel(
    stack: np.ndarray, frame: int, moves: dict[int, tuple[np.ndarray, np.ndarray, int]]
) -> CellUndo:
    """Write new IDs onto known pixels. moves: old_id -> (ys, xs, new_id)."""
    ys = np.concatenate([m[0] for m in moves.values()])
    xs = np.concatenate([m[1] for m in moves.values()])
    plane = stack[frame]
    saved = plane[ys, xs].copy()
    for _old, (my, mx, new_id) in moves.items():
        plane[my, mx] = new_id
    return CellUndo(frame, ys, xs, saved)


def set_id_on_frame(
    stack: np.ndarray, frame: int, src_id: int, dst_id: int
) -> tuple[list[CellUndo] | None, str]:
    """Replace one track ID on a single frame. Other frames stay as they are."""
    if src_id <= 0 or dst_id <= 0:
        return None, "Click a cell."
    if src_id == dst_id:
        return None, f"Track {src_id} already has this ID."
    plane = stack[frame]
    if not np.any(plane == src_id):
        return None, f"Track {src_id} is not on this frame."
    if np.any(plane == dst_id):
        return None, (
            f"Track {dst_id} is already on frame {frame} (another cell). "
            "Use Swap to exchange the two IDs."
        )
    undo = _replace(stack, int(frame), src_id, dst_id)
    return [undo], f"Frame {frame}: track {src_id} is now {dst_id}."


def link_tracks(
    stack: np.ndarray,
    keep_id: int,
    replace_id: int,
    start: int,
    protected: dict[int, np.ndarray] | None = None,
) -> tuple[list[CellUndo] | None, str]:
    """From ``start`` on, ``replace_id`` becomes ``keep_id``.

    If another cell already has ``keep_id`` on one of those frames, the two IDs
    are exchanged from ``start`` on, so no frame ever holds a duplicate ID.
    That other cell is not one the user clicked; if any of its pixels were set by
    an earlier manual edit (``protected``, from ``changed_pixels``), nothing is
    changed and ProtectedCellsError is raised.
    """
    if keep_id <= 0 or replace_id <= 0:
        return None, "Click a cell."
    if keep_id == replace_id:
        return None, "Those are already the same track."
    found = _pixels_by_id(stack, range(int(start), stack.shape[0]), (keep_id, replace_id))
    changed = [t for t, f in found.items() if replace_id in f]
    if not changed:
        return None, "That track does not continue from this frame."
    conflict = any(keep_id in f for f in found.values())
    if conflict and protected:
        width = int(stack.shape[-1])
        for t in sorted(found):
            f = found[t]
            if keep_id not in f or t not in protected:
                continue
            ys, xs = f[keep_id]
            if np.isin(ys.astype(np.int64) * width + xs, protected[t]).any():
                raise ProtectedCellsError(
                    f"Not changed: this would also change a cell you corrected "
                    f"earlier (track {keep_id} on frame {t}). Use Swap and click "
                    "that cell to change it on purpose, or Undo the earlier edit.",
                    t,
                    ys,
                    xs,
                )
    undos: list[CellUndo] = []
    for t in changed:
        f = found[t]
        moves = {replace_id: (*f[replace_id], keep_id)}
        if conflict and keep_id in f:
            moves[keep_id] = (*f[keep_id], replace_id)
        undos.append(_relabel(stack, t, moves))
    if conflict:
        # frames where only the old keep_id exists also move to replace_id
        for t, f in found.items():
            if t not in changed and keep_id in f:
                undos.append(_relabel(stack, t, {keep_id: (*f[keep_id], replace_id)}))
        undos.sort(key=lambda u: u.frame)
        return (
            undos,
            f"Track {replace_id} is now {keep_id} from frame {min(changed)} on; "
            f"the cell that had {keep_id} there now has {replace_id}.",
        )
    return undos, f"Track {replace_id} is now {keep_id} from frame {min(changed)} on."


def break_track(
    stack: np.ndarray, src_id: int, start: int
) -> tuple[list[CellUndo] | None, str]:
    """From ``start`` on, give ``src_id`` a new ID. Earlier frames keep the old one."""
    if src_id <= 0:
        return None, "Click a cell."
    found = _pixels_by_id(stack, range(int(start), stack.shape[0]), (src_id,))
    changed = [t for t, f in found.items() if src_id in f]
    if not changed:
        return None, "That track does not continue from this frame."
    new_id = int(np.max(stack)) + 1
    _ensure_id_fits(stack, new_id)
    undos = [_relabel(stack, t, {src_id: (*found[t][src_id], new_id)}) for t in changed]
    return undos, f"Track {src_id} is {new_id} from frame {min(changed)} on."


def swap_tracks(
    stack: np.ndarray, id_a: int, id_b: int, start: int
) -> tuple[list[CellUndo] | None, str]:
    """From ``start`` on, exchange two track IDs."""
    if id_a <= 0 or id_b <= 0:
        return None, "Click a cell."
    if id_a == id_b:
        return None, "Those are already the same track."
    found = _pixels_by_id(stack, range(int(start), stack.shape[0]), (id_a, id_b))
    changed = [t for t, f in found.items() if f]
    if not changed:
        return None, "Neither track continues from this frame."
    undos: list[CellUndo] = []
    for t in changed:
        f = found[t]
        moves = {}
        if id_a in f:
            moves[id_a] = (*f[id_a], id_b)
        if id_b in f:
            moves[id_b] = (*f[id_b], id_a)
        undos.append(_relabel(stack, t, moves))
    return undos, f"Tracks {id_a} and {id_b} swapped from frame {min(changed)} on."


def restore_cells(stack: np.ndarray, undos: list[CellUndo]) -> list[int]:
    """Write saved cell pixels back. Returns the frame indices touched."""
    frames: list[int] = []
    for undo in undos:
        stack[undo.frame][undo.ys, undo.xs] = undo.values
        frames.append(undo.frame)
    return frames


def _replace(stack: np.ndarray, frame: int, old_id: int, new_id: int) -> CellUndo:
    plane = stack[frame]
    ys, xs = np.nonzero(plane == old_id)
    saved = plane[ys, xs].copy()
    plane[ys, xs] = new_id
    return CellUndo(frame, ys, xs, saved)


def _ensure_id_fits(stack: np.ndarray, new_id: int) -> None:
    if np.iinfo(stack.dtype).max < new_id:
        raise ValueError(
            f"Track id {new_id} does not fit in {stack.dtype}. "
            "Save the Tracked layer and reload it as a 32-bit label image."
        )
