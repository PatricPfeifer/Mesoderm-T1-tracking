"""Edit track IDs on a T,Y,X label stack. The pixel value is the track ID.

Edits rewrite only the pixels of the chosen IDs. Unchanged frames and the
other cells on a frame are left as they are.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CellUndo:
    """Pixels of one frame, saved before an edit."""

    frame: int
    ys: np.ndarray
    xs: np.ndarray
    values: np.ndarray


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
    undo = _replace(stack, int(frame), src_id, dst_id)
    return [undo], f"Frame {frame}: track {src_id} is now {dst_id}."


def link_tracks(
    stack: np.ndarray, keep_id: int, replace_id: int, start: int
) -> tuple[list[CellUndo] | None, str]:
    """From ``start`` on, replace ``replace_id`` with ``keep_id``."""
    if keep_id <= 0 or replace_id <= 0:
        return None, "Click a cell."
    if keep_id == replace_id:
        return None, "Those are already the same track."
    changed: list[int] = []
    for t in range(int(start), stack.shape[0]):
        plane = stack[t]
        has_keep = bool(np.any(plane == keep_id))
        has_replace = bool(np.any(plane == replace_id))
        if has_keep and has_replace:
            return None, f"Track {keep_id} is already on frame {t}."
        if has_replace:
            changed.append(t)
    if not changed:
        return None, "That track does not continue from this frame."
    undos = [_replace(stack, t, replace_id, keep_id) for t in changed]
    return undos, f"Track {replace_id} is now {keep_id} from frame {changed[0]} on."


def break_track(
    stack: np.ndarray, src_id: int, start: int
) -> tuple[list[CellUndo] | None, str]:
    """From ``start`` on, give ``src_id`` a new ID. Earlier frames keep the old one."""
    if src_id <= 0:
        return None, "Click a cell."
    changed = [
        t
        for t in range(int(start), stack.shape[0])
        if np.any(stack[t] == src_id)
    ]
    if not changed:
        return None, "That track does not continue from this frame."
    new_id = int(np.max(stack)) + 1
    _ensure_id_fits(stack, new_id)
    undos = [_replace(stack, t, src_id, new_id) for t in changed]
    return undos, f"Track {src_id} is {new_id} from frame {changed[0]} on."


def swap_tracks(
    stack: np.ndarray, id_a: int, id_b: int, start: int
) -> tuple[list[CellUndo] | None, str]:
    """From ``start`` on, exchange two track IDs."""
    if id_a <= 0 or id_b <= 0:
        return None, "Click a cell."
    if id_a == id_b:
        return None, "Those are already the same track."
    changed = [
        t
        for t in range(int(start), stack.shape[0])
        if np.any(stack[t] == id_a) or np.any(stack[t] == id_b)
    ]
    if not changed:
        return None, "Neither track continues from this frame."
    tmp = int(np.max(stack)) + 1
    _ensure_id_fits(stack, tmp)
    undos: list[CellUndo] = []
    for t in changed:
        plane = stack[t]
        mask = (plane == id_a) | (plane == id_b)
        ys, xs = np.nonzero(mask)
        undos.append(CellUndo(t, ys, xs, plane[ys, xs].copy()))
        plane[plane == id_a] = tmp
        plane[plane == id_b] = id_a
        plane[plane == tmp] = id_b
    return undos, f"Tracks {id_a} and {id_b} swapped from frame {changed[0]} on."


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
