"""Synthetic checks for manual track edits (track_edit.py).

Run with pytest. A tiny T,Y,X label stack: two cells side by side per frame.
"""

from __future__ import annotations

import numpy as np

from napari_tissuekit.track_edit import (
    ProtectedCellsError,
    break_track,
    changed_pixels,
    link_tracks,
    merge_protected,
    restore_cells,
    set_id_on_frame,
    swap_tracks,
)

N_FRAMES = 6


def stack_with(ids_left: list[int], ids_right: list[int]) -> np.ndarray:
    """Left cell (columns 0-3) and right cell (columns 4-7) per frame; 0 = absent."""
    stack = np.zeros((N_FRAMES, 4, 8), dtype=np.int32)
    for t in range(N_FRAMES):
        if ids_left[t]:
            stack[t, :, :4] = ids_left[t]
        if ids_right[t]:
            stack[t, :, 4:] = ids_right[t]
    return stack


def ids_per_frame(stack: np.ndarray) -> list[tuple[int, int]]:
    return [(int(stack[t, 0, 0]), int(stack[t, 0, 7])) for t in range(stack.shape[0])]


def test_link_without_conflict_relabels_to_the_end():
    # cell 1 on the left until frame 2, then the same cell reappears as 7 (frames 3-5)
    stack = stack_with([1, 1, 1, 7, 7, 7], [2] * N_FRAMES)
    undos, msg = link_tracks(stack, keep_id=1, replace_id=7, start=3)
    assert undos is not None, msg
    assert ids_per_frame(stack) == [(1, 2)] * N_FRAMES
    assert "swap" not in msg


def test_link_with_conflict_swaps_ids_from_start():
    # cell 1 should continue into the cell labelled 7, but 1 is already (wrongly)
    # carried by the other cell from frame 3 on
    stack = stack_with([1, 1, 1, 7, 7, 7], [2, 2, 2, 1, 1, 1])
    undos, msg = link_tracks(stack, keep_id=1, replace_id=7, start=3)
    assert undos is not None, msg
    assert ids_per_frame(stack) == [(1, 2), (1, 2), (1, 2), (1, 7), (1, 7), (1, 7)]
    assert "the cell that had 1" in msg
    for t in range(N_FRAMES):  # never a duplicate ID on one frame
        ids = [i for i in np.unique(stack[t]) if i]
        assert len(ids) == len(set(ids)) == 2


def test_link_leaves_earlier_frames_alone():
    stack = stack_with([1, 1, 1, 7, 7, 7], [2, 2, 2, 1, 1, 1])
    before = stack[:3].copy()
    link_tracks(stack, keep_id=1, replace_id=7, start=3)
    assert np.array_equal(stack[:3], before)


def test_link_undo_restores_everything():
    stack = stack_with([1, 1, 1, 7, 7, 7], [2, 2, 2, 1, 1, 1])
    original = stack.copy()
    undos, _ = link_tracks(stack, keep_id=1, replace_id=7, start=3)
    restore_cells(stack, undos)
    assert np.array_equal(stack, original)


def test_link_rejects_bad_clicks():
    stack = stack_with([1] * N_FRAMES, [2] * N_FRAMES)
    assert link_tracks(stack, 0, 2, 1)[0] is None
    assert link_tracks(stack, 1, 1, 1)[0] is None
    assert link_tracks(stack, 1, 99, 1)[0] is None  # 99 does not continue
    assert ids_per_frame(stack) == [(1, 2)] * N_FRAMES


def test_swap_and_break_still_work():
    stack = stack_with([1] * N_FRAMES, [2] * N_FRAMES)
    undos, _ = swap_tracks(stack, 1, 2, 3)
    assert ids_per_frame(stack) == [(1, 2)] * 3 + [(2, 1)] * 3
    restore_cells(stack, undos)
    assert ids_per_frame(stack) == [(1, 2)] * N_FRAMES
    undos, _ = break_track(stack, 1, 4)
    assert ids_per_frame(stack)[:4] == [(1, 2)] * 4
    assert ids_per_frame(stack)[4] == (3, 2)


# --- the threaded implementation must equal a plain whole-frame reference ---------


def _ref_swap(stack, a, b, start):
    for t in range(start, stack.shape[0]):
        plane = stack[t]
        ma, mb = plane == a, plane == b
        plane[ma] = b
        plane[mb] = a


def _ref_link(stack, keep, replace, start):
    conflict = any((stack[t] == keep).any() for t in range(start, stack.shape[0]))
    if conflict:
        _ref_swap(stack, keep, replace, start)
    else:
        for t in range(start, stack.shape[0]):
            stack[t][stack[t] == replace] = keep


def _random_stack(rng: np.random.Generator) -> np.ndarray:
    stack = rng.integers(0, 7, size=(8, 9, 11)).astype(np.int32)
    return stack


def test_edits_match_whole_frame_reference_on_random_stacks():
    rng = np.random.default_rng(0)
    for _ in range(60):
        base = _random_stack(rng)
        a, b = (int(v) for v in rng.integers(1, 7, size=2))
        start = int(rng.integers(0, 8))
        # swap
        new, ref = base.copy(), base.copy()
        undos, _ = swap_tracks(new, a, b, start)
        if a != b:
            _ref_swap(ref, a, b, start)
            assert np.array_equal(new, ref)
            if undos:
                restore_cells(new, undos)
                assert np.array_equal(new, base)
        # link (with or without conflict)
        new, ref = base.copy(), base.copy()
        undos, _ = link_tracks(new, a, b, start)
        if a != b and undos is not None:
            _ref_link(ref, a, b, start)
            assert np.array_equal(new, ref)
            restore_cells(new, undos)
            assert np.array_equal(new, base)
        elif undos is None:
            assert np.array_equal(new, base)


# --- protection of earlier manual corrections ---------------------------------------


def three_cell_stack() -> np.ndarray:
    """Left L, middle M, right R. Frames 0-2: ids 1, 2, 3. From frame 3 the tracker
    swapped the ids of L and M (L=2, M=1); R keeps 3."""
    stack = np.zeros((6, 4, 12), dtype=np.int32)
    for t in range(6):
        left, mid = (1, 2) if t < 3 else (2, 1)
        stack[t, :, 0:4], stack[t, :, 4:8], stack[t, :, 8:12] = left, mid, 3
    return stack


def row(stack, t):
    return [int(stack[t, 0, 1]), int(stack[t, 0, 5]), int(stack[t, 0, 9])]


def test_protected_cells_are_not_undone_by_a_neighbouring_edit():
    stack = three_cell_stack()
    # edit 1: L on frame 3 should be 1 -> the two ids swap, fixing both L and M
    undos1, _ = link_tracks(stack, keep_id=1, replace_id=2, start=3)
    assert [row(stack, t) for t in (2, 3, 5)] == [[1, 2, 3]] * 3
    protected = merge_protected([changed_pixels(undos1, stack.shape[-1])])
    before = stack.copy()
    # edit 2 on the neighbour: "R on frame 3 continues M (id 2)". The cell that
    # holds id 2 from frame 3 on is M, which edit 1 set -> must be refused.
    try:
        link_tracks(stack, keep_id=2, replace_id=3, start=3, protected=protected)
        raise AssertionError("expected ProtectedCellsError")
    except ProtectedCellsError as exc:
        assert exc.frame == 3 and len(exc.ys) == 16  # the whole M cell on frame 3
    assert np.array_equal(stack, before), "a refused edit must not change anything"
    # without protection the old behaviour would have silently broken edit 1
    probe = stack.copy()
    link_tracks(probe, keep_id=2, replace_id=3, start=3)
    assert row(probe, 3) == [1, 3, 2]


def test_protection_goes_away_with_undo_and_swap_overrides_it():
    stack = three_cell_stack()
    undos1, _ = link_tracks(stack, 1, 2, 3)
    protected = merge_protected([changed_pixels(undos1, stack.shape[-1])])
    # Swap = the user selected both tracks on purpose: allowed
    undos2, _ = swap_tracks(stack, 2, 3, 3)
    assert row(stack, 3) == [1, 3, 2]
    restore_cells(stack, undos2)
    # after undoing edit 1 nothing is protected any more
    restore_cells(stack, undos1)
    undos3, _ = link_tracks(stack, 2, 3, 3, protected=merge_protected([]))
    assert undos3 is not None


def test_unprotected_conflicts_and_clean_links_still_work():
    stack = three_cell_stack()
    other = {0: np.array([0, 1, 2])}  # protection somewhere unrelated
    undos, _ = link_tracks(stack, 1, 2, 3, protected=other)
    assert undos is not None and row(stack, 4) == [1, 2, 3]
    # no conflict at all: protection is irrelevant
    stack2 = stack_with([1, 1, 1, 7, 7, 7], [2] * N_FRAMES)
    prot = {t: np.arange(32) for t in range(N_FRAMES)}
    assert link_tracks(stack2, 1, 7, 3, protected=prot)[0] is not None


def test_this_frame_mode_refuses_to_create_a_duplicate_id():
    stack = stack_with([1] * N_FRAMES, [2] * N_FRAMES)
    before = stack.copy()
    undo, msg = set_id_on_frame(stack, 2, 1, 2)
    assert undo is None and "already on frame 2" in msg
    assert np.array_equal(stack, before)
