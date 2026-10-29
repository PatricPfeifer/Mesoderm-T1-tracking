"""Gap closing must never put one track ID on two cells of the same frame (tracks.close_gaps).

Synthetic: a grid of cells; one cell (K) disappears for a frame, comes back as its left half at frame 4
and gets a right half that appears at frame 5, i.e. two newborn tracks near the end of K's track, born
one frame apart. Before the fix K's track (whose end was never updated after the first link) was linked
to both, so the two halves shared one ID from frame 5 on.

Run with pytest (or import and call the test directly).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_divisions import CELL, GRID, grid_frame, run  # noqa: E402

K_LABEL = 3 * GRID + 3 + 1  # grid row 3, col 3


def movie() -> np.ndarray:
    full = grid_frame()
    y0, x0 = 3 * CELL, 3 * CELL
    left = np.s_[y0 : y0 + CELL, x0 : x0 + CELL // 2]
    right = np.s_[y0 : y0 + CELL, x0 + CELL // 2 : x0 + CELL]
    frames = []
    for t in range(12):
        f = full.copy()
        if t == 3:
            f[y0 : y0 + CELL, x0 : x0 + CELL] = 0  # K gone
        elif t == 4:
            f[right] = 0  # left half back
        elif t >= 5:
            f[right] = 100  # right half newborn one frame later
        frames.append(f)
    return np.stack(frames)


def test_no_track_id_on_two_cells_in_one_frame():
    res = run(movie(), gap_close=True, max_gap=3)
    for t, mapping in enumerate(res.label_to_track):
        ids = list(mapping.values())
        assert len(ids) == len(set(ids)), f"duplicate track IDs in frame {t}: {mapping}"


def test_the_gap_is_still_closed_once():
    res = run(movie(), gap_close=True, max_gap=3)
    closed = [e for e in res.events if "gap_close" in (e.note or "")]
    assert len(closed) >= 1  # K's old track is rejoined by exactly one of the newborn tracks, not both
    ends = [(e.frame_from, e.label_from) for e in closed]
    assert len(ends) == len(set(ends)), "the same track end was linked twice"
