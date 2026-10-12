"""Synthetic checks for lineage-mode divisions.

Run with ``python scripts/cellpose_track/tests/test_divisions.py`` (or pytest).
A static grid of square cells; zero motion; one cell splits into two halves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from celltrack.cells import build_frame_cells  # noqa: E402
from celltrack.motion import MotionField  # noqa: E402
from celltrack.tracks import lineage_rows, mark_divisions, render_tracked_rgb  # noqa: E402
from celltrack.tracks import render_tracked_labels, track_movie  # noqa: E402

CELL = 16
GRID = 7
N_FRAMES = 12
SPLIT_AT = 5
SPLIT_CELL = (3, 3)  # grid row, col (an interior cell)


def grid_frame() -> np.ndarray:
    size = CELL * GRID
    lab = np.zeros((size, size), dtype=np.int32)
    k = 1
    for r in range(GRID):
        for c in range(GRID):
            lab[r * CELL : (r + 1) * CELL, c * CELL : (c + 1) * CELL] = k
            k += 1
    return lab


def split(lab: np.ndarray, new_label: int) -> np.ndarray:
    """Split SPLIT_CELL into left / right halves (right half -> new_label)."""
    out = lab.copy()
    r, c = SPLIT_CELL
    y0, x0 = r * CELL, c * CELL
    out[y0 : y0 + CELL, x0 + CELL // 2 : x0 + CELL] = new_label
    return out


def run(masks: np.ndarray, **kw):
    frames = [
        build_frame_cells(masks[t], frame=t, original_frame=t, expand=1)
        for t in range(masks.shape[0])
    ]
    zero = MotionField(
        grid_dy=np.zeros((1, 1)),
        grid_dx=np.zeros((1, 1)),
        tile=masks.shape[1],
        shape_yx=masks.shape[1:],
    )
    kw.setdefault("division_mode", "lineage")
    kw.setdefault("division_penalty", 1.0)
    return track_movie(
        frames, [zero] * (len(frames) - 1), verbose=False, **kw
    )


def movie_with_split() -> np.ndarray:
    base = grid_frame()
    divided = split(base, 100)
    return np.stack([base] * SPLIT_AT + [divided] * (N_FRAMES - SPLIT_AT))


def test_no_divisions():
    masks = np.stack([grid_frame()] * N_FRAMES)
    res = run(masks)
    assert not any(e.kind == "divide" for e in res.events)
    assert res.parents == {}
    assert res.n_tracks == GRID * GRID


def test_split_is_division_with_new_ids():
    masks = movie_with_split()
    res = run(masks)
    divs = [e for e in res.events if e.kind == "divide"]
    assert len(divs) == 1, divs
    ev = divs[0]
    mother_label = SPLIT_CELL[0] * GRID + SPLIT_CELL[1] + 1
    assert ev.frame_to == SPLIT_AT and ev.label_from == mother_label
    mother = res.label_to_track[SPLIT_AT - 1][mother_label]
    d1, d2 = ev.track_id_b, ev.track_id_c
    # Mother track ends, both daughters are new and point to it.
    assert mother not in res.label_to_track[SPLIT_AT].values()
    assert {d1, d2} == set(res.parents) and set(res.parents.values()) == {mother}
    assert d1 not in res.label_to_track[SPLIT_AT - 1].values()
    # Every other cell keeps its ID across the division.
    for lab, track in res.label_to_track[SPLIT_AT - 1].items():
        if lab != mother_label:
            assert res.label_to_track[SPLIT_AT][lab] == track
    assert not ev.flagged
    rows = {r["track_id"]: r for r in lineage_rows(res)}
    assert rows[mother]["divides"] and rows[mother]["end_frame"] == SPLIT_AT - 1
    assert rows[d1]["parent_track_id"] == mother


def test_both_daughters_linked():
    res = run(movie_with_split())
    div_links = [ln for ln in res.links if ln.kind == "divide"]
    assert len(div_links) == 2
    assert {ln.track_id for ln in div_links} == set(res.parents)


def test_over_split_is_flagged():
    base = grid_frame()
    masks = np.stack(
        [base] * SPLIT_AT + [split(base, 100)] + [base] * (N_FRAMES - SPLIT_AT - 1)
    )
    res = run(masks)
    divs = [e for e in res.events if e.kind == "divide"]
    assert len(divs) == 1 and divs[0].flagged, divs
    assert res.flagged_divisions


def test_gap_close_does_not_join_daughter():
    # The right-hand neighbour X ends one frame before the division; a gap
    # closer must not glue the (nearby) daughter onto X's ended track.
    base = grid_frame()
    r, c = SPLIT_CELL
    x_label = r * GRID + (c + 1) + 1
    no_x = base.copy()
    no_x[no_x == x_label] = 0
    masks = np.stack(
        [base] * (SPLIT_AT - 1)
        + [no_x]
        + [split(no_x, 100)] * (N_FRAMES - SPLIT_AT)
    )
    res = run(masks, max_gap=3)
    divs = [e for e in res.events if e.kind == "divide"]
    assert len(divs) == 1, divs
    x_track = res.label_to_track[0][x_label]
    daughters = {divs[0].track_id_b, divs[0].track_id_c}
    assert x_track not in daughters
    assert x_track not in res.label_to_track[-1].values()
    assert set(res.parents) == daughters


def test_event_ids_follow_gap_closing():
    # One cell missing for a frame -> gap closed; event IDs must be canonical.
    base = grid_frame()
    gone = base.copy()
    gone[gone == 1] = 0
    masks = np.stack([base] * 4 + [gone] + [base] * 4)
    res = run(masks, max_gap=2)
    valid = set().union(*[set(m.values()) for m in res.label_to_track])
    for e in res.events:
        for tid in (e.track_id, e.track_id_b, e.track_id_c):
            assert tid is None or tid in valid, (e, tid)


def test_magenta_marks():
    masks = movie_with_split()
    res = run(masks)
    rgb = render_tracked_rgb(render_tracked_labels(masks, res.frames, res.label_to_track))
    mark_divisions(rgb, res)
    magenta = np.all(rgb == (255, 0, 255), axis=-1)
    frames_marked = sorted({int(t) for t in np.nonzero(magenta)[0]})
    assert frames_marked == [SPLIT_AT - 1, SPLIT_AT], frames_marked
    assert magenta[SPLIT_AT - 1].sum() == 9 and magenta[SPLIT_AT].sum() == 18


def test_legacy_mode_unchanged_behaviour():
    res = run(movie_with_split(), division_mode="legacy", division_penalty=0.8)
    assert res.parents == {}
    assert all(e.track_id_c is None for e in res.events)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failed else 0)
