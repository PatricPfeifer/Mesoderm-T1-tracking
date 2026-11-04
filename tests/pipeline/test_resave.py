"""Rebuilding all tracking files after manual track edits (core/celltrack/resave.py).

Run with pytest. Uses the synthetic grid movie from test_divisions.py (one cell
divides at frame SPLIT_AT), tracks it for real, writes the real tables, then
edits the tracked stack with track_edit and rebuilds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
from test_divisions import (  # noqa: E402
    GRID, N_FRAMES, SPLIT_AT, SPLIT_CELL, movie_with_split, run,
)

from napari_tissuekit.core.celltrack.pipeline import _write_tables  # noqa: E402
from napari_tissuekit.core.celltrack.resave import (  # noqa: E402
    id_maps, load_tables, rebuild_tracking, write_tracking,
)
from napari_tissuekit.core.celltrack.tracks import render_tracked_labels  # noqa: E402
from napari_tissuekit.track_edit import break_track, link_tracks, swap_tracks  # noqa: E402

STEM = "m"


def original(tmp_path):
    masks = movie_with_split()
    res = run(masks)
    tracked = render_tracked_labels(masks, res.frames, res.label_to_track)
    _write_tables(res, tmp_path, STEM)
    tables = load_tables(tmp_path, STEM)
    mother_label = SPLIT_CELL[0] * GRID + SPLIT_CELL[1] + 1
    mother = res.label_to_track[SPLIT_AT - 1][mother_label]
    return res, tracked, tables, mother


def other_track(res, mother):
    """Two non-dividing tracks present on every frame."""
    ids = sorted(set(res.label_to_track[0].values()) - {mother})
    return ids[0], ids[1]


def key_events(df):
    return sorted(
        (r.kind, int(r.frame_from), int(r.frame_to), int(r.track_id))
        for r in df.itertuples()
    )


def test_unedited_stack_reproduces_original_tables(tmp_path):
    res, tracked, tables, mother = original(tmp_path)
    out = rebuild_tracking(tracked.copy(), tracked, tables)
    assert out.n_ids_changed == 0 and not out.dropped_divisions
    cols = ["frame", "local_label", "track_id", "area"]
    pd.testing.assert_frame_equal(
        out.tracks.sort_values(cols[:3])[cols].reset_index(drop=True),
        tables["tracks"].sort_values(cols[:3])[cols].reset_index(drop=True),
        check_dtype=False,
    )
    old = tables["lineage"].sort_values("track_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        out.lineage.sort_values("track_id").reset_index(drop=True), old, check_dtype=False
    )
    assert key_events(out.events) == key_events(tables["events"])
    assert not (out.events["note"] == "manual edit").any()
    assert len(out.links) == len(tables["links"])
    assert out.qc["n_divide"] == 1 and out.qc["n_tracks"] == len(old)


def test_break_adds_manual_end_and_start_rows(tmp_path):
    res, tracked, tables, mother = original(tmp_path)
    a, _ = other_track(res, mother)
    edited = tracked.copy()
    undos, msg = break_track(edited, a, 6)
    assert undos is not None, msg
    new = int(edited.max())
    out = rebuild_tracking(edited, tracked, tables)
    man = out.events[out.events["note"] == "manual edit"]
    assert (("disappear", 5, 6, a) in key_events(man)) and (("appear", 5, 6, new) in key_events(man))
    assert not ((out.events["kind"] == "continue") & (out.events["track_id"] == new)
                & (out.events["frame_from"] == 5)).any()
    row = out.lineage.set_index("track_id")
    assert row.loc[a, "end_frame"] == 5 and row.loc[new, "start_frame"] == 6
    assert out.qc["n_divide"] == 1 and not out.dropped_divisions
    assert out.n_ids_changed == 1


def test_swap_links_cells_across_the_swap_frame(tmp_path):
    res, tracked, tables, mother = original(tmp_path)
    a, b = other_track(res, mother)
    edited = tracked.copy()
    undos, msg = swap_tracks(edited, a, b, 6)
    assert undos is not None, msg
    out = rebuild_tracking(edited, tracked, tables)
    cont = out.events[(out.events["kind"] == "continue") & (out.events["track_id"] == a)
                      & (out.events["frame_from"] == 5)]
    assert len(cont) == 1 and cont.iloc[0]["note"] == "manual edit"
    la, lb = cont.iloc[0]["label_from"], cont.iloc[0]["label_to"]
    assert la != lb  # the track now jumps to the other cell
    assert ((out.links["kind"] == "manual") & (out.links["frame_from"] == 5)).sum() == 2


def test_division_that_no_longer_holds_is_dropped_and_reported(tmp_path):
    res, tracked, tables, mother = original(tmp_path)
    d1 = int(tables["events"].query("kind == 'divide'").iloc[0]["track_id_b"])
    edited = tracked.copy()
    undos, msg = link_tracks(edited, mother, d1, SPLIT_AT)
    assert undos is not None, msg
    out = rebuild_tracking(edited, tracked, tables)
    # a daughter now carries the mother's ID, so the division collapsed into one track
    assert len(out.dropped_divisions) == 1 and "merged" in out.dropped_divisions[0]
    assert (out.lineage["parent_track_id"] == 0).all() and not out.lineage["divides"].any()
    assert out.qc["n_divide"] == 0
    assert not (out.events["kind"] == "divide").any()


def test_magenta_dots_follow_held_divisions(tmp_path):
    res, tracked, tables, mother = original(tmp_path)
    a, _ = other_track(res, mother)
    edited = tracked.copy()
    break_track(edited, a, 6)
    out = rebuild_tracking(edited, tracked, tables)
    magenta = (out.rgb == np.array([255, 0, 255], dtype=np.uint8)).all(axis=-1)
    assert magenta[SPLIT_AT - 1].any() and magenta[SPLIT_AT].any() and not magenta[0].any()


def test_write_backs_up_and_keeps_hard_links_intact(tmp_path):
    res, tracked, tables, mother = original(tmp_path)
    labels_path = tmp_path / f"{STEM}_tracked_labels.tif"
    tifffile.imwrite(labels_path, tracked)
    link = tmp_path / "picker_copy.tif"
    os.link(labels_path, link)  # like output/**/picker_ready
    a, _ = other_track(res, mother)
    edited = tracked.copy()
    break_track(edited, a, 6)
    out = rebuild_tracking(edited, tracked, tables)
    written, backup = write_tracking(tmp_path, STEM, out, None)
    assert backup is not None and (backup / labels_path.name).is_file()
    assert np.array_equal(tifffile.imread(labels_path), edited)
    assert np.array_equal(tifffile.imread(link), tracked)  # old content kept
    again = load_tables(tmp_path, STEM)
    assert len(again["lineage"]) == len(out.lineage)
    assert again["qc"]["manual_edit"]["ids_changed"] == 1
    assert not list(tmp_path.glob("._tmp_*"))


def test_id_maps_report_changed_ids_only():
    base = np.zeros((2, 4, 4), np.uint32)
    base[:, :, :2], base[:, :, 2:] = 1, 2
    ed = base.copy()
    ed[1][base[1] == 1], ed[1][base[1] == 2] = 2, 1
    maps, amb = id_maps(base, ed)
    assert maps == [{}, {1: 2, 2: 1}] and not amb
