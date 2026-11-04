"""Test-only comparison script analysis/t1_div_vs_strict_test.py: membership + colours."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
import t1_div_vs_strict_test as m  # noqa: E402


def row(frame_start=2, frame_end=2, ab="1;2", cd="3;4", x=5.0, y=5.0):
    return pd.Series(
        {
            "A": 1, "B": 2, "C": 3, "D": 4,
            "pair_breaking": ab, "pair_building": cd,
            "frame_start": frame_start, "frame_end": frame_end,
            "x": x, "y": y, "type": "completed", "kind": "completed",
        }
    )


def test_membership_classes():
    shared, div_only, strict_only = row(), row(frame_start=5, frame_end=5), row(ab="1;3")
    div = {m.event_key(shared): shared, m.event_key(div_only): div_only}
    strict = {m.event_key(shared): shared, m.event_key(strict_only): strict_only}
    got = {m.event_key(r): c for c, r in m.membership(div, strict)}
    assert got[m.event_key(shared)] == m.BOTH
    assert got[m.event_key(div_only)] == m.DIV_ONLY
    assert got[m.event_key(strict_only)] == m.STRICT_ONLY


def test_strict_subset_gives_no_red():
    a, b = row(), row(frame_start=7, frame_end=7)
    div = {m.event_key(a): a, m.event_key(b): b}
    strict = {m.event_key(a): a}
    classes = [c for c, _ in m.membership(div, strict)]
    assert classes.count(m.STRICT_ONLY) == 0 and set(classes) == {m.BOTH, m.DIV_ONLY}


def two_cell_frames(n=8):
    labels = np.zeros((n, 12, 12), dtype=np.int32)
    labels[:, :, :6] = 1
    labels[:, :, 6:] = 2  # cells 1 and 2 touch: a junction at x = 5/6
    return labels


def painted(color, frame):
    return bool((np.all(frame == np.array(color, dtype=np.uint8), axis=-1)).any())


def test_each_membership_gets_its_colour_and_only_its_colour():
    rgb = np.full((8, 12, 12, 3), 90, dtype=np.uint8)
    labels = two_cell_frames()
    for cls in (m.DIV_ONLY, m.BOTH, m.STRICT_ONLY):
        bright, dark = m.COLORS[cls]
        out = m.render_membership(rgb, labels, [(cls, row(cd="1;2"))], pre=1, post=1, reach=1)
        assert painted(bright, out[1]) and not painted(dark, out[1])  # before: bright
        assert painted(dark, out[3]) and not painted(bright, out[3])  # after: dark
        for other, shades in m.COLORS.items():
            if other != cls:
                assert not any(painted(c, out[1]) or painted(c, out[3]) for c in shades)
        assert not painted(bright, out[6]) and not painted(dark, out[6])  # outside window


def test_bright_before_dark_after_and_yellow_is_gone():
    assert m.COLORS[m.BOTH][0][1] > m.COLORS[m.BOTH][0][0]  # green: G channel dominates
    for bright, dark in m.COLORS.values():
        assert sum(bright) > sum(dark)  # bright really is brighter
    assert (255, 220, 0) not in [c for shades in m.COLORS.values() for c in shades]


def test_event_window_and_vertex_disk():
    rgb = np.full((8, 12, 12, 3), 90, dtype=np.uint8)
    out = m.render_membership(
        rgb, two_cell_frames(),
        # cd = "1;2" too: the only junction that exists in the synthetic labels
        [(m.BOTH, row(frame_start=3, frame_end=4, cd="1;2", x=2.0, y=9.0))],
        pre=1, post=1, reach=1,
    )
    bright, dark = m.COLORS[m.BOTH]
    assert painted(bright, out[2]) and painted(dark, out[5])  # before bright, after dark
    for c in (bright, dark):
        assert not painted(c, out[0]) and not painted(c, out[7])
    assert tuple(out[3][9, 2]) == bright  # bright disk at (y=9, x=2) during the event


def test_comparable_settings_check():
    same = {"labels": "a", "expand": 3, "border_rings": 1}
    assert m.same_settings(same, dict(same)) == []
    assert m.same_settings(same, {**same, "expand": 2}) == ["expand"]
