"""Test-only script analysis/t1_topology_test.py: the topology rule and class colours."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
import t1_topology_test as m  # noqa: E402

PB, PA = frozenset((1, 2)), frozenset((3, 4))  # A-B breaks, C-D builds


def sequence(states, drop_side=None, absent=None):
    """adj/present per frame from a state string: B A-B touch, A C-D touch, X both, O neither."""
    adj, present = [], []
    for t, s in enumerate(states):
        cells = {1, 2, 3, 4} - ({3} if absent == t else set())
        g = {c: set() for c in cells}

        def link(a, b):
            if a in cells and b in cells:
                g[a].add(b)
                g[b].add(a)

        for a, b in ((1, 3), (1, 4), (2, 3), (2, 4)):  # the four sides
            if drop_side != (t, (a, b)):
                link(a, b)
        if s in "BX":
            link(1, 2)
        if s in "AX":
            link(3, 4)
        adj.append(g)
        present.append(cells)
    return adj, present


def check(states, t0, t1, k=2, **kw):
    adj, present = sequence(states, **kw)
    return m.topology_check(PB, PA, t0, t1, adj, present, k)


def test_clean_swap_only_passes():
    assert check("BBBAAA", 2, 2) == (True, "ok")  # swap-only: t_start = t_end = last B frame


def test_clean_four_way_zone_passes():
    assert check("BBXXAA", 2, 3) == (True, "ok")


def test_one_frame_before_state_fails():
    assert check("BABAAA", 2, 2) == (False, "short_before")


def test_one_frame_after_state_fails():
    assert check("BBBAB", 2, 2) == (False, "short_after")


def test_flicker_inside_the_zone_is_allowed_when_the_ends_are_stable():
    assert check("BBBABAAA", 2, 4) == (True, "ok")


def test_side_contact_lost_fails():
    assert check("BBBAAA", 2, 2, drop_side=(3, (1, 3))) == (False, "side_lost")


def test_cell_missing_in_the_zone_fails():
    assert check("BBXXAA", 2, 3, absent=2) == (False, "cell_missing")


def test_one_frame_window_accepts_what_detection_finds():
    assert check("BA", 0, 0, k=1) == (True, "ok")
    assert check("BABAAA", 2, 2, k=1) == (True, "ok")


def test_not_four_cells_rejected():
    adj, present = sequence("BBBAAA")
    assert m.topology_check(frozenset((1, 2)), frozenset((2, 3)), 2, 2, adj, present)[1] == "not_four_cells"


def test_class_from_new_and_strict():
    assert m.classify(False, False) == m.BLUE
    assert m.classify(False, True) == m.RED
    assert m.classify(True, False) == m.ORANGE
    assert m.classify(True, True) == m.GREEN


def test_painted_class_colours_bright_before_dark_after():
    labels = np.zeros((8, 12, 12), dtype=np.int32)
    labels[:, :, :6], labels[:, :, 6:] = 1, 2  # cells 1 and 2 touch
    rgb = np.full((8, 12, 12, 3), 90, dtype=np.uint8)
    ev = pd.Series({"pair_breaking": "1;2", "pair_building": "1;2", "frame_start": 3,
                    "frame_end": 3, "x": 2.0, "y": 9.0})
    painted = lambda c, f: bool((f == np.array(c, dtype=np.uint8)).all(-1).any())
    for cls, (bright, dark) in m.COLORS.items():
        out = m.render_classes(rgb, labels, [(cls, ev)], pre=1, post=1, reach=1)
        assert painted(bright, out[2]) and not painted(dark, out[2])  # before: bright
        assert painted(dark, out[4]) and not painted(bright, out[4])  # after: dark
        for other, shades in m.COLORS.items():
            if other != cls:
                assert not any(painted(c, out[2]) or painted(c, out[4]) for c in shades)


def test_base_mode_has_two_distinct_shades_and_paints_them():
    assert set(m.COLORS_BASE) == {m.KEPT, m.DROPPED}
    for bright, dark in m.COLORS_BASE.values():
        assert sum(bright) > sum(dark)
    four = [c for shades in m.COLORS.values() for c in shades]
    assert not any(c in four for shades in m.COLORS_BASE.values() for c in shades)  # new scheme
    labels = np.zeros((8, 12, 12), dtype=np.int32)
    labels[:, :, :6], labels[:, :, 6:] = 1, 2
    rgb = np.full((8, 12, 12, 3), 90, dtype=np.uint8)
    ev = pd.Series({"pair_breaking": "1;2", "pair_building": "1;2", "frame_start": 3,
                    "frame_end": 3, "x": 2.0, "y": 9.0})
    painted = lambda c, f: bool((f == np.array(c, dtype=np.uint8)).all(-1).any())
    for cls, (bright, dark) in m.COLORS_BASE.items():
        out = m.render_classes(rgb, labels, [(cls, ev)], pre=1, post=1, reach=1,
                               colors=m.COLORS_BASE, order=m.ORDER_BASE)
        assert painted(bright, out[2]) and not painted(dark, out[2])
        assert painted(dark, out[4]) and not painted(bright, out[4])


def test_outline_background_is_black_with_thin_cell_outlines():
    labels = np.zeros((2, 12, 12), dtype=np.int32)
    labels[:, 2:10, 2:10] = 1  # one square cell
    bg = m.outline_stack(labels)
    assert bg.shape == (2, 12, 12, 3) and bg.dtype == np.uint8
    on = (bg[0] != 0).any(-1)
    assert on[2, 2:10].all() and on[9, 2:10].all() and on[2:10, 2].all() and on[2:10, 9].all()  # edge
    assert not on[3:9, 3:9].any()  # inside the cell stays black
    assert not on[0, :].any() and not on[:, 0].any()  # outside stays black
    assert tuple(bg[0][2, 5]) == m.OUTLINE_GREY


def test_render_on_outline_background_keeps_grey_shades_out():
    labels = np.zeros((8, 12, 12), dtype=np.int32)
    labels[:, :, :6], labels[:, :, 6:] = 1, 2
    ev = pd.Series({"pair_breaking": "1;2", "pair_building": "1;2", "frame_start": 3,
                    "frame_end": 3, "x": 2.0, "y": 9.0})
    out = m.render_classes(None, labels, [(m.KEPT, ev)], pre=1, post=1, reach=1,
                           colors=m.COLORS_BASE, order=m.ORDER_BASE,
                           background=m.outline_stack(labels))
    shades = {tuple(p) for p in out.reshape(-1, 3)}
    allowed = {(0, 0, 0), m.OUTLINE_GREY, *m.COLORS_BASE[m.KEPT]}
    assert shades <= allowed, shades - allowed  # only black, outline grey and the two cyan shades
    bright, dark = m.COLORS_BASE[m.KEPT]
    assert (out[2] == np.array(bright, dtype=np.uint8)).all(-1).any()
    assert (out[4] == np.array(dark, dtype=np.uint8)).all(-1).any()
