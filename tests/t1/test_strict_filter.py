"""Strict T1 filter: same four cells, AB/CD end vertices on both sides."""
from napari_tissuekit.core.t1.run_tm_t1 import strict_ok

QUARTET = frozenset((1, 2, 3, 4))
AB = frozenset((1, 2))
CD = frozenset((3, 4))


def vert(cells, y=0.0, x=0.0):
    return {"cell_ids": tuple(sorted(cells)), "y": y, "x": x}


def ab_frame():
    """AB junction: ends {A,B,C} and {A,B,D}."""
    return [vert((1, 2, 3), 0, 0), vert((1, 2, 4), 0, 5)]


def cd_frame():
    """CD junction: ends {C,D,A} and {C,D,B}."""
    return [vert((3, 4, 1), 0, 0), vert((3, 4, 2), 5, 0)]


def make_event(t0=2, t1=2, persist_pre=1, persist_post=1):
    return {
        "t_start": t0, "t_end": t1, "quartet": QUARTET,
        "pair_before": AB, "pair_after": CD,
        "persist_pre": persist_pre, "persist_post": persist_post,
    }


def make_verts(before, after, n=5, t0=2, t1=2):
    verts = [[] for _ in range(n)]
    verts[t0 - 1] = before
    verts[t1 + 1] = after
    return verts


def test_correct_geometry_kept():
    assert strict_ok(make_event(), make_verts(ab_frame(), cd_frame()), 1)


def test_missing_end_vertex_before_dropped():
    before = ab_frame()[:1]
    assert not strict_ok(make_event(), make_verts(before, cd_frame()), 1)


def test_missing_end_vertex_after_dropped():
    after = cd_frame()[:1]
    assert not strict_ok(make_event(), make_verts(ab_frame(), after), 1)


def test_ends_not_covering_all_four_cells_dropped():
    # both ends touch C, none touches D: C,D are not on opposite ends of AB
    before = [vert((1, 2, 3), 0, 0), vert((1, 2, 3), 0, 5)]
    assert not strict_ok(make_event(), make_verts(before, cd_frame()), 1)


def test_wrong_pair_on_far_side_dropped():
    # frame after shows AB junction again instead of CD
    assert not strict_ok(make_event(), make_verts(ab_frame(), ab_frame()), 1)


def test_persistence_below_n_dropped():
    verts = make_verts(ab_frame(), cd_frame())
    assert not strict_ok(make_event(persist_pre=0), verts, 1)
    assert not strict_ok(make_event(persist_post=0), verts, 1)


def test_needs_n_frames_each_side():
    ev = make_event(t0=3, t1=3, persist_pre=2, persist_post=2)
    verts = [[] for _ in range(7)]
    verts[2], verts[1] = ab_frame(), ab_frame()
    verts[4], verts[5] = cd_frame(), cd_frame()
    assert strict_ok(ev, verts, 2)
    verts[1] = []
    assert not strict_ok(ev, verts, 2)


def test_movie_edge_dropped():
    assert not strict_ok(make_event(t0=0, t1=0), make_verts(ab_frame(), cd_frame(), t0=1), 1)


def test_zero_disables():
    assert strict_ok(make_event(persist_pre=0), [[] for _ in range(5)], 0)
