"""State-machine T1 classification: one synthetic length series per class plus the invariants of the specification."""
import numpy as np
import pytest

from napari_tissuekit.core.t1.t1_states import (
    StateParams,
    absorb_short,
    assign_states,
    apply_side_rule,
    classify_quartet,
    hysteresis,
    seconds_to_frames,
)

P = StateParams(l_on=2.0, l_off=1.0, n_dwell=2, n_stable=4)


def series(spec):
    """'O5 V1 N5' -> (len_x, len_y): O has x=5,y=0; N has x=0,y=5; V has both 0; B (both) has 5,5; x absent (NaN)."""
    lx, ly = [], []
    for tok in spec.split():
        k, n = tok[0], int(tok[1:])
        for _ in range(n):
            a, b = {"O": (5.0, 0.0), "N": (0.0, 5.0), "V": (0.0, 0.0), "B": (5.0, 5.0), "x": (np.nan, np.nan)}[k]
            lx.append(a)
            ly.append(b)
    return np.array(lx), np.array(ly)


def klass(spec, p=P):
    ev, _, _ = classify_quartet(*series(spec), p)
    return [e.klass for e in ev]


def test_clean_direct_and_through_vertex_are_identical():
    assert klass("O6 N6") == ["clean"]
    assert klass("O6 V1 N6") == ["clean"]
    assert klass("O6 V3 N6") == ["clean"]  # V of any length is allowed, a missing V equally


def test_v_never_changes_flicker_classes():
    for v in ("", " V1", " V2"):
        assert klass(f"O6{v} N2{v} O2{v} N2{v} O1 N6") == klass("O6 N2 O2 N2 O1 N6")


def test_flicker_resolved_has_three_flips_and_ends_in_n():
    ev, _, _ = classify_quartet(*series("O6 N2 O2 N2 O2 N6"), StateParams(2.0, 1.0, 0, 4))
    assert [e.klass for e in ev] == ["flicker_resolved"]
    assert ev[0].flips == 5


def test_flicker_aborted_ends_in_stable_o():
    ev, _, _ = classify_quartet(*series("O6 N2 O6"), StateParams(2.0, 1.0, 0, 4))
    assert [e.klass for e in ev] == ["flicker_aborted"] and ev[0].flips == 2


def test_abortive_shrink_enters_v_and_returns():
    ev, _, _ = classify_quartet(*series("O6 V2 O6"), P)
    assert [e.klass for e in ev] == ["abortive_shrink"] and ev[0].flips == 0 and ev[0].v_frames == 2


def test_unresolved_at_end_of_movie_and_with_absent_cell():
    assert klass("O6 V2 N2") == ["unresolved"]
    assert klass("O6 V1 x2 V1 N6") == ["unresolved"]


def test_new_event_after_return_to_stable_o():
    # every stable change is its own event with its roles relabelled (a reverse swap is a clean swap too)
    assert klass("O6 N6 O6 N6") == ["clean", "clean", "clean"]
    assert klass("O6 V1 O6 V1 N6") == ["abortive_shrink", "clean"]


def test_short_excursion_absorbed_by_dwell():
    # a 1-frame N blip inside O is absorbed (dwell 2) -> no flip; with dwell 0 it is a flicker
    assert klass("O6 N1 O6") == []
    assert klass("O6 N1 O6", StateParams(2.0, 1.0, 0, 4)) == ["flicker_aborted"]


def test_absorb_into_v_when_surrounding_differ():
    st = np.array(list("OOOONOOOO"))
    assert "".join(absorb_short(st, 2)) == "OOOOOOOOO"
    st = np.array(list("OOOONVVV"))
    assert "".join(absorb_short(st, 2)) == "OOOOVVVV"
    assert "".join(absorb_short(np.array(list("OOOOONVNNN")), 2)) == "OOOOOVVNNN"


def test_hysteresis_prevents_chatter():
    x = np.array([0, 2.1, 1.5, 2.1, 1.2, 1.9, 0.9, 1.5, 2.2])
    assert hysteresis(x, 2.0, 1.0).tolist() == [False, True, True, True, True, True, False, False, True]
    with pytest.raises(ValueError):
        hysteresis(x, 1.0, 2.0)


def test_both_pairs_present_is_v_and_flagged():
    st, both = assign_states(*series("O3 B2 N3"), 2.0, 1.0)
    assert "".join(st) == "OOOVVNNN"
    assert both.tolist() == [False] * 3 + [True] * 2 + [False] * 3


def test_seconds_to_frames():
    assert seconds_to_frames(0, 60) == 0
    assert seconds_to_frames(90, 60) == 2 and seconds_to_frames(90, 38) == 3
    assert seconds_to_frames(60, 60) == 1 and seconds_to_frames(150, 38) == 4


def test_seconds_to_frames_tolerates_a_slightly_short_interval():
    # measured intervals of the 60 s movies are 59.998 / 59.992 / 59.997 s
    for dt in (59.998, 59.992, 59.997):
        assert seconds_to_frames(60, dt) == 1 and seconds_to_frames(120, dt) == 2 and seconds_to_frames(300, dt) == 5
        assert seconds_to_frames(90, dt) == 2 and seconds_to_frames(150, dt) == 3


# ---- group rule: an event ends when the four cells leave their group or a configuration persisted n_end frames
PG = StateParams(l_on=2.0, l_off=1.0, n_dwell=0, n_stable=2, n_end=6)


def gklass(spec, group_ok=None, p=PG):
    lx, ly = series(spec)
    ev, _, _ = classify_quartet(lx, ly, p, np.ones(len(lx), bool) if group_ok is None else np.asarray(group_ok, bool))
    return [(e.klass, e.flips, e.end_reason) for e in ev]


def test_group_rule_swap_back_after_a_persisting_configuration_is_a_new_event():
    # N lasts 6 frames (= n_end): the first swap is complete (clean); the return is a second, separate event
    assert gklass("O6 N6 O6") == [("clean", 1, "stable"), ("clean", 1, "stable")]


def test_group_rule_swap_back_before_n_end_is_flicker():
    assert gklass("O6 N3 O6") == [("flicker_aborted", 2, "stable")]
    assert gklass("O6 N3 O2 N6") == [("flicker_resolved", 3, "stable")]
    assert gklass("O6 N3 V1 O6") == [("flicker_aborted", 2, "stable")]  # V in between changes nothing


def test_group_rule_leaving_the_group_ends_the_event():
    ok = np.ones(9 + 6, bool)
    ok[9:] = False                                    # O6 N3, then a cell leaves
    assert gklass("O6 N3 V6", ok) == [("clean", 1, "group")]
    ok2 = np.ones(8 + 6, bool)
    ok2[8:] = False                                   # O6 V2, then a cell leaves: no swap seen
    assert gklass("O6 V2 V6", ok2) == [("abortive_shrink", 0, "group")]


def test_group_rule_event_still_running_at_the_end_of_the_movie_is_unresolved():
    assert gklass("O6 V2 N2") == [("unresolved", 1, "movie end")]


def test_seconds_to_frames_long_times_at_a_slightly_short_interval():
    for dt in (59.998, 59.992, 59.997):
        assert seconds_to_frames(480, dt) == 8 and seconds_to_frames(300, dt) == 5
    assert seconds_to_frames(480, 37.979) == 13  # Naima: 12.6 frames


# ---- neighbour-relationship (side cells) rule and the return-to-vertex clause -----------------------------------------------
FLICK = "O5 V1 N2 O2 N2 O2 N5"      # O 0-4, V 5, N 6-7, O 8-9, N 10-11, O 12-13, N 14-18: flips at 6, 8, 10, 12, 14


def side_ok_except(n, *bad):
    ok = np.ones(n, dtype=bool)
    ok[list(bad)] = False
    return ok


def run_side(spec, bad, p=P):
    lx, ly = series(spec)
    ev, _, _ = classify_quartet(lx, ly, p, side_ok=side_ok_except(len(lx), *bad))
    assert len(ev) == 1
    return ev[0]


def test_side_rule_is_off_by_default_and_all_valid_changes_nothing():
    ev, _, _ = classify_quartet(*series(FLICK), P)
    assert [e.klass for e in ev] == ["flicker_resolved"] and ev[0].flips == 5
    e = run_side(FLICK, [])
    assert e.klass == "flicker_resolved" and e.flips == 5 and not e.side_truncated and not e.side_dropped


def test_side_rule_first_flip_fails_drops_the_event():
    e = run_side(FLICK, [6])
    assert e.side_dropped and e.flips == 0


def test_side_rule_previous_state_frame_also_counts():
    e = run_side(FLICK, [4])          # last O frame before the first flip
    assert e.side_dropped
    e = run_side(FLICK, [7])          # last N frame before the second flip (frame 8)
    assert e.klass == "clean" and e.flips == 1 and e.end == 8 and e.side_truncated


def test_side_rule_failing_later_flip_ends_the_event_before_it():
    for bad_frame, klass_, flips, final in [(8, "clean", 1, "N"), (10, "flicker_aborted", 2, "O"),
                                            (12, "flicker_resolved", 3, "N"), (14, "flicker_aborted", 4, "O")]:
        e = run_side(FLICK, [bad_frame])
        assert e.klass == klass_ and e.flips == flips and e.final == final and e.end == bad_frame
        assert e.end_reason == "side cells" and len(e.flip_frames) == flips


def test_vertex_clause_needs_a_true_vertex_frame_after_the_first_flip():
    spec = "O5 V1 N2 V2 N5"                  # first flip at 6, V frames 8-9 before the stable N
    lx, ly = series(spec)
    ev, _, _ = classify_quartet(lx, ly, P)
    assert ev[0].klass == "clean"
    ev, _, _ = classify_quartet(lx, ly, P, side_ok_loff=np.zeros(len(lx), dtype=bool))      # V, but the four cells do not meet
    assert ev[0].klass == "clean" and not ev[0].vertex_flag
    vertex = np.zeros(len(lx), dtype=bool)
    vertex[8] = True
    ev, _, _ = classify_quartet(lx, ly, P, side_ok_loff=vertex)                          # a true four-cell vertex after the flip
    assert ev[0].klass == "flicker_resolved" and ev[0].vertex_flag
    vertex[:] = False
    vertex[5] = True                                                                    # the vertex BEFORE the first flip is normal
    ev, _, _ = classify_quartet(lx, ly, P, side_ok_loff=vertex)
    assert ev[0].klass == "clean"
