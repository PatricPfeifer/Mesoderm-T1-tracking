import numpy as np

from napari_tissuekit.core.holes import fill_enclosed_holes


def _frame():
    lab = np.zeros((60, 60), dtype=np.uint16)
    lab[5:55, 5:25] = 1
    lab[5:55, 35:55] = 2
    lab[5:15, 25:35] = 3
    lab[45:55, 25:35] = 4
    return lab  # the gap lab[15:45, 25:35] is enclosed by 1, 2, 3, 4


def test_enclosed_hole_becomes_a_cell():
    lab = _frame()
    out, n = fill_enclosed_holes(lab)
    assert n == 1
    assert (out[15:45, 25:35] == 5).all()
    assert (out[lab > 0] == lab[lab > 0]).all()  # existing cells unchanged
    assert (lab[15:45, 25:35] == 0).all()  # input untouched


def test_region_reaching_the_image_edge_is_not_filled():
    lab = _frame()
    lab[45:55, 25:35] = 0  # now the gap runs off the bottom of the cells ...
    lab[55:, :] = 0
    lab[:, :5] = 0
    out, n = fill_enclosed_holes(lab)  # ... and out to the image edge
    assert n == 0 and (out == lab).all()


def test_thin_gap_and_tiny_hole_stay_empty():
    lab = np.zeros((40, 40), dtype=np.uint16)
    lab[5:35, 5:19] = 1
    lab[5:35, 21:35] = 2  # 2 px wide membrane gap between cells, enclosed by nothing else
    lab[5:35, 19:21] = 0
    lab[5:35, 5] = 0
    out, n = fill_enclosed_holes(lab)
    assert n == 0
    lab2 = np.zeros((40, 40), dtype=np.uint16)
    lab2[5:35, 5:35] = 1
    lab2[18:21, 18:21] = 0  # 9 px, under the minimal cell size
    assert fill_enclosed_holes(lab2)[1] == 0


def test_stack_and_max_area():
    stack = np.stack([_frame(), _frame()])
    out, n = fill_enclosed_holes(stack)
    assert n == 2 and out.shape == stack.shape
    assert fill_enclosed_holes(stack, max_area=100)[1] == 0


def _hole_stack():
    """6 frames of one big cell with carved-out empty areas: a square (a cell), a thin ribbon in frames 0-2 and another in frames 3-4,
    and a thin U that fails both shape rules in every frame."""
    st = np.ones((6, 160, 160), np.uint16)
    st[:, 20:40, 20:40] = 0                      # square, 400 px, radius 10: passes
    st[0:3, 100:106, 20:80] = 0                  # ribbon 6 px wide: radius 3 (< 5), solid: fails the radius rule only, lasts 3 frames
    st[3:5, 130:136, 20:80] = 0                  # same ribbon elsewhere, only 2 frames
    st[:, 20:80, 100:106] = 0                    # U: two legs and a bottom, thin and far from convex: fails both
    st[:, 20:80, 130:136] = 0
    st[:, 74:80, 100:136] = 0
    return st


def test_fill_holes_stack_cells_persistence_and_absorption():
    from napari_tissuekit.core.holes import fill_holes_stack

    out, rep = fill_holes_stack(_hole_stack(), min_area=100, min_radius=5, min_solidity=0.6, min_frames=3)
    assert rep == dict(candidates=17, cells=6, persistent=3, absorbed_both=6, absorbed_short=2)
    for t in range(6):
        sq = out[t, 20:40, 20:40]
        assert sq.min() == sq.max() and sq[0, 0] not in (0, 1)       # the square is a cell of its own in every frame
        assert (out[t, 20:80, 100:106] == 1).all() and (out[t, 74:80, 100:136] == 1).all()  # the U is taken over by the surrounding cell
    for t in range(3):
        rib = out[t, 100:106, 20:80]
        assert rib.min() == rib.max() and rib[0, 0] not in (0, 1)    # the ribbon lasts 3 frames: a cell
    for t in (3, 4):
        assert (out[t, 130:136, 20:80] == 1).all()                  # the same kind of ribbon for 2 frames: taken over
    assert not (out == 0).any()                                     # nothing is left empty


def test_fill_holes_stack_in_one_frame_a_single_rule_failure_is_taken_over():
    from napari_tissuekit.core.holes import fill_holes_stack

    out, rep = fill_holes_stack(_hole_stack()[0], min_area=100, min_radius=5)
    assert rep["cells"] == 1 and rep["persistent"] == 0 and rep["absorbed_short"] == 1 and rep["absorbed_both"] == 1
    assert (out[100:106, 20:80] == 1).all()


def _tracked_frames():
    """3 frames, cells 1-4 in the corners/sides; cell 5 in the middle in frames 0 and 2. In frame 1 cell 5 is not detected: an empty gap."""
    st = np.zeros((3, 60, 60), np.uint16)
    for t in range(3):
        st[t, :, :20] = 1
        st[t, :, 40:] = 2
        st[t, :20, 20:40] = 3
        st[t, 40:, 20:40] = 4
        st[t, 20:40, 20:40] = 5
    st[1, 20:40, 20:40] = 0                       # missed cell: an enclosed gap (nothing touches the image edge)
    # a second spot, on the border between cells 1 and 3 (their corner) in every frame: covered by two cells before and after
    st[:, 5:9, 17:23] = 0
    st[0, 5:9, 17:20] = 1; st[0, 5:9, 20:23] = 3   # frame 0 and 2: half cell 1, half cell 3
    st[2, 5:9, 17:20] = 1; st[2, 5:9, 20:23] = 3
    return st


def test_repair_gaps_restores_a_missed_cell_and_closes_up_the_rest():
    from napari_tissuekit.core.holes import repair_gaps

    out, rep = repair_gaps(_tracked_frames())
    assert rep == dict(gaps=2, restored=1, absorbed=1)
    assert (out[1, 20:40, 20:40] == 5).all()                       # the same cell before and after: the gap is cell 5 again
    assert set(np.unique(out[1, 5:9, 17:23]).tolist()) <= {1, 3}  # covered by two cells: the neighbours take it over
    assert not (out == 0).any() or (out[1] == 0).sum() == 0       # nothing is left empty in the repaired frame


def test_repair_gaps_in_the_last_frame_needs_one_cell_to_cover_most_of_it():
    from napari_tissuekit.core.holes import repair_gaps

    st = _tracked_frames()[:2]                                      # frame 1 is now the last: only frame 0 can vote
    out, rep = repair_gaps(st)
    assert rep["restored"] == 1 and (out[1, 20:40, 20:40] == 5).all()
