"""Analysis region: boundary lines of cells, keep/drop areas, per-frame membership, events, drawing."""
import numpy as np
import pytest

from napari_tissuekit.core.region import (
    Anchor, Geometry, Pick, Region, TRACKING_REQUIRED, burn_region, event_inside, geometry_for, has_tracking,
    keep_stack, positions_from_stack, region_hash,
)

SHAPE = (130, 130)


def grid_positions(n_frames=1, drift=(0.0, 0.0)):
    """Cell id = 10*row + col + 1 on a 6x6 grid (spacing 20, offset 10): y = 10 + 20*row, x = 10 + 20*col."""
    pos = []
    for f in range(n_frames):
        pos.append({10 * r + c + 1: (10 + 20 * r + drift[0] * f, 10 + 20 * c + drift[1] * f)
                    for r in range(6) for c in range(6)})
    return pos


def pick(tid, pos, f=0):
    y, x = pos[f][tid]
    return Pick(tid, f, y, x)


def anchor(tid, pos, mode="keep", f=0):
    y, x = pos[f][tid]
    return Anchor(tid, f, y, x, mode)


def row_cells(r):  # ids of grid row r
    return [10 * r + c + 1 for c in range(6)]


def col_cells(c):
    return [10 * r + c + 1 for r in range(6)]


def inside(geom, f=0):
    return geom.inside_sets()[f]


def test_horizontal_line_keeps_one_side():
    pos = grid_positions()
    # the line goes through row 2 shifted down by half a spacing: use cells of row 2 (y=50) and put the line there
    region = Region(SHAPE, [[pick(i, pos) for i in (row_cells(2)[0], row_cells(2)[3], row_cells(2)[5])]],
                    [anchor(1, pos)])  # cell 1 = top-left
    ins = inside(Geometry(region, pos))
    assert all(i in ins for r in (0, 1) for i in row_cells(r))
    assert not any(i in ins for r in (3, 4, 5) for i in row_cells(r))


def test_disregard_mode_is_the_complement():
    pos = grid_positions()
    line = [pick(i, pos) for i in row_cells(2)]
    keep = inside(Geometry(Region(SHAPE, [line], [anchor(1, pos, "keep")]), pos))
    drop = inside(Geometry(Region(SHAPE, [line], [anchor(1, pos, "drop")]), pos))
    on_line = set(row_cells(2))
    assert (keep | drop) >= set(pos[0]) - on_line
    assert not (keep & drop) - on_line


def test_horizontal_and_vertical_line_keep_one_quadrant():
    pos = grid_positions()
    h = [pick(i, pos) for i in row_cells(2)]
    v = [pick(i, pos) for i in col_cells(2)]
    ins = inside(Geometry(Region(SHAPE, [h, v], [anchor(1, pos)]), pos))  # top-left quadrant
    want = {10 * r + c + 1 for r in (0, 1) for c in (0, 1)}
    on_lines = set(row_cells(2)) | set(col_cells(2))
    assert want <= ins
    assert (ins - on_lines) == want
    # keep two quadrants (union of areas)
    ins2 = inside(Geometry(Region(SHAPE, [h, v], [anchor(1, pos), anchor(10 * 4 + 6, pos)]), pos))
    assert {10 * 4 + 5 + 1, 1} <= ins2 and not ({10 * 0 + 5 + 1, 10 * 4 + 1} & ins2)


def test_three_lines_flexible():
    pos = grid_positions()
    h = [pick(i, pos) for i in row_cells(2)]
    v1 = [pick(i, pos) for i in col_cells(1)]
    v2 = [pick(i, pos) for i in col_cells(4)]
    # keep the cell block between the two vertical lines and above the horizontal one
    ins = inside(Geometry(Region(SHAPE, [h, v1, v2], [anchor(10 * 0 + 3, pos)]), pos))
    on_lines = set(row_cells(2)) | set(col_cells(1)) | set(col_cells(4))
    assert ins - on_lines == {1 * 0 + c + 1 + 10 * r for r in (0, 1) for c in (2, 3)}


def test_curved_line_follows_the_cells():
    pos = grid_positions()
    # a V-shaped line: cells (row 3,col 0) (row 4,col 2) (row 3,col 4)... keep everything above it
    line = [pick(10 * 3 + 0 + 1, pos), pick(10 * 4 + 2 + 1, pos), pick(10 * 3 + 5 + 1, pos)]
    ins = inside(Geometry(Region(SHAPE, [line], [anchor(1, pos)]), pos))
    assert all(i in ins for i in row_cells(0)) and all(i in ins for i in row_cells(2))
    assert 10 * 5 + 3 not in ins and 10 * 5 + 1 not in ins  # bottom row is on the other side


def test_line_moves_with_the_tissue_and_cells_stay_in_or_out():
    pos = grid_positions(n_frames=6, drift=(4.0, 1.0))
    line = [pick(i, pos) for i in row_cells(2)]
    geom = Geometry(Region((200, 200), [line], [anchor(1, pos)]), pos)
    sets = geom.inside_sets()
    on_line = set(row_cells(2))
    assert all(s - on_line == sets[0] - on_line for s in sets)
    assert geom.line_polyline(0, 5)[0][1] > geom.line_polyline(0, 0)[0][1] + 15  # the line moved down


def test_line_cell_lost_in_division_uses_daughters():
    pos = grid_positions(n_frames=4)
    mother = row_cells(2)[2]
    for f in (2, 3):
        y, x = pos[f].pop(mother)
        pos[f][901], pos[f][902] = (y, x - 3), (y, x + 3)
    line = [pick(i, pos) for i in row_cells(2)]
    geom = Geometry(Region(SHAPE, [line], [anchor(1, pos)]), pos, parents={901: mother, 902: mother})
    pts = geom.line_polyline(0, 3)
    assert np.isclose(pts[:, 1].mean(), 50.0, atol=0.5)  # still a horizontal line at y = 50
    without = Geometry(Region(SHAPE, [line], [anchor(1, pos)]), pos)  # no lineage: that cell is skipped, line still there
    assert len(without.line_polyline(0, 3)) == len(pts) - 1


def test_line_with_too_few_cells_is_refused():
    pos = grid_positions()
    with pytest.raises(ValueError):
        Geometry(Region(SHAPE, [[pick(1, pos)]], [anchor(1, pos)]), pos)


def test_keep_mask_matches_cell_membership_and_burn_draws_yellow():
    pos = grid_positions()
    region = Region(SHAPE, [[pick(i, pos) for i in row_cells(2)]], [anchor(1, pos)])
    geom = Geometry(region, pos)
    mask = geom.keep_mask(0)
    assert mask[10, 10] and not mask[110, 10]
    assert abs(mask.mean() - 50 / 130) < 0.01  # kept side = the part above the line at y = 50
    rgb = np.full((1, *SHAPE, 3), 200, np.uint8)
    out = burn_region(rgb, geom)
    assert (out[0, 110, 10] < 200).all()  # outside dimmed
    assert (out[0, 10, 10] == 200).all()  # inside untouched
    yellow = (out[0] == (255, 255, 0)).all(axis=-1)
    assert yellow.any() and abs(np.nonzero(yellow)[0].mean() - 50) < 4  # outline sits on the line at y = 50
    assert keep_stack(geom).shape == (1, *SHAPE)


def test_event_inside_needs_all_four_cells():
    inside_sets = [{1, 2, 3, 4}, {1, 2, 3}]
    ev = {"quartet": frozenset({1, 2, 3, 4}), "t_first_swap": 0}
    assert event_inside(ev, inside_sets)
    assert not event_inside({**ev, "t_first_swap": 1}, inside_sets)
    assert event_inside({"A": 1, "B": 2, "C": 3, "D": 4, "frame_first_swap": 0}, inside_sets)


def test_hash_json_roundtrip_and_tracking_gate(tmp_path):
    pos = grid_positions()
    region = Region(SHAPE, [[pick(i, pos) for i in row_cells(2)]], [anchor(1, pos)])
    path = region.save(tmp_path / "analysis_region" / "region.json")
    again = Region.load(path)
    assert again.to_json() == region.to_json()
    assert region_hash(region) == region_hash(again) != ""
    other = Region(SHAPE, region.lines, [anchor(1, pos, "drop")])
    assert region_hash(other) != region_hash(region)
    assert region_hash(Region(SHAPE)) == "" and region_hash(None) == ""
    assert not has_tracking(None) and not has_tracking(np.zeros((2, 5, 5))) and has_tracking(np.ones((2, 5, 5)))
    assert "Tracking is required" in TRACKING_REQUIRED


def test_positions_from_stack_and_geometry_for():
    stack = np.zeros((2, 40, 40), np.uint32)
    stack[:, 5:10, 5:10] = 1
    stack[:, 5:10, 30:35] = 2
    stack[:, 30:35, 5:10] = 3
    stack[:, 30:35, 30:35] = 4
    pos = positions_from_stack(stack)
    assert pos[0][1] == (7.0, 7.0) and pos[1][4] == (32.0, 32.0)
    region = Region((40, 40), [[Pick(3, 0, 32, 7), Pick(4, 0, 32, 32)]], [Anchor(1, 0, 7, 7, "keep")])
    ins = geometry_for(region, stack).inside_sets()
    assert ins[0] == {1, 2}


def test_cell_properties_only_lists_cells_inside_per_frame():
    from napari_tissuekit.cell_properties import compute_properties

    stack = np.zeros((2, 40, 40), np.uint32)
    stack[:, 5:10, 5:10] = 1
    stack[:, 25:30, 25:30] = 2
    everyone = compute_properties(stack, [1, 2], ["area"])
    assert len(everyone) == 4
    only = compute_properties(stack, [1, 2], ["area"], inside=[{1, 2}, {1}])
    assert {(r.track_id, r.frame) for r in only.itertuples()} == {(1, 0), (2, 0), (1, 1)} and len(only) == 3


# ---- straight best-fit lines ---------------------------------------------------------------------------------------
from napari_tissuekit.core.region import STRAIGHT, fit_line  # noqa: E402


def straight(lines, anchors, pos, shape=SHAPE):
    return Geometry(Region(shape, lines, anchors, STRAIGHT), pos)


def jitter_positions(seed=0, amp=3.0, n_frames=1):
    """The grid with every cell moved by up to `amp` pixels (rows are then only roughly straight)."""
    rng = np.random.default_rng(seed)
    base = grid_positions(n_frames)
    return [{k: (y + rng.uniform(-amp, amp), x + rng.uniform(-amp, amp)) for k, (y, x) in f.items()} for f in base]


def test_fit_line_is_the_least_squares_line_and_needs_two_points():
    c, d = fit_line(np.array([[0, 0], [10, 1], [20, -1], [30, 0.0]]))  # (x, y), nearly horizontal
    assert abs(d[0]) > 0.99 and abs(c[1]) < 1e-9
    assert fit_line(np.array([[5.0, 5.0]])) is None and fit_line(np.array([[5.0, 5.0], [5.0, 5.0]])) is None
    c, d = fit_line(np.array([[3.0, 4.0], [3.0, 40.0]]))  # exactly vertical
    assert abs(d[0]) < 1e-9 and abs(abs(d[1]) - 1) < 1e-9


def test_straight_line_ignores_a_wobbly_row_and_is_two_points():
    pos = jitter_positions()
    geom = straight([[pick(i, pos) for i in row_cells(2)]], [anchor(1, pos)], pos)
    seg = geom.line_polyline(0, 0)
    assert seg.shape == (2, 2)  # one straight segment, border to border
    assert abs(seg[0, 1] - 50) < 4 and abs(seg[1, 1] - 50) < 4 and {seg[0, 0], seg[1, 0]} == {0.0, 130.0}
    ins = geom.inside_sets()[0]
    assert all(i in ins for r in (0, 1) for i in row_cells(r)) and not any(i in ins for r in (4, 5) for i in row_cells(r))
    # the polyline version bends through every wobbly cell, the straight one does not
    poly = Geometry(Region(SHAPE, [[pick(i, pos) for i in row_cells(2)]], [anchor(1, pos)], "polyline"), pos)
    assert len(poly.line_polyline(0, 0)) > 2


def test_two_straight_lines_horizontal_and_vertical_cut_four_corners():
    pos = jitter_positions(seed=1)
    h = [pick(i, pos) for i in row_cells(2)]
    v = [pick(i, pos) for i in col_cells(2)]
    keep_tl = straight([h, v], [anchor(1, pos)], pos).inside_sets()[0]
    want = {10 * r + c + 1 for r in (0, 1) for c in (0, 1)}
    assert want <= keep_tl
    assert not (keep_tl & {10 * r + c + 1 for r in (3, 4, 5) for c in range(6)})
    assert not (keep_tl & {10 * r + c + 1 for r in range(6) for c in (3, 4, 5)})


def test_straight_line_follows_a_moving_and_rotating_tissue_with_stable_sides():
    n = 8
    pos = []
    for f in range(n):  # drift down and rotate by 3 degrees per frame around the centre
        a = np.deg2rad(3.0 * f)
        pos.append({k: (65 + (y - 65) * np.cos(a) + (x - 65) * np.sin(a) + 2.0 * f,
                        65 - (y - 65) * np.sin(a) + (x - 65) * np.cos(a)) for k, (y, x) in grid_positions()[0].items()})
    geom = straight([[pick(i, pos) for i in row_cells(2)]], [anchor(1, pos)], pos, (200, 200))
    sets = geom.inside_sets()
    on_line = set(row_cells(2))
    assert all(s - on_line == sets[0] - on_line for s in sets)  # the same cells stay on the kept side
    dirs = [np.diff(geom.line_polyline(0, f), axis=0)[0] for f in range(n)]
    assert all(dirs[f] @ dirs[f + 1] > 0 for f in range(n - 1))  # direction never flips (side 0 stays side 0)


def test_straight_mode_matches_the_polygon_signatures_for_a_two_cell_line():
    pos = grid_positions()
    line = [pick(4 + 10 * 2 + 1, pos), pick(10 * 3 + 1, pos)]  # a tilted line through two cells
    reg = [[line, [anchor(1, pos)]]][0]
    a = Geometry(Region(SHAPE, [line], [anchor(1, pos)], STRAIGHT), pos)
    b = Geometry(Region(SHAPE, [line], [anchor(1, pos)], "polyline"), pos)
    on_line = {line[0].track_id, line[1].track_id}
    assert a.inside_sets()[0] - on_line == b.inside_sets()[0] - on_line
    ma, mb = a.keep_mask(0), b.keep_mask(0)
    assert (ma != mb).mean() < 0.01  # same area up to pixels on the line itself
    assert reg


def test_inside_ids_equals_inside_sets_and_masks_are_cached():
    pos = grid_positions(n_frames=4, drift=(2.0, 1.0))
    geom = straight([[pick(i, pos) for i in row_cells(2)]], [anchor(1, pos)], pos, (200, 200))
    sets = geom.inside_sets()
    assert all(geom.inside_ids(f) == sets[f] for f in range(4))
    m = geom.keep_mask(2)
    assert geom.keep_mask(2) is m  # second call comes from the cache
    expect = np.isin(geom.signature_image(2), sorted(geom.kept))
    assert (m == expect).all()


def test_line_mode_roundtrip_and_old_regions_stay_polylines(tmp_path):
    pos = grid_positions()
    line = [pick(i, pos) for i in row_cells(2)]
    old = Region(SHAPE, [line], [anchor(1, pos)])  # default: polyline
    new = Region(SHAPE, [line], [anchor(1, pos)], STRAIGHT)
    assert old.line_mode == "polyline" and Region.load(new.save(tmp_path / "r.json")).line_mode == STRAIGHT
    legacy = old.to_json()
    legacy.pop("line_mode")
    assert Region.from_json(legacy).line_mode == "polyline"  # a file written before the option existed
    assert region_hash(old) != region_hash(new) != ""


def test_positions_from_stack_with_uint64_and_empty_frames():
    stack = np.zeros((3, 20, 20), np.uint64)
    stack[0, 2:6, 2:6] = 7
    stack[2, 10:14, 4:8] = 9
    pos = positions_from_stack(stack)
    assert pos[0] == {7: (3.5, 3.5)} and pos[1] == {} and pos[2] == {9: (11.5, 5.5)}


def test_clicked_boundary_cells_are_not_analysed():
    pos = grid_positions()
    for mode in (STRAIGHT, "polyline"):
        geom = Geometry(Region(SHAPE, [[pick(i, pos) for i in row_cells(2)]], [anchor(1, pos)], mode), pos)
        ins = geom.inside_sets()[0]
        assert not (ins & set(row_cells(2)))  # the line's own cells are left out
        assert {10 * r + c + 1 for r in (0, 1) for c in range(6)} <= ins  # the cells above stay in


def test_daughters_of_a_boundary_cell_are_left_out_too():
    pos = grid_positions(n_frames=3)
    mother = row_cells(2)[2]
    for f in (1, 2):
        y, x = pos[f].pop(mother)
        pos[f][901], pos[f][902] = (y, x - 3), (y, x + 3)
    geom = Geometry(Region(SHAPE, [[pick(i, pos) for i in row_cells(2)]], [anchor(1, pos)], STRAIGHT), pos,
                    parents={901: mother, 902: mother})
    assert not ({901, 902} & geom.inside_ids(2)) and not ({901, 902} & geom.inside_ids(1))
