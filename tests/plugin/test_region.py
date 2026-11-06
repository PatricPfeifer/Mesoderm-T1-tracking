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
