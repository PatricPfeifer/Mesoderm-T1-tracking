"""Outline edits are applied exactly as drawn (apply_strokes): untouched lines never move."""
import numpy as np
from scipy import ndimage as ndi
from skimage.draw import circle_perimeter, line
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

from napari_tissuekit.core.outline_io import hand_to_outline
from napari_tissuekit.core.ta_bond_edit import apply_bond_strokes, apply_strokes, thin_hand


def _tissue(d=16, seed=0, size=160):
    """Voronoi cells about ``d`` px across, with empty background around the tissue."""
    rng = np.random.default_rng(seed)
    g = np.arange(d // 2, size, d)
    yy, xx = np.meshgrid(g, g, indexing="ij")
    pts = np.clip(np.stack([yy.ravel(), xx.ravel()], 1) + rng.integers(-d // 4, d // 4 + 1, (yy.size, 2)), 0, size - 1)
    markers = np.zeros((size, size), np.int32)
    markers[pts[:, 0], pts[:, 1]] = np.arange(1, len(pts) + 1)
    lab = watershed(ndi.distance_transform_edt(markers == 0), markers).astype(np.uint16)
    lab[:8], lab[-8:], lab[:, :8], lab[:, -8:] = 0, 0, 0, 0
    return lab


def _cells(hand):
    return int(label(hand, connectivity=1, background=255).max())


def _loose_ends(hand):
    """Line pixels with at most one line pixel around them (open ends), away from the image border."""
    b = hand == 255
    nb = ndi.convolve(b.astype(int), np.ones((3, 3), int), mode="constant") - b
    inner = np.zeros_like(b)
    inner[1:-1, 1:-1] = True
    return int((b & (nb <= 1) & inner).sum())


def _paint(hand, rr, cc):
    """A painted stroke as the plugin sees it: pixels that were outline already are not new."""
    s = np.zeros(hand.shape, bool)
    ok = (rr >= 0) & (cc >= 0) & (rr < hand.shape[0]) & (cc < hand.shape[1])
    s[rr[ok], cc[ok]] = True
    return s & (hand_to_outline(hand) == 0)


def _middle_cell(hand, d):
    lab = label(hand, connectivity=1, background=255)
    c = np.array(hand.shape) / 2
    return min(regionprops(lab), key=lambda r: np.hypot(*(np.array(r.centroid) - c)))


def test_line_across_a_cell_splits_it_and_loose_ends_are_cut_off():
    hand = thin_hand(_tissue())
    r = _middle_cell(hand, 16)
    cy, cx = map(int, r.centroid)
    stroke = _paint(hand, *line(cy, cx - 16, cy, cx + 16))  # overshoots both membranes
    notes = []
    new = apply_strokes(hand, np.zeros_like(stroke), stroke, report=notes)
    assert _cells(new) == _cells(hand) + 1
    assert not ((hand == 255) & ~stroke & (new != 255)).any()  # no existing line moved
    assert _loose_ends(new) == _loose_ends(hand)  # the tails are gone
    assert notes == []


def test_random_lines_never_move_other_lines_or_leave_loose_ends():
    for d in (8, 12, 20):  # small cells too
        hand = thin_hand(_tissue(d, seed=d))
        ends = _loose_ends(hand)
        lab = label(hand, connectivity=1, background=255)
        rng = np.random.default_rng(d)
        for r in regionprops(lab)[::3]:
            y0, x0, y1, x1 = r.bbox
            if y0 < 20 or x0 < 20 or y1 > 140 or x1 > 140:
                continue
            cy, cx = r.coords[rng.integers(len(r.coords))]
            ang, length = rng.random() * np.pi, rng.uniform(0.6, 1.6) * d
            dy, dx = length * np.sin(ang), length * np.cos(ang)
            stroke = _paint(hand, *line(int(cy - dy), int(cx - dx), int(cy + dy), int(cx + dx)))
            notes = []
            new = apply_strokes(hand, np.zeros_like(stroke), stroke, report=notes)
            assert not ((hand == 255) & ~stroke & (new != 255)).any()
            if not notes:
                assert _loose_ends(new) == ends


def test_open_line_stays_as_drawn():
    hand = thin_hand(_tissue())
    cy, cx = map(int, _middle_cell(hand, 16).centroid)
    stroke = _paint(hand, *line(cy, cx - 2, cy, cx + 2))  # touches no line
    notes = []
    new = apply_strokes(hand, np.zeros_like(stroke), stroke, report=notes)
    assert _cells(new) == _cells(hand)
    assert (new[stroke] == 255).all()
    assert notes == [("open", int(stroke.sum()))]


def test_tiny_pieces_are_kept():
    hand = thin_hand(_tissue())
    r = _middle_cell(hand, 16)
    y0, x0, y1, x1 = r.bbox
    col = x0 + 2  # cuts a 1-2 px wide strip off the cell
    stroke = _paint(hand, *line(y0 - 3, col, y1 + 3, col))
    new = apply_strokes(hand, np.zeros_like(stroke), stroke)
    assert _cells(new) == _cells(hand) + 1


def test_closed_loop_around_a_junction_makes_one_cell_and_nothing_outside_moves():
    hand = thin_hand(_tissue())
    b = hand == 255
    junctions = np.argwhere(b & (ndi.convolve(b.astype(int), np.ones((3, 3), int)) >= 4))
    jy, jx = junctions[np.argmin(np.hypot(*(junctions - 80).T))]
    rr, cc = circle_perimeter(int(jy), int(jx), 5)
    ring = np.zeros(hand.shape, bool)
    ring[rr, cc] = True
    inside = ndi.binary_fill_holes(ring) & ~ring
    stroke = _paint(hand, rr, cc)
    notes = []
    new = apply_strokes(hand, np.zeros_like(stroke), stroke, report=notes)
    cells = label(new, connectivity=1, background=255)
    ids = np.unique(cells[inside])
    assert len(ids[ids > 0]) == 1  # one cell, no old line left inside the loop
    assert not ((hand != new) & ~ndi.binary_fill_holes(ring)).any()  # nothing outside the loop changed
    assert [kind for kind, _ in notes] == ["loop"]
    assert _loose_ends(new) == _loose_ends(hand)


def test_loop_with_a_small_gap_still_counts_as_closed():
    hand = thin_hand(_tissue())
    cy, cx = map(int, _middle_cell(hand, 16).centroid)
    rr, cc = circle_perimeter(cy, cx, 4)
    keep = ~((rr < cy) & (cc == cx))  # ends 2 px apart at the top
    notes = []
    new = apply_strokes(hand, np.zeros(hand.shape, bool), _paint(hand, rr[keep], cc[keep]), report=notes)
    assert [kind for kind, _ in notes] == ["loop"]
    assert _cells(new) == _cells(hand) + 1


def test_erasing_part_of_a_membrane_merges_and_removes_its_leftover_bits():
    masks = np.zeros((60, 80), np.uint16)
    masks[10:50, 10:40] = 1
    masks[10:50, 40:70] = 2
    hand = thin_hand(masks)
    bond = (hand == 255) & ~ndi.binary_opening(hand == 255, structure=np.ones((3, 3), bool), border_value=1)
    ys, xs = np.nonzero(bond & (np.arange(80)[None, :] > 30) & (np.arange(80)[None, :] < 50))
    erase = np.zeros(hand.shape, bool)
    middle = (ys > ys.min() + 3) & (ys < ys.max() - 3)  # leave 3 px at each end
    erase[ys[middle], xs[middle]] = True
    new = apply_strokes(hand, erase, np.zeros_like(erase))
    assert _cells(new) == 1
    assert (new[ys, xs] == 0).all()  # the leftover bits went too
    assert not ((hand == 255) & ~bond & (new != 255)).any()  # the outer border did not move


def test_stroke_cutting_off_background_at_the_tissue_edge_makes_a_cell():
    hand = thin_hand(_tissue())
    rr1, cc1 = line(10, 40, 2, 50)
    rr2, cc2 = line(2, 50, 10, 60)
    stroke = _paint(hand, np.concatenate([rr1, rr2]), np.concatenate([cc1, cc2]))
    new = apply_strokes(hand, np.zeros_like(stroke), stroke)
    assert _cells(new) == _cells(hand) + 1


def test_apply_bond_strokes_reports_and_updates_masks():
    masks = _tissue()
    hand = thin_hand(masks)[None]
    outlines = hand_to_outline(hand[0])[None]
    r = _middle_cell(hand[0], 16)
    cy, cx = map(int, r.centroid)
    edited = outlines.copy()
    rr, cc = line(cy, cx - 2, cy, cx + 2)
    edited[0, rr, cc] = 1  # an open line
    rr, cc = line(cy - 30, cx - 16, cy - 30, cx + 16)
    edited[0, rr, cc] = 1  # a split somewhere else
    report = []
    out, rois = apply_bond_strokes(masks[None].copy(), hand, outlines, edited, frames=[0], report=report)
    assert rois and [kind for _t, kind, _n in report] == ["open"]
    assert len(np.unique(out[0])) == len(np.unique(masks)) + 1


def test_frame_bonds_build_on_first_use_and_match_the_full_pass():
    from napari_tissuekit.core.ta_bond_edit import FrameBonds, hand_stack_from_outlines, make_bonds

    masks = np.stack([_tissue(seed=s) for s in range(4)])
    full_hand, outlines = make_bonds(masks)
    bonds = FrameBonds(masks.shape, lambda t: (masks[t], outlines[t]))
    assert not any(bonds.is_built(t) for t in range(4))  # nothing is computed when loading
    assert np.array_equal(hand_to_outline(bonds[2]), outlines[2])  # the saved lines, unchanged
    assert bonds.is_built(2) and not bonds.is_built(1)
    eager, _, fits = hand_stack_from_outlines(masks, outlines)  # what loading did before, for the whole movie
    assert fits.all() and all(np.array_equal(bonds[t], eager[t]) for t in range(4))
    assert full_hand.shape == masks.shape
    assert bonds.rebuilt == set()


def test_editing_works_on_frame_bonds_and_stores_a_copy():
    from napari_tissuekit.core.ta_bond_edit import FrameBonds, make_bonds

    masks = np.stack([_tissue(seed=s) for s in range(3)])
    _, outlines = make_bonds(masks)
    base = outlines.copy()
    bonds = FrameBonds(masks.shape, lambda t: (masks[t], base[t]))
    r = _middle_cell(bonds[1], 16)
    cy, cx = map(int, r.centroid)
    edited = base.copy()
    rr, cc = line(cy, cx - 16, cy, cx + 16)
    edited[1, rr, cc] = 1
    out, rois = apply_bond_strokes(masks.copy(), bonds, base, edited, frames=[1])
    assert rois and len(np.unique(out[1])) == len(np.unique(masks[1])) + 1
    assert not bonds.is_built(0) and not bonds.is_built(2)  # untouched frames were never built
    snap = bonds[1].copy()
    bonds[1] = snap
    bonds[1][0, 0] = 7
    assert snap[0, 0] != 7  # assigning keeps a copy (undo snapshots stay as they were)


def test_frame_bonds_rebuild_a_frame_whose_saved_outlines_do_not_fit():
    from skimage.segmentation import find_boundaries

    from napari_tissuekit.core.ta_bond_edit import FrameBonds, thin_hand

    masks = np.stack([_tissue(seed=s) for s in range(2)])
    cellpose_style = find_boundaries(masks[0], mode="inner").astype(np.uint8)  # not the plugin's 1 px bonds
    shown = {}
    bonds = FrameBonds(masks.shape, lambda t: (masks[t], cellpose_style), on_rebuilt=lambda t, o: shown.update({t: o}))
    assert np.array_equal(bonds[0], thin_hand(masks[0]))
    assert bonds.rebuilt == {0} and np.array_equal(shown[0], hand_to_outline(bonds[0]))
