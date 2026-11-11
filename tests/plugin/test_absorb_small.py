import numpy as np

from napari_tissuekit.core.cell_size import MIN_CELL_AREA, absorb_small_labels


def _tissue():
    lab = np.zeros((60, 60), np.uint16)
    lab[5:55, 5:30] = 1
    lab[5:55, 30:55] = 2
    return lab


def test_small_cell_is_taken_over_by_neighbours():
    lab = _tissue()
    lab[20:25, 28:33] = 3  # 25 px cell on the border between 1 and 2
    out, n = absorb_small_labels(lab)
    assert n == 1 and 3 not in out
    assert (out[5:55, 5:55] > 0).all()  # no empty spot is left
    assert out[22, 27] == 1 and out[22, 33] == 2


def test_tiny_enclosed_gap_is_closed_but_big_hole_and_outside_stay():
    lab = _tissue()
    lab[30:34, 10:14] = 0  # 16 px hole inside cell 1
    lab[10:20, 38:50] = 0  # 120 px hole: kept (fill_enclosed_holes makes it a cell)
    out, _ = absorb_small_labels(lab)
    assert (out[30:34, 10:14] == 1).all()
    assert (out[10:20, 38:50] == 0).all()
    assert (out[:5] == 0).all()  # outside of the tissue untouched


def test_isolated_speck_is_dropped_and_big_cells_unchanged():
    lab = np.pad(_tissue(), 30)
    lab[2:4, 2:4] = 9  # far from every cell
    out, n = absorb_small_labels(lab)
    assert n == 1 and (out[2:4, 2:4] == 0).all()
    assert (out[40:80, 40:80] == lab[40:80, 40:80]).all()
    assert MIN_CELL_AREA == 50
