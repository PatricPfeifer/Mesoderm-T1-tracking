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
