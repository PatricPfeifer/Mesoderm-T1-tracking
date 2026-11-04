import numpy as np

from napari_tissuekit.core.cell_size import MIN_CELL_AREA, remove_small_labels


def test_small_labels_removed_per_frame():
    stack = np.zeros((2, 20, 20), dtype=np.uint16)
    stack[0, :4, :4] = 1          # 16 px: kept
    stack[0, 10:13, 10:15] = 2    # 15 px: kept (the minimum is inclusive)
    stack[0, 18:20, 18:20] = 3    # 4 px: removed
    stack[1, :3, :4] = 1          # 12 px in this frame: removed
    out, n = remove_small_labels(stack)
    assert MIN_CELL_AREA == 15 and n == 2
    assert set(np.unique(out[0])) == {0, 1, 2} and set(np.unique(out[1])) == {0}
    assert stack[0, 18, 18] == 3  # input untouched
    assert remove_small_labels(stack[0])[0].shape == (20, 20)
