import numpy as np

from napari_tissuekit.core.cell_size import MIN_CELL_AREA, remove_small_labels


def test_small_labels_removed_per_frame():
    stack = np.zeros((2, 40, 40), dtype=np.uint16)
    stack[0, :8, :8] = 1          # 64 px: kept
    stack[0, 20:25, 20:30] = 2    # 50 px: kept (the minimum is inclusive)
    stack[0, 38:40, 38:40] = 3    # 4 px: removed
    stack[1, :7, :7] = 1          # 49 px in this frame: removed
    out, n = remove_small_labels(stack)
    assert MIN_CELL_AREA == 50 and n == 2
    assert set(np.unique(out[0])) == {0, 1, 2} and set(np.unique(out[1])) == {0}
    assert stack[0, 38, 38] == 3  # input untouched
    assert remove_small_labels(stack[0])[0].shape == (40, 40)
