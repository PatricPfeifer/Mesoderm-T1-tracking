import numpy as np

from napari_tissuekit.core.cell_size import MIN_CELL_AREA, absorb_small_labels, min_cell_area, remove_small_labels, set_pixel_size


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


def test_minimum_area_is_4_square_micrometres_at_the_movie_pixel_size():
    assert min_cell_area() == MIN_CELL_AREA  # pixel size unknown: the fixed pixel minimum
    assert min_cell_area(0.1032) == 376      # 4 / 0.1032^2 = 375.6 px, rounded up
    assert min_cell_area(0.1625) == 152
    assert min_cell_area(0.4735) == 18
    set_pixel_size(0.1032)
    try:
        assert min_cell_area() == 376
        stack = np.zeros((1, 60, 60), dtype=np.uint16)
        stack[0, :20, :20] = 1   # 400 px: a cell
        stack[0, 30:50, 30:48] = 2  # 360 px: under 4 um^2, not a cell
        out, n = remove_small_labels(stack)
        assert n == 1 and set(np.unique(out)) == {0, 1}
        absorbed = absorb_small_labels(stack)[0]
        assert set(np.unique(absorbed)) == {0, 1}
    finally:
        set_pixel_size(None)
    assert min_cell_area() == MIN_CELL_AREA


def test_flow_motion_field_follows_a_shifted_pattern_and_keeps_its_interface():
    from napari_tissuekit.core.celltrack.costs import warp_labels
    from napari_tissuekit.core.celltrack.motion import estimate_flow

    rng = np.random.default_rng(0)
    base = rng.random((128, 192)).astype(np.float32)
    from scipy import ndimage as ndi

    img0 = ndi.gaussian_filter(base, 3) * 255
    img1 = np.roll(img0, (4, 7), axis=(0, 1))  # the whole pattern moves 4 px down, 7 px right
    field = estimate_flow(img0, img1)
    dense = field.dense()
    assert dense.shape == (128, 192, 2) and dense.dtype == np.float32
    dy, dx = field.sample(np.array([64.0]), np.array([96.0]))
    assert abs(float(dy[0]) - 4) < 1.0 and abs(float(dx[0]) - 7) < 1.0
    labels = np.zeros((128, 192), np.uint16)
    labels[50:70, 80:110] = 5
    moved = warp_labels(labels, field)
    ys, xs = np.nonzero(moved == 5)
    assert abs(ys.mean() - (59.5 + 4)) < 1.5 and abs(xs.mean() - (94.5 + 7)) < 1.5
