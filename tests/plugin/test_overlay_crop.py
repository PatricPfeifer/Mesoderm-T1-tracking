"""The T1 overlay draws each pair on a small window around the two cells; the pixels must equal the full-frame version."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from napari_tissuekit.core.t1.run_tm_t1 import grayscale_rgb, paint_pair


def grid_labels(h=60, w=70, step=10):
    lab = np.zeros((h, w), np.uint32)
    k = 1
    for y in range(0, h - step + 1, step):
        for x in range(0, w - step + 1, step):
            lab[y : y + step - 1, x : x + step - 1] = k  # cells 1 px apart; the first row / column touches the image edge
            k += 1
    return lab


@pytest.mark.parametrize("reach", [1, 3])
def test_paint_pair_with_boxes_equals_full_frame(reach):
    lab = grid_labels()
    boxes = ndi.find_objects(lab.astype(np.int64))
    ids = [int(i) for i in np.unique(lab) if i]
    color = (255, 0, 0)
    n = 0
    for a in ids:
        for b in ids:
            if a >= b:
                continue
            full = np.zeros((*lab.shape, 3), np.uint8)
            fast = full.copy()
            paint_pair(full, lab, frozenset({a, b}), color, reach)
            paint_pair(fast, lab, frozenset({a, b}), color, reach, boxes)
            assert (full == fast).all(), (a, b)
            n += int(full.any())
    assert n > 10  # neighbouring pairs really painted something


def test_paint_pair_absent_cell_draws_nothing():
    lab = grid_labels()
    boxes = ndi.find_objects(lab.astype(np.int64))
    dst = np.zeros((*lab.shape, 3), np.uint8)
    paint_pair(dst, lab, frozenset({1, 9999}), (1, 2, 3), 3, boxes)
    assert not dst.any()


def test_grayscale_rgb_stack_equals_per_frame():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (3, 20, 30, 3), dtype=np.uint8)
    stack = grayscale_rgb(rgb)
    assert stack.shape == rgb.shape and stack.dtype == np.uint8
    assert all((stack[t] == grayscale_rgb(rgb[t])).all() for t in range(3))
