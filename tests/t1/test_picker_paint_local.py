"""Picker junction painting: the cropped (boxes) version must equal the full-frame one."""
import numpy as np
from scipy import ndimage as ndi

from napari_tissuekit.core.t1.run_tm_t1_picker import paint_pair

COLOR = (200, 30, 90)


def random_cells(rng, h=60, w=80, n=14):
    """Voronoi-like labels with background gaps and cells touching the image border."""
    pts = rng.integers(0, [h, w], size=(n, 2))
    yy, xx = np.mgrid[0:h, 0:w]
    d = ((yy[..., None] - pts[:, 0]) ** 2 + (xx[..., None] - pts[:, 1]) ** 2)
    lab = d.argmin(-1) + 1
    nearest = np.sort(d, axis=-1)
    lab[(nearest[..., 1] - nearest[..., 0]) < 25] = 0  # gaps between neighbouring cells
    return lab.astype(np.int32)


def test_local_painting_equals_full_frame_painting():
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(12):
        lab = random_cells(rng)
        boxes = ndi.find_objects(lab)
        ids = [i for i in range(1, lab.max() + 1) if (lab == i).any()]
        for _ in range(25):
            a, b = rng.choice(ids, size=2, replace=False)
            for reach in (1, 3, 5):
                full = np.zeros(lab.shape + (3,), dtype=np.uint8)
                local = np.zeros_like(full)
                paint_pair(full, lab, frozenset((int(a), int(b))), COLOR, reach)
                paint_pair(local, lab, frozenset((int(a), int(b))), COLOR, reach, boxes)
                assert np.array_equal(full, local), (a, b, reach)
                checked += 1
    assert checked == 12 * 25 * 3


def test_absent_cell_paints_nothing():
    lab = random_cells(np.random.default_rng(1))
    boxes = ndi.find_objects(lab)
    out = np.zeros(lab.shape + (3,), dtype=np.uint8)
    paint_pair(out, lab, frozenset((1, 999)), COLOR, 3, boxes)
    assert not out.any()
