"""Cross-check Export Cell Props against hand-computable rectangles.

Run:  pytest tests/test_cell_properties_known_shapes.py -s
Also writes validation_labels.tif + validation_expected.csv next to this file, so you can
load the TIFF in napari, click cells, export, and compare with the expected CSV by eye.
Every expected value is derived from the rectangle size alone (no regionprops).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from napari_tissuekit.cell_properties import FEATURES, compute_properties

HERE = Path(__file__).parent
# id -> {frame: (y0, y1, x0, x1)}  (half-open pixel ranges)
SHAPES = {
    1: {0: (20, 30, 20, 30), 1: (20, 32, 20, 32)},  # 10x10 square, grows to 12x12
    2: {0: (50, 70, 0, 10)},                          # 20 (rows) x 10 (cols), touches left edge
    3: {1: (60, 70, 60, 90)},                         # 10 x 30, only in frame 1
}
H = W = 100


def make_stack() -> np.ndarray:
    st = np.zeros((2, H, W), np.uint16)
    for i, fr in SHAPES.items():
        for t, (y0, y1, x0, x1) in fr.items():
            st[t, y0:y1, x0:x1] = i
    return st


def expected(px: float = 1.0) -> pd.DataFrame:
    rows = []
    for i, fr in SHAPES.items():
        for t, (y0, y1, x0, x1) in fr.items():
            h, w = y1 - y0, x1 - x0
            area, per = h * w, 2 * (h - 1) + 2 * (w - 1)   # pixel-centre perimeter
            # rectangle principal axes: 4*sqrt((n^2-1)/12) along each side
            lh, lw = 4 * np.sqrt((h**2 - 1) / 12), 4 * np.sqrt((w**2 - 1) / 12)
            major, minor = max(lh, lw), min(lh, lw)
            rows.append({
                "track_id": i, "frame": t,
                "area_px2": area, "perimeter_px": per,
                "circularity": min(1, 4 * np.pi * area / per**2),
                "shape_index": per / np.sqrt(area),
                "eccentricity": np.sqrt(1 - (minor / major) ** 2),
                "solidity": 1.0,
                "major_axis_px": major, "minor_axis_px": minor,
                # skimage: 0 = long axis along rows (vertical); +-pi/2 = along columns
                "orientation": 0.0 if h > w else (np.pi / 2 if w > h else np.nan),
                "centroid_x_px": (x0 + x1 - 1) / 2, "centroid_y_px": (y0 + y1 - 1) / 2,
                "touches_edge": int(y0 == 0 or x0 == 0 or y1 == H or x1 == W),
            })
    df = pd.DataFrame(rows).sort_values(["track_id", "frame"]).reset_index(drop=True)
    if px != 1.0:
        for c in list(df.columns):
            if c.endswith("_px2"):
                df[c.replace("_px2", "_um2")] = df.pop(c) * px**2
            elif c.endswith("_px"):
                df[c.replace("_px", "_um")] = df.pop(c) * px
    return df


def test_write_manual_check_files():
    import tifffile
    tifffile.imwrite(HERE / "validation_labels.tif", make_stack())
    expected().round(4).to_csv(HERE / "validation_expected.csv", index=False)


@pytest.mark.parametrize("px", [0.0, 0.5])
def test_matches_hand_values(px):
    got = compute_properties(make_stack(), [1, 2, 3], list(FEATURES), pixel_size_um=px)
    exp = expected(px if px > 0 else 1.0)
    assert list(got["track_id"]) == list(exp["track_id"])
    assert list(got["frame"]) == list(exp["frame"])
    for c in exp.columns:
        if c == "orientation":
            continue  # square has no defined axis; checked below
        np.testing.assert_allclose(got[c], exp[c], rtol=1e-3, atol=1e-6, err_msg=c)
    # non-square rectangles only
    mask = exp["orientation"].notna()
    np.testing.assert_allclose(np.abs(got.loc[mask, "orientation"]), exp.loc[mask, "orientation"], atol=1e-3)
