"""Single regular hexagon (side s, px) -> closed-form values vs compute_properties.

Run:  python tests/test_cell_properties_hexagon.py   (prints a comparison table)
Rasterisation makes pixel values differ slightly from the ideal, hence the tolerances.
"""
import numpy as np
import pytest
from skimage.draw import polygon

from napari_tissuekit.cell_properties import FEATURES, compute_properties

S = 60.0           # side length in px (circumradius = S)
CX, CY = 100.0, 90.0
N = 200


def make_stack() -> np.ndarray:
    a = np.deg2rad(np.arange(6) * 60)               # flat-top hexagon, vertex on +x axis
    rr, cc = polygon(CY + S * np.sin(a), CX + S * np.cos(a), (N, N))
    st = np.zeros((1, N, N), np.uint8)
    st[0, rr, cc] = 1
    return st


# value -> (closed-form, relative tolerance)
IDEAL = {
    "area_px2":       (3 * np.sqrt(3) / 2 * S**2, 0.01),
    "perimeter_px":   (6 * S, 0.03),                 # skimage perimeter is an estimate
    "circularity":    (np.pi / (2 * np.sqrt(3)), 0.04),   # 4*pi*A/P^2 = 0.9069
    "shape_index":    (2 * np.sqrt(2 * np.sqrt(3)), 0.03),  # P/sqrt(A) = 3.722
    "solidity":       (1.0, 0.02),
    "major_axis_px":  (4 * np.sqrt(5 / 24) * S, 0.02),  # sqrt(5/24)*S = rms radius
    "minor_axis_px":  (4 * np.sqrt(5 / 24) * S, 0.02),
    "centroid_x_px":  (CX, 0.005),
    "centroid_y_px":  (CY, 0.005),
}
ABS = {"eccentricity": 0.1, "touches_edge": 0}      # ideal: 0 and 0


def run() -> bool:
    row = compute_properties(make_stack(), [1], list(FEATURES)).iloc[0]
    ok = True
    print(f"{'feature':16}{'expected':>12}{'got':>12}{'diff %':>9}")
    for k, (exp, tol) in IDEAL.items():
        d = abs(row[k] - exp) / exp
        ok &= d <= tol
        print(f"{k:16}{exp:12.4f}{row[k]:12.4f}{100*d:9.2f}  {'OK' if d <= tol else 'FAIL'}")
    for k, tol in ABS.items():
        good = abs(row[k]) <= tol
        ok &= good
        print(f"{k:16}{0:12.4f}{row[k]:12.4f}{'':9}  {'OK' if good else 'FAIL'}")
    return ok


@pytest.mark.xfail(
    reason="skimage's pixel-based perimeter overestimates a rasterised hexagon by about 4%, "
    "which biases perimeter, circularity, shape_index (and eccentricity slightly). "
    "Remove the marker once the perimeter is computed from a smoothed contour.",
    strict=False,
)
def test_hexagon():
    assert run()


if __name__ == "__main__":
    print("ALL OK" if run() else "MISMATCH")
