"""Synthetic tissue with known never-crossed lines: the scan must find the mixing minima at the lines and the spacing."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "lineage_boundaries"))

import metrics  # noqa: E402
import periodicity as per  # noqa: E402
import scan  # noqa: E402
from load import Tissue  # noqa: E402

CFG = {"axes": {"anterior_is_small": True},
       "region": {"border_margin_d": 2.0, "ap_min_d": None, "ap_max_d": None, "dv_min_d": None, "dv_max_d": None},
       "scan": {"x0_step_d": 0.25, "dv_bin_d": 1.0, "min_cells_per_side": 10, "dv_subbands": 3,
                "clean_span_fraction": 0.8, "frame_step": 1},
       "snapping": {"radius_d": 1.0}}
PERIOD, NCOL, NROW, T = 10, 50, 22, 24


def make_tissue(restricted: bool, seed=0) -> Tissue:
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(NCOL) + 0.5, np.arange(NROW) + 0.5)
    x0, y0 = xs.ravel() + rng.normal(0, 0.1, xs.size), ys.ravel() + rng.normal(0, 0.1, xs.size)
    N = x0.size
    lo = (np.floor(x0 / PERIOD)) * PERIOD + 0.3  # compartment limits
    hi = lo + PERIOD - 0.6
    x, y = np.empty((T, N)), np.empty((T, N))
    cx, cy = x0.copy(), y0.copy()
    for t in range(T):
        x[t], y[t] = cx, cy
        cx = cx + rng.normal(0, 0.45, N)
        cy = cy + rng.normal(0, 0.3, N)
        if restricted:
            cx = np.clip(cx, lo, hi)  # never crosses a line
        cy = np.clip(cy, 0.2, NROW - 0.2)
    edges = []
    for t in range(T):
        pairs = np.array(sorted(cKDTree(np.c_[x[t], y[t]]).query_pairs(1.5)))
        edges.append((pairs[:, 0], pairs[:, 1], np.ones(len(pairs))))
    labels = np.ones((T, NROW + 8, NCOL + 8), dtype=np.int32)
    # label image: the tissue sits at pixel (y + 4, x + 4) with pixel size 1; borders are background
    labels[:, :4], labels[:, -4:], labels[:, :, :4], labels[:, :, -4:] = 0, 0, 0, 0
    x, y = x + 4, y + 4  # same shift in the coordinates so that the border distance is measured correctly
    return Tissue("synthetic", 1.0, 60.0, np.arange(1, N + 1), x, y, np.full((T, N), 0.785), np.full(N, -1), edges, labels, 1.0)


def profile(ts):
    cfg = CFG
    win = metrics.Window(ts, cfg, 0, range(T))
    x0s = np.arange(ts.x[0].min() + 3, ts.x[0].max() - 3, 0.25)
    r = scan.run_variants(win, ts.x[0], x0s, cfg, snapping=False)["plain"]
    return x0s, r["m2"]


def test_restricted_lines_are_recovered():
    ts = make_tissue(True)
    x0s, m2 = profile(ts)
    lines = np.arange(PERIOD, NCOL, PERIOD) + 4  # the three interior lines, shifted like the coordinates
    near = np.min(np.abs(x0s[:, None] - lines[None, :]), axis=1) <= 0.4
    assert np.nanmean(m2[near]) < 0.5 * np.nanmean(m2[~near]), "mixing is not lower at the known lines"
    st = per.stats(m2, 0.25, 1.5, 12.0)
    assert abs(st["period_d"] - PERIOD) <= 1.5, st


def test_free_tissue_has_no_line_structure():
    ts = make_tissue(False)
    x0s, m2 = profile(ts)
    lines = np.arange(PERIOD, NCOL, PERIOD) + 4
    near = np.min(np.abs(x0s[:, None] - lines[None, :]), axis=1) <= 0.4
    restricted = profile(make_tissue(True))[1]
    ratio_free = np.nanmean(m2[near]) / np.nanmean(m2[~near])
    ratio_res = np.nanmean(restricted[near]) / np.nanmean(restricted[~near])
    assert ratio_free > 2 * ratio_res
