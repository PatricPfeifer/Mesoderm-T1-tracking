"""Tracking can give empty areas enclosed by cells (missed or deleted cells) an id of their own."""
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
from test_divisions import CELL, GRID, grid_frame  # noqa: E402

from napari_tissuekit.cli import build_parser  # noqa: E402
from napari_tissuekit.core.celltrack.pipeline import track_arrays  # noqa: E402

HOLE = (3, 3)  # grid row, col of the cell that is missing in every frame


def stack(n=4):
    lab = grid_frame()
    r, c = HOLE
    lab[r * CELL : (r + 1) * CELL, c * CELL : (c + 1) * CELL] = 0
    return np.stack([lab] * n)


def tracked(tmp_path, fill):
    track_arrays(stack(), None, tmp_path, stem="t", verbose=False, fill_holes=fill, gap_close=False, write_rgb=False)
    return tifffile.imread(next(tmp_path.glob("*_tracked_labels.tif")))


def test_hole_gets_an_id_when_filled(tmp_path):
    out = tracked(tmp_path, True)
    r, c = HOLE
    area = out[:, r * CELL + 2 : (r + 1) * CELL - 2, c * CELL + 2 : (c + 1) * CELL - 2]
    assert (area > 0).all()  # a cell with an id in every frame
    assert len(set(np.unique(area[0]))) == 1 and len({int(np.unique(area[t])[0]) for t in range(len(out))}) == 1  # same id throughout


def test_hole_stays_empty_without_filling(tmp_path):
    out = tracked(tmp_path, False)
    r, c = HOLE
    assert not out[:, r * CELL : (r + 1) * CELL, c * CELL : (c + 1) * CELL].any()


def test_cli_fills_by_default_and_can_be_switched_off():
    parser = build_parser()
    assert parser.parse_known_args(["track", "m.tif"])[0].fill_holes is True
    assert parser.parse_known_args(["track", "m.tif", "--no-fill-holes"])[0].fill_holes is False
