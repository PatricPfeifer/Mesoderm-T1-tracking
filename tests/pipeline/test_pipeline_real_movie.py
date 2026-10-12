"""Pipeline tests on the real test movie (tests/data).

fast: tracking on stored Cellpose masks.
slow: Cellpose segmentation from the raw movie, then tracking.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tifffile import imread

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data"
MOVIE = DATA / "movie_cut.tif"
MASKS = DATA / "masks_cut_cpsam.tif"
N_FRAMES = 11


def run_script(*args):
    r = subprocess.run(
        [sys.executable, *map(str, args)], capture_output=True, text=True, cwd=ROOT
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r


def track(masks, outdir):
    run_script(
        "-m", "napari_patric.core.tracking",
        "--masks", masks, "--intensity", MOVIE, "--output-dir", outdir,
        "--exclude-border", 2, "--quiet",
    )
    tracks = pd.read_csv(next(outdir.glob("*_tracks.csv")))
    interior = pd.read_csv(next(outdir.glob("*_tracks_interior.csv")))
    return tracks, interior


def check_tracking(tracks, interior):
    frames_per_track = tracks.groupby("track_id").frame.count()
    # about 47 cells per frame; edge slivers create a few extra short tracks
    assert 40 <= tracks.groupby("frame").size().median() <= 55
    assert (frames_per_track == N_FRAMES).sum() >= 40
    # interior tracks (not touching the edge) must all span the whole movie
    per_interior = interior.groupby("track_id").frame.count()
    assert 15 <= len(per_interior) <= 30
    assert (per_interior == N_FRAMES).all()


def test_tracking_on_stored_masks(tmp_path):
    tracks, interior = track(MASKS, tmp_path)
    check_tracking(tracks, interior)


def test_registration_needs_intensity():
    """Without the raw image the tracker cannot undo the frame jumps (see README notes)."""
    from napari_patric.core.celltrack.pipeline import track_arrays

    masks = imread(MASKS)
    kw = dict(stem="x", verbose=False, expand=3, pyramidal_depth=3, w_topo=0.75,
              max_iter=3, appear_cost=1.5, disappear_cost=1.5, division_penalty=0.8,
              max_gap=3, gap_close=True, max_distance=50.0)
    with_int = track_arrays(masks, imread(MOVIE), None, **kw)["qc"]["n_tracks"]
    without = track_arrays(masks, None, None, **kw)["qc"]["n_tracks"]
    assert without > with_int * 1.3


@pytest.mark.slow
def test_segmentation_then_tracking(tmp_path):
    results = tmp_path / "results"
    run_script("-m", "napari_patric.cli", "segment", MOVIE, "--results", results,
               "--preset", "cellflow", "--force-cpu")
    masks_path = next((results / "segmentation").glob("*_cp_masks.tif"))
    masks = imread(masks_path)
    assert masks.shape == (N_FRAMES, 291, 291)
    cells = [len(np.unique(m)) - 1 for m in masks]
    assert all(40 <= c <= 55 for c in cells), cells
    tracks, interior = track(masks_path, tmp_path / "track")
    check_tracking(tracks, interior)
