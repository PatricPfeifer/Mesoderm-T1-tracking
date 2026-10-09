#!/usr/bin/env python3
"""Headless napariCellFlow CellTracker on a label stack.

Example (tmux session 5, conda env cellflow):
  python scripts/cellflow/run_cellflow_track.py \\
    --input output/cellflow/naima_cut/Cri_WT_GAP43_GFP_38sec_Naima_cut_cf_masks.tif \\
    --output-dir output/cellflow/naima_cut
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite

_VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "napariCellFlow" / "napariCellFlow"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from cell_tracking import CellTracker  # noqa: E402
from structure import AnalysisConfig, TrackingParameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="napariCellFlow overlap tracking.")
    p.add_argument("--input", type=Path, required=True, help="T,Y,X integer masks.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--min-overlap-ratio", type=float, default=0.5)
    p.add_argument("--max-displacement", type=float, default=50.0)
    p.add_argument("--min-cell-size", type=int, default=100)
    p.add_argument("--no-gap-closing", action="store_true")
    p.add_argument("--max-frame-gap", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    outdir = args.output_dir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{inp.stem.replace('_cf_masks', '')}_cf_tracked_labels.tif"
    meta_path = outdir / f"{out.stem}_qc.json"

    print("loading", inp, flush=True)
    stack = np.asarray(imread(inp))
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    if stack.ndim != 3:
        raise SystemExit(f"expected T,Y,X labels, got {stack.shape}")
    if not np.issubdtype(stack.dtype, np.integer):
        raise SystemExit(f"expected integer labels, got {stack.dtype}")
    print("shape", stack.shape, stack.dtype, "max", int(stack.max()), flush=True)

    params = TrackingParameters(
        min_overlap_ratio=args.min_overlap_ratio,
        max_displacement=args.max_displacement,
        min_cell_size=args.min_cell_size,
        enable_gap_closing=not args.no_gap_closing,
        max_frame_gap=args.max_frame_gap,
    )
    params.validate()
    tracker = CellTracker(AnalysisConfig())
    tracker.update_parameters(params)
    tracker.set_progress_callback(lambda p, m="": print(f"{p:.0f}% {m}", flush=True))

    print("tracking...", flush=True)
    tracked = tracker.track_cells(stack)
    imwrite(out, np.asarray(tracked, dtype=np.uint16))

    n_ids = int(len(np.unique(tracked)) - (1 if 0 in tracked else 0))
    per_frame = [int(len(np.unique(tracked[t])) - (1 if 0 in tracked[t] else 0)) for t in range(len(tracked))]
    meta = {
        "input": str(inp),
        "output": str(out),
        "shape": list(tracked.shape),
        "n_track_ids": n_ids,
        "cells_per_frame": per_frame,
        "min_overlap_ratio": params.min_overlap_ratio,
        "max_displacement": params.max_displacement,
        "min_cell_size": params.min_cell_size,
        "enable_gap_closing": params.enable_gap_closing,
        "max_frame_gap": params.max_frame_gap,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", out, "n_track_ids", n_ids, flush=True)


if __name__ == "__main__":
    main()
