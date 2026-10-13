#!/usr/bin/env python3
"""Track Cellpose masks like cellpose_track.py, with explicit cell divisions.

Same tracker, plus division handling: when a cell divides, its track ends and
both daughters get new track IDs. Divisions are written to
``{stem}_lineage.csv`` (parent_track_id per track), marked with magenta dots in
the tracked RGB, and ``run_tm_t1.py`` reads the lineage file to exclude the
mother and both daughters from T1 detection.

A division needs two adjacent cells whose union matches the warped mother,
and is kept only if it lowers the assignment cost by more than
``--division-penalty``; divisions stay rare.

Example:
  conda activate TA
  python scripts/cellpose_track/cellpose_track_divisions.py \\
    --masks "output/cellpose_seg/Cri_WT_GAP43_GFP_38sec_Naima_cut_cp_seg/Cri_WT_GAP43_GFP_38sec_Naima_cut_cp_masks.tif" \\
    --intensity "input files/Cri_WT_GAP43_GFP_38sec_Naima_cut.tif" \\
    --output-dir output/cellpose_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_cp_track_divisions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/cellpose_track/cellpose_track.py` without installing.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from napari_tissuekit.core.celltrack.pipeline import run_tracking  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cellpose + TA-logic tracker with divisions: daughters get new "
            "IDs and a lineage table is written for T1 exclusion."
        )
    )
    p.add_argument(
        "--masks",
        type=Path,
        required=True,
        help="T,Y,X Cellpose label stack (*_cp_masks.tif).",
    )
    p.add_argument(
        "--intensity",
        type=Path,
        default=None,
        help="Matching T,Y,X raw stack for registration (strongly advised).",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Only use the first N frames of the on-disk stacks (before blank drop).",
    )
    p.add_argument(
        "--expand",
        type=int,
        default=3,
        help="Grow labels by N px before building adjacency (default: 3).",
    )
    p.add_argument(
        "--pyramidal-depth",
        type=int,
        default=3,
        help="Registration pyramid levels (default: 3).",
    )
    p.add_argument(
        "--w-topo",
        type=float,
        default=0.75,
        help="Weight of neighbour-Dice term in costs (default: 0.75).",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=3,
        help="Topology refinement iterations (default: 3).",
    )
    p.add_argument(
        "--appear-cost",
        type=float,
        default=1.5,
        help="LAP cost for a new appearance (default: 1.5).",
    )
    p.add_argument(
        "--disappear-cost",
        type=float,
        default=1.5,
        help="LAP cost for a disappearance (default: 1.5).",
    )
    p.add_argument(
        "--division-penalty",
        type=float,
        default=1.0,
        help=(
            "Cost a division must save over the best non-division "
            "assignment; higher = fewer divisions (default: 1.0)."
        ),
    )
    p.add_argument(
        "--min-daughter-frames",
        type=int,
        default=3,
        help=(
            "Flag a division if a daughter track is shorter than this "
            "(likely an over-split; default: 3)."
        ),
    )
    p.add_argument(
        "--division-refractory",
        type=int,
        default=10,
        help="Frames after its birth in which a daughter cannot divide (default: 10).",
    )
    p.add_argument(
        "--max-gap",
        type=int,
        default=3,
        help="Max frames to bridge in gap closing (default: 3; 0 disables).",
    )
    p.add_argument(
        "--max-distance",
        type=float,
        default=50.0,
        help="Max centroid distance for candidate links (default: 50).",
    )
    p.add_argument(
        "--backward",
        action="store_true",
        help="Also track reversed movie and report forward/backward agreement.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Less progress output.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_tracking(
        args.masks,
        args.intensity,
        args.output_dir,
        max_frames=args.frames,
        expand=args.expand,
        pyramidal_depth=args.pyramidal_depth,
        w_topo=args.w_topo,
        max_iter=args.max_iter,
        appear_cost=args.appear_cost,
        disappear_cost=args.disappear_cost,
        division_penalty=args.division_penalty,
        max_gap=args.max_gap,
        gap_close=args.max_gap > 0,
        max_distance=args.max_distance,
        run_backward=args.backward,
        verbose=not args.quiet,
        division_mode="lineage",
        division_refractory=args.division_refractory,
        min_daughter_frames=args.min_daughter_frames,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
