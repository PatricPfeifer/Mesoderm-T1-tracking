#!/usr/bin/env python3
"""Track Cellpose masks with looser motion gates for long Δt movies.

Same engine as cellpose_track.py. Defaults assume cells can move farther
between frames: larger candidate radius, higher appear/disappear cost, and
weaker neighbour-topology weight.

Example:
  conda activate TA
  python scripts/cellpose_track/cellpose_track_long_interval.py \\
    --masks "output/cellpose_seg/lateral20250806bCri-Gap43Verena_6_180s-slice7_cp_seg/lateral20250806bCri-Gap43Verena_6_180s-slice7_cp_masks.tif" \\
    --intensity "input files/lateral20250806bCri-Gap43Verena_6_180s-slice7.tif" \\
    --output-dir output/cellpose_track/lateral20250806_slice7_long_interval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
for _p in (_HERE, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from napari_tissuekit.core.celltrack.pipeline import run_tracking  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cellpose tracker tuned for long frame intervals: "
            "max-distance 120 px, appear/disappear 2.0, w-topo 0.4."
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
        default=0.4,
        help="Weight of neighbour-Dice term in costs (default: 0.4; short-Δt is 0.75).",
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
        default=2.0,
        help="LAP cost for a new appearance (default: 2.0; short-Δt is 1.5).",
    )
    p.add_argument(
        "--disappear-cost",
        type=float,
        default=2.0,
        help="LAP cost for a disappearance (default: 2.0; short-Δt is 1.5).",
    )
    p.add_argument(
        "--division-penalty",
        type=float,
        default=0.8,
        help="Extra cost to attach a second daughter (default: 0.8).",
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
        default=120.0,
        help="Max centroid distance for candidate links (default: 120; short-Δt is 50).",
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
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
