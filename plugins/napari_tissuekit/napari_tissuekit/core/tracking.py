#!/usr/bin/env python3
"""Track Cellpose masks with Tissue Analyzer logic (no TA file layout).

Keeps TA's pyramidal motion prior and neighbour-topology score, but replaces
greedy point-probe matching with global LAP assignment over an overlap matrix.

Example:
  conda activate TA
  python scripts/cellpose_track/cellpose_track.py \\
    --masks "output/cellpose_seg/Cri_WT_GAP43_GFP_38sec_Naima_cut_cp_seg/Cri_WT_GAP43_GFP_38sec_Naima_cut_cp_masks.tif" \\
    --intensity "input files/Cri_WT_GAP43_GFP_38sec_Naima_cut.tif" \\
    --output-dir output/cellpose_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_cp_track
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .celltrack.pipeline import run_tracking  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cellpose + TA-logic tracker: overlap matrix, global LAP, "
            "topology refinement, gap closing. No epyseg / TA layout."
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
        default=50.0,
        help="Max centroid distance for candidate links (default: 50).",
    )
    p.add_argument(
        "--backward",
        action="store_true",
        help="Also track reversed movie and report forward/backward agreement.",
    )
    p.add_argument(
        "--exclude-border",
        type=int,
        default=0,
        metavar="PX",
        help="After tracking, also write *_interior files without tracks that touch "
        "the image edge (within PX) in any frame (default: 0 = off).",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="CPU threads for cell tables and motion estimation "
        "(default: 1 = serial; -1 = all cores). Results are identical.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Less progress output.",
    )
    return p.parse_args(argv)


def write_interior_tracks(out_dir: Path, margin: int) -> None:
    """Track everything, then drop whole tracks that touch the image edge in any frame."""
    import numpy as np
    import pandas as pd
    from tifffile import imread, imwrite

    lab_path = next(out_dir.glob("*_tracked_labels.tif"))
    csv_path = next(out_dir.glob("*_tracks.csv"))
    stack = np.array(imread(lab_path))
    edge = np.zeros(stack.shape[1:], dtype=bool)
    edge[:margin], edge[-margin:], edge[:, :margin], edge[:, -margin:] = True, True, True, True
    border_ids = set(np.unique(stack[:, edge]).tolist()) - {0}
    stack[np.isin(stack, list(border_ids))] = 0
    imwrite(out_dir / lab_path.name.replace("_tracked_labels", "_tracked_labels_interior"), stack)
    df = pd.read_csv(csv_path)
    df = df[~df["track_id"].isin(border_ids)]
    df.to_csv(out_dir / csv_path.name.replace("_tracks", "_tracks_interior"), index=False)
    n_all = len(np.unique(imread(lab_path))) - 1
    print(
        f"--exclude-border {margin}px: dropped {len(border_ids)} of {n_all} tracks "
        f"touching the edge; kept {n_all - len(border_ids)} (*_interior files)",
        flush=True,
    )


def main(argv=None) -> None:
    args = parse_args(argv)
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
        n_jobs=args.n_jobs,
        verbose=not args.quiet,
    )
    if args.exclude_border > 0:
        write_interior_tracks(Path(args.output_dir), args.exclude_border)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
