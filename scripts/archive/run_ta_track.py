#!/usr/bin/env python3
"""Track cell identity across time with pyTA (dynamic tissue tracker).

Tracking only. Expects an existing TA layout from run_ta_segment.py:
  output-dir/frames/t####.tif
  output-dir/frames/t####/handCorrection.tif  (corrected in the GUI)

Pipeline:
  load frames + handCorrection -> dynamic tracking -> export RGB / integer ID stacks

Examples:
  conda activate TA
  # after run_ta_segment.py (+ optional GUI mask edits):
  python run_ta_track.py \\
    --output-dir output/ta_seg/naima_cut_cp_masks \\
    --input "input files/Cri_WT_GAP43_GFP_38sec_Naima_cut.tif"

For segmentation + GUI correction, use run_ta_segment.py first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

from ta_common import (
    DEFAULT_INPUT,
    collect_frame_paths,
    export_track_stacks,
    load_stack,
    run_tracking,
    validate_hand_corrections,
    write_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tissue Analyzer: track cell identity from existing "
            "handCorrection.tif masks (no segmentation)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Path to T,Y,X intensity TIFF. Used to refresh/verify frames "
            "under --output-dir/frames/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "TA run directory with frames/ and frames/*/handCorrection.tif "
            "(from run_ta_segment.py)."
        ),
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help="Channel index for multichannel frames (default: None / first).",
    )
    parser.add_argument(
        "--rewrite-frames",
        action="store_true",
        help=(
            "Rewrite intensity frames from --input before tracking "
            "(default: reuse existing frames/ TIFFs)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir = args.output_dir.expanduser().resolve()
    frames_dir = output_dir / "frames"
    if not frames_dir.is_dir():
        raise FileNotFoundError(
            f"Frames directory not found: {frames_dir}. "
            "Run run_ta_segment.py first."
        )

    if args.rewrite_frames:
        print(f"Loading intensity {input_path}", flush=True)
        stack = load_stack(input_path)
        print(f"intensity shape={stack.shape} dtype={stack.dtype}", flush=True)
        print(f"Rewriting frames under {frames_dir}", flush=True)
        frame_paths = write_frames(stack, frames_dir)
    else:
        frame_paths = collect_frame_paths(frames_dir)
        print(
            f"Reusing {len(frame_paths)} frames under {frames_dir}",
            flush=True,
        )

    validate_hand_corrections(frame_paths)

    run_tracking(frame_paths, channel=args.channel)

    rgb_out, ids_out = export_track_stacks(frame_paths, output_dir)
    n_ids = int(np.asarray(tifffile.imread(ids_out)).max())
    print(
        f"Done. frames={len(frame_paths)} max_track_id={n_ids}",
        flush=True,
    )
    print(f"RGB tracks: {rgb_out}", flush=True)
    print(f"ID tracks:  {ids_out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
