#!/usr/bin/env python3
"""Segment a time-lapse TIFF for Tissue Analyzer, then open the pyTA GUI to correct masks.

Pipeline:
  split T,Y,X stack -> EPySeg (GPU if available) or Cellpose --masks
  -> write handCorrection.tif per frame
  -> open pyTA Segmentation tab for manual correction
  -> then run run_ta_track.py on the same --output-dir

Examples:
  conda activate TA
  python run_ta_segment.py \\
    --input "input files/Cri_WT_GAP43_GFP_38sec_Naima_cut.tif"

  python run_ta_segment.py \\
    --masks "output/cellpose_seg/masks/naima_cut_cp_masks.tif" \\
    --input "input files/Cri_WT_GAP43_GFP_38sec_Naima_cut.tif"

  # headless (no GUI; for batch / no DISPLAY):
  python run_ta_segment.py --masks "..." --input "..." --skip-gui

After correcting masks in the GUI, track with:
  python run_ta_track.py --output-dir <same> --input <intensity>

GUI needs X11/VNC (echo $DISPLAY). Prefer a GPU allocation for EPySeg.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ta_common import (
    DEFAULT_INPUT,
    PROJECT_ROOT,
    collect_frame_paths,
    detect_use_cpu,
    launch_correction_gui,
    load_stack,
    run_epyseg_segmentation,
    validate_hand_corrections,
    write_frames,
    write_hand_corrections,
    write_list_lst,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tissue Analyzer / EPySeg: segment a time-lapse TIFF and open "
            "pyTA for manual mask correction (before tracking)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to T,Y,X (or Y,X) intensity TIFF.",
    )
    parser.add_argument(
        "--masks",
        type=Path,
        default=None,
        help=(
            "Optional T,Y,X label mask stack (e.g. Cellpose). "
            "Skips EPySeg; converts labels to TA handCorrection.tif."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Work directory (default: output/ta_seg/<stem>/).",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help="Channel index for multichannel frames (default: None / first).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="EPySeg tile width/height (default: 256).",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=32,
        help="EPySeg tile overlap (default: 32).",
    )
    parser.add_argument(
        "--size-filter",
        type=int,
        default=None,
        help="Optional min cell area (pixels) for EPySeg post-process.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force TensorFlow/EPySeg onto CPU even if a GPU is available.",
    )
    parser.add_argument(
        "--skip-gui",
        action="store_true",
        help="Headless segmentation only; do not open the pyTA correction GUI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    masks_path = args.masks.expanduser().resolve() if args.masks else None
    if masks_path is not None and not masks_path.exists():
        raise FileNotFoundError(f"Masks not found: {masks_path}")

    stem = masks_path.stem if masks_path is not None else input_path.stem
    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "output" / "ta_seg" / stem
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"

    print(f"Loading intensity {input_path}", flush=True)
    stack = load_stack(input_path)
    print(f"intensity shape={stack.shape} dtype={stack.dtype}", flush=True)

    print(f"Writing {stack.shape[0]} frames under {frames_dir}", flush=True)
    frame_paths = write_frames(stack, frames_dir)

    if masks_path is not None:
        print(f"Loading masks {masks_path}", flush=True)
        masks = load_stack(masks_path, allow_rgb_outlines=True)
        print(f"masks shape={masks.shape} dtype={masks.dtype}", flush=True)
        if masks.shape != stack.shape:
            raise ValueError(
                f"Mask shape {masks.shape} does not match intensity {stack.shape}."
            )
        print(
            "Converting Cellpose/instance labels to TA handCorrection.tif...",
            flush=True,
        )
        write_hand_corrections(masks, frame_paths)
        print("Skipping EPySeg (precomputed masks).", flush=True)
        device_label = "CPU(masks)"
    else:
        use_cpu, _gpus = detect_use_cpu(args.force_cpu)
        device_label = "CPU" if use_cpu else "GPU"
        print("Running EPySeg segmentation (TA mode)...", flush=True)
        run_epyseg_segmentation(
            frame_paths,
            use_cpu=use_cpu,
            channel=args.channel,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            size_filter=args.size_filter,
        )

    validate_hand_corrections(frame_paths)
    list_path = write_list_lst(frame_paths, frames_dir)
    print(
        f"Segmentation done. frames={len(frame_paths)} device={device_label}\n"
        f"list.lst: {list_path}",
        flush=True,
    )

    track_cmd = (
        f'python run_ta_track.py \\\n'
        f'  --output-dir "{output_dir}" \\\n'
        f'  --input "{input_path}"'
    )

    if args.skip_gui:
        print(
            "Skipping GUI (--skip-gui). When ready to track:\n" + track_cmd,
            flush=True,
        )
        return

    exit_code = launch_correction_gui(frame_paths)
    print(
        "GUI closed. If you saved mask edits, track with:\n" + track_cmd,
        flush=True,
    )
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
