#!/usr/bin/env python3
"""Minimal Tissue Analyzer: optional EPySeg segmentation + tracking (no GUI).

Modes:
  1) Segment + track (EPySeg → handCorrection → track):
       python run_ta_track_only.py \\
         --input "input files/Cri_WT_GAP43_GFP_38sec_Naima_cut.tif" \\
         --output-dir output/ta_track/minimal_naima_cut

  2) Track only (existing frames/ + handCorrection.tif):
       python run_ta_track_only.py --output-dir output/ta_track/hermetia_ctrl_lateral2

Writes:
  <output-dir>/tracked_cells_rgb.tif
  <output-dir>/tracked_cells_ids.tif
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BG_COLORS = {0, 0xFFFFFF}
_FRAME_RE = re.compile(r"^t(\d+)\.(?:tif|tiff)$", re.IGNORECASE)


def collect_frame_paths(frames_dir: Path) -> list[str]:
    """Ordered list of top-level intensity frame TIFFs under frames/."""
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    frames: list[tuple[int, Path]] = []
    for p in frames_dir.iterdir():
        if not p.is_file():
            continue
        m = _FRAME_RE.match(p.name)
        if m:
            frames.append((int(m.group(1)), p))

    if not frames:
        try:
            from natsort import natsorted

            paths = natsorted(
                [p for p in frames_dir.glob("*.tif") if p.is_file()]
                + [p for p in frames_dir.glob("*.tiff") if p.is_file()],
                key=lambda x: x.name,
            )
        except ImportError:
            paths = sorted(
                list(frames_dir.glob("*.tif")) + list(frames_dir.glob("*.tiff")),
                key=lambda x: x.name,
            )
        paths = [p for p in paths if p.is_file()]
        if not paths:
            raise FileNotFoundError(
                f"No frame TIFFs found in {frames_dir} (expected t0000.tif, ...)."
            )
        return [str(p.resolve()) for p in paths]

    frames.sort(key=lambda x: x[0])
    return [str(p.resolve()) for _, p in frames]


def validate_hand_corrections(frame_paths: list[str]) -> None:
    missing = [
        p
        for p in frame_paths
        if not (Path(p).with_suffix("") / "handCorrection.tif").exists()
    ]
    if missing:
        example = Path(missing[0]).with_suffix("") / "handCorrection.tif"
        raise FileNotFoundError(
            f"Missing handCorrection.tif for {len(missing)}/{len(frame_paths)} "
            f"frames (e.g. {example})."
        )


def run_epyseg_on_input(
    input_path: Path,
    frames_dir: Path,
    *,
    force_cpu: bool,
    channel: int | None,
    tile_size: int,
    tile_overlap: int,
    size_filter: int | None,
) -> list[str]:
    """Write frames from intensity TIFF and run EPySeg (resume missing masks)."""
    from ta_common import (
        detect_use_cpu,
        load_stack,
        run_epyseg_segmentation,
        write_frames,
        write_list_lst,
    )

    print(f"Loading intensity {input_path}", flush=True)
    stack = load_stack(input_path)
    print(f"intensity shape={stack.shape} dtype={stack.dtype}", flush=True)

    if not frames_dir.is_dir() or not any(frames_dir.glob("t*.tif")):
        print(f"Writing {stack.shape[0]} frames under {frames_dir}", flush=True)
        frame_paths = write_frames(stack, frames_dir)
    else:
        frame_paths = collect_frame_paths(frames_dir)
        print(f"Reusing existing frames under {frames_dir}", flush=True)

    missing = [
        p
        for p in frame_paths
        if not (Path(p).with_suffix("") / "handCorrection.tif").exists()
    ]
    print(
        f"EPySeg: {len(missing)}/{len(frame_paths)} frames need segmentation",
        flush=True,
    )

    use_cpu, gpus = detect_use_cpu(force_cpu=force_cpu)
    device = "CPU" if use_cpu else "GPU"
    print(f"device={device} gpus={gpus}", flush=True)

    if missing:
        print("Running EPySeg segmentation (TA mode)...", flush=True)
        run_epyseg_segmentation(
            missing,
            use_cpu=use_cpu,
            channel=channel,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            size_filter=size_filter,
        )

    validate_hand_corrections(frame_paths)
    list_path = write_list_lst(frame_paths, frames_dir)
    print(
        f"Segmentation done. frames={len(frame_paths)} device={device}\n"
        f"list.lst: {list_path}",
        flush=True,
    )
    return frame_paths


def run_tracking(frame_paths: list[str], channel: int | None) -> None:
    from epyseg.ta.tracking.local_to_track_correspondance import (
        add_localID_to_trackID_correspondance_in_DB,
    )
    from epyseg.ta.tracking.tracking_yet_another_approach_pyramidal_registration_n_neo_swapping_correction import (
        track_cells_dynamic_tissue,
    )

    print("Tracking cells (dynamic tissue)...", flush=True)
    track_cells_dynamic_tissue(frame_paths, channel=channel)
    print("Building local-ID ↔ track-ID correspondence...", flush=True)
    add_localID_to_trackID_correspondance_in_DB(frame_paths)


def rgb24_to_dense_ids(rgb) -> "np.ndarray":
    """Map RGB24 track colors to dense uint32 labels (0 = background)."""
    import numpy as np
    from epyseg.img import RGB_to_int24

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected H,W,3 RGB track image; got {rgb.shape}")
    codes = RGB_to_int24(rgb)
    unique = np.unique(codes)
    keep = [int(c) for c in unique if int(c) not in BG_COLORS]
    mapping = {c: i + 1 for i, c in enumerate(keep)}
    out = np.zeros(codes.shape, dtype=np.uint32)
    for c, idx in mapping.items():
        out[codes == c] = idx
    return out


def export_track_stacks(
    frame_paths: list[str], output_dir: Path
) -> tuple[Path, Path]:
    import numpy as np
    import tifffile

    rgb_frames: list = []
    id_frames: list = []

    for path in frame_paths:
        stem_dir = Path(path).with_suffix("")
        track_path = stem_dir / "tracked_cells_resized.tif"
        if not track_path.exists():
            raise FileNotFoundError(
                f"Missing track image after tracking: {track_path}"
            )
        rgb = np.asarray(tifffile.imread(track_path))
        if rgb.ndim == 2:
            raise ValueError(
                f"Expected RGB tracked_cells_resized.tif at {track_path}, "
                f"got shape {rgb.shape}"
            )
        if rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        rgb_frames.append(rgb.astype(np.uint8, copy=False))
        id_frames.append(rgb24_to_dense_ids(rgb))

    rgb_stack = np.stack(rgb_frames, axis=0)
    id_stack = np.stack(id_frames, axis=0)

    rgb_out = output_dir / "tracked_cells_rgb.tif"
    ids_out = output_dir / "tracked_cells_ids.tif"
    tifffile.imwrite(rgb_out, rgb_stack)
    tifffile.imwrite(ids_out, id_stack)
    return rgb_out, ids_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Minimal TA: optional EPySeg segmentation + tracking "
            "(no GUI). Pass --input to segment, or only --output-dir to track."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Intensity TIFF (T,Y,X). If set, run EPySeg segmentation into "
            "--output-dir/frames/ before tracking."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="TA run directory (frames/ + outputs).",
    )
    p.add_argument(
        "--channel",
        type=int,
        default=None,
        help="Channel index for multichannel frames (default: None / first).",
    )
    p.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force EPySeg/TensorFlow onto CPU even if a GPU is available.",
    )
    p.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="EPySeg tile width/height (default: 256).",
    )
    p.add_argument(
        "--tile-overlap",
        type=int,
        default=32,
        help="EPySeg tile overlap (default: 32).",
    )
    p.add_argument(
        "--size-filter",
        type=int,
        default=None,
        help="Optional min cell area (pixels) for EPySeg post-process.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"

    if args.input is not None:
        input_path = args.input.expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")
        frame_paths = run_epyseg_on_input(
            input_path,
            frames_dir,
            force_cpu=args.force_cpu,
            channel=args.channel,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            size_filter=args.size_filter,
        )
    else:
        if not frames_dir.is_dir():
            raise FileNotFoundError(
                f"Frames directory not found: {frames_dir}. "
                "Pass --input to segment, or prepare frames/ + "
                "handCorrection.tif first."
            )
        frame_paths = collect_frame_paths(frames_dir)
        print(f"Found {len(frame_paths)} frames under {frames_dir}", flush=True)
        validate_hand_corrections(frame_paths)

    # Drop stale per-frame tracks so tracking rewrites them.
    for path in frame_paths:
        stale = Path(path).with_suffix("") / "tracked_cells_resized.tif"
        if stale.exists():
            stale.unlink()
    for name in ("tracked_cells_rgb.tif", "tracked_cells_ids.tif"):
        p = output_dir / name
        if p.exists():
            p.unlink()

    run_tracking(frame_paths, channel=args.channel)
    rgb_out, ids_out = export_track_stacks(frame_paths, output_dir)
    import numpy as np
    import tifffile

    n_ids = int(np.asarray(tifffile.imread(ids_out)).max())
    print(f"Done. frames={len(frame_paths)} max_track_id={n_ids}", flush=True)
    print(f"RGB tracks: {rgb_out}", flush=True)
    print(f"ID tracks:  {ids_out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
