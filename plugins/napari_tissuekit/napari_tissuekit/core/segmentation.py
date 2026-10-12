#!/usr/bin/env python
"""Run Cellpose 2D segmentation on each frame of a T,Y,X TIFF stack.

Saves progress after every frame so a short Slurm walltime can be resumed.

Example (``tissuekit segment`` wraps this with the results-folder layout):
  python -m napari_tissuekit.core.segmentation \
    --input movie.tif --output-dir movie_results/segmentation
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from cellpose import models
from tifffile import imread, imwrite

from .outline_io import masks_to_outline_rgb

from .movie_calibration import read_calibration, save_tiff  # noqa: E402


# Cellpose eval settings. "cellflow" = the settings of the older napariCellFlow
# segmentation (auto diameter, stricter flow/cell-probability), used for the Egg1 HPC runs.
PRESETS: dict[str, dict] = {
    "default": {},
    "cellflow": {
        "diameter": None,
        "flow_threshold": 0.6,
        "cellprob_threshold": 0.3,
        "min_size": 25,
        "normalize": True,
    },
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cellpose 2D per-frame segmentation.")
    p.add_argument("--input", type=Path, required=True, help="T,Y,X TIFF stack.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: output/cellpose_seg/<stem>_cp_seg).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Cellpose eval batch_size (default: 8).",
    )
    p.add_argument(
        "--force-cpu",
        action="store_true",
        help="Disable GPU even if available.",
    )
    p.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="default",
        help="Cellpose eval settings (default: cpsam defaults).",
    )
    return p.parse_args(argv)


def as_stack(img: np.ndarray) -> np.ndarray:
    """Return a T,Y,X view. A single 2D frame becomes length 1."""
    img = np.asarray(img)
    if img.ndim == 2:
        return img[np.newaxis, ...]
    if img.ndim != 3:
        raise ValueError(f"expected T,Y,X stack, got shape {img.shape}")
    return img


def segment_stack(
    img: np.ndarray,
    *,
    batch_size: int = 8,
    use_gpu: bool = True,
    start: int = 0,
    masks: np.ndarray | None = None,
    on_frame: Callable[[int, np.ndarray], None] | None = None,
    eval_kwargs: dict | None = None,
) -> np.ndarray:
    """Segment each frame with Cellpose cpsam. Shared by the CLI and napari plugin."""
    img = as_stack(img)
    n_frames, ly, lx = img.shape
    if masks is None:
        masks = np.zeros((n_frames, ly, lx), dtype=np.uint16)
    elif masks.shape != (n_frames, ly, lx):
        raise ValueError(f"masks shape {masks.shape} != {(n_frames, ly, lx)}")
    print(f"loading model cpsam (gpu={use_gpu})", flush=True)
    model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
    for t in range(start, n_frames):
        print(f"frame {t + 1}/{n_frames}", flush=True)
        m = model.eval(img[t], batch_size=batch_size, **(eval_kwargs or {}))[0]
        masks[t] = np.asarray(m, dtype=np.uint16)
        if on_frame is not None:
            on_frame(t, masks)
        print(f"  labels={int(masks[t].max())}", flush=True)
    return masks


def main(argv=None) -> None:
    args = parse_args(argv)
    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    outdir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else Path("output/cellpose_seg") / f"{inp.stem}_cp_seg"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{inp.stem}_cp_masks.tif"
    partial = outdir / f"{inp.stem}_cp_masks_partial.npy"
    prog = outdir / "progress.txt"

    # Pixel size / frame interval of the raw movie, written into the outputs.
    cal = read_calibration(inp)
    print("calibration:", cal.as_dict() if cal else "none found", flush=True)
    print("loading", inp, flush=True)
    img = as_stack(np.asarray(imread(inp)))
    print("shape", img.shape, img.dtype, flush=True)

    n_frames, ly, lx = img.shape
    if partial.exists():
        masks = np.load(partial)
        start = int(prog.read_text().strip()) if prog.exists() else 0
        if masks.shape != (n_frames, ly, lx):
            raise SystemExit(f"partial shape {masks.shape} != {(n_frames, ly, lx)}")
        print(f"resuming from frame {start}/{n_frames}", flush=True)
    else:
        masks = None
        start = 0
        print(f"starting fresh: {n_frames} frames", flush=True)

    def _checkpoint(t: int, frame_masks: np.ndarray) -> None:
        np.save(partial, frame_masks)
        prog.write_text(str(t + 1))

    masks = segment_stack(
        img,
        batch_size=args.batch_size,
        use_gpu=not args.force_cpu,
        start=start,
        masks=masks,
        on_frame=_checkpoint,
        eval_kwargs=PRESETS[args.preset],
    )
    save_tiff(out, masks, cal)
    outlines_path = outdir / f"{inp.stem}_cp_outlines.tif"
    outlines = masks_to_outline_rgb(masks)
    save_tiff(outlines_path, outlines, cal)
    partial.unlink(missing_ok=True)
    prog.unlink(missing_ok=True)
    print("wrote", out, "max_label", int(masks.max()), flush=True)
    print("wrote", outlines_path, flush=True)


if __name__ == "__main__":
    main()
