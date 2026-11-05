#!/usr/bin/env python
"""Run Cellpose 2D segmentation on each frame of a T,Y,X TIFF stack.

Saves progress after every frame so a short Slurm walltime can be resumed.

Writes ``*_cp_masks.tif`` and ``*_cell_outlines.tif`` (1 px cell outlines, 255 on black; the plugin's outline file).

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

from .cell_size import MIN_CELL_AREA
from .holes import fill_enclosed_holes
from .outline_io import hand_stack_to_outline
from .ta_bond_edit import thin_hand_stack

from .movie_calibration import read_calibration, save_tiff  # noqa: E402


# Cellpose eval settings: named presets plus single values (see seg_params.py).
from .seg_params import BUILTIN_PRESETS as PRESETS  # noqa: E402
from .seg_params import all_presets, eval_kwargs  # noqa: E402


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
        default=None,
        help="256x256 px tiles per GPU pass (default: the preset's, else 8).",
    )
    p.add_argument(
        "--force-cpu",
        action="store_true",
        help="Disable GPU even if available.",
    )
    p.add_argument(
        "--no-fill-holes",
        dest="fill_holes",
        action="store_false",
        help="Do not give empty space enclosed by cells (not touching the image edge) a new cell id (default: it does).",
    )
    p.add_argument(
        "--preset",
        default="default",
        help="Named Cellpose settings: default, cellflow or one of your saved presets.",
    )
    p.add_argument(
        "--params",
        default=None,
        help="JSON of single Cellpose settings over the preset, e.g. {\"flow_threshold\": 0.6}.",
    )
    return p.parse_args(argv)


def resolve_all(preset: str, params_json: str | dict | None = None) -> dict:
    """Every setting (Cellpose values and batch size): the preset's, then ``params_json`` over them.

    An unknown preset name is fine when ``params_json`` is given (it is then only a label,
    e.g. a preset saved on another computer)."""
    import json

    from .seg_params import full_params

    extra = json.loads(params_json) if isinstance(params_json, str) else dict(params_json or {})
    presets = all_presets()
    if preset not in presets and not extra:
        raise SystemExit(f"unknown preset {preset!r}; known: {', '.join(presets)}")
    return full_params({**presets.get(preset, {}), **extra})


def resolve_params(preset: str, params_json: str | dict | None = None) -> dict:
    """Cellpose eval settings of ``preset`` + ``params_json`` (without the batch size)."""
    return eval_kwargs(resolve_all(preset, params_json))


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
    fill_holes: bool = True,
) -> np.ndarray:
    """Segment each frame with Cellpose cpsam. Shared by the CLI and napari plugin.

    ``fill_holes`` (on by default): every empty region enclosed by cells (touching no image edge) becomes a cell of its own,
    see :func:`napari_tissuekit.core.holes.fill_enclosed_holes`."""
    img = as_stack(img)
    eval_kwargs = {"min_size": MIN_CELL_AREA, **(eval_kwargs or {})}  # Cellpose drops regions below min_size px
    n_frames, ly, lx = img.shape
    if masks is None:
        masks = np.zeros((n_frames, ly, lx), dtype=np.uint16)
    elif masks.shape != (n_frames, ly, lx):
        raise ValueError(f"masks shape {masks.shape} != {(n_frames, ly, lx)}")
    print(f"loading model cpsam (gpu={use_gpu})", flush=True)
    model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
    for t in range(start, n_frames):
        print(f"frame {t + 1}/{n_frames}", flush=True)
        m = model.eval(img[t], batch_size=batch_size, **eval_kwargs)[0]
        masks[t] = np.asarray(m, dtype=np.uint16)
        if fill_holes:
            masks[t], n_new = fill_enclosed_holes(masks[t])
            if n_new:
                print(f"  filled {n_new} enclosed holes as new cells", flush=True)
        if on_frame is not None:
            on_frame(t, masks)
        print(f"  labels={int(masks[t].max())}", flush=True)
    return masks


def cell_outline_stack(masks: np.ndarray, progress=None) -> np.ndarray:
    """T,Y,X uint8 stack, 255 on the 1 px cell outlines and 0 elsewhere.

    The same file the napari plugin saves as ``*_cell_outlines.tif``: Tissue Analyzer style
    1 px bonds from a watershed of the masks, keeping only bond pixels that touch a cell.
    """
    hand = thin_hand_stack(masks, progress)
    return (hand_stack_to_outline(hand) > 0).astype(np.uint8) * 255


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
        batch_size=args.batch_size or resolve_all(args.preset, args.params)["batch_size"],
        use_gpu=not args.force_cpu,
        start=start,
        masks=masks,
        on_frame=_checkpoint,
        eval_kwargs=resolve_params(args.preset, args.params),
        fill_holes=args.fill_holes,
    )
    save_tiff(out, masks, cal)
    print("wrote", out, "max_label", int(masks.max()), flush=True)

    # Default outline-only output (as in the plugin). The masks are already on disk, so the
    # checkpoint goes first; a failure here still ends the run with an error.
    partial.unlink(missing_ok=True)
    prog.unlink(missing_ok=True)
    cell_path = outdir / f"{inp.stem}_cell_outlines.tif"
    last = [-10]

    def _progress(pct: int, msg: str) -> None:
        if pct >= last[0] + 10:
            last[0] = pct
            print(f"  {msg} ({pct}%)", flush=True)

    save_tiff(cell_path, cell_outline_stack(masks, _progress), cal)
    print("wrote", cell_path, flush=True)


if __name__ == "__main__":
    main()
