#!/usr/bin/env python
"""Run Cellpose 2D segmentation on each frame of a T,Y,X TIFF stack.

Saves progress after every frame so a short Slurm walltime can be resumed.

Writes ``*_cp_masks.tif`` and ``*_cell_outlines.tif`` (1 px cell outlines, 255 on black; the plugin's outline file).
The outlines are part of every segmentation (also in napari): there is no way to skip them.

Example (``tissuekit segment`` wraps this with the results-folder layout):
  python -m napari_tissuekit.core.segmentation \
    --input movie.tif --output-dir movie_results/segmentation
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from cellpose import models
from tifffile import imread, imwrite

from .cell_size import absorb_small_labels, min_cell_area, set_min_area_um2
from .holes import fill_enclosed_holes
from .ta_bond_edit import make_bonds

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
        "--min-area-um2",
        type=float,
        default=None,
        help="Smallest cell in square micrometres (default: 4). Smaller regions are not cells.",
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
    min_area: int | None = None,
) -> np.ndarray:
    """Segment each frame with Cellpose cpsam. Shared by the CLI and napari plugin.

    ``min_area`` (px): the smallest cell; None = ``cell_size.min_cell_area()`` (4 um^2 at the movie's pixel size once known).

    ``fill_holes`` (on by default): every empty region enclosed by cells (touching no image edge) becomes a cell of its own,
    see :func:`napari_tissuekit.core.holes.fill_enclosed_holes`."""
    img = as_stack(img)
    min_area = min_cell_area() if min_area is None else int(min_area)
    eval_kwargs = dict(eval_kwargs or {})
    eval_kwargs["min_size"] = max(int(eval_kwargs.get("min_size", 0)), min_area)  # Cellpose drops regions below min_size px
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
            masks[t], n_new = fill_enclosed_holes(masks[t], min_area=min_area)
            if n_new:
                print(f"  filled {n_new} enclosed holes as new cells", flush=True)
        masks[t] = absorb_small_labels(masks[t], min_area=min_area)[0]  # under the minimal cell size: neighbours take over (no empty spots)
        if on_frame is not None:
            on_frame(t, masks)
        print(f"  labels={int(masks[t].max())}", flush=True)
    return masks


def cell_outline_stack(masks: np.ndarray, progress=None) -> np.ndarray:
    """T,Y,X uint8 stack, 255 on the 1 px cell outlines and 0 elsewhere (``*_cell_outlines.tif``).

    The outline step of every segmentation (:func:`napari_tissuekit.core.ta_bond_edit.make_bonds`): Tissue Analyzer
    style 1 px bonds from a watershed of the masks, keeping only bond pixels that touch a cell.
    """
    return (make_bonds(masks, progress)[1] > 0).astype(np.uint8) * 255


def outline_progress_printer(prefix: str = "  ") -> Callable[[int, str], None]:
    """Progress callback for the outline step that prints every 10 %."""
    last = [-10]

    def _progress(pct: int, msg: str) -> None:
        if pct >= last[0] + 10:
            last[0] = pct
            print(f"{prefix}{msg} ({pct}%)", flush=True)

    return _progress


CHECKPOINT_EVERY = 25  # frames between resume-file writes (env TISSUEKIT_CHECKPOINT_EVERY)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.min_area_um2:
        set_min_area_um2(args.min_area_um2)
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

    # The resume file holds the whole masks stack, so rewriting it after every frame writes
    # N^2/2 frames in total (about 130 GB for 240 frames of 1024x2160). Write it every
    # CHECKPOINT_EVERY frames instead, and when Slurm sends SIGTERM at the time limit.
    done = [start]  # frames finished so far (masks[:done] are complete)
    saved = [start]

    def _write_checkpoint() -> None:
        if done[0] <= saved[0]:
            return
        tmp = partial.with_name(partial.stem + ".tmp.npy")  # replace atomically: a kill never leaves half a file
        np.save(tmp, current[0])
        os.replace(tmp, partial)
        prog.write_text(str(done[0]))
        saved[0] = done[0]

    current: list = [masks]
    every = max(1, int(os.environ.get("TISSUEKIT_CHECKPOINT_EVERY", CHECKPOINT_EVERY)))

    def _checkpoint(t: int, frame_masks: np.ndarray) -> None:
        current[0] = frame_masks
        done[0] = t + 1
        if done[0] % every == 0:
            _write_checkpoint()

    def _on_sigterm(signum, frame) -> None:
        print(f"signal {signum}: saving checkpoint at frame {done[0]}/{n_frames}", flush=True)
        if current[0] is not None:
            _write_checkpoint()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _on_sigterm)

    masks = segment_stack(
        img,
        batch_size=args.batch_size or resolve_all(args.preset, args.params)["batch_size"],
        use_gpu=not args.force_cpu,
        start=start,
        masks=masks,
        on_frame=_checkpoint,
        eval_kwargs=resolve_params(args.preset, args.params),
        fill_holes=args.fill_holes,
        min_area=min_cell_area(getattr(cal, "um_per_px", None)),
    )
    save_tiff(out, masks, cal)
    print("wrote", out, "max_label", int(masks.max()), flush=True)

    # Default outline-only output (as in the plugin). The masks are already on disk, so the
    # checkpoint goes first; a failure here still ends the run with an error.
    partial.unlink(missing_ok=True)
    prog.unlink(missing_ok=True)
    cell_path = outdir / f"{inp.stem}_cell_outlines.tif"
    save_tiff(cell_path, cell_outline_stack(masks, outline_progress_printer()), cal)
    print("wrote", cell_path, flush=True)


if __name__ == "__main__":
    main()
