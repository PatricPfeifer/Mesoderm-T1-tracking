#!/usr/bin/env python3
"""Headless napariCellFlow / Cellpose 4 (cpsam) segmentation.

Cellpose 4 port of the napariCellFlow.SegmentationHandler eval settings (cyto3 -> cpsam),
without the napari GUI.

Example (tmux on a GPU allocation, conda env cellflow):
  python scripts/cellflow/run_cellflow_segment.py \\
    --input "input files/Cri_WT_GAP43_GFP_38sec_Naima_cut.tif" \\
    --output-dir output/cellflow/naima_cut
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from cellpose import models
from tifffile import imread, imwrite


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="napariCellFlow-style Cellpose segmentation.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-type", default="cpsam")
    p.add_argument("--diameter", type=float, default=95.0)
    p.add_argument(
        "--fixed-diameter",
        action="store_true",
        help="Use --diameter as-is (Cellflow default is auto-compute).",
    )
    p.add_argument("--flow-threshold", type=float, default=0.6)
    p.add_argument("--cellprob-threshold", type=float, default=0.3)
    p.add_argument("--min-size", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--force-cpu", action="store_true")
    return p.parse_args()


def as_stack(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img[np.newaxis, ...]
    if img.ndim == 3:
        return img
    raise SystemExit(f"expected 2D or T,Y,X, got shape {img.shape}")


def main() -> None:
    args = parse_args()
    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    outdir = args.output_dir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{inp.stem}_cf_masks.tif"
    partial = outdir / f"{inp.stem}_cf_masks_partial.npy"
    prog = outdir / "progress.txt"
    meta_path = outdir / f"{inp.stem}_cf_seg_qc.json"

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
        masks = np.zeros((n_frames, ly, lx), dtype=np.uint16)
        start = 0
        print(f"starting fresh: {n_frames} frames", flush=True)

    use_gpu = not args.force_cpu
    print(f"loading Cellpose {args.model_type} gpu={use_gpu}", flush=True)
    model = models.CellposeModel(pretrained_model=args.model_type, gpu=use_gpu)
    diameter = None if not args.fixed_diameter else args.diameter
    counts: list[int] = []

    for t in range(start, n_frames):
        print(f"frame {t + 1}/{n_frames}", flush=True)
        m, _flows, _styles = model.eval(
            img[t],
            diameter=diameter,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            min_size=args.min_size,
            batch_size=args.batch_size,
            normalize=True,
        )
        masks[t] = np.asarray(m, dtype=np.uint16)
        nlab = int(masks[t].max())
        counts.append(nlab)
        np.save(partial, masks)
        prog.write_text(str(t + 1))
        print(f"  labels={nlab}", flush=True)

    imwrite(out, masks)
    partial.unlink(missing_ok=True)
    prog.unlink(missing_ok=True)
    meta = {
        "input": str(inp),
        "output": str(out),
        "shape": list(masks.shape),
        "max_label": int(masks.max()),
        "cells_per_frame": counts,
        "model_type": args.model_type,
        "diameter": diameter,
        "flow_threshold": args.flow_threshold,
        "cellprob_threshold": args.cellprob_threshold,
        "min_size": args.min_size,
        "gpu": use_gpu,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", out, "max_label", int(masks.max()), flush=True)


if __name__ == "__main__":
    main()
