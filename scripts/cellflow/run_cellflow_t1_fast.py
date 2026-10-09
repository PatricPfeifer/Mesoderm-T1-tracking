#!/usr/bin/env python3
"""Speedy copy of run_cellflow_t1.py (original left unchanged).

Uses scripts/cellflow/fast_edge_analysis.py instead of vendor napariCellFlow.

Example:
  python scripts/cellflow/run_cellflow_t1_fast.py \\
    --input output/cellpose_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_cp_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_tracked_labels.tif \\
    --output-dir output/cellflow/naima_cut_ta_track_t1_fast
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from tifffile import imread

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fast_edge_analysis import EdgeAnalysisParams, FastEdgeAnalyzer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fast copy of napariCellFlow T1 detection on tracked labels."
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dilation-radius", type=int, default=1)
    p.add_argument("--min-overlap-pixels", type=int, default=5)
    p.add_argument("--min-edge-length", type=float, default=0.0)
    p.add_argument("--no-filter-isolated", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    outdir = args.output_dir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stem = inp.name.replace("_tracked_labels.tif", "").replace(".tif", "")

    print("loading", inp, flush=True)
    stack = np.asarray(imread(inp))
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    if stack.ndim != 3:
        raise SystemExit(f"expected T,Y,X labels, got {stack.shape}")
    print("shape", stack.shape, stack.dtype, "max", int(stack.max()), flush=True)

    params = EdgeAnalysisParams(
        dilation_radius=args.dilation_radius,
        min_overlap_pixels=args.min_overlap_pixels,
        min_edge_length=args.min_edge_length,
        filter_isolated=not args.no_filter_isolated,
    )
    params.validate()
    analyzer = FastEdgeAnalyzer(params)

    def progress(pct, msg):
        print(f"{pct}% {msg}", flush=True)

    results = analyzer.analyze_sequence(stack, progress_callback=progress)

    t1_path = outdir / f"{stem}_cf_t1_events.csv"
    edges_path = outdir / f"{stem}_cf_edge_lengths.csv"
    qc_path = outdir / f"{stem}_cf_t1_qc.json"

    seen = set()
    n_t1 = 0
    with t1_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "event_index",
                "frame",
                "losing_a",
                "losing_b",
                "gaining_c",
                "gaining_d",
                "y",
                "x",
                "edge_id",
            ]
        )
        idx = 0
        for edge_id, edge in results.edges.items():
            for ev in edge.intercalations:
                key = (ev.frame, tuple(ev.losing_cells), tuple(ev.gaining_cells))
                if key in seen:
                    continue
                seen.add(key)
                y, x = (
                    (float(ev.coordinates[0]), float(ev.coordinates[1]))
                    if ev.coordinates is not None
                    else ("", "")
                )
                w.writerow(
                    [
                        idx,
                        int(ev.frame),
                        int(ev.losing_cells[0]),
                        int(ev.losing_cells[1]),
                        int(ev.gaining_cells[0]),
                        int(ev.gaining_cells[1]),
                        y,
                        x,
                        int(edge_id),
                    ]
                )
                idx += 1
        n_t1 = idx

    with edges_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "edge_id",
                "frame",
                "cell_id_1",
                "cell_id_2",
                "signed_length_px",
                "has_intercalation",
            ]
        )
        for edge_id, edge in results.edges.items():
            t1_frames = {int(ev.frame) for ev in edge.intercalations}
            for i, frame in enumerate(edge.frames):
                pair = edge.cell_pairs[i]
                length = edge.lengths[i]
                w.writerow(
                    [
                        int(edge_id),
                        int(frame),
                        int(pair[0]),
                        int(pair[1]),
                        float(length),
                        int(frame) in t1_frames,
                    ]
                )

    n_edges = len(results.edges)
    n_t1_edges = sum(1 for e in results.edges.values() if e.intercalations)
    qc = {
        "input": str(inp),
        "shape": list(stack.shape),
        "n_edges": n_edges,
        "n_edges_with_t1": n_t1_edges,
        "n_t1_events": n_t1,
        "dilation_radius": args.dilation_radius,
        "min_overlap_pixels": args.min_overlap_pixels,
        "min_edge_length": args.min_edge_length,
        "filter_isolated": not args.no_filter_isolated,
        "t1_csv": str(t1_path),
        "edges_csv": str(edges_path),
        "fast_copy": True,
    }
    qc_path.write_text(json.dumps(qc, indent=2) + "\n")
    print("wrote", t1_path, "n_t1", n_t1, flush=True)
    print("wrote", edges_path, "n_edges", n_edges, flush=True)
    print("wrote", qc_path, flush=True)


if __name__ == "__main__":
    main()
