#!/usr/bin/env python3
"""TissueMiner CountT1 logic on a celltrack label stack.

TissueMiner is R + SQLite over Tissue Analyzer databases. This script ports
topology/CountT1.R to Python so we can run it on Cellpose+celltrack output
without a TA sqlite file.

A TissueMiner T1 (often called a half-T1) is a neighbour relationship that
appears or disappears between two consecutive frames while BOTH cells still
exist. Divisions are not modelled here: celltrack IDs already persist the
parent on the better daughter.

Input: T,Y,X tracked_labels.tif (stable track IDs). Optional RGB overlay.
RGB-only stacks are rejected — they have no identity.

Example:
  conda activate tissue_miner
  python scripts/t1/run_tm_count_t1.py \\
    --labels output/cellpose_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_cp_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_tracked_labels.tif \\
    --rgb output/cellpose_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_cp_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_tracked_rgb.tif \\
    --output-dir output/t1/Cri_WT_GAP43_GFP_38sec_Naima_cut_t1/tissue_miner
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

# Allow running as `python scripts/t1/run_tm_count_t1.py`.
_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
_TRACK = _SCRIPTS / "cellpose_track"
for _p in (_HERE, _TRACK, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from celltrack.cells import build_adjacency  # noqa: E402
from output_names import input_stem, tagged_path  # noqa: E402


def load_labels(path: Path) -> np.ndarray:
    img = np.asarray(tifffile.imread(path))
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    if img.ndim == 4 and img.shape[-1] in (3, 4):
        raise ValueError(
            f"{path} looks like RGB (shape {img.shape}). "
            "TissueMiner T1s need stable track IDs. Pass tracked_labels.tif."
        )
    if img.ndim != 3:
        raise ValueError(f"Expected T,Y,X labels; got {img.shape} for {path}")
    return img.astype(np.uint32, copy=False)


def drop_blank(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = [t for t in range(stack.shape[0]) if int(stack[t].max()) > 0]
    if len(keep) < 2:
        raise ValueError(f"Need >=2 non-blank frames; kept {len(keep)}")
    idx = np.asarray(keep, dtype=np.int32)
    dropped = [t for t in range(stack.shape[0]) if t not in set(keep)]
    if dropped:
        print(f"Dropping blank frames {dropped}", flush=True)
    return stack[idx], idx


def centroids_and_ids(labels: np.ndarray) -> dict[int, tuple[float, float]]:
    """track_id -> (y, x) centroid."""
    from skimage.measure import regionprops

    out: dict[int, tuple[float, float]] = {}
    for region in regionprops(labels):
        out[int(region.label)] = (
            float(region.centroid[0]),
            float(region.centroid[1]),
        )
    return out


def neighbor_maps(stack: np.ndarray, expand: int) -> list[dict[int, set[int]]]:
    maps: list[dict[int, set[int]]] = []
    n = stack.shape[0]
    for t in range(n):
        adj = build_adjacency(stack[t], expand=expand)
        maps.append(adj)
        if t == 0 or t == n - 1 or (t + 1) % 20 == 0:
            deg = (
                sum(len(v) for v in adj.values()) / len(adj) if adj else 0.0
            )
            print(
                f"  adjacency frame {t}/{n - 1}: {len(adj)} cells, "
                f"avg_degree={deg:.2f}",
                flush=True,
            )
    return maps


def count_t1s(
    neigh: list[dict[int, set[int]]],
    frame_indices: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    """Port of CountT1.R neighbour gain/loss.

    Returns (event rows, per-frame summary rows).
    """
    events: list[dict] = []
    summaries: list[dict] = []

    for t in range(len(neigh) - 1):
        a, b = neigh[t], neigh[t + 1]
        ids_both = set(a) & set(b)
        n_gained = 0
        n_lost = 0
        n_cells_with_change = 0

        for cell in ids_both:
            na = a[cell] & ids_both
            nb = b[cell] & ids_both
            gained = nb - na
            lost = na - nb
            if not gained and not lost:
                continue
            n_cells_with_change += 1
            n_gained += len(gained)
            n_lost += len(lost)
            for other in sorted(gained):
                events.append(
                    {
                        "frame_from": int(frame_indices[t]),
                        "frame_to": int(frame_indices[t + 1]),
                        "cell_id": int(cell),
                        "neighbor_id": int(other),
                        "kind": "gain",
                    }
                )
            for other in sorted(lost):
                events.append(
                    {
                        "frame_from": int(frame_indices[t]),
                        "frame_to": int(frame_indices[t + 1]),
                        "cell_id": int(cell),
                        "neighbor_id": int(other),
                        "kind": "loss",
                    }
                )

        degrees = [len(a[c]) for c in a]
        summaries.append(
            {
                "frame": int(frame_indices[t]),
                "n_cells": len(a),
                "mean_neighbors": (
                    float(np.mean(degrees)) if degrees else 0.0
                ),
                "sum_t1_gained": n_gained,
                "sum_t1_lost": n_lost,
                "n_cells_with_t1": n_cells_with_change,
                # Each undirected neighbour change is counted twice (once per cell).
                "n_undirected_changes": (n_gained + n_lost) // 2,
            }
        )

    return events, summaries


def pair_t1_events(events: list[dict]) -> list[dict]:
    """Collapse complementary gain/loss pairs into undirected neighbour swaps.

    A TissueMiner half-T1 on cell A vs B is usually mirrored on B vs A.
    """
    groups: dict[tuple[int, int, frozenset], list[dict]] = defaultdict(list)
    for e in events:
        key = (
            e["frame_from"],
            e["frame_to"],
            frozenset((e["cell_id"], e["neighbor_id"])),
        )
        groups[key].append(e)

    rows = []
    for (f0, f1, pair), lst in groups.items():
        a, b = sorted(pair)
        kinds = {x["kind"] for x in lst}
        rows.append(
            {
                "frame_from": f0,
                "frame_to": f1,
                "cell_a": a,
                "cell_b": b,
                "kinds": "+".join(sorted(kinds)),
                "n_records": len(lst),
            }
        )
    rows.sort(key=lambda r: (r["frame_from"], r["cell_a"], r["cell_b"]))
    return rows


def overlay_rgb(
    rgb_path: Path,
    labels: np.ndarray,
    frame_indices: np.ndarray,
    pair_rows: list[dict],
    out_path: Path,
) -> None:
    rgb = np.asarray(tifffile.imread(rgb_path))
    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        rgb = rgb[np.newaxis, ...]
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected T,Y,X,3 RGB; got {rgb.shape}")

    # Align RGB to kept frames if it still includes a blank last plane.
    if rgb.shape[0] != labels.shape[0]:
        if rgb.shape[0] > int(frame_indices.max()) :
            rgb = rgb[frame_indices]
        else:
            print(
                f"RGB frames {rgb.shape[0]} != labels {labels.shape[0]}; "
                "skipping overlay.",
                flush=True,
            )
            return

    from skimage.draw import disk

    out = rgb.astype(np.uint8, copy=True)
    # Rec. 601 grayscale base so markers pop.
    luma = (
        0.299 * out[..., 0].astype(np.float32)
        + 0.587 * out[..., 1].astype(np.float32)
        + 0.114 * out[..., 2].astype(np.float32)
    )
    g = np.clip(luma, 0, 255).astype(np.uint8)
    base = np.stack([g, g, g], axis=-1)

    orig_to_t = {int(f): i for i, f in enumerate(frame_indices.tolist())}
    cents = [centroids_and_ids(labels[t]) for t in range(labels.shape[0])]

    yellow = (255, 255, 0)
    for row in pair_rows:
        t = orig_to_t.get(row["frame_from"])
        if t is None:
            continue
        for cid in (row["cell_a"], row["cell_b"]):
            cyx = cents[t].get(cid)
            if cyx is None:
                continue
            rr, cc = disk(
                (cyx[0], cyx[1]),
                3,
                shape=base[t].shape[:2],
            )
            base[t, rr, cc] = yellow

    tifffile.imwrite(out_path, base)
    print(f"Wrote overlay {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TissueMiner CountT1 neighbour-change detection on tracked labels."
    )
    p.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="T,Y,X tracked_labels.tif (stable track IDs).",
    )
    p.add_argument(
        "--rgb",
        type=Path,
        default=None,
        help="Optional tracked_rgb.tif for a yellow-dot overlay.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--expand",
        type=int,
        default=3,
        help="Grow labels by N px before adjacency (Cellpose gaps; default 3).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = input_stem(args.labels)

    print(f"Loading {args.labels}", flush=True)
    stack, frame_indices = drop_blank(load_labels(args.labels.expanduser()))
    print(f"labels shape={stack.shape} frames={list(frame_indices[[0, -1]])}", flush=True)

    print("Building neighbour graphs (TissueMiner CountT1)...", flush=True)
    neigh = neighbor_maps(stack, expand=args.expand)
    events, summaries = count_t1s(neigh, frame_indices)
    pairs = pair_t1_events(events)

    ev_path = tagged_path(out, stem, "t1_half_events.csv")
    with ev_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_from", "frame_to", "cell_id", "neighbor_id", "kind"],
        )
        writer.writeheader()
        writer.writerows(events)

    pair_path = tagged_path(out, stem, "t1_neighbour_changes.csv")
    with pair_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_from", "frame_to", "cell_a", "cell_b", "kinds", "n_records"],
        )
        writer.writeheader()
        writer.writerows(pairs)

    sum_path = tagged_path(out, stem, "t1_by_frame.csv")
    with sum_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()) if summaries else [])
        if summaries:
            writer.writeheader()
            writer.writerows(summaries)

    qc = {
        "n_frames": int(stack.shape[0]),
        "n_half_t1_records": len(events),
        "n_undirected_neighbour_changes": len(pairs),
        "mean_gained_per_interval": float(
            np.mean([s["sum_t1_gained"] for s in summaries])
        )
        if summaries
        else 0.0,
        "mean_lost_per_interval": float(
            np.mean([s["sum_t1_lost"] for s in summaries])
        )
        if summaries
        else 0.0,
        "expand": args.expand,
        "source": "vendor/tissue_miner/topology/CountT1.R",
        "labels": str(args.labels.resolve()),
    }
    (tagged_path(out, stem, "qc.json")).write_text(json.dumps(qc, indent=2) + "\n")

    print(
        f"half-T1 records={qc['n_half_t1_records']} "
        f"undirected_changes={qc['n_undirected_neighbour_changes']}",
        flush=True,
    )
    print(f"Wrote {ev_path}", flush=True)
    print(f"Wrote {pair_path}", flush=True)
    print(f"Wrote {sum_path}", flush=True)

    if args.rgb is not None and args.rgb.expanduser().is_file():
        overlay_rgb(
            args.rgb.expanduser(),
            stack,
            frame_indices,
            pairs,
            tagged_path(out, stem, "t1_overlay.tif"),
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
