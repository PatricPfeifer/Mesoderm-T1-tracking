#!/usr/bin/env python3
"""T1 overlay from CellFlow *_cf_t1_events.csv (same paint as run_tm_t1).

Grayscale base, orange = losing contact, cyan = gaining contact, red disk
at the CellFlow event frame.

Example:
  python scripts/cellflow/run_cellflow_t1_overlay.py \\
    --t1-dir output/cellflow/naima_cut_ta_track_t1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

_SCRIPTS = Path(__file__).resolve().parents[1]
_T1 = _SCRIPTS / "t1"
for _p in (_T1, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from napari_tissuekit.core.t1.run_tm_fourway_vertices import load_labels  # noqa: E402
from napari_tissuekit.core.t1.run_tm_t1 import (  # noqa: E402
    grayscale_rgb,
    labels_gray_rgb,
    load_rgb,
    render_overlay,
)


def load_cf_events(path: Path) -> list[dict]:
    events: list[dict] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            a, b = int(row["losing_a"]), int(row["losing_b"])
            c, d = int(row["gaining_c"]), int(row["gaining_d"])
            y = float(row["y"]) if row.get("y") not in (None, "") else -1.0
            x = float(row["x"]) if row.get("x") not in (None, "") else -1.0
            ab = frozenset((a, b))
            cd = frozenset((c, d))
            events.append(
                {
                    "t_start": frame,
                    "t_end": frame,
                    "pair_before": ab,
                    "pair_after": cd,
                    "quartet": ab | cd,
                    "y": y,
                    "x": x,
                }
            )
    return events


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paint CellFlow T1 events like run_tm_t1 overlays."
    )
    p.add_argument("--t1-dir", type=Path, required=True)
    p.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Tracked labels. Default: 'input' from *_cf_t1_qc.json.",
    )
    p.add_argument(
        "--rgb",
        type=Path,
        default=None,
        help="tracked_rgb.tif for grayscale base. Default: sibling of labels.",
    )
    p.add_argument("--pre", type=int, default=3)
    p.add_argument("--post", type=int, default=3)
    p.add_argument("--reach", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t1_dir = args.t1_dir.expanduser().resolve()
    if not t1_dir.is_dir():
        raise SystemExit(f"t1-dir not found: {t1_dir}")

    qc_paths = sorted(t1_dir.glob("*_cf_t1_qc.json"))
    csv_paths = sorted(t1_dir.glob("*_cf_t1_events.csv"))
    if not csv_paths:
        raise SystemExit(f"no *_cf_t1_events.csv in {t1_dir}")
    events_path = csv_paths[0]
    qc = {}
    if qc_paths:
        qc = json.loads(qc_paths[0].read_text())

    labels_path = args.labels
    if labels_path is None:
        if not qc.get("input"):
            raise SystemExit("pass --labels (no input in qc json)")
        labels_path = Path(qc["input"])
    labels_path = labels_path.expanduser().resolve()
    if not labels_path.is_file():
        raise SystemExit(f"labels not found: {labels_path}")

    rgb_path = args.rgb
    if rgb_path is None:
        guess = labels_path.with_name(
            labels_path.name.replace("_tracked_labels.tif", "_tracked_rgb.tif")
        )
        rgb_path = guess if guess.is_file() else None
    else:
        rgb_path = rgb_path.expanduser().resolve()
        if not rgb_path.is_file():
            raise SystemExit(f"rgb not found: {rgb_path}")

    print("events", events_path, flush=True)
    print("labels", labels_path, flush=True)
    print("rgb", rgb_path if rgb_path else "(gray from labels)", flush=True)

    events = load_cf_events(events_path)
    print("n_events", len(events), flush=True)
    stack = load_labels(labels_path)
    n, h, w = stack.shape
    print("shape", stack.shape, flush=True)

    frame_indices = np.arange(n, dtype=np.int32)
    if rgb_path is not None:
        rgb = load_rgb(rgb_path, frame_indices, h, w)
    else:
        max_lab = int(stack.max()) if stack.size else 0
        rgb = np.empty((n, h, w, 3), dtype=np.uint8)
        for t in range(n):
            rgb[t] = labels_gray_rgb(stack[t], max_lab)

    four_xy: list[dict] = [{} for _ in range(n)]
    labels_list = [stack[t] for t in range(n)]
    print("rendering overlay...", flush=True)
    overlay = render_overlay(
        rgb,
        labels_list,
        events,
        four_xy,
        pre=args.pre,
        post=args.post,
        reach=args.reach,
    )
    stem = events_path.name.replace("_cf_t1_events.csv", "")
    out_path = t1_dir / f"{stem}_cf_t1_overlay.tif"
    tifffile.imwrite(out_path, overlay)
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
