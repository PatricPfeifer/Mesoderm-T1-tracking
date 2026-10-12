#!/usr/bin/env python3
"""Complement of gated detect: events kept only by pre-matching-gate logic.

Runs run_tm_t1_pre_matching_gate.run_t1_stack, drops any event already in the
current gated detect CSV, and writes complementary CSV + TIFFs to a separate
folder. Does not modify detect/ or picker_ready.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
for _p in (_HERE, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_tm_t1_pre_matching_gate as preg  # noqa: E402
from napari_patric.core.output_names import input_stem, tagged_path  # noqa: E402
from napari_patric.core.t1.run_tm_fourway_vertices import fmt_ids, load_labels  # noqa: E402


def parse_ids(text: str) -> frozenset[int]:
    parts = [p for p in str(text).replace(",", ";").split(";") if p.strip()]
    return frozenset(int(p) for p in parts)


def event_key(e: dict) -> tuple:
    """Identity for set-diff (order-insensitive pairs)."""
    if "quartet" in e:
        quartet = frozenset(e["quartet"])
        breaking = frozenset(e["pair_before"])
        building = frozenset(e["pair_after"])
        kind = str(e.get("kind", "completed"))
    else:
        quartet = frozenset(
            (int(e["A"]), int(e["B"]), int(e["C"]), int(e["D"]))
        )
        breaking = parse_ids(e["pair_breaking"])
        building = parse_ids(e["pair_building"])
        kind = str(e.get("kind", "completed"))
    return (quartet, breaking, building, kind)


def load_current_keys(path: Path) -> set[tuple]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {event_key(r) for r in rows}


def write_events_csv(
    path: Path,
    events: list[dict],
    frame_indices: np.ndarray,
) -> None:
    n = len(frame_indices)
    fields = [
        "frame_start",
        "frame_end",
        "frame_first_swap",
        "frame_last_swap",
        "frame_before",
        "frame_after",
        "A",
        "B",
        "C",
        "D",
        "pair_breaking",
        "pair_building",
        "y",
        "x",
        "source",
        "kind",
        "type",
        "persist_pre",
        "persist_post",
        "n_swaps",
        "has_fourway",
    ]

    def fi(t: int) -> int:
        return int(frame_indices[max(0, min(int(t), n - 1))])

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for e in events:
            writer.writerow(
                {
                    "frame_start": fi(e["t_start"]),
                    "frame_end": fi(e["t_end"]),
                    "frame_first_swap": fi(e.get("t_first_swap", e["t_start"])),
                    "frame_last_swap": fi(e.get("t_last_swap", e["t_end"])),
                    "frame_before": fi(e["t_before"]),
                    "frame_after": fi(e["t_after"]),
                    "A": e["A"],
                    "B": e["B"],
                    "C": e["C"],
                    "D": e["D"],
                    "pair_breaking": fmt_ids(e["pair_before"]),
                    "pair_building": fmt_ids(e["pair_after"]),
                    "y": round(float(e["y"]), 2),
                    "x": round(float(e["x"]), 2),
                    "source": e.get("source", "tm"),
                    "kind": e.get("kind", "completed"),
                    "type": preg.event_type(
                        str(e.get("kind", "completed")),
                        int(e.get("has_fourway", 1)),
                        int(e.get("n_swaps", 0)),
                    ),
                    "persist_pre": int(e.get("persist_pre", 0)),
                    "persist_post": int(e.get("persist_post", 0)),
                    "n_swaps": int(e.get("n_swaps", 0)),
                    "has_fourway": int(e.get("has_fourway", 1)),
                }
            )


def require_compute() -> None:
    if os.environ.get("ALLOW_LOGIN_NODE") == "1":
        print("ALLOW_LOGIN_NODE=1 — skipping compute-node check", flush=True)
        return
    job = os.environ.get("SLURM_JOB_ID")
    host = socket.gethostname()
    if not job:
        raise SystemExit(
            f"Refuse: not inside a Slurm job (host={host}). "
            "Run from tmux on a compute allocation."
        )
    if "login" in host or re.match(r"^uc\d+(\.|$)", host):
        raise SystemExit(
            f"Refuse: login node ({host}, job={job}). "
            "Attach tmux on the compute node, then re-run."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Other-matching / pre-gate-only T1s: complementary to gated detect."
        )
    )
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--rgb", type=Path, default=None)
    p.add_argument(
        "--current-events",
        type=Path,
        required=True,
        help="Gated detect t1_events.csv (current run_tm_t1.py output).",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--frames", type=str, default=None)
    p.add_argument("--expand", type=int, default=3)
    p.add_argument("--bond-cutoff", type=float, default=3.0)
    p.add_argument("--border-rings", type=int, default=1)
    p.add_argument("--search", type=int, default=3)
    p.add_argument("--pre", type=int, default=3)
    p.add_argument("--post", type=int, default=3)
    p.add_argument("--length-window", type=int, default=0)
    p.add_argument("--no-cellflow", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    require_compute()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = input_stem(args.labels)
    current_path = args.current_events.expanduser().resolve()
    if not current_path.is_file():
        raise SystemExit(f"Missing current events CSV: {current_path}")

    print(f"Loading current (gated) keys from {current_path}", flush=True)
    current_keys = load_current_keys(current_path)
    print(f"current events={len(current_keys)}", flush=True)

    print(f"Loading {args.labels}", flush=True)
    full = load_labels(args.labels.expanduser())
    f0, f1 = preg.parse_frames(args.frames, full.shape[0])
    stack = full[f0:f1]
    print(f"slice {f0}:{f1} -> {stack.shape}", flush=True)
    rgb = None
    if args.rgb is not None and args.rgb.expanduser().is_file():
        print(f"Loading RGB {args.rgb}", flush=True)
        rgb = preg.load_rgb(
            args.rgb.expanduser(),
            np.arange(f0, f1),
            stack.shape[1],
            stack.shape[2],
        )

    print("Running pre-matching-gate detect…", flush=True)
    result = preg.run_t1_stack(
        stack,
        rgb=rgb,
        frame_base=f0,
        expand=args.expand,
        bond_cutoff=args.bond_cutoff,
        border_rings=args.border_rings,
        search=args.search,
        length_window=args.length_window,
        pre=args.pre,
        post=args.post,
        include_cellflow=not args.no_cellflow,
    )
    pre_events = list(result["events"])
    pre_keys = {event_key(e) for e in pre_events}
    inter = pre_keys & current_keys
    complement = [e for e in pre_events if event_key(e) not in current_keys]
    complement.sort(key=lambda e: (e["t_start"], e["A"], e["B"]))

    only_current = current_keys - pre_keys
    print(
        f"pre_gate={len(pre_events)} current={len(current_keys)} "
        f"intersection={len(inter)} complement={len(complement)} "
        f"current_only={len(only_current)}",
        flush=True,
    )
    if len(current_keys) + len(complement) != len(pre_events):
        print(
            "NOTE: |current|+|complement| != |pre| "
            f"({len(current_keys)}+{len(complement)}!={len(pre_events)}); "
            f"intersection={len(inter)} current_only={len(only_current)} "
            "(flicker merge / key mismatch).",
            flush=True,
        )

    labels_list = [result["stack"][t] for t in range(result["n_frames"])]
    overlay = preg.render_overlay(
        result["overlay_base"],
        labels_list,
        complement,
        result["four_xy"],
        pre=int(result.get("pre", args.pre)),
        post=int(result.get("post", args.post)),
        reach=int(result.get("expand", args.expand)),
    )
    dots = preg.render_fourway_dots(
        result["stack"], complement, result["four_xy"], rgb=result["overlay_base"]
    )

    ev_path = tagged_path(out, stem, "other_matching_t1_events.csv")
    write_events_csv(ev_path, complement, result["frame_indices"])
    print(f"Wrote {ev_path}", flush=True)

    dots_path = tagged_path(out, stem, "other_matching_t1_fourways.tif")
    tifffile.imwrite(dots_path, dots)
    print(f"Wrote {dots_path}", flush=True)

    ov_path = tagged_path(out, stem, "other_matching_t1_overlay.tif")
    tifffile.imwrite(ov_path, overlay)
    print(f"Wrote {ov_path}", flush=True)

    by_kind = Counter(str(e.get("kind")) for e in complement)
    by_source = Counter(str(e.get("source")) for e in complement)
    by_type = Counter(
        preg.event_type(
            str(e.get("kind", "completed")),
            int(e.get("has_fourway", 1)),
            int(e.get("n_swaps", 0)),
        )
        for e in complement
    )
    n_no4 = sum(1 for e in complement if int(e.get("has_fourway", 1)) == 0)
    summary = {
        "stem": stem,
        "current_events": str(current_path),
        "n_pre_gate": len(pre_events),
        "n_current": len(current_keys),
        "n_intersection": len(inter),
        "n_complement": len(complement),
        "n_current_only": len(only_current),
        "sum_check_ok": (
            len(current_keys) + len(complement) == len(pre_events)
            and len(inter) == len(current_keys)
            and len(only_current) == 0
        ),
        "complement_by_kind": dict(by_kind),
        "complement_by_source": dict(by_source),
        "complement_by_type": dict(by_type),
        "complement_no_fourway": n_no4,
        "labels": str(args.labels.resolve()),
        "rgb": str(args.rgb.resolve()) if args.rgb else None,
    }
    sum_path = out / "complement_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {sum_path}", flush=True)

    readme = out / "README.txt"
    readme.write_text(
        "Other-matching / pre-gate complementary T1s\n"
        "==========================================\n\n"
        "These events appear under the pre-matching-gate detector\n"
        "(run_tm_t1_pre_matching_gate.py) but not in the current gated detect\n"
        "CSV. Together with gated detect they reconstruct the pre-gate set\n"
        "(see complement_summary.json).\n\n"
        "TIFFs use the same coloring as detect t1_overlay / t1_fourways,\n"
        "but only complementary events are drawn.\n\n"
        f"Gated CSV: {current_path}\n"
        f"Complement events: {ev_path.name}\n"
    )
    print(f"Wrote {readme}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
