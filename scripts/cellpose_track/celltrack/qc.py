"""QC metrics and forward/backward agreement."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .tracks import TrackingResult


def track_lengths(result: TrackingResult) -> dict[int, int]:
    """track_id -> number of frames present."""
    lengths: Counter[int] = Counter()
    for mapping in result.label_to_track:
        for track in set(mapping.values()):
            lengths[track] += 1
    return dict(lengths)


def compute_qc(
    result: TrackingResult,
    *,
    backward: TrackingResult | None = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable QC summary."""
    lengths = track_lengths(result)
    n_frames = len(result.frames)
    hist = Counter(lengths.values())
    full = sum(1 for v in lengths.values() if v == n_frames)

    appear = sum(1 for e in result.events if e.kind == "appear")
    disappear = sum(1 for e in result.events if e.kind == "disappear")
    divide = sum(1 for e in result.events if e.kind == "divide")
    divide_flagged = sum(
        1 for e in result.events if e.kind == "divide" and e.flagged
    )
    gap_close = sum(1 for e in result.events if "gap_close" in (e.note or ""))

    qc: dict[str, Any] = {
        "n_frames": n_frames,
        "n_tracks": result.n_tracks,
        "full_length_tracks": full,
        "mean_topology_score": float(np.mean(result.per_frame_scores[1:]))
        if len(result.per_frame_scores) > 1
        else 1.0,
        "per_frame_topology_score": [float(s) for s in result.per_frame_scores],
        "n_appear": appear,
        "n_disappear": disappear,
        "n_divide": divide,
        "n_divide_flagged": divide_flagged,
        "division_mode": result.division_mode,
        "n_gap_close": gap_close,
        "track_length_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "cells_per_frame": [fc.n_cells for fc in result.frames],
    }

    if backward is not None:
        qc["forward_backward"] = forward_backward_agreement(result, backward)

    return qc


def forward_backward_agreement(
    forward: TrackingResult, backward: TrackingResult
) -> dict[str, Any]:
    """Compare forward and reversed-backward track partitions per frame pair.

    For each consecutive pair, build the equivalence relation "same track" from
    both directions and measure the fraction of cell-pairs that agree.
    """
    n = len(forward.frames)
    # Backward result was run on reversed frames; remap index t_fwd <-> n-1-t.
    agreements: list[float] = []
    for t in range(n - 1):
        f_map_a = forward.label_to_track[t]
        f_map_b = forward.label_to_track[t + 1]
        # Backward: frame n-1-t is "later" in reverse time = earlier in forward.
        b_t = n - 1 - (t + 1)
        b_t1 = n - 1 - t
        if b_t < 0 or b_t1 >= n:
            continue
        b_map_a = backward.label_to_track[b_t1]  # corresponds to forward t
        b_map_b = backward.label_to_track[b_t]  # corresponds to forward t+1

        # Cell identity: original labels at each frame.
        # Pair agreement: for each matched pair in forward, check if the same
        # two labels share a track in backward.
        # Build forward pairs (lab_t, lab_t1) that share a track.
        inv_b: dict[int, list[int]] = {}
        for lab, track in f_map_b.items():
            inv_b.setdefault(track, []).append(lab)

        total = 0
        agree = 0
        for lab_a, track in f_map_a.items():
            partners = inv_b.get(track, [])
            for lab_b in partners:
                total += 1
                # In backward, do lab_a and lab_b share a track?
                ta = b_map_a.get(lab_a)
                tb = b_map_b.get(lab_b)
                if ta is not None and tb is not None and ta == tb:
                    agree += 1
        agreements.append(agree / total if total else 1.0)

    return {
        "mean_agreement": float(np.mean(agreements)) if agreements else 1.0,
        "per_pair_agreement": [float(a) for a in agreements],
    }


def write_qc(qc: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(qc, indent=2) + "\n")
