#!/usr/bin/env python3
"""TissueMiner FourWayVertices logic on a celltrack label stack.

TissueMiner counts directed-bond multiplicity at each TA vertex and highlights
anything other than a 3-cell junction (FourWayVertices.R). We do not have that
sqlite, so this finds junctions on tracked_labels.tif:

  - expand Cellpose gaps
  - unique cell IDs in a 3x3 window
  - connected blobs with max unique >= 4 are N-way vertices
  - two 3-cell vertices closer than --bond-cutoff whose cell sets union to 4
    also count as a 4-way (collapsing bond / T1 intermediate)

Junction length: the T1 bond of a 4-cell quartet is the pair shared by both
3-cell vertices (intersection of the two triples). We record
  length_vertex_px  — Euclidean distance between those vertices (0 at a
                      true 4-way point)
  length_contact_px — 4-connected interface length between that pair
and follow every 4-cell quartet through the whole movie while all four
track IDs persist (so a bond can grow or shrink, not only the short 4-way
snapshot). The complementary pair is the swap bond after a T1.

3-cell vertices are not drawn (they dominate a hex lattice). Overlay colours
match TissueMiner: 4=red, 5=darkgreen, 6=darkblue. Short 4-way bonds are
also drawn as a yellow line of the measured vertex length.

Example:
  conda activate tissue_miner
  python scripts/t1/run_tm_fourway_vertices.py \\
    --labels output/cellpose_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_cp_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_tracked_labels.tif \\
    --rgb output/cellpose_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_cp_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_tracked_rgb.tif \\
    --output-dir output/t1/Cri_WT_GAP43_GFP_38sec_Naima_cut_t1/fourway
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
from scipy import ndimage as ndi
from skimage.draw import disk, line
from skimage.segmentation import expand_labels

from ..output_names import input_stem, tagged_path  # noqa: E402

# TissueMiner FourWayVertices.R (3-way skipped in the overlay)
MULT_COLORS = {
    4: (255, 0, 0),
    5: (0, 100, 0),
    6: (0, 0, 139),
}


def load_labels(path: Path) -> np.ndarray:
    img = np.asarray(tifffile.imread(path))
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    if img.ndim == 4 and img.shape[-1] in (3, 4):
        raise ValueError(
            f"{path} looks like RGB (shape {img.shape}). Pass tracked_labels.tif."
        )
    if img.ndim != 3:
        raise ValueError(f"Expected T,Y,X labels; got {img.shape} for {path}")
    return img.astype(np.uint32, copy=False)


def drop_blank(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = [t for t in range(stack.shape[0]) if int(stack[t].max()) > 0]
    if len(keep) < 1:
        raise ValueError("No non-blank frames")
    idx = np.asarray(keep, dtype=np.int32)
    dropped = [t for t in range(stack.shape[0]) if t not in set(keep)]
    if dropped:
        print(f"Dropping blank frames {dropped}", flush=True)
    return stack[idx], idx


def unique_count_3x3(lab: np.ndarray) -> np.ndarray:
    """Number of distinct positive labels in each 3x3 neighbourhood."""
    h, w = lab.shape
    pad = np.pad(lab, 1, mode="constant")
    stacks = np.empty((9, h, w), dtype=lab.dtype)
    k = 0
    for dy in range(3):
        for dx in range(3):
            stacks[k] = pad[dy : dy + h, dx : dx + w]
            k += 1
    stacks.sort(axis=0)
    prev = np.zeros((h, w), dtype=lab.dtype)
    count = np.zeros((h, w), dtype=np.uint8)
    for k in range(9):
        cur = stacks[k]
        nz = cur != 0
        count += (nz & (cur != prev)).astype(np.uint8)
        prev = np.where(nz, cur, prev)
    return count


def _component_records(
    mask: np.ndarray,
    expanded: np.ndarray,
    count: np.ndarray,
) -> list[dict]:
    labeled, n = ndi.label(mask)
    if n == 0:
        return []
    rows = []
    for cid in range(1, n + 1):
        ys, xs = np.where(labeled == cid)
        if ys.size == 0:
            continue
        y = float(ys.mean())
        x = float(xs.mean())
        yi, xi = int(round(y)), int(round(x))
        yi = min(max(yi, 0), expanded.shape[0] - 1)
        xi = min(max(xi, 0), expanded.shape[1] - 1)
        # Labels that actually meet in this blob (3x3 around component pixels).
        ids: set[int] = set()
        h, w = expanded.shape
        for yy, xx in zip(ys, xs):
            y0, y1 = max(yy - 1, 0), min(yy + 2, h)
            x0, x1 = max(xx - 1, 0), min(xx + 2, w)
            patch = expanded[y0:y1, x0:x1]
            ids.update(int(v) for v in np.unique(patch) if v != 0)
        multiplicity = max(int(count[ys, xs].max()), len(ids))
        rows.append(
            {
                "y": y,
                "x": x,
                "multiplicity": multiplicity,
                "n_cells": len(ids),
                "cell_ids": tuple(sorted(ids)),
            }
        )
    return rows


def fmt_ids(ids) -> str:
    return ";".join(str(i) for i in sorted(ids))


def contact_length_map(lab: np.ndarray) -> dict[tuple[int, int], float]:
    """4-connected interface length for every touching label pair in one frame."""
    acc: dict[tuple[int, int], int] = defaultdict(int)
    slices = (
        (lab[:, :-1], lab[:, 1:]),
        (lab[:-1, :], lab[1:, :]),
    )
    mx = int(lab.max()) + 1
    for a, b in slices:
        mask = (a > 0) & (b > 0) & (a != b)
        if not mask.any():
            continue
        ua = a[mask].astype(np.int64, copy=False)
        ub = b[mask].astype(np.int64, copy=False)
        lo = np.minimum(ua, ub)
        hi = np.maximum(ua, ub)
        packed = lo * mx + hi
        uniq, cnt = np.unique(packed, return_counts=True)
        for p, c in zip(uniq.tolist(), cnt.tolist()):
            acc[(int(p // mx), int(p % mx))] += int(c)
    return {k: float(v) for k, v in acc.items()}


def lookup_contact(
    cmap: dict[tuple[int, int], float], a: int, b: int
) -> float:
    if a > b:
        a, b = b, a
    return float(cmap.get((a, b), 0.0))


def pairs_from_triples(
    t1: tuple[int, ...], t2: tuple[int, ...]
) -> tuple[frozenset[int], frozenset[int]] | tuple[None, None]:
    s1, s2 = set(t1), set(t2)
    inter = s1 & s2
    union = s1 | s2
    if len(inter) != 2 or len(union) != 4:
        return None, None
    return frozenset(inter), frozenset(union - inter)


def vertex_bond(
    verts_3: list[dict], pair: frozenset[int], quartet: frozenset[int]
) -> tuple[float | None, dict | None, dict | None]:
    """Distance between the two 3-cell ends of a bond inside a quartet."""
    ends: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    for v in verts_3:
        ids = set(v["cell_ids"])
        if len(ids) != 3:
            continue
        if pair <= ids <= quartet:
            key = tuple(sorted(ids))
            if key in seen:
                continue
            seen.add(key)
            ends.append(v)
    if len(ends) < 2:
        return None, (ends[0] if ends else None), None
    best = None
    pair_ab = None
    for i, a in enumerate(ends):
        for b in ends[i + 1 :]:
            if set(a["cell_ids"]) | set(b["cell_ids"]) != quartet:
                continue
            d = float(np.hypot(a["y"] - b["y"], a["x"] - b["x"]))
            if best is None or d < best:
                best = d
                pair_ab = (a, b)
    if pair_ab is None:
        return None, ends[0], None
    return best, pair_ab[0], pair_ab[1]


def infer_pairs_from_verts3(
    verts_3: list[dict], quartet: frozenset[int]
) -> tuple[frozenset[int], frozenset[int]] | tuple[None, None]:
    triples = [
        v["cell_ids"]
        for v in verts_3
        if len(v["cell_ids"]) == 3 and set(v["cell_ids"]) < quartet
    ]
    for i, a in enumerate(triples):
        for b in triples[i + 1 :]:
            junc, swap = pairs_from_triples(a, b)
            if junc is not None:
                return junc, swap
    return None, None


def infer_pairs_from_contact(
    expanded: np.ndarray, quartet: frozenset[int]
) -> tuple[frozenset[int], frozenset[int]] | tuple[None, None]:
    """Pick the shortest perfect matching as the current T1 bond."""
    cells = tuple(sorted(quartet))
    matchings = [
        (frozenset((cells[0], cells[1])), frozenset((cells[2], cells[3]))),
        (frozenset((cells[0], cells[2])), frozenset((cells[1], cells[3]))),
        (frozenset((cells[0], cells[3])), frozenset((cells[1], cells[2]))),
    ]
    cmap = contact_length_map(expanded)
    best = None
    for p1, p2 in matchings:
        a, b = tuple(p1)
        c, d = tuple(p2)
        l1 = lookup_contact(cmap, a, b)
        l2 = lookup_contact(cmap, c, d)
        score = min(l1 if l1 > 0 else 1e9, l2 if l2 > 0 else 1e9)
        if best is None or score < best[0]:
            best = (score, p1 if (l1 <= l2) else p2, p2 if (l1 <= l2) else p1)
    if best is None:
        return None, None
    return best[1], best[2]


def frame_geometry(
    labels: np.ndarray, expand: int
) -> tuple[np.ndarray, list[dict], list[dict]]:
    expanded = expand_labels(labels, distance=expand) if expand > 0 else labels
    count = unique_count_3x3(expanded)
    junc = count >= 3
    four_mask = junc & (count >= 4)
    three_mask = junc & ~ndi.binary_dilation(four_mask, iterations=1)
    verts_4 = _component_records(four_mask, expanded, count)
    verts_3 = _component_records(three_mask, expanded, count)
    return expanded, verts_3, verts_4


def junctions_one_frame(
    labels: np.ndarray,
    expand: int,
    bond_cutoff: float,
) -> tuple[list[dict], list[dict], np.ndarray]:
    expanded, verts_3, verts_4 = frame_geometry(labels, expand)

    used_3: set[int] = set()
    collapsed: list[dict] = []
    for i, a in enumerate(verts_3):
        if i in used_3:
            continue
        if len(a["cell_ids"]) != 3:
            continue
        best_j = None
        best_d = bond_cutoff + 1.0
        for j, b in enumerate(verts_3):
            if j <= i or j in used_3:
                continue
            if len(b["cell_ids"]) != 3:
                continue
            union = set(a["cell_ids"]) | set(b["cell_ids"])
            if len(union) != 4:
                continue
            d = float(np.hypot(a["y"] - b["y"], a["x"] - b["x"]))
            if d <= bond_cutoff and d < best_d:
                best_d = d
                best_j = j
        if best_j is None:
            continue
        b = verts_3[best_j]
        used_3.add(i)
        used_3.add(best_j)
        union = tuple(sorted(set(a["cell_ids"]) | set(b["cell_ids"])))
        junc, swap = pairs_from_triples(a["cell_ids"], b["cell_ids"])
        collapsed.append(
            {
                "y": 0.5 * (a["y"] + b["y"]),
                "x": 0.5 * (a["x"] + b["x"]),
                "y1": a["y"],
                "x1": a["x"],
                "y2": b["y"],
                "x2": b["x"],
                "multiplicity": 4,
                "n_cells": 4,
                "cell_ids": union,
                "kind": "short_bond",
                "length_vertex_px": best_d,
                "junction_pair": junc,
                "swap_pair": swap,
            }
        )

    out: list[dict] = []
    for v in verts_4:
        n = max(v["n_cells"], v["multiplicity"])
        if n < 4:
            continue
        quartet = frozenset(v["cell_ids"]) if len(v["cell_ids"]) == 4 else None
        junc = swap = None
        length_v: float | None = 0.0
        y1 = x1 = y2 = x2 = None
        if quartet is not None:
            junc, swap = infer_pairs_from_verts3(verts_3, quartet)
            if junc is None:
                junc, swap = infer_pairs_from_contact(expanded, quartet)
            lv, e1, e2 = (
                vertex_bond(verts_3, junc, quartet)
                if junc is not None
                else (None, None, None)
            )
            if lv is not None:
                length_v = lv
            if e1 is not None and e2 is not None:
                y1, x1, y2, x2 = e1["y"], e1["x"], e2["y"], e2["x"]
        out.append(
            {
                **v,
                "multiplicity": min(n, 6),
                "kind": "pixel",
                "length_vertex_px": length_v,
                "junction_pair": junc,
                "swap_pair": swap,
                "y1": y1,
                "x1": x1,
                "y2": y2,
                "x2": x2,
            }
        )
    out.extend(collapsed)
    return out, verts_3, expanded


def overlay_rgb(
    rgb_path: Path,
    frame_indices: np.ndarray,
    events: list[dict],
    n_label_frames: int,
    out_path: Path,
) -> None:
    rgb = np.asarray(tifffile.imread(rgb_path))
    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        rgb = rgb[np.newaxis, ...]
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected T,Y,X,3 RGB; got {rgb.shape}")
    if rgb.shape[0] != n_label_frames:
        if rgb.shape[0] > int(frame_indices.max()):
            rgb = rgb[frame_indices]
        else:
            print(
                f"RGB frames {rgb.shape[0]} != labels {n_label_frames}; "
                "skipping overlay.",
                flush=True,
            )
            return

    luma = (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )
    g = np.clip(luma, 0, 255).astype(np.uint8)
    base = np.stack([g, g, g], axis=-1)

    by_t: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        by_t[int(e["t_index"])].append(e)

    h, w = base.shape[1:3]
    yellow = (255, 255, 0)
    for t, lst in by_t.items():
        for e in lst:
            color = MULT_COLORS.get(int(e["multiplicity"]), (255, 0, 0))
            y1, x1, y2, x2 = e.get("y1"), e.get("x1"), e.get("y2"), e.get("x2")
            if y1 not in (None, "") and y2 not in (None, ""):
                rr, cc = line(
                    int(round(float(y1))),
                    int(round(float(x1))),
                    int(round(float(y2))),
                    int(round(float(x2))),
                )
                ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                base[t, rr[ok], cc[ok]] = yellow
            rr, cc = disk((e["y"], e["x"]), 4, shape=(h, w))
            base[t, rr, cc] = color

    tifffile.imwrite(out_path, base)
    print(f"Wrote overlay {out_path}", flush=True)


def measure_quartet_frame(
    cmap: dict[tuple[int, int], float],
    present: set[int],
    verts_3: list[dict],
    quartet: frozenset[int],
    junc: frozenset[int] | None,
    swap: frozenset[int] | None,
    bond_cutoff: float,
) -> dict:
    all_here = quartet <= present
    row = {
        "cells_present": all_here,
        "length_vertex_px": "",
        "length_contact_px": "",
        "length_swap_contact_px": "",
        "y1": "",
        "x1": "",
        "y2": "",
        "x2": "",
        "is_4way": False,
    }
    if not all_here or junc is None:
        return row
    a, b = tuple(junc)
    row["length_contact_px"] = round(lookup_contact(cmap, a, b), 2)
    if swap is not None:
        c, d = tuple(swap)
        row["length_swap_contact_px"] = round(lookup_contact(cmap, c, d), 2)
    lv, e1, e2 = vertex_bond(verts_3, junc, quartet)
    if lv is not None:
        row["length_vertex_px"] = round(lv, 2)
        row["is_4way"] = lv <= bond_cutoff
    if e1 is not None and e2 is not None:
        row["y1"] = round(e1["y"], 2)
        row["x1"] = round(e1["x"], 2)
        row["y2"] = round(e2["y"], 2)
        row["x2"] = round(e2["x"], 2)
    return row


def write_length_plot(by_frame: list[dict], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    frames = [r["frame"] for r in by_frame]
    mean_v = [r["mean_length_vertex_px"] for r in by_frame]
    mean_c = [r["mean_length_contact_px"] for r in by_frame]
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=120)
    ax.plot(frames, mean_v, color="crimson", label="mean vertex length (4-way dets)")
    ax.plot(frames, mean_c, color="goldenrod", label="mean contact length (4-way dets)")
    ax.set_xlabel("frame")
    ax.set_ylabel("length (px)")
    ax.set_title("4-way junction length vs time")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TissueMiner-style N-way vertex overlay on tracked labels."
    )
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--rgb", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--expand",
        type=int,
        default=3,
        help="Grow labels by N px before junctions (Cellpose gaps; default 3).",
    )
    p.add_argument(
        "--bond-cutoff",
        type=float,
        default=3.0,
        help="Max distance between two 3-cell vertices to count as 4-way.",
    )
    return p.parse_args()


def _pair_str(pair: frozenset[int] | None) -> str:
    return fmt_ids(pair) if pair else ""


def _event_row(v: dict, frame: int, t: int) -> dict:
    lv = v.get("length_vertex_px")
    junc = v.get("junction_pair")
    swap = v.get("swap_pair")
    return {
        "frame": frame,
        "t_index": t,
        "y": round(float(v["y"]), 2),
        "x": round(float(v["x"]), 2),
        "y1": "" if v.get("y1") is None else round(float(v["y1"]), 2),
        "x1": "" if v.get("x1") is None else round(float(v["x1"]), 2),
        "y2": "" if v.get("y2") is None else round(float(v["y2"]), 2),
        "x2": "" if v.get("x2") is None else round(float(v["x2"]), 2),
        "multiplicity": int(v["multiplicity"]),
        "n_cells": int(v["n_cells"]),
        "kind": v["kind"],
        "cell_ids": fmt_ids(v["cell_ids"]),
        "junction_pair": _pair_str(junc),
        "swap_pair": _pair_str(swap),
        "length_vertex_px": "" if lv is None else round(float(lv), 2),
    }


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = input_stem(args.labels)

    print(f"Loading {args.labels}", flush=True)
    stack, frame_indices = drop_blank(load_labels(args.labels.expanduser()))
    print(f"labels shape={stack.shape}", flush=True)

    events: list[dict] = []
    per_frame: list[dict] = []
    geoms: list[tuple[list[dict], np.ndarray]] = []
    n = stack.shape[0]
    for t in range(n):
        verts, verts_3, expanded = junctions_one_frame(
            stack[t], args.expand, args.bond_cutoff
        )
        geoms.append((verts_3, expanded))
        counts = defaultdict(int)
        lengths_v: list[float] = []
        for v in verts:
            counts[int(v["multiplicity"])] += 1
            row = _event_row(v, int(frame_indices[t]), t)
            events.append(row)
            if row["length_vertex_px"] != "":
                lengths_v.append(float(row["length_vertex_px"]))
        per_frame.append(
            {
                "frame": int(frame_indices[t]),
                "n_4way": int(counts[4]),
                "n_5way": int(counts[5]),
                "n_6way": int(counts[6]),
                "n_total": len(verts),
                "mean_length_vertex_px": (
                    float(np.mean(lengths_v)) if lengths_v else 0.0
                ),
                "mean_length_contact_px": 0.0,
            }
        )
        if t == 0 or t == n - 1 or (t + 1) % 20 == 0:
            print(
                f"  frame {t}/{n - 1}: 4-way={counts[4]} "
                f"5-way={counts[5]} 6-way={counts[6]} "
                f"mean_len={per_frame[-1]['mean_length_vertex_px']:.2f}px",
                flush=True,
            )

    # Freeze T1 pair identity per 4-cell quartet from the first good detection.
    quartet_pairs: dict[frozenset[int], tuple[frozenset[int], frozenset[int]]] = {}
    for e in events:
        if e["n_cells"] != 4 or not e["junction_pair"]:
            continue
        q = frozenset(int(x) for x in e["cell_ids"].split(";"))
        if q in quartet_pairs:
            continue
        junc = frozenset(int(x) for x in e["junction_pair"].split(";"))
        swap = (
            frozenset(int(x) for x in e["swap_pair"].split(";"))
            if e["swap_pair"]
            else q - junc
        )
        quartet_pairs[q] = (junc, swap)

    print(
        f"Tracking junction length for {len(quartet_pairs)} 4-cell quartets...",
        flush=True,
    )
    track_rows: list[dict] = []
    for t, (verts_3, expanded) in enumerate(geoms):
        cmap = contact_length_map(expanded)
        present = {int(v) for v in np.unique(stack[t]) if v != 0}
        for q, (junc, swap) in quartet_pairs.items():
            meas = measure_quartet_frame(
                cmap,
                present,
                verts_3,
                q,
                junc,
                swap,
                args.bond_cutoff,
            )
            track_rows.append(
                {
                    "frame": int(frame_indices[t]),
                    "quartet_id": fmt_ids(q),
                    "junction_pair": fmt_ids(junc),
                    "swap_pair": fmt_ids(swap) if swap else "",
                    **meas,
                }
            )

    for rec in per_frame:
        four_vals = [
            float(r["length_contact_px"])
            for r in track_rows
            if r["frame"] == rec["frame"]
            and r["is_4way"]
            and r["length_contact_px"] != ""
        ]
        rec["mean_length_contact_px"] = (
            float(np.mean(four_vals)) if four_vals else 0.0
        )
        rec["n_quartets_present"] = sum(
            1
            for r in track_rows
            if r["frame"] == rec["frame"] and r["cells_present"]
        )

    csv_path = tagged_path(out, stem, "fourway_vertices.csv")
    ev_fields = [
        "frame",
        "y",
        "x",
        "y1",
        "x1",
        "y2",
        "x2",
        "multiplicity",
        "n_cells",
        "kind",
        "cell_ids",
        "junction_pair",
        "swap_pair",
        "length_vertex_px",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ev_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)

    sum_path = tagged_path(out, stem, "fourway_by_frame.csv")
    with sum_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_frame[0].keys()))
        writer.writeheader()
        writer.writerows(per_frame)

    track_path = tagged_path(out, stem, "junction_lengths.csv")
    tr_fields = [
        "frame",
        "quartet_id",
        "junction_pair",
        "swap_pair",
        "length_vertex_px",
        "length_contact_px",
        "length_swap_contact_px",
        "is_4way",
        "cells_present",
        "y1",
        "x1",
        "y2",
        "x2",
    ]
    with track_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tr_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(track_rows)

    n4 = sum(r["n_4way"] for r in per_frame)
    n5 = sum(r["n_5way"] for r in per_frame)
    n6 = sum(r["n_6way"] for r in per_frame)
    det_len = [float(e["length_vertex_px"]) for e in events if e["length_vertex_px"] != ""]
    qc = {
        "n_frames": n,
        "n_4way": n4,
        "n_5way": n5,
        "n_6way": n6,
        "n_quartets": len(quartet_pairs),
        "mean_4way_per_frame": float(n4 / n) if n else 0.0,
        "mean_length_vertex_px": float(np.mean(det_len)) if det_len else 0.0,
        "expand": args.expand,
        "bond_cutoff": args.bond_cutoff,
        "source": "vendor/tissue_miner/topology/FourWayVertices.R",
        "labels": str(args.labels.resolve()),
    }
    (tagged_path(out, stem, "qc.json")).write_text(json.dumps(qc, indent=2) + "\n")
    print(
        f"4-way={n4} 5-way={n5} 6-way={n6} quartets={qc['n_quartets']} "
        f"mean_len={qc['mean_length_vertex_px']:.2f}px",
        flush=True,
    )
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {sum_path}", flush=True)
    print(f"Wrote {track_path}", flush=True)
    write_length_plot(per_frame, tagged_path(out, stem, "junction_length_vs_frame.png"))

    if args.rgb is not None and args.rgb.expanduser().is_file():
        overlay_rgb(
            args.rgb.expanduser(),
            frame_indices,
            events,
            stack.shape[0],
            tagged_path(out, stem, "fourway_overlay.tif"),
        )


if __name__ == "__main__":
    main()
