#!/usr/bin/env python3
"""Detect T1 transitions from a tracked label stack via vertex order changes.

Vertices are detected with TA's own detector (epyseg.ta.tracking
rapid_detection_of_vertices), which expects filled cells separated by a
boundary color (background = 0).

Accepts either:
  - TA export: tracked_cells_ids.tif (+ tracked_cells_rgb.tif)
  - cellpose_track export: tracked_labels.tif (consistent cross-frame IDs;
    RGB is synthesised if tracked_cells_rgb.tif / tracked_rgb.tif is absent)

A T1 is a junction that goes 3-cell -> 4-cell -> 3-cell while the shared bond
swaps partners. The contacting pair must be the same complementary diagonal on
all --search frames before the 4-way (AB) and all --search frames after (CD),
so aborted 4-ways that return to the original neighbours are rejected. As in
TA (`bond_cut_off`), a bond shorter than --bond-cutoff between two 3-cell
vertices counts as a 4-cell junction, since a collapsing bond rarely reduces
to a single 4-way pixel. Writes:
  - t1_highlights.tif  — grayscale track RGB; yellow = the 3-cell junctions
    before/after the swap, red = the 4-cell vertex itself
  - t1_events.csv      — one row per detected event (for QC)

Examples:
  conda activate TA
  python scripts/t1/run_t1_visualize.py \\
    --ids output/archive/ta_track/minimal_naima_cut_epyseg/tracked_cells_ids.tif \\
    --out output/t1/Cri_WT_GAP43_GFP_38sec_Naima_cut_t1/ta_highlights.tif

  python scripts/t1/run_t1_visualize.py \\
    --ta-dir output/archive/ta_track/minimal_naima_cut_epyseg

  python scripts/t1/run_t1_visualize.py \\
    --ids output/cellpose_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_cp_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_tracked_labels.tif
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import tifffile

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from napari_tissuekit.core.output_names import input_stem, tagged, tagged_path  # noqa: E402

CellSet = frozenset  # frozenset of track IDs around a vertex

YELLOW = (255, 255, 0)  # 3-cell junctions before/after the swap
RED = (255, 0, 0)  # 4-cell vertex


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Detect T1 transitions as 3-cell -> 4-cell -> 3-cell vertex "
            "changes in TA tracked_cells_ids.tif and overlay them on a "
            "grayscale copy of tracked_cells_rgb.tif."
        )
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--ids",
        type=Path,
        help=(
            "Path to a T,Y,X integer track-ID stack "
            "(tracked_cells_ids.tif or tracked_labels.tif)."
        ),
    )
    src.add_argument(
        "--ta-dir",
        type=Path,
        help=(
            "Run directory containing tracked_labels.tif or "
            "tracked_cells_ids.tif."
        ),
    )
    p.add_argument(
        "--rgb",
        type=Path,
        default=None,
        help=(
            "Base track RGB TIFF (default: sibling tracked_cells_rgb.tif / "
            "tracked_rgb.tif; synthesised from IDs if neither exists)."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output RGB TIFF (default: <ta-dir or ids parent>/t1_highlights.tif)."
        ),
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output events CSV (default: next to --out as t1_events.csv).",
    )
    p.add_argument(
        "--pre",
        type=int,
        default=3,
        help="Frames before the 4-cell vertex to highlight (default: 3).",
    )
    p.add_argument(
        "--post",
        type=int,
        default=3,
        help="Frames after the 4-cell vertex to highlight (default: 3).",
    )
    p.add_argument(
        "--search",
        type=int,
        default=3,
        help=(
            "Required consecutive frames before and after a 4-way that must "
            "share the same contacting pair (AB then CD; default: 3)."
        ),
    )
    p.add_argument(
        "--frames",
        type=str,
        default=None,
        help=(
            "Process a frame range only, as START:END (end exclusive), e.g. "
            "0:20. Useful for a quick preview."
        ),
    )
    p.add_argument(
        "--bond-cutoff",
        type=float,
        default=3.0,
        help=(
            "Max pixel distance between two 3-cell vertices for the bond "
            "between them to count as a 4-cell junction (default: 3)."
        ),
    )
    return p.parse_args()


def _find_ids_in_dir(ta_dir: Path) -> Path:
    """Prefer cellpose_track's consistent IDs over TA's per-frame remapped ones."""
    for name in ("tracked_labels.tif", "tracked_cells_ids.tif"):
        candidate = ta_dir / name
        if candidate.is_file():
            return candidate
    matches = sorted(ta_dir.glob("*_tracked_labels.tif"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple *_tracked_labels.tif in {ta_dir}: {matches}"
        )
    raise FileNotFoundError(
        f"No tracked_labels.tif / *_tracked_labels.tif or "
        f"tracked_cells_ids.tif in {ta_dir}"
    )


def _find_rgb_in_dir(ta_dir: Path) -> Path | None:
    for name in ("tracked_cells_rgb.tif", "tracked_rgb.tif"):
        candidate = ta_dir / name
        if candidate.is_file():
            return candidate
    matches = sorted(ta_dir.glob("*_tracked_rgb.tif"))
    if len(matches) == 1:
        return matches[0]
    return None


def synthesize_rgb_from_ids(stack: np.ndarray, seed: int = 1) -> np.ndarray:
    """Stable random RGB per track ID (for viewing when no RGB stack exists)."""
    max_id = int(stack.max())
    rng = np.random.default_rng(seed)
    lut = rng.integers(40, 256, size=(max_id + 1, 3), dtype=np.uint8)
    lut[0] = 0
    return lut[stack]


def resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path | None, Path, Path]:
    if args.ta_dir is not None:
        ta_dir = args.ta_dir.expanduser().resolve()
        ids_path = _find_ids_in_dir(ta_dir)
    else:
        ids_path = args.ids.expanduser().resolve()
        ta_dir = ids_path.parent

    if not ids_path.is_file():
        raise FileNotFoundError(f"Track ID stack not found: {ids_path}")

    if args.rgb is not None:
        rgb_path: Path | None = args.rgb.expanduser().resolve()
        if not rgb_path.is_file():
            raise FileNotFoundError(f"RGB stack not found: {rgb_path}")
    else:
        rgb_path = _find_rgb_in_dir(ta_dir)

    stem = input_stem(ids_path)
    out_path = (
        args.out.expanduser().resolve()
        if args.out is not None
        else tagged_path(ta_dir, stem, "t1_highlights.tif")
    )
    csv_path = (
        args.csv.expanduser().resolve()
        if args.csv is not None
        else out_path.with_name(tagged(stem, "t1_events.csv"))
    )
    return ids_path, rgb_path, out_path, csv_path


def load_ids(path: Path) -> np.ndarray:
    stack = np.asarray(tifffile.imread(path))
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    if stack.ndim != 3:
        raise ValueError(f"Expected T,Y,X id stack; got shape {stack.shape}")
    return stack.astype(np.uint32, copy=False)


def load_rgb(path: Path, *, expected_txy: tuple[int, int, int]) -> np.ndarray:
    """Load track RGB as (T,Y,X,3) uint8 matching the ID stack shape."""
    rgb = np.asarray(tifffile.imread(path))
    t, h, w = expected_txy
    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        rgb = rgb[np.newaxis, ...]
    if rgb.ndim == 4 and rgb.shape[1] == 3 and rgb.shape[-1] != 3:
        # (T,3,Y,X) -> (T,Y,X,3)
        rgb = np.moveaxis(rgb, 1, -1)
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected T,Y,X,3 RGB stack; got shape {rgb.shape}")
    if rgb.shape[:3] != (t, h, w):
        raise ValueError(
            f"RGB shape {rgb.shape[:3]} does not match IDs shape {(t, h, w)}"
        )
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def frame_vertices(labels: np.ndarray) -> dict[CellSet, list[tuple[int, int]]]:
    """Vertices of one frame, keyed by the set of cells around them.

    Only 3-cell and 4-cell vertices are kept; the value is the list of vertex
    pixels belonging to that cell set.
    """
    from epyseg.ta.tracking.rapid_detection_of_vertices import (
        associate_vertices_to_cells,
        detect_vertices_and_bonds,
    )

    verts = detect_vertices_and_bonds(labels, boundary_color=0)
    vx_cells = associate_vertices_to_cells(
        labels,
        verts,
        forbidden_colors=(0,),
        output_vertices_and_associated_cells=True,
    )

    by_cells: dict[CellSet, list[tuple[int, int]]] = defaultdict(list)
    for (y, x), ids in vx_cells.items():
        if len(ids) in (3, 4):
            by_cells[frozenset(int(i) for i in ids)].append((int(y), int(x)))
    return dict(by_cells)


def collect_vertices(stack: np.ndarray) -> list[dict[CellSet, list[tuple[int, int]]]]:
    t_count = stack.shape[0]
    print(f"Detecting vertices (TA) for {t_count} frames...", flush=True)
    per_frame: list[dict[CellSet, list[tuple[int, int]]]] = []
    for t in range(t_count):
        fv = frame_vertices(stack[t])
        per_frame.append(fv)
        if (t + 1) % 5 == 0 or t == t_count - 1:
            n3 = sum(1 for cells in fv if len(cells) == 3)
            n4 = sum(1 for cells in fv if len(cells) == 4)
            print(f"  frame {t + 1}/{t_count} v3={n3} v4={n4}", flush=True)
    return per_frame


def vertex_pairs(fv: dict[CellSet, list[tuple[int, int]]]) -> set[CellSet]:
    """Cell pairs that share a vertex, i.e. the neighbour graph of the frame."""
    pairs: set[CellSet] = set()
    for cells in fv:
        for a, b in combinations(sorted(cells), 2):
            pairs.add(frozenset((a, b)))
    return pairs


def _min_distance(p1: list[tuple[int, int]], p2: list[tuple[int, int]]) -> float:
    a = np.asarray(p1, dtype=np.float32)
    b = np.asarray(p2, dtype=np.float32)
    d = np.hypot(
        a[:, 0][:, None] - b[None, :, 0], a[:, 1][:, None] - b[None, :, 1]
    )
    return float(d.min())


def four_cell_junctions(
    fv: dict[CellSet, list[tuple[int, int]]], cutoff: float
) -> dict[CellSet, list[tuple[int, int]]]:
    """Four-cell junctions of one frame, keyed by the four cells around them.

    A genuine 4-cell vertex pixel counts, and so does a bond shorter than
    `cutoff` between two 3-cell vertices -- TA applies the same convention via
    its `bond_cut_off`, since a collapsing bond rarely reduces to one pixel.
    """
    ends: dict[CellSet, list[tuple[CellSet, list[tuple[int, int]]]]] = defaultdict(list)
    for cells, pts in fv.items():
        if len(cells) != 3:
            continue
        for a, b in combinations(sorted(cells), 2):
            ends[frozenset((a, b))].append((cells, pts))

    out: dict[CellSet, list[tuple[int, int]]] = defaultdict(list)
    for lst in ends.values():
        if len(lst) != 2:
            continue
        (cells1, pts1), (cells2, pts2) = lst
        quartet = cells1 | cells2
        if len(quartet) != 4:
            continue
        if _min_distance(pts1, pts2) > cutoff:
            continue
        out[quartet].extend(pts1 + pts2)

    for cells, pts in fv.items():
        if len(cells) == 4:
            out[cells].extend(pts)

    return dict(out)


def pair_split(quartet: CellSet, pairs: set[CellSet]) -> CellSet | None:
    """The one pair of `quartet` that is a neighbour pair while its complement is not."""
    cells = sorted(quartet)
    found: list[CellSet] = []
    for a, b in combinations(cells, 2):
        pair = frozenset((a, b))
        other = quartet - pair
        if pair in pairs and other not in pairs:
            found.append(pair)
    if len(found) == 1:
        return found[0]
    return None


def consecutive_runs(frames: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = prev = frames[0]
    for t in frames[1:]:
        if t == prev + 1:
            prev = t
            continue
        runs.append((start, prev))
        start = prev = t
    runs.append((start, prev))
    return runs


def stable_pair(
    per_frame_pairs: list[set[CellSet]],
    quartet: CellSet,
    start: int,
    end: int,
) -> CellSet | None:
    """Unique contacting pair identical on every frame in [start, end] inclusive.

    Returns None if any frame is out of range, has no unique pair_split, or
    disagrees with the others.
    """
    t_count = len(per_frame_pairs)
    if start > end or start < 0 or end >= t_count:
        return None
    found: CellSet | None = None
    for t in range(start, end + 1):
        pair = pair_split(quartet, per_frame_pairs[t])
        if pair is None:
            return None
        if found is None:
            found = pair
        elif pair != found:
            return None
    return found


def is_complementary(
    quartet: CellSet, before: CellSet, after: CellSet
) -> bool:
    """True T1 diagonals: after is exactly quartet - before (no shared cell)."""
    if before & after:
        return False
    return before | after == quartet and len(before) == 2 and len(after) == 2


def detect_t1_events(
    per_frame: list[dict[CellSet, list[tuple[int, int]]]],
    per_frame_pairs: list[set[CellSet]],
    per_frame_four: list[dict[CellSet, list[tuple[int, int]]]],
    *,
    search: int,
) -> list[dict]:
    """T1 = 4-cell junction with stable complementary AB -> CD across ±search."""
    quartet_frames: dict[CellSet, list[int]] = defaultdict(list)
    for t, fj in enumerate(per_frame_four):
        for quartet in fj:
            quartet_frames[quartet].append(t)

    print(f"Distinct 4-cell junctions seen: {len(quartet_frames)}", flush=True)

    events: list[dict] = []
    for quartet, frames in quartet_frames.items():
        for t_start, t_end in consecutive_runs(sorted(frames)):
            pair_before = stable_pair(
                per_frame_pairs, quartet, t_start - search, t_start - 1
            )
            pair_after = stable_pair(
                per_frame_pairs, quartet, t_end + 1, t_end + search
            )
            if pair_before is None or pair_after is None:
                continue
            if not is_complementary(quartet, pair_before, pair_after):
                continue

            pts = per_frame_four[t_start].get(quartet, [])
            ys = [p[0] for p in pts]
            xs = [p[1] for p in pts]
            cells = sorted(quartet)
            events.append(
                {
                    "quartet": quartet,
                    "t_start": t_start,
                    "t_end": t_end,
                    "A": cells[0],
                    "B": cells[1],
                    "C": cells[2],
                    "D": cells[3],
                    "pair_before": "-".join(str(c) for c in sorted(pair_before)),
                    "pair_after": "-".join(str(c) for c in sorted(pair_after)),
                    "y": int(round(float(np.mean(ys)))) if ys else -1,
                    "x": int(round(float(np.mean(xs)))) if xs else -1,
                }
            )

    events.sort(key=lambda e: (e["t_start"], e["A"], e["B"]))
    return events


def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luma, replicated to 3 channels so overlays stay RGB."""
    luma = (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )
    g = np.clip(luma, 0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def draw_markers(
    dst: np.ndarray,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    radius: int = 2,
) -> None:
    if not points:
        return
    h, w = dst.shape[:2]
    col = np.array(color, dtype=np.uint8)
    for y, x in points:
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        dst[y0:y1, x0:x1] = col


def render_highlights(
    per_frame: list[dict[CellSet, list[tuple[int, int]]]],
    per_frame_four: list[dict[CellSet, list[tuple[int, int]]]],
    events: list[dict],
    rgb_base: np.ndarray,
    *,
    pre: int,
    post: int,
) -> np.ndarray:
    print("Converting base RGB to grayscale...", flush=True)
    out = rgb_to_grayscale(rgb_base)

    if not events:
        print("No T1 events to highlight; writing grayscale copy.", flush=True)
        return out

    t_count = len(per_frame)
    yellow_pts: dict[int, list[tuple[int, int]]] = defaultdict(list)
    red_pts: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for ev in events:
        quartet = ev["quartet"]
        t_start, t_end = ev["t_start"], ev["t_end"]

        for t in range(t_start, t_end + 1):
            red_pts[t].extend(per_frame_four[t].get(quartet, []))

        window = list(range(max(0, t_start - pre), t_start))
        window += list(range(t_end + 1, min(t_count, t_end + post + 1)))
        for t in window:
            for cells, pts in per_frame[t].items():
                if len(cells) == 3 and cells < quartet:
                    yellow_pts[t].extend(pts)

    print(f"Overlaying {len(events)} T1 events...", flush=True)
    for t in range(t_count):
        draw_markers(out[t], yellow_pts.get(t, []), YELLOW, radius=2)
        draw_markers(out[t], red_pts.get(t, []), RED, radius=2)
        if (t + 1) % 20 == 0 or t == t_count - 1:
            print(f"  render {t + 1}/{t_count}", flush=True)

    return out


def write_csv(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "t_start",
        "t_end",
        "A",
        "B",
        "C",
        "D",
        "pair_before",
        "pair_after",
        "y",
        "x",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for ev in events:
            w.writerow(ev)


def main() -> None:
    args = parse_args()
    ids_path, rgb_path, out_path, csv_path = resolve_paths(args)

    print(f"Loading {ids_path}", flush=True)
    stack = load_ids(ids_path)
    print(
        f"ids shape={stack.shape} dtype={stack.dtype} max_id={int(stack.max())}",
        flush=True,
    )

    if rgb_path is not None:
        print(f"Loading base RGB {rgb_path}", flush=True)
        rgb_base = load_rgb(rgb_path, expected_txy=tuple(stack.shape))
    else:
        print(
            "No RGB stack found; synthesising colours from track IDs...",
            flush=True,
        )
        rgb_base = synthesize_rgb_from_ids(stack)
    print(f"rgb shape={rgb_base.shape}", flush=True)

    if args.frames:
        start_s, _, end_s = args.frames.partition(":")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else stack.shape[0]
        stack = stack[start:end]
        rgb_base = rgb_base[start:end]
        print(f"Restricted to frames {start}:{end} -> {stack.shape[0]}", flush=True)

    per_frame = collect_vertices(stack)
    per_frame_pairs = [vertex_pairs(fv) for fv in per_frame]
    per_frame_four = [
        four_cell_junctions(fv, args.bond_cutoff) for fv in per_frame
    ]

    events = detect_t1_events(
        per_frame, per_frame_pairs, per_frame_four, search=args.search
    )
    print(f"Detected T1 events: {len(events)}", flush=True)

    write_csv(csv_path, events)
    print(f"Wrote {csv_path}", flush=True)

    rgb = render_highlights(
        per_frame,
        per_frame_four,
        events,
        rgb_base,
        pre=args.pre,
        post=args.post,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out_path, rgb)
    print(f"Wrote {out_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
