#!/usr/bin/env python3
"""T1 events from TissueMiner four-ways + contact swap.

A 4-way is the intermediate. A completed T1 is:

  before: A touches B, C does not touch D
  during: the four cells meet as a TM 4-way / short bond
  after:  A does not touch B, C touches D

Only complementary diagonals count (A-B -> C-D). Side flips (A-B -> A-C)
are rejected. The contact swap must also match the 4-way junction/swap pair.

Detection writes a baseline of every completed swap (one frame of AB before
and CD after is enough), plus neighbour-swap T1s that need no 4-way vertex
(same graph rule as CellFlow intercalation), plus aborted 4-ways (original
neighbours return). Each row is scored with persist_pre / persist_post and
n_swaps so the napari tab and picker can filter All / Persist 3/5/7 /
Flicker / Aborted without re-detecting. Quartets with n_swaps>=2 are merged
into one event from the first 4-way to the last (including aborted returns).

--length-window (default 0) still drops completed events that lack measurable
vertex lengths. --search is kept for compatibility but no longer drops rows.

Embryo-margin cells (touch background; optional plus-one ring) are excluded.
Overlay colors by type: completed_4way (orange/cyan), flicker (yellow-orange/lime),
aborted (magenta/pink), no_4way (amber/teal), other_matching (red).

Example:
  conda activate tissue_miner
  python scripts/t1/run_tm_t1.py \\
    --labels output/cellpose_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_cp_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_tracked_labels.tif \\
    --rgb output/cellpose_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_cp_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_tracked_rgb.tif \\
    --frames 0:40 \\
    --output-dir output/t1/Cri_WT_GAP43_GFP_38sec_Naima_cut_t1/detect

Step folders under each movie ({stem}_t1/):
  detect/       — this script (events, lengths, overlays)
  picker_ready/ — Mac/PC picker bundle (written automatically)
  picked/       — headless/GUI select export
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import sys
from collections import defaultdict
from collections.abc import Callable
from itertools import combinations
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.draw import disk
from skimage.segmentation import expand_labels


from ..celltrack.cells import build_adjacency  # noqa: E402
from ..output_names import input_stem, tagged_path  # noqa: E402
from ..t1.calibrated_outputs import write_um_files  # noqa: E402
from ..movie_calibration import calibration_for, save_tiff  # noqa: E402
from ..t1.t1_picker_ready import write_picker_ready  # noqa: E402
from ..t1.run_tm_fourway_vertices import (  # noqa: E402
    contact_length_map,
    fmt_ids,
    junctions_one_frame,
    load_labels,
    lookup_contact,
    vertex_bond,
)

BREAKING = (255, 140, 0)  # orange — shrinking AB (completed_4way)
BUILDING = (0, 220, 255)  # cyan — growing CD (completed_4way)
FOURWAY = (0, 180, 220)  # cyan vertex (completed_4way; was red)

# Type-colored junctions / vertices for overlay + fourways.
TYPE_COLORS: dict[str, dict[str, tuple[int, int, int]]] = {
    "completed_4way": {
        "vertex": (0, 180, 220),
        "breaking": (255, 140, 0),
        "building": (0, 220, 255),
    },
    "flicker": {
        "vertex": (255, 220, 0),
        "breaking": (255, 180, 0),
        "building": (140, 255, 0),
    },
    "aborted": {
        "vertex": (220, 0, 220),
        "breaking": (220, 0, 220),
        "building": (255, 120, 200),
    },
    "no_4way": {
        "vertex": (255, 160, 0),
        "breaking": (255, 160, 0),
        "building": (0, 180, 160),
    },
    "other_matching": {
        "vertex": (220, 40, 40),
        "breaking": (255, 80, 60),
        "building": (160, 0, 40),
    },
}


def event_type(
    kind: str,
    has_fourway: int | bool,
    n_swaps: int,
    other_matching: int | bool = 0,
) -> str:
    """Classify a T1 row (other_matching beats flicker; no_4way if never 4-way)."""
    if int(other_matching):
        return "other_matching"
    if int(n_swaps) >= 2:
        return "flicker"
    if kind == "aborted":
        return "aborted"
    if int(has_fourway) == 0:
        return "no_4way"
    return "completed_4way"


def type_colors(
    kind: str,
    has_fourway: int | bool,
    n_swaps: int,
    other_matching: int | bool = 0,
) -> dict[str, tuple[int, int, int]]:
    return TYPE_COLORS[
        event_type(kind, has_fourway, n_swaps, other_matching)
    ]


# Export column order for junction_lengths.csv (matches t1 picker select-all).
TYPE_SORT_ORDER = {
    "completed_4way": 0,
    "flicker": 1,
    "aborted": 2,
    "no_4way": 3,
    "other_matching": 4,
}


def _pair_dash(ids) -> str:
    return "-".join(str(i) for i in sorted(ids))


def _de_num(v) -> str:
    """Format number for German Excel (comma decimal). Flags pass through."""
    if v is None or v == "":
        return ""
    if isinstance(v, str) and not v[:1].isdigit() and v[:1] != "-":
        return v
    if isinstance(v, float):
        s = f"{v:.2f}".rstrip("0").rstrip(".")
        return s.replace(".", ",")
    if isinstance(v, int):
        return str(v)
    return str(v).replace(".", ",")


def write_lengths_dt_first(
    path: Path,
    ordered: list[tuple[int, str]],
    value_map: dict[tuple[int, int], object],
    dts: list[int],
) -> None:
    """Wide Excel CSV: event_id + event_type headers, then dt as first column.

    Layout (matches picker selected_traces_length style):
      sep=;
      event_id;<id>;<id>;...
      event_type;<type>;<type>;...
      <dt>;<len>;<len>;...
    """
    ids = [eid for eid, _ in ordered]
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("sep=;\n")
        handle.write(";".join(["event_id", *[str(eid) for eid in ids]]) + "\n")
        handle.write(
            ";".join(["event_type", *[etype for _, etype in ordered]]) + "\n"
        )
        for dt in dts:
            cells = [str(dt)]
            for eid in ids:
                cells.append(_de_num(value_map.get((eid, dt), "")))
            handle.write(";".join(cells) + "\n")


def pair_sign(
    cmap: dict[tuple[int, int], float],
    ab: frozenset[int],
    cd: frozenset[int],
) -> tuple[frozenset[int] | None, int]:
    """Active T1 pair: +1 AB, -1 CD; ignore side contacts (Preview AB/CD only)."""
    a, b = tuple(ab)
    c, d = tuple(cd)
    ab_on = lookup_contact(cmap, a, b) > 0
    cd_on = lookup_contact(cmap, c, d) > 0
    if ab_on and not cd_on:
        return ab, 1
    if cd_on and not ab_on:
        return cd, -1
    return None, 0


FLAG_FOURWAY = "4way"
FLAG_NV = "nv"
FLAG_NO_ABCD = "no AB/CD contact"


def _pair_touches(adj: dict[int, set[int]], pair: frozenset[int]) -> bool:
    a, b = tuple(pair)
    return b in adj.get(a, set())


def active_abcd_pair(
    adj: dict[int, set[int]],
    ab: frozenset[int],
    cd: frozenset[int],
) -> tuple[frozenset[int] | None, int]:
    """Same as pair_sign but using expanded adjacency instead of a contact map."""
    ab_on = _pair_touches(adj, ab)
    cd_on = _pair_touches(adj, cd)
    if ab_on and not cd_on:
        return ab, 1
    if cd_on and not ab_on:
        return cd, -1
    return None, 0


def length_blank_flag(
    adj: dict[int, set[int]],
    ab: frozenset[int],
    cd: frozenset[int],
) -> str:
    """Why a frame has no AB/CD length: 4-way vs anything else."""
    if _pair_touches(adj, ab) and _pair_touches(adj, cd):
        return FLAG_FOURWAY
    return FLAG_NO_ABCD


def _apply_sign(value: object, sign: int) -> object:
    if value == "" or value is None:
        return ""
    if isinstance(value, str):
        return value
    if sign == 0:
        return ""
    return round(float(value) * sign, 2)


def write_junction_lengths_wide(
    path: Path,
    ordered: list[tuple[int, str, object, object]],
    vertex_pos: dict[tuple[int, int], object],
    vertex_signed: dict[tuple[int, int], object],
    contact_pos: dict[tuple[int, int], object],
    contact_signed: dict[tuple[int, int], object],
    dts: list[int],
) -> None:
    """Write unsigned vertex lengths plus a signed sibling (AB +, CD −)."""
    headers = [(eid, etype) for eid, etype, _, _ in ordered]
    write_lengths_dt_first(path, headers, vertex_pos, dts)
    signed_path = path.with_name(
        path.name.replace("junction_lengths.csv", "junction_lengths_signed.csv")
    )
    if signed_path == path:
        signed_path = path.with_name(f"{path.stem}_signed{path.suffix}")
    write_lengths_dt_first(signed_path, headers, vertex_signed, dts)
    print(f"Wrote {signed_path}", flush=True)
    contact_path = path.with_name(
        path.name.replace("junction_lengths.csv", "junction_contact_lengths.csv")
    )
    if contact_path == path:
        contact_path = path.with_name(f"{path.stem}_contact{path.suffix}")
    write_lengths_dt_first(contact_path, headers, contact_pos, dts)
    print(f"Wrote {contact_path}", flush=True)
    contact_signed_path = path.with_name(
        path.name.replace(
            "junction_lengths.csv", "junction_contact_lengths_signed.csv"
        )
    )
    if contact_signed_path == path:
        contact_signed_path = path.with_name(f"{path.stem}_contact_signed{path.suffix}")
    write_lengths_dt_first(contact_signed_path, headers, contact_signed, dts)
    print(f"Wrote {contact_signed_path}", flush=True)


def parse_frames(spec: str | None, n: int) -> tuple[int, int]:
    if not spec:
        return 0, n
    start_s, _, end_s = spec.partition(":")
    start = int(start_s) if start_s else 0
    end = int(end_s) if end_s else n
    start = max(0, start)
    end = min(n, end)
    if end - start < 3:
        raise ValueError(f"Need >=3 frames; got {start}:{end}")
    return start, end


def drop_blank(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = [t for t in range(stack.shape[0]) if int(stack[t].max()) > 0]
    if len(keep) < 3:
        raise ValueError(f"Need >=3 non-blank frames; kept {len(keep)}")
    idx = np.asarray(keep, dtype=np.int32)
    dropped = [t for t in range(stack.shape[0]) if t not in set(keep)]
    if dropped:
        print(f"Dropping blank frames {dropped}", flush=True)
    return stack[idx], idx


def tissue_border_cells(labels: np.ndarray) -> set[int]:
    """Cells that 4-connect to background on the unexpanded label image."""
    bg = labels == 0
    dil = ndi.binary_dilation(bg, iterations=1)
    ids = {int(v) for v in np.unique(labels[dil]) if v != 0}
    return ids


def border_plus_rings(labels: np.ndarray, adj: dict[int, set[int]], rings: int) -> set[int]:
    border = tissue_border_cells(labels)
    if rings <= 1:
        return border
    extra: set[int] = set(border)
    frontier = set(border)
    for _ in range(rings - 1):
        nxt: set[int] = set()
        for cid in frontier:
            nxt |= adj.get(cid, set())
        nxt -= extra
        extra |= nxt
        frontier = nxt
    return extra


def pair_split(quartet: frozenset[int], adj: dict[int, set[int]]) -> frozenset[int] | None:
    """The unique pair in the quartet that touches while its complement does not."""
    if not quartet <= set(adj):
        return None
    found: list[frozenset[int]] = []
    cells = sorted(quartet)
    for a, b in combinations(cells, 2):
        pair = frozenset((a, b))
        other = quartet - pair
        c, d = tuple(other)
        ab = b in adj.get(a, set())
        cd = d in adj.get(c, set())
        if ab and not cd:
            found.append(pair)
    if len(found) == 1:
        return found[0]
    return None


def consecutive_runs(frames: list[int]) -> list[tuple[int, int]]:
    start = prev = frames[0]
    runs: list[tuple[int, int]] = []
    for t in frames[1:]:
        if t == prev + 1:
            prev = t
            continue
        runs.append((start, prev))
        start = prev = t
    runs.append((start, prev))
    return runs


def stable_pair(
    adj: list[dict[int, set[int]]],
    present: list[set[int]],
    quartet: frozenset[int],
    start: int,
    end: int,
) -> frozenset[int] | None:
    """Unique contacting pair identical on every frame in [start, end] inclusive.

    Returns None if any frame is out of range, missing quartet cells, has no
    unique pair_split, or disagrees with the others.
    """
    n = len(adj)
    if start > end or start < 0 or end >= n:
        return None
    found: frozenset[int] | None = None
    for t in range(start, end + 1):
        if not quartet <= present[t]:
            return None
        pair = pair_split(quartet, adj[t])
        if pair is None:
            return None
        if found is None:
            found = pair
        elif pair != found:
            return None
    return found


def is_complementary(
    quartet: frozenset[int], before: frozenset[int], after: frozenset[int]
) -> bool:
    """True T1 diagonals: after is exactly quartet - before (no shared cell)."""
    if before & after:
        return False
    return before | after == quartet and len(before) == 2 and len(after) == 2


def _edge_set(
    adj: dict[int, set[int]], allowed: set[int] | None = None
) -> set[frozenset[int]]:
    edges: set[frozenset[int]] = set()
    for a, neigh in adj.items():
        ia = int(a)
        if allowed is not None and ia not in allowed:
            continue
        for b in neigh:
            ib = int(b)
            if ia >= ib:
                continue
            if allowed is not None and ib not in allowed:
                continue
            edges.add(frozenset((ia, ib)))
    return edges


def _sides_persist(
    lost: frozenset[int],
    gained: frozenset[int],
    adj0: dict[int, set[int]],
    adj1: dict[int, set[int]],
) -> bool:
    """Four side contacts of the quartet exist in both frames (CellFlow rule)."""
    u1, v1 = tuple(sorted(lost))
    u2, v2 = tuple(sorted(gained))
    for pair in (
        frozenset((u1, u2)),
        frozenset((u1, v2)),
        frozenset((v1, u2)),
        frozenset((v1, v2)),
    ):
        if pair == lost or pair == gained:
            continue
        a, b = tuple(pair)
        if b not in adj0.get(a, ()) or b not in adj1.get(a, ()):
            return False
    return True


def quartet_centroid(
    labels: np.ndarray, quartet: frozenset[int]
) -> tuple[float, float]:
    ys: list[float] = []
    xs: list[float] = []
    for cid in quartet:
        yy, xx = np.nonzero(labels == cid)
        if yy.size == 0:
            continue
        ys.append(float(yy.mean()))
        xs.append(float(xx.mean()))
    if not ys:
        return -1.0, -1.0
    return float(np.mean(ys)), float(np.mean(xs))


def detect_neighbour_swaps(
    adj: list[dict[int, set[int]]],
    present: list[set[int]],
    stack: np.ndarray,
) -> list[dict]:
    """Consecutive-frame complementary neighbour swaps (CellFlow graph rule).

    Lost vs gained contacts, four distinct cells, four sides persist, cells
    present in both frames. No 4-way vertex required.
    """
    rows: list[dict] = []
    n = len(adj)
    for t in range(n - 1):
        both = present[t] & present[t + 1]
        removed = _edge_set(adj[t], both) - _edge_set(adj[t + 1], both)
        added = _edge_set(adj[t + 1], both) - _edge_set(adj[t], both)
        seen_q: set[frozenset[int]] = set()
        for lost in removed:
            for gained in added:
                quartet = lost | gained
                if len(quartet) != 4 or quartet in seen_q:
                    continue
                if not quartet <= both:
                    continue
                if not is_complementary(quartet, lost, gained):
                    continue
                if not _sides_persist(lost, gained, adj[t], adj[t + 1]):
                    continue
                seen_q.add(quartet)
                y, x = quartet_centroid(stack[t], quartet)
                rows.append(
                    {
                        "frame": t,
                        "losing": lost,
                        "gaining": gained,
                        "y": y,
                        "x": x,
                    }
                )
    return rows


def matches_fourway_pairs(
    before: frozenset[int],
    after: frozenset[int],
    junc: frozenset[int] | None,
    swap: frozenset[int] | None,
) -> bool:
    """Contact swap must be the 4-way junction/swap bonds, either orientation."""
    if junc is None or swap is None:
        return False
    if before == junc and after == swap:
        return True
    if before == swap and after == junc:
        return True
    return False


def pair_outline_mask(labels: np.ndarray, a: int, b: int, reach: int) -> np.ndarray:
    """Facing rims of A and B, plus the gap between them — not a chord through cells."""
    ma = labels == a
    mb = labels == b
    if not ma.any() or not mb.any():
        return np.zeros(labels.shape, dtype=bool)
    struct = ndi.generate_binary_structure(2, 1)
    oa = ma & ~ndi.binary_erosion(ma, structure=struct, border_value=0)
    ob = mb & ~ndi.binary_erosion(mb, structure=struct, border_value=0)
    near_b = ndi.binary_dilation(mb, structure=struct, iterations=max(reach, 1))
    near_a = ndi.binary_dilation(ma, structure=struct, iterations=max(reach, 1))
    gap = (labels == 0) & near_a & near_b
    return (oa & near_b) | (ob & near_a) | gap


def paint_pair(
    dst: np.ndarray,
    labels: np.ndarray,
    pair: frozenset[int],
    color: tuple[int, int, int],
    reach: int,
) -> None:
    a, b = tuple(pair)
    mask = pair_outline_mask(labels, a, b, reach)
    if mask.any():
        dst[mask] = color


def grayscale_rgb(rgb: np.ndarray) -> np.ndarray:
    luma = (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )
    g = np.clip(luma, 0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def load_rgb(path: Path, frame_indices: np.ndarray, h: int, w: int) -> np.ndarray:
    rgb = np.asarray(tifffile.imread(path))
    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        rgb = rgb[np.newaxis, ...]
    if rgb.ndim == 4 and rgb.shape[1] == 3 and rgb.shape[-1] != 3:
        rgb = np.moveaxis(rgb, 1, -1)
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected T,Y,X,3 RGB; got {rgb.shape}")
    if int(frame_indices.max()) >= rgb.shape[0]:
        raise ValueError(
            f"RGB has {rgb.shape[0]} frames; need index {int(frame_indices.max())}"
        )
    rgb = rgb[frame_indices]
    if rgb.shape[1:3] != (h, w):
        raise ValueError(f"RGB YX {rgb.shape[1:3]} != labels {(h, w)}")
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def persist_run(
    adj: list[dict[int, set[int]]],
    present: list[set[int]],
    quartet: frozenset[int],
    pair: frozenset[int],
    start: int,
    direction: int,
) -> int:
    """Consecutive frames from ``start`` in ``direction`` where ``pair`` is unique."""
    n = len(adj)
    count = 0
    t = int(start)
    while 0 <= t < n:
        if not quartet <= present[t]:
            break
        found = pair_split(quartet, adj[t])
        if found != pair:
            break
        count += 1
        t += direction
    return count


def neighbour_exchange_frames(
    adj: list[dict[int, set[int]]],
    present: list[set[int]],
    quartet: frozenset[int],
    ab: frozenset[int],
    cd: frozenset[int],
) -> list[int]:
    """Frames where the unique contact flips between this T1's ab and cd only.

    Other complementary matchings of the same quartet are ignored. The recorded
    index is the first frame of the new unique pair (not a 4-way, where neither
    pair is unique).
    """
    allowed = {ab, cd}
    prev: frozenset[int] | None = None
    times: list[int] = []
    for t in range(len(adj)):
        if not quartet <= present[t]:
            prev = None
            continue
        pair = pair_split(quartet, adj[t])
        if pair is None or pair not in allowed:
            continue
        if prev is not None and pair != prev:
            times.append(t)
        prev = pair
    return times


def count_swaps(
    adj: list[dict[int, set[int]]],
    present: list[set[int]],
    quartet: frozenset[int],
    ab: frozenset[int],
    cd: frozenset[int],
) -> int:
    """How many ab↔cd flips this T1 matching makes over the movie."""
    return len(neighbour_exchange_frames(adj, present, quartet, ab, cd))


def annotate_neighbour_exchanges(
    events: list[dict],
    adj: list[dict[int, set[int]]],
    present: list[set[int]],
) -> None:
    """Store first/last ab↔cd swap times for each event's breaking/building pairs."""
    cache: dict[tuple[frozenset[int], frozenset[int], frozenset[int]], list[int]] = {}
    for e in events:
        quartet = e["quartet"]
        ab = e["pair_before"]
        cd = e["pair_after"]
        key = (quartet, ab, cd)
        if key not in cache:
            cache[key] = neighbour_exchange_frames(adj, present, quartet, ab, cd)
        times = cache[key]
        e["t_first_swap"] = times[0] if times else int(e["t_start"])
        e["t_last_swap"] = times[-1] if times else int(e["t_end"])


def length_t0(event: dict) -> int:
    """dt=0: first neighbour swap if it precedes the first 4-way, else the 4-way."""
    t_start = int(event["t_start"])
    t_first = int(event.get("t_first_swap", t_start))
    if t_first < t_start:
        return t_first
    return t_start


def length_csv_span(
    event: dict,
    n: int,
    length_pre: int,
    length_post: int,
) -> tuple[int, int]:
    """±length_pre/post around length_t0, stretched to last neighbour swap."""
    t_anchor = length_t0(event)
    t_last = int(event.get("t_last_swap", event["t_end"]))
    t0 = max(0, t_anchor - length_pre)
    t1 = min(n - 1, t_last + length_post)
    return t0, t1


def _event_xy(
    four_xy: list[dict[frozenset[int], tuple[float, float]]],
    quartet: frozenset[int],
    t0: int,
    t1: int,
) -> tuple[float, float]:
    ys, xs = [], []
    for t in range(t0, t1 + 1):
        if 0 <= t < len(four_xy):
            xy = four_xy[t].get(quartet)
            if xy is not None:
                ys.append(xy[0])
                xs.append(xy[1])
    if not ys:
        return -1.0, -1.0
    return float(np.mean(ys)), float(np.mean(xs))


def _event_row(
    quartet: frozenset[int],
    *,
    t0: int,
    t1: int,
    pair_before: frozenset[int],
    pair_after: frozenset[int],
    y: float,
    x: float,
    kind: str,
    source: str,
    persist_pre: int,
    persist_post: int,
    n_swaps: int,
    has_fourway: bool = True,
    other_matching: bool = False,
) -> dict:
    cells = sorted(quartet)
    return {
        "t_start": t0,
        "t_end": t1,
        "t_before": t0 - persist_pre,
        "t_after": t1 + persist_post,
        "A": cells[0],
        "B": cells[1],
        "C": cells[2],
        "D": cells[3],
        "quartet": quartet,
        "pair_before": pair_before,
        "pair_after": pair_after,
        "y": y,
        "x": x,
        "kind": kind,
        "source": source,
        "persist_pre": int(persist_pre),
        "persist_post": int(persist_post),
        "n_swaps": int(n_swaps),
        "has_fourway": int(bool(has_fourway)),
        "other_matching": int(bool(other_matching)),
    }


def detect_t1s(
    quartet_frames: dict[frozenset[int], list[int]],
    adj: list[dict[int, set[int]]],
    present: list[set[int]],
    four_xy: list[dict[frozenset[int], tuple[float, float]]],
    quartet_bonds: dict[frozenset[int], tuple[frozenset[int], frozenset[int]]],
    *,
    search: int = 1,
) -> list[dict]:
    """Score every 4-way run: completed complementary swaps and aborted returns.

    ``search`` is unused (kept so old callers still work). Persist is measured
    by walking outward from the 4-way rather than requiring a fixed window.
    """
    del search
    events: list[dict] = []
    n = len(adj)
    for quartet, frames in quartet_frames.items():
        bonds = quartet_bonds.get(quartet)
        if bonds is None:
            continue
        junc, swap = bonds
        n_swaps = count_swaps(adj, present, quartet, junc, swap)
        for t0, t1 in consecutive_runs(sorted(frames)):
            if t0 < 1 or t1 + 1 >= n:
                continue
            pair_before = pair_split(quartet, adj[t0 - 1])
            pair_after = pair_split(quartet, adj[t1 + 1])
            if pair_before is None or pair_after is None:
                continue
            persist_pre = persist_run(
                adj, present, quartet, pair_before, t0 - 1, -1
            )
            persist_post = persist_run(
                adj, present, quartet, pair_after, t1 + 1, 1
            )
            y, x = _event_xy(four_xy, quartet, t0, t1)
            if pair_before == pair_after:
                events.append(
                    _event_row(
                        quartet,
                        t0=t0,
                        t1=t1,
                        pair_before=pair_before,
                        pair_after=pair_after,
                        y=y,
                        x=x,
                        kind="aborted",
                        source="tm",
                        persist_pre=persist_pre,
                        persist_post=persist_post,
                        n_swaps=n_swaps,
                    )
                )
                continue
            if not is_complementary(quartet, pair_before, pair_after):
                continue
            if not matches_fourway_pairs(pair_before, pair_after, junc, swap):
                continue
            events.append(
                _event_row(
                    quartet,
                    t0=t0,
                    t1=t1,
                    pair_before=pair_before,
                    pair_after=pair_after,
                    y=y,
                    x=x,
                    kind="completed",
                    source="tm",
                    persist_pre=persist_pre,
                    persist_post=persist_post,
                    n_swaps=n_swaps,
                )
            )
    events.sort(key=lambda e: (e["t_start"], e["A"], e["B"]))
    return events


def _matching_key(e: dict) -> frozenset[frozenset[int]]:
    """Complementary pair set for this event (AB↔CD vs AC↔BD stay distinct)."""
    return frozenset({e["pair_before"], e["pair_after"]})


def merge_flicker_events(events: list[dict]) -> list[dict]:
    """Collapse same-matching rows with n_swaps>=2 (first t0 … last t1)."""
    by_key: dict[
        tuple[frozenset[int], frozenset[frozenset[int]]], list[dict]
    ] = defaultdict(list)
    for e in events:
        by_key[(e["quartet"], _matching_key(e))].append(e)
    out: list[dict] = []
    n_groups = 0
    n_rows = 0
    for (quartet, _matching), group in by_key.items():
        n_swaps = max(int(e.get("n_swaps", 0)) for e in group)
        if n_swaps < 2 or len(group) < 2:
            out.extend(group)
            continue
        group = sorted(group, key=lambda e: (int(e["t_start"]), int(e["t_end"])))
        first, last = group[0], group[-1]
        pair_before = first["pair_before"]
        pair_after = last["pair_after"]
        kind = "aborted" if pair_before == pair_after else "completed"
        has_fourway = any(int(e.get("has_fourway", 1)) for e in group)
        other_matching = any(int(e.get("other_matching", 0)) for e in group)
        sources = {str(e.get("source") or "tm") for e in group}
        if sources <= {"tm"}:
            source = "tm"
        elif sources <= {"swap"}:
            source = "swap"
        else:
            source = "both"
        ys = [float(e["y"]) for e in group if float(e["y"]) >= 0]
        xs = [float(e["x"]) for e in group if float(e["x"]) >= 0]
        y = float(np.mean(ys)) if ys else float(first["y"])
        x = float(np.mean(xs)) if xs else float(first["x"])
        out.append(
            _event_row(
                quartet,
                t0=int(first["t_start"]),
                t1=int(last["t_end"]),
                pair_before=pair_before,
                pair_after=pair_after,
                y=y,
                x=x,
                kind=kind,
                source=source,
                persist_pre=int(first.get("persist_pre", 0)),
                persist_post=int(last.get("persist_post", 0)),
                n_swaps=n_swaps,
                has_fourway=bool(has_fourway),
                other_matching=bool(other_matching),
            )
        )
        n_groups += 1
        n_rows += len(group)
    out.sort(key=lambda e: (e["t_start"], e["A"], e["B"]))
    if n_groups:
        print(
            f"Merged {n_rows} flicker rows into {n_groups} events "
            f"(n_swaps>=2, first 4-way to last, per matching)",
            flush=True,
        )
    return out


T1_FILTER_MODES = (
    ("all", "All (baseline)"),
    ("persist_3", "Persist 3"),
    ("persist_5", "Persist 5"),
    ("persist_7", "Persist 7"),
    ("flicker", "Flicker"),
    ("aborted", "Aborted"),
)


def event_passes_filter(event: dict, mode: str) -> bool:
    """True if a scored event belongs in ``mode`` (plugin / picker)."""
    kind = str(event.get("kind", "completed"))
    persist_pre = int(event.get("persist_pre", -1))
    persist_post = int(event.get("persist_post", -1))
    n_swaps = int(event.get("n_swaps", 0))
    if mode == "all":
        return kind == "completed"
    if mode.startswith("persist_"):
        n = int(mode.split("_", 1)[1])
        if persist_pre < 0 or persist_post < 0:
            return False
        return kind == "completed" and persist_pre >= n and persist_post >= n
    if mode == "flicker":
        return n_swaps >= 2
    if mode == "aborted":
        return kind == "aborted"
    return kind == "completed"


def filter_t1_events(events: list[dict], mode: str) -> list[dict]:
    return [e for e in events if event_passes_filter(e, mode)]


def fourway_counts(events: list[dict]) -> tuple[int, int]:
    """How many events have a 4-cell vertex vs neighbour-swap only."""
    yes = sum(1 for e in events if int(e.get("has_fourway", 1)))
    return yes, len(events) - yes


def _overlaps(t0: int, t1: int, frame: int) -> bool:
    return t0 - 1 <= frame <= t1 + 1


def merge_neighbour_swap_events(
    events: list[dict],
    swap_rows: list[dict],
    adj: list[dict[int, set[int]]],
    present: list[set[int]],
    four_xy: list[dict[frozenset[int], tuple[float, float]]],
    quartet_bonds: dict[frozenset[int], tuple[frozenset[int], frozenset[int]]],
    border_by_t: list[set[int]],
) -> tuple[list[dict], int, int]:
    """Union TM 4-way events with neighbour-swap hits; same quartet+time collapse.

    Complementary swaps that do not match recorded 4-way bonds are kept and
    tagged ``other_matching`` (red overlay). Bond-matching swap-only rows stay
    ``no_4way``. New swap-only rows that touch embryo-border cells are dropped
    (same rule as interior-only 4-ways).

    Returns (events, n_new, n_already_tm).
    """
    n = len(adj)
    n_new = 0
    n_already = 0
    for row in swap_rows:
        losing = row["losing"]
        gaining = row["gaining"]
        quartet = losing | gaining
        if len(quartet) != 4 or len(losing) != 2 or len(gaining) != 2:
            continue
        if not is_complementary(quartet, losing, gaining):
            continue
        frame = int(row["frame"])
        if frame < 0 or frame >= n:
            continue
        matched = None
        for ev in events:
            if ev["quartet"] != quartet:
                continue
            if _overlaps(ev["t_start"], ev["t_end"], frame):
                matched = ev
                break
        if matched is not None:
            n_already += 1
            if matched["source"] == "tm":
                matched["source"] = "both"
            continue
        if quartet & border_by_t[frame]:
            continue
        bonds = quartet_bonds.get(quartet)
        if bonds is None:
            other_matching = True
        else:
            junc, swap = bonds
            other_matching = not matches_fourway_pairs(losing, gaining, junc, swap)
        persist_pre = persist_run(adj, present, quartet, losing, frame, -1)
        persist_post = persist_run(
            adj, present, quartet, gaining, min(frame + 1, n - 1), 1
        )
        y, x = float(row["y"]), float(row["x"])
        if y < 0:
            y, x = _event_xy(four_xy, quartet, frame, frame)
        events.append(
            _event_row(
                quartet,
                t0=frame,
                t1=frame,
                pair_before=losing,
                pair_after=gaining,
                y=y,
                x=x,
                kind="completed",
                source="swap",
                persist_pre=persist_pre,
                persist_post=persist_post,
                n_swaps=count_swaps(adj, present, quartet, losing, gaining),
                has_fourway=False,
                other_matching=other_matching,
            )
        )
        n_new += 1
    events.sort(key=lambda e: (e["t_start"], e["A"], e["B"]))
    return events, n_new, n_already


def length_trace_ok(
    event: dict,
    verts_3_by_t: list[list[dict]],
    window: int,
) -> bool:
    """True if AB/CD vertex lengths exist on all ±window frames around the 4-way."""
    if window <= 0:
        return True
    n = len(verts_3_by_t)
    t0, t1 = event["t_start"], event["t_end"]
    if t0 - window < 0 or t1 + window >= n:
        return False
    quartet = event["quartet"]
    ab = event["pair_before"]
    cd = event["pair_after"]
    for t in range(t0 - window, t0):
        lv, _, _ = vertex_bond(verts_3_by_t[t], ab, quartet)
        if lv is None:
            return False
    for t in range(t1 + 1, t1 + 1 + window):
        lv, _, _ = vertex_bond(verts_3_by_t[t], cd, quartet)
        if lv is None:
            return False
    return True


def render_overlay(
    rgb: np.ndarray,
    labels: list[np.ndarray],
    events: list[dict],
    four_xy: list[dict[frozenset[int], tuple[float, float]]],
    *,
    pre: int,
    post: int,
    reach: int,
) -> np.ndarray:
    out = grayscale_rgb(rgb)
    n = out.shape[0]
    h, w = out.shape[1:3]

    for ev in events:
        colors = type_colors(
            str(ev.get("kind") or "completed"),
            ev.get("has_fourway", 1),
            int(ev.get("n_swaps") or 0),
            ev.get("other_matching", 0),
        )
        breaking = colors["breaking"]
        building = colors["building"]
        vertex = colors["vertex"]
        ab = ev["pair_before"]
        cd = ev["pair_after"]
        t0, t1 = ev["t_start"], ev["t_end"]
        win0 = max(0, t0 - pre)
        win1 = min(n - 1, t1 + post)
        for t in range(win0, win1 + 1):
            dst = out[t]
            if t < t0:
                paint_pair(dst, labels[t], ab, breaking, reach)
            elif t > t1:
                paint_pair(dst, labels[t], cd, building, reach)
            else:
                paint_pair(dst, labels[t], ab, breaking, reach)
                paint_pair(dst, labels[t], cd, building, reach)
                xy = four_xy[t].get(ev["quartet"])
                if xy is None:
                    xy = (ev["y"], ev["x"])
                if xy[0] < 0 or xy[1] < 0:
                    continue
                rr, cc = disk(
                    (int(round(xy[0])), int(round(xy[1]))), 2, shape=(h, w)
                )
                dst[rr, cc] = vertex
    return out


def labels_gray_rgb(labels: np.ndarray, max_lab: int = 0) -> np.ndarray:
    """Fallback gray from label IDs when tracked_rgb is unavailable."""
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if max_lab <= 0:
        return out
    if max_lab <= 255:
        gray = labels.astype(np.uint8, copy=False)
    else:
        gray = np.zeros((h, w), dtype=np.uint8)
        cell = labels > 0
        gray[cell] = np.clip(
            np.rint(labels[cell].astype(np.float64) * (255.0 / max_lab)),
            1,
            255,
        ).astype(np.uint8)
    out[..., 0] = gray
    out[..., 1] = gray
    out[..., 2] = gray
    return out


def render_fourway_dots(
    labels: np.ndarray,
    events: list[dict],
    four_xy: list[dict[frozenset[int], tuple[float, float]]],
    rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Grayscale base + one type-colored pixel per event (4-way or CellFlow xy)."""
    n, h, w = labels.shape
    if rgb is not None:
        if rgb.shape[:3] != (n, h, w) or rgb.shape[-1] != 3:
            raise ValueError(
                f"rgb shape {rgb.shape} does not match labels {(n, h, w)}"
            )
        out = grayscale_rgb(rgb)
    else:
        max_lab = int(labels.max()) if labels.size else 0
        out = np.empty((n, h, w, 3), dtype=np.uint8)
        for t in range(n):
            out[t] = labels_gray_rgb(labels[t], max_lab)
    for ev in events:
        vertex = type_colors(
            str(ev.get("kind") or "completed"),
            ev.get("has_fourway", 1),
            int(ev.get("n_swaps") or 0),
            ev.get("other_matching", 0),
        )["vertex"]
        quartet = ev["quartet"]
        t0, t1 = ev["t_start"], ev["t_end"]
        for t in range(t0, t1 + 1):
            xy = four_xy[t].get(quartet)
            if xy is None:
                xy = (ev["y"], ev["x"])
            y = int(round(xy[0]))
            x = int(round(xy[1]))
            if 0 <= y < h and 0 <= x < w:
                out[t, y, x] = vertex
    return out


def run_t1_stack(
    stack: np.ndarray,
    *,
    rgb: np.ndarray | None = None,
    frame_base: int = 0,
    expand: int = 3,
    bond_cutoff: float = 3.0,
    border_rings: int = 1,
    search: int = 3,
    length_window: int = 0,
    pre: int = 3,
    post: int = 3,
    include_neighbour_swaps: bool = True,
    progress: Callable[[int, str], None] | None = None,
    exclude_ids: set[int] | None = None,
) -> dict:
    """Detect T1s on a label stack. Same rules as the CLI.

    ``stack`` is T,Y,X. Blank frames are dropped. ``rgb``, when given, must
    match ``stack`` before that drop (T,Y,X,3). ``frame_base`` is added to
    kept indices so CSV frames stay aligned with a sliced movie.
    ``search`` is accepted for compatibility and is not used to drop events.
    ``include_neighbour_swaps`` adds neighbour-swap extras that are
    not already the same quartet as a TM 4-way event.
    ``exclude_ids``: track IDs left out of T1 detection, e.g. dividing
    mothers and their daughters from ``{stem}_lineage.csv``. Any T1 whose
    quartet contains one of them is dropped.
    """
    exclude_ids = {int(i) for i in (exclude_ids or ())}
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"expected T,Y,X labels, got {stack.shape}")
    n_in = stack.shape[0]
    if rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.ndim == 3 and rgb.shape[-1] == 3:
            rgb = rgb[np.newaxis, ...]
        if rgb.ndim == 4 and rgb.shape[1] == 3 and rgb.shape[-1] != 3:
            rgb = np.moveaxis(rgb, 1, -1)
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected T,Y,X,3 RGB; got {rgb.shape}")
        if rgb.shape[0] != n_in:
            raise ValueError(
                f"RGB frames {rgb.shape[0]} != label frames {n_in}"
            )
        if rgb.shape[1:3] != stack.shape[1:3]:
            raise ValueError(
                f"RGB YX {rgb.shape[1:3]} != labels {stack.shape[1:3]}"
            )
    stack, kept = drop_blank(stack)
    frame_indices = kept + int(frame_base)
    n = stack.shape[0]
    print(
        f"non-blank frames={n} orig={int(frame_indices[0])}..{int(frame_indices[-1])}",
        flush=True,
    )
    if rgb is not None:
        rgb = np.ascontiguousarray(rgb[kept], dtype=np.uint8)

    adj: list[dict[int, set[int]]] = []
    present: list[set[int]] = []
    border_by_t: list[set[int]] = []
    verts_3_by_t: list[list[dict]] = []
    four_xy: list[dict[frozenset[int], tuple[float, float]]] = []
    quartet_frames: dict[frozenset[int], list[int]] = defaultdict(list)
    quartet_bonds: dict[frozenset[int], tuple[frozenset[int], frozenset[int]]] = {}
    fourway_rows: list[dict] = []
    n4_all = n4_border = n4_kept = 0

    for t in range(n):
        if progress is not None:
            progress(int(100 * t / max(n, 1)), f"frame {int(frame_indices[t])}")
        verts, verts_3, _expanded = junctions_one_frame(stack[t], expand, bond_cutoff)
        verts_3_by_t.append(verts_3)
        a = build_adjacency(stack[t], expand=expand)
        adj.append(a)
        present.append(
            {int(v) for v in np.unique(stack[t]) if v != 0} - exclude_ids
        )
        excluded = (
            border_plus_rings(stack[t], a, border_rings) if border_rings > 0 else set()
        )
        border_by_t.append(excluded)
        xy_map: dict[frozenset[int], tuple[float, float]] = {}
        n4_f = n4b_f = 0
        for v in verts:
            if int(v["n_cells"]) != 4 and int(v["multiplicity"]) < 4:
                continue
            ids = tuple(int(x) for x in v["cell_ids"])
            q = frozenset(ids)
            on_border = bool(q & excluded) if q else True
            junc = v.get("junction_pair")
            swap = v.get("swap_pair")
            if junc is not None and not isinstance(junc, frozenset):
                junc = frozenset(int(x) for x in junc)
            if swap is not None and not isinstance(swap, frozenset):
                swap = frozenset(int(x) for x in swap)
            if (
                int(v["n_cells"]) == 4
                and junc is not None
                and swap is None
                and len(junc) == 2
            ):
                swap = q - junc
            if int(v["n_cells"]) == 4:
                n4_all += 1
                n4_f += 1
                if on_border:
                    n4_border += 1
                    n4b_f += 1
                else:
                    n4_kept += 1
                    quartet_frames[q].append(t)
                    xy_map[q] = (float(v["y"]), float(v["x"]))
                    if (
                        q not in quartet_bonds
                        and junc is not None
                        and swap is not None
                        and len(junc) == 2
                        and len(swap) == 2
                        and junc | swap == q
                        and not (junc & swap)
                    ):
                        quartet_bonds[q] = (junc, swap)
            fourway_rows.append(
                {
                    "frame": int(frame_indices[t]),
                    "t_index": t,
                    "y": round(float(v["y"]), 2),
                    "x": round(float(v["x"]), 2),
                    "n_cells": int(v["n_cells"]),
                    "kind": v["kind"],
                    "cell_ids": fmt_ids(v["cell_ids"]),
                    "on_border": int(on_border),
                    "junction_pair": fmt_ids(junc) if junc else "",
                    "swap_pair": fmt_ids(swap) if swap else "",
                    "length_vertex_px": (
                        ""
                        if v.get("length_vertex_px") is None
                        else round(float(v["length_vertex_px"]), 2)
                    ),
                }
            )
        four_xy.append(xy_map)
        if t == 0 or t == n - 1 or (t + 1) % 10 == 0:
            print(
                f"  frame {int(frame_indices[t])}: 4-way={n4_f} "
                f"border={n4b_f} interior={n4_f - n4b_f}",
                flush=True,
            )

    print(
        f"4-ways all={n4_all} border={n4_border} interior={n4_kept} "
        f"quartets={len(quartet_frames)} with_bonds={len(quartet_bonds)}",
        flush=True,
    )
    events = detect_t1s(
        quartet_frames,
        adj,
        present,
        four_xy,
        quartet_bonds,
        search=search,
    )
    n_tm = len(events)
    if include_neighbour_swaps:
        if progress is not None:
            progress(90, "neighbour-swaps")
        swap_rows = detect_neighbour_swaps(adj, present, stack)
        events, n_new, n_already = merge_neighbour_swap_events(
            events, swap_rows, adj, present, four_xy, quartet_bonds, border_by_t
        )
        swap_note = (
            f"neighbour_swaps={len(swap_rows)} new={n_new} already_tm={n_already}"
        )
        print(swap_note, flush=True)
    else:
        swap_note = "neighbour_swaps skipped (--no-neighbour-swaps)"
        print(swap_note, flush=True)
    events = merge_flicker_events(events)
    annotate_neighbour_exchanges(events, adj, present)
    n_swap = sum(1 for e in events if e["kind"] == "completed")
    if length_window > 0:
        kept: list[dict] = []
        dropped = 0
        for e in events:
            if e["kind"] != "completed":
                kept.append(e)
                continue
            if length_trace_ok(e, verts_3_by_t, length_window):
                kept.append(e)
            else:
                dropped += 1
        events = kept
        print(
            f"Completed T1s={sum(1 for e in events if e['kind']=='completed')} "
            f"(dropped {dropped} lacking ±{length_window} "
            f"measurable lengths; swap-ok={n_swap})",
            flush=True,
        )
    n_excluded = 0
    if exclude_ids:
        before = len(events)
        events = [e for e in events if not (set(e["quartet"]) & exclude_ids)]
        n_excluded = before - len(events)
        print(
            f"Excluded dividing cells: {len(exclude_ids)} track IDs "
            f"(dropped {n_excluded} remaining events with them)",
            flush=True,
        )
    shown = filter_t1_events(events, "all")
    n_4v, n_no4 = fourway_counts(shown)
    print(
        f"Baseline T1s={len(shown)} completed={n_swap} "
        f"aborted={sum(1 for e in events if e['kind']=='aborted')} "
        f"tm_rows={n_tm} 4-cell vertex={n_4v} no 4-way={n_no4}",
        flush=True,
    )

    labels_list = [stack[t] for t in range(n)]
    if rgb is not None:
        overlay_base = grayscale_rgb(rgb)
    else:
        max_lab = int(stack.max()) if stack.size else 0
        overlay_base = np.empty((n, stack.shape[1], stack.shape[2], 3), dtype=np.uint8)
        for t in range(n):
            overlay_base[t] = labels_gray_rgb(stack[t], max_lab)
    overlay = render_overlay(
        overlay_base, labels_list, shown, four_xy, pre=pre, post=post, reach=expand
    )
    dots = render_fourway_dots(stack, shown, four_xy, rgb=rgb)
    if progress is not None:
        progress(100, f"T1s={len(shown)}")
    return {
        "stack": stack,
        "frame_indices": frame_indices,
        "events": events,
        "events_shown": shown,
        "four_xy": four_xy,
        "verts_3_by_t": verts_3_by_t,
        "adj": adj,
        "present": present,
        "fourway_rows": fourway_rows,
        "overlay": overlay,
        "dots": dots,
        "overlay_base": overlay_base,
        "n_frames": n,
        "n_4way": n4_all,
        "n_4way_border_excluded": n4_border,
        "n_4way_kept": n4_kept,
        "n_excluded_ids": len(exclude_ids),
        "n_excluded_events": n_excluded,
        "n_quartets_interior": len(quartet_frames),
        "n_quartets_with_bonds": len(quartet_bonds),
        "n_t1_confirmed": len(shown),
        "n_swap": n_swap,
        "n_aborted": sum(1 for e in events if e["kind"] == "aborted"),
        "n_with_fourway": n_4v,
        "n_without_fourway": n_no4,
        "swap_note": swap_note,
        "expand": expand,
        "pre": pre,
        "post": post,
    }


def paint_t1_filter(result: dict, mode: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Re-paint overlay and dots for a stored ``run_t1_stack`` result."""
    shown = filter_t1_events(result["events"], mode)
    stack = result["stack"]
    labels_list = [stack[t] for t in range(stack.shape[0])]
    overlay = render_overlay(
        result["overlay_base"],
        labels_list,
        shown,
        result["four_xy"],
        pre=int(result.get("pre", 3)),
        post=int(result.get("post", 3)),
        reach=int(result.get("expand", 3)),
    )
    dots = render_fourway_dots(
        stack, shown, result["four_xy"], rgb=result["overlay_base"]
    )
    return overlay, dots, shown


def load_division_ids(path: Path) -> set[int]:
    """Mothers and daughters from a ``*_lineage.csv`` (cellpose_track_divisions)."""
    ids: set[int] = set()
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            parent = int(float(row.get("parent_track_id") or 0))
            if parent:
                ids.add(int(float(row["track_id"])))
                ids.add(parent)
    return ids


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T1s: interior TM 4-ways whose contacting pair swaps."
    )
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--rgb", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--frames",
        type=str,
        default=None,
        help="START:END (end exclusive), e.g. 0:40.",
    )
    p.add_argument("--expand", type=int, default=3)
    p.add_argument("--bond-cutoff", type=float, default=3.0)
    p.add_argument(
        "--border-rings",
        type=int,
        default=1,
        help="0=keep edge 4-ways; 1=drop cells touching background; 2=plus one.",
    )
    p.add_argument(
        "--search",
        type=int,
        default=3,
        help=(
            "Unused; persist is scored per event. Kept so old command lines "
            "still parse."
        ),
    )
    p.add_argument("--pre", type=int, default=3, help="Overlay frames before 4-way.")
    p.add_argument("--post", type=int, default=3, help="Overlay frames after 4-way.")
    p.add_argument(
        "--length-pre",
        type=int,
        default=10,
        help="Junction-length CSV frames before first 4-way / first AB↔CD swap.",
    )
    p.add_argument(
        "--length-post",
        type=int,
        default=10,
        help=(
            "Junction-length CSV frames after the last AB↔CD neighbour swap "
            "(not after the last 4-way vertex)."
        ),
    )
    p.add_argument(
        "--length-window",
        type=int,
        default=0,
        help=(
            "Require measurable AB (before) and CD (after) vertex lengths on "
            "this many frames on each side of the 4-way. 0 disables (baseline)."
        ),
    )
    p.add_argument(
        "--lineage",
        type=Path,
        default=None,
        help=(
            "Lineage CSV from cellpose_track_divisions.py. Default: "
            "{stem}_lineage.csv next to --labels, if present. Mothers and "
            "daughters of every division are excluded from T1s."
        ),
    )
    p.add_argument(
        "--keep-dividing",
        action="store_true",
        help="Do not exclude dividing cells even if a lineage CSV exists.",
    )
    p.add_argument(
        "--no-neighbour-swaps",
        "--no-cellflow",  # old name
        dest="no_neighbour_swaps",
        action="store_true",
        help="Skip neighbour-swap extras (TM 4-ways only).",
    )
    return p.parse_args(argv)


def require_compute() -> None:
    """Refuse login-node runs. Override: ALLOW_LOGIN_NODE=1."""
    if os.environ.get("ALLOW_LOGIN_NODE") == "1":
        print("ALLOW_LOGIN_NODE=1 — skipping compute-node check", flush=True)
        return
    job = os.environ.get("SLURM_JOB_ID")
    host = socket.gethostname()
    if not job:
        raise SystemExit(
            f"Refuse: not inside a Slurm job (host={host}). "
            "Run T1 from tmux on a compute allocation "
            "(bash scripts/hpc_alloc_tmux.sh)."
        )
    if "login" in host or re.match(r"^uc\d+(\.|$)", host):
        raise SystemExit(
            f"Refuse: login node ({host}, job={job}). "
            "Attach tmux on the compute node, then re-run."
        )


def main(argv=None) -> None:
    args = parse_args(argv)
    require_compute()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = input_stem(args.labels)
    # Pixel size / frame interval: from the labels TIFF, else the raw movie.
    cal = calibration_for(args.labels.expanduser(), stem=stem)
    print(f"Calibration: {cal.as_dict() if cal else 'none found (pixel/frame output only)'}", flush=True)

    print(f"Loading {args.labels}", flush=True)
    full = load_labels(args.labels.expanduser())
    f0, f1 = parse_frames(args.frames, full.shape[0])
    stack = full[f0:f1]
    print(f"slice {f0}:{f1} -> {stack.shape}", flush=True)
    rgb = None
    if args.rgb is not None and args.rgb.expanduser().is_file():
        print(f"Loading RGB {args.rgb}", flush=True)
        rgb = load_rgb(
            args.rgb.expanduser(),
            np.arange(f0, f1),
            stack.shape[1],
            stack.shape[2],
        )
    exclude_ids: set[int] = set()
    if not args.keep_dividing:
        lineage = args.lineage
        if lineage is None:
            lineage = tagged_path(args.labels.expanduser().parent, stem, "lineage.csv")
        if lineage.expanduser().is_file():
            exclude_ids = load_division_ids(lineage.expanduser())
            print(
                f"Lineage {lineage}: excluding {len(exclude_ids)} dividing "
                "track IDs (mothers + daughters)",
                flush=True,
            )
        elif args.lineage is not None:
            raise SystemExit(f"--lineage not found: {args.lineage}")
    result = run_t1_stack(
        stack,
        rgb=rgb,
        frame_base=f0,
        exclude_ids=exclude_ids,
        expand=args.expand,
        bond_cutoff=args.bond_cutoff,
        border_rings=args.border_rings,
        search=args.search,
        length_window=args.length_window,
        pre=args.pre,
        post=args.post,
        include_neighbour_swaps=not args.no_neighbour_swaps,
    )
    stack = result["stack"]
    frame_indices = result["frame_indices"]
    n = result["n_frames"]
    events = result["events"]
    fourway_rows = result["fourway_rows"]

    ev_path = tagged_path(out, stem, "t1_events.csv")
    ev_fields = [
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
        "other_matching",
    ]
    with ev_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ev_fields)
        writer.writeheader()
        for e in events:
            writer.writerow(
                {
                    "frame_start": int(frame_indices[e["t_start"]]),
                    "frame_end": int(frame_indices[e["t_end"]]),
                    "frame_first_swap": int(
                        frame_indices[
                            max(0, min(int(e.get("t_first_swap", e["t_start"])), n - 1))
                        ]
                    ),
                    "frame_last_swap": int(
                        frame_indices[
                            max(0, min(int(e.get("t_last_swap", e["t_end"])), n - 1))
                        ]
                    ),
                    "frame_before": int(
                        frame_indices[max(0, min(e["t_before"], n - 1))]
                    ),
                    "frame_after": int(
                        frame_indices[max(0, min(e["t_after"], n - 1))]
                    ),
                    "A": e["A"],
                    "B": e["B"],
                    "C": e["C"],
                    "D": e["D"],
                    "pair_breaking": fmt_ids(e["pair_before"]),
                    "pair_building": fmt_ids(e["pair_after"]),
                    "y": round(e["y"], 2),
                    "x": round(e["x"], 2),
                    "source": e.get("source", "tm"),
                    "kind": e.get("kind", "completed"),
                    "type": event_type(
                        str(e.get("kind", "completed")),
                        int(e.get("has_fourway", 1)),
                        int(e.get("n_swaps", 0)),
                        int(e.get("other_matching", 0)),
                    ),
                    "persist_pre": int(e.get("persist_pre", 0)),
                    "persist_post": int(e.get("persist_post", 0)),
                    "n_swaps": int(e.get("n_swaps", 0)),
                    "has_fourway": int(e.get("has_fourway", 1)),
                    "other_matching": int(e.get("other_matching", 0)),
                }
            )

    fw_path = tagged_path(out, stem, "fourway_vertices.csv")
    with fw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "t_index",
                "y",
                "x",
                "n_cells",
                "kind",
                "cell_ids",
                "on_border",
                "junction_pair",
                "swap_pair",
                "length_vertex_px",
            ],
        )
        writer.writeheader()
        writer.writerows(fourway_rows)

    # Wide junction_lengths.csv (positive) + junction_lengths_signed.csv (AB +, CD −).
    verts_3_by_t = result["verts_3_by_t"]
    adj = result["adj"]
    vertex_pos: dict[tuple[int, int], object] = {}
    vertex_signed: dict[tuple[int, int], object] = {}
    contact_pos: dict[tuple[int, int], object] = {}
    contact_signed: dict[tuple[int, int], object] = {}
    meta: list[tuple[int, str, int, object, object]] = []
    dts_all: set[int] = set()
    needed: set[int] = set()
    windows: list[tuple[int, dict, int, int, int]] = []
    for eid, e in enumerate(events):
        ab = e["pair_before"]
        cd = e["pair_after"]
        etype = event_type(
            str(e.get("kind", "completed")),
            int(e.get("has_fourway", 1)),
            int(e.get("n_swaps", 0)),
            int(e.get("other_matching", 0)),
        )
        t_zero = length_t0(e)
        frame_zero = int(frame_indices[max(0, min(t_zero, n - 1))])
        meta.append((eid, etype, frame_zero, ab, cd))
        t0, t1 = length_csv_span(e, n, args.length_pre, args.length_post)
        needed.update(range(t0, t1 + 1))
        windows.append((eid, e, t0, t1, frame_zero))
    expand = int(args.expand)
    cmaps: dict[int, dict[tuple[int, int], float]] = {}
    for t in needed:
        lab = stack[t]
        expanded = (
            expand_labels(lab, distance=expand) if expand > 0 else lab
        )
        cmaps[t] = contact_length_map(expanded)
    for eid, e, t0, t1, frame_zero in windows:
        ab = e["pair_before"]
        cd = e["pair_after"]
        quartet = e["quartet"]
        for t in range(t0, t1 + 1):
            frame = int(frame_indices[t])
            dt = frame - frame_zero
            dts_all.add(dt)
            pair, sign = active_abcd_pair(adj[t], ab, cd)
            if sign == 0 or pair is None:
                flag = length_blank_flag(adj[t], ab, cd)
                vertex_pos[(eid, dt)] = flag
                vertex_signed[(eid, dt)] = flag
                contact_pos[(eid, dt)] = flag
                contact_signed[(eid, dt)] = flag
                continue
            a, b = tuple(pair)
            lv, _, _ = vertex_bond(verts_3_by_t[t], pair, quartet)
            contact = lookup_contact(cmaps[t], a, b)
            mag_v: object = FLAG_NV if lv is None else round(float(lv), 2)
            mag_c: object = (
                round(contact, 2) if contact > 0 else FLAG_NV
            )
            vertex_pos[(eid, dt)] = mag_v
            vertex_signed[(eid, dt)] = _apply_sign(mag_v, sign)
            contact_pos[(eid, dt)] = mag_c
            contact_signed[(eid, dt)] = _apply_sign(mag_c, sign)
    ordered = sorted(
        meta,
        key=lambda m: (
            TYPE_SORT_ORDER.get(m[1], 99),
            m[2],
            m[0],
        ),
    )
    ordered_cols = [(eid, etype, ab, cd) for eid, etype, _, ab, cd in ordered]
    len_path = tagged_path(out, stem, "junction_lengths.csv")
    write_junction_lengths_wide(
        len_path,
        ordered_cols,
        vertex_pos,
        vertex_signed,
        contact_pos,
        contact_signed,
        sorted(dts_all),
    )
    qc = {
        "n_frames": n,
        "frame_slice": [f0, f1],
        "n_4way": result["n_4way"],
        "n_4way_border_excluded": result["n_4way_border_excluded"],
        "n_4way_kept": result["n_4way_kept"],
        "n_quartets_interior": result["n_quartets_interior"],
        "n_quartets_with_bonds": result["n_quartets_with_bonds"],
        "n_t1_confirmed": result["n_t1_confirmed"],
        "n_aborted": result.get("n_aborted", 0),
        "n_with_fourway": result.get("n_with_fourway", 0),
        "n_without_fourway": result.get("n_without_fourway", 0),
        "neighbour_swaps": result.get("swap_note", ""),
        "strict_complementary": True,
        "border_rings": args.border_rings,
        "expand": args.expand,
        "bond_cutoff": args.bond_cutoff,
        "search": args.search,
        "length_window": args.length_window,
        "length_pre": args.length_pre,
        "length_post": args.length_post,
        "signed_lengths": True,
        "overlay": {
            "completed_4way": "cyan vertex; orange break / cyan build",
            "flicker": "yellow vertex; yellow-orange break / lime build (n_swaps>=2)",
            "aborted": "magenta vertex; magenta break / pink build",
            "no_4way": "amber vertex; amber break / teal build (no 4-cell vertex)",
            "other_matching": (
                "red vertex; red-orange break / dark red build "
                "(other complementary matching)"
            ),
            "t1_overlay.tif": (
                "grayscale tracked_rgb + type-colored junctions + vertex disks"
            ),
            "t1_fourways.tif": (
                "same grayscale + one type-colored pixel per event frame"
            ),
        },
        "labels": str(args.labels.resolve()),
    }
    (tagged_path(out, stem, "qc.json")).write_text(json.dumps(qc, indent=2) + "\n")
    print(f"Wrote {ev_path}", flush=True)
    print(f"Wrote {fw_path}", flush=True)
    print(f"Wrote {len_path}", flush=True)
    # Second set of tables in µm and seconds, next to the pixel/frame ones.
    write_um_files(out, stem, cal)

    print(
        "Writing t1_fourways.tif (grayscale + type-colored event pixels)...",
        flush=True,
    )
    dots_path = tagged_path(out, stem, "t1_fourways.tif")
    save_tiff(dots_path, result["dots"], cal)
    print(f"Wrote {dots_path}", flush=True)

    if rgb is not None:
        print(
            "Writing t1_overlay.tif (grayscale + junctions + 4-way disks)...",
            flush=True,
        )
        ov_path = tagged_path(out, stem, "t1_overlay.tif")
        save_tiff(ov_path, result["overlay"], cal)
        print(f"Wrote {ov_path}", flush=True)

    write_picker_ready(
        t1_output_dir=out,
        stem=stem,
        events=ev_path,
        labels=args.labels.expanduser().resolve(),
        rgb=(
            args.rgb.expanduser().resolve()
            if args.rgb is not None
            else None
        ),
        fourways=dots_path,
        calibration=cal.as_dict() if cal else None,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
