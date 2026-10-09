#!/usr/bin/env python3
"""Tissue Analyzer's cell tracking algorithm, extracted to run on Cellpose masks.

Source of the algorithm (pyTA / EPySeg, Benoit Aigouy):
  epyseg/ta/tracking/tracking_yet_another_approach_pyramidal_registration_n_neo_swapping_correction.py
      track_cells_dynamic_tissue()          -> main frame-to-frame loop
  epyseg/ta/tracking/registration.py
      pre_register_images(), get_pyramidal_registration()
  epyseg/ta/tracking/tracking_error_detector_and_fixer.py
      associate_cell_to_its_neighbors2(), compute_neighbor_score(),
      pairwise_score(), optimize_score()
  epyseg/ta/tracking/tools.py
      first_image_tracking(), assign_random_ID_to_missing_cells()
  epyseg/ta/measurements/.../point_on_surface()

Two stages per frame pair, exactly like TA:

  1. PROPAGATION. Estimate a dense (per-pixel) translation field between the raw
     images at t-1 and t by pyramidal phase cross-correlation. For every cell in
     t-1, take an interior point, shift it by the local translation, and read
     which cell sits there in t. That cell inherits the track ID. Cells in t that
     nobody claimed get fresh track IDs.

  2. NEIGHBOUR-GRAPH CORRECTION. Build the cell adjacency graph for both frames.
     A cell's score is the Dice overlap between its neighbour ID set at t and at
     t-1 (1.0 = neighbourhood fully preserved). Cells below a threshold are
     candidates for having been swapped or misassigned; TA proposes swaps /
     relabellings, applies them one at a time, and keeps a change only if the
     summed score over the whole frame goes up. Repeat until no improvement.

Differences from TA (needed because Cellpose labels are not TA handCorrection):
  * TA reads per-frame `handCorrection.tif` (0 = cell, 255 = 1 px bond) and
    relabels with `measure.label(..., background=255)`. Here we consume a Cellpose
    label stack directly, so no relabelling happens.
  * TA stores track IDs as 24-bit RGB colours in `tracked_cells_resized.tif`.
    Here track IDs are plain integers in a uint32 label stack (RGB is optional
    and only for viewing).
  * Cellpose labels are eroded and do not touch (~33 % background here), so the
    adjacency graph is built on labels grown by `--expand` px, and the propagated
    point gets a small `--search-radius` fallback. TA needed neither.

Input: a T,Y,X integer label stack (Cellpose `*_cp_masks.tif`), labels
independent per frame. Optionally the matching T,Y,X raw intensity stack, which
makes the registration stage much more reliable.

Example:
  conda activate TA
  python scripts/ta_tracking_extracted.py \
    --masks "output/cellpose_seg/masks/naima_cut_cp_masks.tif" \
    --intensity "input files/Cri_WT_GAP43_GFP_38sec_Naima_cut.tif" \
    --output-dir output/ta_track/extracted_naima_cut
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import tifffile
from skimage import transform
from skimage.measure import regionprops
from skimage.registration import phase_cross_correlation
from skimage.segmentation import expand_labels

# ---------------------------------------------------------------------------
# Stage 1a: registration (from epyseg/ta/tracking/registration.py)
# ---------------------------------------------------------------------------


def _phase_shift(ref: np.ndarray, mov: np.ndarray) -> tuple[float, float]:
    """Sub-pixel translation from ref to mov, tolerant of skimage API changes."""
    out = phase_cross_correlation(ref, mov)
    shift = out[0] if isinstance(out, tuple) else out
    return float(shift[0]), float(shift[1])


def apply_translation(img: np.ndarray, y: float, x: float) -> np.ndarray:
    tf = transform.AffineTransform(translation=(x, y))
    return transform.warp(img, inverse_map=tf, order=0, preserve_range=True)


def shifted_crop(
    img: np.ndarray, y: int, x: int, height: int, width: int, dy: int, dx: int
) -> np.ndarray:
    """img[y+dy : y+dy+height, x+dx : x+dx+width], zero-padded outside.

    Equivalent to cropping TA's nearest-neighbour full-image warp, without
    warping the pixels we then throw away.
    """
    out = np.zeros((height, width), dtype=img.dtype)
    src_y0, src_x0 = max(0, y + dy), max(0, x + dx)
    src_y1 = min(img.shape[0], y + dy + height)
    src_x1 = min(img.shape[1], x + dx + width)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out
    dst_y0, dst_x0 = max(0, -(y + dy)), max(0, -(x + dx))
    out[
        dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)
    ] = img[src_y0:src_y1, src_x0:src_x1]
    return out


def get_pyramidal_registration(
    orig_t0: np.ndarray,
    orig_t1: np.ndarray,
    depth: int = 3,
    threshold_translation: float = 20.0,
    block_size: int = 256,
    legacy_warp: bool = False,
) -> np.ndarray:
    """Dense H,W,2 translation field mapping positions in t0 onto t1.

    Starts from one global rigid shift, then refines it on a pyramid of shrinking
    tiles (256, 128, 64 px for depth=3). Per-tile shifts larger than
    `threshold_translation` are discarded as registration failures, so a bad tile
    simply keeps its parent's coarser shift.

    `legacy_warp=True` reproduces TA literally, warping the entire image once per
    tile (~170 full-image warps at depth 3, and the run's main cost).
    """
    translation_matrix = np.zeros((*orig_t0.shape, 2), dtype=np.float32)

    global_y, global_x = _phase_shift(orig_t0, orig_t1)
    translation_matrix[..., 0] += -global_y
    translation_matrix[..., 1] += -global_x

    height, width = orig_t0.shape[:2]

    for level in range(depth):
        tile = max(block_size // (2**level), 16)
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                tile_t1 = orig_t1[y : y + tile, x : x + tile]
                if tile_t1.size == 0:
                    continue

                if legacy_warp:
                    tf = transform.AffineTransform(
                        translation=(
                            -translation_matrix[y, x, 1],
                            -translation_matrix[y, x, 0],
                        )
                    )
                    warped = transform.warp(
                        orig_t0, inverse_map=tf, order=0, preserve_range=True
                    )
                    tile_t0 = warped[y : y + tile, x : x + tile]
                else:
                    tile_t0 = shifted_crop(
                        orig_t0,
                        y,
                        x,
                        tile_t1.shape[0],
                        tile_t1.shape[1],
                        int(round(-translation_matrix[y, x, 0])),
                        int(round(-translation_matrix[y, x, 1])),
                    )

                if tile_t0.shape != tile_t1.shape:
                    continue
                if tile_t0.std() == 0 or tile_t1.std() == 0:
                    continue

                dy, dx = _phase_shift(tile_t0, tile_t1)
                dy, dx = -dy, -dx
                if abs(dy) > threshold_translation or abs(dx) > threshold_translation:
                    continue

                translation_matrix[y : y + tile, x : x + tile, 0] += dy
                translation_matrix[y : y + tile, x : x + tile, 1] += dx

    return translation_matrix


# ---------------------------------------------------------------------------
# Stage 1b: interior representative point (TA point_on_surface)
# ---------------------------------------------------------------------------


def point_on_surface(region, labels: np.ndarray) -> tuple[int, int]:
    """A pixel guaranteed to lie inside the region.

    Prefers the centroid; for concave cells whose centroid falls outside, falls
    back to the middle of the longest horizontal run inside the cell.
    """
    cell_id = labels[region.coords[0][0], region.coords[0][1]]

    cy, cx = region.centroid
    cy, cx = int(round(cy)), int(round(cx))
    if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1]:
        if labels[cy, cx] == cell_id:
            return cy, cx

    min_row, min_col, max_row, max_col = region.bbox
    best_len = 0
    best = (int(region.coords[0][0]), int(region.coords[0][1]))
    sub = labels[min_row:max_row, min_col:max_col] == cell_id
    for row in range(sub.shape[0]):
        line = sub[row]
        start = None
        for col in range(sub.shape[1] + 1):
            inside = col < sub.shape[1] and line[col]
            if inside and start is None:
                start = col
            elif not inside and start is not None:
                run = col - start
                if run > best_len:
                    best_len = run
                    best = (min_row + row, min_col + start + run // 2)
                start = None
    return best


# ---------------------------------------------------------------------------
# Stage 2a: adjacency graph (TA associate_cell_to_its_neighbors2)
# ---------------------------------------------------------------------------


def build_adjacency(labels: np.ndarray, expand: int = 3) -> dict[int, set[int]]:
    """label -> set of touching labels.

    TA walks every bond pixel of the 1 px-wide watershed skeleton and pairs the
    labels it sees in a 3x3 window. Cellpose labels are shrunk and separated by
    background, so we first grow them by `expand` px to close the gaps, then read
    adjacency off shifted comparisons (same result, much faster).
    """
    work = expand_labels(labels, distance=expand) if expand > 0 else labels

    adj: dict[int, set[int]] = {int(v): set() for v in np.unique(labels) if v != 0}
    valid = set(adj)

    height, width = work.shape
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = work[: height - dy, max(0, -dx) : width - max(0, dx)]
        b = work[dy:, max(0, dx) : width + min(0, dx)]
        mask = (a > 0) & (b > 0) & (a != b)
        if not mask.any():
            continue
        for u, v in zip(a[mask].tolist(), b[mask].tolist()):
            if u in valid and v in valid:
                adj[u].add(v)
                adj[v].add(u)
    return adj


# ---------------------------------------------------------------------------
# Stage 2b: neighbourhood scoring (TA pairwise_score / compute_neighbor_score)
# ---------------------------------------------------------------------------


def pairwise_score(a: set[int], b: set[int]) -> float:
    """Dice overlap of two neighbour-ID sets: 2*|a n b| / (|a| + |b|)."""
    total = len(a) + len(b)
    if total == 0:
        return 0.0
    return (len(a & b) * 2) / total


def track_neighbourhoods(
    adj: dict[int, set[int]], label_to_track: dict[int, int]
) -> dict[int, set[int]]:
    """Adjacency expressed in track-ID space instead of local-label space."""
    return {
        label_to_track[lab]: {
            label_to_track[n] for n in neighbours if n in label_to_track
        }
        for lab, neighbours in adj.items()
        if lab in label_to_track
    }


def compute_neighbor_score(
    cur: dict[int, set[int]], prev: dict[int, set[int]]
) -> tuple[float, dict[int, float]]:
    """Summed and per-cell score of frame t against frame t-1.

    A track absent from t-1 scores 0, which is what marks new / misassigned cells
    as candidates for correction.
    """
    per_cell: dict[int, float] = {}
    total = 0.0
    for track, neighbours in cur.items():
        if track in prev:
            score = pairwise_score(neighbours, prev[track])
        else:
            score = 0.0
        per_cell[track] = score
        total += score
    return total, per_cell


# ---------------------------------------------------------------------------
# Stage 2c: swap / assignment correction (TA optimize_score)
# ---------------------------------------------------------------------------


def optimize_score(
    adj: dict[int, set[int]],
    label_to_track: dict[int, int],
    prev: dict[int, set[int]],
    threshold: float = 0.8,
    max_iter: int = 15,
    verbose: bool = True,
) -> tuple[dict[int, int], float, int]:
    """Greedily swap/relabel low-scoring cells while the frame total improves.

    Returns the corrected label -> track mapping, the final score and the number
    of accepted changes.
    """
    mapping = dict(label_to_track)
    total, _ = compute_neighbor_score(track_neighbourhoods(adj, mapping), prev)
    accepted = 0

    for iteration in range(max_iter):
        cur = track_neighbourhoods(adj, mapping)
        total, per_cell = compute_neighbor_score(cur, prev)
        track_to_label = {t: l for l, t in mapping.items()}

        low = sorted(
            (t for t, s in per_cell.items() if s <= threshold),
            key=lambda t: per_cell[t],
        )
        if not low:
            break

        # Propose one partner per low-scoring cell, as TA does: the cell whose
        # current neighbourhood best explains the reference neighbourhood.
        proposals: list[tuple[int, int]] = []
        for track in low:
            current = per_cell[track]
            if current >= 1.0:
                continue
            reference = prev.get(track) or cur.get(track)
            if not reference:
                continue
            best_partner = None
            best = current
            for other, neighbours in cur.items():
                if other == track:
                    continue
                score = pairwise_score(neighbours, reference)
                if score >= best:
                    best = score
                    best_partner = other
            if best_partner is not None:
                proposals.append((track, best_partner))

        # Tracks that vanished between t-1 and t: offer their ID to the current
        # cell that best matches their old neighbourhood (TA's "assignment").
        for track in set(prev) - set(cur):
            reference = prev[track]
            best_partner = None
            best = 0.0
            for other, neighbours in cur.items():
                if per_cell.get(other, 0.0) >= 1.0:
                    continue
                score = pairwise_score(neighbours, reference)
                if score >= max(best, per_cell.get(other, 0.0)):
                    best = score
                    best_partner = other
            if best_partner is not None:
                proposals.append((best_partner, track))

        applied_this_round = 0
        for track, partner in proposals:
            label_a = track_to_label.get(track)
            if label_a is None:
                continue
            label_b = track_to_label.get(partner)

            if label_b is None:
                # `partner` is unused in this frame: pure relabelling.
                mapping[label_a] = partner
            else:
                mapping[label_a], mapping[label_b] = partner, track

            new_total, _ = compute_neighbor_score(
                track_neighbourhoods(adj, mapping), prev
            )
            if new_total <= total:
                if label_b is None:
                    mapping[label_a] = track
                else:
                    mapping[label_a], mapping[label_b] = track, partner
                continue

            total = new_total
            accepted += 1
            applied_this_round += 1
            track_to_label = {t: l for l, t in mapping.items()}

        if verbose:
            print(
                f"    iter {iteration}: score={total:.2f} "
                f"candidates={len(low)} accepted={applied_this_round}",
                flush=True,
            )
        if applied_this_round == 0:
            break

    return mapping, total, accepted


# ---------------------------------------------------------------------------
# Stage 1c: translation-driven ID propagation (TA main loop body)
# ---------------------------------------------------------------------------


def propagate_ids(
    labels_prev: np.ndarray,
    labels_cur: np.ndarray,
    prev_label_to_track: dict[int, int],
    translation_matrix: np.ndarray,
    search_radius: int = 3,
) -> dict[int, int]:
    """Carry track IDs from t-1 into t through the translation field.

    For each cell of t-1 we shift its interior point by the local translation and
    claim whichever cell of t is hit. First claim wins (TA's behaviour): a second
    cell landing on an already-claimed target is dropped and will be treated as a
    new cell instead.
    """
    height, width = labels_cur.shape
    mapping: dict[int, int] = {}
    claimed: set[int] = set()

    for region in regionprops(labels_prev):
        label_prev = int(labels_prev[region.coords[0][0], region.coords[0][1]])
        track = prev_label_to_track.get(label_prev)
        if track is None:
            continue

        cy, cx = point_on_surface(region, labels_prev)
        ty = int(cy + translation_matrix[cy, cx, 0])
        tx = int(cx + translation_matrix[cy, cx, 1])
        if not (0 <= ty < height and 0 <= tx < width):
            continue

        target = int(labels_cur[ty, tx])
        if target == 0 and search_radius > 0:
            # Cellpose leaves background between cells; look just around.
            y0, y1 = max(0, ty - search_radius), min(height, ty + search_radius + 1)
            x0, x1 = max(0, tx - search_radius), min(width, tx + search_radius + 1)
            patch = labels_cur[y0:y1, x0:x1]
            nz = patch[patch > 0]
            if nz.size:
                values, counts = np.unique(nz, return_counts=True)
                target = int(values[counts.argmax()])

        if target == 0 or target in claimed:
            continue

        mapping[target] = track
        claimed.add(target)

    return mapping


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_stack(path: Path) -> np.ndarray:
    img = np.asarray(tifffile.imread(path))
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    if img.ndim == 4:
        # T,Y,X,C or T,C,Y,X -> keep first channel
        if img.shape[-1] <= 4:
            img = img[..., 0]
        else:
            img = img[:, 0]
    if img.ndim != 3:
        raise ValueError(f"Expected a T,Y,X stack; got shape {img.shape} for {path}")
    return img


def registration_images(
    masks: np.ndarray, intensity: np.ndarray | None
) -> np.ndarray:
    """What the pyramidal registration is run on.

    TA registers the raw membrane signal. Without an intensity stack we fall back
    to the binary cell mask, which still has enough structure for phase
    correlation but is noticeably less precise.
    """
    if intensity is not None:
        if intensity.shape != masks.shape:
            raise ValueError(
                f"Intensity stack {intensity.shape} does not match masks "
                f"{masks.shape}."
            )
        return intensity.astype(np.float32)
    print(
        "No --intensity given: registering on the binary masks "
        "(less accurate than TA, which uses the raw signal).",
        flush=True,
    )
    return (masks > 0).astype(np.float32)


def track_colors(max_track: int, seed: int = 1) -> np.ndarray:
    """Stable random RGB per track ID, mimicking TA's colour-coded output."""
    rng = np.random.default_rng(seed)
    lut = rng.integers(40, 256, size=(max_track + 1, 3), dtype=np.uint8)
    lut[0] = 0
    return lut


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def track_stack(
    masks: np.ndarray,
    reg_images: np.ndarray,
    *,
    expand: int,
    depth: int,
    threshold_translation: float,
    score_threshold: float,
    max_iter: int,
    search_radius: int,
    fix_swaps: bool,
    legacy_warp: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    n_frames = masks.shape[0]
    tracked = np.zeros(masks.shape, dtype=np.uint32)
    rows: list[dict] = []

    # First frame: every cell gets a brand new track ID (TA first_image_tracking).
    labels_0 = masks[0]
    ids_0 = [int(v) for v in np.unique(labels_0) if v != 0]
    label_to_track = {lab: i + 1 for i, lab in enumerate(ids_0)}
    next_track_id = len(ids_0) + 1

    prev_neigh = track_neighbourhoods(
        build_adjacency(labels_0, expand=expand), label_to_track
    )
    tracked[0] = relabel(labels_0, label_to_track)
    rows.extend(frame_rows(0, labels_0, label_to_track, 1.0))
    print(f"frame 0: {len(ids_0)} cells, new tracks 1..{len(ids_0)}", flush=True)

    for t in range(1, n_frames):
        start = timer()
        labels_prev, labels_cur = masks[t - 1], masks[t]

        translation_matrix = get_pyramidal_registration(
            reg_images[t - 1],
            reg_images[t],
            depth=depth,
            threshold_translation=threshold_translation,
            legacy_warp=legacy_warp,
        )

        mapping = propagate_ids(
            labels_prev,
            labels_cur,
            label_to_track,
            translation_matrix,
            search_radius=search_radius,
        )

        cur_labels = [int(v) for v in np.unique(labels_cur) if v != 0]
        matched = len(mapping)
        for lab in cur_labels:
            if lab not in mapping:
                mapping[lab] = next_track_id
                next_track_id += 1

        adj_cur = build_adjacency(labels_cur, expand=expand)
        total, per_cell = compute_neighbor_score(
            track_neighbourhoods(adj_cur, mapping), prev_neigh
        )
        print(
            f"frame {t}: {len(cur_labels)} cells, propagated {matched}, "
            f"new {len(cur_labels) - matched}, score {total:.2f}",
            flush=True,
        )

        accepted = 0
        if fix_swaps:
            mapping, total, accepted = optimize_score(
                adj_cur,
                mapping,
                prev_neigh,
                threshold=score_threshold,
                max_iter=max_iter,
            )
            _, per_cell = compute_neighbor_score(
                track_neighbourhoods(adj_cur, mapping), prev_neigh
            )
            # Relabelling can free up an ID and duplicate another; make unique.
            mapping, next_track_id = deduplicate(mapping, next_track_id)

        tracked[t] = relabel(labels_cur, mapping)
        rows.extend(frame_rows(t, labels_cur, mapping, per_cell))

        label_to_track = mapping
        prev_neigh = track_neighbourhoods(adj_cur, mapping)
        print(
            f"  corrected score {total:.2f}, swaps accepted {accepted}, "
            f"{timer() - start:.1f}s",
            flush=True,
        )

    return tracked, rows


def deduplicate(
    mapping: dict[int, int], next_track_id: int
) -> tuple[dict[int, int], int]:
    seen: dict[int, int] = {}
    out = dict(mapping)
    for label, track in mapping.items():
        if track in seen:
            out[label] = next_track_id
            next_track_id += 1
        else:
            seen[track] = label
    return out, next_track_id


def relabel(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    lut = np.zeros(int(labels.max()) + 1, dtype=np.uint32)
    for label, track in mapping.items():
        if label < lut.size:
            lut[label] = track
    return lut[labels]


def frame_rows(
    frame: int,
    labels: np.ndarray,
    mapping: dict[int, int],
    scores,
) -> list[dict]:
    rows = []
    for region in regionprops(labels):
        label = int(labels[region.coords[0][0], region.coords[0][1]])
        track = mapping.get(label)
        if track is None:
            continue
        score = scores if isinstance(scores, float) else scores.get(track, 0.0)
        rows.append(
            {
                "frame": frame,
                "track_id": track,
                "local_label": label,
                "area": int(region.area),
                "centroid_y": round(float(region.centroid[0]), 2),
                "centroid_x": round(float(region.centroid[1]), 2),
                "neighbour_score": round(float(score), 4),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Tissue Analyzer tracking (pyramidal registration + neighbour-graph "
            "swap correction) applied to a Cellpose label stack."
        )
    )
    p.add_argument(
        "--masks",
        type=Path,
        required=True,
        help="T,Y,X integer label stack, e.g. Cellpose *_cp_masks.tif.",
    )
    p.add_argument(
        "--intensity",
        type=Path,
        default=None,
        help="Matching T,Y,X raw stack used for registration (strongly advised).",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Only track the first N frames (quick test).",
    )
    p.add_argument(
        "--expand",
        type=int,
        default=3,
        help="Grow labels by N px to close Cellpose gaps before building the "
        "adjacency graph (default: 3; use 0 for watershed-style masks).",
    )
    p.add_argument(
        "--pyramidal-depth",
        type=int,
        default=3,
        help="Registration pyramid levels, tiles 256/128/64 px (TA default: 3).",
    )
    p.add_argument(
        "--max-translation",
        type=float,
        default=20.0,
        help="Reject per-tile shifts above this many px (TA default: 20).",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.8,
        help="Cells with a neighbourhood score at or below this are candidates "
        "for swap correction (TA default: 0.8).",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=15,
        help="Max correction rounds per frame (TA default: 15).",
    )
    p.add_argument(
        "--search-radius",
        type=int,
        default=3,
        help="If the propagated point lands on background, take the majority "
        "label within this radius (default: 3; 0 disables).",
    )
    p.add_argument(
        "--legacy-warp",
        action="store_true",
        help="Use TA's original full-image warp per registration tile "
        "(much slower, kept to verify equivalence).",
    )
    p.add_argument(
        "--no-swap-correction",
        action="store_true",
        help="Skip stage 2 and keep the raw registration-based propagation.",
    )
    p.add_argument(
        "--rgb",
        action="store_true",
        help="Also write a TA-style random-colour RGB stack for viewing.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    masks_path = args.masks.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading masks {masks_path}", flush=True)
    masks = load_stack(masks_path)
    intensity = None
    if args.intensity is not None:
        intensity_path = args.intensity.expanduser().resolve()
        print(f"Loading intensity {intensity_path}", flush=True)
        intensity = load_stack(intensity_path)

    if args.frames is not None:
        masks = masks[: args.frames]
        if intensity is not None:
            intensity = intensity[: args.frames]

    if masks.shape[0] < 2:
        raise ValueError("Need at least 2 frames to track.")
    print(
        f"masks shape={masks.shape} dtype={masks.dtype} "
        f"cells/frame~{int(np.unique(masks[0]).size - 1)}",
        flush=True,
    )

    reg_images = registration_images(masks, intensity)

    start = timer()
    tracked, rows = track_stack(
        masks,
        reg_images,
        expand=args.expand,
        depth=args.pyramidal_depth,
        threshold_translation=args.max_translation,
        score_threshold=args.score_threshold,
        max_iter=args.max_iter,
        search_radius=args.search_radius,
        fix_swaps=not args.no_swap_correction,
        legacy_warp=args.legacy_warp,
    )

    labels_out = output_dir / "tracked_labels.tif"
    tifffile.imwrite(labels_out, tracked)

    csv_out = output_dir / "tracks.csv"
    with csv_out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "track_id",
                "local_label",
                "area",
                "centroid_y",
                "centroid_x",
                "neighbour_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if args.rgb:
        lut = track_colors(int(tracked.max()))
        rgb_out = output_dir / "tracked_rgb.tif"
        tifffile.imwrite(rgb_out, lut[tracked])
        print(f"RGB stack:    {rgb_out}", flush=True)

    n_tracks = int(tracked.max())
    lengths: dict[int, int] = {}
    for row in rows:
        lengths[row["track_id"]] = lengths.get(row["track_id"], 0) + 1
    complete = sum(1 for v in lengths.values() if v == masks.shape[0])
    print(
        f"Done in {timer() - start:.1f}s. frames={masks.shape[0]} "
        f"tracks={n_tracks} full-length_tracks={complete}",
        flush=True,
    )
    print(f"Label stack:  {labels_out}", flush=True)
    print(f"Track table:  {csv_out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
