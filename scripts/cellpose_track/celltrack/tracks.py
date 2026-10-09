"""Link table, track assembly, and gap closing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .cells import FrameCells, track_neighbourhoods
from .events import Event, classify_events
from .motion import MotionField
from .topology import pairwise_score, topology_refined_assignment


@dataclass
class Link:
    frame_from: int  # original on-disk index
    label_from: int
    frame_to: int
    label_to: int
    kind: str
    cost: float | None = None
    topology_score: float | None = None
    track_id: int | None = None


@dataclass
class TrackingResult:
    """Full movie tracking output (pre-render)."""

    # Per movie-index frame: original_label -> track_id
    label_to_track: list[dict[int, int]]
    links: list[Link]
    events: list[Event]
    per_frame_scores: list[float]
    frames: list[FrameCells]
    n_tracks: int


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
        return ra


def track_movie(
    frames: list[FrameCells],
    fields: list[MotionField],
    *,
    w_topo: float = 0.75,
    max_iter: int = 3,
    appear_cost: float = 1.5,
    disappear_cost: float = 1.5,
    division_penalty: float = 0.8,
    max_gap: int = 3,
    gap_close: bool = True,
    cost_kwargs: dict | None = None,
    verbose: bool = True,
) -> TrackingResult:
    """Run forward tracking across all consecutive frame pairs."""
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames")
    if len(fields) != len(frames) - 1:
        raise ValueError(
            f"Expected {len(frames) - 1} motion fields, got {len(fields)}"
        )

    # Frame 0: every cell gets a fresh track.
    label_to_track: list[dict[int, int]] = []
    t0 = {
        int(frames[0].original_labels[i + 1]): i + 1
        for i in range(frames[0].n_cells)
    }
    label_to_track.append(t0)
    next_track_id = frames[0].n_cells + 1

    links: list[Link] = []
    events: list[Event] = []
    scores: list[float] = [1.0]  # frame 0 has perfect self-score

    for t in range(1, len(frames)):
        prev, cur, field = frames[t - 1], frames[t], fields[t - 1]
        assignment, cur_map, next_track_id, score, per_score = (
            topology_refined_assignment(
                prev,
                cur,
                field,
                label_to_track[t - 1],
                next_track_id,
                w_topo=w_topo,
                max_iter=max_iter,
                appear_cost=appear_cost,
                disappear_cost=disappear_cost,
                division_penalty=division_penalty,
                cost_kwargs=cost_kwargs,
            )
        )
        label_to_track.append(cur_map)
        scores.append(score / max(cur.n_cells, 1))

        fe = classify_events(
            assignment,
            prev,
            cur,
            label_to_track[t - 1],
            cur_map,
            per_track_score=per_score,
        )
        events.extend(fe)

        for e in fe:
            if e.kind in ("continue", "divide") and e.label_from and e.label_to:
                links.append(
                    Link(
                        frame_from=e.frame_from,
                        label_from=e.label_from,
                        frame_to=e.frame_to,
                        label_to=e.label_to,
                        kind=e.kind,
                        cost=e.cost,
                        topology_score=e.topology_score,
                        track_id=e.track_id,
                    )
                )

        if verbose:
            n_div = sum(1 for e in fe if e.kind == "divide")
            n_app = sum(1 for e in fe if e.kind == "appear")
            n_dis = sum(1 for e in fe if e.kind == "disappear")
            n_cont = sum(1 for e in fe if e.kind == "continue")
            print(
                f"frame {prev.original_frame}->{cur.original_frame}: "
                f"cells={cur.n_cells} cont={n_cont} div={n_div} "
                f"appear={n_app} disappear={n_dis} "
                f"topo={scores[-1]:.3f}",
                flush=True,
            )

    if gap_close and max_gap > 0:
        label_to_track, links, events, next_track_id = close_gaps(
            frames,
            fields,
            label_to_track,
            links,
            events,
            next_track_id,
            max_gap=max_gap,
            appear_cost=appear_cost,
            verbose=verbose,
        )

    return TrackingResult(
        label_to_track=label_to_track,
        links=links,
        events=events,
        per_frame_scores=scores,
        frames=frames,
        n_tracks=next_track_id - 1,
    )


def close_gaps(
    frames: list[FrameCells],
    fields: list[MotionField],
    label_to_track: list[dict[int, int]],
    links: list[Link],
    events: list[Event],
    next_track_id: int,
    *,
    max_gap: int = 3,
    appear_cost: float = 1.5,
    max_distance: float = 40.0,
    min_dice: float = 0.3,
    verbose: bool = True,
) -> tuple[list[dict[int, int]], list[Link], list[Event], int]:
    """Reconnect tracks that disappear and reappear within max_gap frames.

    For each gap size k=1..max_gap, find tracks ending at t and tracks starting
    at t+k+1 (i.e. appeared at that frame with no prior history), score by
    warped centroid distance + neighbour Dice, solve LAP, merge track IDs.
    """
    # Build track lifespan: track_id -> (start_frame_idx, end_frame_idx, ...)
    track_start: dict[int, int] = {}
    track_end: dict[int, int] = {}
    track_start_label: dict[int, int] = {}
    track_end_label: dict[int, int] = {}
    track_end_centroid: dict[int, np.ndarray] = {}
    track_start_centroid: dict[int, np.ndarray] = {}
    track_end_neigh: dict[int, set[int]] = {}
    track_start_neigh: dict[int, set[int]] = {}

    for t, mapping in enumerate(label_to_track):
        neigh = track_neighbourhoods(frames[t].adjacency, mapping)
        for lab, track in mapping.items():
            idx = frames[t].index_of(lab)
            cent = (
                frames[t].centroids[idx]
                if idx is not None
                else np.zeros(2)
            )
            if track not in track_start:
                track_start[track] = t
                track_start_label[track] = lab
                track_start_centroid[track] = cent
                track_start_neigh[track] = neigh.get(track, set())
            track_end[track] = t
            track_end_label[track] = lab
            track_end_centroid[track] = cent
            track_end_neigh[track] = neigh.get(track, set())

    uf = UnionFind()
    for track in track_start:
        uf.find(track)

    n_closed = 0
    for k in range(1, max_gap + 1):
        # Ends at t, starts at t+k+1 means gap of k intermediate frames.
        # Movie index: end at t, start at t + k + 1.
        for t in range(len(frames) - k - 1):
            t_end = t
            t_start = t + k + 1
            endings = [
                tr
                for tr, end in track_end.items()
                if end == t_end and track_start[tr] <= t_end
            ]
            beginnings = [
                tr
                for tr, start in track_start.items()
                if start == t_start
            ]
            # Only close tracks that currently look terminated / newborn.
            endings = [
                tr
                for tr in endings
                if track_end[tr] == t_end
                and uf.find(tr) == tr  # canonical
            ]
            beginnings = [
                tr
                for tr in beginnings
                if track_start[tr] == t_start and uf.find(tr) == tr
            ]
            if not endings or not beginnings:
                continue

            # Accumulate motion across the gap.
            # Approximate: sum successive fields' samples at ending centroid.
            n_e, n_b = len(endings), len(beginnings)
            cost = np.full((n_e, n_b), np.inf, dtype=np.float64)
            for i, tr_e in enumerate(endings):
                cy, cx = track_end_centroid[tr_e]
                pred_y, pred_x = float(cy), float(cx)
                for g in range(k + 1):
                    field = fields[t_end + g]
                    dy, dx = field.sample(
                        np.array([pred_y]), np.array([pred_x])
                    )
                    pred_y += float(dy[0])
                    pred_x += float(dx[0])
                for j, tr_b in enumerate(beginnings):
                    by, bx = track_start_centroid[tr_b]
                    dist = float(
                        np.hypot(pred_y - by, pred_x - bx)
                    )
                    dice = pairwise_score(
                        track_end_neigh.get(tr_e, set()),
                        track_start_neigh.get(tr_b, set()),
                    )
                    if dist > max_distance and dice < min_dice:
                        continue
                    cost[i, j] = dist / max_distance + 1.0 - dice

            # Pad for non-matches.
            n = n_e + n_b
            big = np.full((n, n), 0.0)
            large = appear_cost + 5.0
            finite = np.where(np.isfinite(cost), cost, large)
            big[:n_e, :n_b] = finite
            big[:n_e, n_b:] = large
            np.fill_diagonal(big[:n_e, n_b : n_b + n_e], appear_cost)
            big[n_e:, :n_b] = large
            np.fill_diagonal(big[n_e : n_e + n_b, :n_b], appear_cost)

            row_ind, col_ind = linear_sum_assignment(big)
            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                if r >= n_e or c >= n_b:
                    continue
                if not np.isfinite(cost[r, c]):
                    continue
                if cost[r, c] >= appear_cost:
                    continue
                tr_e, tr_b = endings[r], beginnings[c]
                if uf.find(tr_e) == uf.find(tr_b):
                    continue
                uf.union(tr_e, tr_b)
                n_closed += 1
                links.append(
                    Link(
                        frame_from=frames[t_end].original_frame,
                        label_from=track_end_label[tr_e],
                        frame_to=frames[t_start].original_frame,
                        label_to=track_start_label[tr_b],
                        kind="gap_close",
                        cost=float(cost[r, c]),
                        track_id=tr_e,
                    )
                )
                events.append(
                    Event(
                        kind="continue",
                        frame_from=frames[t_end].original_frame,
                        frame_to=frames[t_start].original_frame,
                        label_from=track_end_label[tr_e],
                        label_to=track_start_label[tr_b],
                        track_id=tr_e,
                        note=f"gap_close k={k}",
                    )
                )

    if n_closed:
        # Remap all label_to_track through union-find.
        remapped: list[dict[int, int]] = []
        canon_ids: dict[int, int] = {}
        new_id = 1
        for mapping in label_to_track:
            new_map: dict[int, int] = {}
            for lab, track in mapping.items():
                root = uf.find(track)
                if root not in canon_ids:
                    canon_ids[root] = new_id
                    new_id += 1
                new_map[lab] = canon_ids[root]
            remapped.append(new_map)
        label_to_track = remapped
        # Update link track ids.
        for link in links:
            if link.track_id is not None and link.track_id in uf.parent:
                root = uf.find(link.track_id)
                link.track_id = canon_ids.get(root, link.track_id)
        next_track_id = new_id

    if verbose:
        print(f"Gap closing: merged {n_closed} track pairs", flush=True)

    return label_to_track, links, events, next_track_id


def render_tracked_labels(
    masks: np.ndarray,
    frames: list[FrameCells],
    label_to_track: list[dict[int, int]],
) -> np.ndarray:
    """Build a T,Y,X uint32 stack with consistent track IDs."""
    out = np.zeros(masks.shape, dtype=np.uint32)
    for t, fc in enumerate(frames):
        mapping = label_to_track[t]
        plane = masks[t]
        # Vectorised via LUT on original labels.
        max_lab = int(plane.max()) if plane.size else 0
        lut = np.zeros(max_lab + 1, dtype=np.uint32)
        for lab, track in mapping.items():
            if 0 < lab <= max_lab:
                lut[lab] = track
        out[t] = lut[plane]
    return out


def render_tracked_rgb(tracked: np.ndarray, seed: int = 1) -> np.ndarray:
    """Stable random RGB per track ID for viewing identity over time.

    Same track keeps the same colour across frames; background stays black.
    """
    max_id = int(tracked.max()) if tracked.size else 0
    rng = np.random.default_rng(seed)
    lut = rng.integers(40, 256, size=(max_id + 1, 3), dtype=np.uint8)
    lut[0] = 0
    return lut[tracked]
