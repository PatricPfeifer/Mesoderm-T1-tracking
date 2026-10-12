"""Division candidates (epithelial signature) and hypothesis testing.

A division is two *adjacent* current cells whose union overlaps the warped
mother. Each candidate is accepted only if it lowers the total assignment
cost of the frame pair by more than ``division_penalty`` (btrack / LAP-split
style), which keeps divisions rare.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cells import FrameCells
from .costs import PairCosts
from .motion import MotionField


def _dice(a: set[int], b: set[int]) -> float:
    total = len(a) + len(b)
    return (2 * len(a & b) / total) if total else 0.0


@dataclass
class DivisionCandidate:
    parent: int  # prev cell index (0-based)
    d1: int  # cur cell index
    d2: int  # cur cell index
    union_cost: float
    union_iou: float


def find_division_candidates(
    prev: FrameCells,
    cur: FrameCells,
    pair: PairCosts,
    field: MotionField,
    *,
    min_inside: float = 0.5,
    min_union_iou: float = 0.5,
    max_area_logratio: float = float(np.log(1.6)),
    min_size_ratio: float = 0.3,
    max_candidates: int = 20,
    blocked_parents: set[int] | None = None,
) -> list[DivisionCandidate]:
    """Adjacent cur-cell pairs that together look like the warped prev cell."""
    if pair.inter is None or pair.n_prev == 0 or pair.n_cur < 2:
        return []
    blocked_parents = blocked_parents or set()
    weights = pair.weights or {}
    w_iou = float(weights.get("w_iou", 1.0))
    w_dist = float(weights.get("w_dist", 0.5))
    w_area = float(weights.get("w_area", 0.25))
    max_distance = float(weights.get("max_distance", 50.0))

    inter = pair.inter
    a_cur = np.maximum(cur.areas, 1.0)
    inside = inter / a_cur[None, :]

    cy = prev.centroids[:, 0]
    cx = prev.centroids[:, 1]
    dy, dx = field.sample(cy, cx)
    pred = np.stack([cy + dy, cx + dx], axis=1)

    label_of = cur.original_labels
    idx_of = {int(label_of[j + 1]): j for j in range(cur.n_cells)}

    out: list[DivisionCandidate] = []
    for p in range(pair.n_prev):
        if p in blocked_parents:
            continue
        kids = np.nonzero(inside[p] >= min_inside)[0]
        if kids.size < 2:
            continue
        kid_set = set(kids.tolist())
        a_p = float(prev.areas[p])
        for c1 in kids.tolist():
            for n_lab in cur.adjacency.get(int(label_of[c1 + 1]), ()):
                c2 = idx_of.get(int(n_lab))
                if c2 is None or c2 <= c1 or c2 not in kid_set:
                    continue
                a1, a2 = float(cur.areas[c1]), float(cur.areas[c2])
                if min(a1, a2) / max(a1, a2, 1.0) < min_size_ratio:
                    continue
                area_term = abs(np.log(max(a_p, 1.0) / max(a1 + a2, 1.0)))
                if area_term > max_area_logratio:
                    continue
                i1, i2 = float(inter[p, c1]), float(inter[p, c2])
                union = a_p + a1 + a2 - i1 - i2
                u_iou = (i1 + i2) / union if union > 0 else 0.0
                if u_iou < min_union_iou:
                    continue
                cen = (
                    cur.centroids[c1] * a1 + cur.centroids[c2] * a2
                ) / (a1 + a2)
                dist = float(np.hypot(*(pred[p] - cen)))
                cost = (
                    w_iou * (1.0 - u_iou)
                    + w_dist * dist / max(max_distance, 1e-6)
                    + w_area * area_term
                )
                out.append(DivisionCandidate(p, c1, c2, float(cost), float(u_iou)))
    out.sort(key=lambda d: d.union_cost)
    return out[:max_candidates]


def union_topology_dice(
    cand: DivisionCandidate,
    prev: FrameCells,
    cur: FrameCells,
    matches: list[tuple[int, int]],
) -> float:
    """Dice of the mother's prev neighbours (mapped through ``matches``) vs
    the union's neighbours, in cur-label space."""
    p_lab = int(prev.original_labels[cand.parent + 1])
    l1 = int(cur.original_labels[cand.d1 + 1])
    l2 = int(cur.original_labels[cand.d2 + 1])
    prev_to_cur = {
        int(prev.original_labels[p + 1]): int(cur.original_labels[c + 1])
        for p, c in matches
    }
    mapped = {
        prev_to_cur[n]
        for n in prev.adjacency.get(p_lab, ())
        if n in prev_to_cur
    } - {l1, l2}
    union_neigh = (
        set(cur.adjacency.get(l1, ())) | set(cur.adjacency.get(l2, ()))
    ) - {l1, l2}
    return _dice(mapped, union_neigh)


def select_divisions(
    pair: PairCosts,
    candidates: list[DivisionCandidate],
    prev: FrameCells,
    cur: FrameCells,
    *,
    appear_cost: float,
    disappear_cost: float,
    division_penalty: float,
    min_union_dice: float = 0.5,
) -> list[DivisionCandidate]:
    """Greedy hypothesis test: accept a candidate only if it lowers the total
    assignment objective (with its union cost + penalty). Accepted ones are
    locked for later candidates."""
    from .assign import solve_lap

    if not candidates:
        return []
    matches, best = solve_lap(
        pair.cost, appear_cost=appear_cost, disappear_cost=disappear_cost
    )
    accepted: list[DivisionCandidate] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    extra = 0.0
    for cand in candidates:
        if cand.parent in used_rows or cand.d1 in used_cols or cand.d2 in used_cols:
            continue
        if union_topology_dice(cand, prev, cur, matches) < min_union_dice:
            continue
        rows = used_rows | {cand.parent}
        cols = used_cols | {cand.d1, cand.d2}
        add = cand.union_cost + division_penalty
        new_matches, obj = solve_lap(
            pair.cost,
            appear_cost=appear_cost,
            disappear_cost=disappear_cost,
            drop_rows=rows,
            drop_cols=cols,
        )
        if obj + extra + add < best - 1e-9:
            accepted.append(cand)
            used_rows, used_cols = rows, cols
            extra += add
            best = obj + extra
            matches = new_matches
    return accepted
