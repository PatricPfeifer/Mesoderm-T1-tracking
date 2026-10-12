"""Neighbour-set topology scoring folded into assignment costs."""

from __future__ import annotations

import numpy as np

from .assign import Assignment, solve_assignment
from .cells import FrameCells, track_neighbourhoods
from .costs import PairCosts, build_pair_costs
from .motion import MotionField


def pairwise_score(a: set[int], b: set[int]) -> float:
    """Dice overlap of two neighbour-ID sets (TA pairwise_score)."""
    total = len(a) + len(b)
    if total == 0:
        return 0.0
    return (len(a & b) * 2) / total


def compute_neighbor_score(
    cur: dict[int, set[int]], prev: dict[int, set[int]]
) -> tuple[float, dict[int, float]]:
    """Summed and per-track neighbour score of frame t against t-1."""
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


def _tracks_from_assignment(
    assignment: Assignment,
    prev: FrameCells,
    cur: FrameCells,
    prev_label_to_track: dict[int, int],
    next_track_id: int,
    parents: dict[int, int] | None = None,
) -> tuple[dict[int, int], int, dict[int, int]]:
    """Build cur_label -> track_id from an Assignment.

    Returns (cur_label_to_track, next_track_id, prev_idx_to_track).
    For legacy divisions the parent keeps its track on the higher-IoU /
    lower-cost daughter; the other daughter gets a fresh track ID.
    For ``split_divisions`` the parent track ends and both daughters get
    fresh IDs; ``parents`` (if given) is filled with daughter -> parent track.
    """
    cur_map: dict[int, int] = {}
    prev_idx_to_track = {
        prev.index_of(lab): track
        for lab, track in prev_label_to_track.items()
        if prev.index_of(lab) is not None
    }

    # Group matches by parent.
    by_parent: dict[int, list[tuple[int, float]]] = {}
    for p, c in assignment.matches:
        cost = assignment.costs.get((p, c), 0.0)
        by_parent.setdefault(p, []).append((c, cost))

    for p, kids in by_parent.items():
        kids_sorted = sorted(kids, key=lambda x: x[1])  # lowest cost first
        parent_track = prev_idx_to_track.get(p)
        if parent_track is None:
            parent_track = next_track_id
            next_track_id += 1
        # First (best) daughter inherits parent track.
        best_c = kids_sorted[0][0]
        cur_lab = int(cur.original_labels[best_c + 1])
        cur_map[cur_lab] = parent_track
        for c, _ in kids_sorted[1:]:
            cur_lab = int(cur.original_labels[c + 1])
            cur_map[cur_lab] = next_track_id
            next_track_id += 1

    for p, d1, d2, _cost in assignment.split_divisions:
        parent_track = prev_idx_to_track.get(p)
        for c in (d1, d2):
            cur_lab = int(cur.original_labels[c + 1])
            cur_map[cur_lab] = next_track_id
            if parents is not None and parent_track is not None:
                parents[next_track_id] = parent_track
            next_track_id += 1

    for j in assignment.appeared:
        cur_lab = int(cur.original_labels[j + 1])
        cur_map[cur_lab] = next_track_id
        next_track_id += 1

    return cur_map, next_track_id, prev_idx_to_track


def topology_refined_assignment(
    prev: FrameCells,
    cur: FrameCells,
    field: MotionField,
    prev_label_to_track: dict[int, int],
    next_track_id: int,
    *,
    w_topo: float = 0.75,
    max_iter: int = 3,
    appear_cost: float = 1.5,
    disappear_cost: float = 1.5,
    division_penalty: float = 0.8,
    cost_kwargs: dict | None = None,
    division_mode: str = "legacy",
    blocked_parents: set[int] | None = None,
    division_kwargs: dict | None = None,
    parents: dict[int, int] | None = None,
) -> tuple[Assignment, dict[int, int], int, float, dict[int, float]]:
    """Solve LAP, fold neighbour Dice into costs, re-solve up to max_iter.

    Accepts a new solution only if the summed topology score improved
    (TA's good habit). Stops early when score does not improve.

    ``division_mode="lineage"``: divisions are detected once on the base
    costs (see divisions.py), locked for every refinement solve, and both
    daughters get new track IDs (``parents`` receives daughter -> parent).
    ``blocked_parents`` are prev cell indices that may not divide.
    """
    cost_kwargs = dict(cost_kwargs or {})
    pair = build_pair_costs(prev, cur, field, **cost_kwargs)
    lineage = division_mode == "lineage"
    locked: list[tuple[int, int, int, float]] = []
    if lineage:
        from .divisions import find_division_candidates, select_divisions

        dkw = dict(division_kwargs or {})
        min_union_dice = float(dkw.pop("min_union_dice", 0.5))
        candidates = find_division_candidates(
            prev, cur, pair, field, blocked_parents=blocked_parents, **dkw
        )
        accepted = select_divisions(
            pair,
            candidates,
            prev,
            cur,
            appear_cost=appear_cost,
            disappear_cost=disappear_cost,
            division_penalty=division_penalty,
            min_union_dice=min_union_dice,
        )
        locked = [(d.parent, d.d1, d.d2, d.union_cost) for d in accepted]
    locked_rows = {d[0] for d in locked}
    locked_cols = {d[1] for d in locked} | {d[2] for d in locked}

    def solve(pc: PairCosts) -> Assignment:
        if lineage:
            return solve_assignment(
                pc,
                appear_cost=appear_cost,
                disappear_cost=disappear_cost,
                allow_division=False,
                locked_divisions=locked,
            )
        return solve_assignment(
            pc,
            appear_cost=appear_cost,
            disappear_cost=disappear_cost,
            division_penalty=division_penalty,
        )

    assignment = solve(pair)

    prev_neigh = track_neighbourhoods(prev.adjacency, prev_label_to_track)

    best_parents: dict[int, int] = {}
    cur_map, next_id, _ = _tracks_from_assignment(
        assignment, prev, cur, prev_label_to_track, next_track_id, best_parents
    )
    cur_neigh = track_neighbourhoods(cur.adjacency, cur_map)
    best_score, best_per = compute_neighbor_score(cur_neigh, prev_neigh)
    best_assignment = assignment
    best_map = cur_map
    best_next = next_id

    for iteration in range(max_iter):
        # Build topology-augmented cost: add w_topo * (1 - dice) for each
        # candidate pair, where dice is scored as if that pair continued the
        # parent's track into the daughter.
        aug = pair.cost.copy()
        for i in range(pair.n_prev):
            if i in locked_rows:
                continue
            prev_lab = int(prev.original_labels[i + 1])
            parent_track = prev_label_to_track.get(prev_lab)
            if parent_track is None:
                continue
            parent_prev_neigh = prev_neigh.get(parent_track, set())
            for j in range(pair.n_cur):
                if j in locked_cols or not np.isfinite(aug[i, j]):
                    continue
                cur_lab = int(cur.original_labels[j + 1])
                # Neighbours of cur_lab expressed as tracks under current map,
                # but substituting parent_track for cur_lab itself.
                tentative = dict(best_map)
                tentative[cur_lab] = parent_track
                cur_lab_neigh_tracks = {
                    tentative[n]
                    for n in cur.adjacency.get(cur_lab, set())
                    if n in tentative
                }
                dice = pairwise_score(cur_lab_neigh_tracks, parent_prev_neigh)
                aug[i, j] = float(pair.cost[i, j]) + w_topo * (1.0 - dice)

        aug_pair = PairCosts(
            cost=aug,
            n_prev=pair.n_prev,
            n_cur=pair.n_cur,
        )
        new_assignment = solve(aug_pair)
        new_parents: dict[int, int] = {}
        new_map, new_next, _ = _tracks_from_assignment(
            new_assignment, prev, cur, prev_label_to_track, next_track_id,
            new_parents,
        )
        new_neigh = track_neighbourhoods(cur.adjacency, new_map)
        new_score, new_per = compute_neighbor_score(new_neigh, prev_neigh)

        if new_score <= best_score + 1e-9:
            # No improvement — early stop (fixes TA's equality-only break).
            break

        best_score = new_score
        best_per = new_per
        best_assignment = new_assignment
        best_map = new_map
        best_next = new_next
        best_parents = new_parents

    if parents is not None:
        parents.update(best_parents)
    return best_assignment, best_map, best_next, best_score, best_per
