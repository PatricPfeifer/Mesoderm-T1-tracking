"""Global linear assignment with appear / disappear / division."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .costs import PairCosts


@dataclass
class Assignment:
    """Result of one frame-pair LAP.

    ``matches``: list of (prev_idx, cur_idx) 0-based cell indices.
    ``disappeared``: prev indices with no match.
    ``appeared``: cur indices with no match.
    ``divisions``: list of (prev_idx, (cur_a, cur_b)) where prev maps to two curs.
    """

    matches: list[tuple[int, int]] = field(default_factory=list)
    disappeared: list[int] = field(default_factory=list)
    appeared: list[int] = field(default_factory=list)
    divisions: list[tuple[int, tuple[int, int]]] = field(default_factory=list)
    costs: dict[tuple[int, int], float] = field(default_factory=dict)
    # Lineage-mode divisions: (prev_idx, cur_a, cur_b, union_cost). The parent
    # track ends and both daughters start new tracks (not in ``matches``).
    split_divisions: list[tuple[int, int, int, float]] = field(default_factory=list)


def solve_lap(
    cost: np.ndarray,
    *,
    appear_cost: float,
    disappear_cost: float,
    drop_rows: set[int] | frozenset[int] = frozenset(),
    drop_cols: set[int] | frozenset[int] = frozenset(),
) -> tuple[list[tuple[int, int]], float]:
    """Padded appear/disappear LAP on ``cost`` (inf = not allowed).

    Rows / cols in ``drop_*`` are left out. Returns (matches, objective) where
    matches are (row, col) in the original indexing and objective is the sum
    of match costs plus appear/disappear costs of unmatched kept cells.
    """
    n_prev_all, n_cur_all = cost.shape
    rows = [i for i in range(n_prev_all) if i not in drop_rows]
    cols = [j for j in range(n_cur_all) if j not in drop_cols]
    n_prev, n_cur = len(rows), len(cols)
    if n_prev == 0 or n_cur == 0:
        return [], n_prev * disappear_cost + n_cur * appear_cost
    sub = cost[np.ix_(rows, cols)]

    # Padded square matrix of size (n_prev + n_cur):
    #   [ real (n_prev x n_cur) | diag(disappear) ]
    #   [ diag(appear)          | zeros           ]
    n = n_prev + n_cur
    big = np.full((n, n), 0.0, dtype=np.float64)
    # Replace inf with a large finite value so LAP still works, then filter.
    large = appear_cost + disappear_cost + 10.0
    big[:n_prev, :n_cur] = np.where(np.isfinite(sub), sub, large)
    big[:n_prev, n_cur:] = large
    np.fill_diagonal(big[:n_prev, n_cur : n_cur + n_prev], disappear_cost)
    big[n_prev:, :n_cur] = large
    np.fill_diagonal(big[n_prev : n_prev + n_cur, :n_cur], appear_cost)

    row_ind, col_ind = linear_sum_assignment(big)
    matches: list[tuple[int, int]] = []
    objective = 0.0
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if r < n_prev and c < n_cur:
            value = float(sub[r, c])
            if not np.isfinite(value):
                # Forced match via large filler — treat as unmatched.
                continue
            matches.append((rows[r], cols[c]))
            objective += value
    objective += (n_prev - len(matches)) * disappear_cost
    objective += (n_cur - len(matches)) * appear_cost
    return matches, objective


def solve_assignment(
    pair: PairCosts,
    *,
    appear_cost: float = 1.5,
    disappear_cost: float = 1.5,
    division_penalty: float = 0.8,
    allow_division: bool = True,
    locked_divisions: list[tuple[int, int, int, float]] | None = None,
) -> Assignment:
    """Solve padded LAP for one frame pair.

    Matrix layout (square of size n_prev+n_cur + optional n_prev division rows
    compressed into the padding scheme):

      [ real_costs (n_prev x n_cur) | diag(disappear) (n_prev x n_prev) ]
      [ diag(appear)  (n_cur x n_cur) | zeros                          ]

    When ``allow_division`` is True we add ``n_prev`` extra *rows* that are
    copies of the real-cost rows plus ``division_penalty``. A previous cell
    that is assigned both its primary and its division row to two current
    cells is recorded as a division; the weaker of the two links is kept as
    the daughter, the stronger as the continuing parent.

    In practice a simpler scheme works well and stays O((n_p+n_c)^3):
    solve the standard appear/disappear padding first, then for each unmatched
    current cell, try attaching it as a second daughter of the nearest matched
    previous cell if the division cost beats appear_cost.

    ``locked_divisions`` (lineage mode): (prev, cur_a, cur_b, union_cost)
    divisions found beforehand; their rows/cols are left out of the LAP and
    they are returned in ``split_divisions``. Pass ``allow_division=False``
    with it to skip the legacy second-daughter attachment.
    """
    n_prev, n_cur = pair.n_prev, pair.n_cur
    result = Assignment()

    if n_prev == 0:
        result.appeared = list(range(n_cur))
        return result
    if n_cur == 0:
        result.disappeared = list(range(n_prev))
        return result

    locked = list(locked_divisions or [])
    drop_rows = {d[0] for d in locked}
    drop_cols = {d[1] for d in locked} | {d[2] for d in locked}
    matches, _ = solve_lap(
        pair.cost,
        appear_cost=appear_cost,
        disappear_cost=disappear_cost,
        drop_rows=drop_rows,
        drop_cols=drop_cols,
    )
    matched_prev: set[int] = set(drop_rows)
    matched_cur: set[int] = set(drop_cols)
    for r, c in matches:
        result.matches.append((r, c))
        result.costs[(r, c)] = float(pair.cost[r, c])
        matched_prev.add(r)
        matched_cur.add(c)
    result.split_divisions = [
        (int(d[0]), int(d[1]), int(d[2]), float(d[3])) for d in locked
    ]

    result.disappeared = [i for i in range(n_prev) if i not in matched_prev]
    result.appeared = [j for j in range(n_cur) if j not in matched_cur]

    if allow_division and result.appeared and matched_prev:
        result = _attach_divisions(
            result,
            pair,
            division_penalty=division_penalty,
            appear_cost=appear_cost,
        )

    return result


def _attach_divisions(
    result: Assignment,
    pair: PairCosts,
    *,
    division_penalty: float,
    appear_cost: float,
) -> Assignment:
    """Try to attach unmatched current cells as daughters of matched parents.

    A candidate parent must already have exactly one match. The division is
    accepted when cost(parent, daughter) + division_penalty < appear_cost.
    Each parent may gain at most one extra daughter.
    """
    children_of: dict[int, list[int]] = {}
    for p, c in result.matches:
        children_of.setdefault(p, []).append(c)

    still_appeared: list[int] = []
    used_parents: set[int] = set()

    # Greedily assign each appeared cell to its best eligible parent.
    candidates: list[tuple[float, int, int]] = []  # (cost, parent, daughter)
    for j in result.appeared:
        best_p = None
        best_c = np.inf
        for p in children_of:
            if len(children_of[p]) >= 2:
                continue
            c = float(pair.cost[p, j])
            if not np.isfinite(c):
                continue
            if c < best_c:
                best_c = c
                best_p = p
        if best_p is not None and best_c + division_penalty < appear_cost:
            candidates.append((best_c + division_penalty, best_p, j))
        else:
            still_appeared.append(j)

    candidates.sort()
    for total, p, j in candidates:
        if p in used_parents:
            still_appeared.append(j)
            continue
        if len(children_of.get(p, [])) >= 2:
            still_appeared.append(j)
            continue
        # Accept division.
        used_parents.add(p)
        children_of.setdefault(p, []).append(j)
        result.matches.append((p, j))
        result.costs[(p, j)] = float(pair.cost[p, j])
        # Mark in divisions list.
        kids = tuple(sorted(children_of[p]))
        # Replace any prior entry for this parent.
        result.divisions = [d for d in result.divisions if d[0] != p]
        if len(kids) == 2:
            result.divisions.append((p, kids))  # type: ignore[arg-type]

    result.appeared = still_appeared
    return result
