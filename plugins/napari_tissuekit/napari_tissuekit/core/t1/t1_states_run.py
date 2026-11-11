"""State-machine T1 classification of a tracked movie (see t1_states.py for the definitions).

  labels (T,Y,X tracked)  ->  contact lengths of all touching pairs per frame (um, cached)
                          ->  candidate quartets from pairs that lose / gain a stable interface
                          ->  per quartet: A-B / C-D length series -> states -> events -> classes
                          ->  one row per event (new columns + the legacy columns of *_t1_events.csv)

Everything is measured on the label image grown by `expand` px (as the existing detector does); the interface
length of a pair is its 4-connected contact-pixel count times the pixel size, for BOTH pairs of the quartet in the
same frame. Vertex-to-vertex lengths are not used for the classification, only in the descriptive junction-length tables
(`junction_length_traces` / `write_length_tables`). The definition in force is `DEFINITION` below.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import yaml
from scipy import ndimage as ndi
from skimage.segmentation import expand_labels

from .run_tm_fourway_vertices import contact_length_map
from .run_tm_t1 import load_division_frames, pair_outline_mask, tissue_border_cells
from .t1_states import StateParams, classify_quartet, hysteresis, seconds_to_frames

# The LATEST event definition (committed 2026-11-02). A junction (pair) is present from `on_px` contact pixels on and stays present down to
# `off_px` (hysteresis; contact length = 4-connected pixel-pair count of the labels grown by `expand` px, times the pixel size; the
# thresholds are taken 0.001 px lower so that an exact pixel count passes the >= test). Events start when a quartet leaves a configuration
# that lasted >= t_stable_s (2 frames in every movie we have) and end when the four cells leave their group, a configuration persisted
# t_end_s, or the movie ends. Side rule: while one pair touches, the two flanking cells must both touch both cells of it (checked at each
# flip, at the flip frame and the last frame of the previous O/N state); a failing first flip drops the event, a later one ends it before
# that flip. Clause: a true four-cell vertex frame (state V with all four side pairs in contact >= off_px) after the first flip makes a
# clean event flicker_resolved. Events without a swap and events unresolved at the end of the movie are not counted.
DEFINITION = dict(on_px=3.0, off_px=2.0, t_dwell_s=0.0, t_stable_s=76.0, t_end_s=480.0, side_rule=True, vertex_clause=True, expand=3)

CLASS_TO_LEGACY = {  # (kind, type) written to the legacy columns
    "clean": ("completed", "completed"),
    "flicker_resolved": ("completed", "flicker_resolved"),
    "flicker_aborted": ("aborted", "flicker_aborted"),
    "abortive_shrink": ("aborted", "abortive_shrink"),
    "unresolved": ("aborted", "unresolved"),
    "complex": ("aborted", "complex"),
}


@dataclass
class MovieData:
    stem: str
    T: int
    um_per_px: float
    dt_s: float
    labels: np.ndarray                      # (T, Y, X) tracked labels
    present: np.ndarray                     # (T, L) bool
    border: np.ndarray                      # (T, L) bool: cell touches background
    cy: np.ndarray                          # (T, L) centroid, px
    cx: np.ndarray
    pair_index: dict                        # (a, b) a < b -> row of `length`
    length: np.ndarray                      # (P, T) um; NaN where a cell is absent, 0 where present but not touching
    expand: int

    def pair_length(self, a: int, b: int) -> np.ndarray:
        a, b = (a, b) if a < b else (b, a)
        row = self.pair_index.get((a, b))
        if row is not None:
            return self.length[row]
        return np.where(self.present[:, a] & self.present[:, b], 0.0, np.nan)

    def neighbours(self, t: int, min_len: float) -> dict[int, dict[int, float]]:
        """cell -> {neighbour: length} of the pairs touching with at least min_len in frame t (cached)."""
        cache = self.__dict__.setdefault("_nb", {})
        key = (t, round(min_len, 6))
        if key not in cache:
            col = self.length[:, t]
            nb: dict[int, dict[int, float]] = {}
            for (a, b), r in self.pair_index.items():
                v = col[r]
                if v >= min_len:
                    nb.setdefault(a, {})[b] = float(v)
                    nb.setdefault(b, {})[a] = float(v)
            cache[key] = nb
        return cache[key]


def contact_length_map_euclid(lab: np.ndarray) -> dict[tuple[int, int], float]:
    """Direction-corrected interface length (px) of every touching pair: sqrt(n_h**2 + n_v**2) with n_h / n_v the numbers of
    horizontal / vertical neighbour links across the interface. A straight boundary of length L at angle a to the pixel grid
    crosses L|sin a| horizontal and L|cos a| vertical links, so this returns L for any direction, where the plain 4-connected
    count n_h + n_v returns up to 1.41 L (diagonal)."""
    acc = {}
    for axis, (a, b) in enumerate(((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :]))):
        mask = (a > 0) & (b > 0) & (a != b)
        if not mask.any():
            continue
        ua, ub = a[mask].astype(np.int64), b[mask].astype(np.int64)
        lo, hi = np.minimum(ua, ub), np.maximum(ua, ub)
        mx = int(lab.max()) + 1
        uniq, cnt = np.unique(lo * mx + hi, return_counts=True)
        for q, c in zip(uniq.tolist(), cnt.tolist()):
            key = (int(q // mx), int(q % mx))
            v = acc.setdefault(key, [0, 0])
            v[axis] += int(c)
    return {k: math.hypot(v[0], v[1]) for k, v in acc.items()}


def default_workers(n_tasks: int = 1 << 30, requested: int | None = None) -> int:
    """CPU cores to use: `requested`, else $TISSUEKIT_N_JOBS, else the cores this process may use (respects a Slurm allocation), at most `n_tasks`."""
    n = requested or int(os.environ.get("TISSUEKIT_N_JOBS", 0) or 0)
    if not n:
        n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, min(int(n), int(n_tasks)))


def build_movie_data(labels: np.ndarray, um_per_px: float, dt_s: float, stem: str, expand: int = 3,
                     cache: Path | None = None, metric: str = "pairs", workers: int | None = None) -> MovieData:
    """Contact lengths, presence, border cells and centroids of every frame (cached as an .npz).
    metric 'pairs' = 4-connected contact-pixel count (default); 'euclid' = direction-corrected length (contact_length_map_euclid)."""
    length_map = contact_length_map if metric == "pairs" else contact_length_map_euclid
    T = labels.shape[0]
    L = int(labels.max()) + 1
    if cache is not None and Path(cache).is_file():
        z = np.load(cache, allow_pickle=True)
        if int(z["expand"]) == expand and z["length"].shape[1] == T and int(z["L"]) == L:
            pairs = [tuple(p) for p in z["pairs"]]
            return MovieData(stem, T, um_per_px, dt_s, labels, z["present"], z["border"], z["cy"], z["cx"],
                             {p: i for i, p in enumerate(pairs)}, z["length"], expand)
    present = np.zeros((T, L), dtype=bool)
    border = np.zeros((T, L), dtype=bool)
    cy = np.full((T, L), np.nan, dtype=np.float32)
    cx = np.full((T, L), np.nan, dtype=np.float32)
    contact: dict[tuple[int, int], dict[int, float]] = {}
    yy, xx = np.indices(labels.shape[1:])

    def measure(t):  # one frame (frames are independent; numpy / scipy release the GIL, so frames run in threads)
        lab = labels[t].astype(np.int64)
        cnt = np.bincount(lab.ravel(), minlength=L).astype(float)
        ids = np.flatnonzero(cnt[1:]) + 1
        sy = np.bincount(lab.ravel(), weights=yy.ravel(), minlength=L)
        sx = np.bincount(lab.ravel(), weights=xx.ravel(), minlength=L)
        work = expand_labels(lab, distance=expand) if expand > 0 else lab
        return ids, sy[ids] / cnt[ids], sx[ids] / cnt[ids], tissue_border_cells(lab), length_map(work)

    n_workers = default_workers(T, workers)
    if n_workers > 1:
        with cf.ThreadPoolExecutor(n_workers) as pool:
            results = list(pool.map(measure, range(T)))
    else:
        results = map(measure, range(T))
    for t, (ids, cyv, cxv, bcells, pairs_t) in enumerate(results):
        present[t, ids] = True
        cy[t, ids], cx[t, ids] = cyv, cxv
        for c in bcells:
            border[t, c] = True
        for pair, px in pairs_t.items():
            contact.setdefault(pair, {})[t] = px * um_per_px
    pairs = sorted(contact)
    length = np.full((len(pairs), T), np.nan, dtype=np.float32)
    for r, (a, b) in enumerate(pairs):
        length[r] = np.where(present[:, a] & present[:, b], 0.0, np.nan)
        for t, v in contact[(a, b)].items():
            length[r, t] = v
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, present=present, border=border, cy=cy, cx=cx, length=length,
                            pairs=np.array(pairs, dtype=np.int64), expand=expand, L=L)
    return MovieData(stem, T, um_per_px, dt_s, labels, present, border, cy, cx, {p: i for i, p in enumerate(pairs)},
                     length, expand)


def _runs_true(x: np.ndarray):
    out, start = [], None
    for i, v in enumerate(x):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(x)))
    return out


def candidate_quartets(md: MovieData, l_on: float, l_off: float, min_run: int = 2) -> dict:
    """Quartets {X, Y} (two complementary pairs) around every pair that loses or gains an interface (hysteresis
    thresholds, present for at least `min_run` frames). The flanking cells are the two common neighbours with the
    longest shared interface. Returns {frozenset({X, Y}): n_common_neighbours}."""
    out: dict = {}

    def flank(a, b, f, need_apart):
        nb = md.neighbours(f, l_off)
        common = (set(nb.get(a, {})) & set(nb.get(b, {}))) - {a, b}
        if len(common) < 2:
            return None
        score = sorted(common, key=lambda c: -min(nb[a][c], nb[b][c]))[:4]
        best, bs = None, -1.0
        for i, c in enumerate(score):
            for d in score[i + 1:]:
                if need_apart and md.pair_length(c, d)[f] >= l_on:
                    continue
                s = min(nb[a][c], nb[b][c], nb[a][d], nb[b][d])
                if s > bs:
                    best, bs = (c, d), s
        return None if best is None else (best, len(common))

    for (a, b), r in md.pair_index.items():
        pres = hysteresis(md.length[r], l_on, l_off)
        for s, e in _runs_true(pres):
            if e - s < min_run:
                continue
            for f, need_apart in ((e - 1, True), (s, False)):  # last frame before it is lost / first frame it exists
                if (f == e - 1 and e >= md.T) or (f == s and s == 0):
                    continue  # still present at the end of the movie / present from the start: no transition seen
                got = flank(a, b, f, need_apart)
                if got is None:
                    continue
                (c, d), ncommon = got
                if md.border[f, [a, b, c, d]].any():
                    continue
                key = frozenset((frozenset((a, b)), frozenset((c, d))))
                out[key] = max(out.get(key, 0), ncommon)
    return out


def params_for(cfg: dict, stem: str, dt_s: float, t_dwell_s: float, t_stable_s: float, l_scale: float = 1.0,
               t_end_s: float = 0.0) -> StateParams:
    lc = cfg["lengths_um"].get(stem) or cfg["lengths_um"]["default"]
    return StateParams(l_on=float(lc["l_on"]) * l_scale, l_off=float(lc["l_off"]) * l_scale,
                       n_dwell=seconds_to_frames(t_dwell_s, dt_s), n_stable=seconds_to_frames(t_stable_s, dt_s), n_end=seconds_to_frames(t_end_s, dt_s))


def _angle(md: MovieData, expanded_cache: dict, t: int, a: int, b: int):
    from .run_tm_t1_picker import angle_from_mask  # lazy: heavy import
    if t < 0 or t >= md.T or not (md.present[t, a] and md.present[t, b]):
        return float("nan")
    key = (t, a, b)
    if key not in expanded_cache:
        # grown labels on a crop around the two cells only (a whole frame is ~10x slower; the angle only reads pixels near A and B)
        lab = md.labels[t]
        ys, xs = np.nonzero((lab == a) | (lab == b))
        m = 2 * int(md.expand) + 6
        y0, y1 = max(int(ys.min()) - m, 0), min(int(ys.max()) + m + 1, lab.shape[0])
        x0, x1 = max(int(xs.min()) - m, 0), min(int(xs.max()) + m + 1, lab.shape[1])
        sub = lab[y0:y1, x0:x1].astype(np.int64)
        expanded_cache.clear()
        expanded_cache[key] = expand_labels(sub, distance=md.expand) if md.expand > 0 else sub
    ang = angle_from_mask(pair_outline_mask(expanded_cache[key], a, b, md.expand + 1))
    return float("nan") if ang is None else float(ang)


def junction_centre(md: MovieData, pair: tuple, f0: int, f1: int) -> tuple[float, float, int]:
    """(x, y, n_frames): where the junction of `pair` is, averaged over the frames f0 .. f1 - 1 in which the two cells touch.

    The junction is the set of contact pixel pairs of the two cells in the labels grown by `md.expand` px (the contact the detector
    measures); per frame the mean x and y of those pixels (a horizontal pair sits between x and x + 1, a vertical one between y and y + 1),
    then the mean over the frames. n_frames = 0 (and NaN) when the cells never touch in that range."""
    a, b = int(pair[0]), int(pair[1])
    xs_f, ys_f = [], []
    for t in range(max(f0, 0), min(f1, md.T)):
        if not (md.present[t, a] and md.present[t, b]):
            continue
        cy, cx = (float(np.nanmean(md.cy[t, [a, b]])), float(np.nanmean(md.cx[t, [a, b]])))
        half = int(max(40.0, math.hypot(float(md.cy[t, a] - md.cy[t, b]), float(md.cx[t, a] - md.cx[t, b])) / 2 + 20 + md.expand))
        y0, y1 = max(0, int(cy) - half), min(md.labels.shape[1], int(cy) + half + 1)
        x0, x1 = max(0, int(cx) - half), min(md.labels.shape[2], int(cx) + half + 1)
        crop = md.labels[t, y0:y1, x0:x1].astype(np.int64)
        g = expand_labels(crop, distance=md.expand) if md.expand > 0 else crop
        ma, mb = g == a, g == b
        h = (ma[:, :-1] & mb[:, 1:]) | (mb[:, :-1] & ma[:, 1:])
        v = (ma[:-1] & mb[1:]) | (mb[:-1] & ma[1:])
        (hy, hx), (vy, vx) = np.nonzero(h), np.nonzero(v)
        if not len(hy) and not len(vy):
            continue
        xs_f.append(float(np.r_[hx + 0.5, vx].mean()) + x0)
        ys_f.append(float(np.r_[hy, vy + 0.5].mean()) + y0)
    if not xs_f:
        return float("nan"), float("nan"), 0
    return float(np.mean(xs_f)), float(np.mean(ys_f)), len(xs_f)


def group_ok_series(md: MovieData, cells: list[int], l_off: float) -> np.ndarray:
    """Per frame: all four cells present and the four of them one connected group (contact >= L_off between at least
    the pairs needed to connect them)."""
    T = md.T
    E = np.zeros((T, 4, 4), dtype=bool)
    present = np.ones(T, dtype=bool)
    for i in range(4):
        present &= md.present[:T, cells[i]]
        for j in range(i + 1, 4):
            on = np.nan_to_num(md.pair_length(cells[i], cells[j]).astype(float), nan=0.0) >= l_off
            E[:, i, j] = E[:, j, i] = on
    reach = np.zeros((T, 4), dtype=bool)
    reach[:, 0] = True
    for _ in range(3):
        reach = reach | (E & reach[:, None, :]).any(axis=2)
    return present & reach.all(axis=1)


def side_series(md: MovieData, pa: tuple, pb: tuple, min_len: float, strict: bool = False) -> np.ndarray:
    """Per frame: all four side pairs (each cell of pa with each cell of pb) are in contact (>= min_len, or > min_len if `strict`)."""
    L = np.stack([np.nan_to_num(md.pair_length(c1, c2).astype(float), nan=0.0) for c1 in pa for c2 in pb])
    return (L > min_len if strict else L >= min_len).all(axis=0)


_CLS: dict = {}   # per-quartet function and quartet list handed to the forked worker processes of classify_movie


def _classify_worker(i: int):
    return _CLS["one"](_CLS["items"][i])


def classify_movie(md: MovieData, p: StateParams, division_frames: dict[int, list[int]] | None = None,
                   cand: dict | None = None, orientation: bool = True, complex_pair_min: float | None = None,
                   apply_rules: bool = True, side_rule: bool = False, vertex_clause: bool = False, workers: int | None = None,
                   on_progress=None) -> pd.DataFrame:
    """One row per event of every candidate quartet. Classes: clean, flicker_resolved, flicker_aborted,
    abortive_shrink, unresolved, complex (complex has priority, then unresolved, see t1_states.py).
    `on_progress(done, total)` (optional) is called while the quartets are classified."""
    cand = candidate_quartets(md, p.l_on, p.l_off) if cand is None else cand
    division_frames = division_frames or {}
    cache = {}

    def one(item):  # all events of one candidate quartet (quartets are independent of each other)
        key, ncommon = item
        rows, dropped = [], []
        p0, p1 = sorted((tuple(sorted(x)) for x in key))
        cells = sorted(p0 + p1)
        lx, ly = md.pair_length(*p0).astype(float), md.pair_length(*p1).astype(float)
        gok = group_ok_series(md, cells, p.l_off) if p.n_end > 0 else None
        sok = side_series(md, p0, p1, 0.0, strict=True) if side_rule else None   # any contact, all four side pairs at once
        sl = side_series(md, p0, p1, p.l_off) if vertex_clause else None         # true four-cell vertex frames
        events, states, both = classify_quartet(lx, ly, p, gok, sok, sl)
        for ev in events:
            orig, other = (p0, p1) if ev.original == "O" else (p1, p0)
            fin = (other if ev.final != ev.original else orig) if ev.final is not None else None
            f_a = max(ev.start - 1, 0)                       # last frame of the stable original configuration
            f_b = ev.end if ev.end is not None else md.T - 1
            frames = range(ev.start, f_b + 1)
            # complex: a division of a quartet cell during the event, or a THIRD common neighbour E of a pair that is
            # in contact (interface >= L_on, as in the state definition; A-B before, C-D after the swap): E touches both cells of the pair (>= L_on each) in addition
            # to the two flanking quartet cells, the cell geometry of a 5-cell vertex (rosette)
            divides = any(ev.start <= f <= f_b for c in cells for f in division_frames.get(c, []))
            fifth = False
            for f in frames:
                nb = md.neighbours(f, p.l_on)
                for pa in (orig, other):
                    if md.pair_length(*pa)[f] < (p.l_on if complex_pair_min is None else complex_pair_min):  # pair not in contact
                        continue
                    if (set(nb.get(pa[0], {})) & set(nb.get(pa[1], {}))) - set(cells):
                        fifth = True
                        break
                if fifth:
                    break
            is_complex = bool(divides or fifth)
            # without the counting rules complex is its own class; with them it is only a flag (columns
            # complex_division / complex_fifth_cell / complex_flag): a complex event goes into the class its swaps give
            klass = "complex" if (is_complex and not apply_rules) else ev.klass
            drop_reason, from_unresolved = "", 0
            if apply_rules:
                # counting rules: an event without a single swap is dropped whatever it is (also a 4-cell vertex that
                # goes back); an unresolved event (never stable for T_stable, or a cell lost) is dropped if the movie
                # ended before it could have become stable (last observed frame < T_stable before the last frame), else
                # it counts by the swaps seen: 1 -> clean, >= 2 -> flicker_resolved (odd) / flicker_aborted (even)
                if ev.side_dropped:
                    drop_reason = "side cells not touching at the first flip"
                elif ev.flips == 0:
                    drop_reason = "0 swaps"
                elif ev.klass == "unresolved":
                    if p.n_end > 0 or md.T - 1 - ev.last_obs < p.n_stable:
                        drop_reason = "unresolved at the end of the movie"
                    else:
                        klass = "clean" if ev.flips == 1 else ("flicker_resolved" if ev.flips % 2 else "flicker_aborted")
                        from_unresolved = 1
            if md.border[ev.start, cells].any():  # as the current detector: cells touching the tissue border are excluded
                continue
            sel = [c for c in cells if md.present[f_a, c]] or cells
            cy = float(np.nanmean(md.cy[f_a, sel])) if len(sel) else float("nan")
            cx = float(np.nanmean(md.cx[f_a, sel])) if len(sel) else float("nan")
            if drop_reason:
                dropped.append(dict(frame_start=ev.start, A=cells[0], B=cells[1], C=cells[2], D=cells[3],
                                    t1_class_before_rules=klass, n_flips=(ev.flips_raw if ev.side_dropped else ev.flips), drop_reason=drop_reason,
                                    state_sequence=ev.sequence, last_obs=ev.last_obs))
                continue
            kind, typ = CLASS_TO_LEGACY[klass]
            o_start = _angle(md, cache, f_a, *orig) if orientation else float("nan")
            o_end = (_angle(md, cache, ev.end, *fin) if (orientation and ev.end is not None and fin) else float("nan"))
            dt = md.dt_s
            first_swap = ev.flip_frames[0] if ev.flip_frames else ev.start
            # event position = the shrinking junction (the breaking pair) before the first swap; the four-cell centre where it never touches
            jx, jy, n_junction = junction_centre(md, orig, f_a, first_swap)
            cx_cells, cy_cells = cx, cy
            if n_junction:
                cx, cy = jx, jy
            last_swap = ev.flip_frames[-1] if ev.flip_frames else ev.start
            rows.append(dict(
                # legacy columns (as *_t1_events.csv of the current detector)
                frame_start=ev.start, frame_end=f_b, frame_first_swap=first_swap, frame_last_swap=last_swap,
                frame_before=f_a, frame_after=f_b, A=cells[0], B=cells[1], C=cells[2], D=cells[3],
                pair_breaking=";".join(map(str, orig)), pair_building=";".join(map(str, other)),
                y=cy, x=cx, x_cells=cx_cells, y_cells=cy_cells, n_junction_frames=n_junction, source="states", kind=kind, type=typ, persist_pre=ev.pre_frames,
                persist_post=ev.post_frames, n_swaps=ev.flips, has_fourway=int(ev.v_frames > 0), other_matching=0,
                involves_division=int(divides), division_offset_frames="",
                # new columns
                t1_class=klass, n_flips=ev.flips, t_start_s=ev.start * dt,
                t_end_s=(ev.end * dt if ev.end is not None else float("nan")), time_in_V_s=ev.v_frames * dt,
                time_to_first_N_s=((ev.first_new - ev.start) * dt if ev.first_new is not None else float("nan")),
                orient_start_deg=o_start, orient_end_deg=o_end, centre_x_um=cx * md.um_per_px,
                centre_y_um=cy * md.um_per_px, state_sequence=ev.sequence, censored=int(ev.censored),
                complex_division=int(divides), complex_fifth_cell=int(fifth), both_frames=ev.both_frames,
                n_common_neighbours=ncommon, original_pair=";".join(map(str, orig)), new_pair=";".join(map(str, other)),
                flip_frames=";".join(map(str, ev.flip_frames)), end_reason=ev.end_reason, resolved_from_unresolved=from_unresolved, complex_flag=int(is_complex),
                n_flips_raw=(ev.flips_raw if ev.flips_raw >= 0 else ev.flips), side_truncated=int(ev.side_truncated),
                vertex_clause_flag=int(ev.vertex_flag),
            ))
        return rows, dropped

    items = sorted(cand.items(), key=lambda kv: sorted(sorted(x) for x in kv[0]))
    rows, dropped = [], []
    n_workers = default_workers(len(items), workers)
    total, last = len(items), -1

    def tick(done: int) -> None:  # report at most once per percent
        nonlocal last
        if on_progress is not None and (done * 100 // max(total, 1) != last or done == total):
            last = done * 100 // max(total, 1)
            on_progress(done, total)

    tick(0)
    if n_workers > 1 and "fork" in mp.get_all_start_methods():
        # plain-Python work per quartet: separate processes (forked, so md is shared); results come back in quartet order
        _CLS["one"], _CLS["items"] = one, items
        try:
            with cf.ProcessPoolExecutor(n_workers, mp_context=mp.get_context("fork")) as pool:
                for k, (r, d) in enumerate(pool.map(_classify_worker, range(len(items)), chunksize=16), 1):
                    rows += r
                    dropped += d
                    tick(k)
        finally:
            _CLS.clear()
    else:
        for k, item in enumerate(items, 1):
            r, d = one(item)
            rows += r
            dropped += d
            tick(k)
    df = pd.DataFrame(rows)
    df = df.sort_values(["frame_start", "A", "B", "C", "D"]).reset_index(drop=True) if len(df) else df
    df.attrs["dropped"] = pd.DataFrame(dropped)
    return df


def definition_params(dt_s: float, um_per_px: float, definition: dict | None = None) -> StateParams:
    """StateParams of the committed definition (DEFINITION, optionally overridden) for a movie with pixel size `um_per_px`."""
    d = {**DEFINITION, **(definition or {})}
    return StateParams(l_on=(d["on_px"] - 0.001) * um_per_px, l_off=(d["off_px"] - 0.001) * um_per_px,
                       n_dwell=seconds_to_frames(d["t_dwell_s"], dt_s), n_stable=seconds_to_frames(d["t_stable_s"], dt_s),
                       n_end=seconds_to_frames(d["t_end_s"], dt_s))


def classify_with_definition(md: MovieData, division_frames=None, definition: dict | None = None, orientation: bool = True,
                             cand: dict | None = None, workers: int | None = None, on_progress=None) -> pd.DataFrame:
    """classify_movie with the committed definition: hysteresis thresholds in pixels, group rule, side rule, true-vertex clause."""
    d = {**DEFINITION, **(definition or {})}
    p = definition_params(md.dt_s, md.um_per_px, d)
    return classify_movie(md, p, division_frames, cand, orientation=orientation, side_rule=bool(d["side_rule"]), vertex_clause=bool(d["vertex_clause"]),
                         workers=workers, on_progress=on_progress)


def _crop_geometry(lab: np.ndarray, quartet: frozenset, expand: int):
    """(3-cell vertices, adjacency) of one frame, computed only on a crop around the four cells of `quartet`.

    The crop holds every pixel of the four cells plus a margin wider than the label growing (`expand`), so the grown labels, the
    adjacency of the quartet cells and the triple points between them are the same as on the whole frame (the other cells in the
    crop are cut at its border, but nothing reads them). Whole frames cost about 6 s each, the crop a few milliseconds."""
    from ..celltrack.cells import build_adjacency
    from .run_tm_fourway_vertices import frame_geometry
    mask = np.isin(lab, list(quartet))
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return [], {}
    m = int(expand) + 8
    y0, y1 = max(int(ys.min()) - m, 0), min(int(ys.max()) + m + 1, lab.shape[0])
    x0, x1 = max(int(xs.min()) - m, 0), min(int(xs.max()) + m + 1, lab.shape[1])
    sub = lab[y0:y1, x0:x1]
    grown = expand_labels(sub, distance=expand) if expand > 0 else sub   # grown once, used for the vertices and the adjacency
    verts3 = frame_geometry(grown, 0)[1]
    for v in verts3:  # back to frame coordinates
        v["y"] += y0
        v["x"] += x0
    return verts3, build_adjacency(grown, expand=0)


_JL: dict = {}   # labels / expand handed to the forked worker processes of junction_length_traces


def _junction_event_frame(job):
    eid, t, quartet = job
    v, a = _crop_geometry(_JL["labels"][t].astype(np.int64), quartet, _JL["expand"])
    return eid, t, v, a


def junction_length_traces(md: MovieData, ev: pd.DataFrame, pre: int = 3, post: int = 3, anchor: str = "start",
                           workers: int | None = None, on_progress=None) -> pd.DataFrame:
    """Junction length of the active pair around every event, long table (one row per event and frame). Window = anchor - pre ..
    last swap + post (anchor 'start' = frame_start, 'first_swap' = frame_first_swap); dt = frame - anchor. In every frame the active pair is the
    original pair (sign +1) or the new pair (sign -1) when exactly one of them touches (any contact on the grown labels, as the legacy tables);
    otherwise the length is blank and `flag` says '4way' (both touch) or 'no AB/CD contact'. vertex_px = vertex-to-vertex length of the
    junction, contact_px = 4-connected contact-pixel count (both in pixels, *_um in um); signed_* = sign * length.
    Only a crop around each event's four cells is analysed in each frame (see `_crop_geometry`).
    `on_progress(done, total)` is called after every measured (event, frame) of the serial path (the plugin runs it in a Qt thread)."""
    from .run_tm_fourway_vertices import vertex_bond
    from .run_tm_t1 import FLAG_NV, active_abcd_pair, length_blank_flag
    T, um = md.T, md.um_per_px
    wins, jobs = [], []
    for eid, r in enumerate(ev.itertuples(index=False)):
        r = r._asdict()
        first = int(r["frame_first_swap"] if anchor == "first_swap" else r["frame_start"])
        t0, t1 = max(0, first - pre), min(T - 1, int(r["frame_last_swap"]) + post)
        quartet = frozenset(int(r[k]) for k in "ABCD")
        jobs += [(eid, t, quartet) for t in range(t0, t1 + 1)]
        wins.append((eid, r, first, t0, t1))
    geo = {}
    n_workers = default_workers(len(jobs), workers)
    if n_workers > 1 and "fork" in mp.get_all_start_methods():
        _JL.update(labels=md.labels, expand=md.expand)
        try:
            with cf.ProcessPoolExecutor(n_workers, mp_context=mp.get_context("fork")) as pool:
                for eid, t, v, a in pool.map(_junction_event_frame, jobs, chunksize=8):
                    geo[eid, t] = (v, a)
        finally:
            _JL.clear()
    else:
        for i, (eid, t, quartet) in enumerate(jobs):
            geo[eid, t] = _crop_geometry(md.labels[t].astype(np.int64), quartet, md.expand)
            if on_progress is not None:
                on_progress(i + 1, len(jobs))
    rows = []
    for eid, r, first, t0, t1 in wins:
        quartet = frozenset(int(r[k]) for k in "ABCD")
        ab = frozenset(int(v) for v in str(r["pair_breaking"]).split(";"))
        cd = frozenset(int(v) for v in str(r["pair_building"]).split(";"))
        for t in range(t0, t1 + 1):
            verts3, adj = geo[eid, t]
            pair, sign = active_abcd_pair(adj, ab, cd)
            row = dict(event_id=eid, type=r["type"], t1_class=r["t1_class"], frame=t, dt=t - first, t_s=(t - first) * md.dt_s, sign=sign,
                       flag="", vertex_px=np.nan, contact_px=np.nan)
            if sign == 0 or pair is None:
                row["flag"] = length_blank_flag(adj, ab, cd)
            else:
                a, b = tuple(pair)
                lv, _, _ = vertex_bond(verts3, pair, quartet)
                c = float(md.pair_length(a, b)[t]) / um
                if lv is None:
                    row["flag"] = FLAG_NV
                else:
                    row["vertex_px"] = float(lv)
                if c > 0:
                    row["contact_px"] = c
                else:
                    row["flag"] = row["flag"] or FLAG_NV
            rows.append(row)
    df = pd.DataFrame(rows)
    for k in ("vertex", "contact"):
        df[f"{k}_um"] = df[f"{k}_px"] * um
        df[f"signed_{k}_px"] = df[f"{k}_px"] * df["sign"]
        df[f"signed_{k}_um"] = df[f"{k}_um"] * df["sign"]
    return df


def write_length_tables(md: MovieData | None, ev: pd.DataFrame, out: Path, stem: str, pre: int = 3, post: int = 3, anchor: str = "start",
                        workers: int | None = None, traces: pd.DataFrame | None = None) -> pd.DataFrame:
    """Legacy wide tables <stem>_junction_lengths{,_signed,_contact_lengths,_contact_lengths_signed}.csv (px) for the states events.
    `traces` (from `junction_length_traces` on the same `ev`) skips the measuring; `md` may then be None."""
    from ..output_names import tagged_path
    from .run_tm_t1 import FLAG_NV, TYPE_SORT_ORDER, write_junction_lengths_wide
    tr = traces if traces is not None else junction_length_traces(md, ev, pre, post, anchor, workers)
    vp, vs, cp, cs = {}, {}, {}, {}
    for r in tr.itertuples(index=False):
        k = (r.event_id, r.dt)
        if r.sign == 0:
            vp[k] = vs[k] = cp[k] = cs[k] = r.flag
            continue
        mv = FLAG_NV if np.isnan(r.vertex_px) else round(r.vertex_px, 2)
        mc = FLAG_NV if np.isnan(r.contact_px) else round(r.contact_px, 2)
        vp[k], vs[k] = mv, (mv if isinstance(mv, str) else round(mv * r.sign, 2))
        cp[k], cs[k] = mc, (mc if isinstance(mc, str) else round(mc * r.sign, 2))
    meta = []
    for eid, r in enumerate(ev.itertuples(index=False)):
        first = int(r.frame_first_swap if anchor == "first_swap" else r.frame_start)
        quartet = [int(r.A), int(r.B), int(r.C), int(r.D)]
        meta.append((eid, r.type, first, frozenset(int(v) for v in str(r.pair_breaking).split(";")), frozenset(int(v) for v in str(r.pair_building).split(";"))))
    ordered = sorted(meta, key=lambda m: (TYPE_SORT_ORDER.get(m[1], 99), m[2], m[0]))
    cols = [(eid, etype, ab, cd) for eid, etype, _, ab, cd in ordered]
    write_junction_lengths_wide(tagged_path(out, stem, "junction_lengths.csv"), cols, vp, vs, cp, cs, sorted(set(tr.dt)))
    return tr


def run_states_stack(stack: np.ndarray, *, um_per_px: float, dt_s: float, rgb: np.ndarray | None = None, definition: dict | None = None,
                     pre: int = 3, post: int = 3, exclude_ids: set[int] | None = None, division_frames: dict[int, list[int]] | None = None,
                     inside: list | None = None, progress=None, length_tables: bool = False) -> dict:
    """Plugin counterpart of `run_t1_stack` for the state classifier: classify a (T,Y,X) tracked label stack with the committed definition
    (DEFINITION, optionally overridden) and return the same kind of result dict (events as dicts with t_start / t_end / quartet / pair_before /
    pair_after / type, overlay, dots, ...), so the plugin's filter, overlay and autosave code works on it unchanged. `um_per_px` and `dt_s`
    are needed: the thresholds are pixel counts but the persistence times are seconds. Events have the legacy `type` (completed / flicker_resolved /
    flicker_aborted) and the new `t1_class` (clean / flicker_resolved / flicker_aborted); pair_after is the pair the quartet ends on.
    `inside` (optional): per frame the set of track IDs in the analysis region (core/region.py); only events whose four cells are
    all inside at the frame of the first swap are kept (the others are counted in `n_outside_region`).
    `length_tables`: also measure the junction length of every kept event (`length_traces` in the result, for `write_length_tables`;
    one vertex / adjacency pass per frame near an event, minutes on a long movie)."""
    from ..region import event_inside
    from .run_tm_t1 import grayscale_rgb, labels_gray_rgb, render_fourway_dots, render_overlay
    d = {**DEFINITION, **(definition or {})}
    n = int(stack.shape[0])
    # progress(overall %, step name, step %): overall 0-10 contacts, 10-80 classifying, 80-100 overlay and results
    def report(pct, name, step=-1):
        if progress is not None:
            progress(int(pct), name, int(step))

    report(1, "Measuring contacts", 0)
    md = build_movie_data(np.asarray(stack), float(um_per_px), float(dt_s), "plugin", int(d["expand"]))
    report(10, "Finding candidate quartets", 100)
    cand, n_skipped = None, 0
    if inside is not None:
        # a counted event needs all four cells inside at one frame: a quartet with a cell that is never inside cannot
        # be counted, so it is not classified at all (same events as filtering afterwards, much less work)
        ever = set().union(*inside) if inside else set()
        p = definition_params(md.dt_s, md.um_per_px, d)
        cand_all = candidate_quartets(md, p.l_on, p.l_off)
        cand = {k: v for k, v in cand_all.items() if set().union(*[set(x) for x in k]) <= ever}
        n_skipped = len(cand_all) - len(cand)

    def on_quartets(done, total):
        report(10 + 70 * done / max(total, 1), f"Classifying events ({done}/{total} quartets)", 100 * done / max(total, 1))

    report(10, "Classifying events", 0)
    df = classify_with_definition(md, division_frames, d, cand=cand, workers=1,  # serial: the plugin runs this in a Qt worker thread (no forking there)
                                  on_progress=on_quartets)
    report(80, "Collecting events", 0)
    dropped = df.attrs.get("dropped", pd.DataFrame())
    exclude_ids = exclude_ids or set()
    events, four_xy = [], [dict() for _ in range(n)]
    n_excluded = n_outside = 0
    for r in df.to_dict("records"):
        quartet = frozenset(int(r[k]) for k in "ABCD")
        if quartet & exclude_ids:
            n_excluded += 1
            continue
        if inside is not None and not event_inside({"quartet": quartet, "t_first_swap": int(r["frame_first_swap"])}, inside):
            n_outside += 1
            continue
        orig = frozenset(int(v) for v in str(r["pair_breaking"]).split(";"))
        new = frozenset(int(v) for v in str(r["pair_building"]).split(";"))
        r.update(quartet=quartet, pair_before=orig, pair_after=(new if int(r["n_flips"]) % 2 else orig),
                 t_start=int(r["frame_start"]), t_end=int(r["frame_end"]), t_first_swap=int(r["frame_first_swap"]),
                 t_before=int(r["frame_before"]), t_after=int(r["frame_after"]))
        sel = sorted(quartet)
        for t in range(r["t_start"], r["t_end"] + 1):
            ok = [c for c in sel if md.present[t, c]]
            if ok:
                four_xy[t][quartet] = (float(np.nanmean(md.cy[t, ok])), float(np.nanmean(md.cx[t, ok])))
        events.append(r)
    length_traces = None
    if length_tables and events:
        report(82, "Measuring junction lengths", 0)
        length_traces = junction_length_traces(
            md, pd.DataFrame(events), 3, 3, "start", workers=1,  # serial: Qt worker thread, no forking
            on_progress=lambda done, total: report(82 + 8 * done / max(total, 1), f"Measuring junction lengths ({done}/{total} frames)",
                                                   100 * done / max(total, 1)))
    report(90, "Drawing the overlay", 0)
    labels_list = [stack[t] for t in range(n)]
    if rgb is not None:
        overlay_base = grayscale_rgb(rgb)
    else:
        max_lab = int(stack.max()) if stack.size else 0
        overlay_base = np.empty((n, stack.shape[1], stack.shape[2], 3), dtype=np.uint8)
        for t in range(n):
            overlay_base[t] = labels_gray_rgb(stack[t], max_lab)
    overlay = render_overlay(overlay_base, labels_list, events, four_xy, pre=pre, post=post, reach=int(d["expand"]))
    report(95, "Drawing the overlay", 70)
    dots = render_fourway_dots(stack, events, four_xy, rgb=rgb)
    classes = pd.Series([e["t1_class"] for e in events]).value_counts().to_dict()
    reasons = dropped["drop_reason"].value_counts().to_dict() if len(dropped) else {}
    note = (f"Event definition: {d['on_px']:g} px on / {d['off_px']:g} px off, new state held {d['t_stable_s']:g} s, event ends after "
            f"{d['t_end_s']:g} s of persistence; side-cell rule {'on' if d['side_rule'] else 'off'}, vertex clause {'on' if d['vertex_clause'] else 'off'}. "
            f"Classes {classes}; not counted: {reasons}."
            + (f" {n_outside} event(s) outside the analysis region left out." if inside is not None else "")
            + (f" {n_skipped} candidate quartet(s) with a cell outside the region were skipped." if n_skipped else ""))
    report(100, f"T1s={len(events)}", 100)
    return {"classifier": "states", "definition": d, "stack": stack, "frame_indices": np.arange(n), "events": events, "events_shown": events,
            "four_xy": four_xy, "adj": [], "present": [], "overlay": overlay, "dots": dots, "overlay_base": overlay_base, "n_frames": n,
            "n_t1_confirmed": len(events), "n_excluded_events": n_excluded, "n_outside_region": n_outside, "classes": classes, "dropped": reasons, "swap_note": note,
            "expand": int(d["expand"]), "pre": pre, "post": post, "length_traces": length_traces}


def load_config(path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_movie(cfg: dict, stem: str, root: Path, cache_dir: Path | None = None, subsample: int = 1, metric: str = "pairs"):
    """MovieData and lineage divisions of one configured movie; `subsample` keeps every n-th frame."""
    mv = cfg["movies"][stem]
    tdir = root / mv["tracking_dir"]
    cal = json.loads((root / mv["calibration"]).read_text())
    labels = tifffile.imread(next(tdir.glob("*_tracked_labels.tif")))
    labels = labels[:int(mv.get("n_frames", labels.shape[0]))]
    um, dt = float(cal["um_per_px"]), float(cal["s_per_frame"])
    div = load_division_frames(next(tdir.glob("*_lineage.csv"))) if next(tdir.glob("*_lineage.csv"), None) else {}
    if subsample > 1:
        labels = labels[::subsample]
        dt *= subsample
        div = {k: sorted({int(math.ceil(f / subsample)) for f in v}) for k, v in div.items()}  # division frame in subsampled index
    tag = "" if metric == "pairs" else f"_{metric}"
    cache = None if cache_dir is None else cache_dir / f"{stem}_contacts_sub{subsample}{tag}.npz"
    return build_movie_data(labels, um, dt, stem, int(cfg.get("expand", 3)), cache, metric), div


def run_states_cli(*, labels_path: Path, rgb: Path | None, out: Path, stem: str, cal, lineage: Path | None,
                   config: Path | None = None, t_dwell: float | None = None, t_stable: float | None = None, t_end: float | None = None,
                   on_px: float | None = None, off_px: float | None = None, side_rule: bool | None = None,
                   vertex_clause: bool | None = None, tracked_stem: str | None = None, n_jobs: int | None = None,
                   region: Path | None = None) -> Path:
    """`run_tm_t1.py --classifier states`: classify one movie with the committed definition (DEFINITION; any argument that is not None
    overrides it) and write <stem>_t1_events.csv (legacy columns + new ones), <stem>_qc.json, the junction-length tables of the events
    (vertex-to-vertex and contact length of the active pair, as the legacy tables), the calibrated twins and the picker-ready bundle.
    `region` (analysis_region/region.json from the plugin's Analysis cropping tab): only events whose four cells are inside the kept area at
    the frame of their first swap are written. `config` (YAML) is only read for `expand` and kept for compatibility. `n_jobs`: CPU cores (default: $TISSUEKIT_N_JOBS or all cores of the allocation)."""
    from .calibrated_outputs import write_um_files
    from .t1_picker_ready import _write_one_picker_dir
    cfg = load_config(config) if config is not None and Path(config).is_file() else {}
    over = {k: v for k, v in dict(t_dwell_s=t_dwell, t_stable_s=t_stable, t_end_s=t_end, on_px=on_px, off_px=off_px, side_rule=side_rule,
                                  vertex_clause=vertex_clause).items() if v is not None}
    d = {**DEFINITION, **cfg.get("definition", {}), **over}
    labels = np.asarray(tifffile.imread(labels_path))
    um = float(cal.um_per_px) if cal is not None and getattr(cal, "um_per_px", None) else 1.0
    dt = float(cal.s_per_frame) if cal is not None and getattr(cal, "s_per_frame", None) else 1.0
    md = build_movie_data(labels, um, dt, stem, int(cfg.get("expand", d["expand"])), workers=n_jobs)
    p = definition_params(dt, um, d)
    div = load_division_frames(lineage) if lineage is not None and Path(lineage).is_file() else {}
    df = classify_with_definition(md, div, d, workers=n_jobs)
    dropped = df.attrs.get("dropped", pd.DataFrame())
    df.attrs = {}
    n_outside = 0
    if region is not None:
        from ..region import Region, event_inside, geometry_for

        lin_df = pd.read_csv(lineage) if lineage is not None and Path(lineage).is_file() else None
        inside = geometry_for(Region.load(region), labels, lin_df).inside_sets()
        keep = [event_inside({"quartet": [r.A, r.B, r.C, r.D], "t_first_swap": r.frame_first_swap}, inside) for r in df.itertuples()]
        n_outside = int(len(df) - sum(keep))
        df = df[keep].reset_index(drop=True)  # event_id = row index in the length tables, so renumber after filtering
    out.mkdir(parents=True, exist_ok=True)
    ev_path = out / f"{stem}_t1_events.csv"
    df.to_csv(ev_path, index=False)
    qc = dict(classifier="states", definition=d, l_on_um=p.l_on, l_off_um=p.l_off, n_dwell_frames=p.n_dwell, n_stable_frames=p.n_stable,
              n_end_frames=p.n_end, n_events=len(df), classes=df["t1_class"].value_counts().to_dict() if len(df) else {},
              dropped=dropped["drop_reason"].value_counts().to_dict() if len(dropped) else {}, labels=str(labels_path),
              note="abortive_shrink is detected less reliably at coarse time resolution",
              **({"analysis_region": str(region), "n_outside_region": n_outside} if region is not None else {}))
    (out / f"{stem}_qc.json").write_text(json.dumps(qc, indent=2) + "\n")
    print(f"Wrote {ev_path}: {qc['classes']}; dropped {qc['dropped']}", flush=True)
    if len(df):
        write_length_tables(md, df, out, stem, workers=n_jobs)
    write_um_files(out, stem, cal)
    # picker bundle next to the detection folder ONLY (not the shared output/t1_picker_ready/<stem> mirror that the
    # legacy detector writes: that would overwrite the bundle of the legacy run of the same movie)
    if rgb is not None and Path(rgb).is_file():
        dest = out.parent / "picker_ready" if out.name == "detect" else out / f"{stem}_t1_picker"
        _write_one_picker_dir(dest, stem=stem, events=ev_path, labels=Path(labels_path), rgb_path=Path(rgb),
                              four_path=None, calibration=cal.as_dict() if cal else None)
        print(f"Picker-ready: {dest}", flush=True)
    return ev_path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="State-machine T1 classification of one movie")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--t-dwell", type=float, required=True, help="seconds")
    ap.add_argument("--t-stable", type=float, required=True, help="seconds")
    ap.add_argument("--subsample", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True, help="output CSV")
    ap.add_argument("--root", type=Path, default=Path.cwd())
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    md, div = load_movie(cfg, a.stem, a.root, a.root / cfg.get("cache_dir", "output/t1_states/cache"), a.subsample)
    p = params_for(cfg, a.stem, md.dt_s, a.t_dwell, a.t_stable)
    df = classify_movie(md, p, div)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{a.stem}: {len(df)} events; {df['t1_class'].value_counts().to_dict() if len(df) else {}}")


if __name__ == "__main__":
    main()
