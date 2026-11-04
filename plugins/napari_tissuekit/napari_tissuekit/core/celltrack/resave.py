"""Rebuild every tracking output after manual track edits.

Track edits relabel whole cells, so the ID change is recovered by comparing the
edited stack with the original ``{stem}_tracked_labels.tif`` (per frame, old ID
-> new ID by pixel overlap). Existing tables are remapped with those maps;
what the edit changed (new continuations, new track ends) is added as
``manual`` rows. Divisions are carried over to the corrected IDs and dropped,
with a reason, if an edit no longer lets them hold.

``rebuild_tracking`` is pure (arrays and DataFrames); ``load_tables`` and
``write_tracking`` do the file work.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.measure import regionprops

from ..movie_calibration import save_tiff
from ..output_names import tagged_path
from .tracks import MAGENTA, render_tracked_rgb

ROLES = (
    "tracked_labels.tif",
    "tracked_rgb.tif",
    "tracks.csv",
    "links.csv",
    "track_events.csv",
    "lineage.csv",
    "qc.json",
)
MANUAL = "manual edit"


@dataclass
class Rebuilt:
    tracked: np.ndarray
    rgb: np.ndarray
    tracks: pd.DataFrame
    links: pd.DataFrame | None
    events: pd.DataFrame | None
    lineage: pd.DataFrame | None
    qc: dict
    dropped_divisions: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    n_ids_changed: int = 0


def id_maps(baseline: np.ndarray, edited: np.ndarray):
    """Per frame ``{old_id: new_id}`` for IDs the edit changed, plus warnings."""
    if baseline.shape != edited.shape:
        raise ValueError(f"edited {edited.shape} differs from original {baseline.shape}")
    maps: list[dict[int, int]] = []
    ambiguous: list[str] = []
    for t in range(edited.shape[0]):
        b, e = baseline[t], edited[t]
        sel = b > 0
        if not sel.any():
            maps.append({})
            continue
        bv = b[sel].astype(np.int64)
        ev = e[sel].astype(np.int64)
        k = int(ev.max()) + 1
        keys, counts = np.unique(bv * k + ev, return_counts=True)
        old, new = keys // k, keys % k
        best: dict[int, tuple[int, int]] = {}
        total: dict[int, int] = {}
        for o, n, c in zip(old.tolist(), new.tolist(), counts.tolist()):
            total[o] = total.get(o, 0) + c
            if o not in best or c > best[o][1]:
                best[o] = (n, c)
        for o, n, c in zip(old.tolist(), new.tolist(), counts.tolist()):
            if n != best[o][0] and c / total[o] > 0.05:
                ambiguous.append(f"frame {t}: cell {o} now has several IDs")
                break
        maps.append({o: n for o, (n, _) in best.items() if o != n and n > 0})
    return maps, ambiguous


def _presence(edited: np.ndarray):
    pres: list[set[int]] = []
    for t in range(edited.shape[0]):
        ids = np.unique(edited[t])
        pres.append({int(i) for i in ids if i > 0})
    return pres


def _start_end(pres: list[set[int]]):
    start: dict[int, int] = {}
    end: dict[int, int] = {}
    for t, ids in enumerate(pres):
        for i in ids:
            start.setdefault(i, t)
            end[i] = t
    return start, end


def _tracks_table(edited, pres, maps, old_tracks, seg_masks):
    local: dict[tuple[int, int], int] = {}
    score: dict[int, float] = {}
    old_id_at: dict[tuple[int, int], int] = {}
    if old_tracks is not None and len(old_tracks):
        for f, lab, tid, sc in zip(
            old_tracks["frame"], old_tracks["local_label"],
            old_tracks["track_id"], old_tracks["topology_score"],
        ):
            f, lab, tid = int(f), int(lab), int(tid)
            old_id_at[(f, lab)] = tid
            new = maps[f].get(tid, tid) if f < len(maps) else tid
            local[(f, new)] = lab
            if f not in score and pd.notna(sc):
                score[f] = float(sc)
    rows = []
    for t in range(edited.shape[0]):
        if not pres[t]:
            continue
        for rp in regionprops(edited[t].astype(np.int64)):
            tid = int(rp.label)
            lab = local.get((t, tid))
            if lab is None:
                lab = 0
                if seg_masks is not None:
                    co = rp.coords
                    lab = int(np.bincount(seg_masks[t][co[:, 0], co[:, 1]]).argmax())
            rows.append(
                {
                    "frame": t,
                    "local_label": lab,
                    "track_id": tid,
                    "area": int(rp.area),
                    "centroid_y": round(float(rp.centroid[0]), 2),
                    "centroid_x": round(float(rp.centroid[1]), 2),
                    "topology_score": score.get(t),
                }
            )
    cols = ["frame", "local_label", "track_id", "area", "centroid_y", "centroid_x", "topology_score"]
    return pd.DataFrame(rows, columns=cols), local, old_id_at


def _lineage(old_lineage, maps, pres, frames_nz):
    """Carry divisions over to corrected IDs. Returns (table, held, dropped)."""
    start, end = _start_end(pres)
    old = old_lineage.set_index("track_id")
    by_parent: dict[int, list[int]] = {}
    for tid, row in old.iterrows():
        if int(row["parent_track_id"]) > 0:
            by_parent.setdefault(int(row["parent_track_id"]), []).append(int(tid))
    old_flagged = {int(t) for t in old.index[old["flagged"].astype(bool)]}
    held: list[dict] = []
    dropped: list[str] = []
    for parent, kids in by_parent.items():
        f = min(int(old.loc[k, "start_frame"]) for k in kids)
        e_p = int(old.loc[parent, "end_frame"]) if parent in old.index else f - 1
        p_new = maps[e_p].get(parent, parent) if e_p < len(maps) else parent
        d_new = [maps[int(old.loc[k, "start_frame"])].get(k, k) for k in kids]
        prev = max((x for x in frames_nz if x < f), default=None)
        label = f"parent {parent} -> {kids} at frame {f}"
        if len(set(d_new)) != len(d_new) or p_new in d_new:
            dropped.append(f"{label}: cells merged into one track")
        elif prev is None or end.get(p_new) != prev:
            dropped.append(f"{label}: parent track {p_new} no longer ends at frame {prev}")
        elif any(start.get(d) != f for d in d_new):
            dropped.append(f"{label}: a daughter no longer starts at frame {f}")
        else:
            held.append(
                {"parent": p_new, "daughters": d_new, "frame_from": prev, "frame_to": f,
                 "flagged": parent in old_flagged}
            )
    parents = {d: h["parent"] for h in held for d in h["daughters"]}
    dividing = {h["parent"] for h in held}
    flagged = {h["parent"] for h in held if h["flagged"]}
    rows = [
        {
            "track_id": tid,
            "start_frame": start[tid],
            "end_frame": end[tid],
            "parent_track_id": parents.get(tid, 0),
            "divides": tid in dividing,
            "flagged": tid in flagged,
        }
        for tid in sorted(start)
    ]
    cols = ["track_id", "start_frame", "end_frame", "parent_track_id", "divides", "flagged"]
    return pd.DataFrame(rows, columns=cols), held, dropped


def _events(old_events, maps, pres, frames_nz, held, local, old_id_at, lineage_mode):
    """Remap valid events; add ``manual`` rows for what the edit changed."""
    nxt = dict(zip(frames_nz[:-1], frames_nz[1:]))
    heldset = {(h["parent"], frozenset(h["daughters"])) for h in held}

    def nm(f, tid):
        if tid is None or pd.isna(tid):
            return None
        tid = int(tid)
        return maps[f].get(tid, tid) if 0 <= f < len(maps) else tid

    def here(f, tid):
        return tid is not None and 0 <= f < len(pres) and tid in pres[f]

    kept: list[dict] = []
    out_cov: set[tuple[int, int]] = set()
    in_cov: set[tuple[int, int]] = set()
    if old_events is not None:
        for r in old_events.to_dict("records"):
            f0, f1, kind = int(r["frame_from"]), int(r["frame_to"]), r["kind"]
            lab0 = r["label_from"]
            p_old = old_id_at.get((f0, int(lab0))) if pd.notna(lab0) else None
            p_new = nm(f0, p_old)
            new = dict(r)
            if kind == "divide":
                if pd.notna(r["track_id_c"]) and lineage_mode:
                    p = nm(f0, r["track_id"])
                    b, c = nm(f1, r["track_id_b"]), nm(f1, r["track_id_c"])
                    if (p, frozenset((b, c))) not in heldset:
                        continue
                    new.update(track_id=p, track_id_b=b, track_id_c=c)
                    out_cov.add((f0, p))
                    in_cov.update({(f1, b), (f1, c)})
                else:
                    a, b = nm(f1, r["track_id"]), nm(f1, r["track_id_b"])
                    if not (here(f1, a) and here(f1, b)):
                        continue
                    new.update(track_id=a, track_id_b=b)
                    in_cov.update({(f1, a), (f1, b)})
                    if p_new is not None:
                        out_cov.add((f0, p_new))
            elif kind == "continue":
                to = nm(f1, r["track_id"])
                if p_new is None or p_new != to or not (here(f0, to) and here(f1, to)):
                    continue
                new["track_id"] = to
                out_cov.add((f0, to))
                in_cov.add((f1, to))
            elif kind == "appear":
                to = nm(f1, r["track_id"])
                if not here(f1, to) or here(f0, to):
                    continue
                new["track_id"] = to
                in_cov.add((f1, to))
            elif kind == "disappear":
                fr = nm(f0, r["track_id"])
                if not here(f0, fr) or here(f1, fr):
                    continue
                new["track_id"] = fr
                out_cov.add((f0, fr))
            else:  # merge or anything unknown: remap only
                new["track_id"] = nm(f1, r["track_id"])
            kept.append(new)

    manual: list[dict] = []

    def row(kind, f0, f1, tid):
        return {
            "kind": kind, "frame_from": f0, "frame_to": f1,
            "label_from": local.get((f0, tid)) if kind != "appear" else None,
            "label_to": local.get((f1, tid)) if kind != "disappear" else None,
            "track_id": tid, "track_id_b": None, "track_id_c": None,
            "cost": None, "topology_score": None, "flagged": False, "note": MANUAL,
        }

    for t, t2 in nxt.items():
        for tid in sorted(pres[t]):
            if (t, tid) in out_cov:
                continue
            if tid in pres[t2]:
                manual.append(row("continue", t, t2, tid))
                in_cov.add((t2, tid))
            else:
                manual.append(row("disappear", t, t2, tid))
            out_cov.add((t, tid))
        for tid in sorted(pres[t2]):
            if (t2, tid) not in in_cov and tid not in pres[t]:
                manual.append(row("appear", t, t2, tid))
                in_cov.add((t2, tid))

    cols = ["kind", "frame_from", "frame_to", "label_from", "label_to", "track_id",
            "track_id_b", "track_id_c", "cost", "topology_score", "flagged", "note"]
    df = pd.DataFrame(kept + manual, columns=cols)
    return df.sort_values(["frame_from", "frame_to"], kind="stable").reset_index(drop=True), manual


def _links(old_links, maps, manual_events, held, old_id_at):
    cols = ["frame_from", "label_from", "frame_to", "label_to", "kind", "cost",
            "topology_score", "track_id"]
    rows = []
    heldset = {(h["parent"], d) for h in held for d in h["daughters"]}

    def nm(f, tid):
        tid = int(tid)
        return maps[f].get(tid, tid) if 0 <= f < len(maps) else tid

    if old_links is not None:
        for r in old_links.to_dict("records"):
            f0, f1 = int(r["frame_from"]), int(r["frame_to"])
            frm = old_id_at.get((f0, int(r["label_from"])))
            if frm is None or pd.isna(r["track_id"]):
                continue
            a, b = nm(f0, frm), nm(f1, r["track_id"])
            ok = (a, b) in heldset if r["kind"] == "divide" else a == b
            if ok:
                r = dict(r)
                r["track_id"] = b
                rows.append(r)
    for e in manual_events:
        if e["kind"] == "continue":
            rows.append(
                {"frame_from": e["frame_from"], "label_from": e["label_from"],
                 "frame_to": e["frame_to"], "label_to": e["label_to"], "kind": "manual",
                 "cost": None, "topology_score": None, "track_id": e["track_id"]}
            )
    return pd.DataFrame(rows, columns=cols)


def _mark_divisions(rgb, tracks, held):
    """3 px magenta dots: mother on her last frame, daughters on their first."""
    height, width = rgb.shape[1:3]
    cent = {
        (int(f), int(t)): (y, x)
        for f, t, y, x in zip(tracks["frame"], tracks["track_id"],
                              tracks["centroid_y"], tracks["centroid_x"])
    }
    for h in held:
        spots = [(h["frame_from"], h["parent"])] + [(h["frame_to"], d) for d in h["daughters"]]
        for f, tid in spots:
            if (f, tid) not in cent:
                continue
            cy, cx = np.rint(cent[(f, tid)]).astype(int)
            rgb[f, max(0, cy - 1):min(height, cy + 2), max(0, cx - 1):min(width, cx + 2)] = MAGENTA
    return rgb


def rebuild_tracking(edited, baseline, tables, *, seg_masks=None) -> Rebuilt:
    """All tracking outputs for ``edited`` (T,Y,X) given the originals.

    ``baseline`` is the original tracked_labels stack; ``tables`` has the keys
    ``tracks``, ``links``, ``events``, ``lineage`` (DataFrames or None) and
    ``qc`` (dict).
    """
    edited = np.asarray(edited)
    baseline = np.asarray(baseline)
    maps, ambiguous = id_maps(baseline, edited)
    pres = _presence(edited)
    frames_nz = [t for t, ids in enumerate(pres) if ids]

    tracks, local, old_id_at = _tracks_table(
        edited, pres, maps, tables.get("tracks"), seg_masks
    )
    old_lineage = tables.get("lineage")
    held: list[dict] = []
    dropped: list[str] = []
    lineage = None
    if old_lineage is not None:
        lineage, held, dropped = _lineage(old_lineage, maps, pres, frames_nz)
    events, manual = _events(
        tables.get("events"), maps, pres, frames_nz, held, local, old_id_at,
        lineage_mode=old_lineage is not None,
    )
    links = _links(tables.get("links"), maps, manual, held, old_id_at)

    rgb = render_tracked_rgb(edited)
    _mark_divisions(rgb, tracks, held)

    start, end = _start_end(pres)
    lengths = {i: sum(1 for ids in pres if i in ids) for i in start}
    hist: dict[str, int] = {}
    for v in lengths.values():
        hist[str(v)] = hist.get(str(v), 0) + 1
    changed = {o for m in maps for o in m}
    qc = dict(tables.get("qc") or {})
    n_div = int((events["kind"] == "divide").sum())
    qc.update(
        n_frames=len(frames_nz),
        n_tracks=len(start),
        full_length_tracks=sum(1 for v in lengths.values() if v == len(frames_nz)),
        n_appear=int((events["kind"] == "appear").sum()),
        n_disappear=int((events["kind"] == "disappear").sum()),
        n_divide=n_div,
        n_divide_flagged=int(events.loc[events["kind"] == "divide", "flagged"].astype(bool).sum()),
        track_length_histogram=dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
        cells_per_frame=[len(pres[t]) for t in frames_nz],
        manual_edit={
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "ids_changed": len(changed),
            "manual_events": len(manual),
            "dropped_divisions": dropped,
        },
    )
    return Rebuilt(
        tracked=edited, rgb=rgb, tracks=tracks, links=links, events=events,
        lineage=lineage, qc=qc, dropped_divisions=dropped, ambiguous=ambiguous,
        n_ids_changed=len(changed),
    )


def _read_csv(path: Path):
    return pd.read_csv(path) if Path(path).is_file() else None


def load_tables(out_dir: Path, stem: str) -> dict:
    """Existing tracking tables next to the labels (None where absent)."""
    out_dir = Path(out_dir)
    qc_path = tagged_path(out_dir, stem, "qc.json")
    return {
        "tracks": _read_csv(tagged_path(out_dir, stem, "tracks.csv")),
        "links": _read_csv(tagged_path(out_dir, stem, "links.csv")),
        "events": _read_csv(tagged_path(out_dir, stem, "track_events.csv")),
        "lineage": _read_csv(tagged_path(out_dir, stem, "lineage.csv")),
        "qc": json.loads(qc_path.read_text()) if qc_path.is_file() else {},
    }


def _replace(path: Path, writer) -> None:
    """Write to a temp file then rename, so hard links to the old file (the
    picker_ready folders link tracked_labels/rgb) keep the old contents."""
    tmp = path.with_name("._tmp_" + path.name)
    writer(tmp)
    os.replace(tmp, path)


def backup_files(out_dir: Path, stem: str) -> Path | None:
    """Copy the existing tracking files to ``previous_<timestamp>/``."""
    out_dir = Path(out_dir)
    existing = [tagged_path(out_dir, stem, r) for r in ROLES]
    existing = [p for p in existing if p.is_file()]
    if not existing:
        return None
    dest = out_dir / f"previous_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    dest.mkdir()
    for p in existing:
        shutil.copy2(p, dest / p.name)
    return dest


def write_tracking(out_dir: Path, stem: str, rebuilt: Rebuilt, cal=None, *, backup=True):
    """Write all files; returns (written paths, backup dir)."""
    out_dir = Path(out_dir)
    saved = backup_files(out_dir, stem) if backup else None
    written: list[Path] = []

    def put(role, writer):
        p = tagged_path(out_dir, stem, role)
        _replace(p, writer)
        written.append(p)

    put("tracked_labels.tif", lambda p: save_tiff(p, rebuilt.tracked.astype(np.uint32, copy=False), cal))
    put("tracked_rgb.tif", lambda p: save_tiff(p, rebuilt.rgb, cal))
    put("tracks.csv", lambda p: rebuilt.tracks.to_csv(p, index=False))
    if rebuilt.links is not None:
        put("links.csv", lambda p: rebuilt.links.to_csv(p, index=False))
    if rebuilt.events is not None:
        put("track_events.csv", lambda p: rebuilt.events.to_csv(p, index=False))
    if rebuilt.lineage is not None:
        put("lineage.csv", lambda p: rebuilt.lineage.to_csv(p, index=False))
    put("qc.json", lambda p: p.write_text(json.dumps(rebuilt.qc, indent=2) + "\n"))
    return written, saved
