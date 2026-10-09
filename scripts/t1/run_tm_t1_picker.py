#!/usr/bin/env python3
"""Qt picker: click detected T1s, then export length/angle traces.

Loads ``*_t1_events.csv`` from run_tm_t1.py (or the older t1_events.csv).
Shows cell outlines (from labels) with markers for 4-way events. Click to
select; Preview shows the export look for one T1 (filled quartet + outlines
+ orange/cyan/red); Confirm measures junction length (vertex-vertex chord)
and angle ±window frames around each selected T1.

Windows / macOS: a normal desktop session is enough (no DISPLAY).
Linux cluster: needs X11/VNC (DISPLAY).

Uses PySide6 or PyQt5. This file is standalone (no other .py files required).

If the three input files sit next to this script, you can omit paths:

  python run_tm_t1_picker.py --output-dir t1_picked --window 10

They may be either the old names (t1_events.csv, tracked_labels.tif,
tracked_rgb.tif) or ``{input_stem}_t1_events.csv`` /
``{input_stem}_tracked_labels.tif`` / ``{input_stem}_tracked_rgb.tif``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.segmentation import expand_labels


def _input_stem(path: Path) -> str:
    stem = path.stem
    for suf in ("_tracked_labels", "_tracked_rgb", "_t1_events"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    if stem in ("tracked_labels", "tracked_rgb", "t1_events"):
        return ""
    return stem


def _tagged_path(directory: Path, stem: str, role: str) -> Path:
    if stem:
        return directory / f"{stem}_{role}"
    return directory / role


_SCRIPTS = Path(__file__).resolve().parent
_ROLE_FILES = {
    "events": "t1_events.csv",
    "labels": "tracked_labels.tif",
    "rgb": "tracked_rgb.tif",
}


def unique_count_3x3(lab: np.ndarray) -> np.ndarray:
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
    mask: np.ndarray, expanded: np.ndarray
) -> list[dict]:
    """Connected vertex blobs -> {y, x, cell_ids} via vectorized neighborhood IDs."""
    labeled, n = ndi.label(mask)
    if n == 0:
        return []
    ys, xs = np.nonzero(labeled)
    if ys.size == 0:
        return []
    blob = labeled[ys, xs].astype(np.int64)
    cnt = np.bincount(blob, minlength=n + 1)[1:]
    cy = np.bincount(blob, weights=ys.astype(np.float64), minlength=n + 1)[1:] / cnt
    cx = np.bincount(blob, weights=xs.astype(np.float64), minlength=n + 1)[1:] / cnt

    h, w = expanded.shape
    mx = int(expanded.max()) + 1
    keys: list[np.ndarray] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            v = expanded[
                np.clip(ys + dy, 0, h - 1),
                np.clip(xs + dx, 0, w - 1),
            ]
            nz = v != 0
            if nz.any():
                keys.append(blob[nz] * mx + v[nz].astype(np.int64))
    if not keys:
        return [
            {"y": float(cy[i]), "x": float(cx[i]), "cell_ids": ()}
            for i in range(n)
            if cnt[i] > 0
        ]
    uniq = np.unique(np.concatenate(keys))
    owner, cell = np.divmod(uniq, mx)
    starts = np.searchsorted(owner, np.arange(1, n + 1), side="left")
    stops = np.searchsorted(owner, np.arange(1, n + 1), side="right")
    rows: list[dict] = []
    for i in range(n):
        if cnt[i] == 0:
            continue
        ids = tuple(int(c) for c in cell[starts[i] : stops[i]])
        rows.append({"y": float(cy[i]), "x": float(cx[i]), "cell_ids": ids})
    return rows


def frame_geometry(
    labels: np.ndarray, expand: int
) -> tuple[np.ndarray, list[dict]]:
    """Expanded labels + 3-cell vertices (4-ways excluded from the 3-way mask)."""
    expanded = expand_labels(labels, distance=expand) if expand > 0 else labels
    count = unique_count_3x3(expanded)
    junc = count >= 3
    four_mask = junc & (count >= 4)
    three_mask = junc & ~ndi.binary_dilation(four_mask, iterations=1)
    verts_3 = _component_records(three_mask, expanded)
    return expanded, verts_3


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


def vertex_bond(
    verts_3: list[dict], pair: frozenset[int], quartet: frozenset[int]
) -> tuple[float | None, dict | None, dict | None]:
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


def pair_outline_mask(labels: np.ndarray, a: int, b: int, reach: int) -> np.ndarray:
    """Facing rims of A and B, plus the gap between them."""
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


def _search_dirs() -> list[Path]:
    dirs: list[Path] = []
    for d in (Path.cwd(), _SCRIPTS):
        try:
            resolved = d.resolve()
        except OSError:
            resolved = d
        if resolved not in dirs:
            dirs.append(resolved)
    return dirs


def _uniq_files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved.is_file() and resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def discover_role(role: str, stem: str | None = None) -> Path | None:
    """Find t1_events.csv / tracked_labels.tif / tracked_rgb.tif or tagged twins."""
    name = _ROLE_FILES[role]
    found: list[Path] = []
    for folder in _search_dirs():
        exact = folder / name
        if exact.is_file():
            found.append(exact)
        if stem:
            tagged = folder / f"{stem}_{name}"
            if tagged.is_file():
                found.append(tagged)
        else:
            found.extend(folder.glob(f"*_{name}"))
    uniq = _uniq_files(found)
    if len(uniq) == 1:
        return uniq[0]
    if len(uniq) > 1:
        listing = "\n  ".join(str(p) for p in uniq)
        raise FileNotFoundError(
            f"Multiple {name} files found. Pass --{role} explicitly:\n  {listing}"
        )
    return None


def resolve_input(path: Path) -> Path:
    """Find a file in cwd, next to this script, or by filename / tagged glob."""
    raw = Path(path).expanduser()
    name = raw.name
    tried: list[Path] = []
    for cand in (
        raw,
        Path.cwd() / raw,
        _SCRIPTS / raw,
        Path.cwd() / name,
        _SCRIPTS / name,
    ):
        try:
            resolved = cand.resolve()
        except OSError:
            resolved = cand
        if resolved not in tried:
            tried.append(resolved)
        if resolved.is_file():
            return resolved
    for role, role_name in _ROLE_FILES.items():
        if name == role_name:
            hit = discover_role(role)
            if hit is not None:
                return hit
    here = Path.cwd()
    script_files = sorted(p.name for p in _SCRIPTS.iterdir() if p.is_file())[:40]
    cwd_files = sorted(p.name for p in here.iterdir() if p.is_file())[:40]
    raise FileNotFoundError(
        f"Could not find '{path}'.\n"
        f"Looked at:\n  "
        + "\n  ".join(str(p) for p in tried)
        + f"\nCurrent folder: {here}\n"
        f"Script folder: {_SCRIPTS}\n"
        f"Files in current folder: {cwd_files or '(none)'}\n"
        f"Files next to the .py: {script_files or '(none)'}\n"
        "Put the T1 CSV and tracked TIFFs in the SAME folder as "
        "run_tm_t1_picker.py. Names may be generic (t1_events.csv, "
        "tracked_labels.tif, tracked_rgb.tif) or tagged "
        "({input_stem}_t1_events.csv, {input_stem}_tracked_labels.tif, "
        "{input_stem}_tracked_rgb.tif). Then:\n"
        "  python run_tm_t1_picker.py --output-dir t1_picked --window 10"
    )


def resolve_picker_inputs(
    events: Path | None,
    labels: Path | None,
    rgb: Path | None,
) -> tuple[Path, Path, Path]:
    events_path = resolve_input(events) if events is not None else discover_role("events")
    stem = _input_stem(events_path) if events_path is not None else None
    labels_path = (
        resolve_input(labels)
        if labels is not None
        else discover_role("labels", stem)
    )
    if labels_path is not None:
        stem = _input_stem(labels_path) or stem
    rgb_path = (
        resolve_input(rgb) if rgb is not None else discover_role("rgb", stem)
    )
    missing: list[str] = []
    if events_path is None:
        missing.append("*_t1_events.csv or t1_events.csv")
    if labels_path is None:
        missing.append("*_tracked_labels.tif or tracked_labels.tif")
    if rgb_path is None:
        missing.append("*_tracked_rgb.tif or tracked_rgb.tif")
    if missing:
        raise FileNotFoundError(
            "Could not auto-detect picker inputs: "
            + ", ".join(missing)
            + ".\nPass --events / --labels / --rgb, or put those files "
            "next to run_tm_t1_picker.py."
        )
    return events_path, labels_path, rgb_path


def import_qt():
    """PySide6 (preferred) or PyQt5."""
    try:
        from PySide6.QtCore import Qt, QPointF
        from PySide6.QtGui import QColor, QImage, QIntValidator, QPainter, QPen, QPixmap
        from PySide6.QtWidgets import (
            QAbstractItemView,
            QApplication,
            QComboBox,
            QDialog,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSlider,
            QVBoxLayout,
            QWidget,
        )

        return {
            "Qt": Qt,
            "QPointF": QPointF,
            "QColor": QColor,
            "QImage": QImage,
            "QIntValidator": QIntValidator,
            "QPainter": QPainter,
            "QPen": QPen,
            "QPixmap": QPixmap,
            "QAbstractItemView": QAbstractItemView,
            "QApplication": QApplication,
            "QComboBox": QComboBox,
            "QDialog": QDialog,
            "QHBoxLayout": QHBoxLayout,
            "QLabel": QLabel,
            "QLineEdit": QLineEdit,
            "QListWidget": QListWidget,
            "QListWidgetItem": QListWidgetItem,
            "QMainWindow": QMainWindow,
            "QMessageBox": QMessageBox,
            "QPushButton": QPushButton,
            "QSlider": QSlider,
            "QVBoxLayout": QVBoxLayout,
            "QWidget": QWidget,
        }
    except ImportError:
        pass
    try:
        from PyQt5.QtCore import Qt, QPointF
        from PyQt5.QtGui import QColor, QImage, QIntValidator, QPainter, QPen, QPixmap
        from PyQt5.QtWidgets import (
            QAbstractItemView,
            QApplication,
            QComboBox,
            QDialog,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSlider,
            QVBoxLayout,
            QWidget,
        )

        return {
            "Qt": Qt,
            "QPointF": QPointF,
            "QColor": QColor,
            "QImage": QImage,
            "QIntValidator": QIntValidator,
            "QPainter": QPainter,
            "QPen": QPen,
            "QPixmap": QPixmap,
            "QAbstractItemView": QAbstractItemView,
            "QApplication": QApplication,
            "QComboBox": QComboBox,
            "QDialog": QDialog,
            "QHBoxLayout": QHBoxLayout,
            "QLabel": QLabel,
            "QLineEdit": QLineEdit,
            "QListWidget": QListWidget,
            "QListWidgetItem": QListWidgetItem,
            "QMainWindow": QMainWindow,
            "QMessageBox": QMessageBox,
            "QPushButton": QPushButton,
            "QSlider": QSlider,
            "QVBoxLayout": QVBoxLayout,
            "QWidget": QWidget,
        }
    except ImportError as exc:
        raise RuntimeError(
            "Need PySide6 or PyQt5.  pip install PySide6"
        ) from exc

# Marker / selection colours (RGB for QColor / overlay)
CYAN = (0, 180, 220)
YELLOW = (255, 220, 0)
WHITE = (255, 255, 255)
OUTLINE = (90, 90, 90)  # dark gray — stays readable over pastel fills
# Preview junctions: saturated brights on muted pastel cell fills.
PREVIEW_BREAKING = (255, 230, 0)  # hot yellow — shrinking AB
PREVIEW_BUILDING = (0, 255, 255)  # electric cyan — growing CD
PREVIEW_FOURWAY = (255, 0, 255)  # magenta — 4-way vertex
# Export junctions: classic grayscale-overlay palette (legacy names).
EXPORT_BREAKING = (255, 140, 0)  # orange — shrinking AB
EXPORT_BUILDING = (0, 220, 255)  # cyan — growing CD
EXPORT_FOURWAY = (0, 180, 220)  # cyan vertex (completed_4way)
LENGTH_CHORD = (255, 0, 0)
LENGTH_CHORD_ALPHA = 0.5
MARKER_R = 6
CLICK_R = 12.0

# Type-colored markers / junctions (matches scripts/t1/run_tm_t1.py).
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
}


def event_type(kind: str, has_fourway: int | bool, n_swaps: int) -> str:
    """Classify a T1 row (flicker beats aborted; no_4way only if never 4-way)."""
    if int(n_swaps) >= 2:
        return "flicker"
    if kind == "aborted":
        return "aborted"
    if int(has_fourway) == 0:
        return "no_4way"
    return "completed_4way"


def type_colors(
    kind: str, has_fourway: int | bool, n_swaps: int
) -> dict[str, tuple[int, int, int]]:
    return TYPE_COLORS[event_type(kind, has_fourway, n_swaps)]
# Soft pastels for quartet fills (avoid yellow / cyan / magenta clash).
QUARTET_PALETTE = (
    (170, 130, 190),  # lavender
    (130, 175, 155),  # sage
    (130, 155, 195),  # periwinkle
    (195, 150, 140),  # dusty rose
)


def _opt_int(row: dict, key: str, default: int) -> int:
    raw = row.get(key)
    if raw in (None, ""):
        return default
    return int(raw)


@dataclass
class T1Event:
    event_id: int
    frame_start: int
    frame_end: int
    frame_first_swap: int
    frame_last_swap: int
    frame_before: int
    frame_after: int
    A: int
    B: int
    C: int
    D: int
    pair_breaking: frozenset[int]
    pair_building: frozenset[int]
    y: float
    x: float
    source: str = "tm"
    kind: str = "completed"
    persist_pre: int = -1
    persist_post: int = -1
    n_swaps: int = 0
    has_fourway: int = 1

    @property
    def quartet(self) -> frozenset[int]:
        return frozenset((self.A, self.B, self.C, self.D))

    def etype(self) -> str:
        return event_type(self.kind, self.has_fourway, self.n_swaps)

    def colors(self) -> dict[str, tuple[int, int, int]]:
        return type_colors(self.kind, self.has_fourway, self.n_swaps)

    def label(self) -> str:
        br = ";".join(str(i) for i in sorted(self.pair_breaking))
        bu = ";".join(str(i) for i in sorted(self.pair_building))
        return (
            f"#{self.event_id}  [{self.etype()}]  "
            f"f{self.frame_start}-{self.frame_end}  "
            f"break {br} → build {bu}"
        )


def parse_pair(s: str) -> frozenset[int]:
    return frozenset(int(x) for x in s.split(";") if x.strip())


def load_events(path: Path) -> list[T1Event]:
    rows: list[T1Event] = []
    with path.open(newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            rows.append(
                T1Event(
                    event_id=i,
                    frame_start=int(row["frame_start"]),
                    frame_end=int(row["frame_end"]),
                    frame_first_swap=_opt_int(
                        row, "frame_first_swap", int(row["frame_start"])
                    ),
                    frame_last_swap=_opt_int(
                        row, "frame_last_swap", int(row["frame_end"])
                    ),
                    frame_before=int(row["frame_before"]),
                    frame_after=int(row["frame_after"]),
                    A=int(row["A"]),
                    B=int(row["B"]),
                    C=int(row["C"]),
                    D=int(row["D"]),
                    pair_breaking=parse_pair(row["pair_breaking"]),
                    pair_building=parse_pair(row["pair_building"]),
                    y=float(row["y"]),
                    x=float(row["x"]),
                    source=str(row.get("source") or "tm"),
                    kind=str(row.get("kind") or "completed"),
                    persist_pre=int(row["persist_pre"])
                    if row.get("persist_pre") not in (None, "")
                    else -1,
                    persist_post=int(row["persist_post"])
                    if row.get("persist_post") not in (None, "")
                    else -1,
                    n_swaps=int(row["n_swaps"]) if row.get("n_swaps") not in (None, "") else 0,
                    has_fourway=int(row["has_fourway"])
                    if row.get("has_fourway") not in (None, "")
                    else (0 if str(row.get("source") or "") == "cellflow" else 1),
                )
            )
    return rows


# Export / select-all order: type groups, then time within each type.
TYPE_SORT_ORDER = {
    "completed_4way": 0,
    "flicker": 1,
    "aborted": 2,
    "no_4way": 3,
}


def sort_events_by_type(events: list[T1Event]) -> list[T1Event]:
    """Order by type (completed_4way → flicker → aborted → no_4way), then frame."""
    return sorted(
        events,
        key=lambda e: (
            TYPE_SORT_ORDER.get(e.etype(), 99),
            int(e.frame_start),
            int(e.frame_end),
            int(e.event_id),
        ),
    )


def event_passes_filter(ev: T1Event, mode: str) -> bool:
    if mode == "all_kinds":
        return True
    if mode == "all":
        return ev.kind == "completed"
    if mode.startswith("persist_"):
        n = int(mode.split("_", 1)[1])
        if ev.persist_pre < 0 or ev.persist_post < 0:
            return False
        return (
            ev.kind == "completed"
            and ev.persist_pre >= n
            and ev.persist_post >= n
        )
    if mode == "flicker":
        return ev.kind == "completed" and ev.n_swaps >= 2
    if mode == "aborted":
        return ev.kind == "aborted"
    if mode == "no_4way":
        return ev.kind == "completed" and int(ev.has_fourway) == 0
    return ev.kind == "completed"


def load_rgb_stack(path: Path) -> np.ndarray:
    rgb = np.asarray(tifffile.imread(path))
    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        rgb = rgb[np.newaxis, ...]
    if rgb.ndim == 4 and rgb.shape[1] == 3 and rgb.shape[-1] != 3:
        rgb = np.moveaxis(rgb, 1, -1)
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected T,Y,X,3 RGB; got {rgb.shape}")
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def load_labels_stack(path: Path) -> np.ndarray:
    img = np.asarray(tifffile.imread(path))
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    if img.ndim != 3:
        raise ValueError(f"Expected T,Y,X labels; got {img.shape}")
    return img.astype(np.uint32, copy=False)


def undirected_angle_deg(dx: float, dy: float) -> float | None:
    """Undirected angle from image horizontal in [0, 180).

    0° = parallel to x (bottom edge), 90° = vertical. Negatives and
    θ+180° flips collapse via modulo 180.
    """
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    # atan2 with y-down image: negate dy so 0° = +x, 90° = up
    a = float(np.degrees(np.arctan2(-dy, dx))) % 180.0
    return a


def angle_from_points(
    p1: tuple[float, float] | None, p2: tuple[float, float] | None
) -> float | None:
    if p1 is None or p2 is None:
        return None
    dy = p2[0] - p1[0]
    dx = p2[1] - p1[1]
    return undirected_angle_deg(dx, dy)


def angle_from_mask(mask: np.ndarray) -> float | None:
    ys, xs = np.where(mask)
    if ys.size < 3:
        return None
    pts = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    pts -= pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    vx, vy = float(vt[0, 0]), float(vt[0, 1])
    return undirected_angle_deg(vx, vy)


FLAG_FOURWAY = "4way"
FLAG_NV = "nv"
FLAG_NO_ABCD = "no AB/CD contact"


def length_blank_flag_cmap(
    cmap: dict[tuple[int, int], float],
    ab: frozenset[int],
    cd: frozenset[int],
) -> str:
    a, b = tuple(ab)
    c, d = tuple(cd)
    if lookup_contact(cmap, a, b) > 0 and lookup_contact(cmap, c, d) > 0:
        return FLAG_FOURWAY
    return FLAG_NO_ABCD


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


def length_t0(ev: T1Event) -> int:
    """dt=0: first neighbour swap if it precedes the first 4-way, else the 4-way."""
    if ev.frame_first_swap < ev.frame_start:
        return ev.frame_first_swap
    return ev.frame_start


def length_trace_span(ev: T1Event, n: int, window: int) -> tuple[int, int]:
    """±window around length_t0, stretched to last AB↔CD neighbour swap."""
    t_anchor = length_t0(ev)
    t0 = max(0, t_anchor - window)
    t1 = min(n - 1, ev.frame_last_swap + window)
    return t0, t1


def measure_pair_frame(
    labels: np.ndarray,
    verts_3: list[dict],
    pair: frozenset[int],
    quartet: frozenset[int],
    expand: int,
) -> tuple[float | None, float | None, dict | None, dict | None]:
    """Vertex-vertex chord length and angle for one pair on one frame.

    Length is Euclidean distance between the two 3-cell endpoints of the bond.
    Blank when fewer than two endpoints are found. Angle uses those endpoints
    when available, else the principal axis of the facing interface mask.
    """
    lv, e1, e2 = vertex_bond(verts_3, pair, quartet)
    angle = None
    if e1 is not None and e2 is not None:
        angle = angle_from_points((e1["y"], e1["x"]), (e2["y"], e2["x"]))
    if angle is None:
        a, b = tuple(pair)
        mask = pair_outline_mask(labels, a, b, expand)
        angle = angle_from_mask(mask)
    if lv is None or e1 is None or e2 is None:
        return None, angle, e1, e2
    return float(lv), angle, e1, e2


def measure_selection(
    events: list[T1Event],
    labels: np.ndarray,
    *,
    window: int,
    expand: int,
) -> list[dict]:
    """Length/angle traces: geometry once per frame, vertex-vertex length."""
    n = labels.shape[0]
    plan: dict[int, list[T1Event]] = defaultdict(list)
    for ev in events:
        t0, t1 = length_trace_span(ev, n, window)
        for t in range(t0, t1 + 1):
            plan[t].append(ev)

    frames = sorted(plan)
    rows: list[dict] = []
    for i, t in enumerate(frames):
        print(
            f"  measured frame {t} ({i + 1}/{len(frames)})",
            flush=True,
        )
        expanded, verts_3 = frame_geometry(labels[t], expand)
        cmap = contact_length_map(expanded)
        for ev in plan[t]:
            pair, sign = pair_sign(
                cmap, ev.pair_breaking, ev.pair_building
            )
            if sign == 0 or pair is None:
                flag = length_blank_flag_cmap(
                    cmap, ev.pair_breaking, ev.pair_building
                )
                rows.append(
                    {
                        "t1_id": ev.event_id,
                        "frame": t,
                        "dt": t - length_t0(ev),
                        "length_px": flag,
                        "length_signed_px": flag,
                        "length_contact_px": flag,
                        "length_contact_signed_px": flag,
                        "angle_deg": "",
                        "y1": "",
                        "x1": "",
                        "y2": "",
                        "x2": "",
                    }
                )
                continue
            length, angle, e1, e2 = measure_pair_frame(
                labels[t], verts_3, pair, ev.quartet, expand
            )
            a, b = tuple(pair)
            contact = lookup_contact(cmap, a, b)
            mag_v: object = FLAG_NV if length is None else round(length, 2)
            mag_c: object = (
                round(contact, 2) if contact > 0 else FLAG_NV
            )
            signed_v = (
                mag_v
                if isinstance(mag_v, str)
                else round(float(mag_v) * sign, 2)
            )
            signed_c = (
                mag_c
                if isinstance(mag_c, str)
                else round(float(mag_c) * sign, 2)
            )
            rows.append(
                {
                    "t1_id": ev.event_id,
                    "frame": t,
                    "dt": t - length_t0(ev),
                    "length_px": mag_v,
                    "length_signed_px": signed_v,
                    "length_contact_px": mag_c,
                    "length_contact_signed_px": signed_c,
                    "angle_deg": "" if angle is None else round(angle, 2),
                    "y1": "" if e1 is None or e2 is None else float(e1["y"]),
                    "x1": "" if e1 is None or e2 is None else float(e1["x"]),
                    "y2": "" if e1 is None or e2 is None else float(e2["y"]),
                    "x2": "" if e1 is None or e2 is None else float(e2["x"]),
                }
            )
    rows.sort(key=lambda r: (r["t1_id"], r["frame"]))
    return rows


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


def _pair_dash(ids) -> str:
    return "-".join(str(i) for i in sorted(ids))


def write_traces_wide_csv(
    out_dir: Path,
    selected: list[T1Event],
    traces: list[dict],
    stem: str,
) -> tuple[Path, Path]:
    """Write length and angle as wide CSVs: event_id/type headers, dt first col."""
    ids = [ev.event_id for ev in selected]
    types = [ev.etype() for ev in selected]
    dts = sorted({int(r["dt"]) for r in traces})

    len_map: dict[tuple[int, int], object] = {}
    len_signed_map: dict[tuple[int, int], object] = {}
    contact_map: dict[tuple[int, int], object] = {}
    contact_signed_map: dict[tuple[int, int], object] = {}
    ang_map: dict[tuple[int, int], object] = {}
    for r in traces:
        key = (int(r["t1_id"]), int(r["dt"]))
        len_map[key] = r["length_px"]
        len_signed_map[key] = r.get("length_signed_px", "")
        contact_map[key] = r.get("length_contact_px", "")
        contact_signed_map[key] = r.get("length_contact_signed_px", "")
        ang_map[key] = r["angle_deg"]

    def _write_dt_first(
        path: Path, value_map: dict[tuple[int, int], object]
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            handle.write("sep=;\n")
            handle.write(";".join(["event_id", *[str(eid) for eid in ids]]) + "\n")
            handle.write(";".join(["event_type", *types]) + "\n")
            for dt in dts:
                cells = [str(dt)]
                for eid in ids:
                    cells.append(_de_num(value_map.get((eid, dt), "")))
                handle.write(";".join(cells) + "\n")

    len_path = _tagged_path(out_dir, stem, "selected_traces_length.csv")
    len_signed_path = _tagged_path(
        out_dir, stem, "selected_traces_length_signed.csv"
    )
    contact_path = _tagged_path(out_dir, stem, "selected_traces_contact.csv")
    contact_signed_path = _tagged_path(
        out_dir, stem, "selected_traces_contact_signed.csv"
    )
    ang_path = _tagged_path(out_dir, stem, "selected_traces_angle.csv")
    _write_dt_first(len_path, len_map)
    _write_dt_first(len_signed_path, len_signed_map)
    _write_dt_first(contact_path, contact_map)
    _write_dt_first(contact_signed_path, contact_signed_map)
    _write_dt_first(ang_path, ang_map)
    return len_path, ang_path


def cell_outline_mask(labels: np.ndarray) -> np.ndarray:
    """True where a cell pixel touches a different label (4-neighbour)."""
    lab = labels
    edge = np.zeros(lab.shape, dtype=bool)
    edge[:, :-1] |= (lab[:, :-1] != lab[:, 1:]) & (lab[:, :-1] > 0)
    edge[:, 1:] |= (lab[:, 1:] != lab[:, :-1]) & (lab[:, 1:] > 0)
    edge[:-1, :] |= (lab[:-1, :] != lab[1:, :]) & (lab[:-1, :] > 0)
    edge[1:, :] |= (lab[1:, :] != lab[:-1, :]) & (lab[1:, :] > 0)
    return edge


def outline_rgb_frame(labels: np.ndarray) -> np.ndarray:
    """Black background with gray cell outlines."""
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[cell_outline_mask(labels)] = OUTLINE
    return out


def quartet_colors(quartet: frozenset[int]) -> dict[int, tuple[int, int, int]]:
    """Stable distinct colors for the four cells of a T1 quartet."""
    cells = sorted(quartet)
    return {cid: QUARTET_PALETTE[i % len(QUARTET_PALETTE)] for i, cid in enumerate(cells)}


def fill_cells(
    dst: np.ndarray,
    labels: np.ndarray,
    colors: dict[int, tuple[int, int, int]],
) -> None:
    for cid, color in colors.items():
        mask = labels == cid
        if mask.any():
            dst[mask] = color


def paint_event_junctions(
    dst: np.ndarray,
    labels: np.ndarray,
    ev: T1Event,
    t: int,
    *,
    expand: int,
    breaking: tuple[int, int, int] | None = None,
    building: tuple[int, int, int] | None = None,
    fourway: tuple[int, int, int] | None = None,
) -> None:
    """Paint AB / CD / vertex for one event on frame t (type colors by default)."""
    from skimage.draw import disk

    colors = ev.colors()
    if breaking is None:
        breaking = colors["breaking"]
    if building is None:
        building = colors["building"]
    if fourway is None:
        fourway = colors["vertex"]
    ab = ev.pair_breaking
    cd = ev.pair_building
    t0, t1 = ev.frame_start, ev.frame_end
    h, w = labels.shape
    if t < t0:
        paint_pair(dst, labels, ab, breaking, expand)
    elif t > t1:
        paint_pair(dst, labels, cd, building, expand)
    else:
        paint_pair(dst, labels, ab, breaking, expand)
        paint_pair(dst, labels, cd, building, expand)
        if ev.y >= 0 and ev.x >= 0:
            rr, cc = disk(
                (int(round(ev.y)), int(round(ev.x))), 2, shape=(h, w)
            )
            dst[rr, cc] = fourway


def render_event_preview_frame(
    labels: np.ndarray,
    ev: T1Event,
    t: int,
    *,
    expand: int,
) -> np.ndarray:
    """Preview frame: outlines + filled quartet + yellow/cyan/magenta junctions."""
    out = outline_rgb_frame(labels)
    fill_cells(out, labels, quartet_colors(ev.quartet))
    # Re-draw outlines on top of fills so boundaries stay visible.
    out[cell_outline_mask(labels)] = OUTLINE
    paint_event_junctions(out, labels, ev, t, expand=expand)
    return out


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


def render_selected_overlay(
    rgb: np.ndarray,
    labels: np.ndarray,
    selected: list[T1Event],
    *,
    window: int,
    expand: int,
) -> np.ndarray:
    """Grayscale movie; type-colored junction outlines over ±window."""
    out = grayscale_rgb(rgb)
    n = out.shape[0]
    for ev in selected:
        t0, t1 = ev.frame_start, ev.frame_end
        win0 = max(0, t0 - window)
        win1 = min(n - 1, t1 + window)
        for t in range(win0, win1 + 1):
            paint_event_junctions(
                out[t],
                labels[t],
                ev,
                t,
                expand=expand,
            )
    return out


def blend_red_chord(
    dst: np.ndarray,
    e1: dict,
    e2: dict,
    *,
    alpha: float = LENGTH_CHORD_ALPHA,
) -> None:
    """Blend the vertex-vertex chord onto dst (same points as length_px)."""
    from skimage.draw import disk, line

    h, w = dst.shape[:2]
    y1, x1 = int(round(float(e1["y"]))), int(round(float(e1["x"])))
    y2, x2 = int(round(float(e2["y"]))), int(round(float(e2["x"])))
    mask = np.zeros((h, w), dtype=bool)
    rr, cc = line(y1, x1, y2, x2)
    ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    mask[rr[ok], cc[ok]] = True
    mask = ndi.binary_dilation(mask, iterations=1)
    for ey, ex in ((y1, x1), (y2, x2)):
        rr, cc = disk((ey, ex), 2, shape=(h, w))
        mask[rr, cc] = True
    if not mask.any():
        return
    color = np.asarray(LENGTH_CHORD, dtype=np.float32)
    pix = dst[mask].astype(np.float32)
    dst[mask] = np.clip((1.0 - alpha) * pix + alpha * color, 0, 255).astype(
        np.uint8
    )


def render_length_overlay(rgb: np.ndarray, traces: list[dict]) -> np.ndarray:
    """Grayscale movie with the measured length chords (semi-transparent red)."""
    out = grayscale_rgb(rgb)
    for row in traces:
        if row["y1"] == "" or row["y2"] == "":
            continue
        t = int(row["frame"])
        if t < 0 or t >= out.shape[0]:
            continue
        blend_red_chord(
            out[t],
            {"y": row["y1"], "x": row["x1"]},
            {"y": row["y2"], "x": row["x2"]},
        )
    return out


def _overwrite_tif(path: Path, stack: np.ndarray) -> None:
    if path.exists():
        try:
            path.unlink()
        except OSError as exc:
            raise OSError(
                f"Cannot overwrite {path} (file in use?). "
                "Close it in ImageJ/Fiji or another viewer, then Confirm again."
            ) from exc
    tifffile.imwrite(path, stack)


def export_selection(
    selected: list[T1Event],
    labels: np.ndarray,
    rgb: np.ndarray,
    out_dir: Path,
    *,
    window: int,
    expand: int,
    stem: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = sort_events_by_type(list(selected))
    sel_path = _tagged_path(out_dir, stem, "selected_t1s.csv")
    fields = [
        "event_id",
        "type",
        "kind",
        "source",
        "has_fourway",
        "n_swaps",
        "persist_pre",
        "persist_post",
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
    ]
    with sel_path.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for ev in selected:
            w.writerow(
                {
                    "event_id": ev.event_id,
                    "type": ev.etype(),
                    "kind": ev.kind,
                    "source": ev.source,
                    "has_fourway": int(ev.has_fourway),
                    "n_swaps": ev.n_swaps,
                    "persist_pre": ev.persist_pre,
                    "persist_post": ev.persist_post,
                    "frame_start": ev.frame_start,
                    "frame_end": ev.frame_end,
                    "frame_first_swap": ev.frame_first_swap,
                    "frame_last_swap": ev.frame_last_swap,
                    "frame_before": ev.frame_before,
                    "frame_after": ev.frame_after,
                    "A": ev.A,
                    "B": ev.B,
                    "C": ev.C,
                    "D": ev.D,
                    "pair_breaking": ";".join(
                        str(i) for i in sorted(ev.pair_breaking)
                    ),
                    "pair_building": ";".join(
                        str(i) for i in sorted(ev.pair_building)
                    ),
                    "y": ev.y,
                    "x": ev.x,
                }
            )
    print(f"Wrote {sel_path}", flush=True)

    print(
        f"Measuring {len(selected)} event(s), window=±{window} ...",
        flush=True,
    )
    traces = measure_selection(
        selected,
        labels,
        window=window,
        expand=expand,
    )
    len_path, ang_path = write_traces_wide_csv(
        out_dir, selected, traces, stem
    )
    n_dt = len({int(r["dt"]) for r in traces}) if traces else 0
    print(
        f"Wrote {len_path} and {ang_path} "
        f"({len(selected)} T1 columns, {n_dt} dt rows)",
        flush=True,
    )

    print("Writing selected_overlay.tif ...", flush=True)
    overlay = render_selected_overlay(
        rgb, labels, selected, window=window, expand=expand
    )
    ov_path = _tagged_path(out_dir, stem, "selected_overlay.tif")
    _overwrite_tif(ov_path, overlay)
    print(f"Wrote {ov_path}", flush=True)

    print("Writing selected_length.tif (measurement chords) ...", flush=True)
    length_ov = render_length_overlay(rgb, traces)
    len_ov_path = _tagged_path(out_dir, stem, "selected_length.tif")
    _overwrite_tif(len_ov_path, length_ov)
    print(f"Wrote {len_ov_path}", flush=True)

    qc = {
        "n_selected": len(selected),
        "n_trace_long_rows": len(traces),
        "n_dt_rows": n_dt,
        "window": window,
        "expand": expand,
        "event_ids": [e.event_id for e in selected],
        "traces_length": str(len_path),
        "traces_angle": str(ang_path),
        "overlay": str(ov_path),
        "length_overlay": str(len_ov_path),
        "overlay_colors": {
            "base": "grayscale tracked_rgb",
            "completed_4way": "cyan vertex; orange break / cyan build",
            "flicker": "yellow vertex; yellow-orange break / lime build",
            "aborted": "magenta vertex; magenta break / pink build",
            "no_4way": "amber vertex; amber break / teal build",
        },
        "length_overlay_colors": {
            "base": "grayscale tracked_rgb",
            "red": (
                f"vertex-vertex length chord (alpha {LENGTH_CHORD_ALPHA})"
            ),
        },
    }
    ( _tagged_path(out_dir, stem, "qc.json") ).write_text(json.dumps(qc, indent=2) + "\n")
    print(f"Export done under {out_dir}", flush=True)


# ---------------------------------------------------------------------------
# Qt GUI
# ---------------------------------------------------------------------------


def run_gui(
    events: list[T1Event],
    labels: np.ndarray,
    rgb: np.ndarray,
    out_dir: Path,
    *,
    window: int,
    expand: int,
    stem: str,
) -> int:
    qt = import_qt()
    Qt = qt["Qt"]
    QPointF = qt["QPointF"]
    QColor = qt["QColor"]
    QImage = qt["QImage"]
    QPainter = qt["QPainter"]
    QPen = qt["QPen"]
    QPixmap = qt["QPixmap"]
    QApplication = qt["QApplication"]
    QDialog = qt["QDialog"]
    QHBoxLayout = qt["QHBoxLayout"]
    QLabel = qt["QLabel"]
    QLineEdit = qt["QLineEdit"]
    QIntValidator = qt["QIntValidator"]
    QListWidget = qt["QListWidget"]
    QListWidgetItem = qt["QListWidgetItem"]
    QMainWindow = qt["QMainWindow"]
    QMessageBox = qt["QMessageBox"]
    QPushButton = qt["QPushButton"]
    QSlider = qt["QSlider"]
    QVBoxLayout = qt["QVBoxLayout"]
    QWidget = qt["QWidget"]
    QAbstractItemView = qt["QAbstractItemView"]
    QComboBox = qt["QComboBox"]

    n_frames, h, w = labels.shape
    # Cache outline RGB so the frame slider stays responsive.
    outline_cache: list[np.ndarray | None] = [None] * n_frames

    def outline_frame(t: int) -> np.ndarray:
        cached = outline_cache[t]
        if cached is None:
            cached = outline_rgb_frame(labels[t])
            outline_cache[t] = cached
        return cached

    if sys.platform in ("win32", "darwin"):
        os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
        os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    class ImageView(QLabel):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumSize(640, 400)
            self.setAlignment(Qt.AlignCenter)
            self.setMouseTracking(True)
            self._frame = 0
            self._selected: set[int] = set()
            self._shown_ids: set[int] | None = None
            self._hover: int | None = None
            self._scale = 1.0
            self._ox = 0
            self._oy = 0
            self.on_toggle = None  # callback(event_id)

        def set_frame(self, t: int) -> None:
            self._frame = int(t)
            self._redraw()

        def set_selected(self, ids: set[int]) -> None:
            self._selected = set(ids)
            self._redraw()

        def set_shown(self, ids: set[int]) -> None:
            self._shown_ids = set(ids)
            self._redraw()

        def visible_events(self) -> list[T1Event]:
            # Markers only while the 4-way vertex is present.
            t = self._frame
            shown = self._shown_ids
            return [
                ev
                for ev in events
                if ev.frame_start <= t <= ev.frame_end
                and (shown is None or ev.event_id in shown)
            ]

        def _redraw(self) -> None:
            frame = outline_frame(self._frame).copy()
            vis = self.visible_events()
            for ev in vis:
                paint_event_junctions(
                    frame, labels[self._frame], ev, self._frame, expand=expand
                )
            buf = np.ascontiguousarray(frame)
            qimg = QImage(buf.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            painter = QPainter(qimg)
            painter.setRenderHint(QPainter.Antialiasing)
            for ev in vis:
                base = ev.colors()["vertex"]
                if ev.event_id in self._selected:
                    color = YELLOW
                    r = MARKER_R + 2
                elif ev.event_id == self._hover:
                    color = WHITE
                    r = MARKER_R + 1
                else:
                    color = base
                    r = MARKER_R
                # Type fill under a selection/hover ring.
                fill = QColor(base[0], base[1], base[2], 100)
                pen = QPen(QColor(*color))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.setBrush(fill)
                if ev.x >= 0 and ev.y >= 0:
                    painter.drawEllipse(
                        QPointF(ev.x, ev.y), float(r), float(r)
                    )
            painter.end()

            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self._scale = scaled.width() / w if w else 1.0
            self._ox = (self.width() - scaled.width()) // 2
            self._oy = (self.height() - scaled.height()) // 2
            self.setPixmap(scaled)

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            self._redraw()

        def _to_image(self, pos) -> tuple[float, float] | None:
            x = (pos.x() - self._ox) / self._scale
            y = (pos.y() - self._oy) / self._scale
            if x < 0 or y < 0 or x >= w or y >= h:
                return None
            return float(x), float(y)

        def _nearest(self, xy: tuple[float, float]) -> int | None:
            best = None
            best_d = CLICK_R
            for ev in self.visible_events():
                d = float(np.hypot(ev.x - xy[0], ev.y - xy[1]))
                if d < best_d:
                    best_d = d
                    best = ev.event_id
            return best

        def mouseMoveEvent(self, event) -> None:
            xy = self._to_image(event.position() if hasattr(event, "position") else event.pos())
            hover = self._nearest(xy) if xy else None
            if hover != self._hover:
                self._hover = hover
                self._redraw()

        def mousePressEvent(self, event) -> None:
            xy = self._to_image(event.position() if hasattr(event, "position") else event.pos())
            if xy is None:
                return
            eid = self._nearest(xy)
            if eid is None:
                return
            if event.button() == Qt.LeftButton and self.on_toggle:
                self.on_toggle(eid)
            elif event.button() == Qt.RightButton and self.on_toggle:
                # right-click: deselect if selected
                if eid in self._selected:
                    self.on_toggle(eid)

    class PreviewDialog(QDialog):
        def __init__(self, ev: T1Event, parent=None):
            super().__init__(parent)
            self.setWindowTitle(f"Preview T1 #{ev.event_id}")
            self._ev = ev
            t0, t1 = length_trace_span(ev, n_frames, window)
            self._t0 = t0
            self._t1 = t1

            layout = QVBoxLayout(self)
            self.view = QLabel()
            self.view.setMinimumSize(640, 400)
            self.view.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.view, stretch=1)

            self.slider = QSlider(Qt.Horizontal)
            self.slider.setMinimum(t0)
            self.slider.setMaximum(t1)
            self.slider.setValue(ev.frame_start)
            self.slider.valueChanged.connect(self._on_frame)
            layout.addWidget(self.slider)

            self.frame_lbl = QLabel()
            layout.addWidget(self.frame_lbl)

            hint = QLabel(
                "Filled = pastel quartet cells; gray = other outlines; "
                "yellow/cyan = breaking/building junction; magenta = 4-way. "
                "Preview only — Confirm exports grayscale + orange/cyan/red."
            )
            hint.setWordWrap(True)
            layout.addWidget(hint)

            btn_close = QPushButton("Close")
            btn_close.clicked.connect(self.accept)
            layout.addWidget(btn_close)

            self.resize(900, 700)
            self._on_frame(ev.frame_start)

        def _on_frame(self, t: int) -> None:
            t = int(t)
            self.frame_lbl.setText(
                f"frame {t}  (window {self._t0}–{self._t1}, "
                f"4-way {self._ev.frame_start}–{self._ev.frame_end})"
            )
            frame = render_event_preview_frame(
                labels[t], self._ev, t, expand=expand
            )
            buf = np.ascontiguousarray(frame)
            qimg = QImage(buf.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(
                self.view.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.view.setPixmap(scaled)

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            self._on_frame(self.slider.value())

    class Main(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("T1 junction picker")
            self._selected: dict[int, T1Event] = {}
            self._confirmed = False
            self._filter_mode = "all"

            central = QWidget()
            self.setCentralWidget(central)
            layout = QHBoxLayout(central)

            left = QVBoxLayout()
            self.view = ImageView()
            self.view.on_toggle = self.toggle
            left.addWidget(self.view, stretch=1)

            self.slider = QSlider(Qt.Horizontal)
            self.slider.setMinimum(0)
            self.slider.setMaximum(n_frames - 1)
            self.slider.setValue(0)
            self.slider.valueChanged.connect(self.on_frame)
            left.addWidget(self.slider)

            self.frame_lbl = QLabel("frame 0 / %d" % (n_frames - 1))
            left.addWidget(self.frame_lbl)
            hint = QLabel(
                "View = cell outlines + type-colored junctions. Markers on "
                "event frames (4-way span, or intercalation frame if no "
                "4-way). Colors: cyan=completed 4-way, yellow=flicker, "
                "magenta=aborted, amber=no 4-way. Left-click to "
                "select/deselect. Preview = pastel quartet. Confirm exports "
                "length/angle CSVs and overlays (±window)."
            )
            hint.setWordWrap(True)
            left.addWidget(hint)
            layout.addLayout(left, stretch=4)

            right = QVBoxLayout()
            right.addWidget(QLabel("T1 filter"))
            self.filter_box = QComboBox()
            for key, label in (
                ("all", "Completed (baseline)"),
                ("all_kinds", "All kinds"),
                ("persist_3", "Persist 3"),
                ("persist_5", "Persist 5"),
                ("persist_7", "Persist 7"),
                ("flicker", "Flicker"),
                ("aborted", "Aborted"),
                ("no_4way", "No 4-way"),
            ):
                self.filter_box.addItem(label, key)
            self.filter_box.currentIndexChanged.connect(self.on_filter)
            right.addWidget(self.filter_box)
            self.filter_lbl = QLabel("")
            self.filter_lbl.setWordWrap(True)
            right.addWidget(self.filter_lbl)
            right.addWidget(QLabel("Go to event_id"))
            goto_row = QHBoxLayout()
            self.goto_edit = QLineEdit()
            self.goto_edit.setValidator(QIntValidator(0, 2_000_000_000, self.goto_edit))
            self.goto_edit.setPlaceholderText("CSV #N")
            self.goto_edit.returnPressed.connect(self.goto_event)
            goto_row.addWidget(self.goto_edit, stretch=1)
            btn_goto = QPushButton("Go")
            btn_goto.clicked.connect(self.goto_event)
            goto_row.addWidget(btn_goto)
            right.addLayout(goto_row)
            right.addWidget(QLabel("Selected T1s"))
            self.list = QListWidget()
            self.list.setSelectionMode(QAbstractItemView.SingleSelection)
            right.addWidget(self.list, stretch=1)

            btn_preview = QPushButton("Preview")
            btn_preview.clicked.connect(self.preview)
            right.addWidget(btn_preview)

            btn_all = QPushButton("Select all")
            btn_all.clicked.connect(self.select_all_shown)
            right.addWidget(btn_all)

            btn_rm = QPushButton("Remove selected from list")
            btn_rm.clicked.connect(self.remove_listed)
            right.addWidget(btn_rm)

            btn_ok = QPushButton("Confirm & export")
            btn_ok.clicked.connect(self.confirm)
            right.addWidget(btn_ok)

            btn_cancel = QPushButton("Cancel")
            btn_cancel.clicked.connect(self.close)
            right.addWidget(btn_cancel)
            layout.addLayout(right, stretch=1)

            self.view.set_frame(0)
            self.on_filter()
            self.resize(1200, 700)

        def keyPressEvent(self, event) -> None:
            if event.key() == Qt.Key_Left:
                self.slider.setValue(max(0, self.slider.value() - 1))
            elif event.key() == Qt.Key_Right:
                self.slider.setValue(min(n_frames - 1, self.slider.value() + 1))
            else:
                super().keyPressEvent(event)

        def on_filter(self, *_args) -> None:
            mode = str(self.filter_box.currentData() or "all")
            shown = {e.event_id for e in events if event_passes_filter(e, mode)}
            self.view.set_shown(shown)
            shown_ev = [e for e in events if e.event_id in shown]
            n_completed = sum(1 for e in shown_ev if e.etype() == "completed_4way")
            n_flicker = sum(1 for e in shown_ev if e.etype() == "flicker")
            n_aborted = sum(1 for e in shown_ev if e.etype() == "aborted")
            n_no4 = sum(1 for e in shown_ev if e.etype() == "no_4way")
            self.filter_lbl.setText(
                f"Showing {len(shown)} of {len(events)}. "
                f"completed 4-way {n_completed}, flicker {n_flicker}, "
                f"aborted {n_aborted}, no 4-way {n_no4}."
            )
            dropped = [eid for eid in list(self._selected) if eid not in shown]
            if dropped:
                for eid in dropped:
                    del self._selected[eid]
                self.refresh_list()
                self.view.set_selected(set(self._selected))

        def on_frame(self, t: int) -> None:
            self.frame_lbl.setText(f"frame {t} / {n_frames - 1}")
            self.view.set_frame(t)

        def toggle(self, eid: int) -> None:
            shown = self.view._shown_ids
            if shown is not None and eid not in shown:
                return
            if eid in self._selected:
                del self._selected[eid]
            else:
                ev = next(e for e in events if e.event_id == eid)
                self._selected[eid] = ev
            self.refresh_list()
            self.view.set_selected(set(self._selected))

        def select_all_shown(self) -> None:
            mode = str(self.filter_box.currentData() or "all")
            for ev in events:
                if event_passes_filter(ev, mode):
                    self._selected[ev.event_id] = ev
            self.refresh_list()
            self.view.set_selected(set(self._selected))

        def refresh_list(self) -> None:
            self.list.clear()
            for ev in sort_events_by_type(list(self._selected.values())):
                item = QListWidgetItem(ev.label())
                item.setData(Qt.UserRole, ev.event_id)
                self.list.addItem(item)

        def _current_event(self) -> T1Event | None:
            item = self.list.currentItem()
            if item is not None:
                eid = int(item.data(Qt.UserRole))
                return self._selected.get(eid)
            if len(self._selected) == 1:
                return next(iter(self._selected.values()))
            return None

        def goto_event(self) -> None:
            raw = self.goto_edit.text().strip()
            if not raw:
                QMessageBox.warning(
                    self, "Go to event_id", "Type an event_id from the length CSV."
                )
                return
            try:
                eid = int(raw)
            except ValueError:
                QMessageBox.warning(
                    self, "Go to event_id", "event_id must be an integer."
                )
                return
            ev = next((e for e in events if e.event_id == eid), None)
            if ev is None:
                QMessageBox.warning(
                    self, "Go to event_id", f"Unknown event_id {eid}."
                )
                return
            mode = str(self.filter_box.currentData() or "all")
            if not event_passes_filter(ev, mode):
                idx = self.filter_box.findData("all_kinds")
                if idx >= 0:
                    self.filter_box.setCurrentIndex(idx)
            self._selected[eid] = ev
            self.refresh_list()
            self.view.set_selected(set(self._selected))
            for i in range(self.list.count()):
                item = self.list.item(i)
                if int(item.data(Qt.UserRole)) == eid:
                    self.list.setCurrentItem(item)
                    break
            self.slider.setValue(length_t0(ev))
            self.preview()

        def preview(self) -> None:
            ev = self._current_event()
            if ev is None:
                QMessageBox.warning(
                    self,
                    "Nothing to preview",
                    "Select a T1 in the list (or click one marker first).",
                )
                return
            dlg = PreviewDialog(ev, parent=self)
            dlg.exec() if hasattr(dlg, "exec") else dlg.exec_()

        def remove_listed(self) -> None:
            item = self.list.currentItem()
            if item is None:
                return
            eid = int(item.data(Qt.UserRole))
            if eid in self._selected:
                del self._selected[eid]
            self.refresh_list()
            self.view.set_selected(set(self._selected))

        def confirm(self) -> None:
            if not self._selected:
                QMessageBox.warning(
                    self, "Nothing selected", "Click one or more T1 markers first."
                )
                return
            selected = sort_events_by_type(list(self._selected.values()))
            self._confirmed = True
            self.hide()
            QApplication.processEvents()
            try:
                export_selection(
                    selected,
                    labels,
                    rgb,
                    out_dir,
                    window=window,
                    expand=expand,
                    stem=stem,
                )
                QMessageBox.information(
                    self,
                    "Done",
                    f"Exported {len(selected)} T1(s) to\n{out_dir}\n"
                    f"({stem + '_' if stem else ''}selected_traces_length.csv, "
                    f"{stem + '_' if stem else ''}selected_traces_angle.csv, "
                    f"{stem + '_' if stem else ''}selected_overlay.tif, "
                    f"{stem + '_' if stem else ''}selected_length.tif)",
                )
            except Exception as exc:
                QMessageBox.critical(self, "Export failed", str(exc))
                raise
            finally:
                self.close()

    app = QApplication.instance() or QApplication(sys.argv)
    win = Main()
    win.show()
    exec_fn = getattr(app, "exec", None) or getattr(app, "exec_")
    code = int(exec_fn())
    return 0 if win._confirmed else code


FILTER_CHOICES = (
    "all",
    "all_kinds",
    "persist_3",
    "persist_5",
    "persist_7",
    "flicker",
    "aborted",
    "no_4way",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Click T1s on cell outlines and export length/angle traces. "
            "Inputs may be generic (t1_events.csv, tracked_labels.tif, "
            "tracked_rgb.tif) or tagged ({stem}_t1_events.csv, …). "
            "If omitted, files are auto-detected next to this script or in cwd. "
            "Use --headless --select-all for a no-GUI export of every event "
            "matching --filter (includes overlay TIFFs)."
        )
    )
    p.add_argument(
        "--events",
        type=Path,
        default=None,
        help="t1_events.csv or {stem}_t1_events.csv (auto-detected if omitted).",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="tracked_labels.tif or {stem}_tracked_labels.tif.",
    )
    p.add_argument(
        "--rgb",
        type=Path,
        default=None,
        help="tracked_rgb.tif or {stem}_tracked_rgb.tif (grayscale overlay base).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("t1_picked"),
        help="Export folder (default: t1_picked).",
    )
    p.add_argument(
        "--window",
        type=int,
        default=10,
        help=(
            "Frames before first 4-way / first AB↔CD swap and after the last "
            "neighbour swap (default 10; flicker stretches the window)."
        ),
    )
    p.add_argument("--expand", type=int, default=3)
    p.add_argument(
        "--headless",
        action="store_true",
        help="No GUI; export without Qt/DISPLAY (implies --select-all).",
    )
    p.add_argument(
        "--select-all",
        action="store_true",
        help="Export every event matching --filter (required for --headless).",
    )
    p.add_argument(
        "--filter",
        choices=FILTER_CHOICES,
        default="all",
        help="Event filter for --select-all / --headless (default: all = completed).",
    )
    return p.parse_args()


def run_headless(
    events: list[T1Event],
    labels: np.ndarray,
    rgb: np.ndarray,
    out_dir: Path,
    *,
    window: int,
    expand: int,
    stem: str,
    filter_mode: str,
) -> int:
    selected = sort_events_by_type(
        [e for e in events if event_passes_filter(e, filter_mode)]
    )
    print(
        f"Headless select-all filter={filter_mode!r}: "
        f"{len(selected)} of {len(events)} events",
        flush=True,
    )
    if not selected:
        print("No events match filter; nothing to export.", flush=True)
        return 1
    export_selection(
        selected,
        labels,
        rgb,
        out_dir,
        window=window,
        expand=expand,
        stem=stem,
    )
    print(f"Done. Exported {len(selected)} T1(s) to {out_dir}", flush=True)
    return 0


def main() -> None:
    args = parse_args()
    headless = bool(args.headless) or bool(args.select_all)
    if bool(args.select_all) and not bool(args.headless):
        # Select-all export is headless-only (no GUI confirm step).
        headless = True

    # DISPLAY is required on Linux/X11 for the GUI only.
    if (
        not headless
        and sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
    ):
        raise RuntimeError(
            "No DISPLAY set. On Linux the T1 picker needs X11/VNC "
            "(echo $DISPLAY). Windows/macOS do not need this. "
            "Or use --headless --select-all."
        )

    events_path, labels_path, rgb_path = resolve_picker_inputs(
        args.events, args.labels, args.rgb
    )
    stem = _input_stem(labels_path) or _input_stem(events_path)

    print(f"Loading events {events_path}", flush=True)
    events = load_events(events_path)
    print(f"  {len(events)} T1 events", flush=True)

    print(f"Loading labels {labels_path}", flush=True)
    labels = load_labels_stack(labels_path)
    print(f"Loading RGB {rgb_path}", flush=True)
    rgb = load_rgb_stack(rgb_path)
    if rgb.shape[0] != labels.shape[0]:
        raise ValueError(
            f"RGB T={rgb.shape[0]} != labels T={labels.shape[0]}"
        )
    if rgb.shape[1:3] != labels.shape[1:3]:
        raise ValueError(
            f"RGB YX {rgb.shape[1:3]} != labels {labels.shape[1:3]}"
        )

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if headless:
        code = run_headless(
            events,
            labels,
            rgb,
            out,
            window=args.window,
            expand=args.expand,
            stem=stem,
            filter_mode=str(args.filter),
        )
        raise SystemExit(code)

    print("Launching picker (outlines; click T1s, Preview, Confirm)...", flush=True)
    code = run_gui(
        events,
        labels,
        rgb,
        out,
        window=args.window,
        expand=args.expand,
        stem=stem,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
