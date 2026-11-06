"""Analysis region: keep part of the embryo for T1 detection and cell properties.

The user picks cells that form a boundary *line* (on any frame). Every picked cell is a track ID, so the line can be
redrawn in every frame through the cells' centroids of that frame: it moves with the tissue. A line is a polyline
through the cells (ordered once along the line), extended straight to the image border at both ends. Each line splits
the frame into two sides; with several lines a cell or pixel has one *signature* (one side per line). The user then
clicks cells in the areas to keep (or to disregard); their signatures define the kept area:

    kept = signatures of the "keep" clicks minus those of the "drop" clicks
           (no "keep" click at all: every signature except the "drop" ones)

A cell counts as inside in a frame when its centroid in that frame has a kept signature. Nothing is cut from the
images or labels: T1 events and cell properties just use the cells inside, and exported tiffs show the kept area as a
yellow outline with the rest dimmed (`burn_region`).

Nothing in here depends on napari or Qt, so it can be tested on its own.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

YELLOW = (255, 255, 0)
TRACKING_REQUIRED = "Tracking is required before analysis cropping. Run the Tracking tab first."
REGION_FILE = "region.json"

Positions = list  # per frame: {track_id: (y, x)}


# ---- cell positions ---------------------------------------------------------------------------------------------
def has_tracking(stack) -> bool:
    """True when `stack` is a tracked (T,Y,X) label stack with at least one cell."""
    return stack is not None and getattr(stack, "ndim", 0) == 3 and int(np.max(stack)) > 0


def positions_from_stack(stack: np.ndarray) -> Positions:
    """Centroid (y, x) of every track ID in every frame of a tracked label stack."""
    out: Positions = []
    for frame in np.asarray(stack):
        ys, xs = np.nonzero(frame)
        if not len(ys):
            out.append({})
            continue
        ids, inv = np.unique(frame[ys, xs], return_inverse=True)
        n = np.bincount(inv)
        cy, cx = np.bincount(inv, weights=ys) / n, np.bincount(inv, weights=xs) / n
        out.append({int(i): (float(y), float(x)) for i, y, x in zip(ids, cy, cx)})
    return out


def positions_from_tracks(df, n_frames: int) -> Positions:
    """Same from a `<stem>_tracks.csv` table (columns frame, track_id, centroid_y, centroid_x)."""
    out: Positions = [dict() for _ in range(n_frames)]
    g = df.groupby(["frame", "track_id"])[["centroid_y", "centroid_x"]].mean()
    for (f, tid), row in g.iterrows():
        if 0 <= int(f) < n_frames:
            out[int(f)][int(tid)] = (float(row.centroid_y), float(row.centroid_x))
    return out


def parents_from_lineage(df) -> dict:
    """{daughter track: parent track} from a `<stem>_lineage.csv` table."""
    if df is None or "parent_track_id" not in df:
        return {}
    d = df[df["parent_track_id"] > 0]
    return {int(t): int(p) for t, p in zip(d["track_id"], d["parent_track_id"])}


def _children(parents: dict) -> dict:
    out: dict = {}
    for child, parent in parents.items():
        out.setdefault(parent, []).append(child)
    return out


def cell_position(pos: Positions, parents: dict, children: dict, tid: int, frame: int):
    """(y, x) of track `tid` in `frame`; None if it cannot be found.

    A track that is absent in the frame falls back to its daughters (after a division: their mean position) or to its
    parent (before it starts), so a boundary cell that divides keeps the line anchored."""
    p = pos[frame].get(tid)
    if p is not None:
        return p
    found, todo, depth = [], [(c, 0) for c in children.get(tid, ())], 0
    while todo:
        c, depth = todo.pop()
        if c in pos[frame]:
            found.append(pos[frame][c])
        elif depth < 6:
            todo.extend((g, depth + 1) for g in children.get(c, ()))
    if found:
        return tuple(np.mean(found, axis=0))
    a = parents.get(tid)
    for _ in range(6):
        if a is None:
            break
        if a in pos[frame]:
            return pos[frame][a]
        a = parents.get(a)
    return None


# ---- the region definition -------------------------------------------------------------------------------------
@dataclass
class Pick:
    track_id: int
    frame: int
    y: float
    x: float


@dataclass
class Anchor(Pick):
    mode: str = "keep"  # "keep" or "drop"


@dataclass
class Region:
    shape: tuple  # (height, width) of the frames
    lines: list = field(default_factory=list)  # list of list[Pick]
    anchors: list = field(default_factory=list)  # list[Anchor]

    @property
    def active(self) -> bool:
        return bool(self.lines) and bool(self.anchors)

    def to_json(self) -> dict:
        return {
            "version": 1,
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "lines": [[vars(p) | {} for p in line] for line in self.lines],
            "anchors": [vars(a) | {} for a in self.anchors],
        }

    @classmethod
    def from_json(cls, d: dict) -> "Region":
        lines = [[Pick(int(p["track_id"]), int(p["frame"]), float(p["y"]), float(p["x"])) for p in line]
                 for line in d.get("lines", [])]
        anchors = [Anchor(int(a["track_id"]), int(a["frame"]), float(a["y"]), float(a["x"]), str(a.get("mode", "keep")))
                   for a in d.get("anchors", [])]
        return cls(tuple(d["shape"]), lines, anchors)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "Region":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def region_hash(region: Region | None) -> str:
    """Short id of what is selected (cells of each line, anchors). Changes whenever the analysis would change."""
    if region is None or not region.active:
        return ""
    key = {"lines": [sorted({p.track_id for p in line}) for line in region.lines],
           "anchors": sorted((a.track_id, a.mode) for a in region.anchors)}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


# ---- geometry ------------------------------------------------------------------------------------------------------
def _order_line(line, pos, parents, children) -> list:
    """Track IDs of a line ordered along it (principal axis of the positions at the line's first pick frame)."""
    ref = line[0].frame
    ids, pts = [], []
    for p in line:
        if p.track_id in ids:
            continue
        q = cell_position(pos, parents, children, p.track_id, ref) or (p.y, p.x)
        ids.append(p.track_id)
        pts.append(q)
    pts = np.asarray(pts, float)
    if len(ids) < 2:
        raise ValueError("a boundary line needs at least two cells")
    c = pts - pts.mean(axis=0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    return [ids[i] for i in np.argsort(c @ axis)]


def _ray_to_border(p, d, w, h):
    """Point where the ray p + t*d (t > 0) leaves the rectangle [0, w] x [0, h]."""
    ts = []
    for k, hi in ((0, w), (1, h)):
        if d[k] > 1e-12:
            ts.append((hi - p[k]) / d[k])
        elif d[k] < -1e-12:
            ts.append((0 - p[k]) / d[k])
    t = min(t for t in ts if t >= 0) if ts else 0.0
    q = p + t * d
    return np.clip(q, [0, 0], [w, h])


def _perimeter(pt, w, h) -> float:
    """Position of a border point along the rectangle, clockwise from the top-left corner."""
    x, y = pt
    tol = 1e-6 * max(w, h)
    if abs(y) <= tol:
        return float(x)
    if abs(x - w) <= tol:
        return float(w + y)
    if abs(y - h) <= tol:
        return float(w + h + (w - x))
    return float(2 * w + h + (h - y))


def extended_polyline(points_xy: np.ndarray, shape) -> np.ndarray:
    """Polyline through the points (x, y) with both ends extended straight to the image border."""
    h, w = shape
    pts = [points_xy[0]]
    for q in points_xy[1:]:
        if np.hypot(*(q - pts[-1])) > 1e-6:
            pts.append(q)
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        raise ValueError("a boundary line needs at least two different positions")
    a = _ray_to_border(pts[0], pts[0] - pts[1], w, h)
    b = _ray_to_border(pts[-1], pts[-1] - pts[-2], w, h)
    return np.vstack([a, pts, b])


def side_polygon(poly: np.ndarray, shape) -> np.ndarray:
    """Polygon (x, y) of "side 0" of the directed polyline: the polyline closed along the border from its end back
    to its start in the clockwise direction. Side 1 is the rest of the frame."""
    h, w = shape
    per = 2 * (w + h)
    s_end, s_start = _perimeter(poly[-1], w, h), _perimeter(poly[0], w, h)
    corners = [(0.0, (0.0, 0.0)), (float(w), (float(w), 0.0)), (float(w + h), (float(w), float(h))),
               (float(2 * w + h), (0.0, float(h)))]
    span = (s_start - s_end) % per
    between = sorted(((s - s_end) % per, xy) for s, xy in corners if 0 < (s - s_end) % per < span)
    return np.vstack([poly] + [np.asarray(xy, float)[None] for _, xy in between])


class Geometry:
    """Boundary lines of one region over all frames of one movie."""

    def __init__(self, region: Region, positions: Positions, parents: dict | None = None):
        self.region, self.pos = region, positions
        self.parents = parents or {}
        self.children = _children(self.parents)
        self.shape = tuple(region.shape)
        self.n_frames = len(positions)
        self.orders = [_order_line(line, positions, self.parents, self.children) for line in region.lines]
        self.polys = [self._line_over_time(order) for order in self.orders]
        self.n_sigs = 2 ** len(region.lines)
        self.kept = self._kept_signatures()

    # -- lines ---------------------------------------------------------------------------------------------------
    def _line_over_time(self, order) -> list:
        raw = []
        for f in range(self.n_frames):
            pts = [cell_position(self.pos, self.parents, self.children, t, f) for t in order]
            pts = np.array([[x, y] for (y, x) in (p for p in pts if p is not None)], float)
            raw.append(extended_polyline(pts, self.shape) if len(pts) >= 2 and
                       len(np.unique(pts.round(6), axis=0)) >= 2 else None)
        ok = [f for f, r in enumerate(raw) if r is not None]
        if not ok:
            raise ValueError("a boundary line has fewer than two cells in every frame")
        return [raw[min(ok, key=lambda g: abs(g - f))] if raw[f] is None else raw[f] for f in range(self.n_frames)]

    def line_polyline(self, line_index: int, frame: int) -> np.ndarray:
        """Extended polyline (x, y) of one boundary line in one frame."""
        return self.polys[line_index][frame]

    # -- signatures ----------------------------------------------------------------------------------------------
    def signature_points(self, frame: int, pts_yx: np.ndarray) -> np.ndarray:
        """Signature (bit i = side of line i) of points (y, x) in a frame."""
        from skimage.measure import points_in_poly

        pts_yx = np.atleast_2d(np.asarray(pts_yx, float))
        sig = np.zeros(len(pts_yx), dtype=np.int64)
        xy = pts_yx[:, ::-1]
        for i, polys in enumerate(self.polys):
            poly = side_polygon(polys[frame], self.shape)
            on_side0 = points_in_poly(xy, poly)
            sig |= (~on_side0).astype(np.int64) << i
        return sig

    def signature_image(self, frame: int) -> np.ndarray:
        from skimage.draw import polygon2mask

        sig = np.zeros(self.shape, dtype=np.int16)
        for i, polys in enumerate(self.polys):
            poly = side_polygon(polys[frame], self.shape)
            side0 = polygon2mask(self.shape, poly[:, ::-1])
            sig |= (~side0).astype(np.int16) << i
        return sig

    def _kept_signatures(self) -> set:
        keep, drop = set(), set()
        for a in self.region.anchors:
            f = int(np.clip(a.frame, 0, self.n_frames - 1))
            p = cell_position(self.pos, self.parents, self.children, a.track_id, f) or (a.y, a.x)
            s = int(self.signature_points(f, np.array([p]))[0])
            (keep if a.mode == "keep" else drop).add(s)
        base = keep if keep else set(range(self.n_sigs))
        return base - drop

    # -- what is inside ------------------------------------------------------------------------------------------
    def keep_mask(self, frame: int) -> np.ndarray:
        """(Y, X) bool: True where the frame belongs to the kept area."""
        return np.isin(self.signature_image(frame), sorted(self.kept))

    def inside_sets(self) -> list:
        """Per frame: set of track IDs whose centroid is in the kept area."""
        out = []
        for f in range(self.n_frames):
            ids = list(self.pos[f])
            if not ids:
                out.append(set())
                continue
            sig = self.signature_points(f, np.array([self.pos[f][i] for i in ids]))
            out.append({i for i, s in zip(ids, sig) if int(s) in self.kept})
        return out


def event_inside(event: dict, inside: list) -> bool:
    """True when all four cells of a T1 event are inside at the frame of its first swap."""
    quartet = event.get("quartet") or [event[k] for k in "ABCD"]
    f = int(event.get("t_first_swap", event.get("frame_first_swap")))
    f = min(max(f, 0), len(inside) - 1)
    return all(int(c) in inside[f] for c in quartet)


# ---- drawing ---------------------------------------------------------------------------------------------------------
def burn_frame(rgb: np.ndarray, keep: np.ndarray, dim: float = 0.45, width: int = 3) -> np.ndarray:
    """One RGB frame (Y,X,3 uint8) with the area outside `keep` dimmed and the kept area outlined in yellow (new array)."""
    from scipy import ndimage as ndi
    from skimage.segmentation import find_boundaries

    out = np.array(rgb, copy=True)
    out[~keep] = (out[~keep].astype(np.float32) * dim).astype(out.dtype)
    edge = find_boundaries(keep, mode="inner")
    if width > 1:
        edge = ndi.binary_dilation(edge, iterations=max(1, width // 2))
    out[edge] = YELLOW
    return out


def burn_region(rgb: np.ndarray, geom: Geometry, dim: float = 0.45, width: int = 3) -> np.ndarray:
    """Copy of a (T,Y,X,3) uint8 stack with the kept area outlined in yellow and the rest dimmed."""
    rgb = np.asarray(rgb)
    if rgb.ndim == 3:
        rgb = rgb[None]
    out = np.empty_like(rgb)
    for f in range(rgb.shape[0]):
        out[f] = burn_frame(rgb[f], geom.keep_mask(min(f, geom.n_frames - 1)), dim, width)
    return out


def keep_stack(geom: Geometry) -> np.ndarray:
    """(T,Y,X) uint8 stack, 1 inside the kept area (written as `<stem>_analysis_region.tif`)."""
    return np.stack([geom.keep_mask(f) for f in range(geom.n_frames)]).astype(np.uint8)


def geometry_for(region: Region, stack: np.ndarray, lineage_df=None) -> Geometry:
    """Convenience: geometry of `region` on a tracked label stack (and optional lineage table)."""
    return Geometry(region, positions_from_stack(stack), parents_from_lineage(lineage_df))
