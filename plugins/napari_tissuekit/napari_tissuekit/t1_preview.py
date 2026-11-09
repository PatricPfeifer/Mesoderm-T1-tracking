"""T1 preview: click an event on the T1 overlay and watch it in a looping popup (like the T1 picker's Preview).

The pure helpers (`find_event_at`, `event_frames`, `sort_events`, `render_preview_frame`) need no Qt; `T1PreviewDialog` is the
popup. Events are the dicts of a plugin T1 result (`run_states_stack`): `quartet`, `pair_before`, `pair_after`, `pair_building`,
`t_start`, `t_end`, `t_first_swap`, `type`, `y`, `x`, ...
"""
from __future__ import annotations

import numpy as np

from .core.t1.run_tm_t1 import TYPE_COLORS, TYPE_SORT_ORDER, paint_pair

OUTLINE = (90, 90, 90)
QUARTET_PALETTE = ((170, 130, 190), (130, 175, 155), (130, 155, 195), (195, 150, 140))  # as the picker
CLICK_RADIUS = 15  # pixels: a click this close to an event's four-way vertex counts as on the event


# ---- finding and ordering events -----------------------------------------------------------------------------------
def event_frames(ev: dict, pre: int, post: int, n_frames: int) -> range:
    """Frames the preview of an event plays: from `pre` frames before the event to `post` frames after it."""
    return range(max(0, int(ev["t_start"]) - int(pre)), min(n_frames - 1, int(ev["t_end"]) + int(post)) + 1)


def sort_events(events) -> list:
    """Like the picker: completed, flicker_resolved, flicker_aborted, then by start frame and cells."""
    return sorted(events, key=lambda e: (TYPE_SORT_ORDER.get(e.get("type"), 9), int(e["t_start"]), sorted(e["quartet"])))


def _vertex(ev: dict, four_xy, frame: int):
    if four_xy and 0 <= frame < len(four_xy):
        p = four_xy[frame].get(ev["quartet"])
        if p is not None:
            return p
    return (float(ev["y"]), float(ev["x"]))


def find_event_at(events, four_xy, frame: int, y: float, x: float, label: int, pre: int, post: int,
                  radius: float = CLICK_RADIUS):
    """The event a click at (frame, y, x) belongs to, or None.

    Candidates are events visible on that frame (`t_start - pre .. t_end + post`) whose quartet contains the clicked cell
    (a junction pixel belongs to one of its two cells) or whose four-way vertex is within `radius` pixels. The nearest vertex wins."""
    best, best_d = None, None
    for ev in events:
        if not (int(ev["t_start"]) - int(pre) <= frame <= int(ev["t_end"]) + int(post)):
            continue
        vy, vx = _vertex(ev, four_xy, frame)
        d = float(np.hypot(vy - y, vx - x))
        if not (label in ev["quartet"] or d <= radius):
            continue
        if best is None or d < best_d:
            best, best_d = ev, d
    return best


# ---- drawing --------------------------------------------------------------------------------------------------------
def _outline(labels: np.ndarray) -> np.ndarray:
    e = np.zeros(labels.shape, dtype=bool)
    e[:, :-1] |= (labels[:, :-1] != labels[:, 1:]) & (labels[:, :-1] > 0)
    e[:, 1:] |= (labels[:, 1:] != labels[:, :-1]) & (labels[:, 1:] > 0)
    e[:-1, :] |= (labels[:-1, :] != labels[1:, :]) & (labels[:-1, :] > 0)
    e[1:, :] |= (labels[1:, :] != labels[:-1, :]) & (labels[1:, :] > 0)
    return e


def event_box(ev: dict, labels_stack, margin: int = 14, min_size: int = 70):
    """(y0, y1, x0, x1) window around the quartet at the frame of the first swap (same window for every frame of the event)."""
    n = labels_stack.shape[0]
    f = min(max(int(ev.get("t_first_swap", ev["t_start"])), 0), n - 1)
    ys, xs = np.nonzero(np.isin(labels_stack[f], list(ev["quartet"])))
    h, w = labels_stack.shape[-2:]
    if not len(ys):
        cy, cx = int(round(ev["y"])), int(round(ev["x"]))
        ys, xs = np.array([cy]), np.array([cx])
    y0, y1, x0, x1 = ys.min() - margin, ys.max() + margin + 1, xs.min() - margin, xs.max() + margin + 1
    for lo, hi, size in ((0, 1, h), (2, 3, w)):
        box = [y0, y1, x0, x1]
        if box[hi] - box[lo] < min_size:
            mid = (box[hi] + box[lo]) // 2
            box[lo], box[hi] = mid - min_size // 2, mid + (min_size - min_size // 2)
        y0, y1, x0, x1 = box
    y0, x0 = max(int(y0), 0), max(int(x0), 0)
    return y0, min(int(y1), h), x0, min(int(x1), w)


def render_preview_frame(labels: np.ndarray, ev: dict, t: int, four_xy=None, *, reach: int = 3, box=None,
                         scale: int | None = None) -> np.ndarray:
    """One preview frame (RGB uint8): outlines, the four quartet cells filled, the breaking / building junctions and the vertex."""
    from skimage.draw import disk

    if box is None:
        box = (0, labels.shape[0], 0, labels.shape[1])
    y0, y1, x0, x1 = box
    lab = np.asarray(labels[y0:y1, x0:x1])
    out = np.zeros((*lab.shape, 3), np.uint8)
    for i, cid in enumerate(sorted(ev["quartet"])):
        out[lab == cid] = QUARTET_PALETTE[i % len(QUARTET_PALETTE)]
    out[_outline(lab)] = OUTLINE
    colors = TYPE_COLORS.get(ev.get("type"), TYPE_COLORS["completed"])
    t0, t1 = int(ev["t_start"]), int(ev["t_end"])
    building = ev.get("pair_building")
    if isinstance(building, str):
        building = frozenset(int(v) for v in building.split(";"))
    if t < t0:
        paint_pair(out, lab, ev["pair_before"], colors["breaking"], reach)
    elif t > t1:
        paint_pair(out, lab, ev["pair_after"], colors["building"], reach)
    else:
        paint_pair(out, lab, ev["pair_before"], colors["breaking"], reach)
        paint_pair(out, lab, building or ev["pair_after"], colors["building"], reach)
        vy, vx = _vertex(ev, four_xy, t)
        rr, cc = disk((int(round(vy)) - y0, int(round(vx)) - x0), 2, shape=lab.shape)
        out[rr, cc] = colors["vertex"]
    if scale is None:
        scale = int(np.clip(480 // max(lab.shape), 2, 8))
    return np.repeat(np.repeat(out, scale, axis=0), scale, axis=1)


def event_info(ev: dict, index: int, total: int) -> str:
    a, b, c, d = (int(ev[k]) for k in "ABCD")
    cls = ev.get("t1_class") or ev.get("type")
    return (f"Event {index + 1} of {total}  ·  {cls}  ·  cells {a}, {b}, {c}, {d}\n"
            f"frames {int(ev['t_start'])} to {int(ev['t_end'])}, first swap {int(ev['t_first_swap'])}"
            f"  ·  breaking {'-'.join(map(str, sorted(ev['pair_before'])))}"
            f"  →  building {'-'.join(map(str, sorted(ev['pair_after'])))}")


# ---- the popup ------------------------------------------------------------------------------------------------------
from qtpy.QtCore import Qt, QTimer, Signal  # noqa: E402
from qtpy.QtGui import QImage, QPixmap  # noqa: E402
from qtpy.QtWidgets import (  # noqa: E402
    QDialog, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout,
)


class T1PreviewDialog(QDialog):
    """Looping preview of one T1 event with Previous / Next event buttons (non-modal; one window is re-used)."""

    event_changed = Signal(object)  # the event now shown (the plugin jumps the napari view to it)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("T1 preview")
        self.setWindowFlag(Qt.Tool, True)
        self.events: list = []
        self.stack = None
        self.four_xy = None
        self.pre = self.post = 3
        self.reach = 3
        self.index = 0
        self.frames: range = range(0)
        self.t = 0
        self._box = None
        self.image = QLabel(alignment=Qt.AlignCenter)
        self.image.setMinimumSize(320, 320)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.frame_label = QLabel("")
        self.slider = QSlider(Qt.Horizontal)
        self.play_btn = QPushButton("Pause")
        self.prev_btn = QPushButton("◀ Previous event")
        self.next_btn = QPushButton("Next event ▶")
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.5, 20.0)
        self.speed.setValue(4.0)
        self.speed.setSuffix(" frames/s")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.slider.valueChanged.connect(self._slider_moved)
        self.play_btn.clicked.connect(self.toggle_play)
        self.prev_btn.clicked.connect(lambda: self.show_event(self.index - 1))
        self.next_btn.clicked.connect(lambda: self.show_event(self.index + 1))
        self.speed.valueChanged.connect(self._restart_timer)
        row = QHBoxLayout()
        for w in (self.prev_btn, self.play_btn, self.next_btn, self.speed):
            row.addWidget(w)
        lay = QVBoxLayout()
        lay.addWidget(self.info)
        lay.addWidget(self.image, 1)
        lay.addWidget(self.frame_label)
        lay.addWidget(self.slider)
        lay.addLayout(row)
        self.setLayout(lay)

    # -- data ---------------------------------------------------------------------------------------------------------
    def set_data(self, events, stack, four_xy, pre: int, post: int, reach: int = 3) -> None:
        self.events = sort_events(events)
        self.stack, self.four_xy = stack, four_xy
        self.pre, self.post, self.reach = int(pre), int(post), int(reach)

    def show_event(self, index: int, start_frame: int | None = None, emit: bool = True) -> None:
        if not self.events:
            return
        self.index = index % len(self.events)
        ev = self.events[self.index]
        self.frames = event_frames(ev, self.pre, self.post, self.stack.shape[0])
        self._box = event_box(ev, self.stack)
        self.slider.blockSignals(True)
        self.slider.setRange(self.frames.start, self.frames.stop - 1)
        self.slider.blockSignals(False)
        self.info.setText(event_info(ev, self.index, len(self.events)))
        self._set_frame(self.frames.start if start_frame is None else min(max(start_frame, self.frames.start), self.frames.stop - 1))
        self._restart_timer()
        if emit:
            self.event_changed.emit(ev)

    def show_event_object(self, ev: dict, start_frame: int | None = None) -> None:
        self.show_event(next((i for i, e in enumerate(self.events) if e is ev), 0), start_frame)

    # -- playing ------------------------------------------------------------------------------------------------------
    def _set_frame(self, t: int) -> None:
        self.t = int(t)
        ev = self.events[self.index]
        img = np.ascontiguousarray(render_preview_frame(self.stack[self.t], ev, self.t, self.four_xy, reach=self.reach, box=self._box))
        h, w = img.shape[:2]
        self._img = img  # keep the buffer alive for the QImage
        self.image.setPixmap(QPixmap.fromImage(QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()))
        where = "before" if self.t < int(ev["t_start"]) else ("after" if self.t > int(ev["t_end"]) else "during")
        self.frame_label.setText(f"frame {self.t}  ({where}; first swap at {int(ev['t_first_swap'])})")
        self.slider.blockSignals(True)
        self.slider.setValue(self.t)
        self.slider.blockSignals(False)

    def _slider_moved(self, value: int) -> None:
        if self.events:
            self._set_frame(value)

    def _tick(self) -> None:
        if self.events and len(self.frames):
            self._set_frame(self.frames.start if self.t + 1 >= self.frames.stop else self.t + 1)

    def _restart_timer(self, *_) -> None:
        if self.play_btn.text() == "Pause" and self.events:
            self.timer.start(int(1000 / self.speed.value()))

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            self.play_btn.setText("Pause")
            self._restart_timer()

    def closeEvent(self, event) -> None:
        self.timer.stop()
        super().closeEvent(event)
