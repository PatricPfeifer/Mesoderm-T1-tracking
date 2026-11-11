"""Analysis cropping tab: choose which part of the embryo counts for T1 detection and cell properties.

Needs finished tracking (picks are track IDs). The user clicks cells that form a boundary line (any frame; the line is
redrawn through the same cells in every frame, so it moves with the tissue; by default it is the straight best-fit
line through them), adds more lines if needed, then clicks a
cell in each area to keep (or to disregard). The geometry lives in `core/region.py`; this file is only the napari/Qt
side: pick handling, preview layers, saving to `<results>/analysis_region/region.json`.
"""
from __future__ import annotations

from pathlib import Path

import napari
import numpy as np
from napari.layers import Labels
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .core.project import LEGACY_RUN
from .core.region import (
    POLYLINE, REGION_FILE, STRAIGHT, TRACKING_REQUIRED, Anchor, Geometry, Pick, Region, cell_position, has_tracking, keep_stack,
    parents_from_lineage, positions_from_stack, region_hash,
)

LINES_LAYER = "Analysis region lines"
CELLS_LAYER = "Analysis region cells"
AREA_LAYER = "Analysis region"
LINE_COLORS = ("#ffd400", "#ff9f1c", "#2ec4b6", "#e71d36")


class AnalysisRegionWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        # Set by the main widget:
        self.tracked_provider = lambda: None  # -> Labels layer with the tracked stack
        self.dir_provider = lambda: None  # -> results folder (Path) or None
        self.project_provider = lambda: None  # -> WorkingFolder (for settings.json) or None
        self.run_provider = lambda: None  # -> name of the segmentation run the shown tracking belongs to
        self.tracking_dir_provider = lambda: None  # -> folder of the shown tracking (lineage.csv lives there)
        self._region_run = None  # run the lines/areas on screen were picked on (track IDs differ between runs)
        self.on_changed = None  # called after the region changed (stale-results reminder)
        self.lines: list[list[Pick]] = []
        self.anchors: list[Anchor] = []
        self.active_line: int | None = None
        self.mode: str | None = None  # None, "line" or "area"
        self.geom: Geometry | None = None
        self._pos_cache: tuple | None = None
        self._tracking_ok = False
        self._busy = False
        self._drawn_key = None  # what the preview layers show now: (frame, geometry, anchors); skip identical redraws
        self._area_cache: tuple | None = None  # (geometry, {frame: RGBA image}) of the area preview
        self._auto_hidden: set[str] = set()  # preview layers this widget hid (empty); the user's own toggle is left alone
        self.line_mode = STRAIGHT  # mode of the region on screen; a loaded old region keeps "polyline"

        self.notice = QLabel(TRACKING_REQUIRED)
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet("color: #e05252; font-weight: bold;")
        self.stale = QLabel("")
        self.stale.setWordWrap(True)
        self.stale.setStyleSheet("color: #e0a030; font-weight: bold;")

        # 1 · Boundary lines
        self.new_line_btn = QPushButton("New line: click its cells")
        self.new_line_btn.setCheckable(True)
        self.new_line_btn.setToolTip(
            "Click the cells that mark the boundary, on any frame (two or more, in a row). The line is drawn through "
            "them (the best-fit straight line, or through every cell with 'Straight lines' off) and stays on the same "
            "cells in every frame. The picked cells sit on the line: their side of it does not matter, they count as inside when the area they are in is kept (by the other lines). Click a picked cell again to "
            "remove it."
        )
        self.add_cells_btn = QPushButton("Add cells to selected line")
        self.add_cells_btn.setCheckable(True)
        self.undo_btn = QPushButton("Undo last cell")
        self.delete_line_btn = QPushButton("Delete line")
        self.straight_box = QCheckBox("Straight lines (best fit)")
        self.straight_box.setChecked(True)
        self.straight_box.setToolTip(
            "On: each boundary line is the straight line that fits its cells best (refitted in every frame, so it "
            "follows the tissue). Use one line that runs roughly horizontally and one roughly vertically to cut the "
            "embryo into corners. Off: the line bends through every picked cell."
        )
        self.line_list = QListWidget()
        self.line_list.setMaximumHeight(90)
        l1 = QVBoxLayout()
        r = QHBoxLayout()
        r.addWidget(self.new_line_btn)
        r.addWidget(self.add_cells_btn)
        l1.addLayout(r)
        r = QHBoxLayout()
        r.addWidget(self.undo_btn)
        r.addWidget(self.delete_line_btn)
        l1.addLayout(r)
        l1.addWidget(self.straight_box)
        l1.addWidget(self.line_list)
        box1 = QGroupBox("1 · Boundary lines")
        box1.setLayout(l1)

        # 2 · Area
        self.keep_radio = QRadioButton("Keep this area")
        self.drop_radio = QRadioButton("Disregard this area")
        self.keep_radio.setChecked(True)
        self.area_btn = QPushButton("Click a cell in an area")
        self.area_btn.setCheckable(True)
        self.area_btn.setToolTip(
            "Click one cell inside each area that should be kept (or disregarded). With several areas picked, "
            "all 'keep' areas are used together. Without any 'keep' click, everything except the disregarded areas "
            "is used."
        )
        self.remove_anchor_btn = QPushButton("Remove selected")
        self.anchor_list = QListWidget()
        self.anchor_list.setMaximumHeight(90)
        group = QButtonGroup(self)
        group.addButton(self.keep_radio)
        group.addButton(self.drop_radio)
        l2 = QVBoxLayout()
        r = QHBoxLayout()
        r.addWidget(self.keep_radio)
        r.addWidget(self.drop_radio)
        l2.addLayout(r)
        r = QHBoxLayout()
        r.addWidget(self.area_btn)
        r.addWidget(self.remove_anchor_btn)
        l2.addLayout(r)
        l2.addWidget(self.anchor_list)
        box2 = QGroupBox("2 · Area to use")
        box2.setLayout(l2)

        # 3 · Save
        self.save_btn = QPushButton("Save region")
        self.save_btn.setToolTip(
            "Writes analysis_region/region.json and the kept-area mask next to the other results. "
            "T1 detection and Export Cell Props then use only the cells inside; exported tiffs show the kept area "
            "as a yellow outline."
        )
        self.load_btn = QPushButton("Load saved region")
        self.clear_btn = QPushButton("Clear region")
        l3 = QHBoxLayout()
        for b in (self.save_btn, self.load_btn, self.clear_btn):
            l3.addWidget(b)
        box3 = QGroupBox("3 · Save / apply")
        box3.setLayout(l3)

        self.status = QLabel("Pick two or more cells on a boundary, then click a cell in the area to keep.")
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.controls = QWidget()
        cl = QVBoxLayout()
        cl.setContentsMargins(0, 0, 0, 0)
        for w in (box1, box2, box3):
            cl.addWidget(w)
        self.controls.setLayout(cl)
        lay = QVBoxLayout()
        lay.addWidget(self.notice)
        lay.addWidget(self.stale)
        lay.addWidget(self.controls)
        lay.addWidget(self.status)
        lay.addStretch(1)
        self.setLayout(lay)

        self.new_line_btn.clicked.connect(self._new_line_clicked)
        self.add_cells_btn.clicked.connect(self._add_cells_clicked)
        self.area_btn.clicked.connect(self._area_clicked)
        self.undo_btn.clicked.connect(self._undo_cell)
        self.delete_line_btn.clicked.connect(self._delete_line)
        self.remove_anchor_btn.clicked.connect(self._remove_anchor)
        self.save_btn.clicked.connect(self.save)
        self.load_btn.clicked.connect(self.load)
        self.clear_btn.clicked.connect(self.clear)
        self.line_list.currentRowChanged.connect(self._line_selected)
        self.straight_box.toggled.connect(self._straight_toggled)
        self.viewer.mouse_drag_callbacks.append(self._on_click)
        self.viewer.dims.events.current_step.connect(self._draw)
        self.viewer.layers.events.inserted.connect(self.refresh_gate)
        self.viewer.layers.events.removed.connect(self.refresh_gate)
        self.refresh_gate()

    # ---- tracking gate -------------------------------------------------------------------------------------------
    def _stack(self):
        layer = self.tracked_provider()
        if layer is None or not isinstance(layer, Labels):
            return None
        data = np.asarray(layer.data)
        return data if data.ndim == 3 else None

    def refresh_gate(self, *_) -> bool:
        """Enable the tab only when a tracked label stack is loaded; otherwise explain that tracking is required."""
        stack = self._stack()
        ok = has_tracking(stack)
        self._tracking_ok = ok
        self.notice.setVisible(not ok)
        self.controls.setEnabled(ok)
        if not ok:
            self._set_mode(None)
        return ok

    # ---- picking ---------------------------------------------------------------------------------------------------
    def _set_mode(self, mode: str | None) -> None:
        self.mode = mode
        self.new_line_btn.setChecked(mode == "line" and self._new_line_pending)
        self.add_cells_btn.setChecked(mode == "line" and not self._new_line_pending)
        self.area_btn.setChecked(mode == "area")

    _new_line_pending = False

    def _new_line_clicked(self, on: bool) -> None:
        if not on:
            self._set_mode(None)
            return
        self.lines.append([])
        self.active_line = len(self.lines) - 1
        self._new_line_pending = True
        self._set_mode("line")
        self._refresh_lists()
        self.status.setText("Click the cells that form the boundary (any frame). Click the button again when done.")

    def _add_cells_clicked(self, on: bool) -> None:
        if not on:
            self._set_mode(None)
            return
        row = self.line_list.currentRow()
        if row < 0 or row >= len(self.lines):
            self.add_cells_btn.setChecked(False)
            self.status.setText("Select a line in the list first.")
            return
        self.active_line = row
        self._new_line_pending = False
        self._set_mode("line")

    def _area_clicked(self, on: bool) -> None:
        if on and not self.lines:
            self.area_btn.setChecked(False)
            self.status.setText("Add at least one boundary line first.")
            return
        self._new_line_pending = False
        self._set_mode("area" if on else None)
        if on:
            self.status.setText("Click one cell in each area. Click a marked cell again to remove it.")

    def _line_selected(self, row: int) -> None:
        if 0 <= row < len(self.lines):
            self.active_line = row
            self._draw()

    def _cell_at(self, position):
        layer = self.tracked_provider()
        if layer is None:
            return None
        data = np.asarray(layer.data)
        c = np.asarray(layer.world_to_data(position), dtype=float)
        y, x = int(round(c[-2])), int(round(c[-1]))
        frame = int(np.clip(round(float(self.viewer.dims.point[0])), 0, data.shape[0] - 1))
        if not (0 <= y < data.shape[-2] and 0 <= x < data.shape[-1]):
            return None
        return frame, y, x, int(data[frame, y, x])

    def _on_click(self, viewer, event):
        if self.mode is None or not self._tracking_ok or "Shift" in event.modifiers:
            return
        moved = False
        yield
        while event.type == "mouse_move":
            moved = True
            yield
        if moved:  # a drag (pan), not a click
            return
        hit = self._cell_at(event.position)
        if hit is None or hit[3] == 0:
            return
        frame, y, x, tid = hit
        if self.mode == "line" and self.active_line is not None:
            line = self.lines[self.active_line]
            same = [p for p in line if p.track_id == tid]
            if same:
                line.remove(same[0])
            else:
                line.append(Pick(tid, frame, float(y), float(x)))
        elif self.mode == "area":
            same = [a for a in self.anchors if a.track_id == tid]
            if same:
                self.anchors.remove(same[0])
            else:
                self.anchors.append(Anchor(tid, frame, float(y), float(x), "keep" if self.keep_radio.isChecked() else "drop"))
        self._changed()

    def _undo_cell(self) -> None:
        if self.mode == "area" or (self.active_line is None and self.anchors):
            if self.anchors:
                self.anchors.pop()
        elif self.active_line is not None and self.lines[self.active_line]:
            self.lines[self.active_line].pop()
        self._changed()

    def _delete_line(self) -> None:
        row = self.line_list.currentRow()
        if 0 <= row < len(self.lines):
            del self.lines[row]
            self.active_line = None
            self._set_mode(None)
            self._changed()

    def _straight_toggled(self, on: bool) -> None:
        self.line_mode = STRAIGHT if on else POLYLINE
        if self.lines:
            self._changed()

    def _remove_anchor(self) -> None:
        row = self.anchor_list.currentRow()
        if 0 <= row < len(self.anchors):
            del self.anchors[row]
            self._changed()

    # ---- state -> region -> geometry --------------------------------------------------------------------------
    def region(self) -> Region | None:
        stack = self._stack()
        if stack is None:
            return None
        lines = [list(line) for line in self.lines if len({p.track_id for p in line}) >= 2]
        return Region(tuple(stack.shape[-2:]), lines, list(self.anchors), self.line_mode)

    def _positions(self):
        stack = self._stack()
        key = (id(self.tracked_provider()), stack.shape, hash(stack[:, ::16, ::16].tobytes()))
        if self._pos_cache is None or self._pos_cache[0] != key:
            self.status.setText("Reading cell positions from the tracked labels...")
            self._pos_cache = (key, positions_from_stack(stack))
        return self._pos_cache[1]

    def invalidate(self) -> None:
        """Forget cached cell positions (call after the tracked labels changed)."""
        self._pos_cache = None
        self.geom = None
        self.refresh_gate()
        if self._tracking_ok and self.lines:
            self._rebuild()

    def _parents(self) -> dict:
        folder = self.tracking_dir_provider()
        if folder is None:
            return {}
        try:
            import pandas as pd

            found = sorted(Path(folder).glob("*_lineage.csv"))
            return parents_from_lineage(pd.read_csv(found[0])) if found else {}
        except Exception:
            return {}

    def _rebuild(self) -> None:
        self.geom = None
        region = self.region()
        if region is None or not region.lines:
            return
        try:
            self.geom = Geometry(region, self._positions(), self._parents())
        except ValueError as exc:
            self.status.setText(str(exc))

    def _changed(self) -> None:
        self._rebuild()
        self._refresh_lists()
        self._draw(force=True)
        n_in = None
        if self.geom is not None and self.anchors:
            n_in = len(self.geom.inside_ids(self._frame()))
        msg = f"{len(self.lines)} line(s), {len(self.anchors)} area click(s)."
        if n_in is not None:
            msg += f" Cells inside on this frame: {n_in}."
        if self.lines and not self.anchors:
            msg += " Now click a cell in the area to keep (or disregard)."
        self.status.setText(msg)
        self._notify()

    def _notify(self) -> None:
        if callable(self.on_changed):
            self.on_changed()

    def _refresh_lists(self) -> None:
        self.line_list.blockSignals(True)
        self.line_list.clear()
        for i, line in enumerate(self.lines):
            n = len({p.track_id for p in line})
            self.line_list.addItem(f"Line {i + 1} · {n} cell(s)" + ("  (needs 2+)" if n < 2 else ""))
        if self.active_line is not None and self.active_line < len(self.lines):
            self.line_list.setCurrentRow(self.active_line)
        self.line_list.blockSignals(False)
        self.anchor_list.clear()
        for a in self.anchors:
            self.anchor_list.addItem(f"{'keep' if a.mode == 'keep' else 'disregard'} · cell {a.track_id} (frame {a.frame + 1})")

    # ---- preview in the viewer ---------------------------------------------------------------------------------
    def _transform(self):
        t = self.tracked_provider()
        if t is None:
            return (1.0, 1.0), (0.0, 0.0)
        return tuple(t.scale[-2:]), tuple(t.translate[-2:])

    def _frame(self) -> int:
        n = len(self.geom.pos) if self.geom is not None else 1
        return int(np.clip(round(float(self.viewer.dims.point[0])), 0, n - 1))

    def _draw(self, *_, force: bool = False) -> None:
        if self._busy or not self._tracking_ok:
            return
        # the dims event also fires for other axes / the same frame again: nothing to redraw then
        key = (self._frame(), id(self.geom), self.active_line, len(self.anchors),
               tuple(a.track_id for a in self.anchors), len(self.viewer.layers))
        if not force and key == self._drawn_key:
            return
        self._busy = True
        try:
            self._draw_inner()
            self._drawn_key = key
        finally:
            self._busy = False

    def _draw_inner(self) -> None:
        scale, translate = self._transform()
        paths, colors = [], []
        pts, pcol = [], []
        frame = self._frame()
        if self.geom is not None:
            for i in range(len(self.geom.polys)):
                p = self.geom.line_polyline(i, frame)
                paths.append(p[:, ::-1])
                colors.append(LINE_COLORS[i % len(LINE_COLORS)])
            for i, order in enumerate(self.geom.orders):
                for tid in order:
                    q = cell_position(self.geom.pos, self.geom.parents, self.geom.children, tid, frame)
                    if q is not None:
                        pts.append(q)
                        pcol.append(LINE_COLORS[i % len(LINE_COLORS)])
            for a in self.anchors:
                q = cell_position(self.geom.pos, self.geom.parents, self.geom.children, a.track_id, frame)
                if q is not None:
                    pts.append(q)
                    pcol.append("#3ddc84" if a.mode == "keep" else "#ff4d4d")
        self._set_shapes(paths, colors, scale, translate)
        self._set_points(pts, pcol, scale, translate)
        self._set_area(frame, scale, translate)

    def _layer(self, name):
        return self.viewer.layers[name] if name in self.viewer.layers else None

    def _hide(self, layer) -> None:
        """Hide a preview layer that has nothing to show now (remembered, so it is shown again when it has content)."""
        if layer.visible:
            layer.visible = False
            self._auto_hidden.add(layer.name)

    def _show(self, layer) -> None:
        """New content for a preview layer: only turn it back on if this widget hid it. A layer the user toggled
        off in the layer list stays off while the frame changes."""
        if layer.name in self._auto_hidden:
            self._auto_hidden.discard(layer.name)
            layer.visible = True

    def _set_shapes(self, paths, colors, scale, translate) -> None:
        layer = self._layer(LINES_LAYER)
        if not paths:
            if layer is not None:
                self._hide(layer)
            return
        if layer is None:
            active = self.viewer.layers.selection.active
            layer = self.viewer.add_shapes(paths, shape_type="path", edge_color=colors, edge_width=3, name=LINES_LAYER,
                                           scale=scale, translate=translate)
            layer.editable = False
            if active is not None and active in self.viewer.layers:
                self.viewer.layers.selection.active = active
        else:
            layer.data = []
            layer.add(paths, shape_type="path", edge_color=colors, edge_width=3)
            self._show(layer)

    def _set_points(self, pts, cols, scale, translate) -> None:
        layer = self._layer(CELLS_LAYER)
        if not pts:
            if layer is not None:
                self._hide(layer)
            return
        data = np.asarray(pts, float)
        if layer is not None and len(layer.data) != len(data):
            # napari redraws a Points layer with the old slice indices while the data shrinks ("index 2 is out of bounds
            # for axis 0 with size 2"): a changed number of points gets a fresh layer instead
            self.viewer.layers.remove(layer)
            layer = None
        if layer is None:
            active = self.viewer.layers.selection.active
            layer = self.viewer.add_points(data, face_color=cols, size=14, name=CELLS_LAYER, scale=scale,
                                           translate=translate)
            layer.editable = False
            if active is not None and active in self.viewer.layers:
                self.viewer.layers.selection.active = active
        else:
            layer.data = data
            layer.face_color = cols
            self._show(layer)

    def _set_area(self, frame: int, scale, translate) -> None:
        from scipy import ndimage as ndi
        from skimage.segmentation import find_boundaries

        layer = self._layer(AREA_LAYER)
        if self.geom is None or not self.anchors:
            if layer is not None:
                self._hide(layer)
            return
        if self._area_cache is None or self._area_cache[0] is not self.geom:
            self._area_cache = (self.geom, {})
        images = self._area_cache[1]
        img = images.get(frame)
        if img is None:  # scrubbing back and forth reuses the images already made for this geometry
            keep = self.geom.keep_mask(frame)
            img = np.zeros((*keep.shape, 4), np.uint8)
            img[~keep] = (0, 0, 0, 120)
            edge = ndi.binary_dilation(find_boundaries(keep, mode="inner"), iterations=1)
            img[edge] = (255, 255, 0, 255)
            if len(images) >= 12:
                images.pop(next(iter(images)))
            images[frame] = img
        if layer is None:
            active = self.viewer.layers.selection.active
            layer = self.viewer.add_image(img, rgb=True, name=AREA_LAYER, scale=scale, translate=translate)
            if active is not None and active in self.viewer.layers:
                self.viewer.layers.selection.active = active
        else:
            layer.data = img
            self._show(layer)

    # ---- results use -------------------------------------------------------------------------------------------
    def geometry(self) -> Geometry | None:
        """Geometry of the current region when it is complete (a line and a kept/disregarded area), else None."""
        if not self._tracking_ok or not self.anchors:
            return None
        if self.geom is None:
            self._rebuild()
        return self.geom

    def inside(self):
        """(per-frame sets of track IDs inside, region hash) for T1 / cell props, or None when no region is set."""
        geom = self.geometry()
        if geom is None:
            return None
        return geom.inside_sets(), region_hash(geom.region)

    def current_hash(self) -> str:
        region = self.region()
        return region_hash(region) if region is not None and self.geometry() is not None else ""

    # ---- save / load -------------------------------------------------------------------------------------------
    def _folder(self):
        """analysis_region/<segmentation run>/: track IDs belong to one run's tracking, so the region does too."""
        root = self.dir_provider()
        return None if root is None else Path(root) / "analysis_region" / (self.run_provider() or LEGACY_RUN)

    def tracking_changed(self) -> None:
        """Call after tracking was run or loaded. Lines and areas picked on another run's track IDs are dropped (they
        would point at different cells); a region saved for this run is loaded again."""
        run = self.run_provider()
        if run != self._region_run:
            self._reset_state()
            self._region_run = run
        self.refresh_gate()
        folder = self._folder()
        if self._tracking_ok and not self.lines and folder is not None and (folder / REGION_FILE).is_file():
            self.load()
        self._notify()

    def _reset_state(self) -> None:
        self.lines, self.anchors, self.active_line = [], [], None
        self._set_mode(None)
        self.geom = None
        self._pos_cache = None
        self._drawn_key, self._area_cache = None, None
        for name in (LINES_LAYER, CELLS_LAYER, AREA_LAYER):
            layer = self._layer(name)
            if layer is not None:
                self.viewer.layers.remove(layer)
        self._refresh_lists()

    def save(self) -> None:
        geom = self.geometry()
        folder = self._folder()
        if geom is None and not self.lines and not self.anchors and folder is not None and (folder / REGION_FILE).is_file():
            for f in (folder / REGION_FILE, folder / "analysis_region_mask.tif"):  # an emptied region removes the saved one
                f.unlink(missing_ok=True)
            project = self.project_provider()
            if project is not None:
                project.write_settings("analysis_region", {"hash": ""})
            self.status.setText("Saved: no analysis region (the saved region file was removed).")
            self._notify()
            return
        if geom is None:
            self.status.setText("Nothing to save: add a boundary line (2+ cells) and click the area to keep.")
            return
        if folder is None:
            self.status.setText("Not saved: no results folder chosen (use Folder...).")
            return
        import tifffile

        self._rebuild()  # cell positions stay cached (the cache key notices changed labels)
        geom = self.geometry()
        geom.region.save(folder / REGION_FILE)
        tifffile.imwrite(folder / "analysis_region_mask.tif", keep_stack(geom), compression="zlib")
        project = self.project_provider()
        if project is not None:
            project.write_settings("analysis_region", {"hash": region_hash(geom.region), "lines": len(geom.region.lines),
                                                       "areas": len(geom.region.anchors)})
        n_in = len(geom.inside_ids(0))
        self.status.setText(f"Saved the region ({n_in} of {len(geom.pos[0])} cells inside in frame 1) → {folder}")
        self._notify()

    def has_unsaved(self) -> bool:
        """The region differs from the saved file (new, changed, or emptied while a file exists)."""
        folder = self._folder()
        saved = folder is not None and (folder / REGION_FILE).is_file()
        if self.geometry() is not None:
            if not saved:
                return folder is not None
            try:
                return region_hash(Region.load(folder / REGION_FILE)) != self.current_hash()
            except Exception:
                return True
        return saved and not self.lines and not self.anchors

    def load(self) -> None:
        folder = self._folder()
        path = None if folder is None else folder / REGION_FILE
        if path is None or not path.is_file():
            self.status.setText("No saved region found in the results folder.")
            return
        if not self.refresh_gate():
            self.status.setText(TRACKING_REQUIRED)
            return
        region = Region.load(path)
        self.lines, self.anchors = [list(line) for line in region.lines], list(region.anchors)
        self.line_mode = region.line_mode  # an old saved region keeps its polylines until the box is ticked again
        self.straight_box.blockSignals(True)
        self.straight_box.setChecked(region.line_mode == STRAIGHT)
        self.straight_box.blockSignals(False)
        self.active_line = None
        self._set_mode(None)
        self._changed()
        self.status.setText(f"Loaded {path.name}: {len(self.lines)} line(s), {len(self.anchors)} area click(s).")

    def clear(self) -> None:
        self._reset_state()
        folder = self._folder()
        saved = folder is not None and (folder / REGION_FILE).is_file()
        self.status.setText("Region cleared." + (" The saved file is kept until you press Save region." if saved else ""))
        self._notify()

    def set_stale_message(self, text: str) -> None:
        self.stale.setText(text)
        self.stale.setVisible(bool(text))
