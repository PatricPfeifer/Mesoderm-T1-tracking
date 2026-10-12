"""Cell properties dock: pick tracked cells, tick features, export CSV.

The selection is just a list of track IDs (track ID == label value in the
tracked-labels stack), so it can be saved to JSON and restored later, then
edited by adding or removing individual cells.
"""
from __future__ import annotations

import json
from pathlib import Path

import napari
import numpy as np
import pandas as pd
from napari.layers import Labels
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from skimage.measure import regionprops

STATE_VERSION = 1
HELP_DIR = Path(__file__).parent / "feature_help"


def help_tooltip(key: str, label: str) -> str:
    """Tooltip with an explanatory image if feature_help/<key>.(png|jpg|jpeg) exists."""
    for ext in (".png", ".jpg", ".jpeg"):
        img = HELP_DIR / f"{key}{ext}"
        if img.is_file():
            # the card already carries its own title and definition
            return f'<img src="{img.as_posix()}" width="320" height="240">'
    return label
SELECTED_LAYER = "Selected cells"
HOVER_LAYER = "Hovered cell"
DIMMED_OPACITY = 0.4
# features that can colour the selected cells -> matplotlib colormap name
COLORABLE = {
    "area": "viridis",
    "perimeter": "viridis",
    "circularity": "viridis",
    "shape_index": "viridis",
    "eccentricity": "viridis",
    "solidity": "viridis",
    "major_axis": "viridis",
    "minor_axis": "viridis",
    "orientation": "twilight",
    "touches_edge": "viridis",
}

# key -> (label, default ticked)
FEATURES: dict[str, tuple[str, bool]] = {
    "area": ("Area", True),
    "perimeter": ("Perimeter", False),
    "circularity": ("Circularity", True),
    "shape_index": ("Shape index", True),
    "eccentricity": ("Eccentricity", False),
    "solidity": ("Solidity", False),
    "major_axis": ("Major axis length", False),
    "minor_axis": ("Minor axis length", False),
    "orientation": ("Orientation (rad)", False),
    "centroid": ("Centroid (x, y)", False),
    "touches_edge": ("Touches image edge", True),
}


def as_tyx(data) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"expected T,Y,X labels, got shape {arr.shape}")
    return arr


def track_frames(stack: np.ndarray, ids) -> dict[int, list[int]]:
    """Frames (0-based) in which each track ID is present."""
    out: dict[int, list[int]] = {int(i): [] for i in ids}
    if not out:
        return out
    wanted = np.fromiter(out, dtype=stack.dtype)
    for t in range(stack.shape[0]):
        for i in np.intersect1d(np.unique(stack[t]), wanted):
            out[int(i)].append(t)
    return out


def compute_properties(
    stack: np.ndarray,
    ids,
    features,
    pixel_size_um: float = 0.0,
    frame_interval_s: float = 0.0,
) -> pd.DataFrame:
    """One row per selected track and frame. Lengths in um if pixel_size_um > 0."""
    stack = as_tyx(stack)
    ids = [int(i) for i in ids]
    feats = [f for f in FEATURES if f in set(features)]
    h, w = stack.shape[1:]
    cal = pixel_size_um > 0
    ls = pixel_size_um if cal else 1.0
    lu, au = ("um", "um2") if cal else ("px", "px2")
    rows = []
    for t in range(stack.shape[0]):
        frame = stack[t]
        present = np.intersect1d(np.unique(frame), np.asarray(ids, dtype=frame.dtype))
        if not len(present):
            continue
        masked = np.where(np.isin(frame, present), frame, 0)
        for rp in regionprops(masked):
            area = float(rp.area)
            per = float(rp.perimeter)
            row: dict = {"track_id": int(rp.label), "frame": t}
            if frame_interval_s > 0:
                row["time_s"] = t * frame_interval_s
            for f in feats:
                if f == "area":
                    row[f"area_{au}"] = area * ls * ls
                elif f == "perimeter":
                    row[f"perimeter_{lu}"] = per * ls
                elif f == "circularity":
                    row[f] = min(1.0, 4 * np.pi * area / per**2) if per > 0 else np.nan
                elif f == "shape_index":
                    row[f] = per / np.sqrt(area) if area > 0 else np.nan
                elif f == "eccentricity":
                    row[f] = float(rp.eccentricity)
                elif f == "solidity":
                    row[f] = float(rp.solidity)
                elif f == "major_axis":
                    row[f"major_axis_{lu}"] = float(rp.major_axis_length) * ls
                elif f == "minor_axis":
                    row[f"minor_axis_{lu}"] = float(rp.minor_axis_length) * ls
                elif f == "orientation":
                    row[f] = float(rp.orientation)
                elif f == "centroid":
                    row[f"centroid_x_{lu}"] = float(rp.centroid[1]) * ls
                    row[f"centroid_y_{lu}"] = float(rp.centroid[0]) * ls
                elif f == "touches_edge":
                    y0, x0, y1, x1 = rp.bbox
                    row[f] = int(y0 == 0 or x0 == 0 or y1 == h or x1 == w)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["track_id", "frame"]).reset_index(drop=True)


def feature_cards() -> list[Path]:
    """Help card per feature, in FEATURES order (missing ones are skipped)."""
    out = []
    for key in FEATURES:
        for ext in (".png", ".jpg", ".jpeg"):
            f = HELP_DIR / f"{key}{ext}"
            if f.is_file():
                out.append(f)
                break
    return out


class FeatureGuide(QDialog):
    """Non-modal overview of all feature help cards."""

    def __init__(self, parent=None, columns: int = 3):
        super().__init__(parent)
        self.setWindowTitle("Feature guide")
        grid = QGridLayout()
        grid.setSpacing(8)
        for n, f in enumerate(feature_cards()):
            lbl = QLabel()
            lbl.setPixmap(QPixmap(str(f)))
            grid.addWidget(lbl, n // columns, n % columns)
        inner = QWidget()
        inner.setLayout(grid)
        inner.setStyleSheet("background: #262930;")
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(scroll)
        self.setLayout(lay)
        self.resize(min(columns, max(1, len(feature_cards()))) * 336 + 40, 560)


def value_column(df: pd.DataFrame) -> str:
    """Name of the single feature column in a one-feature compute_properties result."""
    return [c for c in df.columns if c not in ("track_id", "frame", "time_s")][0]


def colormap_image(cmap_name: str, width: int = 256, height: int = 14) -> QPixmap:
    import matplotlib

    cm = matplotlib.colormaps[cmap_name]
    rgb = (cm(np.linspace(0, 1, width))[:, :3] * 255).astype(np.uint8)
    rgb = np.ascontiguousarray(np.tile(rgb[np.newaxis], (height, 1, 1)))
    img = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


def fingerprint(stack: np.ndarray) -> dict:
    stack = as_tyx(stack)
    return {
        "shape": list(stack.shape),
        "max_label": int(stack.max()),
        "n_tracks": int(len(np.unique(stack)) - (1 if (stack == 0).any() else 0)),
    }


class CellPropertiesWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._selected: list[int] = []
        self._state_path: Path | None = None
        self._cb_added = False
        self.dir_provider = None  # callable -> Path | None (the results folder)
        self.color_feature: str | None = None
        self._hover_id = 0
        self._dimmed: tuple[Labels, float] | None = None

        self.labels_combo = QComboBox()
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setDecimals(4)
        self.pixel_size.setRange(0.0, 1000.0)
        self.pixel_size.setSpecialValueText("0 (use pixels)")
        self.interval = QDoubleSpinBox()
        self.interval.setDecimals(3)
        self.interval.setRange(0.0, 1e6)
        self.interval.setSpecialValueText("0 (use frames)")
        form = QFormLayout()
        form.addRow("Tracked labels", self.labels_combo)
        form.addRow("Pixel size (um)", self.pixel_size)
        form.addRow("Frame interval (s)", self.interval)

        self.select_cb = QCheckBox("Select cells (click a cell to add/remove)")
        self.select_cb.toggled.connect(self._select_toggled)
        self.hover = QLabel("Hover: -")
        self.hover.setWordWrap(True)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.ExtendedSelection)
        self.list.itemDoubleClicked.connect(self._jump_to_track)
        self.list.setMinimumHeight(110)
        remove = QPushButton("Remove highlighted")
        remove.clicked.connect(self._remove_highlighted)
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear)
        row = QHBoxLayout()
        row.addWidget(remove)
        row.addWidget(clear)

        self.feature_boxes: dict[str, QCheckBox] = {}
        self.color_buttons: dict[str, QToolButton] = {}
        fbox = QGroupBox("Features  (tick = export, Color = paint cells)")
        flay = QVBoxLayout()
        for key, (label, default) in FEATURES.items():
            cb = QCheckBox(label)
            cb.setChecked(default)
            cb.toggled.connect(self._autosave)
            cb.setToolTip(help_tooltip(key, label))
            self.feature_boxes[key] = cb
            frow = QHBoxLayout()
            frow.addWidget(cb, 1)
            if key in COLORABLE:
                btn = QToolButton()
                btn.setText("Color")
                btn.setCheckable(True)
                btn.setToolTip("Color the selected cells by this feature (fixed scale over the movie)")
                btn.toggled.connect(lambda on, k=key: self._color_toggled(k, on))
                self.color_buttons[key] = btn
                frow.addWidget(btn)
            flay.addLayout(frow)
        guide = QPushButton("Feature guide (all cards)")
        guide.setToolTip("Open an overview of all feature cards")
        guide.clicked.connect(self._show_guide)
        flay.addWidget(guide)
        fbox.setLayout(flay)
        self.legend_box = QGroupBox("Color legend")
        self.legend_title = QLabel("No feature selected for coloring.")
        self.legend_title.setWordWrap(True)
        self.legend_bar = QLabel()
        self.legend_bar.setFixedHeight(14)
        self.legend_bar.setScaledContents(True)
        self.legend_min = QLabel("")
        self.legend_max = QLabel("")
        self.legend_max.setAlignment(Qt.AlignRight)
        self.zero_cb = QCheckBox("Scale starts at 0")
        self.zero_cb.toggled.connect(lambda *_: self._rebuild())
        lrow = QHBoxLayout()
        lrow.addWidget(self.legend_min)
        lrow.addStretch(1)
        lrow.addWidget(self.legend_max)
        llay = QVBoxLayout()
        llay.addWidget(self.legend_title)
        llay.addWidget(self.legend_bar)
        llay.addLayout(lrow)
        llay.addWidget(self.zero_cb)
        self.legend_box.setLayout(llay)

        export = QPushButton("Export CSV...")
        export.clicked.connect(self._export_csv)
        save = QPushButton("Save selection...")
        save.clicked.connect(self._save_state_dialog)
        load = QPushButton("Load selection...")
        load.clicked.connect(self._load_state_dialog)
        srow = QHBoxLayout()
        srow.addWidget(save)
        srow.addWidget(load)

        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        lay = QVBoxLayout()
        lay.addLayout(form)
        lay.addWidget(self.select_cb)
        lay.addWidget(self.hover)
        lay.addWidget(QLabel("Selected tracks (double-click: jump to first frame)"))
        lay.addWidget(self.list)
        lay.addLayout(row)
        lay.addWidget(fbox)
        lay.addWidget(self.legend_box)
        lay.addWidget(export)
        lay.addLayout(srow)
        lay.addWidget(self.status)
        lay.addStretch(1)
        self.setLayout(lay)

        self.pixel_size.valueChanged.connect(lambda *_: self._rebuild())
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.labels_combo.currentTextChanged.connect(lambda *_: self._rebuild())
        self.viewer.dims.events.current_step.connect(self._on_frame_change)
        self.viewer.mouse_move_callbacks.append(self._on_move)
        self.viewer.mouse_drag_callbacks.append(self._on_click)
        self._refresh_layers()

    # ---- layers -----------------------------------------------------
    def _labels_layer(self) -> Labels | None:
        name = self.labels_combo.currentText()
        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Labels):
            return self.viewer.layers[name]
        return None

    def _stack(self) -> np.ndarray | None:
        layer = self._labels_layer()
        return None if layer is None else as_tyx(layer.data)

    def _refresh_layers(self, *_a) -> None:
        names = [
            ly.name
            for ly in self.viewer.layers
            if isinstance(ly, Labels) and ly.name != SELECTED_LAYER
        ]
        current = self.labels_combo.currentText()
        self.labels_combo.blockSignals(True)
        self.labels_combo.clear()
        self.labels_combo.addItems(names)
        if current in names:
            self.labels_combo.setCurrentText(current)
        elif "Tracked" in names:
            self.labels_combo.setCurrentText("Tracked")
        self.labels_combo.blockSignals(False)

    def _on_frame_change(self, *_a) -> None:
        self._hover_id = 0
        self._clear_hover()

    # ---- mouse ------------------------------------------------------
    def _value_at(self, event) -> int:
        layer = self._labels_layer()
        if layer is None:
            return 0
        val = layer.get_value(
            event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        return int(val) if val is not None else 0

    def _on_move(self, viewer, event) -> None:
        if not self.select_cb.isChecked():
            return
        v = self._value_at(event)
        if v == 0:
            self.hover.setText("Hover: -")
        else:
            mark = " (selected)" if v in self._selected else ""
            self.hover.setText(f"Hover: track {v}{mark}")
        if v != self._hover_id:
            self._hover_id = v
            self._update_hover(int(round(event.position[0])) if len(event.position) > 2 else 0)

    def _update_hover(self, t: int) -> None:
        """Show only the hovered cell (current frame) as a bright 2D layer."""
        src = self._labels_layer()
        stack = self._stack()
        if src is None or stack is None or self._hover_id == 0:
            self._clear_hover()
            return
        t = min(max(t, 0), stack.shape[0] - 1)
        data = (stack[t] == self._hover_id).astype(np.uint8)
        rgba = np.asarray(src.get_color(self._hover_id), dtype=float).copy()
        rgba[3] = 1.0  # same hue as the cell, fully opaque
        cmap = DirectLabelColormap(color_dict={None: [0, 0, 0, 0], 1: rgba})
        if HOVER_LAYER in self.viewer.layers:
            lay = self.viewer.layers[HOVER_LAYER]
            lay.colormap = cmap
            lay.data = data
        else:
            self.viewer.add_labels(
                data,
                name=HOVER_LAYER,
                opacity=0.85,
                scale=src.scale[-2:],
                translate=src.translate[-2:],
                colormap=cmap,
            )
            self.viewer.layers[HOVER_LAYER].editable = False
        self._raise_hover()

    def _clear_hover(self) -> None:
        if HOVER_LAYER in self.viewer.layers:
            self.viewer.layers.remove(HOVER_LAYER)

    def _raise_hover(self) -> None:
        layers = self.viewer.layers
        if HOVER_LAYER in layers and layers.index(HOVER_LAYER) != len(layers) - 1:
            active = layers.selection.active
            layers.move(layers.index(HOVER_LAYER), len(layers))
            if active is not None and active.name != HOVER_LAYER:
                layers.selection.active = active

    def _dim_background(self, on: bool) -> None:
        if on:
            layer = self._labels_layer()
            if layer is not None and self._dimmed is None:
                self._dimmed = (layer, layer.opacity)
                layer.opacity = DIMMED_OPACITY
        elif self._dimmed is not None:
            layer, op = self._dimmed
            if layer in self.viewer.layers:
                layer.opacity = op
            self._dimmed = None

    def _on_click(self, viewer, event):
        if not self.select_cb.isChecked() or "Shift" in event.modifiers:
            return
        moved = False
        yield
        while event.type == "mouse_move":
            moved = True
            yield
        if moved:  # a drag (pan), not a click
            return
        v = self._value_at(event)
        if v == 0:
            return
        if v in self._selected:
            self._selected.remove(v)
        else:
            self._selected.append(v)
        self._changed()

    def _select_toggled(self, on: bool) -> None:
        self._dim_background(on)
        if on:
            layer = self._labels_layer()
            if layer is not None:
                self.viewer.layers.selection.active = layer
            self.status.setText("Select mode on: click a cell to add or remove its track.")
        else:
            self.hover.setText("Hover: -")
            self._hover_id = 0
            self._clear_hover()
            self.status.setText("Select mode off.")

    # ---- selection --------------------------------------------------
    def _changed(self) -> None:
        self._rebuild()
        self._autosave()

    def _rebuild(self) -> None:
        stack = self._stack()
        frames = track_frames(stack, self._selected) if stack is not None else {}
        n = stack.shape[0] if stack is not None else 0
        self.list.clear()
        for i in self._selected:
            f = frames.get(i, [])
            if not f:
                text = f"Track {i}: not found in this layer"
            else:
                gaps = (f[-1] - f[0] + 1) - len(f)
                text = f"Track {i}: frames {f[0]}-{f[-1]} ({len(f)}/{n})"
                if gaps:
                    text += f", {gaps} gap(s)"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, i)
            self.list.addItem(item)
        self._update_overlay(stack)

    def _show_guide(self) -> None:
        if getattr(self, "_guide", None) is None:
            self._guide = FeatureGuide(self)
        self._guide.show()
        self._guide.raise_()
        self._guide.activateWindow()

    def _color_toggled(self, key: str, on: bool) -> None:
        if on:
            for k, btn in self.color_buttons.items():
                if k != key and btn.isChecked():
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
            self.color_feature = key
        elif self.color_feature == key:
            self.color_feature = None
        self._rebuild()

    def _color_overlay(self, stack: np.ndarray):
        """Instance-labelled stack (one label per selected track and frame) + colormap."""
        import matplotlib

        key = self.color_feature
        df = compute_properties(
            stack, self._selected, [key], self.pixel_size.value(), self.interval.value()
        )
        if df.empty:
            return None
        col = value_column(df)
        vals = df[col].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if not len(finite):
            return None
        vmin = 0.0 if self.zero_cb.isChecked() else float(finite.min())
        vmax = float(finite.max())
        if vmax <= vmin:
            vmax = vmin + 1e-9
        cm = matplotlib.colormaps[COLORABLE[key]]
        rgba = cm(np.clip((vals - vmin) / (vmax - vmin), 0, 1))
        rgba[~np.isfinite(vals)] = [0.5, 0.5, 0.5, 1.0]
        inst = np.arange(1, len(df) + 1, dtype=np.uint32)
        data = np.zeros(stack.shape, dtype=np.uint32)
        lut = np.zeros(int(stack.max()) + 1, dtype=np.uint32)
        for t, grp in df.groupby("frame"):
            lut[:] = 0
            lut[grp["track_id"].to_numpy()] = inst[grp.index.to_numpy()]
            data[int(t)] = lut[stack[int(t)]]
        color_dict = {None: [0, 0, 0, 0]}
        color_dict.update({int(i): rgba[i - 1] for i in inst})
        return data, DirectLabelColormap(color_dict=color_dict), col, vmin, vmax

    def _set_legend(self, info) -> None:
        if info is None:
            self.legend_title.setText("No feature selected for coloring.")
            self.legend_bar.clear()
            self.legend_min.setText("")
            self.legend_max.setText("")
            return
        _data, _cmap, col, vmin, vmax = info
        self.legend_title.setText(
            f"{col}: fixed scale over all frames of the {len(self._selected)} selected track(s)"
        )
        self.legend_bar.setPixmap(colormap_image(COLORABLE[self.color_feature]))
        self.legend_min.setText(f"{vmin:.3g}")
        self.legend_max.setText(f"{vmax:.3g}")

    def _update_overlay(self, stack: np.ndarray | None) -> None:
        if SELECTED_LAYER in self.viewer.layers:
            self.viewer.layers.remove(SELECTED_LAYER)
        src = self._labels_layer()
        if stack is None or src is None or not self._selected:
            self._set_legend(None)
            return
        info = self._color_overlay(stack) if self.color_feature else None
        self._set_legend(info)
        if info is not None:
            data, cmap = info[0], info[1]
        else:
            sel = np.isin(stack, np.asarray(self._selected, dtype=stack.dtype))
            data, cmap = np.where(sel, stack, 0), None
        kwargs = {"colormap": cmap} if cmap is not None else {}
        lay = self.viewer.add_labels(
            data,
            name=SELECTED_LAYER,
            opacity=0.9,
            scale=src.scale,
            translate=src.translate,
            **kwargs,
        )
        lay.editable = False
        lay.mouse_pan = True
        self.viewer.layers.selection.active = src
        self._raise_hover()

    def _remove_highlighted(self) -> None:
        for item in self.list.selectedItems():
            i = item.data(Qt.UserRole)
            if i in self._selected:
                self._selected.remove(i)
        self._changed()

    def _clear(self) -> None:
        self._selected.clear()
        self._changed()

    def _jump_to_track(self, item) -> None:
        stack = self._stack()
        if stack is None:
            return
        i = int(item.data(Qt.UserRole))
        f = track_frames(stack, [i]).get(i, [])
        if f:
            self.viewer.dims.set_point(0, f[0])

    # ---- export -----------------------------------------------------
    def _features(self) -> list[str]:
        return [k for k, cb in self.feature_boxes.items() if cb.isChecked()]

    def _default_dir(self) -> Path:
        if self.dir_provider is not None:
            d = self.dir_provider()
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)
                return d
        layer = self._labels_layer()
        src = getattr(getattr(layer, "source", None), "path", None) if layer else None
        if src:
            return Path(src).parent
        return Path.cwd()

    def _export_csv(self) -> None:
        stack = self._stack()
        if stack is None or not self._selected:
            self.status.setText("Select at least one cell first.")
            return
        if not self._features():
            self.status.setText("Tick at least one feature.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", str(self._default_dir() / "cell_properties.csv"), "CSV (*.csv)"
        )
        if not path:
            return
        df = compute_properties(
            stack,
            self._selected,
            self._features(),
            self.pixel_size.value(),
            self.interval.value(),
        )
        df.to_csv(path, index=False)
        self.status.setText(f"Wrote {len(df)} rows for {len(self._selected)} tracks to {path}")

    # ---- state ------------------------------------------------------
    def get_state(self) -> dict:
        stack = self._stack()
        return {
            "version": STATE_VERSION,
            "labels_layer": self.labels_combo.currentText(),
            "selected_track_ids": list(self._selected),
            "features": self._features(),
            "pixel_size_um": self.pixel_size.value(),
            "frame_interval_s": self.interval.value(),
            "labels_fingerprint": fingerprint(stack) if stack is not None else None,
        }

    def set_state(self, state: dict) -> str:
        msg = ""
        self._selected = [int(i) for i in state.get("selected_track_ids", [])]
        feats = set(state.get("features", []))
        for k, cb in self.feature_boxes.items():
            cb.blockSignals(True)
            cb.setChecked(k in feats)
            cb.blockSignals(False)
        self.pixel_size.setValue(float(state.get("pixel_size_um", 0.0)))
        self.interval.setValue(float(state.get("frame_interval_s", 0.0)))
        name = state.get("labels_layer")
        if name and self.labels_combo.findText(name) >= 0:
            self.labels_combo.setCurrentText(name)
        stack = self._stack()
        fp = state.get("labels_fingerprint")
        if stack is not None and fp and fp != fingerprint(stack):
            msg = (
                " Warning: the tracked labels differ from when this selection was saved "
                "(track IDs may have changed); check each track."
            )
        self._rebuild()
        return msg

    def _autosave(self, *_a) -> None:
        if self._state_path is not None:
            self._write_state(self._state_path)

    def _write_state(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.get_state(), indent=2) + "\n", encoding="utf-8")

    def _save_state_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save selection",
            str(self._default_dir() / "cell_selection.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        self._state_path = Path(path)
        self._write_state(self._state_path)
        self.status.setText(f"Saved to {path}. Further changes are saved there automatically.")

    def _load_state_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load selection", str(self._default_dir()), "JSON (*.json)"
        )
        if not path:
            return
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        warn = self.set_state(state)
        self._state_path = Path(path)
        self.status.setText(
            f"Loaded {len(self._selected)} tracks from {path}. "
            f"Changes are saved back automatically.{warn}"
        )
