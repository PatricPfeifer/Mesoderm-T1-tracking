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
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from skimage.measure import regionprops
from skimage.segmentation import expand_labels

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
    "aspect_ratio": "viridis",
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
    "aspect_ratio": ("Aspect ratio (major / minor)", False),
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
    inside=None,
) -> pd.DataFrame:
    """One row per selected track and frame. Lengths in um if pixel_size_um > 0.

    ``inside``: optional list (one set of track IDs per frame) of the cells in the analysis region; only those cells
    get a row in that frame."""
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
        if inside is not None and len(present):
            keep = inside[min(t, len(inside) - 1)]
            present = np.asarray([i for i in present if int(i) in keep], dtype=frame.dtype)
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
                elif f == "aspect_ratio":
                    minor = float(rp.minor_axis_length)
                    row[f] = float(rp.major_axis_length) / minor if minor > 0 else np.nan
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


JUNCTION_EXPAND = 3  # px: cells are grown by this much before their contact is counted (the T1 tab's Expand)


def junction_lengths(stack: np.ndarray, a: int, b: int, expand: int = JUNCTION_EXPAND) -> pd.DataFrame:
    """Length of the junction between tracks ``a`` and ``b`` in every frame in which they touch.

    The same measure as the T1 tab: both cells (with their neighbours) are grown by ``expand`` px to close the
    gaps between segmented cells, then the 4-connected contact pixels between the two are counted. Frames in which
    they do not touch (or one is missing) have no row. Columns: frame, contact_px.
    """
    stack = as_tyx(stack)
    a, b = int(a), int(b)
    pad = int(expand) + 2
    rows = []
    for t in range(stack.shape[0]):
        frame = stack[t]
        mask = (frame == a) | (frame == b)
        if not (np.any(frame == a) and np.any(frame == b)):
            continue
        ys, xs = np.nonzero(mask)
        y0, y1 = max(0, ys.min() - pad), min(frame.shape[0], ys.max() + 1 + pad)
        x0, x1 = max(0, xs.min() - pad), min(frame.shape[1], xs.max() + 1 + pad)
        crop = frame[y0:y1, x0:x1]
        grown = expand_labels(crop, distance=expand) if expand > 0 else crop
        ga, gb = grown == a, grown == b
        n = int(np.count_nonzero(ga[:, :-1] & gb[:, 1:]) + np.count_nonzero(gb[:, :-1] & ga[:, 1:])
                + np.count_nonzero(ga[:-1] & gb[1:]) + np.count_nonzero(gb[:-1] & ga[1:]))
        if n > 0:
            rows.append({"frame": t, "contact_px": n})
    return pd.DataFrame(rows, columns=["frame", "contact_px"])


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
        self.region_provider = None  # callable -> (per-frame sets of track IDs inside, region hash) | None
        self.on_exported = None  # called after an export (analysis-region bookkeeping)
        self.region_hash_used: str | None = None  # region of the last export in this session
        self.color_feature: str | None = None
        self._hover_id = 0
        self._dimmed: tuple[Labels, float] | None = None
        self._junctions: list[tuple[int, int]] = []  # pairs of track IDs whose junction length is followed
        self._picking: list[int] | None = None  # None = not adding a junction; else the cell(s) clicked so far
        self._auto_cal = (0.0, 0.0)  # what was last filled in automatically (pixel size, frame interval)
        # Computed feature table + instance stack, reused while only the colour
        # scale changes. Valid for one (layer, data version, selection, feature,
        # calibration) key; the version is bumped whenever a labels layer changes.
        self._overlay_cache: dict | None = None
        self._data_version = 0
        self._watched: set[int] = set()

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
        select_all = QPushButton("Select all cells")
        select_all.setToolTip("Add every track in the tracked labels to the selection")
        select_all.clicked.connect(self._select_all)
        self.use_region_cb = QCheckBox("Only cells inside the analysis region")
        self.use_region_cb.setChecked(True)
        self.use_region_cb.setToolTip(
            "Use the area chosen in the Analysis cropping tab: 'Select all cells' picks only cells inside it, and "
            "exports only list a cell in the frames where it is inside. No effect when no region is set. "
            "Export again after changing the region."
        )
        row = QHBoxLayout()
        row.addWidget(select_all)
        row.addWidget(remove)
        row.addWidget(clear)

        self.feature_boxes: dict[str, QCheckBox] = {}
        self.color_buttons: dict[str, QToolButton] = {}
        fbox = QGroupBox("2 · Features")
        flay = QVBoxLayout()
        flay.addWidget(QLabel("Tick = export, Color = paint cells"))
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
        # manual colour-scale limits per feature: {key: {"lo": float, "hi": float}}
        self._manual_range: dict[str, dict[str, float]] = {}
        self.legend_min = self._limit_box("lo")
        self.legend_max = self._limit_box("hi")
        self.legend_max.setAlignment(Qt.AlignRight)
        self.legend_reset = QToolButton()
        self.legend_reset.setText("Auto")
        self.legend_reset.setToolTip("Reset min and max to the automatic range")
        self.legend_reset.clicked.connect(self._reset_range)
        self.zero_cb = QCheckBox("Scale starts at 0")
        self.zero_cb.toggled.connect(lambda *_: self._recolor())
        lrow = QHBoxLayout()
        lrow.addWidget(self.legend_min)
        lrow.addStretch(1)
        lrow.addWidget(self.legend_reset)
        lrow.addStretch(1)
        lrow.addWidget(self.legend_max)
        llay = QVBoxLayout()
        llay.addWidget(self.legend_title)
        llay.addWidget(self.legend_bar)
        llay.addLayout(lrow)
        llay.addWidget(self.zero_cb)
        self.legend_box.setLayout(llay)

        jbox = QGroupBox("3 · Junction length over time")
        jlay = QVBoxLayout()
        self.junction_add = QPushButton("Add junction")
        self.junction_add.setCheckable(True)
        self.junction_add.setToolTip(
            "Click, then click two neighbouring cells. The junction between them is measured in every frame in "
            "which they touch."
        )
        self.junction_add.toggled.connect(self._junction_add_toggled)
        self.junction_expand = QSpinBox()
        self.junction_expand.setRange(0, 20)
        self.junction_expand.setValue(JUNCTION_EXPAND)
        self.junction_expand.setToolTip("Cells are grown by this many pixels before their contact is counted (as the T1 tab's Expand).")
        jform = QFormLayout()
        jform.addRow("Expand (px)", self.junction_expand)
        self.junction_list = QListWidget()
        self.junction_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.junction_list.setMinimumHeight(70)
        self.junction_list.itemDoubleClicked.connect(self._jump_to_junction)
        junction_remove = QPushButton("Remove highlighted")
        junction_remove.clicked.connect(self._remove_junctions)
        junction_export = QPushButton("Export junction lengths...")
        junction_export.setToolTip("Writes a CSV (one row per junction and frame) and a PNG plot of length over time.")
        junction_export.clicked.connect(self._export_junctions)
        jlay.addWidget(self.junction_add)
        jlay.addLayout(jform)
        jlay.addWidget(QLabel("Junctions (double-click: jump to the first frame they touch)"))
        jlay.addWidget(self.junction_list)
        jlay.addWidget(junction_remove)
        jlay.addWidget(junction_export)
        jbox.setLayout(jlay)

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

        select_box = QGroupBox("1 · Select cells")
        sel_lay = QVBoxLayout()
        sel_lay.addLayout(form)
        sel_lay.addWidget(self.select_cb)
        sel_lay.addWidget(self.use_region_cb)
        sel_lay.addWidget(self.hover)
        sel_lay.addWidget(QLabel("Selected tracks (double-click: jump to first frame)"))
        sel_lay.addWidget(self.list)
        sel_lay.addLayout(row)
        select_box.setLayout(sel_lay)

        export_box = QGroupBox("4 · Export")
        exp_lay = QVBoxLayout()
        exp_lay.addWidget(export)
        exp_lay.addLayout(srow)
        export_box.setLayout(exp_lay)

        lay = QVBoxLayout()
        lay.addWidget(select_box)
        lay.addWidget(fbox)
        lay.addWidget(self.legend_box)
        lay.addWidget(jbox)
        lay.addWidget(export_box)
        lay.addWidget(self.status)
        lay.addStretch(1)
        self.setLayout(lay)

        self.pixel_size.valueChanged.connect(lambda *_: self._rebuild())
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.labels_combo.currentTextChanged.connect(lambda *_: self._rebuild())
        self.labels_combo.currentTextChanged.connect(lambda *_: self._junctions_changed())
        self.viewer.dims.events.current_step.connect(self._on_frame_change)
        self.viewer.mouse_move_callbacks.append(self._on_move)
        self.viewer.mouse_drag_callbacks.append(self._on_click)
        self._refresh_layers()

    def autofill_calibration(self, um_per_px, s_per_frame) -> None:
        """Fill the pixel size / frame interval boxes from the movie. A value typed by hand is left alone;
        a value that was filled in automatically before is replaced when another movie is loaded."""
        old_um, old_dt = self._auto_cal
        for box, value, old in ((self.pixel_size, um_per_px, old_um), (self.interval, s_per_frame, old_dt)):
            if value and (box.value() == 0.0 or abs(box.value() - old) < 1e-9):
                box.setValue(float(value))
        self._auto_cal = (float(um_per_px or 0.0) if self.pixel_size.value() else 0.0,
                          float(s_per_frame or 0.0) if self.interval.value() else 0.0)

    # ---- layers -----------------------------------------------------
    def _labels_layer(self) -> Labels | None:
        name = self.labels_combo.currentText()
        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Labels):
            return self.viewer.layers[name]
        return None

    def _stack(self) -> np.ndarray | None:
        layer = self._labels_layer()
        return None if layer is None else as_tyx(layer.data)

    def _bump_version(self, *_a) -> None:
        self._data_version += 1

    def _watch_layers(self) -> None:
        """Invalidate the overlay cache when any labels layer is repainted or replaced."""
        for ly in self.viewer.layers:
            if isinstance(ly, Labels) and ly.name not in (SELECTED_LAYER, HOVER_LAYER) and id(ly) not in self._watched:
                self._watched.add(id(ly))
                for ev in ("data", "set_data", "paint"):
                    getattr(ly.events, ev).connect(self._bump_version)

    def _refresh_layers(self, *_a) -> None:
        self._watch_layers()
        names = [
            ly.name
            for ly in self.viewer.layers
            if isinstance(ly, Labels) and ly.name not in (SELECTED_LAYER, HOVER_LAYER)
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

    def _frame_key(self) -> tuple:
        """The non-spatial part of the current dims position (the time frame)."""
        return tuple(self.viewer.dims.current_step[:-2])

    def _on_frame_change(self, *_a) -> None:
        # Adding/removing a layer (e.g. the 2D hover layer itself) re-emits current_step without the
        # frame actually changing; only drop the hover when the frame really moved.
        key = self._frame_key()
        if key == getattr(self, "_hover_frame", None):
            return
        self._hover_frame = key
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
        if not (self.select_cb.isChecked() or self._picking is not None):
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
        self._hover_frame = self._frame_key()
        if HOVER_LAYER in self.viewer.layers:
            lay = self.viewer.layers[HOVER_LAYER]
            lay.colormap = cmap
            lay.data = data
        else:
            lay = self.viewer.add_labels(
                data,
                name=HOVER_LAYER,
                opacity=0.85,
                scale=src.scale[-2:],
                translate=src.translate[-2:],
                colormap=cmap,
            )
            lay.editable = False
        self._raise_hover()

    def _clear_hover(self) -> None:
        if HOVER_LAYER in self.viewer.layers:
            self.viewer.layers.remove(HOVER_LAYER)

    def _raise_hover(self) -> None:
        layers = self.viewer.layers
        try:
            idx = layers.index(HOVER_LAYER)
        except ValueError:
            return
        if idx != len(layers) - 1:
            active = layers.selection.active
            layers.move(idx, len(layers))
            if active is not None and active.name != HOVER_LAYER and active in layers:
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
        if not (self.select_cb.isChecked() or self._picking is not None) or "Shift" in event.modifiers:
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
        if self._picking is not None:
            self._junction_clicked(v)
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

    # ---- junctions --------------------------------------------------
    def _junction_add_toggled(self, on: bool) -> None:
        if on:
            layer = self._labels_layer()
            if layer is None:
                self.junction_add.blockSignals(True)
                self.junction_add.setChecked(False)
                self.junction_add.blockSignals(False)
                self.status.setText("Select a tracked labels layer first.")
                return
            self._picking = []
            self.viewer.layers.selection.active = layer
            self.junction_add.setText("Click the first cell...")
            self.status.setText("Click the first of the two cells.")
        else:
            self._picking = None
            self.junction_add.setText("Add junction")
            self.hover.setText("Hover: -")
            self._hover_id = 0
            if not self.select_cb.isChecked():
                self._clear_hover()

    def _junction_clicked(self, track_id: int) -> None:
        if not self._picking:
            self._picking = [int(track_id)]
            self.junction_add.setText("Click the second cell...")
            self.status.setText(f"Cell {track_id} picked. Click the cell it shares a junction with.")
            return
        first = self._picking[0]
        if int(track_id) == first:
            self.status.setText("That is the same cell: click a different one.")
            return
        pair = (min(first, int(track_id)), max(first, int(track_id)))
        self.junction_add.setChecked(False)  # also resets the picking state
        if pair in self._junctions:
            self.status.setText(f"Junction {pair[0]}-{pair[1]} is already in the list.")
            return
        self._junctions.append(pair)
        self._junctions_changed()
        df = self._junction_table(pair)
        if df.empty:
            self.status.setText(
                f"Added {pair[0]}-{pair[1]}, but these cells never touch (try a larger Expand if they should)."
            )
        else:
            self.status.setText(
                f"Added junction {pair[0]}-{pair[1]}: touching in {len(df)} frames ({int(df.frame.min())}-{int(df.frame.max())})."
            )

    def _junction_table(self, pair) -> pd.DataFrame:
        stack = self._stack()
        if stack is None:
            return pd.DataFrame(columns=["frame", "contact_px"])
        return junction_lengths(stack, pair[0], pair[1], int(self.junction_expand.value()))

    def _junctions_changed(self) -> None:
        self.junction_list.clear()
        for pair in self._junctions:
            df = self._junction_table(pair)
            if df.empty:
                text = f"Junction {pair[0]} - {pair[1]}: never touching"
            else:
                f0, f1 = int(df.frame.min()), int(df.frame.max())
                text = f"Junction {pair[0]} - {pair[1]}: frames {f0}-{f1} ({len(df)} touching)"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, pair)
            self.junction_list.addItem(item)
        self._autosave()

    def _jump_to_junction(self, item) -> None:
        df = self._junction_table(item.data(Qt.UserRole))
        if not df.empty:
            self.viewer.dims.set_point(0, int(df.frame.min()))

    def _remove_junctions(self) -> None:
        for item in self.junction_list.selectedItems():
            pair = tuple(item.data(Qt.UserRole))
            if pair in self._junctions:
                self._junctions.remove(pair)
        self._junctions_changed()

    def junction_long_table(self) -> pd.DataFrame:
        """One row per junction and frame in which the two cells touch."""
        um = self.pixel_size.value()
        dt = self.interval.value()
        parts = []
        for a, b in self._junctions:
            df = self._junction_table((a, b))
            if df.empty:
                continue
            df.insert(0, "cell_b", b)
            df.insert(0, "cell_a", a)
            if dt > 0:
                df.insert(3, "time_s", df["frame"] * dt)
            df["length_um" if um > 0 else "length_px"] = df["contact_px"] * (um if um > 0 else 1.0)
            parts.append(df)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def _export_junctions(self) -> None:
        df = self.junction_long_table()
        if df.empty:
            self.status.setText("Add at least one junction whose cells touch.")
            return
        region = self._region()
        if region is not None:
            inside = region[0]
            keep = [int(f) in range(len(inside)) and int(a) in inside[int(f)] and int(b) in inside[int(f)]
                    for f, a, b in zip(df["frame"], df["cell_a"], df["cell_b"])]
            df = df[keep]
            if df.empty:
                self.status.setText("No junction has both cells inside the analysis region.")
                return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export junction lengths", str(self._default_dir() / "junction_lengths.csv"), "CSV (*.csv)"
        )
        if not path:
            return
        df.to_csv(path, index=False)
        png = Path(path).with_suffix(".png")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xcol = "time_s" if "time_s" in df else "frame"
            ycol = "length_um" if "length_um" in df else "length_px"
            fig, ax = plt.subplots(figsize=(7, 4))
            for (a, b), grp in df.groupby(["cell_a", "cell_b"]):
                ax.plot(grp[xcol], grp[ycol], marker="o", ms=3, label=f"{a} - {b}")
            ax.set_xlabel("time (s)" if xcol == "time_s" else "frame")
            ax.set_ylabel("junction length (um)" if ycol == "length_um" else "junction length (px)")
            ax.legend(title="cells", fontsize=8)
            fig.tight_layout()
            fig.savefig(png, dpi=150)
            plt.close(fig)
            tail = f" and plot {png.name}"
        except Exception as exc:  # the CSV is what matters
            tail = f" (plot not written: {exc})"
        self.status.setText(f"Wrote {len(df)} rows for {len(self._junctions)} junction(s) to {path}{tail}")
        self._exported(region)

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

    def _cache_key(self) -> tuple:
        return (
            id(self._labels_layer()),
            self._data_version,
            tuple(self._selected),
            self.color_feature,
            self.pixel_size.value(),
            self.interval.value(),
        )

    def _color_base(self, stack: np.ndarray) -> dict | None:
        """Feature values and instance-labelled stack; cached while the inputs are unchanged."""
        key = self._cache_key()
        cache = self._overlay_cache
        if cache is not None and cache["key"] == key:
            return cache
        self._overlay_cache = None
        df = compute_properties(
            stack,
            self._selected,
            [self.color_feature],
            self.pixel_size.value(),
            self.interval.value(),
        )
        if df.empty:
            return None
        col = value_column(df)
        vals = df[col].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if not len(finite):
            return None
        inst = np.arange(1, len(df) + 1, dtype=np.uint32)
        data = np.zeros(stack.shape, dtype=np.uint32)
        lut = np.zeros(int(stack.max()) + 1, dtype=np.uint32)
        for t, grp in df.groupby("frame"):
            lut[:] = 0
            lut[grp["track_id"].to_numpy()] = inst[grp.index.to_numpy()]
            data[int(t)] = lut[stack[int(t)]]
        self._overlay_cache = {
            "key": key,
            "col": col,
            "vals": vals,
            "inst": inst,
            "data": data,
            "lo": float(finite.min()),
            "hi": float(finite.max()),
        }
        return self._overlay_cache

    def _color_overlay(self, stack: np.ndarray):
        """Instance-labelled stack (one label per selected track and frame) + colormap."""
        import matplotlib

        key = self.color_feature
        base = self._color_base(stack)
        if base is None:
            return None
        col, vals, inst, data = base["col"], base["vals"], base["inst"], base["data"]
        vmin = 0.0 if self.zero_cb.isChecked() else base["lo"]
        vmax = base["hi"]
        manual = self._manual_range.get(key, {})
        vmin = manual.get("lo", vmin)
        vmax = manual.get("hi", vmax)
        if vmax <= vmin:
            vmax = vmin + 1e-9
        cm = matplotlib.colormaps[COLORABLE[key]]
        rgba = cm(np.clip((vals - vmin) / (vmax - vmin), 0, 1))
        rgba[~np.isfinite(vals)] = [0.5, 0.5, 0.5, 1.0]
        color_dict = {None: [0, 0, 0, 0]}
        color_dict.update({int(i): rgba[i - 1] for i in inst})
        return data, DirectLabelColormap(color_dict=color_dict), col, vmin, vmax

    def _limit_box(self, which: str) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-1e9, 1e9)
        box.setDecimals(4)
        box.setButtonSymbols(QDoubleSpinBox.NoButtons)
        box.setKeyboardTracking(False)
        box.setMaximumWidth(90)
        box.setEnabled(False)
        box.setToolTip("Click to type a value and amplify a range; values beyond it saturate")
        box.valueChanged.connect(lambda v, w=which: self._limit_edited(w, v))
        return box

    def _limit_edited(self, which: str, value: float) -> None:
        if self.color_feature is None:
            return
        self._manual_range.setdefault(self.color_feature, {})[which] = float(value)
        self._recolor()

    def _reset_range(self) -> None:
        if self.color_feature is not None:
            self._manual_range.pop(self.color_feature, None)
            self._recolor()

    def _recolor(self) -> None:
        """Re-apply the colour scale to the existing overlay layer without recomputing."""
        if self.color_feature is None:
            return
        cache = self._overlay_cache
        stack = self._stack()
        if (
            stack is None
            or cache is None
            or cache["key"] != self._cache_key()
            or SELECTED_LAYER not in self.viewer.layers
        ):
            self._rebuild()
            return
        info = self._color_overlay(stack)
        if info is None:
            self._rebuild()
            return
        self.viewer.layers[SELECTED_LAYER].colormap = info[1]
        self._set_legend(info)

    def _set_legend(self, info) -> None:
        boxes = (self.legend_min, self.legend_max)
        for b in boxes:
            b.blockSignals(True)
        try:
            if info is None:
                self.legend_title.setText("No feature selected for coloring.")
                self.legend_bar.clear()
                for b in boxes:
                    b.setValue(0.0)
                    b.setEnabled(False)
                self.legend_reset.setEnabled(False)
                return
            _data, _cmap, col, vmin, vmax = info
            manual = bool(self._manual_range.get(self.color_feature))
            scale = "manual scale" if manual else "fixed scale over all frames"
            self.legend_title.setText(
                f"{col}: {scale} of the {len(self._selected)} selected track(s)"
            )
            self.legend_bar.setPixmap(colormap_image(COLORABLE[self.color_feature]))
            self.legend_min.setValue(vmin)
            self.legend_max.setValue(vmax)
            for b in boxes:
                b.setEnabled(True)
            self.legend_reset.setEnabled(manual)
        finally:
            for b in boxes:
                b.blockSignals(False)

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

    def _region(self):
        """(per-frame inside sets, hash) when the region is set and wanted for the export, else None."""
        if self.region_provider is None or not self.use_region_cb.isChecked():
            return None
        return self.region_provider()

    def _exported(self, region) -> None:
        self.region_hash_used = region[1] if region else ""
        if callable(self.on_exported):
            self.on_exported()

    def _select_all(self) -> None:
        stack = self._stack()
        if stack is None:
            self.status.setText("Select a tracked labels layer first.")
            return
        ids = np.unique(stack)
        ids = [int(i) for i in ids if i != 0]
        region = self._region()
        if region is not None:
            ever_inside = set().union(*region[0]) if region[0] else set()
            ids = [i for i in ids if i in ever_inside]
        self._selected = ids
        self._changed()
        where = " inside the analysis region" if region is not None else ""
        self.status.setText(f"Selected all {len(self._selected)} tracks{where}.")

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
        region = self._region()
        df = compute_properties(
            stack,
            self._selected,
            self._features(),
            self.pixel_size.value(),
            self.interval.value(),
            inside=region[0] if region else None,
        )
        df.to_csv(path, index=False)
        where = " (only cells inside the analysis region)" if region else ""
        self.status.setText(f"Wrote {len(df)} rows for {len(self._selected)} tracks{where} to {path}")
        self._exported(region)

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
            "junctions": [list(p) for p in self._junctions],
            "junction_expand": int(self.junction_expand.value()),
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
        self._junctions = [tuple(sorted(int(i) for i in p)) for p in state.get("junctions", [])]
        self.junction_expand.setValue(int(state.get("junction_expand", JUNCTION_EXPAND)))
        self._rebuild()
        self._junctions_changed()
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
