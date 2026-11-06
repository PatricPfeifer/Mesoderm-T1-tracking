"""Napari dock: Cellpose cpsam, cellpose_track, and the original T1 detector."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import traceback
from collections import deque
from pathlib import Path

import napari
import numpy as np
from napari.layers import Image, Labels
from qtpy.QtCore import QObject, QProcess, QSize, QThread, QTimer, Qt, Signal
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .cell_properties import CellPropertiesWidget
from .core.cell_size import MIN_CELL_AREA
from .core.project import PER_RUN_STEPS, WorkingFolder, find_results_root, movie_stem
from .remote_widget import RemoteRunner
from .track_edit import (
    CellUndo,
    ProtectedCellsError,
    break_track,
    changed_pixels,
    link_tracks,
    merge_protected,
    restore_cells,
    swap_tracks,
)


class _Job(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            self.finished.emit(self._fn(self.progress.emit))
        except Exception:
            self.failed.emit(traceback.format_exc())


def _as_tyx(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 4 and arr.shape[-1] <= 4:
        arr = arr[..., 0]
    elif arr.ndim == 4 and arr.shape[1] <= 4:
        arr = arr[:, 0]
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"expected T,Y,X image, got {arr.shape}")
    return arr


def _as_tyx_shape(shape, rgb: bool = False) -> tuple[int, int, int]:
    """T,Y,X of a layer from its shape only (no pixel loading), like ``_as_tyx``."""
    shape = tuple(int(s) for s in shape)
    if rgb or (len(shape) == 4 and shape[-1] <= 4):
        shape = shape[:-1]
    elif len(shape) == 4 and shape[1] <= 4:
        shape = (shape[0], *shape[2:])
    if len(shape) == 2:
        shape = (1, *shape)
    return shape[0], shape[-2], shape[-1]


def _as_rgb(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[np.newaxis, ...]
    if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
        arr = np.moveaxis(arr, 1, -1)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected T,Y,X,3 RGB, got {arr.shape}")
    return arr


HELP_HTML = """
<h3>Workflow</h3>
<ol>
<li><b>Open the movie</b> in napari, or press <b>Folder...</b> and pick its <code>*_results</code> folder: saved results load and the drop-downs fill in by themselves.</li>
<li><b>Segmentation</b>: find the cells and correct their outlines.</li>
<li><b>Tracking</b>: give every cell the same ID in all frames.</li>
<li><b>T1</b>: find neighbour exchanges (T1 transitions).</li>
</ol>
<p>Everything is saved in the results folder next to the movie. Progress and errors show in the line at the bottom.</p>

<h3>Top bar</h3>
<ul>
<li><b>Folder...</b>: choose the results folder. If it holds saved results they are loaded.</li>
<li><b>Load results</b>: load the saved movie, segmentation, outlines and tracking.</li>
<li><b>Help</b>: this window.</li>
</ul>

<h3>Segmentation tab</h3>
<ul>
<li><b>Image</b>: the movie. <b>Run locally</b> finds the cells (Cellpose) on this computer; <b>Run on cluster</b> does it on a bwUniCluster GPU (<b>Cluster settings</b>: fetch results, job log, settings, one-time setup). <b>Preset</b>: saved settings (<i>default</i> = Cellpose's own, <i>cellflow</i>, <i>hpc</i> = big batches for the cluster); open <b>Parameters</b> to change single values. <i>Batch size</i>: 256x256 px tiles per GPU pass (tiles within a frame; speed and GPU memory only, not the result). Run locally uses this computer's GPU when it has one, else the CPU; the cluster always uses its GPU. <i>Fill enclosed holes as new cells</i> (under Parameters): empty space that is closed in by cells becomes a cell of its own (empty space that reaches the image edge, like the outside of the embryo, never does). Every run is saved in its own folder; <b>Runs</b>: <i>Show</i> makes a run the one you edit and track, <i>Compare</i> overlays another run's outlines.</li>
<li><b>Masks / Outlines</b>: the cell layer and the outline layer to edit (set automatically after loading).</li>
<li><b>Start outline editing (for separately imported outlines)</b>: only needed for outlines or masks you dragged in yourself and picked above; after Run locally / Run on cluster or loading a results folder editing starts by itself. It shows the yellow outlines for editing. Without masks the cells are the regions between the outlines. Wait for 100% before editing.</li>
<li><b>Paint / Erase / Pan-zoom</b>: pick the tool. Paint adds a bond, Erase removes one. With Paint, the <b>right mouse button</b> erases.</li>
<li><b>Apply bond edits (Enter)</b>: turn your painted or erased bonds into cell changes.</li>
<li><b>Save segmentation (Ctrl+S)</b>: write the masks and the 1 px outlines (<code>*_cp_masks.tif</code>, <code>*_cell_outlines.tif</code>).</li>
<li><b>Export TIFF</b>: same as Save.</li>
<li>A cell needs at least <b>15 px</b> in every step (segmentation, outline edits, manual drawing, tracking); smaller regions are dropped or refused.</li>
<li><b>Ctrl/Cmd+Z</b> takes back the last paint stroke that is not applied yet; otherwise the last applied bond edit (Enter) or cell deletion.</li>
<li><b>M</b> hides or shows the outlines. <i>Watershed pad</i>: area around an edit that is recomputed (smaller = faster).</li>
<li><b>Delete cell</b>: hold <b>Option</b> and left-click a cell to delete it (Cmd/Ctrl+Z undoes it).</li>
</ul>

<h3>Tracking tab</h3>
<ul>
<li><b>Labels</b>: the segmentation to track. <b>Intensity</b>: the movie (or none).</li>
<li><b>Run tracking</b>: track the cells. The defaults are fine to start with; hover over a setting for what it does.</li>
<li><b>Divisions</b>: dividing cells end their track and the two daughters get new IDs.</li>
<li><b>Export TIFF</b>: write the tracked layers to the results folder.</li>
<li><b>Correct tracks</b>: turn on <i>Edit tracks</i>, pick a mode and click cells on the Tracked layer.
 <i>Link</i> joins two tracks, <i>Break</i> starts a new track, <i>Swap</i> exchanges two IDs.
 <b>Undo</b> takes back the last edit, <b>Clear protection (lets corrected cells be changed again)</b>: corrected cells are protected so a later edit on a neighbour can't silently change them; this lifts that. <b>Save corrected tracks</b> writes the corrected tracking (the old files go to a <code>previous_...</code> folder).</li>
</ul>

<h3>T1 tab</h3>
<ul>
<li><b>Labels</b>: the Tracked layer. <b>RGB</b>: the Tracked RGB image used for the overlay.</li>
<li><b>Run T1</b>: detect T1 transitions. Needs the pixel size and frame interval. They are read from the movie or from the results files (also ones made on a server); the line under <i>RGB</i> shows what was found. If a value is missing, type it in the boxes.</li>
<li><b>T1 filter</b>: show all counted events, only clean ones, flicker resolved, or flicker aborted.</li>
<li><b>Overlay pre / post</b>: frames shown before and after each event. <b>Expand</b>: how far cells are grown to find contacts.</li>
<li><b>Exclude dividing cells</b>: leave out mothers and daughters (needs tracking with Divisions).</li>
<li><b>Export TIFF</b>: write the T1 layers to the results folder.</li>
</ul>

<h3>Export Cell Props tab</h3>
<p>Measure cell properties on the Tracked layer. The pixel size and frame interval are filled in from the movie (change them if needed). Click cells (<i>Select cells</i>) or use <b>Select all cells</b>,
tick the features you want and use <b>Color</b> to colour the cells by a feature. <b>Feature guide</b> explains each feature.
<b>Export CSV...</b> writes the table. <b>Add junction</b>: click it, then click two neighbouring cells; the length of the junction between them is measured in every frame in which they touch (listed under <i>Junction length over time</i>, exported as CSV + plot with <b>Export junction lengths...</b>). <b>Save selection...</b> / <b>Load selection...</b> keep your chosen cells for later.</p>
"""

TRACK_TIPS = {
    "expand": (
        "Pixels each cell is grown before neighbours are looked up. Bridges the "
        "thin background gap between Cellpose masks so touching cells count as "
        "neighbours. Too large adds false neighbours."
    ),
    "depth": (
        "Levels of the motion-correction pyramid. The movie shift between frames "
        "is measured on 256 px tiles, then 128, then 64 (depth 3). Higher is "
        "finer but slower and noisier; the finest tile is 16 px."
    ),
    "w_topo": (
        "How much a cell must keep the same neighbours as in the previous frame. "
        "0 = only overlap and distance count; higher = neighbour layout counts more."
    ),
    "iter": (
        "Maximum rounds of re-matching with the neighbour penalty. Stops early "
        "when the neighbour score no longer improves."
    ),
    "appear": (
        "Cost of starting a new track for a cell with no match in the previous "
        "frame. Higher forces more matches; lower allows more new tracks."
    ),
    "disappear": (
        "Cost of ending a track whose cell has no match in the next frame. "
        "Higher keeps tracks alive more; lower ends them sooner."
    ),
    "division": (
        "Extra cost for giving a cell a second daughter. Higher means fewer "
        "divisions. (1.0 is used when 'Divisions: new daughter IDs' is on.)"
    ),
    "gap": (
        "Maximum number of frames a track may vanish and still be reconnected. "
        "0 turns gap closing off."
    ),
    "jobs": (
        "CPU threads used for the per-frame steps (cell tables and motion "
        "estimation). 1 = one core. Results are identical; only speed changes."
    ),
    "dist": (
        "Distance scale in pixels for matching. A cell may only link to cells "
        "within this distance (unless they overlap strongly); larger distances "
        "also count less against a link."
    ),
}


class TissueKitWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._thread: QThread | None = None
        self._job: _Job | None = None
        self._proc: QProcess | None = None
        self._seg_dir: Path | None = None
        self._seg_out: Path | None = None
        self._seg_base_masks: np.ndarray | None = None
        self._seg_base_outlines: np.ndarray | None = None
        self._autosave_outlines_ok = True  # False while editing outlines loaded from a file: never rewrite the file before the user saves
        self._pending_outlines: np.ndarray | None = None  # imported outlines shown after the bonds are built; their differences apply on Enter
        self._session_cal = None  # pixel size / frame interval of the movie in use (from any file read or run)
        self._seg_cal = None  # calibration read from a loaded outlines file (pixel size / frame interval)
        # True while the plugin itself pushes masks into the correction widget, so
        # only the user's cell edits are mirrored onto the Outlines layer.
        self._outline_sync_busy = False
        self._last_job_error = ""
        # TA handCorrection stack (255 = 1 px bond / background): the bonds
        # shown in Outlines and used by the watershed.
        self._seg_base_hand: np.ndarray | None = None
        self._hand_msg = ""
        self._seg_stem: str = "segmentation"
        # Segmentation run (segmentation/<run>/) the Segmentation layer belongs to; None = not saved yet.
        self._seg_run: str | None = None
        self._seg_pending: dict | None = None  # settings of a finished run, written when it is saved
        self._compare_colors = iter(())  # colours for "Add for comparison" layers
        self._track_pick: tuple[int, int, int, int] | None = None
        self._track_undo: deque = deque(maxlen=20)
        # Pixels set by manual edits, one record per edit (same order as the undo
        # stack). Later edits may not change them as a side effect.
        self._protect_log: list = []
        self._overlay_msg = ""  # second line of the viewer text overlay
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_flash)
        self._t1_result: dict | None = None
        self.project = WorkingFolder()
        self._track_run: tuple | None = None
        # Mother + daughter track IDs from the last division-aware tracking.
        self._division_ids: set[int] = set()
        # The tracked_labels file the "Tracked" layer came from (the originals
        # that "Save corrected tracks" compares against), and unsaved-edit flag.
        self._tracking_files: Path | None = None
        self._track_dirty = False
        # Segmentation run the shown tracking belongs to (tracking/<run>/, t1/<run>/).
        self._tracked_run: str | None = None
        # Cluster (bwUniCluster) segmentation runs in its own thread, so local work goes on.
        self._remote = RemoteRunner(self)
        self._remote_tmp: Path | None = None
        self._remote_fetch_job = None
        self._remote_events: list[str] = []  # timestamped steps for the Log window
        self._remote_cluster_text = ""
        self._remote_event_key = ""
        self._remote_log_dialog = None
        self._batch_job_items: list[dict] | None = None  # items of the batch job being watched

        tabs = QTabWidget()
        tabs.addTab(self._scrollable(self._segmentation_tab()), "Segmentation")
        tabs.addTab(self._scrollable(self._tracking_tab()), "Tracking")
        tabs.addTab(self._scrollable(self._t1_tab()), "T1")
        self.cell_props = CellPropertiesWidget(self.viewer)
        self.cell_props.dir_provider = self._cell_props_dir
        tabs.addTab(self._scrollable(self.cell_props), "Export Cell Props")
        from .batch_widget import BatchPanel

        self.batch = BatchPanel(
            current_values=self.seg_params.values,
            fill_holes=self.seg_fill_holes.isChecked,
            open_egg=self._open_batch_egg,
            run_items=self._run_batch,
            preview_items=self._preview_batch,
        )
        batch_index = tabs.addTab(self.batch, "Batch")
        # read the folder again when the tab is shown (presets saved, runs made meanwhile)
        tabs.currentChanged.connect(lambda i: self.batch.refresh() if i == batch_index else None)
        version = QLabel(f"v{__version__}")
        version.setToolTip("napari-tissuekit version")
        version.setStyleSheet("color: gray; padding: 0 6px;")
        tabs.setCornerWidget(version, Qt.TopRightCorner)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout = QVBoxLayout()
        layout.addLayout(self._folder_bar())
        layout.addWidget(tabs)
        layout.addWidget(self.status)
        self.setLayout(layout)
        self._apply_section_style()
        self.viewer.events.theme.connect(self._apply_section_style)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.viewer.mouse_drag_callbacks.append(self._on_track_click)
        self.viewer.dims.events.current_step.connect(self._update_overlay)
        self.viewer.layers.events.inserted.connect(self._update_overlay)
        self.viewer.layers.events.removed.connect(self._update_overlay)
        self._update_overlay()
        self._install_bond_shortcuts()
        self._refresh_layers()
        self._announce_waiting_cluster_jobs()

    def _announce_waiting_cluster_jobs(self) -> None:
        """After a lost connection or a restart: say that a cluster job can still be fetched."""
        try:
            from .core.remote import load_jobs

            jobs = load_jobs()
        except Exception:
            return
        if jobs:
            names = ", ".join(j.label for j in jobs[:3])
            self.status.setText(
                f"{len(jobs)} cluster job(s) waiting ({names}): use Cluster settings > Fetch cluster results. "
                "They keep running on the cluster while napari is closed."
            )

    def _apply_section_style(self, *_args) -> None:
        """Coloured section titles and clearer borders on every QGroupBox."""
        try:
            from napari.utils.theme import get_theme

            bg = get_theme(self.viewer.theme).background.as_rgb_tuple()
            dark = sum(bg[:3]) < 384
        except Exception:
            dark = self.viewer.theme != "light"
        # light yellow on dark (echoes the yellow outlines); dark amber on light
        title, border = ("#f5d67b", "#6a7380") if dark else ("#8a5a00", "#9a9fa6")
        self.setStyleSheet(
            f"""
            QGroupBox {{
                border: 1px solid {border};
                border-radius: 4px;
                margin-top: 16px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 8px;
                padding: 0 4px;
                color: {title};
            }}
            """
        )

    def sizeHint(self) -> QSize:
        return QSize(320, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        return QSize(240, super().minimumSizeHint().height())

    def _refresh_layers(self, *_args) -> None:
        if hasattr(self, 'folder_label'):
            self._refresh_folder_label()
        images = [ly.name for ly in self.viewer.layers if isinstance(ly, Image)]
        # "Seg · <run>" layers are read-only comparison overlays, never inputs.
        labels = [ly.name for ly in self.viewer.layers if isinstance(ly, Labels) and not ly.name.startswith("Seg · ")]
        # Masks: Labels plus non-RGB Images (imported *_cp_masks.tif often opens as Image).
        mask_candidates = list(labels)
        for ly in self.viewer.layers:
            if isinstance(ly, Image) and ly.name not in mask_candidates:
                # Use layer metadata; np.asarray(ly.data) would load lazy data.
                if not ly.rgb and ly.ndim in (2, 3):
                    mask_candidates.append(ly.name)
        outline_candidates = list(dict.fromkeys([*images, *labels]))
        self._fill(self.seg_image, images, keep=True)
        self._fill(self.seg_masks, mask_candidates, keep=True)
        self._fill(self.seg_outlines, outline_candidates, keep=True)
        self._fill(self.track_labels, labels, keep=True)
        self._fill(self.track_intensity, ["(none)", *images], keep=True)
        self._fill(self.t1_labels, labels, keep=True)
        self._fill(self.t1_rgb, ["(none)", *images], keep=True)
        self._show_calibration()

    @staticmethod
    def _fill(box: QComboBox, names: list[str], *, keep: bool) -> None:
        current = box.currentText() if keep else ""
        box.blockSignals(True)
        box.clear()
        box.addItems(names)
        if current:
            idx = box.findText(current)
            if idx >= 0:
                box.setCurrentIndex(idx)
        box.blockSignals(False)

    def _layer(self, name: str):
        if not name or name == "(none)":
            return None
        for ly in self.viewer.layers:
            if ly.name == name:
                return ly
        return None

    def _job_running(self) -> bool:
        if self._thread is not None and self._thread.isRunning():
            return True
        return (
            self._proc is not None
            and self._proc.state() != QProcess.NotRunning
        )

    def _start(self, fn, on_done) -> None:
        if self._job_running():
            self.status.setText("A job is already running.")
            return
        self._set_busy(True)
        self.status.setText("Starting...")
        self._job = _Job(fn)
        self._thread = QThread(self)
        self._job.moveToThread(self._thread)
        self._thread.started.connect(self._job.run)
        self._job.progress.connect(self._on_progress)
        self._job.finished.connect(on_done)
        self._job.finished.connect(self._thread.quit)
        self._job.failed.connect(self._on_failed)
        self._job.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._clear_job)
        self._thread.start()

    def _on_progress(self, percent: int, message: str) -> None:
        self.status.setText(f"{percent}% {message}")

    def _on_failed(self, text: str) -> None:
        self._last_job_error = text.strip().splitlines()[-1]
        self.status.setText(self._last_job_error)
        print(text, flush=True)

    def _clear_job(self) -> None:
        self._set_busy(False)
        self._thread = None
        self._job = None

    def _set_busy(self, busy: bool) -> None:
        for button in (self.seg_run, self.track_run, self.t1_run, self.track_save):
            button.setEnabled(not busy)

    def _segmentation_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()
        form = QFormLayout()
        form.setVerticalSpacing(4)
        self.seg_image = QComboBox()
        self.seg_fill_holes = QCheckBox("Fill enclosed holes as new cells")
        self.seg_fill_holes.setToolTip(
            "Empty space that is closed in by cells becomes a cell of its own. Space that reaches the image edge "
            "(the outside of the embryo) is never filled. Recorded in each run's params.json."
        )
        self.seg_fill_holes.setChecked(True)
        from .seg_params_widget import SegParamsPanel

        # One form, one label column: Image, Preset, Run name, Parameters (incl. batch size).
        self.seg_params = SegParamsPanel(name_for=self._next_run_name)
        form.addRow("Image", self.seg_image)
        self.seg_params.add_rows(form)
        # In the Parameters pane, but not saved in presets.
        self.seg_params.add_body_row("", self.seg_fill_holes)
        segment_box = QGroupBox("1 · Segment")
        segment_inner = QVBoxLayout()
        segment_inner.setSpacing(6)
        segment_inner.addLayout(form)

        # Run here or on the cluster; the cluster's occasional actions sit in one menu.
        self.seg_run = QPushButton("Run locally")
        self.seg_run.setToolTip("Segment the movie with Cellpose on this computer")
        self.seg_run.clicked.connect(self._run_segmentation)
        self.seg_remote = QPushButton("Run on cluster")
        self.seg_remote.setToolTip(
            "Log in to bwUniCluster once (password + OTP), upload the movie, segment it on a GPU "
            "node and download the results here. Runs in the background; you can keep working."
        )
        self.seg_remote.clicked.connect(lambda: self._run_remote_segmentation())
        cluster_btn = QPushButton("Cluster settings")  # a push button sizes itself to fit its text and menu arrow
        cluster_btn.setToolTip("Fetch results, job log, cluster settings and one-time setup")
        cluster_menu = QMenu(cluster_btn)
        self.seg_remote_fetch = cluster_menu.addAction("Fetch cluster results")
        self.seg_remote_fetch.setToolTip("Collect a cluster job whose connection was lost (laptop asleep, Wi-Fi).")
        self.seg_remote_fetch.triggered.connect(lambda: self._fetch_remote_segmentation())
        remote_log = cluster_menu.addAction("Log...")
        remote_log.setToolTip("Live view of what the cluster job is doing (this computer's steps and the job log)")
        remote_log.triggered.connect(lambda: self._show_remote_log())
        cluster_menu.addSeparator()
        remote_cfg = cluster_menu.addAction("Settings...")
        remote_cfg.setToolTip("Cluster user name, GPU queue, conda env")
        remote_cfg.triggered.connect(lambda: self._remote_settings())
        self.seg_remote_setup = cluster_menu.addAction("Set up cluster...")
        self.seg_remote_setup.setToolTip(
            "One time: install Miniforge and a conda env (torch, Cellpose, this plugin) in your "
            "cluster home via a CPU job (10-20 min). Later runs keep the code up to date by themselves."
        )
        self.seg_remote_setup.triggered.connect(lambda: self._setup_remote())
        cluster_menu.setToolTipsVisible(True)
        cluster_btn.setMenu(cluster_menu)
        cluster_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        run_row = QHBoxLayout()
        run_row.setSpacing(4)
        run_row.addWidget(self.seg_run, 1)
        run_row.addWidget(self.seg_remote, 1)
        run_row.addWidget(cluster_btn)
        segment_inner.addLayout(run_row)
        self.seg_remote_bar = QProgressBar()
        self.seg_remote_bar.setTextVisible(True)
        self.seg_remote_bar.hide()  # shown while a cluster job runs
        segment_inner.addWidget(self.seg_remote_bar)

        # Saved runs (one folder per parameter set): pick the one to edit / track, or overlay others.
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        segment_inner.addWidget(line)
        runs_form = QFormLayout()
        runs_row = QHBoxLayout()
        runs_row.setSpacing(4)
        self.seg_runs = QComboBox()
        self.seg_runs.setToolTip("Segmentation runs saved in the results folder (segmentation/<run>/)")
        self.seg_runs.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.seg_runs.setMinimumContentsLength(10)
        self.seg_run_show = QPushButton("Show")
        self.seg_run_show.setToolTip(
            "Make this run the active one: load it as the editable Segmentation + Outlines. "
            "Tracking and Load results use the active run."
        )
        self.seg_run_show.clicked.connect(lambda: self._show_seg_run())
        self.seg_run_compare = QPushButton("Compare")
        self.seg_run_compare.setToolTip(
            "Overlay this run's outlines as a read-only layer 'Seg · <run>' (own colour) "
            "to compare it with the active segmentation"
        )
        self.seg_run_compare.clicked.connect(lambda: self._compare_seg_run())
        runs_row.addWidget(self.seg_runs, 1)
        runs_row.addWidget(self.seg_run_show)
        runs_row.addWidget(self.seg_run_compare)
        runs_form.addRow("Runs", runs_row)
        segment_inner.addLayout(runs_form)
        segment_box.setLayout(segment_inner)
        layout.addWidget(segment_box)

        outline_box = QGroupBox("2 · Correct outlines")
        outline_inner = QVBoxLayout()
        outline_help = QLabel(
            "Tissue Analyzer style. Paint bonds (label 1) / erase bonds (label 0) on yellow Outlines "
            "(right mouse button erases while Paint is on). "
            "Enter = Apply (local watershed, like TA). "
            "M = toggle Outlines. Deleting a cell also redraws the outlines."
        )
        outline_help.setWordWrap(True)
        outline_help.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        outline_inner.addWidget(outline_help)
        import_form = QFormLayout()
        self.seg_masks = QComboBox()
        self.seg_outlines = QComboBox()
        import_form.addRow("Masks", self.seg_masks)
        import_form.addRow("Outlines", self.seg_outlines)
        self.seg_pad = QSpinBox()
        self.seg_pad.setRange(8, 512)
        self.seg_pad.setValue(128)
        self.seg_pad.setToolTip(
            "Context margin (px) around each edit for the local watershed. "
            "TA uses 128; smaller is faster."
        )
        import_form.addRow("Watershed pad", self.seg_pad)
        outline_inner.addLayout(import_form)
        tools = QHBoxLayout()
        for text, mode in (("Paint", "paint"), ("Erase", "erase"), ("Pan/zoom", "pan_zoom")):
            button = QPushButton(text)
            button.setToolTip("Select the Outlines layer and switch napari's tool")
            button.clicked.connect(lambda _=False, m=mode: self._set_outline_tool(m))
            tools.addWidget(button)
        outline_inner.addLayout(tools)
        self.seg_start_edit = QPushButton("Start outline editing (for separately imported outlines)")
        self.seg_start_edit.setToolTip("Only needed for outlines or masks you dragged in yourself and picked above. Not needed after Run locally / Run on cluster or loading a results folder: editing starts there by itself.")
        self.seg_start_edit.clicked.connect(self._start_outline_editing)
        outline_inner.addWidget(self.seg_start_edit)
        self.seg_commit = QPushButton("Apply bond edits (Enter)")
        self.seg_commit.clicked.connect(lambda: self._apply_bond_edits())
        outline_inner.addWidget(self.seg_commit)
        self._add_correction(outline_inner)
        outline_box.setLayout(outline_inner)
        layout.addWidget(outline_box)

        save_box = QGroupBox("3 · Save / export")
        save_row = QHBoxLayout()
        self.seg_save = QPushButton("Save segmentation")
        self.seg_save.setToolTip(
            "Ctrl+S. Writes the masks (*_cp_masks.tif) and the 1 px outlines "
            "(*_cell_outlines.tif) to the results folder."
        )
        self.seg_save.clicked.connect(self._save_segmentation)
        self.seg_export = QPushButton("Export TIFF")
        self.seg_export.clicked.connect(lambda: self._export_step_tiffs("segmentation"))
        save_row.addWidget(self.seg_save)
        save_row.addWidget(self.seg_export)
        save_box.setLayout(save_row)
        layout.addWidget(save_box)

        layout.addStretch()
        page.setLayout(layout)
        return page

    def _add_correction(self, inner: QVBoxLayout) -> None:
        """Delete-cell tool, appended to the outline-correction section."""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        inner.addWidget(line)
        title = QLabel("<b>Delete cell</b>")
        inner.addWidget(title)
        try:
            from .correction import CellCorrectionWidget, DataManager, VisualizationManager
        except Exception as exc:
            self.correction = None
            missing = QLabel(
                "The cell-delete tool is not importable "
                f"({exc}). Outline edit/save still work."
            )
            missing.setWordWrap(True)
            missing.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
            inner.addWidget(missing)
        else:
            self._data_manager = DataManager()
            self._vis = VisualizationManager(self.viewer, self._data_manager)
            self.correction = CellCorrectionWidget(
                self.viewer, self._data_manager, self._vis
            )
            self.correction.layout().setContentsMargins(0, 0, 0, 0)
            inner.addWidget(self.correction)
            self._mirror_cell_edits_on_outlines()

    def _run_segmentation(self) -> None:
        if self._job_running():
            self.status.setText("A job is already running.")
            return
        layer = self._layer(self.seg_image.currentText())
        if layer is None:
            self.status.setText("Select an image layer.")
            return
        self._seg_stem = self._safe_stem(layer.name)
        self._autosave_outlines_ok = True
        self._remember_calibration(layer)
        image = _as_tyx(np.asarray(layer.data))
        batch = int(self.seg_params.values()["batch_size"])
        use_gpu = True  # always: Cellpose uses the CPU by itself when this computer has no GPU
        fill_holes = self.seg_fill_holes.isChecked()
        work = Path(tempfile.mkdtemp(prefix="tissuekit_seg_"))
        image_path = work / "image.npy"
        out_path = work / "masks.npy"
        np.save(image_path, np.asarray(image))
        self._seg_dir = work
        self._seg_out = out_path
        # Settings of this run (preset + every Cellpose value); the run folder is named when it is saved.
        self._seg_pending = {**self.seg_params.run_info(), "run_base": self.seg_params.run_base(),
                             "batch_size": batch, "gpu": "auto", "fill_holes": fill_holes, "where": "local"}
        import json

        from .core.seg_params import eval_kwargs

        kwargs_json = json.dumps(eval_kwargs(self.seg_params.values()))
        code = (
            "import json\n"
            "import numpy as np\n"
            "from napari_tissuekit.core.segmentation import segment_stack\n"
            f"img = np.load({str(image_path)!r})\n"
            "masks = segment_stack(\n"
            "    img,\n"
            f"    batch_size={batch},\n"
            f"    use_gpu={use_gpu},\n"
            f"    eval_kwargs=json.loads({kwargs_json!r}),\n"
            f"    fill_holes={fill_holes},\n"
            ")\n"
            f"np.save({str(out_path)!r}, masks)\n"
            "print('SEGMENT_DONE', int(masks.max()), flush=True)\n"
        )
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._seg_output)
        self._proc.finished.connect(self._seg_finished)
        self._set_busy(True)
        self.status.setText(
            f"Starting segmentation (batch size {batch}"
            + ("; on a laptop GPU this may run out of memory, lower it under Parameters if it fails" if batch > 16 and use_gpu else "")
            + ")..."
        )
        self._proc.start(sys.executable, ["-u", "-c", code])

    def _seg_output(self) -> None:
        if self._proc is None:
            return
        text = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("frame "):
                self.status.setText(line)
            elif line.startswith("SEGMENT_DONE"):
                self.status.setText(line)
            elif line:
                print(line, flush=True)

    def _seg_finished(self, exit_code: int, _status) -> None:
        out = self._seg_out
        work = self._seg_dir
        self._proc = None
        self._set_busy(False)
        try:
            if exit_code != 0 or out is None or not out.is_file():
                self.status.setText(
                    f"Segmentation failed (exit {exit_code}). See the terminal log."
                )
                return
            masks = np.load(out)
            self._segmentation_done(masks)
        finally:
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)
            self._seg_dir = None
            self._seg_out = None

    def _segmentation_done(self, masks: np.ndarray) -> None:
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        self._seg_base_masks = masks.copy()
        self._seg_run = None  # a new run folder is made when it is saved (below)
        self._show_segmentation_layers(
            masks,
            f"Segmentation done. max label {int(np.max(masks))}. "
            "Paint/erase yellow Outlines, Enter to Apply, then Save / track.",
        )
        self._refresh_layers()
        self._autosave_segmentation()

    # ---- cluster (bwUniCluster) segmentation ------------------------------
    def _remote_settings(self, cfg=None) -> bool:
        from .core.remote import RemoteConfig
        from .remote_widget import RemoteSettingsDialog

        cfg = cfg if cfg is not None else RemoteConfig.load()
        return RemoteSettingsDialog(cfg, self).exec_() == QDialog.Accepted and bool(cfg.user)

    def _remote_cfg(self):
        from .core.remote import RemoteConfig

        cfg = RemoteConfig.load()
        if not cfg.user and not self._remote_settings(cfg):
            self.status.setText("Cluster settings needed (at least your user name).")
            return None
        return cfg

    def _remote_movie(self, layer) -> Path:
        """A .tif to upload: the layer's own file, or its data written to a temp file."""
        path = next((p for p in self._layer_paths(layer) if p), None)
        if path and str(path).lower().endswith((".tif", ".tiff")):
            return Path(path)
        import tifffile

        work = Path(tempfile.mkdtemp(prefix="tissuekit_remote_"))
        self._remote_tmp = work
        out = work / f"{movie_stem(path) if path else self._safe_stem(layer.name)}.tif"
        tifffile.imwrite(out, _as_tyx(np.asarray(layer.data)))
        return out

    def _run_remote_segmentation(self) -> None:
        from .core import remote
        from .remote_widget import LoginDialog

        if self._remote.busy:
            self.status.setText("A cluster job is already running.")
            return
        layer = self._layer(self.seg_image.currentText())
        if layer is None:
            self.status.setText("Select an image layer.")
            return
        wf = self._working_folder(layer)
        if wf is None:
            self.status.setText("Choose a results folder first (Folder...).")
            return
        cfg = self._remote_cfg()
        if cfg is None:
            return
        shape = _as_tyx_shape(layer.data.shape, getattr(layer, "rgb", False))
        creds = LoginDialog(
            cfg,
            f"Segment '{layer.name}' ({shape[0]} frames, {shape[1]}x{shape[2]} px) on {cfg.partition}, "
            f"time limit {cfg.time} (Settings... to change).",
            self,
        ).credentials()
        if creds is None:
            return
        password, otp = creds
        self.status.setText("[cluster] preparing the movie ...")
        movie = self._remote_movie(layer)
        results = Path(wf.root)
        # Preset, Cellpose values and run folder from this tab (the cluster has no saved presets).
        from dataclasses import replace

        cfg = replace(cfg, preset=self.seg_params.preset_name().replace(" ", "_"),
                      batch_size=int(self.seg_params.values()["batch_size"]))
        run = wf.new_run_name(self.seg_params.run_base())
        params = self.seg_params.values()
        fill_holes = self.seg_fill_holes.isChecked()

        def work(progress, log):
            return remote.run_segmentation(
                cfg, password, otp, movie, results, progress, on_log=log, run=run, params=params,
                fill_holes=fill_holes,
            )

        self._start_remote(work)

    def _fetch_remote_segmentation(self, job=None) -> None:
        from .core import remote
        from .remote_widget import LoginDialog

        if self._remote.busy:
            self.status.setText("A cluster job is already running.")
            return
        if job is None:
            jobs = remote.load_jobs()
            if not jobs:
                self.status.setText("No cluster jobs waiting to be fetched.")
                return
            job = jobs[0]
            if len(jobs) > 1:
                names = [
                    f"{j.job_id}  {j.label}  ({time.strftime('%d.%m. %H:%M', time.localtime(j.submitted))})"
                    for j in jobs
                ]
                chosen, ok = QInputDialog.getItem(
                    self, "Fetch cluster results", "Job:", names, len(names) - 1, False
                )
                if not ok:
                    return
                job = jobs[names.index(chosen)]
        cfg = self._remote_cfg()
        if cfg is None:
            return
        creds = LoginDialog(
            cfg, f"Fetch job {job.job_id} ({job.label}). If it is still running, "
            "it is watched again and downloaded when it ends.", self,
        ).credentials()
        if creds is None:
            return
        password, otp = creds
        self._remote_fetch_job = job
        if job.is_batch:
            self._batch_job_items = list(job.items)
            self.batch.set_live({(remote.movie_stem(it["movie_name"]), it["profile"]): "queued"
                                 for it in job.items if it.get("status") == "pending"})

        def work(progress, log):
            return remote.fetch_job(cfg, job, progress, password, otp, on_log=log)

        self._start_remote(work)

    def _setup_remote(self) -> None:
        from .core import remote
        from .remote_widget import LoginDialog

        if self._remote.busy:
            self.status.setText("A cluster job is already running.")
            return
        cfg = self._remote_cfg()
        if cfg is None:
            return
        creds = LoginDialog(
            cfg, f"One-time setup: install the conda env '{cfg.conda_env}' on the cluster "
            f"({cfg.setup_partition} job, about 10-20 min).", self,
        ).credentials()
        if creds is None:
            return
        password, otp = creds

        def work(progress, log):
            return remote.setup_cluster(cfg, password, otp, progress, on_log=log)

        self._start_remote(work, on_done=self._remote_setup_done)

    def _remote_setup_done(self, summary) -> None:
        self._remote_finish()
        self.status.setText(f"[cluster] Set up: {summary}. 'Run on cluster' is ready.")

    def _start_remote(self, work, on_done=None) -> None:
        for button in (self.seg_remote, self.seg_remote_fetch, self.seg_remote_setup, self.batch.run_button):
            button.setEnabled(False)
        self._remote_event_key = ""
        self._remote_event("---- started")
        self.seg_remote_bar.setRange(0, 0)  # busy until the first measurable step
        self.seg_remote_bar.setFormat("%p%")
        self.seg_remote_bar.show()
        self._remote.start(
            work, self._remote_progress, on_done or self._remote_done, self._remote_failed,
            on_log=self._remote_cluster_log,
        )

    def _remote_event(self, message: str) -> None:
        """One timestamped line in the log window (kept while the window is closed)."""
        line = f"{time.strftime('%H:%M:%S')}  {message}"
        self._remote_events.append(line)
        if self._remote_log_dialog is not None:
            self._remote_log_dialog.add_event(line)

    def _remote_cluster_log(self, text: str) -> None:
        self._remote_cluster_text = text
        if self._remote_log_dialog is not None:
            self._remote_log_dialog.set_cluster_log(text)

    def _show_remote_log(self) -> None:
        from .remote_widget import RemoteLogDialog

        if self._remote_log_dialog is None:
            self._remote_log_dialog = RemoteLogDialog(self)
            for line in self._remote_events:
                self._remote_log_dialog.add_event(line)
            if self._remote_cluster_text:
                self._remote_log_dialog.set_cluster_log(self._remote_cluster_text)
        self._remote_log_dialog.show()
        self._remote_log_dialog.raise_()

    def _remote_progress(self, percent: int, message: str) -> None:
        import re

        self.status.setText(f"[cluster] {percent}% {message}" if percent else f"[cluster] {message}")
        if percent > 0:
            self.seg_remote_bar.setRange(0, 100)
            self.seg_remote_bar.setValue(percent)
        else:
            self.seg_remote_bar.setRange(0, 0)  # queued / loading: busy animation
        # log each new kind of step once (not every MB of an upload or every minute in the queue)
        key = re.sub(r"\d+", "#", message)
        if key != self._remote_event_key or "item" in message and "FAILED" in message:
            self._remote_event_key = key
            self._remote_event(message)
        if self._batch_job_items is not None:
            self._batch_progress(percent, message)

    def _batch_progress(self, percent: int, message: str) -> None:
        """Batch tab: the running item's cell, rows of downloaded items, the batch progress bar."""
        import re

        items = self._batch_job_items
        key = lambda i: (movie_stem(items[i - 1]["movie_name"]), items[i - 1]["profile"])  # noqa: E731
        m = re.match(r"(downloaded )?item (\d+)/\d+ ", message)
        if m and 0 < int(m.group(2)) <= len(items):
            i = int(m.group(2))
            if m.group(1):
                self.batch.set_live({key(i): None}, refresh_egg=key(i)[0])
            elif "FAILED" in message:
                self.batch.set_live({key(i): "failed"})
            elif self.batch.live.get(key(i)) == "queued":  # stays "running" until downloaded / failed
                self.batch.set_live({key(i): "running"})
        self.batch.show_progress(percent, message)

    def _remote_finish(self) -> None:
        self.batch.show_progress(None, "")
        for button in (self.seg_remote, self.seg_remote_fetch, self.seg_remote_setup, self.batch.run_button):
            button.setEnabled(True)
        self.seg_remote_bar.hide()
        if self._remote_tmp is not None:
            shutil.rmtree(self._remote_tmp, ignore_errors=True)
            self._remote_tmp = None

    def _remote_failed(self, exc) -> None:
        from .core.remote import RemoteError

        self._remote_finish()
        text = str(exc) if isinstance(exc, RemoteError) else f"{type(exc).__name__}: {exc}"
        self._remote_event(f"FAILED: {text}")  # full report (job folder, log end) in the Log window
        self.status.setText(f"[cluster] {text.splitlines()[0]} (details: Log...)")
        if self._batch_job_items is not None:
            self._batch_job_items = None
            self.batch.set_live(None)  # a lost connection: the job is remembered, fetch it later
            self.batch.refresh()

    def _remote_done(self, result) -> None:
        from .core.remote import BatchResult

        if isinstance(result, BatchResult):
            self._batch_done(result)
            return
        self._remote_finish()
        masks_path = Path(result[1] if isinstance(result, tuple) else result)
        # .../<movie>_results/segmentation/<run>/<stem>_cp_masks.tif
        results = next((p.parent for p in masks_path.parents if p.name == "segmentation"), masks_path.parent.parent)
        here = self.project.root is not None and Path(self.project.root).resolve() == results.resolve()
        run = masks_path.parent.name if masks_path.parent.parent.name == "segmentation" else "original"
        self._refresh_seg_runs()
        if not here or self._job_running():
            self.status.setText(f"[cluster] Done. Saved run '{run}' in {results}; open it with Folder... / Load results.")
            return
        if self._seg_base_masks is not None:
            answer = QMessageBox.question(
                self, "Cluster segmentation ready",
                f"{masks_path.name} is ready (saved in the results folder).\n"
                "Replace the segmentation shown now with it?",
            )
            if answer != QMessageBox.Yes:
                self.status.setText(f"[cluster] Done. Saved run '{run}' in {results} (pick it under Runs, then Show).")
                return
        import tifffile

        from .core.project import SEG_MASK_SUFFIXES, _strip_suffix

        masks = np.asarray(tifffile.imread(masks_path))
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        self._seg_stem = (_strip_suffix(masks_path.name, SEG_MASK_SUFFIXES) or "segmentation").replace("/", "_")
        self._replace_labels("Segmentation", masks)
        self._seg_base_masks = masks.copy()
        self._seg_run = run
        self._seg_pending = None
        self.project.set_active_seg_run(run)
        self._refresh_seg_runs()
        self._pending_outlines = None
        self._autosave_outlines_ok = False  # the cluster already wrote the outlines; only Save rewrites them
        self._show_segmentation_layers(
            masks, f"[cluster] Segmentation run '{run}' done on the GPU (max label {int(masks.max())}). "
            "Bonds are being prepared: wait for 100%, then edit / track.",
        )
        self._refresh_layers()
        self._pick(self.seg_masks, "Segmentation")
        self._pick(self.track_labels, "Segmentation")

    # ---- Batch tab ---------------------------------------------------------------------
    def _preview_batch(self, items, skipped) -> None:
        """What the batch would submit: items, time limit, the Slurm script (nothing is sent)."""
        from .batch_widget import DryRunDialog

        cfg = self._remote_cfg()
        if cfg is None:
            return
        try:
            DryRunDialog(items, skipped, cfg, self).exec_()
        except ValueError as exc:  # e.g. a time limit in the settings that is not hh:mm:ss
            self.status.setText(f"Batch: cannot read the time limit '{cfg.time}' ({exc}); fix it in Cluster settings.")

    def _run_batch(self, items, skipped) -> None:
        """One login, one GPU job for every (egg, profile) item; results come back item by item."""
        from .core import remote
        from .remote_widget import LoginDialog

        if self._remote.busy:
            self.status.setText("A cluster job is already running.")
            return
        cfg = self._remote_cfg()
        if cfg is None:
            return
        try:
            total = remote.batch_time(cfg.time, len(items))
        except ValueError:
            self.status.setText(f"Batch: cannot read the time limit '{cfg.time}'; fix it in Cluster settings.")
            return
        problem = remote.partition_problem(cfg.partition, total)
        if problem:
            QMessageBox.warning(self, "Batch", f"The batch does not fit the queue:\n\n{problem}")
            return
        eggs = len({it.movie for it in items})
        creds = LoginDialog(
            cfg,
            f"Batch: {len(items)} item(s) ({eggs} egg(s)) one after another in one job on {cfg.partition}, "
            f"time limit {total} ({cfg.time} per item)"
            + (f"; {len(skipped)} pair(s) skipped as already done" if skipped else "")
            + ". Results are downloaded item by item while the job runs.",
            self,
        ).credentials()
        if creds is None:
            return
        password, otp = creds
        self._batch_job_items = [{"movie_name": it.movie_name, "profile": it.profile} for it in items]
        self.batch.set_live({(it.egg, it.profile): "queued" for it in items})

        def work(progress, log):
            return remote.run_batch(cfg, password, otp, items, progress, on_log=log)

        self._start_remote(work)

    def _batch_done(self, result) -> None:
        self._remote_finish()
        self._batch_job_items = None
        self.batch.set_live(None)
        self.batch.refresh()
        self._refresh_seg_runs()  # the egg shown now may have new runs
        ok, bad = len(result.downloaded), len(result.failed)
        for it in result.failed:
            self._remote_event(f"not done: {it['movie_name']} · {it['profile']}: {it.get('note', '')}")
        if result.note:
            self._remote_event(f"batch ended early: {result.note}")
        text = f"[cluster] Batch done: {ok} run(s) saved"
        if bad:
            text += f", {bad} without results (they show '–'; select them again, details: Log...)"
        if result.note:
            text += f". The job ended early: {result.note.splitlines()[0]}"
        self.status.setText(text + ".")

    def _open_batch_egg(self, egg, run=None) -> None:
        """Open one egg of the Batch tab: its movie and saved results (``run`` made active first)."""
        if len(self.viewer.layers):
            # One egg at a time (shared results folder, layer names, Runs list): replace the layers.
            # Ask only when unsaved track edits would be lost.
            if self._track_dirty and QMessageBox.question(
                self, "Open egg",
                f"The Tracked layer has unsaved edits. Discard them and open {egg.name}?",
                QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel,
            ) != QMessageBox.Ok:
                return
            self.viewer.layers.clear()
            self._seg_base_masks = None
            self._seg_base_hand = None
            self._seg_run = None
            self._seg_pending = None
            self._tracking_files = None
            self._tracked_run = None
            self._track_dirty = False
        self.project.set_manual(egg.results)
        if run:
            self.project.set_active_seg_run(run)
        self._refresh_folder_label()
        if self.project.found():
            self._load_saved_results()  # also opens the movie
        else:
            before = {id(ly) for ly in self.viewer.layers}
            self.viewer.open(str(egg.movie), plugin="napari")
            self._refresh_layers()
            new = next((ly for ly in self.viewer.layers if id(ly) not in before and isinstance(ly, Image)), None)
            if new is not None:
                self._pick(self.seg_image, new.name)
            self.status.setText(f"Opened {egg.name} (no results yet). Results go to {egg.results}.")

    def _start_outline_editing(self) -> None:
        masks_ly = self._layer(self.seg_masks.currentText())
        outlines_ly = self._layer(self.seg_outlines.currentText())
        if outlines_ly is None:
            self.status.setText("Select an Outlines layer (a *_cp_outlines.tif or *_cell_outlines.tif).")
            return
        try:
            from .core.outline_io import outline_labels_to_rgb, outlines_rgb_to_masks, rgb_or_labels_to_outline_labels
        except Exception as exc:
            self.status.setText(f"outline_io import failed: {exc}")
            return
        try:
            outlines = rgb_or_labels_to_outline_labels(np.asarray(outlines_ly.data))
        except Exception as exc:
            self.status.setText(f"Could not read outlines: {exc}")
            return
        # No masks (none selected, the outlines layer itself, or a binary image): the cells are the regions between the outlines.
        from_outlines = masks_ly is None or masks_ly is outlines_ly
        if not from_outlines:
            try:
                first = np.asarray(masks_ly.data[0] if np.ndim(masks_ly.data) == 3 else masks_ly.data)
                from_outlines = np.unique(first).size <= 2
            except Exception:
                pass
        if from_outlines:
            masks = outlines_rgb_to_masks(outline_labels_to_rgb(outlines), background="auto")
            base_name = outlines_ly.name
        else:
            try:
                masks = _as_tyx(np.asarray(masks_ly.data))
            except ValueError as exc:
                self.status.setText(f"Masks layer must be T,Y,X labels: {exc}")
                return
            base_name = masks_ly.name
            if outlines.shape != masks.shape:
                self.status.setText(
                    f"Outlines shape {outlines.shape} != masks shape {masks.shape}."
                )
                return
        for suffix in ("_cp_outlines", "_cell_outlines", "_cp_masks"):
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
        self._seg_stem = base_name.replace("/", "_") if from_outlines else self._safe_stem(base_name)  # outlines file: keep its own name
        self._autosave_outlines_ok = not from_outlines
        self._pending_outlines = None if from_outlines else outlines
        # Prefer raw image name for save stem when selected.
        img_ly = self._layer(self.seg_image.currentText())
        if img_ly is not None and not from_outlines:
            self._seg_stem = self._safe_stem(img_ly.name)
        self._seg_base_masks = masks.copy()
        # Imported by hand: saved as a new run ("edited", "edited_2" ...), never into an existing one.
        self._seg_run = None
        self._seg_pending = None
        # The imported outlines are the starting bonds where they are the plugin's own 1 px bonds; other
        # frames get 1 px TA bonds rebuilt from the masks (the cells between the outlines when there are no masks).
        self._show_segmentation_layers(
            masks,
            f"Outline editing ready{' from the outlines' if from_outlines else ''} (max label {int(masks.max())}). "
            "Paint/erase bonds, Enter to Apply (TA watershed), then Save.",
            saved_outlines=outlines,
        )
        self._refresh_layers()

    def _install_bond_shortcuts(self) -> None:
        # ApplicationShortcut so Enter works while the canvas has focus (TA-like).
        sc_enter = QShortcut(QKeySequence(Qt.Key_Return), self)
        sc_enter.setContext(Qt.ApplicationShortcut)
        sc_enter.activated.connect(lambda: self._apply_bond_edits())
        sc_pad = QShortcut(QKeySequence(Qt.Key_Enter), self)
        sc_pad.setContext(Qt.ApplicationShortcut)
        sc_pad.activated.connect(lambda: self._apply_bond_edits())
        sc_toggle = QShortcut(QKeySequence(Qt.Key_M), self)
        sc_toggle.setContext(Qt.ApplicationShortcut)
        sc_toggle.activated.connect(self._toggle_outlines_visibility)
        sc_save = QShortcut(QKeySequence("Ctrl+S"), self)
        sc_save.setContext(Qt.ApplicationShortcut)
        sc_save.activated.connect(self._save_segmentation)
        # Ctrl/Cmd+Z for the whole outline editor. napari's own Labels undo (bound on the active Outlines layer) would
        # otherwise win and replay old paint strokes; this one first takes back unapplied strokes, then applied steps.
        sc_undo = QShortcut(QKeySequence.Undo, self)
        sc_undo.setContext(Qt.ApplicationShortcut)
        sc_undo.activated.connect(self._undo_edit)
        self._bond_shortcuts = (sc_enter, sc_pad, sc_toggle, sc_save, sc_undo)

    def _undo_edit(self) -> None:
        """Ctrl/Cmd+Z: take back strokes not applied yet (napari's paint history), else the last applied step
        (a bond edit with Enter, or a cell deletion)."""
        ol = self._layer("Outlines")
        history = getattr(ol, "_undo_history", None)
        if isinstance(ol, Labels) and history and self._seg_base_outlines is not None:
            from .core.ta_bond_edit import changed_frames

            edited = np.asarray(ol.data)
            if edited.shape == self._seg_base_outlines.shape and changed_frames(self._seg_base_outlines, edited):
                ol.undo()
                self.status.setText("Undid the last paint stroke.")
                return
        if self.correction is not None:
            self.correction.undo_last_action()

    def _toggle_outlines_visibility(self) -> None:
        ol = self._layer("Outlines")
        if isinstance(ol, Labels):
            ol.visible = not ol.visible
            self.status.setText(
                f"Outlines {'shown' if ol.visible else 'hidden'} (M)."
            )

    def _show_segmentation_layers(self, masks: np.ndarray, message: str, saved_outlines: np.ndarray | None = None) -> None:
        """Show masks, then build 1 px TA bonds in the background for Outlines. ``saved_outlines`` (0/1, T,Y,X) are kept
        as the starting bonds where they are the plugin's own 1 px bonds, so loading a saved edit moves no line."""
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        if self._job_running():
            self.status.setText("A job is already running.")
            return
        self._seg_base_hand = None
        self._seg_base_outlines = None

        if self.correction is not None:
            # A "Segmentation" layer that already exists (e.g. loaded from the results folder) is the one the cell
            # editing tools must use; otherwise they add a second layer ("Segmentation [1]") and edit that one.
            existing = self._layer("Segmentation")
            vis = getattr(self.correction, "vis_manager", None)
            if isinstance(existing, Labels) and vis is not None and vis.tracking_layer is not existing:
                vis.tracking_layer = existing
                self.correction.masks_layer = existing
            self.correction.undo_stack.clear()  # older steps belong to the previous masks
            self._data_manager.initialize_stack(int(masks.shape[0]))
            self._data_manager.segmentation_data = masks
            self._outline_sync_busy = True
            try:
                self.correction.set_masks_layer(masks)
            finally:
                self._outline_sync_busy = False

        # Own copy: edits are written in place and must not touch the input layer.
        self._replace_labels("Segmentation", masks.copy())
        seg = self._layer("Segmentation")
        if isinstance(seg, Labels):
            seg.opacity = 0.25
            seg.visible = False

        self._hand_msg = message

        # Show outlines right away from the masks; the 1 px TA bonds replace them
        # when the background job finishes (or they stay if that job fails).
        from .core.outline_io import masks_to_outline_labels

        interim = (np.asarray(saved_outlines) > 0).astype(np.uint8) if saved_outlines is not None and np.shape(saved_outlines) == masks.shape else masks_to_outline_labels(masks)
        self._replace_labels("Outlines", interim)
        self._style_outlines_layer(self._layer("Outlines"))

        def work(progress):
            from .core.outline_io import hand_stack_to_outline
            from .core.ta_bond_edit import hand_stack_from_outlines, thin_hand_stack

            if saved_outlines is None:
                hand = thin_hand_stack(masks, progress)
                return hand, hand_stack_to_outline(hand), False
            hand, outlines, fits = hand_stack_from_outlines(masks, saved_outlines, progress)
            # most frames fit: the file is the plugin's own, so its other frames are not edits made elsewhere
            return hand, outlines, bool(fits.mean() >= 0.5)

        self._start(work, self._hand_ready)

    def _hand_ready(self, result: tuple) -> None:
        hand, outlines, own_file = result
        self._seg_base_hand = hand
        self._seg_base_outlines = outlines.copy()
        self._replace_labels("Outlines", outlines)
        self._style_outlines_layer(self._layer("Outlines"))
        pending, self._pending_outlines = self._pending_outlines, None
        if own_file:
            pending = None  # nothing in it to apply: the bonds shown are the file's (or rebuilt from the masks where it was out of date)
        if pending is not None and not np.array_equal(pending > 0, outlines > 0):
            from .core.outline_io import outlines_are_edits

            if outlines_are_edits(pending, outlines):
                # outlines edited outside the plugin: show them; Enter applies the differences to the cells like painted strokes
                self._replace_labels("Outlines", (pending > 0).astype(outlines.dtype))
                self._style_outlines_layer(self._layer("Outlines"))
                self._hand_msg = f"{self._hand_msg} The loaded outlines differ from the masks: press Enter to apply those changes."
            else:
                self._hand_msg = (f"{self._hand_msg} The loaded outlines are not 1 px bonds of these masks "
                                  "(e.g. Cellpose-style *_cp_outlines.tif): bonds are rebuilt from the masks.")
        if self._autosave_outlines_ok:
            self._autosave_outlines()
        self.status.setText(self._hand_msg)

    def _set_outline_tool(self, mode: str) -> None:
        """Paint / erase / pan on the Outlines layer without napari's own tool panel."""
        ol = self._layer("Outlines")
        if not isinstance(ol, Labels):
            self.status.setText("No Outlines layer yet: run segmentation or Start outline editing.")
            return
        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(ol)
        ol.visible = True
        if mode != "pan_zoom":
            ol.selected_label = 1  # 1 = outline
            ol.brush_size = max(int(ol.brush_size), 2)
        ol.mode = mode
        self.status.setText(f"Outlines: {mode.replace('_', '/')} tool active. Enter applies the edit.")

    @staticmethod
    def _right_click_erase(layer, event):
        """Right mouse button erases while the Paint tool is active (like Tissue Analyzer); the tool buttons stay."""
        if getattr(event, "button", 1) not in (2, "right") or getattr(layer, "mode", "") != "paint":
            return
        from napari.layers.labels._labels_constants import Mode

        layer._mode = Mode.ERASE  # napari's brush reads this when the stroke starts; the tool buttons do not change
        try:
            yield
            while event.type == "mouse_move":
                yield
        finally:
            layer._mode = Mode.PAINT

    def _style_outlines_layer(self, ol) -> None:
        """Yellow 1 px outlines, brush ready for painting."""
        if isinstance(ol, Labels):
            if self._right_click_erase not in ol.mouse_drag_callbacks:
                ol.mouse_drag_callbacks.insert(0, self._right_click_erase)  # before napari's own brush
            ol.opacity = 1.0
            ol.visible = True
            try:
                ol.contour = 0
            except Exception:
                pass
            try:
                ol.color = {0: None, 1: "yellow"}
            except Exception:
                try:
                    ol.colormap = {0: [0, 0, 0, 0], 1: [1, 1, 0, 1]}
                except Exception:
                    pass
            try:
                self.viewer.layers.selection.clear()
                self.viewer.layers.selection.add(ol)
                ol.mode = "paint"
                ol.brush_size = 3
            except Exception:
                pass

    def _autosave_outlines(self) -> None:
        """Write the outline-only TIFF as soon as the 1 px outlines exist (no click needed)."""
        outlines = self._seg_base_outlines
        if outlines is None or self.project.root is None:
            return
        try:
            from .core.movie_calibration import save_tiff

            stem = self._seg_stem or "segmentation"
            cal = self._calibration(
                self._layer(self.seg_image.currentText()),
                self._layer(self.seg_masks.currentText()),
                stem=self._seg_stem,
            ) or self._seg_cal
            path = self.project.step_dir("segmentation") / f"{stem}_cell_outlines.tif"
            save_tiff(path, (outlines > 0).astype(np.uint8) * 255, cal)
            self._hand_msg = f"{self._hand_msg} Outlines saved to {path.name}."
        except Exception:  # saving must never break editing
            print(traceback.format_exc(), flush=True)

    def _bonds_ready(self) -> bool:
        if self._seg_base_masks is None:
            self.status.setText(
                "No base masks — run segmentation or Start outline editing first."
            )
            return False
        if self._seg_base_hand is None or self._seg_base_outlines is None:
            if self._last_job_error and not self._job_running():
                self.status.setText(f"Bond preparation failed: {self._last_job_error}")
            else:
                self.status.setText("Bonds are still being prepared — wait for 100%.")
            return False
        return True

    def _apply_bond_edits(self) -> bool:
        """Apply painted bond strokes. Returns True if any frame changed."""
        if not self._bonds_ready():
            return False
        ol = self._layer("Outlines")
        if not isinstance(ol, Labels):
            self.status.setText("Missing Outlines layer.")
            return False

        try:
            from .core.ta_bond_edit import apply_bond_strokes, changed_frames
        except Exception as exc:
            self.status.setText(f"TA bond edit import failed: {exc}")
            return False

        t0 = time.perf_counter()
        edited = np.asarray(ol.data)
        if edited.ndim == 2:
            edited = edited[np.newaxis, ...]
        frames = changed_frames(self._seg_base_outlines, edited)
        # cells, bonds and outlines of the frames about to change, for Undo (the edit writes into these arrays in place)
        before = {int(t): (self._seg_base_masks[t].copy(), self._seg_base_hand[t].copy(), self._seg_base_outlines[t].copy()) for t in frames}
        if not frames:
            self.status.setText("No bond edits to apply.")
            return False
        try:
            updated, rois = apply_bond_strokes(
                self._seg_base_masks,
                self._seg_base_hand,
                self._seg_base_outlines,
                edited,
                frames=frames,
                min_seed_area=MIN_CELL_AREA - 1,  # the watershed removes regions of up to this many px
                pad=int(self.seg_pad.value()),
            )
        except Exception as exc:
            self.status.setText(f"Apply bond edits failed: {exc}")
            print(traceback.format_exc(), flush=True)
            return False

        self._commit_seg(updated, frames, rois)
        if rois:
            self._push_bond_undo(before)
        if not rois:
            self.status.setText(
                "No cells changed. A stroke outside the tissue must enclose "
                "an area (closed loop, or closed against the tissue edge)."
            )
            return False
        self.status.setText(
            f"Applied bond edits on {len(frames)} frame(s) (TA wshed) "
            f"in {time.perf_counter() - t0:.2f}s. max label "
            f"{int(self._seg_base_masks.max())}."
        )
        return True

    def _push_bond_undo(self, before: dict) -> None:
        """Make the bond edit just applied one Undo step (Ctrl/Cmd+Z), shared with the cell deletions."""
        if self.correction is None:
            return
        from .correction import ActionType, UndoAction

        def restore() -> None:
            masks, hand, outlines = self._seg_base_masks, self._seg_base_hand, self._seg_base_outlines
            if masks is None or hand is None or outlines is None:
                return
            frames = [t for t in before if t < masks.shape[0]]
            for t in frames:
                masks[t], hand[t], outlines[t] = before[t]
            self._commit_seg(masks, frames, [])
            self.status.setText(f"Undid the bond edit on {len(frames)} frame(s).")

        self.correction.undo_stack.append(UndoAction(
            action_type=ActionType.BOND, frame=None, description=f"bond edit on {len(before)} frame(s)",
            affected_cell_ids=set(), restore=restore,
        ))

    def _commit_seg(self, updated: np.ndarray, frames: list[int], rois: list) -> None:
        """Push changed frames into the Segmentation / Outlines layers in place."""
        from .core.outline_io import update_outline_roi

        updated = np.asarray(updated)
        if updated.ndim == 2:
            updated = updated[np.newaxis, ...]
        reallocated = updated is not self._seg_base_masks
        self._seg_base_masks = updated
        outlines = self._seg_base_outlines
        for t, ys, xs in rois:
            update_outline_roi(outlines[t], self._seg_base_hand[t], ys, xs)

        seg = self._layer("Segmentation")
        ol = self._layer("Outlines")
        if reallocated or not isinstance(seg, Labels):
            # Labels outgrew the dtype: fall back to a full layer replace.
            self._replace_labels("Segmentation", updated.copy())
        else:
            seg_data = self._writable_stack(seg)
            for t in frames:
                seg_data[t] = updated[t]
            seg.refresh()
        if isinstance(ol, Labels):
            ol_data = self._writable_stack(ol)
            # Whole frame: also resets stray painted pixels outside any ROI.
            for t in frames:
                ol_data[t] = outlines[t]
            for history in (getattr(ol, "_undo_history", None), getattr(ol, "_redo_history", None)):
                if history is not None:
                    history.clear()  # the strokes it holds are applied (or undone) now
            ol.refresh()
        else:
            self._replace_labels("Outlines", outlines.copy())
        if self.correction is not None:
            self._data_manager.segmentation_data = self._seg_base_masks
            self._outline_sync_busy = True
            try:
                self.correction.set_masks_layer(self._seg_base_masks)
            except Exception:
                pass
            finally:
                self._outline_sync_busy = False

    def _mirror_cell_edits_on_outlines(self) -> None:
        """Make the cell-delete tool (delete / undo a cell) also
        redraw that frame's 1 px cell outlines on the Outlines layer."""
        vis = self._vis
        update = vis.update_tracking_visualization

        def update_and_sync(data):
            update(data)
            if not self._outline_sync_busy:
                try:
                    self._sync_outlines_to_masks(data)
                except Exception:  # an outline refresh must never break a cell edit
                    print(traceback.format_exc(), flush=True)

        vis.update_tracking_visualization = update_and_sync

    def _sync_outlines_to_masks(self, data) -> None:
        """Rebuild bonds/outlines of the frames whose Segmentation labels differ
        from the base masks (``data`` is a (frame, index) pair or a full stack)."""
        seg = self._layer("Segmentation")
        if (
            self._seg_base_masks is None
            or self._seg_base_hand is None
            or self._seg_base_outlines is None
            or not isinstance(seg, Labels)
        ):
            return
        from .core.outline_io import hand_to_outline
        from .core.ta_bond_edit import thin_hand

        masks = np.asarray(seg.data)
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        if masks.shape != self._seg_base_masks.shape:
            return
        if isinstance(data, tuple):
            candidates = [int(data[1])]
        else:
            candidates = range(masks.shape[0])
        frames = [t for t in candidates if not np.array_equal(masks[t], self._seg_base_masks[t])]
        if not frames:
            return
        if masks.dtype != self._seg_base_masks.dtype:
            self._seg_base_masks = self._seg_base_masks.astype(masks.dtype)
        ol = self._layer("Outlines")
        ol_data = self._writable_stack(ol) if isinstance(ol, Labels) else None
        for t in frames:
            self._seg_base_masks[t] = masks[t]
            self._seg_base_hand[t] = thin_hand(masks[t])
            self._seg_base_outlines[t] = hand_to_outline(self._seg_base_hand[t])
            if ol_data is not None:
                ol_data[t] = self._seg_base_outlines[t]
        if ol_data is not None:
            ol.refresh()
        self.status.setText(f"Outlines redrawn on {len(frames)} frame(s) after the cell edit.")

    def _save_segmentation(self) -> None:
        if self._seg_base_masks is None and self._layer("Segmentation") is None:
            self.status.setText(
                "Nothing to save — run segmentation or Start outline editing first."
            )
            return
        # Auto-apply pending bond edits before writing (no-op when none).
        if (
            isinstance(self._layer("Outlines"), Labels)
            and self._seg_base_hand is not None
        ):
            self._apply_bond_edits()
        seg = self._layer("Segmentation")
        if not isinstance(seg, Labels):
            self.status.setText("Missing Segmentation labels layer.")
            return
        masks = np.asarray(seg.data)
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        try:
            from .core.movie_calibration import save_tiff
            from .core.outline_io import masks_to_outline_rgb
        except Exception as exc:
            self.status.setText(f"Save imports failed: {exc}")
            return
        stem = self._seg_stem or "segmentation"
        wf = self._working_folder(
            self._layer(self.seg_image.currentText()),
            self._layer(self.seg_masks.currentText()),
        )
        if wf is None:
            self.status.setText("Not saved: no results folder chosen (use Folder...).")
            return
        new_run = self._seg_run is None
        if new_run:
            # A fresh segmentation (or outlines imported by hand) gets its own run folder;
            # a run loaded from the folder is saved back into its own folder.
            info = dict(self._seg_pending or {"preset": None, "where": "outlines imported in napari"})
            self._seg_run = wf.new_run_name(info.pop("run_base", None) or "edited")
            wf.write_run_params(self._seg_run, info)
            self._seg_pending = None
        outdir = wf.seg_run_dir(self._seg_run)
        masks_path = outdir / f"{stem}_cp_masks.tif"
        # Keep the raw movie's pixel size / frame interval in the saved files.
        cal = self._calibration(
            self._layer(self.seg_image.currentText()),
            self._layer(self.seg_masks.currentText()),
            stem=self._seg_stem,
        ) or self._seg_cal
        save_tiff(masks_path, masks.astype(np.uint16, copy=False), cal)
        bonds = self._seg_base_outlines
        # Outline file: T,Y,X uint8, 255 on the 1 px cell outlines, 0 elsewhere.
        outline_only = (
            bonds if bonds is not None and bonds.shape == masks.shape and self._seg_base_hand is not None
            else (masks_to_outline_rgb(masks).max(axis=-1) > 0)
        )
        cell_outlines_path = outdir / f"{stem}_cell_outlines.tif"
        save_tiff(cell_outlines_path, (np.asarray(outline_only) > 0).astype(np.uint8) * 255, cal)
        self._seg_base_masks = masks.copy()
        wf.set_active_seg_run(self._seg_run)
        self._refresh_seg_runs()
        self.status.setText(
            f"Saved run '{self._seg_run}': {masks_path.name} and {cell_outlines_path.name} → {outdir}"
        )

    # step -> [(layer name, file tag, dtype or None)] written by "Export TIFF"
    _EXPORTS = {
        "tracking": [("Tracked", "tracked_labels", None), ("Tracked RGB", "tracked_rgb", np.uint8)],
        "t1": [("T1 overlay", "t1_overlay", np.uint8), ("T1 four-ways", "t1_fourways", np.uint8)],
    }

    def _export_step_tiffs(self, step: str) -> None:
        """Write the current layers of a pipeline step as calibrated TIFFs into
        ``<results>/<step>/`` (picks up manual edits, unlike the autosaves)."""
        if step == "segmentation":
            self._save_segmentation()
            return
        from .core.movie_calibration import save_tiff
        from .core.output_names import input_stem

        movie = self._layer(self.seg_image.currentText())
        wf = self._working_folder(movie)
        if wf is None:
            self.status.setText("Not exported: no results folder chosen (use Folder...).")
            return
        paths = [p for p in self._layer_paths(movie) if p]
        stem = input_stem(paths[0]) if paths else (self._seg_stem or "movie")
        outdir = wf.run_step_dir(step, self._tracked_run) if step in PER_RUN_STEPS else wf.step_dir(step)
        cal = self._calibration(movie, stem=stem)
        written = []
        for name, tag, dtype in self._EXPORTS[step]:
            layer = self._layer(name)
            if layer is None:
                continue
            data = np.asarray(layer.data)
            if dtype is not None and data.dtype != dtype:
                data = data.astype(dtype)
            path = outdir / f"{stem}_{tag}.tif"
            save_tiff(path, data, cal)
            written.append(path.name)
        self.status.setText(
            f"Exported {', '.join(written)} → {outdir}" if written
            else f"Nothing to export for {step}: run the step first."
        )

    def _calibration(self, *layers, stem: str | None = None):
        """Pixel size / frame interval from the layers' source files, else
        from ``input/<stem>.tif``; ``None`` if neither has it."""
        try:
            from .core.movie_calibration import calibration_for

            from .core.movie_calibration import merge_calibrations

            paths = []
            for ly in layers:
                path = getattr(getattr(ly, "source", None), "path", None)
                if path:
                    paths.append(path)
            cal = calibration_for(*paths, stem=stem)
            # Layers made inside the plugin (Tracked, Segmentation, ...) have no source file: what the movie,
            # a loaded results folder or the last run gave is used for whatever is still missing.
            return merge_calibrations(cal, self._session_cal)
        except Exception:
            return None

    def _remember_calibration(self, *sources) -> None:
        """Remember the pixel size / frame interval found in these layers or files (the newest source wins)."""
        try:
            from .core.movie_calibration import calibration_for, merge_calibrations

            paths = []
            for src in sources:
                path = getattr(getattr(src, "source", None), "path", None) if not isinstance(src, (str, Path)) else src
                if path:
                    paths.append(path)
            found = calibration_for(*paths) if paths else None
            if found is not None:
                self._session_cal = merge_calibrations(found, self._session_cal)
        except Exception:
            pass
        self._show_calibration()

    def _show_calibration(self) -> None:
        """Show what was found in the T1 tab and fill the Cell props boxes."""
        cal = self._calibration(self._layer(self.t1_labels.currentText()) if hasattr(self, "t1_labels") else None)
        um = getattr(cal, "um_per_px", None)
        dt = getattr(cal, "s_per_frame", None)
        if hasattr(self, "t1_cal_label"):
            parts = [f"{um:.4g} \u00b5m/px" if um else "pixel size not found",
                     f"{dt:.4g} s/frame" if dt else "frame interval not found"]
            note = "" if (um and dt) else " - type the missing value(s) below"
            self.t1_cal_label.setText("Movie calibration: " + ", ".join(parts) + note)
        if hasattr(self, "cell_props"):
            self.cell_props.autofill_calibration(um, dt)

    @staticmethod
    def _safe_stem(name: str) -> str:
        text = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))
        text = text.strip("_")
        return text or "segmentation"

    def _tracking_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()
        form = QFormLayout()
        self.track_labels = QComboBox()
        self.track_intensity = QComboBox()
        self.track_expand = self._spin(1, 20, 3)
        self.track_depth = self._spin(1, 8, 2)
        self.track_w = self._dspin(0.0, 5.0, 0.75, 0.05)
        self.track_iter = self._spin(1, 20, 3)
        self.track_appear = self._dspin(0.0, 20.0, 1.5, 0.1)
        self.track_disappear = self._dspin(0.0, 20.0, 1.5, 0.1)
        self.track_division = self._dspin(0.0, 20.0, 0.8, 0.1)
        self.track_gap = self._spin(0, 20, 3)
        self.track_dist = self._dspin(1.0, 500.0, 50.0, 1.0)
        _max_cores = max(os.cpu_count() or 1, 1)
        self.track_jobs = self._spin(1, _max_cores, min(8, _max_cores))
        self.track_out = QLineEdit()
        self.track_out.setPlaceholderText("optional output folder")
        form.addRow("Labels", self.track_labels)
        form.addRow("Intensity", self.track_intensity)
        def tipped(label: str, widget: QWidget, key: str) -> None:
            # Tooltip on the control and on its label text.
            widget.setToolTip(TRACK_TIPS[key])
            form.addRow(label, widget)
            form.labelForField(widget).setToolTip(TRACK_TIPS[key])

        tipped("Expand", self.track_expand, "expand")
        tipped("Pyramid depth", self.track_depth, "depth")
        tipped("Topology weight", self.track_w, "w_topo")
        tipped("Iterations", self.track_iter, "iter")
        tipped("Appear cost", self.track_appear, "appear")
        tipped("Disappear cost", self.track_disappear, "disappear")
        tipped("Division penalty", self.track_division, "division")
        self.track_divisions = QCheckBox("Divisions: new daughter IDs")
        self.track_divisions.setToolTip(
            "Dividing cells end their track; both daughters get new IDs "
            "(magenta dots in Tracked RGB). The T1 tab can then exclude "
            "mothers and daughters."
        )
        self.track_divisions.toggled.connect(self._track_divisions_toggled)
        form.addRow("", self.track_divisions)
        self.track_show_edge = QCheckBox("Show edge cells (Not recommended)")
        self.track_show_edge.setToolTip(
            "Off (default): cells whose track touches the image edge in any frame "
            "are hidden in the Tracked layers, since their shape and neighbours "
            "are cut off. Files written to disk always keep every track. "
            "Applies when tracking is run."
        )
        form.addRow("", self.track_show_edge)
        tipped("Max gap", self.track_gap, "gap")
        tipped("Max distance", self.track_dist, "dist")
        tipped("CPU cores", self.track_jobs, "jobs")
        form.addRow("Save to", self.track_out)
        track_box = QGroupBox("1 · Track")
        track_inner = QVBoxLayout()
        track_inner.addLayout(form)
        self.track_run = QPushButton("Run tracking")
        self.track_run.clicked.connect(self._run_tracking)
        track_inner.addWidget(self.track_run)
        track_box.setLayout(track_inner)
        layout.addWidget(track_box)
        layout.addWidget(self._track_edit_box())

        save_box = QGroupBox("3 · Save / export")
        save_row = QHBoxLayout()
        self.track_save = QPushButton("Save corrected tracks")
        self.track_save.setToolTip(
            "Write the corrected tracking next to the original files so the later "
            "steps use it: tracked labels and RGB, tracks, links, events, lineage "
            "(divisions follow your edits) and QC. The previous files are copied to "
            "a previous_<time> folder first."
        )
        self.track_save.clicked.connect(self._save_track_edits)
        self.track_export = QPushButton("Export TIFF")
        self.track_export.clicked.connect(lambda: self._export_step_tiffs("tracking"))
        save_row.addWidget(self.track_save)
        save_row.addWidget(self.track_export)
        save_box.setLayout(save_row)
        layout.addWidget(save_box)
        layout.addStretch()
        page.setLayout(layout)
        return page

    def _track_edit_box(self) -> QGroupBox:
        group = QGroupBox("2 · Correct tracks")
        inner = QVBoxLayout()
        self.track_edit = QCheckBox("Edit tracks")
        self.track_edit.toggled.connect(self._track_edit_toggled)
        inner.addWidget(self.track_edit)
        modes = QVBoxLayout()  # one way to correct per line
        self.track_modes = QButtonGroup(self)
        self.track_modes.setExclusive(True)
        for i, (key, label) in enumerate(
            (
                ("link", "Link"),
                ("break", "Break"),
                ("swap", "Swap"),
            )
        ):
            button = QRadioButton(label)
            button.setProperty("mode", key)
            self.track_modes.addButton(button)
            button.toggled.connect(self._on_track_mode_toggled)
            # `clicked` fires only for the user, not for setChecked() at start-up
            button.clicked.connect(self._enable_track_editing)
            modes.addWidget(button)
        inner.addLayout(modes)
        self.track_help = QLabel("")
        self.track_help.setWordWrap(True)
        self.track_help.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.track_msg = QLabel("Turn on Edit tracks, then click a cell on the Tracked layer.")
        self.track_msg.setWordWrap(True)
        self.track_msg.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        inner.addWidget(self.track_help)
        inner.addWidget(self.track_msg)
        undo = QPushButton("Undo")
        undo.clicked.connect(self._undo_track)
        inner.addWidget(undo, 0, Qt.AlignLeft)
        unprotect = QPushButton("Clear protection (lets corrected cells be changed again)")
        unprotect.setToolTip(
            "Cells you corrected are protected: a later edit on a neighbouring "
            "cell cannot silently change them. This lifts that protection."
        )
        unprotect.clicked.connect(self._clear_protection)
        inner.addWidget(unprotect, 0, Qt.AlignLeft)
        group.setLayout(inner)
        self.track_modes.buttons()[0].setChecked(True)
        return group

    def _track_mode(self) -> str:
        button = self.track_modes.checkedButton()
        if button is None:
            return "link"
        return str(button.property("mode"))

    def _enable_track_editing(self, *_args) -> None:
        """Choosing how to edit turns editing on."""
        if not self.track_edit.isChecked():
            self.track_edit.setChecked(True)

    def _on_track_mode_toggled(self, checked: bool) -> None:
        if checked:
            self._track_mode_changed()

    def _track_mode_changed(self) -> None:
        self._clear_track_pick()
        help_text = {
            "link": (
                "Click the cell with the right ID (frame x), then the same "
                "cell in a later frame (x+1). It takes the first ID from the "
                "second click through the end of the movie. If another cell "
                "already has that ID there, the two IDs swap."
            ),
            "break": (
                "Click the cell on the first frame that should be a new track. "
                "From this frame on, it gets a new ID."
            ),
            "swap": (
                "Click one cell, then the other. Their IDs exchange from the "
                "second click through the end of the movie."
            ),
        }
        self.track_help.setText(help_text[self._track_mode()])
        self.track_msg.setText("Click a cell on the Tracked layer.")

    def _track_edit_toggled(self, on: bool) -> None:
        if on:
            if self._ensure_tracked_labels():
                self._set_overlay_msg("Edit tracks: click a cell.")
            else:
                self._set_overlay_msg(self._tracked_missing_msg())
            self._update_overlay()
        else:
            self._clear_track_pick()
            self._clear_flash()
            self._set_overlay_msg("")
            self.track_msg.setText(
                "Turn on Edit tracks, then click a cell on the Tracked layer."
            )

    def _on_track_click(self, viewer, event):
        if not self.track_edit.isChecked():
            return
        button = getattr(event, "button", 1)
        if button not in (1, "left"):
            return
        start = np.array(event.position, dtype=float)
        yield
        if getattr(event, "type", "") != "mouse_release":
            return
        end = np.array(event.position, dtype=float)
        # 5 screen pixels of wiggle still counts as a click (not 3 image pixels,
        # which a zoomed-out view turns into a 1-2 pixel trackpad jitter).
        zoom = max(float(getattr(self.viewer.camera, "zoom", 1.0) or 1.0), 1e-6)
        if np.linalg.norm(end - start) > 5.0 / zoom:
            return
        self._apply_track_click(event.position)

    def _apply_track_click(self, position) -> None:
        self._ensure_tracked_labels()
        hit = self._cell_at_tracked(position)
        if hit is None:
            self.track_msg.setText(self._tracked_missing_msg())
            self._set_overlay_msg(self._tracked_missing_msg())
            return
        frame, y, x, track_id = hit
        if track_id <= 0:
            if self._track_pick is None:
                self.track_msg.setText("Click a cell.")
            self._set_overlay_msg("That is background - click a cell.")
            return
        mode = self._track_mode()
        if mode == "break":
            self._commit_track(break_track, (track_id, frame))
            return
        if self._track_pick is None:
            self._track_pick = (frame, y, x, track_id)
            self._show_track_pick(frame, y, x, track_id)
            if mode == "link":
                scope = "The next click changes this frame and every later frame."
                hint = "Now go to the next frame and click the same cell."
            else:
                scope = "The next click changes this frame and every later frame."
                hint = "Now click the other cell."
            self.track_msg.setText(f"Track {track_id} selected. {scope}")
            self._set_overlay_msg(
                f"Picked track {track_id} on frame {frame} (marked). {hint}"
            )
            self._toast(f"Picked track {track_id} (frame {frame}). {hint}")
            return
        src_frame, _y, _x, src_id = self._track_pick
        if mode == "link" and frame <= src_frame:
            msg = f"Click the cell on a later frame than frame {src_frame}."
            self.track_msg.setText(msg)
            self._set_overlay_msg(msg)
            return
        self._clear_track_pick()
        if mode == "link":
            self._commit_track(link_tracks, (src_id, track_id, frame))
        else:
            self._commit_track(swap_tracks, (src_id, track_id, frame))

    def _commit_track(self, fn, args: tuple) -> None:
        layer = self._tracked_layer()
        if layer is None:
            self.track_msg.setText(self._tracked_missing_msg())
            self._set_overlay_msg(self._tracked_missing_msg())
            return
        stack = self._writable_stack(layer)
        kwargs = {}
        if fn is link_tracks and self._protect_log:
            kwargs["protected"] = merge_protected(self._protect_log)
        self._set_overlay_msg("Working ...")
        try:
            undos, message = fn(stack, *args, **kwargs)
        except ProtectedCellsError as exc:
            self._show_blocked(exc, stack)
            return
        except ValueError as exc:
            self.track_msg.setText(str(exc))
            self.status.setText(str(exc))
            self._set_overlay_msg(str(exc))
            return
        self.track_msg.setText(message)
        self.status.setText(message)
        if not undos:
            self._set_overlay_msg(message)
            return
        self._track_undo.append(undos)
        self._track_dirty = True
        self._protect_log.append(changed_pixels(undos, stack.shape[-1]))
        self._refresh_edited(layer, stack, undos)
        self._flash_edit(undos)
        n_frames = len({u.frame for u in undos})
        self._set_overlay_msg(f"{message}  [{n_frames} frame(s); Undo available]")
        self._toast(message)

    @staticmethod
    def _division_ids_from(labels_path) -> set[int]:
        """Mother + daughter track IDs from the lineage.csv next to the labels."""
        lineage = Path(labels_path)
        lineage = lineage.with_name(lineage.name.replace("_tracked_labels.tif", "_lineage.csv"))
        if not lineage.is_file():
            return set()
        try:
            from .core.t1.run_tm_t1 import load_division_ids

            return load_division_ids(lineage)
        except Exception:
            return set()

    def _save_track_edits(self) -> None:
        layer = self._tracked_layer()
        if layer is None:
            self.track_msg.setText(self._tracked_missing_msg())
            return
        src = self._tracking_files
        if src is None:
            src = getattr(getattr(layer, "source", None), "path", None)
        if not src or not Path(src).name.endswith("_tracked_labels.tif") or not Path(src).is_file():
            msg = (
                "Don't know which saved tracking this layer came from. Run tracking "
                "with a working folder, or use Load results, then edit and save."
            )
            self.track_msg.setText(msg)
            self.status.setText(msg)
            return
        if not self._track_dirty:
            self.track_msg.setText("No edits to save.")
            return
        labels_path = Path(src)
        edited = self._writable_stack(layer).copy()
        from .core.movie_calibration import read_calibration
        from .core.output_names import input_stem

        stem = input_stem(labels_path)
        cal = read_calibration(labels_path) or self._calibration(layer, stem=stem)

        def work(progress):
            import tifffile

            from .core.celltrack import resave

            out_dir = labels_path.parent
            progress(5, "reading the original tracking")
            baseline = tifffile.imread(labels_path)
            tables = resave.load_tables(out_dir, stem)
            progress(20, "rebuilding tracks, links, events and lineage")
            rebuilt = resave.rebuild_tracking(edited, baseline, tables)
            progress(70, "writing files")
            written, backup = resave.write_tracking(out_dir, stem, rebuilt, cal)
            return {"rebuilt": rebuilt, "written": written, "backup": backup}

        self._start(work, self._track_edits_saved)

    def _track_edits_saved(self, result: dict) -> None:
        rebuilt, written, backup = result["rebuilt"], result["written"], result["backup"]
        self._replace_image("Tracked RGB", rebuilt.rgb)
        lin = rebuilt.lineage
        if lin is not None:
            kids = lin[lin["parent_track_id"] > 0]
            self._division_ids = set(kids["track_id"]) | set(kids["parent_track_id"])
        self._track_undo.clear()
        self._protect_log.clear()
        self._track_dirty = False
        msg = f"Saved {len(written)} files to {written[0].parent}."
        if backup is not None:
            msg += f" Previous versions: {backup.name}."
        if rebuilt.dropped_divisions:
            msg += (
                f" {len(rebuilt.dropped_divisions)} division(s) no longer hold and were "
                "removed from the lineage: " + "; ".join(rebuilt.dropped_divisions[:3])
            )
        if rebuilt.ambiguous:
            msg += f" Warning: {len(rebuilt.ambiguous)} cell(s) had unclear IDs."
        self.track_msg.setText(msg)
        self.status.setText(msg)
        self._set_overlay_msg(msg)

    def _show_blocked(self, exc: ProtectedCellsError, stack: np.ndarray) -> None:
        """Explain a refused edit and light up the protected cell in red."""
        msg = str(exc)
        self.track_msg.setText(msg)
        self.status.setText(msg)
        self._set_overlay_msg(msg)
        self._toast(msg)
        mask = np.zeros(stack.shape[-2:], dtype=bool)
        mask[exc.ys, exc.xs] = True
        self._cell_outline_image("Track edit (cells)", mask, "red")
        self._flash_timer.start(3500)
        # jump to the frame where the protected cell is, so it can be seen
        try:
            self.viewer.dims.set_point(0, exc.frame)
        except Exception:
            pass

    def _clear_protection(self) -> None:
        n = len(self._protect_log)
        self._protect_log.clear()
        msg = f"Protection cleared ({n} earlier edit(s) are no longer protected)."
        self.track_msg.setText(msg)
        self._set_overlay_msg(msg)

    def _undo_track(self) -> None:
        layer = self._tracked_layer()
        if layer is None or not self._track_undo:
            self.track_msg.setText("Nothing to undo.")
            return
        undos: list[CellUndo] = self._track_undo.pop()
        if self._protect_log:
            self._protect_log.pop()
        stack = self._writable_stack(layer)
        restore_cells(stack, undos)
        self._refresh_edited(layer, stack, undos)
        self._clear_track_pick()
        self._flash_edit(undos)
        self.track_msg.setText("Undid the last track edit.")
        self.status.setText("Undid the last track edit.")
        self._set_overlay_msg("Undid the last track edit.")

    def _writable_stack(self, layer) -> np.ndarray:
        stack = np.asarray(layer.data)
        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]
        if not stack.flags.writeable:
            stack.setflags(write=True)
        return stack

    def _refresh_edited(self, layer, stack: np.ndarray, undos: list[CellUndo]) -> None:
        """Redraw only the cells that were just edited."""
        self._recolor_cells(stack, undos)
        layer.refresh()
        data_event = getattr(layer.events, "data", None)
        if data_event is not None:
            data_event()

    def _recolor_cells(self, stack: np.ndarray, undos: list[CellUndo]) -> None:
        rgb = self._tracked_rgb_layer()
        if rgb is None or not undos:
            return
        max_id = int(np.max(stack))
        rng = np.random.default_rng(1)
        lut = rng.integers(40, 256, size=(max_id + 1, 3), dtype=np.uint8)
        lut[0] = 0
        picture = np.asarray(rgb.data)
        if not picture.flags.writeable:
            picture.setflags(write=True)
        for undo in undos:
            colors = lut[stack[undo.frame][undo.ys, undo.xs]]
            if picture.ndim == 3:
                picture[undo.ys, undo.xs] = colors
            else:
                picture[undo.frame, undo.ys, undo.xs] = colors
        rgb.refresh()
        data_event = getattr(rgb.events, "data", None)
        if data_event is not None:
            data_event()

    def _tracked_layer(self):
        """The Labels layer to edit: 'Tracked', else a single '*tracked*' Labels layer."""
        layer = self._layer("Tracked")
        if isinstance(layer, Labels):
            return layer
        found = [
            ly
            for ly in self.viewer.layers
            if isinstance(ly, Labels)
            and "tracked" in ly.name.lower()
            and "rgb" not in ly.name.lower()
        ]
        if not found:
            return None
        active = self.viewer.layers.selection.active
        return active if active in found else found[0]

    def _tracked_rgb_layer(self):
        layer = self._layer("Tracked RGB")
        if isinstance(layer, Image):
            return layer
        for ly in self.viewer.layers:
            if isinstance(ly, Image) and "tracked_rgb" in ly.name.lower():
                return ly
        return None

    def _ensure_tracked_labels(self) -> bool:
        """Make sure there is a Tracked Labels layer; convert an opened TIFF if needed.

        A *_tracked_labels.tif dragged into napari opens as an Image layer. If one is
        there (integer data, time axis), turn it into the Labels layer 'Tracked'.
        """
        if self._tracked_layer() is not None:
            return True
        for ly in list(self.viewer.layers):
            name = ly.name.lower()
            data = getattr(ly, "data", None)
            if (
                isinstance(ly, Image)
                and "tracked" in name
                and "rgb" not in name
                and not getattr(ly, "rgb", False)
                and data is not None
                and np.ndim(data) >= 3
                and np.issubdtype(np.asarray(data[0:1]).dtype, np.integer)
            ):
                converted = self.viewer.add_labels(
                    data, name="Tracked", scale=ly.scale, translate=ly.translate
                )
                self.viewer.layers.remove(ly)
                self.viewer.layers.selection.active = converted
                self.track_msg.setText(f"Opened {ly.name} as the Tracked labels layer.")
                return True
        return False

    def _tracked_missing_msg(self) -> str:
        return (
            "No tracked labels layer found. Drag in the *_tracked_labels.tif "
            "(or use Load saved results); it should have a time axis."
        )

    def _cell_at_tracked(self, position):
        layer = self._tracked_layer()
        if layer is None:
            return None
        data = np.asarray(layer.data)
        coords = np.asarray(layer.world_to_data(position), dtype=float)
        y = int(round(coords[-2]))
        x = int(round(coords[-1]))
        if data.ndim == 2:
            frame = 0
            height, width = data.shape
        else:
            frame = int(round(float(self.viewer.dims.point[0])))
            frame = int(np.clip(frame, 0, data.shape[0] - 1))
            height, width = data.shape[-2], data.shape[-1]
        if not (0 <= y < height and 0 <= x < width):
            return frame, y, x, 0
        track_id = int(data[y, x] if data.ndim == 2 else data[frame, y, x])
        return frame, y, x, track_id

    # --- feedback: frame counter, pick marker, edit flash ------------------------

    def _n_frames(self) -> int:
        layer = self._tracked_layer()
        if layer is not None and np.ndim(layer.data) >= 3:
            return int(np.shape(layer.data)[0])
        nsteps = getattr(self.viewer.dims, "nsteps", None)
        if nsteps is not None and len(nsteps) >= 3:
            return int(nsteps[0])
        return 1

    def _update_overlay(self, *_args) -> None:
        """Frame counter (top left of the viewer) plus the latest edit message."""
        try:
            overlay = self.viewer.text_overlay
        except AttributeError:  # very old napari without a text overlay
            return
        n = self._n_frames()
        if n <= 1 or self.viewer.dims.ndim < 3:
            overlay.visible = False
            return
        frame = int(round(float(self.viewer.dims.point[0])))
        lines = [f"Frame {frame}  (0-{n - 1})"]
        if self._overlay_msg:
            lines.append(self._overlay_msg)
        overlay.text = "\n".join(lines)
        overlay.position = "top_left"
        overlay.color = "white"
        overlay.font_size = 16
        overlay.visible = True

    def _set_overlay_msg(self, msg: str) -> None:
        self._overlay_msg = msg
        self._update_overlay()

    def _set_marker(self, name: str, data, **style):
        """Create or update a small non-editable Points layer, keeping the active layer."""
        data = np.asarray(data, dtype=float)
        layer = self._layer(name)
        if layer is None:
            active = self.viewer.layers.selection.active
            layer = self.viewer.add_points(data, name=name, **style)
            layer.editable = False
            if active is not None and active in self.viewer.layers:
                self.viewer.layers.selection.active = active
        else:
            layer.data = data
            layer.visible = True
        return layer

    def _hide_marker(self, name: str) -> None:
        layer = self._layer(name)
        if layer is None:
            return
        layer.data = np.zeros((0, int(layer.ndim)), dtype=float)
        layer.visible = False

    def _marker_size(self) -> float:
        """Dot size in image pixels: about 1.5% of the longer image side, at least 14."""
        layer = self._tracked_layer()
        if layer is None:
            return 20.0
        shape = np.shape(layer.data)
        return float(max(14.0, 0.015 * max(shape[-2:])))

    def _toast(self, text: str) -> None:
        """Short pop-up in the napari window (also goes to the status bar)."""
        try:
            self.viewer.status = text
        except Exception:
            pass
        try:
            from napari.utils.notifications import show_info

            show_info(text)
        except Exception:
            pass

    def _cell_outline_image(self, name: str, mask: np.ndarray, colormap: str) -> None:
        """Highlight a cell: translucent fill plus a thick outline, on every frame.

        An additive 2D Image layer, so black stays transparent and the cell shows
        through; it is non-interactive and keeps the active layer.
        """
        from scipy import ndimage as ndi

        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            self._hide_image_marker(name)
            return
        pad = 6
        y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
        crop = mask[y0:y1, x0:x1]
        inner = ndi.binary_erosion(crop, iterations=3, border_value=0)
        edge = crop & ~inner
        img = np.zeros(mask.shape, dtype=np.float32)
        img[y0:y1, x0:x1] = 0.35 * crop + 0.65 * edge
        tracked = self._tracked_layer()
        scale = tracked.scale[-2:] if tracked is not None else (1.0, 1.0)
        translate = tracked.translate[-2:] if tracked is not None else (0.0, 0.0)
        layer = self._layer(name)
        if layer is None:
            active = self.viewer.layers.selection.active
            layer = self.viewer.add_image(
                img,
                name=name,
                colormap=colormap,
                blending="additive",
                contrast_limits=(0.0, 1.0),
                opacity=0.9,
                scale=scale,
                translate=translate,
            )
            if active is not None and active in self.viewer.layers:
                self.viewer.layers.selection.active = active
        else:
            layer.data = img
            layer.visible = True

    def _hide_image_marker(self, name: str) -> None:
        layer = self._layer(name)
        if layer is not None:
            layer.visible = False

    def _show_track_pick(self, frame: int, y: int, x: int, track_id: int = 0) -> None:
        size = self._marker_size()
        # whole cell lit up (yellow) - stays visible when you scrub to the next frame
        tracked = self._tracked_layer()
        if tracked is not None and track_id > 0:
            data = np.asarray(tracked.data)
            plane = data[frame] if data.ndim >= 3 else data
            self._cell_outline_image("Picked cell", plane == track_id, "yellow")
        # a clear dot where you clicked (picked frame only) ...
        self._set_marker(
            "Track selection",
            [[frame, y, x]],
            size=size,
            face_color="yellow",
            border_color="black",
            symbol="disc",
        )
        # ... and a big ring there on every frame
        self._set_marker(
            "Track selection (all frames)",
            [[y, x]],
            size=size * 2.5,
            face_color="transparent",
            border_color="yellow",
            symbol="ring",
        )

    def _clear_track_pick(self) -> None:
        self._track_pick = None
        self._hide_marker("Track selection")
        self._hide_marker("Track selection (all frames)")
        self._hide_image_marker("Picked cell")

    def _flash_edit(self, undos: list[CellUndo]) -> None:
        """Light up the cells that just changed (green) for about two seconds."""
        first = undos[0]
        tracked = self._tracked_layer()
        if tracked is None:
            return
        shape = np.shape(tracked.data)[-2:]
        mask = np.zeros(shape, dtype=bool)
        mask[first.ys, first.xs] = True
        self._cell_outline_image("Track edit (cells)", mask, "green")
        points = []
        for value in np.unique(first.values):
            sel = first.values == value
            ys, xs = first.ys[sel], first.xs[sel]
            cy, cx = ys.mean(), xs.mean()
            i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))  # a pixel inside the cell
            points.append([first.frame, int(ys[i]), int(xs[i])])
        if points:
            self._set_marker(
                "Track edit",
                points,
                size=self._marker_size() * 1.2,
                face_color="lime",
                border_color="black",
                symbol="disc",
            )
        self._flash_timer.start(2500)

    def _clear_flash(self) -> None:
        self._hide_marker("Track edit")
        self._hide_image_marker("Track edit (cells)")

    def _run_tracking(self) -> None:
        labels = self._layer(self.track_labels.currentText())
        if not isinstance(labels, Labels):
            self.status.setText("Select a labels layer to track.")
            return
        masks = np.asarray(labels.data).copy()
        intensity_layer = self._layer(self.track_intensity.currentText())
        intensity = None
        if isinstance(intensity_layer, Image):
            intensity = _as_tyx(np.asarray(intensity_layer.data)).copy()
        out_text = self.track_out.text().strip()
        movie_path = next(
            (p for p in self._layer_paths(intensity_layer, labels) if p), None
        )
        stem = movie_stem(movie_path) if movie_path else labels.name
        self._remember_calibration(intensity_layer, labels)
        cal = self._calibration(intensity_layer, labels)
        wf = None
        out_dir = Path(out_text) if out_text else None
        run = None
        if out_dir is None:
            wf = self._working_folder(intensity_layer, labels)
            if wf is not None:
                # Each segmentation run keeps its own tracking: tracking/<run>/.
                run = (self._seg_run if self.track_labels.currentText() == "Segmentation" else None)                     or wf.active_seg_run()
                out_dir = wf.run_step_dir("tracking", run)
                run = out_dir.name
        kwargs = dict(
            calibration=cal,
            expand=int(self.track_expand.value()),
            pyramidal_depth=int(self.track_depth.value()),
            w_topo=float(self.track_w.value()),
            max_iter=int(self.track_iter.value()),
            appear_cost=float(self.track_appear.value()),
            disappear_cost=float(self.track_disappear.value()),
            division_penalty=float(self.track_division.value()),
            max_gap=int(self.track_gap.value()),
            gap_close=int(self.track_gap.value()) > 0,
            max_distance=float(self.track_dist.value()),
            division_mode="lineage" if self.track_divisions.isChecked() else "legacy",
            n_jobs=int(self.track_jobs.value()),
        )

        def work(progress):
            from .core.celltrack.pipeline import track_arrays
            from pathlib import Path

            progress(1, "tracking")
            return track_arrays(
                masks,
                intensity,
                out_dir,
                stem=stem,
                verbose=True,
                progress=progress,
                **kwargs,
            )

        self._track_run = (
            wf,
            {**{k: v for k, v in kwargs.items() if k not in ("calibration", "n_jobs")},
             "labels_layer": self.track_labels.currentText()},
            movie_path,
            run,
        )
        self._start(work, self._tracking_done)

    def _track_divisions_toggled(self, on: bool) -> None:
        # Lineage mode uses a different penalty scale (see cellpose_track_divisions).
        value = self.track_division.value()
        if on and abs(value - 0.8) < 1e-9:
            self.track_division.setValue(1.0)
        elif not on and abs(value - 1.0) < 1e-9:
            self.track_division.setValue(0.8)

    def _tracking_done(self, result: dict) -> None:
        saved = ""
        self._tracked_run = None
        if self._track_run is not None and self._track_run[0] is not None:
            wf, params, movie_path, run = self._track_run
            wf.write_step_params("tracking", run, params, movie=movie_path)
            self._tracked_run = run
            saved = f" Saved to {wf.run_step_dir('tracking', run, create=False)} (run '{run}')"
        self._track_run = None
        tracked, rgb = result["tracked"], result.get("rgb")
        hidden = 0
        if not self.track_show_edge.isChecked():
            from .core.tracking import edge_track_ids

            edge_ids = edge_track_ids(tracked)
            if edge_ids:
                hide = np.isin(tracked, list(edge_ids))
                tracked = np.where(hide, 0, tracked).astype(tracked.dtype)
                if rgb is not None:
                    rgb = np.where(hide[..., np.newaxis], 0, rgb).astype(rgb.dtype)
                hidden = len(edge_ids)
        self._replace_labels("Tracked", tracked)
        labels_out = result.get("labels_path")
        self._tracking_files = Path(labels_out) if labels_out else None
        if rgb is not None:
            self._replace_image("Tracked RGB", rgb)
        qc = result.get("qc") or {}
        track_result = result["result"]
        parents = getattr(track_result, "parents", {}) or {}
        self._division_ids = set(parents) | set(parents.values())
        extra = ""
        if getattr(track_result, "division_mode", "legacy") == "lineage":
            extra = (
                f" divisions={qc.get('n_divide', 0)} "
                f"(flagged {qc.get('n_divide_flagged', 0)})"
            )
        if hidden:
            extra += f" (hid {hidden} edge tracks)"
        self.status.setText(
            f"Tracking done. tracks={qc.get('n_tracks', track_result.n_tracks)}{extra}{saved}"
        )
        self._refresh_layers()
        self._select_t1_layers()

    def _t1_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()
        form = QFormLayout()
        self.t1_labels = QComboBox()
        self.t1_rgb = QComboBox()
        self.t1_expand = self._spin(1, 20, 3)
        self.t1_filter = QComboBox()
        for label, key in (("All counted events", "all"), ("Clean", "clean"), ("Flicker resolved", "flicker_resolved"),
                           ("Flicker aborted", "flicker_aborted")):
            self.t1_filter.addItem(label, key)
        self.t1_filter.currentIndexChanged.connect(self._apply_t1_filter)
        self.t1_pre = self._spin(0, 40, 3)
        self.t1_post = self._spin(0, 40, 3)
        self.t1_um = QDoubleSpinBox()
        self.t1_um.setDecimals(4)
        self.t1_um.setRange(0.0, 1000.0)
        self.t1_um.setSpecialValueText("from the movie")
        self.t1_um.setToolTip("Pixel size in micrometre. Only needed when the movie's own value was not found.")
        self.t1_dt = QDoubleSpinBox()
        self.t1_dt.setDecimals(3)
        self.t1_dt.setRange(0.0, 1e6)
        self.t1_dt.setSpecialValueText("from the movie")
        self.t1_dt.setToolTip("Time between frames in seconds. Only needed when the movie's own value was not found.")
        self.t1_cal_label = QLabel("")
        self.t1_cal_label.setWordWrap(True)
        self.t1_cal_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        form.addRow("Labels", self.t1_labels)
        form.addRow("RGB", self.t1_rgb)
        form.addRow(self.t1_cal_label)
        form.addRow("Pixel size (\u00b5m)", self.t1_um)
        form.addRow("Frame interval (s)", self.t1_dt)
        form.addRow("Expand", self.t1_expand)
        form.addRow("T1 filter", self.t1_filter)
        form.addRow("Overlay pre", self.t1_pre)
        form.addRow("Overlay post", self.t1_post)
        self.t1_exclude_div = QCheckBox("Exclude dividing cells")
        self.t1_exclude_div.setChecked(True)
        self.t1_exclude_div.setToolTip(
            "Leave mothers and daughters out of T1 detection. Uses the "
            "divisions from the last Tracking run (Divisions on) on the "
            "Tracked layer. After manual edits use Save corrected tracks to "
            "update them."
        )
        form.addRow("", self.t1_exclude_div)
        detect_box = QGroupBox("1 · Detect T1 events")
        detect_inner = QVBoxLayout()
        detect_inner.addLayout(form)
        self.t1_run = QPushButton("Run T1")
        self.t1_run.clicked.connect(self._run_t1)
        detect_inner.addWidget(self.t1_run)
        detect_box.setLayout(detect_inner)
        layout.addWidget(detect_box)
        export_box = QGroupBox("2 · Export")
        export_inner = QVBoxLayout()
        self.t1_export = QPushButton("Export TIFF")
        self.t1_export.clicked.connect(lambda: self._export_step_tiffs("t1"))
        export_inner.addWidget(self.t1_export)
        export_box.setLayout(export_inner)
        layout.addWidget(export_box)
        layout.addStretch()
        page.setLayout(layout)
        return page

    def _run_t1(self) -> None:
        labels = self._layer(self.t1_labels.currentText())
        if not isinstance(labels, Labels):
            self.status.setText("Select tracked labels for T1.")
            return
        stack = np.asarray(labels.data).copy()
        rgb_layer = self._layer(self.t1_rgb.currentText())
        rgb = None
        if isinstance(rgb_layer, Image):
            rgb = _as_rgb(np.asarray(rgb_layer.data)).copy()
        cal = self._calibration(labels, rgb_layer)
        um = float(self.t1_um.value()) or getattr(cal, "um_per_px", None)  # typed value wins over the movie's
        dt = float(self.t1_dt.value()) or getattr(cal, "s_per_frame", None)
        if not um or not dt:
            missing = " and ".join(n for n, v in (("pixel size", um), ("frame interval", dt)) if not v)
            self.status.setText(
                f"The T1 definition needs the {missing} of the movie. It was not found in the movie or in the "
                "results files: type it in the boxes above Run T1."
            )
            return
        exclude = set(self._division_ids) if (
            self.t1_exclude_div.isChecked() and self._division_ids and labels.name == "Tracked") else None
        pre, post = int(self.t1_pre.value()), int(self.t1_post.value())
        definition = {"expand": int(self.t1_expand.value())}

        def work(progress):
            from .core.t1.t1_states_run import run_states_stack

            return run_states_stack(stack, um_per_px=float(um), dt_s=float(dt), rgb=rgb, definition=definition,
                                    pre=pre, post=post, exclude_ids=exclude, progress=progress)

        self._start(work, self._t1_done)

    def _t1_done(self, result: dict) -> None:
        self._t1_result = result
        self._autosave_t1(result)
        self._refresh_layers()
        self._apply_t1_filter()
        note = result.get("swap_note", "")
        if note:
            self.status.setText(f"{self.status.text()} {note}")

    def _apply_t1_filter(self) -> None:
        result = self._t1_result
        if not result:
            return
        from .core.t1.run_tm_t1 import paint_t1_filter

        mode = str(self.t1_filter.currentData() or "all")
        overlay, dots, shown = paint_t1_filter(result, mode)
        self._replace_image("T1 overlay", overlay)
        self._replace_image("T1 four-ways", dots)
        total = len(result.get("events") or [])
        self.status.setText(
            f"T1 filter {self.t1_filter.currentText()}: showing {len(shown)} of {total}. "
            f"Classes {result.get('classes', {})}; not counted {result.get('dropped', {})}."
        )

    # ---- working folder ---------------------------------------------------
    def _folder_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.folder_label = QLabel("Results: set automatically next to the movie")
        self.folder_label.setWordWrap(True)
        self.folder_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        choose = QPushButton("Folder...")
        choose.setToolTip("Choose where results are saved (default: next to the movie)")
        choose.clicked.connect(self._choose_folder)
        load = QPushButton("Load results")
        load.setToolTip("Load segmentation and tracking already saved in the results folder")
        load.clicked.connect(self._load_saved_results)
        help_button = QPushButton("Help")
        help_button.setToolTip("How to use this panel: the workflow and what every button does")
        help_button.clicked.connect(self._show_help)
        bar.addWidget(self.folder_label, 1)
        bar.addWidget(choose)
        bar.addWidget(load)
        bar.addWidget(help_button)
        return bar

    def _show_help(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"napari-tissuekit v{__version__}: how to use")
        dialog.resize(560, 640)
        text = QTextBrowser()
        text.setHtml(HELP_HTML)
        box = QVBoxLayout()
        box.addWidget(text)
        dialog.setLayout(box)
        dialog.show()
        self._help_dialog = dialog  # keep a reference so the window stays open

    @staticmethod
    def _scrollable(page: QWidget) -> QScrollArea:
        """Tab page that scrolls vertically, so the panel fits on laptop screens."""
        scroll = QScrollArea()
        scroll.setWidget(page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    @staticmethod
    def _layer_paths(*layers) -> list:
        return [getattr(getattr(ly, "source", None), "path", None) for ly in layers if ly is not None]

    def _ask_folder(self) -> str:
        return QFileDialog.getExistingDirectory(self, "Choose a folder for the results")

    def _working_folder(self, *layers, ask: bool = True):
        """The results folder for these layers' movie (asks once if it cannot be derived)."""
        root = self.project.resolve(self._layer_paths(*layers))
        if root is None and ask:
            chosen = self._ask_folder()
            if chosen:
                root = self.project.set_manual(chosen)
        self._refresh_folder_label()
        return self.project if root is not None else None

    def _refresh_folder_label(self) -> None:
        layer = self._layer(self.seg_image.currentText()) if hasattr(self, "seg_image") else None
        root = self.project.resolve(self._layer_paths(layer))
        self.folder_label.setText(
            f"Results: {root}" if root is not None
            else "Results: set automatically next to the movie"
        )
        self.folder_label.setToolTip(str(root) if root is not None else "")
        self._refresh_seg_runs()

    def _choose_folder(self) -> None:
        chosen = self._ask_folder()
        if chosen:
            root = self.project.set_manual(find_results_root(chosen))
            self._refresh_folder_label()
            if self.project.found():
                self._load_saved_results()
            else:
                self.status.setText(
                    f"No saved results found in {root} (looked for a tracking/ or segmentation/ "
                    "folder inside it). Pick the *_results folder itself. New results will be "
                    "saved there."
                )

    def _cell_props_dir(self):
        return self.project.root / "cell_props" if self.project.root is not None else None

    def _autosave_segmentation(self) -> None:
        wf = self._working_folder(self._layer(self.seg_image.currentText()))
        if wf is None:
            self.status.setText(f"{self.status.text()} (not saved: no results folder chosen)")
            return
        info = dict(self._seg_pending or {})
        self._save_segmentation()
        info.pop("run_base", None)
        wf.write_settings(
            "segmentation",
            {"model": "cpsam", "run": self._seg_run, **info,
             "batch_size": int(self.seg_params.values()["batch_size"]), "gpu": "auto",
             "fill_holes": self.seg_fill_holes.isChecked()},
            movie=next((p for p in self._layer_paths(self._layer(self.seg_image.currentText())) if p), None),
        )

    def _autosave_t1(self, result: dict) -> None:
        wf = self.project if self.project.root is not None else self._working_folder()
        events = result.get("events") or []
        if wf is None or not events:
            return
        try:
            import pandas as pd

            rows = [
                {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                 for k, v in e.items()}
                for e in events
            ]
            out = wf.run_step_dir("t1", self._tracked_run)  # T1 of the shown tracking's run
            pd.DataFrame(rows).to_csv(out / "t1_events.csv", index=False)
            wf.write_step_params(
                "t1", out.name,
                {"expand": int(self.t1_expand.value()), "definition": result.get("definition")},
            )
            self.status.setText(f"{self.status.text()} Saved {len(rows)} T1 events to {out}")
        except Exception as exc:  # saving must never break the analysis
            self.status.setText(f"{self.status.text()} (T1 events not saved: {exc})")

    def _load_saved_results(self, only_segmentation: bool = False) -> None:
        """Load the movie, the active segmentation run and that run's tracking from the results folder.

        ``only_segmentation``: just the active run and its tracking (Runs > Show); the movie stays.
        Tracked layers of another run are removed when this run has no tracking."""
        wf = self._working_folder(self._layer(self.seg_image.currentText()), ask=False)
        found = wf.found() if wf is not None else {}
        if only_segmentation:
            found = {k: v for k, v in found.items() if k in ("segmentation", "outlines", "tracking")}
        if not found:
            self.status.setText("No saved results found in the results folder.")
            return
        import tifffile

        loaded = []
        run = wf.active_seg_run()
        if "segmentation" in found or "outlines" in found:
            self._seg_run = run
            self._seg_pending = None
            loaded.append(f"run '{run}'")
        # e.g. results made on the HPC: no settings.json, so guess from the mask file names
        movie = self.project.movie_path() or self.project.guess_movie()
        movie_layer = None
        if movie and Path(movie).is_file():
            for ly in self.viewer.layers:
                path = getattr(getattr(ly, "source", None), "path", None)
                if isinstance(ly, Image) and path and Path(path).resolve() == Path(movie).resolve():
                    movie_layer = ly
        if movie and Path(movie).is_file() and movie_layer is None and not only_segmentation:
            before = {id(ly) for ly in self.viewer.layers}
            try:
                self.viewer.open(str(movie), plugin="napari")
            except Exception:
                self.viewer.add_image(tifffile.imread(movie), name=movie_stem(movie))
            movie_layer = next((ly for ly in self.viewer.layers if id(ly) not in before and isinstance(ly, Image)), None)
            loaded.append("movie")
        # Pixel size / frame interval: from the movie, else from the result files (also those made on a server)
        self._session_cal = None
        self._remember_calibration(*([movie] if movie else []), *(f for files in found.values() for f in files[:1]))
        if "segmentation" in found:
            masks = tifffile.imread(found["segmentation"][0])
            self._replace_labels("Segmentation", masks)
            self._seg_base_masks = np.asarray(masks).copy()
            loaded.append("segmentation")
        outline_note = ""
        self._autosave_outlines_ok = False  # loading never rewrites the saved files; only Save does
        if "segmentation" in found:
            from .core.project import SEG_MASK_SUFFIXES, _strip_suffix as _strip

            self._seg_stem = (_strip(found["segmentation"][0].name, SEG_MASK_SUFFIXES) or "segmentation").replace("/", "_")
        if "outlines" not in found and "segmentation" in found:
            self._pending_outlines = None
            self._show_segmentation_layers(
                np.asarray(masks), f"Outline editing ready from the masks (max label {int(np.asarray(masks).max())}). "
                "Paint/erase bonds, Enter to Apply, then Save."
            )
            outline_note = " Bonds are being prepared: wait for 100% before editing."
        if "outlines" in found:
            from .core.movie_calibration import read_calibration
            from .core.outline_io import outline_labels_to_rgb, outlines_rgb_to_masks, rgb_or_labels_to_outline_labels
            from .core.project import OUTLINE_SUFFIXES, _strip_suffix

            ofile = Path(found["outlines"][0])
            outlines = rgb_or_labels_to_outline_labels(tifffile.imread(ofile))
            self._seg_cal = read_calibration(ofile)
            self._seg_stem = (_strip_suffix(ofile.name, OUTLINE_SUFFIXES) or ofile.stem).replace("/", "_")  # keep the file's own name
            if "segmentation" in found:
                # masks and outlines: editing starts right away; differences between the file and the masks apply on Enter
                self._pending_outlines = outlines
                self._show_segmentation_layers(
                    np.asarray(masks), f"Outline editing ready from {ofile.name} (max label {int(np.asarray(masks).max())}). "
                    "Paint/erase bonds, Enter to Apply, then Save.", saved_outlines=outlines,
                )
                outline_note = " Bonds are being prepared: wait for 100% before editing."
                loaded.append(f"outlines {ofile.name}")
            else:
                self._pending_outlines = None
                # outlines only: the cells are the regions between them, ready for outline editing and saving
                masks = outlines_rgb_to_masks(outline_labels_to_rgb(outlines), background="auto")
                self._seg_base_masks = masks.copy()
                self._show_segmentation_layers(
                    masks, f"Outline editing ready from {ofile.name} (max label {int(masks.max())}). "
                    "Paint/erase bonds, Enter to Apply, then Save (writes *_cp_masks.tif and *_cell_outlines.tif).",
                    saved_outlines=outlines,
                )
                outline_note = " Bonds are being prepared: wait for 100% before editing."
                loaded.append(f"outlines {ofile.name} (cells derived from the outlines)")
        if "tracking" not in found and self._tracked_run not in (None, run):
            self._drop_tracking_layers()  # they belong to another run
        if "tracking" in found:
            labels = found["tracking"][0]
            self._replace_labels("Tracked", tifffile.imread(labels))
            self._tracking_files = Path(labels)
            self._tracked_run = run
            self._division_ids = self._division_ids_from(labels)
            rgb = labels.with_name(labels.name.replace("_tracked_labels", "_tracked_rgb"))
            if rgb.is_file():
                self._replace_image("Tracked RGB", tifffile.imread(rgb))
            loaded.append("tracking")
        self._refresh_layers()
        self._refresh_seg_runs()
        self._select_loaded_layers(movie_layer)
        self.status.setText(f"Loaded {', '.join(loaded)} from {wf.root}.{outline_note}{self._tracking_run_note(wf)}")

    def _tracking_run_note(self, wf) -> str:
        """Warning when the shown tracking is an older one that may not come from the active run."""
        files = self._tracking_files
        if files is None or self._layer("Tracked") is None or files.parent.name == self._tracked_run:
            return ""
        if wf.legacy_tracking_run() is None:  # older flat tracking/ folder, run not recorded
            return (" Note: this tracking is from before tracking was kept per run; it may come from "
                    "another segmentation run. Run tracking again to save it under this run.")
        return ""

    def _drop_tracking_layers(self) -> None:
        """Remove the Tracked and T1 layers (switching to a run without its own tracking)."""
        for name in ("Tracked", "Tracked RGB", "T1 overlay", "T1 four-ways"):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        self._tracking_files = None
        self._tracked_run = None
        self._track_dirty = False
        self._track_undo.clear()
        self._protect_log.clear()
        self._division_ids = set()
        self._t1_result = None

    # ---- segmentation runs -------------------------------------------------------
    def _next_run_name(self, base: str) -> str:
        """Next free run name in the results folder (for the Run name hint)."""
        layer = self._layer(self.seg_image.currentText()) if hasattr(self, "seg_image") else None
        root = self.project.resolve(self._layer_paths(layer))
        if root is None:
            from .core.seg_params import run_name

            return run_name(base)
        return self.project.new_run_name(base)

    def _refresh_seg_runs(self) -> None:
        """Fill the Runs box from the results folder (active run selected)."""
        if not hasattr(self, "seg_runs"):
            return
        runs = self.project.seg_runs() if self.project.root is not None else []
        active = self.project.active_seg_run() if runs else None
        keep = self.seg_runs.currentData()
        self.seg_runs.blockSignals(True)
        try:
            self.seg_runs.clear()
            for run in runs:
                info = self.project.read_run_params(run)
                self.seg_runs.addItem(f"{run}  (active)" if run == active else run, run)
                tip = run
                if info:
                    vals = ", ".join(f"{k} {v}" for k, v in (info.get("params") or {}).items())
                    tip = f"{run}: preset {info.get('preset')}; {vals}; {info.get('time', '')}"
                self.seg_runs.setItemData(self.seg_runs.count() - 1, tip, Qt.ToolTipRole)
            pick = keep if keep in runs else active
            if pick is not None:
                self.seg_runs.setCurrentIndex(runs.index(pick))
        finally:
            self.seg_runs.blockSignals(False)
        for w in (self.seg_run_show, self.seg_run_compare):
            w.setEnabled(bool(runs))
        self.seg_params.refresh_run_hint()

    def _segmentation_edited(self) -> bool:
        seg = self._layer("Segmentation")
        base = self._seg_base_masks
        if not isinstance(seg, Labels) or base is None:
            return False
        data = np.asarray(seg.data)
        return data.shape != base.shape or not np.array_equal(data, base)

    def _show_seg_run(self) -> None:
        """Make the chosen run the active one and load it for editing / tracking."""
        run = self.seg_runs.currentData()
        if run is None or self.project.root is None:
            self.status.setText("No segmentation runs in the results folder yet.")
            return
        if self._job_running():
            self.status.setText("A job is already running.")
            return
        if run != self._seg_run and self._segmentation_edited():
            answer = QMessageBox.question(
                self, "Unsaved edits",
                f"The segmentation shown (run '{self._seg_run or 'not saved'}') has unsaved edits.\n"
                f"Discard them and show run '{run}'?",
            )
            if answer != QMessageBox.Yes:
                return
        if run != self._tracked_run and self._track_dirty and QMessageBox.question(
            self, "Unsaved edits",
            f"The Tracked layer has unsaved edits. Discard them and show run '{run}' with its own tracking?",
        ) != QMessageBox.Yes:
            return
        self.project.set_active_seg_run(run)
        self._load_saved_results(only_segmentation=True)

    _COMPARE_COLORS = ("cyan", "magenta", "lime", "orange", "deepskyblue", "violet", "white")

    def _compare_seg_run(self) -> None:
        """Overlay a run's outlines as a read-only layer in its own colour."""
        run = self.seg_runs.currentData()
        if run is None or self.project.root is None:
            self.status.setText("No segmentation runs in the results folder yet.")
            return
        import tifffile

        found = self.project.found(run=run)
        if "outlines" in found:
            from .core.outline_io import rgb_or_labels_to_outline_labels

            lines = rgb_or_labels_to_outline_labels(tifffile.imread(found["outlines"][0])) > 0
        elif "segmentation" in found:
            from .core.outline_io import masks_to_outline_labels

            lines = masks_to_outline_labels(np.asarray(tifffile.imread(found["segmentation"][0]))) > 0
        else:
            self.status.setText(f"Run '{run}' has no masks or outlines.")
            return
        name = f"Seg · {run}"
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        used = {ly.metadata.get("compare_color") for ly in self.viewer.layers if ly.name.startswith("Seg · ")}
        color = next((c for c in self._COMPARE_COLORS if c not in used), "white")
        active = self.viewer.layers.selection.active
        layer = self.viewer.add_labels(
            lines.astype(np.uint8), name=name, opacity=1.0,
            metadata={"compare_color": color, "segmentation_run": run},  # set before "inserted" fires
        )
        try:
            from napari.utils.colormaps import DirectLabelColormap

            layer.colormap = DirectLabelColormap(color_dict={0: "transparent", 1: color, None: "transparent"})
        except Exception:
            pass
        layer.editable = False
        if active is not None:  # keep painting on the layer that was selected (e.g. Outlines)
            self.viewer.layers.selection.active = active
        self.status.setText(f"Added run '{run}' as '{name}' ({color} outlines, read-only).")

    @staticmethod
    def _pick(box: QComboBox, name: str | None) -> None:
        idx = box.findText(name) if name else -1
        if idx >= 0:
            box.setCurrentIndex(idx)

    def _select_t1_layers(self) -> None:
        """T1 tab: the tracked labels and, when there is one, the tracked RGB."""
        if self._layer("Tracked") is not None:
            self._pick(self.t1_labels, "Tracked")
        if self._layer("Tracked RGB") is not None:
            self._pick(self.t1_rgb, "Tracked RGB")

    def _select_loaded_layers(self, movie_layer) -> None:
        """Point the layer drop-downs at the layers that loading a results folder just made."""
        pick = self._pick
        pick(self.seg_image, movie_layer.name if movie_layer is not None else None)
        if self._layer("Segmentation") is not None:
            pick(self.seg_masks, "Segmentation")
            pick(self.track_labels, "Segmentation")  # tracking starts from the segmentation, not from earlier tracks
        if self._layer("Outlines") is not None:
            pick(self.seg_outlines, "Outlines")
        if movie_layer is not None:
            pick(self.track_intensity, movie_layer.name)
        self._select_t1_layers()

    def _replace_labels(self, name: str, data: np.ndarray) -> None:
        if name == "Tracked":  # new tracks: old undo steps and protection are stale
            self._track_undo.clear()
            self._protect_log.clear()
            self._track_dirty = False
        layer = self._layer(name)
        if isinstance(layer, Labels):
            layer.data = data
        else:
            self.viewer.add_labels(data, name=name)

    def _replace_image(self, name: str, data: np.ndarray) -> None:
        layer = self._layer(name)
        if isinstance(layer, Image):
            layer.data = data
        else:
            self.viewer.add_image(data, name=name, rgb=data.ndim == 4 and data.shape[-1] == 3)

    @staticmethod
    def _spin(lo: int, hi: int, value: int) -> QSpinBox:
        box = QSpinBox()
        box.setRange(lo, hi)
        box.setValue(value)
        return box

    @staticmethod
    def _dspin(lo: float, hi: float, value: float, step: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setDecimals(2)
        box.setSingleStep(step)
        box.setValue(value)
        return box
