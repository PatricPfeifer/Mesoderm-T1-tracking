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
from qtpy.QtCore import QObject, QProcess, QProcessEnvironment, QSize, QThread, Qt, Signal
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QShortcut,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .cell_properties import CellPropertiesWidget
from .paths import ensure_scripts, repo_root
from .project import WorkingFolder, find_results_root, movie_stem
from .track_edit import (
    CellUndo,
    break_track,
    link_tracks,
    restore_cells,
    set_id_on_frame,
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


def _as_rgb(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[np.newaxis, ...]
    if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
        arr = np.moveaxis(arr, 1, -1)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected T,Y,X,3 RGB, got {arr.shape}")
    return arr


class PatricWidget(QWidget):
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
        # TA handCorrection stack (255 = 1 px bond / background): the bonds
        # shown in Outlines and used by the watershed.
        self._seg_base_hand: np.ndarray | None = None
        self._hand_msg = ""
        self._seg_stem: str = "segmentation"
        self._track_pick: tuple[int, int, int, int] | None = None
        self._track_undo: deque = deque(maxlen=20)
        self._pick_layer = None
        self._t1_result: dict | None = None
        self.project = WorkingFolder()
        self._track_run: tuple | None = None
        # Mother + daughter track IDs from the last division-aware tracking.
        self._division_ids: set[int] = set()
        ensure_scripts()

        tabs = QTabWidget()
        tabs.addTab(self._segmentation_tab(), "Segmentation")
        tabs.addTab(self._tracking_tab(), "Tracking")
        tabs.addTab(self._t1_tab(), "T1")
        self.cell_props = CellPropertiesWidget(self.viewer)
        self.cell_props.dir_provider = self._cell_props_dir
        tabs.addTab(self.cell_props, "Export Cell Props")
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout = QVBoxLayout()
        layout.addLayout(self._folder_bar())
        layout.addWidget(tabs)
        layout.addWidget(self.status)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.viewer.mouse_drag_callbacks.append(self._on_track_click)
        self._install_bond_shortcuts()
        self._refresh_layers()

    def sizeHint(self) -> QSize:
        return QSize(320, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        return QSize(240, super().minimumSizeHint().height())

    def _refresh_layers(self, *_args) -> None:
        if hasattr(self, 'folder_label'):
            self._refresh_folder_label()
        images = [ly.name for ly in self.viewer.layers if isinstance(ly, Image)]
        labels = [ly.name for ly in self.viewer.layers if isinstance(ly, Labels)]
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
        self.status.setText(text.strip().splitlines()[-1])
        print(text, flush=True)

    def _clear_job(self) -> None:
        self._set_busy(False)
        self._thread = None
        self._job = None

    def _set_busy(self, busy: bool) -> None:
        for button in (self.seg_run, self.track_run, self.t1_run):
            button.setEnabled(not busy)

    def _segmentation_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()
        form = QFormLayout()
        self.seg_image = QComboBox()
        self.seg_batch = QSpinBox()
        self.seg_batch.setRange(1, 64)
        self.seg_batch.setValue(8)
        self.seg_gpu = QCheckBox("Use GPU")
        self.seg_gpu.setChecked(False)
        form.addRow("Image", self.seg_image)
        form.addRow("Batch size", self.seg_batch)
        form.addRow("", self.seg_gpu)
        layout.addLayout(form)
        self.seg_run = QPushButton("Run segmentation")
        self.seg_run.clicked.connect(self._run_segmentation)
        layout.addWidget(self.seg_run)

        outline_box = QGroupBox("Outline correction (Tissue Analyzer style)")
        outline_inner = QVBoxLayout()
        outline_inner.addWidget(
            QLabel(
                "Paint bonds (label 1) / erase bonds (label 0) on yellow Outlines. "
                "Enter = Apply (local watershed, like TA). "
                "Shift+Enter = Apply + drop small cells. "
                "Ctrl+M = seeded local watershed (paint ≥2 seed dots first). "
                "M = toggle Outlines. Ctrl+S / Save writes masks+outlines."
            )
        )
        import_form = QFormLayout()
        self.seg_masks = QComboBox()
        self.seg_outlines = QComboBox()
        self.seg_min_area = QSpinBox()
        self.seg_min_area.setRange(0, 100000)
        self.seg_min_area.setValue(10)
        import_form.addRow("Masks", self.seg_masks)
        import_form.addRow("Outlines", self.seg_outlines)
        import_form.addRow("Min cell area", self.seg_min_area)
        self.seg_pad = QSpinBox()
        self.seg_pad.setRange(8, 512)
        self.seg_pad.setValue(128)
        self.seg_pad.setToolTip(
            "Context margin (px) around each edit for the local watershed. "
            "TA uses 128; smaller is faster."
        )
        import_form.addRow("Watershed pad", self.seg_pad)
        outline_inner.addLayout(import_form)
        self.seg_start_edit = QPushButton("Start outline editing")
        self.seg_start_edit.clicked.connect(self._start_outline_editing)
        outline_inner.addWidget(self.seg_start_edit)
        self.seg_commit = QPushButton("Apply bond edits (Enter)")
        self.seg_commit.clicked.connect(lambda: self._apply_bond_edits(False))
        self.seg_apply_shift = QPushButton("Apply + drop small cells (Shift+Enter)")
        self.seg_apply_shift.clicked.connect(lambda: self._apply_bond_edits(True))
        self.seg_seeded = QPushButton("Seeded local watershed (Ctrl+M)")
        self.seg_seeded.clicked.connect(self._seeded_local_watershed)
        self.seg_save = QPushButton("Save segmentation")
        self.seg_save.clicked.connect(self._save_segmentation)
        outline_inner.addWidget(self.seg_commit)
        outline_inner.addWidget(self.seg_apply_shift)
        outline_inner.addWidget(self.seg_seeded)
        outline_inner.addWidget(self.seg_save)
        outline_box.setLayout(outline_inner)
        layout.addWidget(outline_box)

        self._add_correction(layout)
        layout.addStretch()
        page.setLayout(layout)
        return page

    def _add_correction(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("Manual correction (filled labels, optional)")
        inner = QVBoxLayout()
        try:
            from napariCellFlow.cell_correction_widget import CellCorrectionWidget
            from napariCellFlow.data_manager import DataManager
            from napariCellFlow.visualization_manager import VisualizationManager
        except Exception as exc:
            self.correction = None
            inner.addWidget(
                QLabel(
                    "CellFlow correction tools are not importable "
                    f"({exc}). Outline edit/save still work."
                )
            )
        else:
            self._data_manager = DataManager()
            self._vis = VisualizationManager(self.viewer, self._data_manager)
            self.correction = CellCorrectionWidget(
                self.viewer, self._data_manager, self._vis
            )
            inner.addWidget(self.correction)
        group.setLayout(inner)
        layout.addWidget(group)

    def _run_segmentation(self) -> None:
        if self._job_running():
            self.status.setText("A job is already running.")
            return
        layer = self._layer(self.seg_image.currentText())
        if layer is None:
            self.status.setText("Select an image layer.")
            return
        self._seg_stem = self._safe_stem(layer.name)
        image = _as_tyx(np.asarray(layer.data))
        batch = int(self.seg_batch.value())
        use_gpu = self.seg_gpu.isChecked()
        work = Path(tempfile.mkdtemp(prefix="patric_seg_"))
        image_path = work / "image.npy"
        out_path = work / "masks.npy"
        np.save(image_path, np.asarray(image))
        self._seg_dir = work
        self._seg_out = out_path
        root = repo_root()
        pythonpath = os.pathsep.join(
            [
                str(root / "scripts" / "cellpose_seg"),
                str(root / "scripts"),
                os.environ.get("PYTHONPATH", ""),
            ]
        )
        code = (
            "import numpy as np\n"
            "from run_cellpose_naima import segment_stack\n"
            f"img = np.load({str(image_path)!r})\n"
            "masks = segment_stack(\n"
            "    img,\n"
            f"    batch_size={batch},\n"
            f"    use_gpu={use_gpu},\n"
            ")\n"
            f"np.save({str(out_path)!r}, masks)\n"
            "print('SEGMENT_DONE', int(masks.max()), flush=True)\n"
        )
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONPATH", pythonpath)
        self._proc = QProcess(self)
        self._proc.setProcessEnvironment(env)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._seg_output)
        self._proc.finished.connect(self._seg_finished)
        self._set_busy(True)
        self.status.setText("Starting segmentation...")
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
        self._show_segmentation_layers(
            masks,
            f"Segmentation done. max label {int(np.max(masks))}. "
            "Paint/erase yellow Outlines, Enter to Apply, then Save / track.",
        )
        self._refresh_layers()
        self._autosave_segmentation()

    def _start_outline_editing(self) -> None:
        masks_ly = self._layer(self.seg_masks.currentText())
        outlines_ly = self._layer(self.seg_outlines.currentText())
        if masks_ly is None:
            self.status.setText("Select a Masks layer (imported *_cp_masks.tif).")
            return
        if outlines_ly is None:
            self.status.setText("Select an Outlines layer (imported *_cp_outlines.tif).")
            return
        try:
            from outline_io import rgb_or_labels_to_outline_labels
        except Exception as exc:
            self.status.setText(f"outline_io import failed: {exc}")
            return
        try:
            masks = _as_tyx(np.asarray(masks_ly.data))
        except ValueError as exc:
            self.status.setText(f"Masks layer must be T,Y,X labels: {exc}")
            return
        try:
            outlines = rgb_or_labels_to_outline_labels(np.asarray(outlines_ly.data))
        except Exception as exc:
            self.status.setText(f"Could not read outlines: {exc}")
            return
        if outlines.shape != masks.shape:
            self.status.setText(
                f"Outlines shape {outlines.shape} != masks shape {masks.shape}."
            )
            return
        self._seg_stem = self._safe_stem(masks_ly.name)
        # Prefer raw image name for save stem when selected.
        img_ly = self._layer(self.seg_image.currentText())
        if img_ly is not None:
            self._seg_stem = self._safe_stem(img_ly.name)
        self._seg_base_masks = masks.copy()
        # Bonds are rebuilt as 1 px TA bonds from the masks; the imported
        # outlines are only shape-checked.
        self._show_segmentation_layers(
            masks,
            f"Outline editing ready (max label {int(masks.max())}). "
            "Paint/erase bonds, Enter to Apply (TA watershed), then Save.",
        )
        self._refresh_layers()

    def _install_bond_shortcuts(self) -> None:
        # ApplicationShortcut so Enter works while the canvas has focus (TA-like).
        sc_enter = QShortcut(QKeySequence(Qt.Key_Return), self)
        sc_enter.setContext(Qt.ApplicationShortcut)
        sc_enter.activated.connect(lambda: self._apply_bond_edits(False))
        sc_pad = QShortcut(QKeySequence(Qt.Key_Enter), self)
        sc_pad.setContext(Qt.ApplicationShortcut)
        sc_pad.activated.connect(lambda: self._apply_bond_edits(False))
        sc_shift = QShortcut(QKeySequence("Shift+Return"), self)
        sc_shift.setContext(Qt.ApplicationShortcut)
        sc_shift.activated.connect(lambda: self._apply_bond_edits(True))
        sc_shift2 = QShortcut(QKeySequence("Shift+Enter"), self)
        sc_shift2.setContext(Qt.ApplicationShortcut)
        sc_shift2.activated.connect(lambda: self._apply_bond_edits(True))
        sc_m = QShortcut(QKeySequence("Ctrl+M"), self)
        sc_m.setContext(Qt.ApplicationShortcut)
        sc_m.activated.connect(self._seeded_local_watershed)
        sc_toggle = QShortcut(QKeySequence(Qt.Key_M), self)
        sc_toggle.setContext(Qt.ApplicationShortcut)
        sc_toggle.activated.connect(self._toggle_outlines_visibility)
        sc_save = QShortcut(QKeySequence("Ctrl+S"), self)
        sc_save.setContext(Qt.ApplicationShortcut)
        sc_save.activated.connect(self._save_segmentation)
        self._bond_shortcuts = (sc_enter, sc_pad, sc_shift, sc_shift2, sc_m, sc_toggle, sc_save)

    def _toggle_outlines_visibility(self) -> None:
        ol = self._layer("Outlines")
        if isinstance(ol, Labels):
            ol.visible = not ol.visible
            self.status.setText(
                f"Outlines {'shown' if ol.visible else 'hidden'} (M)."
            )

    def _show_segmentation_layers(self, masks: np.ndarray, message: str) -> None:
        """Show masks, then build 1 px TA bonds in the background for Outlines."""
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        if self._job_running():
            self.status.setText("A job is already running.")
            return
        self._seg_base_hand = None
        self._seg_base_outlines = None

        if self.correction is not None:
            self._data_manager.initialize_stack(int(masks.shape[0]))
            self._data_manager.segmentation_data = masks
            self.correction.set_masks_layer(masks)

        # Own copy: edits are written in place and must not touch the input layer.
        self._replace_labels("Segmentation", masks.copy())
        seg = self._layer("Segmentation")
        if isinstance(seg, Labels):
            seg.opacity = 0.25
            seg.visible = False

        self._hand_msg = message

        def work(progress):
            from outline_io import hand_stack_to_outline
            from ta_bond_edit import thin_hand_stack

            hand = thin_hand_stack(masks, progress)
            return hand, hand_stack_to_outline(hand)

        self._start(work, self._hand_ready)

    def _hand_ready(self, result: tuple) -> None:
        hand, outlines = result
        self._seg_base_hand = hand
        self._seg_base_outlines = outlines.copy()
        self._replace_labels("Outlines", outlines)
        ol = self._layer("Outlines")
        if isinstance(ol, Labels):
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
        self.status.setText(self._hand_msg)

    def _bonds_ready(self) -> bool:
        if self._seg_base_masks is None:
            self.status.setText(
                "No base masks — run segmentation or Start outline editing first."
            )
            return False
        if self._seg_base_hand is None or self._seg_base_outlines is None:
            self.status.setText("Bonds are still being prepared — wait for 100%.")
            return False
        return True

    def _apply_bond_edits(self, drop_small: bool) -> bool:
        """Apply painted bond strokes. Returns True if any frame changed."""
        if not self._bonds_ready():
            return False
        ol = self._layer("Outlines")
        if not isinstance(ol, Labels):
            self.status.setText("Missing Outlines layer.")
            return False

        try:
            from ta_bond_edit import apply_bond_strokes, changed_frames
        except Exception as exc:
            self.status.setText(f"TA bond edit import failed: {exc}")
            return False

        t0 = time.perf_counter()
        edited = np.asarray(ol.data)
        if edited.ndim == 2:
            edited = edited[np.newaxis, ...]
        frames = changed_frames(self._seg_base_outlines, edited)
        min_area = int(self.seg_min_area.value()) if drop_small else 0
        if min_area and not frames:
            # Drop small cells on the frame being viewed, not the whole movie.
            frames = [self._current_frame(self._seg_base_masks.shape[0])]
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
                min_seed_area=min_area,
                pad=int(self.seg_pad.value()),
            )
        except Exception as exc:
            self.status.setText(f"Apply bond edits failed: {exc}")
            print(traceback.format_exc(), flush=True)
            return False

        self._commit_seg(updated, frames, rois)
        if not rois:
            self.status.setText(
                "No cells changed. A stroke outside the tissue must enclose "
                "an area (closed loop, or closed against the tissue edge)."
            )
            return False
        extra = f", min_area={min_area}" if drop_small else ""
        self.status.setText(
            f"Applied bond edits on {len(frames)} frame(s) (TA wshed{extra}) "
            f"in {time.perf_counter() - t0:.2f}s. max label "
            f"{int(self._seg_base_masks.max())}."
        )
        return True

    def _current_frame(self, n_frames: int) -> int:
        frame = int(round(float(self.viewer.dims.point[0]))) if n_frames > 1 else 0
        return int(np.clip(frame, 0, n_frames - 1))

    def _commit_seg(self, updated: np.ndarray, frames: list[int], rois: list) -> None:
        """Push changed frames into the Segmentation / Outlines layers in place."""
        from outline_io import update_outline_roi

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
            ol.refresh()
        else:
            self._replace_labels("Outlines", outlines.copy())
        if self.correction is not None:
            self._data_manager.segmentation_data = self._seg_base_masks
            try:
                self.correction.set_masks_layer(self._seg_base_masks)
            except Exception:
                pass

    def _seeded_local_watershed(self) -> None:
        if not self._bonds_ready():
            return
        ol = self._layer("Outlines")
        img_ly = self._layer(self.seg_image.currentText())
        if not isinstance(ol, Labels):
            self.status.setText("Missing Outlines layer.")
            return
        if img_ly is None:
            self.status.setText("Select an Image layer for seeded watershed.")
            return
        try:
            from ta_bond_edit import changed_frames, seeded_local_wshed
        except Exception as exc:
            self.status.setText(f"TA bond edit import failed: {exc}")
            return
        t0 = time.perf_counter()
        try:
            image = _as_tyx(np.asarray(img_ly.data))
            seeds = np.asarray(ol.data)
            if seeds.ndim == 2:
                seeds = seeds[np.newaxis, ...]
            frames = changed_frames(self._seg_base_outlines, seeds)
            if not frames:
                self.status.setText("Paint seed dots first (Ctrl+M).")
                return
            updated, rois = seeded_local_wshed(
                image,
                self._seg_base_masks,
                self._seg_base_hand,
                seeds,
                self._seg_base_outlines,
                frames=frames,
            )
        except Exception as exc:
            self.status.setText(f"Seeded watershed failed: {exc}")
            print(traceback.format_exc(), flush=True)
            return

        self._commit_seg(updated, frames, rois)
        self.status.setText(
            f"Seeded local watershed done on {len(frames)} frame(s) in "
            f"{time.perf_counter() - t0:.2f}s. max label "
            f"{int(self._seg_base_masks.max())}."
        )

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
            self._apply_bond_edits(False)
        seg = self._layer("Segmentation")
        if not isinstance(seg, Labels):
            self.status.setText("Missing Segmentation labels layer.")
            return
        masks = np.asarray(seg.data)
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        try:
            from movie_calibration import save_tiff
            from outline_io import masks_to_outline_rgb, outline_labels_to_rgb
        except Exception as exc:
            self.status.setText(f"Save imports failed: {exc}")
            return
        stem = self._seg_stem or "segmentation"
        wf = self._working_folder(
            self._layer(self.seg_image.currentText()),
            self._layer(self.seg_masks.currentText()),
        )
        if wf is not None:
            outdir = wf.step_dir("segmentation")
        else:  # no folder chosen: old behaviour, inside the repo
            outdir = repo_root() / "output" / "cellpose_seg" / f"{stem}_cp_seg"
            outdir.mkdir(parents=True, exist_ok=True)
        masks_path = outdir / f"{stem}_cp_masks.tif"
        outlines_path = outdir / f"{stem}_cp_outlines.tif"
        # Keep the raw movie's pixel size / frame interval in the saved files.
        cal = self._calibration(
            self._layer(self.seg_image.currentText()),
            self._layer(self.seg_masks.currentText()),
            stem=self._seg_stem,
        )
        save_tiff(masks_path, masks.astype(np.uint16, copy=False), cal)
        bonds = self._seg_base_outlines
        if self._seg_base_hand is not None and bonds is not None and bonds.shape == masks.shape:
            save_tiff(outlines_path, outline_labels_to_rgb(bonds), cal)  # 1 px TA bonds
        else:
            save_tiff(outlines_path, masks_to_outline_rgb(masks), cal)
        self._seg_base_masks = masks.copy()
        self.status.setText(f"Saved {masks_path.name} and {outlines_path.name} → {outdir}")

    def _calibration(self, *layers, stem: str | None = None):
        """Pixel size / frame interval from the layers' source files, else
        from ``input/<stem>.tif``; ``None`` if neither has it."""
        try:
            from movie_calibration import calibration_for

            paths = []
            for ly in layers:
                path = getattr(getattr(ly, "source", None), "path", None)
                if path:
                    paths.append(path)
            return calibration_for(*paths, stem=stem)
        except Exception:
            return None

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
        self.track_depth = self._spin(1, 8, 3)
        self.track_w = self._dspin(0.0, 5.0, 0.75, 0.05)
        self.track_iter = self._spin(1, 20, 3)
        self.track_appear = self._dspin(0.0, 20.0, 1.5, 0.1)
        self.track_disappear = self._dspin(0.0, 20.0, 1.5, 0.1)
        self.track_division = self._dspin(0.0, 20.0, 0.8, 0.1)
        self.track_gap = self._spin(0, 20, 3)
        self.track_dist = self._dspin(1.0, 500.0, 50.0, 1.0)
        self.track_out = QLineEdit()
        self.track_out.setPlaceholderText("optional output folder")
        form.addRow("Labels", self.track_labels)
        form.addRow("Intensity", self.track_intensity)
        form.addRow("Expand", self.track_expand)
        form.addRow("Pyramid depth", self.track_depth)
        form.addRow("Topology weight", self.track_w)
        form.addRow("Iterations", self.track_iter)
        form.addRow("Appear cost", self.track_appear)
        form.addRow("Disappear cost", self.track_disappear)
        form.addRow("Division penalty", self.track_division)
        self.track_divisions = QCheckBox("Divisions: new daughter IDs")
        self.track_divisions.setToolTip(
            "Dividing cells end their track; both daughters get new IDs "
            "(magenta dots in Tracked RGB). The T1 tab can then exclude "
            "mothers and daughters."
        )
        self.track_divisions.toggled.connect(self._track_divisions_toggled)
        form.addRow("", self.track_divisions)
        form.addRow("Max gap", self.track_gap)
        form.addRow("Max distance", self.track_dist)
        form.addRow("Save to", self.track_out)
        layout.addLayout(form)
        self.track_run = QPushButton("Run tracking")
        self.track_run.clicked.connect(self._run_tracking)
        layout.addWidget(self.track_run)
        layout.addWidget(self._track_edit_box())
        layout.addStretch()
        page.setLayout(layout)
        return page

    def _track_edit_box(self) -> QGroupBox:
        group = QGroupBox("Correct tracks")
        inner = QVBoxLayout()
        self.track_edit = QCheckBox("Edit tracks")
        self.track_edit.toggled.connect(self._track_edit_toggled)
        inner.addWidget(self.track_edit)
        modes = QGridLayout()
        self.track_modes = QButtonGroup(self)
        self.track_modes.setExclusive(True)
        for i, (key, label) in enumerate(
            (
                ("frame", "This frame"),
                ("link", "Link"),
                ("break", "Break"),
                ("swap", "Swap"),
            )
        ):
            button = QRadioButton(label)
            button.setProperty("mode", key)
            self.track_modes.addButton(button)
            button.toggled.connect(self._on_track_mode_toggled)
            modes.addWidget(button, i // 2, i % 2)
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
        inner.addWidget(undo)
        group.setLayout(inner)
        self.track_modes.buttons()[0].setChecked(True)
        return group

    def _track_mode(self) -> str:
        button = self.track_modes.checkedButton()
        if button is None:
            return "frame"
        return str(button.property("mode"))

    def _on_track_mode_toggled(self, checked: bool) -> None:
        if checked:
            self._track_mode_changed()

    def _track_mode_changed(self) -> None:
        self._clear_track_pick()
        help_text = {
            "frame": (
                "Click the cell to change, then click a cell that already has "
                "the ID you want. Only this frame changes."
            ),
            "link": (
                "Click the cell with the right ID, then click the cell that "
                "should become it. That second ID changes from the second "
                "click through the end of the movie."
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
        if not on:
            self._clear_track_pick()
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
        if np.linalg.norm(end - start) > 3:
            return
        self._apply_track_click(event.position)

    def _apply_track_click(self, position) -> None:
        hit = self._cell_at_tracked(position)
        if hit is None:
            self.track_msg.setText("The Tracked layer is not loaded.")
            return
        frame, y, x, track_id = hit
        if track_id <= 0:
            if self._track_pick is None:
                self.track_msg.setText("Click a cell.")
            return
        mode = self._track_mode()
        if mode == "break":
            self._commit_track(break_track, (track_id, frame))
            return
        if self._track_pick is None:
            self._track_pick = (frame, y, x, track_id)
            self._show_track_pick(frame, y, x)
            if mode == "frame":
                scope = "The next click changes only this frame."
            else:
                scope = "The next click changes this frame and every later frame."
            self.track_msg.setText(f"Track {track_id} selected. {scope}")
            return
        src_frame, _y, _x, src_id = self._track_pick
        self._clear_track_pick()
        if mode == "frame":
            self._commit_track(set_id_on_frame, (src_frame, src_id, track_id))
        elif mode == "link":
            self._commit_track(link_tracks, (src_id, track_id, frame))
        else:
            self._commit_track(swap_tracks, (src_id, track_id, frame))

    def _commit_track(self, fn, args: tuple) -> None:
        layer = self._tracked_layer()
        if layer is None:
            self.track_msg.setText("The Tracked layer is not loaded.")
            return
        stack = self._writable_stack(layer)
        try:
            undos, message = fn(stack, *args)
        except ValueError as exc:
            self.track_msg.setText(str(exc))
            self.status.setText(str(exc))
            return
        self.track_msg.setText(message)
        self.status.setText(message)
        if not undos:
            return
        self._track_undo.append(undos)
        self._refresh_edited(layer, stack, undos)

    def _undo_track(self) -> None:
        layer = self._tracked_layer()
        if layer is None or not self._track_undo:
            self.track_msg.setText("Nothing to undo.")
            return
        undos: list[CellUndo] = self._track_undo.pop()
        stack = self._writable_stack(layer)
        restore_cells(stack, undos)
        self._refresh_edited(layer, stack, undos)
        self._clear_track_pick()
        self.track_msg.setText("Undid the last track edit.")
        self.status.setText("Undid the last track edit.")

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
        rgb = self._layer("Tracked RGB")
        if not isinstance(rgb, Image) or not undos:
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
        layer = self._layer("Tracked")
        return layer if isinstance(layer, Labels) else None

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

    def _show_track_pick(self, frame: int, y: int, x: int) -> None:
        point = np.array([[frame, y, x]], dtype=float)
        layer = self._pick_layer
        if layer is None or layer not in self.viewer.layers:
            self._pick_layer = self.viewer.add_points(
                point,
                name="Track selection",
                size=14,
                face_color="white",
                border_color="red",
                symbol="ring",
            )
            return
        layer.data = point
        layer.visible = True

    def _clear_track_pick(self) -> None:
        self._track_pick = None
        layer = self._pick_layer
        if layer is None or layer not in self.viewer.layers:
            self._pick_layer = None
            return
        width = int(layer.data.shape[1]) if np.asarray(layer.data).ndim == 2 else 3
        layer.data = np.zeros((0, width), dtype=float)
        layer.visible = False

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
        cal = self._calibration(intensity_layer, labels)
        wf = None
        out_dir = Path(out_text) if out_text else None
        if out_dir is None:
            wf = self._working_folder(intensity_layer, labels)
            if wf is not None:
                out_dir = wf.step_dir("tracking")
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
        )

        def work(progress):
            from celltrack.pipeline import track_arrays
            from pathlib import Path

            progress(1, "tracking")
            return track_arrays(
                masks,
                intensity,
                out_dir,
                stem=stem,
                verbose=True,
                **kwargs,
            )

        self._track_run = (
            wf,
            {k: v for k, v in kwargs.items() if k != "calibration"},
            movie_path,
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
        if self._track_run is not None and self._track_run[0] is not None:
            wf, params, movie_path = self._track_run
            wf.write_settings("tracking", params, movie=movie_path)
            saved = f" Saved to {wf.step_dir('tracking', create=False)}"
        self._track_run = None
        self._replace_labels("Tracked", result["tracked"])
        if result.get("rgb") is not None:
            self._replace_image("Tracked RGB", result["rgb"])
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
        self.status.setText(
            f"Tracking done. tracks={qc.get('n_tracks', track_result.n_tracks)}{extra}{saved}"
        )
        self._refresh_layers()

    def _t1_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout()
        form = QFormLayout()
        self.t1_labels = QComboBox()
        self.t1_rgb = QComboBox()
        self.t1_expand = self._spin(1, 20, 3)
        self.t1_bond = self._dspin(0.0, 50.0, 3.0, 0.5)
        self.t1_rings = self._spin(0, 5, 1)
        self.t1_filter = QComboBox()
        self.t1_filter.addItem("All (baseline)", "all")
        self.t1_filter.addItem("Persist 3", "persist_3")
        self.t1_filter.addItem("Persist 5", "persist_5")
        self.t1_filter.addItem("Persist 7", "persist_7")
        self.t1_filter.addItem("Flicker", "flicker")
        self.t1_filter.addItem("Aborted", "aborted")
        self.t1_filter.currentIndexChanged.connect(self._apply_t1_filter)
        self.t1_pre = self._spin(0, 40, 3)
        self.t1_post = self._spin(0, 40, 3)
        form.addRow("Labels", self.t1_labels)
        form.addRow("RGB", self.t1_rgb)
        form.addRow("Expand", self.t1_expand)
        form.addRow("Bond cutoff", self.t1_bond)
        form.addRow("Border rings", self.t1_rings)
        form.addRow("T1 filter", self.t1_filter)
        form.addRow("Overlay pre", self.t1_pre)
        form.addRow("Overlay post", self.t1_post)
        self.t1_exclude_div = QCheckBox("Exclude dividing cells")
        self.t1_exclude_div.setChecked(True)
        self.t1_exclude_div.setToolTip(
            "Leave mothers and daughters out of T1 detection. Uses the "
            "divisions from the last Tracking run (Divisions on) on the "
            "Tracked layer; manual track edits do not update it."
        )
        form.addRow("", self.t1_exclude_div)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.t1_run = QPushButton("Run T1")
        self.t1_run.clicked.connect(self._run_t1)
        row.addWidget(self.t1_run)
        layout.addLayout(row)
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
        kwargs = dict(
            expand=int(self.t1_expand.value()),
            bond_cutoff=float(self.t1_bond.value()),
            border_rings=int(self.t1_rings.value()),
            pre=int(self.t1_pre.value()),
            post=int(self.t1_post.value()),
            length_window=0,
        )
        if (
            self.t1_exclude_div.isChecked()
            and self._division_ids
            and labels.name == "Tracked"
        ):
            kwargs["exclude_ids"] = set(self._division_ids)

        def work(progress):
            from run_tm_t1 import run_t1_stack

            return run_t1_stack(stack, rgb=rgb, progress=progress, **kwargs)

        self._start(work, self._t1_done)

    def _t1_done(self, result: dict) -> None:
        self._t1_result = result
        self._autosave_t1(result)
        self._refresh_layers()
        self._apply_t1_filter()
        note = result.get("cellflow_note", "")
        if note:
            self.status.setText(f"{self.status.text()} {note}")

    def _apply_t1_filter(self) -> None:
        result = self._t1_result
        if not result:
            return
        from run_tm_t1 import fourway_counts, paint_t1_filter

        mode = str(self.t1_filter.currentData() or "all")
        overlay, dots, shown = paint_t1_filter(result, mode)
        self._replace_image("T1 overlay", overlay)
        self._replace_image("T1 four-ways", dots)
        total = len(result.get("events") or [])
        n_4v, n_no4 = fourway_counts(shown)
        self.status.setText(
            f"T1 filter {self.t1_filter.currentText()}: "
            f"showing {len(shown)} of {total}. "
            f"4-cell vertex {n_4v}, no 4-way {n_no4}."
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
        bar.addWidget(self.folder_label, 1)
        bar.addWidget(choose)
        bar.addWidget(load)
        return bar

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

    def _choose_folder(self) -> None:
        chosen = self._ask_folder()
        if chosen:
            root = self.project.set_manual(find_results_root(chosen))
            self._refresh_folder_label()
            if self.project.found():
                self._load_saved_results()
            else:
                self.status.setText(f"Results will be saved to {root}")

    def _cell_props_dir(self):
        return self.project.root / "cell_props" if self.project.root is not None else None

    def _autosave_segmentation(self) -> None:
        wf = self._working_folder(self._layer(self.seg_image.currentText()))
        if wf is None:
            self.status.setText(f"{self.status.text()} (not saved: no results folder chosen)")
            return
        self._save_segmentation()
        wf.write_settings(
            "segmentation",
            {"model": "cpsam", "batch_size": int(self.seg_batch.value()), "gpu": self.seg_gpu.isChecked()},
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
            pd.DataFrame(rows).to_csv(wf.step_dir("t1") / "t1_events.csv", index=False)
            wf.write_settings(
                "t1",
                {"expand": int(self.t1_expand.value()), "bond_cutoff": float(self.t1_bond.value()),
                 "border_rings": int(self.t1_rings.value())},
            )
            self.status.setText(f"{self.status.text()} Saved {len(rows)} T1 events to {wf.step_dir('t1', create=False)}")
        except Exception as exc:  # saving must never break the analysis
            self.status.setText(f"{self.status.text()} (T1 events not saved: {exc})")

    def _load_saved_results(self) -> None:
        wf = self._working_folder(self._layer(self.seg_image.currentText()), ask=False)
        found = wf.found() if wf is not None else {}
        if not found:
            self.status.setText("No saved results found in the results folder.")
            return
        import tifffile

        loaded = []
        movie = self.project.read_settings().get("movie")
        if movie and Path(movie).is_file() and Path(movie).resolve() not in {
            Path(p).resolve() for p in self._layer_paths(*self.viewer.layers) if p
        }:
            try:
                self.viewer.open(str(movie), plugin="napari")
            except Exception:
                self.viewer.add_image(tifffile.imread(movie), name=movie_stem(movie))
            loaded.append("movie")
        if "segmentation" in found:
            masks = tifffile.imread(found["segmentation"][0])
            self._replace_labels("Segmentation", masks)
            self._seg_base_masks = np.asarray(masks).copy()
            loaded.append("segmentation")
        if "tracking" in found:
            labels = found["tracking"][0]
            self._replace_labels("Tracked", tifffile.imread(labels))
            rgb = labels.with_name(labels.name.replace("_tracked_labels", "_tracked_rgb"))
            if rgb.is_file():
                self._replace_image("Tracked RGB", tifffile.imread(rgb))
            loaded.append("tracking")
        self._refresh_layers()
        self.status.setText(f"Loaded {', '.join(loaded)} from {wf.root}")

    def _replace_labels(self, name: str, data: np.ndarray) -> None:
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
