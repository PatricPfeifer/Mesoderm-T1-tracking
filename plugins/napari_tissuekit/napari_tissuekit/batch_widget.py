"""Batch tab: a folder of eggs (movies) x segmentation profiles (presets), run on the cluster.

Rows are eggs, columns the checked profiles. A cell shows the run that holds that
profile's result (from the egg's ``params.json`` files), so the table is also the
overview of what is done and what is still missing.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QBrush, QColor, QFont
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .core import batch as core_batch
from .core import seg_params

STATE_PATH = Path.home() / ".napari_tissuekit" / "batch.json"
CURRENT = "custom"  # the Segmentation tab's values when they are no saved preset (runs: custom, custom_2 ...)

# Cell colours (semi-transparent: readable on napari's dark and light themes).
COLORS = {
    "done": QColor(60, 170, 90, 90),
    "running": QColor(70, 130, 220, 100),
    "queued": QColor(220, 170, 40, 90),
    "failed": QColor(220, 70, 60, 100),
}


def _load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(data: dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass  # remembering the folder is a convenience only


def _fmt_values(values: dict | None) -> str:
    if not values:
        return "no params.json (made before runs recorded their settings)"
    return ", ".join(f"{k} {v}" for k, v in values.items())


class BatchPanel(QWidget):
    """The Batch tab. The host widget gives it the Segmentation tab's state and actions."""

    def __init__(
        self,
        current_values: Callable[[], dict],
        fill_holes: Callable[[], bool],
        open_egg: Callable[[core_batch.EggStatus, str | None], None],
        run_items: Callable[[list[core_batch.BatchItem], list], None],
        preview_items: Callable[[list[core_batch.BatchItem], list], None] | None = None,
        cancel_all: Callable[[], None] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._current_values = current_values
        self._fill_holes = fill_holes
        self._open_egg = open_egg
        self._run_items = run_items
        self._preview_items = preview_items
        self._cancel_all = cancel_all
        state = _load_state()
        self.folder: Path | None = Path(state["folder"]) if state.get("folder") else None
        self._checked_profiles: set[str] = set(state.get("profiles") or ["default"])
        self.eggs: list[core_batch.EggStatus] = []
        self._egg_checked: set[str] = set()
        self._profile_boxes: dict[str, QCheckBox] = {}
        self._profiles: dict[str, dict] = {}
        self.live: dict[tuple[str, str], str] = {}  # (egg, profile) -> queued / running / failed
        self._scan_error = ""
        self._remembered: dict[tuple[str, str], str] = {}

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)

        # folder
        folder_row = QHBoxLayout()
        self.folder_label = QLabel()
        self.folder_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        pick = QPushButton("Folder…")
        pick.setToolTip("The folder with the movies (.tif); every egg's results go next to it as <egg>_results")
        pick.clicked.connect(self._choose_folder)
        refresh = QPushButton("Refresh")
        refresh.setToolTip("Read the folder again (new movies, runs made elsewhere)")
        refresh.clicked.connect(self.refresh)
        folder_row.addWidget(self.folder_label, 1)
        folder_row.addWidget(pick)
        folder_row.addWidget(refresh)
        layout.addLayout(folder_row)

        # profiles
        self.profile_box = QGroupBox("Profiles (segmentation presets)")
        self.profile_box.setToolTip(
            "Each checked profile is a column. Make and save profiles in the Segmentation tab "
            "(Preset > Save). Profiles with the same Cellpose values share their runs."
        )
        self.profile_grid = QGridLayout()
        self.profile_grid.setContentsMargins(6, 4, 6, 4)
        self.profile_box.setLayout(self.profile_grid)
        layout.addWidget(self.profile_box)

        # the matrix
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(180)
        self.table.setTextElideMode(Qt.ElideMiddle)  # long egg names: start…end
        self.table.setWordWrap(False)
        self._fitting = False
        self._user_widths: dict[str, int] = {}  # column title -> width you dragged it to
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(36)
        header.sectionResized.connect(self._column_dragged)
        header.sectionDoubleClicked.connect(lambda _c: self._reset_widths())
        header.setToolTip("Drag a column edge to change its width; double-click a title to fit all columns again")
        self.table.viewport().installEventFilter(self)
        self.table.itemChanged.connect(self._item_changed)
        self.table.cellDoubleClicked.connect(self._double_clicked)
        self.table.setToolTip("Double-click an egg to open it; double-click a done cell to open the egg with that run")
        layout.addWidget(self.table, 1)

        # selection
        sel = QHBoxLayout()
        sel.addWidget(QLabel("Select:"))
        for text, tip, fn in (
            ("All", "Tick every egg", lambda: self._select(lambda e: True)),
            ("None", "Untick every egg", lambda: self._select(lambda e: False)),
            ("Missing", "Tick the eggs that lack at least one of the checked profiles", self._select_missing),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            sel.addWidget(b)
        sel.addStretch(1)
        layout.addLayout(sel)
        self.rerun = QCheckBox("Re-run pairs that are already done (as a new run, e.g. cellflow_2)")
        self.rerun.toggled.connect(self._update_summary)
        layout.addWidget(self.rerun)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: gray;")
        layout.addWidget(self.summary)

        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run on cluster")
        self.run_button.setToolTip("All ticked eggs x checked profiles, one after another in one GPU job")
        self.run_button.clicked.connect(self._run)
        self.preview_button = QPushButton("Preview…")
        self.preview_button.setToolTip("Show the items, the time limit and the Slurm script; nothing is sent")
        self.preview_button.clicked.connect(self._preview)
        self.open_button = QPushButton("Open egg")
        self.open_button.setToolTip("Open the selected egg in the viewer (movie and its active run)")
        self.open_button.clicked.connect(lambda: self._open_selected())
        buttons.addWidget(self.run_button, 1)
        buttons.addWidget(self.preview_button)
        buttons.addWidget(self.open_button)
        layout.addLayout(buttons)
        self.cancel_button = QPushButton("Cancel all jobs in cluster")
        self.cancel_button.setToolTip(
            "Log in and cancel every waiting and running job of your cluster user (scancel -u). "
            "Items a running batch already finished are kept."
        )
        self.cancel_button.clicked.connect(lambda: self._cancel_all() if self._cancel_all else None)
        self.cancel_button.setEnabled(cancel_all is not None)
        layout.addWidget(self.cancel_button)
        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        self.bar.hide()  # shown while a batch runs
        self.bar_text = QLabel()
        self.bar_text.setWordWrap(True)
        self.bar_text.hide()
        layout.addWidget(self.bar)
        layout.addWidget(self.bar_text)
        self.setLayout(layout)

        self.reload_profiles()
        self.refresh()

    # ---- folder and profiles -------------------------------------------------------
    def _choose_folder(self) -> None:
        start = str(self.folder) if self.folder else ""
        chosen = QFileDialog.getExistingDirectory(self, "Folder with the egg movies", start)
        if chosen:
            self.folder = Path(chosen)
            self._egg_checked.clear()
            self._remember()
            self.refresh()

    def _remember(self) -> None:
        _save_state({"folder": str(self.folder) if self.folder else "",
                     "profiles": sorted(self._checked_profiles)})

    def reload_profiles(self) -> None:
        """Checkboxes for every preset, plus the Segmentation tab's values if they are unsaved."""
        presets = seg_params.all_presets()
        current = self._current_values()
        profiles = dict(presets)
        if seg_params.matching_preset(current, presets) is None:
            profiles[CURRENT] = current
        else:
            self._checked_profiles.discard(CURRENT)
        self._profiles = profiles
        while self.profile_grid.count():
            w = self.profile_grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self._profile_boxes = {}
        for i, (name, values) in enumerate(profiles.items()):
            box = QCheckBox(f"{name} (unsaved)" if name == CURRENT else name)
            box.setChecked(name in self._checked_profiles)
            tip = _fmt_values(core_batch.quality_values(values))
            if name == CURRENT:
                tip = ("The Segmentation tab's current values, not saved as a preset (runs are named "
                       "'custom'). Save them as a preset there to give them a name: " + tip)
            box.setToolTip(tip)
            box.toggled.connect(lambda on, n=name: self._profile_toggled(n, on))
            self.profile_grid.addWidget(box, i // 3, i % 3)
            self._profile_boxes[name] = box

    def _profile_toggled(self, name: str, on: bool) -> None:
        (self._checked_profiles.add if on else self._checked_profiles.discard)(name)
        self._remember()
        self._fill_table()

    def checked_profiles(self) -> dict[str, dict]:
        return {n: v for n, v in self._profiles.items() if n in self._checked_profiles}

    # ---- table -----------------------------------------------------------------------
    def refresh(self) -> None:
        """Read the folder again and redraw."""
        self.reload_profiles()
        if self.folder is None:
            self.folder_label.setText("Folder: (none, press Folder…)")
            self.eggs = []
        else:
            self.folder_label.setText(f"Folder: {self.folder}")
            self.folder_label.setToolTip(str(self.folder))
            try:
                self.eggs = core_batch.scan_folder(self.folder)
                self._scan_error = ""
            except core_batch.BatchError as exc:
                self.eggs = []
                self._scan_error = str(exc)
        self._egg_checked &= {e.name for e in self.eggs}
        self._remembered = self._remembered_items()
        self._fill_table()

    @staticmethod
    def _remembered_items() -> dict[tuple[str, str], str]:
        """(egg, profile) of batch items still on the cluster -> job id (fetch them later)."""
        try:
            from .core.project import movie_stem
            from .core.remote import load_jobs

            return {(movie_stem(it["movie_name"]), it["profile"]): j.job_id
                    for j in load_jobs() for it in j.items if it.get("status") == "pending"}
        except Exception:
            return {}

    def _profile_columns(self) -> list[str]:
        return [n for n in self._profiles if n in self._checked_profiles]

    def _fill_table(self) -> None:
        profiles = self._profile_columns()
        headers = ["Egg", *profiles]
        self.table.blockSignals(True)
        try:
            self.table.clear()
            self.table.setColumnCount(len(headers))
            self.table.setHorizontalHeaderLabels(headers)
            self.table.setRowCount(len(self.eggs))
            for row, egg in enumerate(self.eggs):
                name = QTableWidgetItem(egg.name)
                name.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                name.setCheckState(Qt.Checked if egg.name in self._egg_checked else Qt.Unchecked)
                name.setToolTip(f"{egg.movie}\nresults: {egg.results}\nactive run: {egg.active_run or '-'}")
                self.table.setItem(row, 0, name)
                for col, profile in enumerate(profiles, 1):
                    self.table.setItem(row, col, self._cell(egg, profile))
        finally:
            self.table.blockSignals(False)
        self._fit_columns()
        self._update_summary()

    # ---- column widths: everything visible, every column draggable ---------------------
    def _fit_columns(self) -> None:
        """Columns you dragged keep their width; the others fit their content, and the Egg
        column gets the space that is left (long names are shortened in the middle)."""
        if self.table.columnCount() == 0:
            return
        self._fitting = True
        try:
            header = self.table.horizontalHeader()
            others = 0
            for col in range(1, self.table.columnCount()):
                name = self.table.horizontalHeaderItem(col).text()
                if name in self._user_widths:
                    self.table.setColumnWidth(col, self._user_widths[name])
                else:
                    self.table.resizeColumnToContents(col)
                    self.table.setColumnWidth(col, min(self.table.columnWidth(col), 160))
                others += self.table.columnWidth(col)
            if "Egg" in self._user_widths:
                egg = self._user_widths["Egg"]
            else:
                self.table.resizeColumnToContents(0)
                room = self.table.viewport().width() - others - 2
                egg = max(90, min(self.table.columnWidth(0), room))
            self.table.setColumnWidth(0, egg)
            header.setSectionResizeMode(QHeaderView.Interactive)
        finally:
            self._fitting = False

    def _column_dragged(self, col: int, _old: int, new: int) -> None:
        if self._fitting:
            return
        item = self.table.horizontalHeaderItem(col)
        if item is not None:
            self._user_widths[item.text()] = new

    def _reset_widths(self) -> None:
        self._user_widths.clear()
        self._fit_columns()

    def eventFilter(self, obj, event) -> bool:  # refit when the dock gets wider or narrower
        from qtpy.QtCore import QEvent

        if obj is self.table.viewport() and event.type() == QEvent.Resize and not self._fitting:
            self._fit_columns()
        return super().eventFilter(obj, event)

    def _cell(self, egg: core_batch.EggStatus, profile: str) -> QTableWidgetItem:
        live = self.live.get((egg.name, profile))
        done = egg.run_for(self._profiles[profile], profile)
        waiting = self._remembered.get((egg.name, profile))
        if live:
            text = "running…" if live == "running" else live
            text, color, tip = text, COLORS.get(live), f"{profile} on {egg.name}: {live}"
        elif waiting and done is None:
            text, color = "on cluster", COLORS["queued"]
            tip = (f"Submitted in batch job {waiting}, not downloaded yet: "
                   "Cluster settings > Fetch cluster results picks it up.")
        elif done is not None:
            star = " ★" if done.name == egg.active_run else ""
            name = "" if done.name == profile else f" {done.name}"  # name only when it differs
            text, color = f"✓{name}{star}", COLORS["done"]
            tip = (f"run '{done.name}'{' (active)' if star else ''}, made {done.time or '?'} "
                   f"from preset '{done.preset or '?'}'"
                   f"{'' if done.fill_holes is None else ', fill holes ' + ('on' if done.fill_holes else 'off')}\n"
                   f"{_fmt_values(done.values)}\nDouble-click: open {egg.name} with this run")
        else:
            text, color, tip = "–", None, f"{profile} not run on {egg.name} yet"
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        item.setToolTip(tip)
        item.setData(Qt.UserRole, done.name if done is not None else None)
        if color is not None:
            item.setBackground(QBrush(color))
        return item

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        name = self.eggs[item.row()].name
        (self._egg_checked.add if item.checkState() == Qt.Checked else self._egg_checked.discard)(name)
        self._update_summary()

    def _select(self, keep) -> None:
        self._egg_checked = {e.name for e in self.eggs if keep(e)}
        self._fill_table()

    def _select_missing(self) -> None:
        profiles = self.checked_profiles()
        self._select(lambda e: any(e.run_for(v) is None for v in profiles.values()))

    def checked_eggs(self) -> list[core_batch.EggStatus]:
        return [e for e in self.eggs if e.name in self._egg_checked]

    # ---- summary and run ----------------------------------------------------------------
    def plan(self):
        return core_batch.plan_items(self.checked_eggs(), self.checked_profiles(),
                                     rerun=self.rerun.isChecked(), fill_holes=self._fill_holes())

    def _update_summary(self, *_a) -> None:
        if self.folder is not None and not self.eggs:
            self.summary.setText(self._scan_error or f"No .tif movies in {self.folder}.")
            return
        eggs, profiles = self.checked_eggs(), self.checked_profiles()
        if not eggs or not profiles:
            self.summary.setText("Tick eggs (left column) and profiles (above) to build a batch.")
            return
        items, skipped = self.plan()
        n = len(items)
        text = (f"Will run {n} item{'s' * (n != 1)} ({len(eggs)} egg{'s' * (len(eggs) != 1)} × "
                f"{len(profiles)} profile{'s' * (len(profiles) != 1)}"
                + (f", {len(skipped)} skipped as done" if skipped else "") + ")")
        if n:
            try:
                from .core.remote import RemoteConfig, batch_time, partition_problem

                cfg = RemoteConfig.load()
                total = batch_time(cfg.time, n)
                text += f" · time limit {total} on {cfg.partition} · fill holes {'on' if self._fill_holes() else 'off'}"
                problem = partition_problem(cfg.partition, total)
                if problem:
                    text += f"\n⚠ {problem}"
            except Exception as exc:  # a bad time limit in the settings
                text += f" · (time limit: {exc})"
        self.summary.setText(text)

    def _planned(self):
        items, skipped = self.plan()
        if not items:
            self.summary.setText("Nothing to run: tick eggs and profiles (done pairs are skipped "
                                 "unless 'Re-run' is ticked).")
            return None
        return items, skipped

    def _run(self) -> None:
        planned = self._planned()
        if planned:
            self._run_items(*planned)

    def _preview(self) -> None:
        planned = self._planned()
        if planned and self._preview_items is not None:
            self._preview_items(*planned)

    # ---- a batch on the cluster --------------------------------------------------------------
    def set_live(self, states: dict | None, refresh_egg: str | None = None) -> None:
        """Cluster state per (egg, profile): "queued", "running", "failed"; None removes it.

        ``states=None`` clears all. ``refresh_egg``: read that egg's folder again (an item came in).
        """
        if states is None:
            self.live.clear()
        else:
            for key, value in states.items():
                if value is None:
                    self.live.pop(key, None)
                else:
                    self.live[key] = value
        if refresh_egg is not None:
            self.eggs = [core_batch.egg_status(e.movie) if e.name == refresh_egg else e for e in self.eggs]
        self._fill_table()

    def show_progress(self, percent: int | None, message: str) -> None:
        """The batch's progress bar under the buttons (``percent=None`` hides it)."""
        if percent is None:
            self.bar.hide()
            self.bar_text.hide()
            return
        self.bar.show()
        self.bar_text.show()
        if percent > 0:
            self.bar.setRange(0, 100)
            self.bar.setValue(percent)
        else:
            self.bar.setRange(0, 0)  # queued / uploading: busy animation
        self.bar_text.setText(message)

    # ---- opening an egg ------------------------------------------------------------------
    def _open_selected(self, run: str | None = None) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.eggs):
            ticked = self.checked_eggs()
            if len(ticked) != 1:
                self.summary.setText("Select an egg in the table first (click its row).")
                return
            egg = ticked[0]
        else:
            egg = self.eggs[row]
        self._open_egg(egg, run)

    def _double_clicked(self, row: int, col: int) -> None:
        item = self.table.item(row, col)
        run = item.data(Qt.UserRole) if item is not None and 0 < col <= len(self._profile_columns()) else None
        self.table.setCurrentCell(row, col)
        self._open_selected(run)


class DryRunDialog(QDialog):
    """What a batch would send to the cluster (step 1: nothing is sent yet)."""

    def __init__(self, items, skipped, cfg, parent=None):
        from .core.remote import batch_job_script, batch_time, partition_problem

        super().__init__(parent)
        self.setWindowTitle("Batch (dry run)")
        self.resize(760, 600)
        n = len(items)
        total = batch_time(cfg.time, n)
        lines = [f"{n} item(s) in one job on {cfg.partition}, time limit {total} "
                 f"({cfg.time} per item). Movies to upload: {len({it.movie for it in items})}.", ""]
        problem = partition_problem(cfg.partition, total)
        if problem:
            lines += [f"⚠ {problem}", ""]
        lines += [f"  {i:>2}. {it.egg} · {it.profile}  →  {it.egg}_results/segmentation/{it.run}/"
                  for i, it in enumerate(items, 1)]
        if skipped:
            lines += ["", "Skipped (already done):"]
            lines += [f"      {egg} · {profile}  (run '{run}')" for egg, profile, run in skipped]
        remote_dir = f"~/{cfg.remote_root}/batch_{time.strftime('%Y%m%d-%H%M%S')}"
        script = batch_job_script(cfg, remote_dir, items)
        lines += ["", "Slurm script it would submit:", "", script]
        text = QPlainTextEdit("\n".join(lines))
        text.setReadOnly(True)
        text.setLineWrapMode(QPlainTextEdit.NoWrap)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        text.setFont(mono)
        note = QLabel("Preview: nothing was sent to the cluster. 'Run on cluster' submits this.")
        note.setStyleSheet("color: gray;")
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout()
        layout.addWidget(note)
        layout.addWidget(text, 1)
        layout.addWidget(buttons)
        self.setLayout(layout)
