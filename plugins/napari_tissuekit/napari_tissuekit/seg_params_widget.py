"""Segmentation tab: preset choice, collapsible "Parameters" pane and run name.

The preset row sits outside the Parameters pane. Changing a value that no preset has
switches the preset box to "Custom"; "Save as preset..." stores the current values in
``~/.napari_tissuekit/segmentation_presets.json`` (see core/seg_params.py).
"""
from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QToolButton,
    QWidget,
)

from .core import seg_params

CUSTOM = "Custom"


class SegParamsPanel(QWidget):
    """Preset + Parameters + Run name. ``values()`` gives every Cellpose setting.

    The rows go into the caller's form with :meth:`add_rows`, so their labels line up
    with the rest of the section (this widget itself is only their owner, never shown)."""

    changed = Signal()

    def __init__(self, name_for: Callable[[str], str] | None = None, parent=None):
        super().__init__(parent)
        self._name_for = name_for  # base name -> next free run name in the results folder
        self._presets: dict[str, dict] = {}
        self._quiet = False

        # ---- preset row (outside the Parameters pane)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.preset = QComboBox()
        self.preset.setToolTip("Saved parameter sets. 'default' = Cellpose's own settings.")
        self.preset.currentTextChanged.connect(self._preset_chosen)
        self.save_btn = QPushButton("Save as preset...")
        self.save_btn.setToolTip("Store the current parameters under a name (for every movie)")
        self.save_btn.clicked.connect(self._save_preset)
        self.del_btn = QPushButton("Delete")
        self.del_btn.setToolTip("Delete this saved preset (built-in presets cannot be deleted)")
        self.del_btn.clicked.connect(self._delete_preset)
        row.addWidget(self.preset, 1)
        row.addWidget(self.save_btn)
        row.addWidget(self.del_btn)
        self._preset_row = row

        # ---- collapsible Parameters pane
        self.toggle = QToolButton()
        self.toggle.setText("Parameters")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(False)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.RightArrow)
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.toggle.toggled.connect(self._toggle)
        self.body = QWidget()
        form = QFormLayout()
        form.setContentsMargins(14, 0, 0, 4)
        form.setVerticalSpacing(4)
        self.fields: dict[str, QWidget] = {}
        for spec in seg_params.SPECS:
            if spec.kind is bool:
                w = QCheckBox()
                w.toggled.connect(self._edited)
            elif spec.kind is int:
                w = QSpinBox()
                w.setRange(int(spec.minimum), int(spec.maximum))
                w.setSingleStep(int(spec.step))
                w.valueChanged.connect(self._edited)
            else:
                w = QDoubleSpinBox()
                w.setDecimals(2)
                w.setRange(spec.minimum, spec.maximum)
                w.setSingleStep(spec.step)
                w.valueChanged.connect(self._edited)
            if spec.key == "diameter":
                w.setSpecialValueText("0 (auto)")
            w.setToolTip(spec.tip)
            self.fields[spec.key] = w
            form.addRow(spec.label, w)
        self.body.setLayout(form)
        self._body_form = form
        self.body.setVisible(False)

        # ---- run name
        self.run_name = QLineEdit()
        self.run_name.setToolTip(
            "Folder for this run: <results>/segmentation/<run name>/. Leave empty to use the preset "
            "name (cellflow, cellflow_2, ...). An existing run is never overwritten."
        )
        self.reload_presets("default")

    def add_rows(self, form: QFormLayout) -> None:
        """Preset, Run name and the collapsible Parameters pane as rows of ``form``."""
        form.addRow("Preset", self._preset_row)
        form.addRow("Run name", self.run_name)
        form.addRow(self.toggle)
        form.addRow(self.body)

    def add_body_row(self, label: str, widget: QWidget) -> None:
        """An extra row in the Parameters pane that is not part of a preset (e.g. Fill enclosed holes)."""
        self._body_form.addRow(label, widget)

    # ---- presets ---------------------------------------------------------------
    def reload_presets(self, select: str | None = None) -> None:
        self._presets = seg_params.all_presets()
        current = select or self.preset.currentText() or "default"
        self._quiet = True
        try:
            self.preset.clear()
            self.preset.addItems(list(self._presets))
        finally:
            self._quiet = False
        if current in self._presets:
            self.preset.setCurrentText(current)
            self._preset_chosen(current)
        else:
            self._sync_preset_box()

    def _preset_chosen(self, name: str) -> None:
        if self._quiet or name not in self._presets:
            return
        self.set_values(self._presets[name])

    def set_values(self, values: dict) -> None:
        self._quiet = True
        try:
            for key, value in seg_params.full_params(values).items():
                w = self.fields[key]
                if isinstance(w, QCheckBox):
                    w.setChecked(bool(value))
                else:
                    w.setValue(value)
        finally:
            self._quiet = False
        self._sync_preset_box()

    def _edited(self, *_a) -> None:
        if not self._quiet:
            self._sync_preset_box()

    def _sync_preset_box(self) -> None:
        """Show the preset these values belong to, or "Custom"."""
        name = seg_params.matching_preset(self.values(), self._presets)
        self._quiet = True
        try:
            idx = self.preset.findText(CUSTOM)
            if name is None:
                if idx < 0:
                    self.preset.addItem(CUSTOM)
                self.preset.setCurrentText(CUSTOM)
            else:
                if idx >= 0:
                    self.preset.removeItem(idx)
                self.preset.setCurrentText(name)
        finally:
            self._quiet = False
        self.del_btn.setEnabled(name is not None and name not in seg_params.BUILTIN_PRESETS)
        self.refresh_run_hint()
        self.changed.emit()

    def _save_preset(self) -> None:
        suggestion = "" if self.preset_name() in (CUSTOM, *seg_params.BUILTIN_PRESETS) else self.preset_name()
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:", text=suggestion)
        name = name.strip()
        if not ok or not name:
            return
        if name in seg_params.BUILTIN_PRESETS or name == CUSTOM:
            QMessageBox.warning(self, "Save preset", f"'{name}' is a built-in name; choose another.")
            return
        if name in self._presets and QMessageBox.question(
            self, "Save preset", f"Replace the preset '{name}'?"
        ) != QMessageBox.Yes:
            return
        seg_params.save_user_preset(name, self.values())
        self.reload_presets(name)

    def _delete_preset(self) -> None:
        name = self.preset_name()
        if name in seg_params.BUILTIN_PRESETS or name == CUSTOM:
            return
        if QMessageBox.question(self, "Delete preset", f"Delete the preset '{name}'?") != QMessageBox.Yes:
            return
        values = self.values()
        seg_params.delete_user_preset(name)
        self.reload_presets("default")
        self.set_values(values)  # keep the values on screen (now "Custom")

    # ---- Parameters pane -----------------------------------------------------------
    def _toggle(self, open_: bool) -> None:
        self.toggle.setArrowType(Qt.DownArrow if open_ else Qt.RightArrow)
        self.body.setVisible(open_)

    # ---- results ---------------------------------------------------------------------
    def values(self) -> dict:
        out = {}
        for key, w in self.fields.items():
            out[key] = w.isChecked() if isinstance(w, QCheckBox) else w.value()
        return seg_params.full_params(out)

    def preset_name(self) -> str:
        return self.preset.currentText() or "default"

    def run_base(self) -> str:
        """What the run folder is named after: the typed name, else the preset ("custom")."""
        typed = self.run_name.text().strip()
        if typed:
            return typed
        return "custom" if self.preset_name() == CUSTOM else self.preset_name()

    def refresh_run_hint(self) -> None:
        base = "custom" if self.preset_name() == CUSTOM else self.preset_name()
        try:
            hint = self._name_for(base) if self._name_for else base
        except Exception:
            hint = base
        self.run_name.setPlaceholderText(f"automatic: {hint}")

    def run_info(self) -> dict:
        """What is written to the run's params.json (and settings.json)."""
        return {"preset": self.preset_name(), "params": self.values()}
