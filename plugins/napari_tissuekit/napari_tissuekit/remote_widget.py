"""Qt side of "Run on cluster GPU": settings dialog, login dialog, background runner."""
from __future__ import annotations

from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
)

from .core.remote import RemoteConfig

_FIELDS = (
    # (attribute, label, tooltip)
    ("user", "User name", "Your bwUniCluster login, e.g. ho_abcd123"),
    ("host", "Host", "Cluster login host"),
    ("remote_root", "Cluster folder", "Where movies and results go (relative = inside your cluster home)"),
    ("conda_sh", "conda.sh", "Path to conda.sh on the cluster"),
    ("conda_env", "Conda env", "Env on the cluster with napari-tissuekit installed (`tissuekit` on PATH)"),
    ("partition", "Partition", "Slurm GPU queue: dev_gpu_h100 (starts fast, max 30 min) or gpu_h100 (up to 72 h)"),
    ("gres", "GPUs (--gres)", "e.g. gpu:1"),
    ("mem", "Memory", "e.g. 32G"),
    ("time", "Time limit per item", "hh:mm:ss for one movie with one profile. 5 min is plenty for normal movies. "
     "A batch job gets this times its number of items; dev queues allow 00:30:00 at most (use gpu_h100 for longer batches)"),
    ("setup_partition", "Setup partition", "CPU partition for the one-time env install, e.g. cpu, dev_cpu"),
    ("setup_time", "Setup time limit", "hh:mm:ss for the one-time env install"),
)


class RemoteSettingsDialog(QDialog):
    def __init__(self, cfg: RemoteConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster settings")
        self.cfg = cfg
        form = QFormLayout()
        self._edits: dict[str, QLineEdit] = {}
        for attr, label, tip in _FIELDS:
            edit = QLineEdit(str(getattr(cfg, attr)))
            edit.setToolTip(tip)
            form.addRow(label, edit)
            self._edits[attr] = edit
        self._cpus = self._spin(1, 128, cfg.cpus)
        self._batch = self._spin(1, 64, cfg.batch_size)
        self._poll = self._spin(5, 600, cfg.poll_seconds)
        form.addRow("CPUs", self._cpus)
        form.addRow("Batch size", self._batch)
        form.addRow("Check every (s)", self._poll)
        self._delete = QCheckBox("Delete the cluster folder after download")
        self._delete.setChecked(cfg.delete_remote_after)
        form.addRow("", self._delete)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    @staticmethod
    def _spin(lo, hi, value) -> QSpinBox:
        box = QSpinBox()
        box.setRange(lo, hi)
        box.setValue(int(value))
        return box

    def _save(self) -> None:
        for attr, edit in self._edits.items():
            setattr(self.cfg, attr, edit.text().strip())
        self.cfg.cpus = self._cpus.value()
        self.cfg.batch_size = self._batch.value()
        self.cfg.poll_seconds = self._poll.value()
        self.cfg.delete_remote_after = self._delete.isChecked()
        self.cfg.save()
        self.accept()


class LoginDialog(QDialog):
    """Password + OTP. Nothing typed here is stored."""

    def __init__(self, cfg: RemoteConfig, purpose: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster login")
        info = QLabel(f"{purpose}\n\nLog in as {cfg.user}@{cfg.host}.")
        info.setWordWrap(True)
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.Password)
        self.otp = QLineEdit()
        self.otp.setEchoMode(QLineEdit.Password)
        self.otp.setPlaceholderText("6 digits from your token app")
        form = QFormLayout()
        form.addRow("Password", self.password)
        form.addRow("OTP", self.otp)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Log in and start")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout()
        layout.addWidget(info)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)
        self.password.setFocus()

    def credentials(self) -> tuple[str, str] | None:
        """(password, otp) or None if cancelled; the fields are cleared either way."""
        ok = self.exec_() == QDialog.Accepted
        creds = (self.password.text(), self.otp.text().strip())
        self.password.clear()
        self.otp.clear()
        return creds if ok and all(creds) else None


class RemoteLogDialog(QDialog):
    """Live view: the desktop's steps (with times) and the newest lines of the cluster job log."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster job log")
        self.resize(720, 560)
        self.events = QPlainTextEdit()
        self.events.setReadOnly(True)
        self.events.setMaximumBlockCount(500)
        self.cluster = QPlainTextEdit()
        self.cluster.setReadOnly(True)
        self.cluster.setLineWrapMode(QPlainTextEdit.NoWrap)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self.cluster.setFont(mono)
        self.cluster.setPlainText("(the job log appears here once the job runs; it refreshes at every check)")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("<b>This computer</b>"))
        layout.addWidget(self.events, 1)
        layout.addWidget(QLabel("<b>Cluster job log</b> (newest lines)"))
        layout.addWidget(self.cluster, 3)
        self.setLayout(layout)

    def add_event(self, line: str) -> None:
        self.events.appendPlainText(line)

    def set_cluster_log(self, text: str) -> None:
        bar = self.cluster.verticalScrollBar()
        at_end = bar.value() >= bar.maximum() - 2
        self.cluster.setPlainText(text.rstrip())
        if at_end:  # follow the log unless the user scrolled up to read
            bar.setValue(bar.maximum())


class _RemoteJob(QObject):
    progress = Signal(int, str)
    log = Signal(str)
    finished = Signal(object)
    failed = Signal(object)  # the exception itself, so the caller can react to its type

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            self.finished.emit(self._fn(self.progress.emit, self.log.emit))
        except Exception as exc:  # reported to the widget, never raised in the thread
            import traceback

            traceback.print_exc()
            self.failed.emit(exc)


class RemoteRunner(QObject):
    """Runs one cluster task in its own thread, independent of the widget's local jobs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._job: _RemoteJob | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self, fn, on_progress, on_done, on_failed, on_log=None) -> bool:
        """Run ``fn(progress, log)`` in a thread; the callbacks run in the GUI thread."""
        if self.busy:
            return False
        self._job = _RemoteJob(fn)
        self._thread = QThread(self)
        self._job.moveToThread(self._thread)
        self._thread.started.connect(self._job.run)
        self._job.progress.connect(on_progress)
        if on_log is not None:
            self._job.log.connect(on_log)
        self._job.finished.connect(on_done)
        self._job.failed.connect(on_failed)
        self._job.finished.connect(self._thread.quit)
        self._job.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._clear)
        self._thread.start()
        return True

    def _clear(self) -> None:
        self._thread = None
        self._job = None
