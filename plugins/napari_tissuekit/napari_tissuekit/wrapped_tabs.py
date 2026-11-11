"""Tab pages with a row of buttons that wraps, so every tab name is always shown in full.

A QTabWidget cuts long names ("Analysis cropping", "Export Cell Props") or hides tabs behind scroll arrows when the
napari dock is narrow. `WrappedTabs` shows one checkable button per page in a flow layout (as many per row as fit, the
rest on further rows) above a stack of the pages. It offers the part of the QTabWidget interface the plugin uses:
`addTab`, `currentChanged`, `currentIndex`, `setCurrentIndex`, `tabText`, `count`, `widget`, `setCornerWidget`.
"""
from __future__ import annotations

from qtpy.QtCore import QPoint, QRect, QSize, Qt, Signal
from qtpy.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLayout,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

BUTTON_STYLE = "QPushButton:checked { font-weight: bold; border: 2px solid #f5d67b; }"


class FlowLayout(QLayout):
    """Left-to-right layout that starts a new row when the next widget does not fit (Qt's flow layout example)."""

    def __init__(self, parent=None, spacing: int = 4):
        super().__init__(parent)
        self._items: list = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        try:
            return Qt.Orientations(0)  # PyQt5 / PySide2
        except (AttributeError, TypeError):
            return Qt.Orientation(0)  # Qt6

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, apply=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect: QRect, apply: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, row_h = area.x(), area.y(), 0
        gap = self.spacing()
        for item in self._items:
            hint = item.sizeHint()
            if x > area.x() and x + hint.width() > area.right() + 1:  # next row
                x, y, row_h = area.x(), y + row_h + gap, 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + gap
            row_h = max(row_h, hint.height())
        return y + row_h - rect.y() + m.bottom()


class WrappedTabs(QWidget):
    currentChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bar = QWidget()
        self._flow = FlowLayout(self._bar)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: list[QPushButton] = []
        self._stack = QStackedWidget()
        self._corner_box = QHBoxLayout()
        self._corner_box.setContentsMargins(0, 0, 0, 0)
        self._bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self._bar, 1, Qt.AlignTop)
        head.addLayout(self._corner_box)
        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(head)
        lay.addWidget(self._stack, 1)
        self.setLayout(lay)
        self._current = -1

    def addTab(self, widget: QWidget, text: str) -> int:
        index = self._stack.addWidget(widget)
        button = QPushButton(text)
        button.setCheckable(True)
        button.setFlat(False)
        button.setStyleSheet(BUTTON_STYLE)
        button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        button.setToolTip(text)
        self._group.addButton(button, index)
        self._flow.addWidget(button)
        self._buttons.append(button)
        button.clicked.connect(lambda _checked=False, i=index: self.setCurrentIndex(i))
        if self._current < 0:
            self.setCurrentIndex(index)
        return index

    def setCornerWidget(self, widget: QWidget, _corner=None) -> None:
        self._corner_box.addWidget(widget, 0, Qt.AlignTop | Qt.AlignRight)

    def count(self) -> int:
        return self._stack.count()

    def widget(self, index: int) -> QWidget:
        return self._stack.widget(index)

    def tabText(self, index: int) -> str:
        return self._buttons[index].text() if 0 <= index < len(self._buttons) else ""

    def currentIndex(self) -> int:
        return self._current

    def setCurrentIndex(self, index: int) -> None:
        if not 0 <= index < self._stack.count():
            return
        self._buttons[index].setChecked(True)
        if index == self._current:
            return
        self._current = index
        self._stack.setCurrentIndex(index)
        self.currentChanged.emit(index)
