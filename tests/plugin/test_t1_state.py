"""T1 state in the plugin widget: cleared when the tracking is replaced, flagged after track edits, optional event selection."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def _widget():
    from napari.components import ViewerModel
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])  # noqa: F841
    from napari_tissuekit._widget import TissueKitWidget

    w = TissueKitWidget(ViewerModel())
    yield w
    w.close()
    w.deleteLater()
    app.processEvents()


@pytest.fixture()
def widget(_widget):
    """One widget for the module (a fresh Qt widget per test crashes the offscreen platform at exit); T1 state reset each time."""
    _widget._invalidate_t1()
    _widget._tracked_run = None
    return _widget


def fake_result(n_events=4, tracked_run=None):
    stack = np.zeros((6, 20, 20), np.uint32)
    stack[:, 2:10, 2:10] = 1
    events = [dict(quartet=frozenset((1, 2, 3, 4)), A=1, B=2, C=3, D=4, t_start=1, t_end=2, t_first_swap=1 + i % 2, type="completed",
                   y=5.0, x=5.0, pair_before=frozenset((1, 4)), pair_after=frozenset((2, 3))) for i in range(n_events)]
    return dict(events=events, stack=stack, pre=3, post=3, expand=3, four_xy=[{} for _ in range(6)], tracked_run=tracked_run, stale=False)


def show_t1(widget, result):
    widget._t1_result = result
    widget._t1_shown = list(result["events"])
    widget._fill_t1_selector(result)
    widget.viewer.add_image(np.zeros((6, 20, 20, 3), np.uint8), name="T1 overlay", rgb=True)
    widget.viewer.add_image(np.zeros((6, 20, 20, 3), np.uint8), name="T1 four-ways", rgb=True)


def test_new_tracks_clear_the_t1_state(widget):
    show_t1(widget, fake_result())
    assert widget.t1_select_list.count() == 4
    widget._replace_labels("Tracked", np.zeros((6, 20, 20), np.uint32))
    assert widget._t1_result is None and widget._t1_shown == []
    assert "T1 overlay" not in widget.viewer.layers and "T1 four-ways" not in widget.viewer.layers
    assert widget.t1_select_list.count() == 0


def test_track_edits_and_other_tracking_make_the_result_stale(widget):
    show_t1(widget, fake_result())
    assert not widget._t1_stale()
    widget._mark_t1_stale()
    assert widget._t1_stale()
    widget._t1_result["stale"] = False
    widget._tracked_run = "other_run"  # tracking of another run in the viewer
    assert widget._t1_stale()
    widget.t1_click_preview.setChecked(True)
    assert widget._t1_click_active() is False


def test_selection_is_optional_and_picks_the_ticked_events(widget):
    show_t1(widget, fake_result())
    assert widget._t1_selection(4) is None  # selector off: all events
    widget.t1_select_box.setChecked(True)
    assert widget._t1_selection(4) is None  # everything ticked: all events
    widget._t1_select_set_all(False)
    for row in (0, 2):
        widget.t1_select_list.item(row).setCheckState(__import__("qtpy.QtCore", fromlist=["Qt"]).Qt.Checked)
    assert widget._t1_selection(4) == [0, 2]
    assert widget._t1_selection(5) is None  # list no longer matches the events: save all, never a wrong subset
    assert "2 of 4" in widget.t1_select_info.text()
