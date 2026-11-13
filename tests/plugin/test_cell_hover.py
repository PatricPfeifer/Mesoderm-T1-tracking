"""Hover and selection in the Cell properties widget, with the correction tool active.

Regression: the correction widget renamed every newly added Labels layer to "Segmentation", so the
'Hovered cell' layer became 'Segmentation [N]' right after it was added -> KeyError
"'Hovered cell' is not in list", and picked cells never showed up in the 'Selected cells' layer.
"""
import numpy as np
import pytest

napari = pytest.importorskip("napari")

from napari_tissuekit.cell_properties import HOVER_LAYER, SELECTED_LAYER, CellPropertiesWidget  # noqa: E402
from napari_tissuekit.correction import CellCorrectionWidget, DataManager, VisualizationManager  # noqa: E402


@pytest.fixture
def setup():
    viewer = napari.Viewer(show=False)
    stack = np.zeros((3, 40, 40), np.uint16)
    stack[:, 5:15, 5:15] = 1
    stack[:, 20:30, 20:30] = 2
    viewer.add_image(np.random.rand(3, 40, 40), name="raw")
    viewer.add_labels(stack, name="Tracked")
    dm = DataManager()
    vis = VisualizationManager(viewer, dm)
    corr = CellCorrectionWidget(viewer, dm, vis)  # hooks layers.events.inserted (the renamer)
    corr.set_masks_layer(stack.astype(np.int32))
    masks_before = corr.masks_layer
    w = CellPropertiesWidget(viewer)
    w.labels_combo.setCurrentText("Tracked")
    yield viewer, w, corr, masks_before
    viewer.close()


def test_hover_layer_keeps_its_name(setup):
    viewer, w, corr, masks_before = setup
    viewer.dims.set_current_step(0, 1)
    w._hover_id = 2
    w._update_hover(1)
    assert HOVER_LAYER in viewer.layers
    assert viewer.layers[-1].name == HOVER_LAYER
    assert viewer.layers[HOVER_LAYER].data.sum() == 100
    assert corr.masks_layer is masks_before  # not hijacked as the masks layer
    items = [w.labels_combo.itemText(i) for i in range(w.labels_combo.count())]
    assert HOVER_LAYER not in items

    viewer.dims.set_current_step(0, 2)  # a real frame change clears it
    assert HOVER_LAYER not in viewer.layers


def test_selected_cells_go_to_selected_layer(setup):
    viewer, w, corr, masks_before = setup
    w._selected.append(2)
    w._changed()
    assert SELECTED_LAYER in viewer.layers
    sel = np.asarray(viewer.layers[SELECTED_LAYER].data)
    assert set(np.unique(sel)) - {0} == {2}
    assert corr.masks_layer is masks_before
    assert not any(ly.name.startswith("Segmentation") and ly is not masks_before for ly in viewer.layers)
