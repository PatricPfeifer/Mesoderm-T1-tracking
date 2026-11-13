"""Server segmentation writes the plugin's default outline file (*_cell_outlines.tif)."""
import numpy as np

from napari_tissuekit.core.outline_io import hand_stack_to_outline
from napari_tissuekit.core.segmentation import cell_outline_stack
from napari_tissuekit.core.ta_bond_edit import thin_hand_stack


def _masks():
    masks = np.zeros((2, 30, 30), np.uint16)
    masks[:, 2:14, 2:28] = 1
    masks[:, 16:28, 2:28] = 2
    return masks


def test_cell_outlines_are_255_on_black_and_match_the_plugin_path():
    masks = _masks()
    out = cell_outline_stack(masks)
    assert out.shape == masks.shape and out.dtype == np.uint8
    assert set(np.unique(out)) == {0, 255}
    plugin = (hand_stack_to_outline(thin_hand_stack(masks)) > 0).astype(np.uint8) * 255
    assert np.array_equal(out, plugin)
    assert out[0, 15, 10] == 255  # bond between the two cells
    assert out[0, 8, 10] == 0  # cell interior stays empty


def test_napari_and_cluster_share_one_outline_step():
    from napari_tissuekit.core.ta_bond_edit import hand_from_outlines, make_bonds

    masks = _masks()
    hand, outlines = make_bonds(masks)  # what a napari segmentation hands to the editor
    assert np.array_equal((outlines > 0).astype(np.uint8) * 255, cell_outline_stack(masks))  # the cluster's file
    for t in range(masks.shape[0]):  # loading the cluster's file gives the same bonds back (no line moves)
        assert np.array_equal(hand_from_outlines(masks[t], outlines[t]), hand[t])


def test_progress_callback_is_called():
    seen = []
    cell_outline_stack(_masks(), lambda pct, msg: seen.append(pct))
    assert seen and seen[-1] == 100
