import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.segmentation import find_boundaries, watershed

from napari_tissuekit.core.outline_io import hand_to_outline
from napari_tissuekit.core.ta_bond_edit import apply_drawing_hand, hand_stack_from_outlines, hand_from_outlines, relabel_changed, thin_hand


def _tissue():
    rng = np.random.default_rng(0)
    pts = rng.integers(10, [190, 210], size=(30, 2))
    markers = np.zeros((200, 220), np.int32)
    markers[pts[:, 0], pts[:, 1]] = np.arange(1, 31)
    lab = watershed(ndi.distance_transform_edt(markers == 0), markers)
    lab[:8], lab[-8:], lab[:, :8], lab[:, -8:] = 0, 0, 0, 0
    return lab.astype(np.uint16)


def test_saved_outlines_are_the_starting_bonds():
    masks = _tissue()
    hand = thin_hand(masks)
    stroke = np.zeros(masks.shape, np.uint8)
    stroke[60:140, 100:103] = 1
    new = apply_drawing_hand(hand, np.zeros_like(stroke), stroke, min_seed_area=14, pad=128)
    labels, (ys, xs) = relabel_changed(masks, hand, new)
    edited = masks.copy()
    edited[ys, xs] = labels[ys, xs]
    saved = hand_to_outline(new)
    again = hand_from_outlines(edited, saved)
    assert again is not None and np.array_equal(again, new)  # nothing moves when a saved edit is loaded
    stack = hand_stack_from_outlines(edited[None], saved[None])
    assert np.array_equal(hand_to_outline(stack[0]), saved)


def test_other_outlines_fall_back_to_rebuilt_bonds():
    masks = _tissue()
    cellpose_style = find_boundaries(masks, mode="inner").astype(np.uint8)  # inside the cells, not the plugin's bonds
    assert hand_from_outlines(masks, cellpose_style) is None
    stack = hand_stack_from_outlines(masks[None], cellpose_style[None])
    assert np.array_equal(stack[0], thin_hand(masks))


def test_enclosed_gap_between_cells_gets_one_membrane():
    masks = _tissue()
    masks[60:120, 100:104] = 0  # a 4 px wide gap the segmentation left between cells, fully inside the tissue
    hand = thin_hand(masks)
    strip = hand[70:110, 100:104] == 255
    assert strip.mean() < 0.4  # the cells meet in the gap: a 1 px line, not an empty strip with a line on each side
    big = masks.copy()
    big[40:140, 90:130] = 0  # a large hole stays empty
    assert (thin_hand(big)[60:120, 100:120] == 255).mean() > 0.9
    sizes = np.bincount(label(thin_hand(masks), connectivity=1, background=255).ravel())[1:]
    assert sizes.min() >= 4  # closing the gap leaves no stray 1 px cells
