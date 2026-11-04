"""Watershed used by the outline/bond editing tools (``ta_bond_edit``).

Ported from EPySeg / pyTA ``epyseg/ta/segmentation/neo_wshed.py`` (``wshed`` only), so the
plugin no longer needs EPySeg, numba or TensorFlow. The optional ``fix_scipy_wshed`` pass
and the demo code were left out; nothing here calls them.

EPySeg Copyright (c) 2020-2023, Benoit Aigouy. BSD 3-Clause licence: redistribution and
use in source and binary forms, with or without modification, are permitted provided that
this copyright notice, the licence conditions and the disclaimer are retained
(https://github.com/baigouy/EPySeg/blob/master/LICENSE).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure
from skimage.segmentation import watershed
from skimage.util import invert


def wshed(
    img,
    channel=None,
    weak_blur=None,
    strong_blur=None,
    seeds="mask",
    min_seed_area=None,
    is_white_bg=False,
    force_dual_pass=False,
):
    """Watershed segmentation that returns the 255-valued watershed lines (uint8).

    seeds: 'mask' (every connected 0-region of ``img``), a list of (row, col) points, or
    a label image of the same shape. Any other value picks local minima as seeds.
    min_seed_area: after the first pass, regions this small are merged away and the
    watershed is rerun (dual pass). weak_blur / strong_blur: Gaussian sigmas.
    """
    if min_seed_area is None and force_dual_pass:
        min_seed_area = 0

    if weak_blur is None and strong_blur is not None:
        weak_blur, strong_blur = strong_blur, None

    if (strong_blur is not None or weak_blur is not None) and isinstance(seeds, str):
        seeds = None

    if channel and len(img.shape) > 2:
        img = img[..., channel]

    if is_white_bg:
        img = invert(img)

    weak = filters.gaussian(img, sigma=weak_blur, preserve_range=True, mode="wrap") if weak_blur else img
    strong = (
        filters.gaussian(img, sigma=strong_blur, preserve_range=True, mode="wrap") if strong_blur else None
    )

    if isinstance(seeds, str) and seeds == "mask":
        markers = measure.label(weak, connectivity=1, background=255)
    elif isinstance(seeds, list):
        marker = np.zeros_like(weak, dtype=np.int32)
        pts = np.asarray(seeds)
        marker[pts[:, 0], pts[:, 1]] = 1
        markers = measure.label(marker, connectivity=1, background=0)
    elif isinstance(seeds, np.ndarray):
        if seeds.shape == weak.shape:
            markers = seeds
        else:
            marker = np.zeros_like(weak, dtype=np.int32)
            marker[seeds[:, 0], seeds[:, 1]] = 1
            markers = measure.label(marker, connectivity=1, background=0)
    else:
        ref = strong if strong is not None else weak
        extreme = ndi.maximum_filter if is_white_bg else ndi.minimum_filter
        markers = measure.label(extreme(ref, size=3) == ref)

    labels_ws = watershed(weak, markers=markers, watershed_line=True)
    lines = np.zeros_like(labels_ws)
    lines[labels_ws == 0] = 255
    labels_ws = lines

    if (min_seed_area is not None and min_seed_area > 0) or force_dual_pass:
        wshed_mask = np.copy(labels_ws)
        regions = measure.label(wshed_mask, connectivity=1, background=255)
        ids, count = np.unique(regions, return_counts=True)
        small = ids[count <= min_seed_area]
        if small.size == 0 and not force_dual_pass:
            final_mask = wshed_mask
        else:
            for cell in small:
                wshed_mask[regions == cell] = 255
            wshed_mask = watershed(wshed_mask, markers=None, watershed_line=True)
            final_mask = np.zeros_like(wshed_mask)
            final_mask[wshed_mask == 0] = 255
    else:
        final_mask = labels_ws

    if final_mask.max() == final_mask.min():
        return np.zeros_like(final_mask, dtype=np.uint8)
    return final_mask.astype(np.uint8)
