"""The plugin must not depend on vendor/ (napariCellFlow, EPySeg) any more."""
import ast
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "plugins" / "napari_tissuekit" / "napari_tissuekit"
NEO_WSHED = ROOT / "vendor" / "EPySeg" / "epyseg" / "ta" / "segmentation" / "neo_wshed.py"
BANNED = ("napariCellFlow", "epyseg", "numba")


def test_package_has_no_vendor_imports():
    for path in PKG.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0] not in BANNED, f"{path.name} imports {name}"
        assert "vendor/EPySeg" not in path.read_text().replace('"source": "vendor/tissue_miner', ""), path.name


def _blocked(monkeypatch):
    for name in BANNED:
        monkeypatch.setitem(sys.modules, name, None)  # import raises ImportError


def _cells(h=40, w=40):
    """Hand-correction style image: 255 bonds, 0 inside cells (2x2 grid of cells)."""
    hand = np.zeros((h, w), np.uint8)
    hand[h // 2, :] = 255
    hand[:, w // 2] = 255
    hand[[0, -1], :] = 255
    hand[:, [0, -1]] = 255
    return hand


def test_wshed_runs_with_vendor_packages_blocked(monkeypatch):
    _blocked(monkeypatch)
    from napari_tissuekit.core._wshed import wshed

    out = wshed(_cells(), seeds="mask")
    assert out.dtype == np.uint8 and out.shape == (40, 40)
    assert out[18:23, :].any() and out[:, 18:23].any()  # lines between the four cells
    assert out[10, 10] == 0 and out[30, 30] == 0  # cell interiors stay empty


def test_bond_edit_thin_hand_with_vendor_packages_blocked(monkeypatch):
    _blocked(monkeypatch)
    from napari_tissuekit.core.ta_bond_edit import thin_hand

    labels = np.zeros((30, 30), np.int32)
    labels[2:14, 2:28] = 1
    labels[16:28, 2:28] = 2
    thin = thin_hand(labels)
    assert thin.shape == labels.shape and thin.dtype == np.uint8
    assert thin[15, 10] == 255  # a bond between the two cells


def _load_original_wshed():
    """Import EPySeg's neo_wshed.py as is, with its heavy imports stubbed."""
    stubs = {}
    img = types.ModuleType("epyseg.img")
    img.Img = type("Img", (), {})
    log = types.ModuleType("epyseg.tools.logger")
    log.TA_logger = lambda *a, **k: None
    numba = types.ModuleType("numba")
    numba.njit = numba.jit = lambda f=None, **k: (f if callable(f) else (lambda g: g))
    for name, mod in (("epyseg", types.ModuleType("epyseg")), ("epyseg.tools", types.ModuleType("epyseg.tools")),
                      ("epyseg.img", img), ("epyseg.tools.logger", log), ("numba", numba)):
        stubs[name] = sys.modules.get(name)
        sys.modules[name] = mod
    try:
        spec = importlib.util.spec_from_file_location("neo_wshed_orig", NEO_WSHED)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.wshed
    finally:
        for name, old in stubs.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


@pytest.mark.skipif(not NEO_WSHED.exists(), reason="vendor/EPySeg not present")
def test_wshed_matches_epyseg_original():
    from napari_tissuekit.core._wshed import wshed

    orig = _load_original_wshed()
    rng = np.random.default_rng(0)
    bonds = _cells(60, 60)
    noisy = (rng.random((60, 60)) * 255).astype(np.uint8)
    seeds_img = np.zeros((60, 60), np.int32)
    seeds_img[10, 10], seeds_img[45, 45] = 1, 2
    cases = [
        (bonds, dict(seeds="mask")),
        (bonds, dict(seeds="mask", min_seed_area=5)),
        (bonds, dict(seeds="mask", force_dual_pass=True)),
        (bonds, dict(seeds=[[10, 10], [45, 45]])),
        (bonds, dict(seeds=seeds_img, weak_blur=1.0)),
        (noisy, dict(seeds=None, weak_blur=1.0)),
        (noisy, dict(seeds=None, strong_blur=2.0, weak_blur=1.0, is_white_bg=True)),
        (np.zeros((20, 20), np.uint8), dict(seeds="mask")),
    ]
    for img, kw in cases:
        a, b = orig(img.copy(), **kw), wshed(img.copy(), **kw)
        assert a.dtype == b.dtype and np.array_equal(a, b), kw


def test_correction_managers_edit_a_labels_layer(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    _blocked(monkeypatch)
    from napari.components import ViewerModel
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])  # noqa: F841
    from napari_tissuekit.correction import CellCorrectionWidget, DataManager, VisualizationManager

    viewer = ViewerModel()
    masks = np.zeros((3, 20, 20), np.int32)
    masks[:, 2:10, 2:10] = 1
    dm, vis = DataManager(), VisualizationManager(viewer, DataManager())
    vis.data_manager = dm
    dm.initialize_stack(3)
    dm.segmentation_data = masks
    assert dm.segmentation_data.shape == (3, 20, 20)

    vis.update_tracking_visualization(masks)
    assert vis.tracking_layer is not None and vis.tracking_layer in viewer.layers
    frame = masks[1].copy()
    frame[12:18, 12:18] = 2
    vis.update_tracking_visualization((frame, 1))
    assert int(vis.tracking_layer.data[1].max()) == 2 and int(vis.tracking_layer.data[0].max()) == 1
    viewer.layers.remove(vis.tracking_layer)
    assert vis.tracking_layer is None

    widget = CellCorrectionWidget(viewer, dm, vis)
    widget.set_masks_layer(masks)
    assert widget.masks_layer is not None
