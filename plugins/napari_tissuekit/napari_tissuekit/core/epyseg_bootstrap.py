"""Make vendored EPySeg ``wshed`` importable without full pip install.

``neo_wshed`` only needs ``Img`` at import time (unused by ``wshed()`` itself).
We stub ``epyseg.img`` and ``numba`` so cellflow can call the real algorithm.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def _vendor_epyseg_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "vendor" / "EPySeg"
        if (cand / "epyseg").is_dir():
            return cand
    raise RuntimeError("Could not find vendor/EPySeg above cellpose_seg.")


def _ensure_numba_shim() -> None:
    try:
        import numba  # noqa: F401
        return
    except ImportError:
        pass
    if "numba" in sys.modules and getattr(sys.modules["numba"], "_tissuekit_stub", False):
        return
    stub = types.ModuleType("numba")
    stub._tissuekit_stub = True

    def njit(*args, **_kwargs):
        if args and callable(args[0]):
            return args[0]

        def deco(fn):
            return fn

        return deco

    stub.njit = njit
    stub.jit = njit
    sys.modules["numba"] = stub


def _ensure_img_stub() -> None:
    """Avoid importing real ``epyseg.img`` (pulls czifile, read_lif, …)."""
    existing = sys.modules.get("epyseg.img")
    if existing is not None and getattr(existing, "_tissuekit_stub", False):
        return
    # Drop a failed/partial real import if present.
    if existing is not None and not getattr(existing, "_tissuekit_stub", False):
        # Keep real module if it already loaded successfully.
        if hasattr(existing, "Img"):
            return

    stub = types.ModuleType("epyseg.img")
    stub._tissuekit_stub = True
    stub.Img = type("Img", (), {})
    sys.modules["epyseg.img"] = stub


def ensure_epyseg_on_path() -> Path:
    root = _vendor_epyseg_root()
    path = str(root)
    if path not in sys.path:
        sys.path.insert(0, path)
    _ensure_numba_shim()
    return root


def import_wshed():
    """Return EPySeg ``neo_wshed.wshed`` after bootstrap."""
    ensure_epyseg_on_path()
    _ensure_img_stub()
    # Import package first so stub attaches cleanly.
    import epyseg  # noqa: F401

    if getattr(sys.modules.get("epyseg.img"), "_tissuekit_stub", False):
        epyseg.img = sys.modules["epyseg.img"]  # type: ignore[attr-defined]

    # Fresh import of neo_wshed if a previous attempt failed mid-import.
    name = "epyseg.ta.segmentation.neo_wshed"
    if name in sys.modules and not hasattr(sys.modules[name], "wshed"):
        del sys.modules[name]

    mod = importlib.import_module(name)
    return mod.wshed
