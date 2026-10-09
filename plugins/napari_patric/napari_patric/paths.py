"""Locate the patric repo so the plugin can import the existing scripts."""
from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        marker = parent / "scripts" / "t1" / "run_tm_t1.py"
        if marker.is_file():
            return parent
    raise RuntimeError(
        "Could not find scripts/t1/run_tm_t1.py above the napari plugin."
    )


def ensure_scripts() -> Path:
    root = repo_root()
    for rel in (
        "scripts",
        "scripts/cellpose_seg",
        "scripts/cellpose_track",
        "scripts/t1",
    ):
        path = str(root / rel)
        if path not in sys.path:
            sys.path.insert(0, path)
    return root
