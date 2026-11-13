"""Cellpose segmentation parameters and presets (named, saved parameter sets).

Built-in presets live here; presets you save in the plugin go to
``~/.napari_tissuekit/segmentation_presets.json`` so they work for every movie.
A preset only needs to name the values it changes; :func:`full_params` fills in
the rest from :data:`DEFAULTS`. No napari / Qt in here.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .cell_size import MIN_CELL_AREA

USER_PRESETS_PATH = Path.home() / ".napari_tissuekit" / "segmentation_presets.json"


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    kind: type  # float, int or bool
    default: float | int | bool
    minimum: float = 0.0
    maximum: float = 0.0
    step: float = 1.0
    tip: str = ""


# Cellpose 4 (cpsam) eval settings the plugin exposes, with Cellpose's own defaults.
SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("diameter", "Diameter (px)", float, 0.0, 0.0, 1000.0, 1.0,
              "Expected cell diameter in pixels. 0 = automatic (cpsam does not need it)."),
    ParamSpec("flow_threshold", "Flow threshold", float, 0.4, 0.0, 3.0, 0.05,
              "Maximum flow error per cell. Higher keeps more (also worse-shaped) cells; lower is stricter."),
    ParamSpec("cellprob_threshold", "Cell prob. threshold", float, 0.0, -6.0, 6.0, 0.1,
              "Pixels above this cell probability count as cell. Lower finds more / larger cells; "
              "higher fewer / smaller ones."),
    ParamSpec("min_size", "Min size (px)", int, MIN_CELL_AREA, MIN_CELL_AREA, 100000, 1,
              f"Cells with fewer pixels are dropped (at least {MIN_CELL_AREA} px; with the movie's pixel size the minimum is also at least 4 um^2)."),
    ParamSpec("normalize", "Normalize intensity", bool, True,
              tip="Rescale each frame's intensity (1st-99th percentile) before segmentation."),
    # Speed / memory only, not the result: passed to eval as batch_size, not with the settings above.
    ParamSpec("batch_size", "Batch size", int, 8, 1, 256, 1,
              "256x256 px tiles per GPU pass (tiles within a frame). Higher is faster on a big GPU "
              "(the cluster: try 64) but needs more GPU memory; does not change the result."),
)
# Settings that only change speed: not Cellpose quality settings.
SPEED_KEYS = ("batch_size",)
DEFAULTS: dict = {s.key: s.default for s in SPECS}

BUILTIN_PRESETS: dict[str, dict] = {
    "default": {},
    # Settings of the older napariCellFlow segmentation (used for the Egg1 HPC runs).
    "cellflow": {"diameter": 0.0, "flow_threshold": 0.6, "cellprob_threshold": 0.3, "min_size": 25,
                 "normalize": True},
    # Cellpose defaults with big batches, for the cluster's large GPUs.
    "hpc": {"batch_size": 64},
}


def full_params(values: dict | None = None) -> dict:
    """Every parameter, ``values`` over the defaults (unknown keys dropped, types fixed)."""
    out = dict(DEFAULTS)
    for spec in SPECS:
        if values and values.get(spec.key) is not None:
            v = values[spec.key]
            out[spec.key] = bool(v) if spec.kind is bool else spec.kind(v)
    out["min_size"] = max(int(out["min_size"]), MIN_CELL_AREA)
    return out


def eval_kwargs(params: dict) -> dict:
    """Cellpose quality settings for ``CellposeModel.eval`` (diameter 0 = automatic).

    Without the batch size: :func:`batch_size` gives it, and it is passed on its own."""
    p = full_params(params)
    p["diameter"] = p["diameter"] or None
    for key in SPEED_KEYS:
        p.pop(key, None)
    return p


def batch_size(params: dict) -> int:
    return int(full_params(params)["batch_size"])


# ---- presets -------------------------------------------------------------------

def load_user_presets(path: Path | None = None) -> dict[str, dict]:
    try:
        data = json.loads(Path(path or USER_PRESETS_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): full_params(v) for k, v in data.items() if isinstance(v, dict) and k not in BUILTIN_PRESETS}


def all_presets(path: Path | None = None) -> dict[str, dict]:
    """Built-in presets first, then yours; every value filled in."""
    out = {name: full_params(p) for name, p in BUILTIN_PRESETS.items()}
    out.update(load_user_presets(path))
    return out


def save_user_preset(name: str, params: dict, path: Path | None = None) -> None:
    name = name.strip()
    if not name or name in BUILTIN_PRESETS:
        raise ValueError(f"{name!r} cannot be used as a preset name")
    path = Path(path or USER_PRESETS_PATH)
    presets = load_user_presets(path)
    presets[name] = full_params(params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(presets, indent=2) + "\n", encoding="utf-8")


def delete_user_preset(name: str, path: Path | None = None) -> None:
    path = Path(path or USER_PRESETS_PATH)
    presets = load_user_presets(path)
    if presets.pop(name, None) is not None:
        path.write_text(json.dumps(presets, indent=2) + "\n", encoding="utf-8")


def matching_preset(params: dict, presets: dict[str, dict]) -> str | None:
    """Name of the preset whose values equal ``params``, if any."""
    p = full_params(params)
    return next((name for name, values in presets.items() if full_params(values) == p), None)


def run_name(text: str) -> str:
    """A safe folder name for a segmentation run."""
    name = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text.strip()).strip("._")
    return name or "run"
