"""Working folder for one movie: where every pipeline step saves its results.

Layout (next to the input movie by default)::

    <movie_dir>/<movie_stem>_results/
        segmentation/   masks + outlines
        tracking/       tracked labels, RGB, tracks/links/events CSVs, qc.json
        t1/             T1 events
        cell_props/     exports and selections from "Export Cell Props"
        settings.json   parameters and time of the last run of every step

Nothing in here depends on napari or Qt, so it can be tested on its own.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

STEPS = ("segmentation", "tracking", "t1", "cell_props")

# Naming contract with the pipeline scripts (scripts/, HPC runs). Anything a script
# writes must be matched here, or "Load results" will not see it (tests/plugin/test_project.py
# checks this against scripts/output_names.py). Longest suffix first.
SEG_MASK_SUFFIXES = ("_cp_masks.tif", "_cf_masks.tif")
TRACK_LABEL_SUFFIXES = ("_cf_tracked_labels.tif", "_tracked_labels.tif")
T1_PATTERN = "*.csv"
# Folder names a step may live in: the plugin's own first, then HPC/script layouts.
STEP_FOLDERS = {
    "segmentation": ("segmentation", "cellflow_seg", "cellpose_seg"),
    "tracking": ("tracking", "cellpose_track", "cellflow_track"),
    "t1": ("t1",),
}
# Per-movie folders written by scripts/output_names.step_dir (``<stem>_cp_seg`` ...).
LEGACY_FOLDER_GLOBS = {
    "segmentation": ("*_cp_seg", "*_cf_seg"),
    "tracking": ("*_cp_track", "*_cf_track"),
}
MOVIE_EXTENSIONS = (".ome.tif", ".ome.tiff", ".tif", ".tiff", ".czi", ".nd2", ".lif")


def movie_stem(path: str | Path) -> str:
    """File name without its image extension (handles .ome.tif / .tiff)."""
    name = Path(path).name
    for ext in (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".czi", ".nd2", ".lif"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return Path(path).stem


def results_dir_for(movie_path: str | Path) -> Path:
    """Default working folder: ``<movie_dir>/<movie_stem>_results``."""
    p = Path(movie_path)
    return p.parent / f"{movie_stem(p)}_results"


def find_results_root(path: str | Path) -> Path:
    """The results folder itself, or the single ``*_results`` folder inside ``path``."""
    p = Path(path)
    if (
        (p / "settings.json").is_file()
        or any((p / s).is_dir() for s in STEPS)
        or any(_step_dirs(p, step) for step in STEP_FOLDERS)
    ):
        return p
    kids = [k for k in p.glob("*_results") if k.is_dir()]
    return kids[0] if len(kids) == 1 else p


def _step_dirs(root: Path, step: str) -> list[Path]:
    """Existing folders of ``step`` under ``root`` (own layout first, then script layouts)."""
    dirs = [root / name for name in STEP_FOLDERS.get(step, (step,)) if (root / name).is_dir()]
    for pat in LEGACY_FOLDER_GLOBS.get(step, ()):
        dirs += sorted(k for k in root.glob(pat) if k.is_dir())
    return dirs


def _strip_suffix(name: str, suffixes) -> str | None:
    for suf in sorted(suffixes, key=len, reverse=True):
        if name.endswith(suf):
            return name[: -len(suf)]
    return None


class WorkingFolder:
    """The working folder of the movie that is currently being processed."""

    def __init__(self) -> None:
        self.root: Path | None = None
        self.manual = False  # chosen by the user: do not replace it automatically

    # ---- choosing the folder ---------------------------------------------
    def set_manual(self, path: str | Path) -> Path:
        self.root = Path(path)
        self.manual = True
        return self.root

    def resolve(self, movie_paths) -> Path | None:
        """Manual choice if there is one, else derived from the first known movie path."""
        if self.manual and self.root is not None:
            return self.root
        for p in movie_paths:
            if p:
                self.root = results_dir_for(p)
                return self.root
        return self.root

    # ---- folders and files -------------------------------------------------
    def step_dir(self, step: str, create: bool = True) -> Path:
        assert self.root is not None, "no working folder set"
        d = self.root / step
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- settings.json -----------------------------------------------------
    @property
    def settings_path(self) -> Path:
        assert self.root is not None
        return self.root / "settings.json"

    def read_settings(self) -> dict:
        try:
            return json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def write_settings(self, step: str, params: dict, movie: str | None = None) -> None:
        assert self.root is not None
        self.root.mkdir(parents=True, exist_ok=True)
        data = self.read_settings()
        if movie:
            data["movie"] = str(movie)
        data[step] = {"time": datetime.now().isoformat(timespec="seconds"), "params": params}
        self.settings_path.write_text(
            json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8"
        )

    # ---- what is already there --------------------------------------------
    def found(self) -> dict[str, list[Path]]:
        """Result files already in the folder, per step (empty lists omitted)."""
        out: dict[str, list[Path]] = {}
        if self.root is None or not self.root.is_dir():
            return out
        patterns = {
            "segmentation": tuple(f"*{s}" for s in SEG_MASK_SUFFIXES),
            "tracking": tuple(f"*{s}" for s in TRACK_LABEL_SUFFIXES),
            "t1": (T1_PATTERN,),
        }
        for step, pats in patterns.items():
            files = sorted(
                {f for d in _step_dirs(self.root, step) for pat in pats for f in d.glob(pat)}
            )
            if files:
                out[step] = files
        return out

    def guess_movie(self) -> Path | None:
        """The movie behind saved results when ``settings.json`` does not name it.

        Derived from a mask / tracked-label file name; looked up in the results
        folder and the folder above it.
        """
        found = self.found()
        for step, suffixes in (("segmentation", SEG_MASK_SUFFIXES), ("tracking", TRACK_LABEL_SUFFIXES)):
            for f in found.get(step, []):
                stem = _strip_suffix(f.name, suffixes)
                if not stem:
                    continue
                for folder in (self.root, self.root.parent):
                    for ext in MOVIE_EXTENSIONS:
                        cand = folder / f"{stem}{ext}"
                        if cand.is_file():
                            return cand
        return None
