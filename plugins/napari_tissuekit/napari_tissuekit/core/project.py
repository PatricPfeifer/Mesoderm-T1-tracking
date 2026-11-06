"""Working folder for one movie: where every pipeline step saves its results.

Layout (next to the input movie by default)::

    <movie_dir>/<movie_stem>_results/
        segmentation/   one folder per run (parameter set): <run>/ masks + outlines + params.json;
                        files directly in segmentation/ (older layout, cluster/HPC) are the run "original"
        tracking/<run>/ tracking of one segmentation run: tracked labels, RGB, tracks/links/events
                        CSVs, qc.json, params.json. Files directly in tracking/ (older layout: one
                        tracking per egg) belong to the run named in settings.json, if any.
        t1/<run>/       T1 events of that run's tracking (files directly in t1/: older layout)
        cell_props/     exports and selections from "Export Cell Props"
        settings.json   parameters and time of the last run of every step, active segmentation run

Nothing in here depends on napari or Qt, so it can be tested on its own.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

STEPS = ("segmentation", "tracking", "t1", "cell_props")

# Naming contract with the pipeline scripts (scripts/, HPC runs). Anything a script
# writes must be matched here, or "Load results" will not see it (tests/plugin/test_project.py
# checks this against scripts/output_names.py). Longest suffix first.
SEG_MASK_SUFFIXES = ("_cp_masks.tif", "_cf_masks.tif")
TRACK_LABEL_SUFFIXES = ("_cf_tracked_labels.tif", "_tracked_labels.tif")
# Outline files: 0/255 1 px bonds *_cell_outlines.tif (loaded first) or yellow RGB *_cp_outlines.tif (Cellpose style: a line inside each cell).
OUTLINE_SUFFIXES = ("_cell_outlines.tif", "_cp_outlines.tif")
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
# Steps that are kept per segmentation run: <step>/<run>/ (files directly in <step>/ = older layout).
PER_RUN_STEPS = ("tracking", "t1")
# Segmentation runs: files directly in the segmentation folder(s) form this run.
LEGACY_RUN = "original"
RUN_PARAMS = "params.json"
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
    if len(kids) == 1:
        return kids[0]
    # A step folder itself (…/tracking) was picked: use the results folder above it.
    if p.name in STEPS or any(p.name in names for names in STEP_FOLDERS.values()):
        parent = p.parent
        if parent.name.endswith("_results") or (parent / "settings.json").is_file():
            return parent
    return p


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


def _seg_files(dirs) -> list[Path]:
    """Mask and outline files directly inside ``dirs``."""
    sufs = (*SEG_MASK_SUFFIXES, *OUTLINE_SUFFIXES)
    return sorted({f for d in dirs if d.is_dir() for s in sufs for f in d.glob(f"*{s}")})


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

    def run_step_dir(self, step: str, run: str | None = None, create: bool = True) -> Path:
        """``tracking/<run>/`` or ``t1/<run>/``: every segmentation run keeps its own (``run``: else the active run)."""
        assert self.root is not None, "no working folder set"
        assert step in PER_RUN_STEPS, step
        d = self.root / step / (run or self.active_seg_run() or LEGACY_RUN)
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def write_step_params(self, step: str, run: str | None, params: dict, movie: str | None = None) -> Path:
        """``params.json`` of a per-run step (and the movie in settings.json)."""
        d = self.run_step_dir(step, run)
        data = {"segmentation_run": d.name, "time": datetime.now().isoformat(timespec="seconds"), "params": params}
        (d / RUN_PARAMS).write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
        if movie:
            settings = self.read_settings()
            settings["movie"] = self._relative_movie(movie)
            self.settings_path.write_text(json.dumps(settings, indent=2, default=str) + "\n", encoding="utf-8")
        return d / RUN_PARAMS

    def legacy_tracking_run(self) -> str | None:
        """The run an older, flat ``tracking/`` folder was made from (settings.json), if recorded."""
        return ((self.read_settings().get("tracking") or {}).get("params") or {}).get("segmentation_run")

    def _per_run_dirs(self, step: str, run: str | None) -> list[Path]:
        """Where ``step`` of ``run`` is: its own folder if it holds files, else an older flat folder
        when that one belongs to ``run`` (or to no recorded run)."""
        own = self.root / step / run if run else None
        if own is not None and own.is_dir() and any(f.is_file() for f in own.iterdir()):
            return [own]
        owner = self.legacy_tracking_run()
        if owner is None or owner == run:
            return _step_dirs(self.root, step)
        return []

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
            data["movie"] = self._relative_movie(movie)
        data[step] = {"time": datetime.now().isoformat(timespec="seconds"), "params": params}
        self.settings_path.write_text(
            json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8"
        )

    def _relative_movie(self, movie: str | Path) -> str:
        """Movie path relative to the results folder, with ``/`` on every OS.

        A shared or synced results folder then names the same movie on every machine
        (an absolute path only exists on the computer that wrote it).
        """
        assert self.root is not None
        try:
            rel = os.path.relpath(Path(movie).resolve(), self.root.resolve())
        except ValueError:  # Windows: movie on another drive than the results folder
            return str(movie)
        return Path(rel).as_posix()

    def movie_path(self) -> Path | None:
        """The movie named in ``settings.json``, if it exists on this computer.

        Relative entries are resolved against the results folder. Older files store an
        absolute path from the computer that wrote them; when that path is missing here,
        the file of the same name next to (or inside) the results folder is used.
        """
        saved = self.read_settings().get("movie")
        if not saved or self.root is None:
            return None
        path = Path(saved)
        if not path.is_absolute() and not PureWindowsPath(saved).is_absolute():
            path = self.root / saved
        if path.is_file():
            return path
        name = (PureWindowsPath if "\\" in saved else PurePosixPath)(saved).name
        for folder in (self.root.parent, self.root):
            if (folder / name).is_file():
                return folder / name
        return None

    # ---- segmentation runs --------------------------------------------------
    def _legacy_seg_dirs(self) -> list[Path]:
        """Folders whose own files form the run "original" (masks may also sit in the results folder itself)."""
        return [*_step_dirs(self.root, "segmentation"), self.root]

    def _run_dirs(self, run: str | None) -> list[Path]:
        if run is None:
            run = self.active_seg_run()
        if run is None or run == LEGACY_RUN:
            return self._legacy_seg_dirs()
        return [self.root / "segmentation" / run]

    def seg_runs(self) -> list[str]:
        """Segmentation runs that hold masks or outlines ("original" first, then by name)."""
        if self.root is None or not self.root.is_dir():
            return []
        runs = [LEGACY_RUN] if _seg_files(self._legacy_seg_dirs()) else []
        base = self.root / "segmentation"
        if base.is_dir():
            runs += sorted(d.name for d in base.iterdir() if d.is_dir() and _seg_files([d]))
        return runs

    def seg_run_dir(self, run: str, create: bool = True) -> Path:
        """Folder of ``run``. "original" is the folder that already holds its masks (else segmentation/)."""
        assert self.root is not None, "no working folder set"
        if run == LEGACY_RUN:
            files = _seg_files(self._legacy_seg_dirs())
            d = files[0].parent if files else self.root / "segmentation"
        else:
            d = self.root / "segmentation" / run
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def new_run_name(self, base: str) -> str:
        """``base`` made safe and unused: ``cellflow``, then ``cellflow_2`` ..."""
        from .seg_params import run_name

        base = run_name(base)
        taken = set(self.seg_runs()) | {LEGACY_RUN}
        if self.root is not None and (self.root / "segmentation").is_dir():
            taken |= {d.name for d in (self.root / "segmentation").iterdir() if d.is_dir()}
        name, i = base, 2
        while name in taken:
            name, i = f"{base}_{i}", i + 1
        return name

    def active_seg_run(self) -> str | None:
        """The run the plugin shows and tracks: the one saved in settings.json, else the newest."""
        runs = self.seg_runs()
        if not runs:
            return None
        saved = self.read_settings().get("active_segmentation_run")
        if saved in runs:
            return saved
        newest = {r: max(f.stat().st_mtime for f in _seg_files(self._run_dirs(r))) for r in runs}
        return max(runs, key=newest.get)

    def set_active_seg_run(self, run: str) -> None:
        assert self.root is not None
        self.root.mkdir(parents=True, exist_ok=True)
        data = self.read_settings()
        data["active_segmentation_run"] = run
        self.settings_path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")

    def write_run_params(self, run: str, params: dict) -> Path:
        path = self.seg_run_dir(run) / RUN_PARAMS
        data = {"run": run, "time": datetime.now().isoformat(timespec="seconds"), **params}
        path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
        return path

    def read_run_params(self, run: str) -> dict:
        try:
            return json.loads((self.seg_run_dir(run, create=False) / RUN_PARAMS).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    # ---- what is already there --------------------------------------------
    def found(self, run: str | None = None) -> dict[str, list[Path]]:
        """Result files already in the folder, per step (empty lists omitted).

        Segmentation masks and outlines, tracking and T1 come from one run: ``run``, else the active run.
        """
        out: dict[str, list[Path]] = {}
        if self.root is None or not self.root.is_dir():
            return out
        run = run or self.active_seg_run()
        seg_dirs = self._run_dirs(run)
        masks = sorted({f for d in seg_dirs for s in SEG_MASK_SUFFIXES for f in d.glob(f"*{s}")})
        if masks:
            out["segmentation"] = masks
        tracks = sorted({f for d in self._per_run_dirs("tracking", run) for s in TRACK_LABEL_SUFFIXES
                         for f in d.glob(f"*{s}")})
        if tracks:
            out["tracking"] = tracks
        t1 = sorted({f for d in self._per_run_dirs("t1", run) for f in d.glob(T1_PATTERN)})
        if t1:
            out["t1"] = t1
        outlines = sorted(
            {f for d in seg_dirs for suf in OUTLINE_SUFFIXES for f in d.glob(f"*{suf}")},
            key=lambda f: (not f.name.endswith(OUTLINE_SUFFIXES[0]), f.name),  # *_cell_outlines.tif first
        )
        if outlines:
            out["outlines"] = outlines
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
