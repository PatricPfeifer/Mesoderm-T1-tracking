"""Batch: many eggs (movies in one folder) x many profiles (segmentation presets).

Two phases of an analysis use the same idea:

1. finding the profile: several presets on one or two eggs, then compare the runs;
2. production: one preset on many eggs.

A batch is a list of (egg, profile) items. Each item becomes one segmentation run
``<folder>/<egg>_results/segmentation/<run>/``, the same layout as a single run, so
the Segmentation tab, Load results and the CLI see batch results like any other.

Whether an egg already has a profile is read from disk only: a run whose
``params.json`` holds the same Cellpose values. No extra state file. No napari / Qt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .project import WorkingFolder, movie_stem, results_dir_for
from .seg_params import SPEED_KEYS, full_params, run_name

BATCH_EXTENSIONS = (".tif", ".tiff")  # what the cluster job reads (``tissuekit segment``)


class BatchError(ValueError):
    """A batch cannot be built; the message is meant for the user."""


def find_movies(folder: str | Path) -> list[Path]:
    """The ``.tif`` movies directly in ``folder``, sorted by name.

    Refuses two movies with the same stem (``a.tif`` and ``a.ome.tif``): they would
    share one ``a_results`` folder.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise BatchError(f"Not a folder: {folder}")
    movies = sorted(
        (p for p in folder.iterdir() if p.is_file() and p.name.lower().endswith(BATCH_EXTENSIONS)),
        key=lambda p: p.name.lower(),
    )
    seen: dict[str, Path] = {}
    for m in movies:
        stem = movie_stem(m)
        if stem in seen:
            raise BatchError(
                f"{seen[stem].name} and {m.name} would share the results folder {stem}_results. "
                "Rename one of them."
            )
        seen[stem] = m
    return movies


def quality_values(params: dict | None) -> dict:
    """The values that change the segmentation (batch size only changes speed)."""
    p = full_params(params or {})
    for key in SPEED_KEYS:
        p.pop(key, None)
    return p


def _run_values(info: dict) -> dict | None:
    """Cellpose values stored in a run's params.json (None if it has none)."""
    params = info.get("params")
    return quality_values(params) if isinstance(params, dict) else None


@dataclass
class RunInfo:
    name: str
    preset: str = ""
    time: str = ""
    fill_holes: bool | None = None
    values: dict | None = None  # quality values, None for runs without params.json


@dataclass
class EggStatus:
    movie: Path
    results: Path
    runs: list[RunInfo] = field(default_factory=list)
    active_run: str | None = None
    has_tracking: bool = False
    has_t1: bool = False

    @property
    def name(self) -> str:
        return movie_stem(self.movie)

    def run_for(self, profile_values: dict) -> RunInfo | None:
        """The run made with these Cellpose values (the active one if several, else the newest)."""
        want = quality_values(profile_values)
        hits = [r for r in self.runs if r.values == want]
        if not hits:
            return None
        active = next((r for r in hits if r.name == self.active_run), None)
        return active or max(hits, key=lambda r: r.time)


def egg_status(movie: str | Path) -> EggStatus:
    """What is already saved for one egg (reads its ``<stem>_results`` folder)."""
    movie = Path(movie)
    results = results_dir_for(movie)
    status = EggStatus(movie=movie, results=results)
    if not results.is_dir():
        return status
    wf = WorkingFolder()
    wf.set_manual(results)
    for name in wf.seg_runs():
        info = wf.read_run_params(name)
        status.runs.append(RunInfo(
            name=name, preset=str(info.get("preset") or ""), time=str(info.get("time") or ""),
            fill_holes=info.get("fill_holes"), values=_run_values(info),
        ))
    status.active_run = wf.active_seg_run()
    found = wf.found()
    status.has_tracking = "tracking" in found
    status.has_t1 = "t1" in found
    return status


def scan_folder(folder: str | Path) -> list[EggStatus]:
    return [egg_status(m) for m in find_movies(folder)]


@dataclass
class BatchItem:
    """One (egg, profile) pair: one segmentation run on the cluster."""

    movie: str          # local movie path
    local_results: str  # the egg's *_results folder
    profile: str        # preset name (a label on the cluster; the values travel in params)
    run: str            # run folder name, free in the egg's results folder
    params: dict        # every Cellpose value incl. batch size
    fill_holes: bool = True
    status: str = "pending"  # pending | done | failed | downloaded

    @property
    def movie_name(self) -> str:
        return Path(self.movie).name

    @property
    def egg(self) -> str:
        return movie_stem(self.movie)


def plan_items(eggs: list[EggStatus], profiles: dict[str, dict], *, rerun: bool = False,
               fill_holes: bool = True) -> tuple[list[BatchItem], list[tuple[str, str, str]]]:
    """Items for every (egg, profile), grouped by egg (each movie is uploaded once).

    Returns ``(items, skipped)``; ``skipped`` lists ``(egg, profile, existing run)`` for
    pairs that are already done (unless ``rerun``). Run names never reuse a folder that
    exists: a second ``cellflow`` run becomes ``cellflow_2``.
    """
    items: list[BatchItem] = []
    skipped: list[tuple[str, str, str]] = []
    for egg in eggs:
        wf = WorkingFolder()
        wf.set_manual(egg.results)
        taken: set[str] = set()
        for profile, values in profiles.items():
            done = egg.run_for(values)
            if done is not None and not rerun:
                skipped.append((egg.name, profile, done.name))
                continue
            run = _free_name(wf, run_name(profile), taken)
            taken.add(run)
            items.append(BatchItem(
                movie=str(egg.movie), local_results=str(egg.results), profile=profile, run=run,
                params=full_params(values), fill_holes=fill_holes,
            ))
    return items, skipped


def _free_name(wf: WorkingFolder, base: str, taken: set[str]) -> str:
    """``wf.new_run_name`` that also avoids names given to earlier items of this batch."""
    name, i = base, 2
    while name in taken or wf.new_run_name(name) != name:
        name, i = f"{base}_{i}", i + 1
    return name
