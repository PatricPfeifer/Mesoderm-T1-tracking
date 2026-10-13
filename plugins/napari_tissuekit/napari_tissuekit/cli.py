"""``tissuekit``: run the pipeline steps from the command line (laptop or HPC).

Every step reads and writes the same results folder the napari widget uses, so a
job run on the HPC can be opened with "Folder..." / "Load results" afterwards::

    tissuekit segment movie.tif --preset cellflow     # -> movie_results/segmentation/
    tissuekit track   movie.tif                       # -> movie_results/tracking/
    tissuekit t1      movie.tif                       # -> movie_results/t1/

``TARGET`` is the movie (results folder = ``<movie_dir>/<movie_stem>_results``) or an
existing results folder. Options this wrapper does not know are passed on to the step
(see ``python -m napari_tissuekit.core.tracking --help`` and ``...core.t1.run_tm_t1 --help``).
No napari or Qt is imported here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.project import WorkingFolder, find_results_root, results_dir_for


def _folder(target: str, results: str | None) -> tuple[WorkingFolder, Path | None]:
    """Working folder for ``target`` plus the movie when ``target`` is one."""
    path = Path(target).expanduser()
    movie = path if path.is_file() else None
    if results:
        root = Path(results).expanduser()
    elif movie is not None:
        root = results_dir_for(movie)
    elif path.is_dir():
        root = find_results_root(path)
    else:
        raise SystemExit(f"not a movie file or folder: {path}")
    wf = WorkingFolder()
    wf.set_manual(root)
    return wf, movie


def _movie(wf: WorkingFolder, movie: Path | None) -> Path | None:
    if movie is not None:
        return movie
    saved = wf.read_settings().get("movie")
    if saved and Path(saved).is_file():
        return Path(saved)
    return wf.guess_movie()


def _need(wf: WorkingFolder, step: str, hint: str) -> Path:
    files = wf.found().get(step)
    if not files:
        raise SystemExit(f"no {step} result in {wf.root}. Run `{hint}` first.")
    if len(files) > 1:
        print(f"{len(files)} {step} files, using {files[0].name}", flush=True)
    return files[0]


def cmd_segment(args, extra) -> None:
    from .core import segmentation

    wf, movie = _folder(args.target, args.results)
    if movie is None:
        raise SystemExit("segment needs the movie file as TARGET")
    out = wf.step_dir("segmentation")
    argv = ["--input", str(movie), "--output-dir", str(out), "--preset", args.preset,
            "--batch-size", str(args.batch_size)]
    if args.force_cpu:
        argv.append("--force-cpu")
    segmentation.main(argv + extra)
    wf.write_settings(
        "segmentation",
        {"preset": args.preset, "batch_size": args.batch_size, "gpu": not args.force_cpu},
        movie=str(movie),
    )


def cmd_track(args, extra) -> None:
    from .core import tracking

    wf, movie = _folder(args.target, args.results)
    masks = _need(wf, "segmentation", f"tissuekit segment {args.target}")
    movie = _movie(wf, movie)
    argv = ["--masks", str(masks), "--output-dir", str(wf.step_dir("tracking"))]
    if movie is not None:
        argv += ["--intensity", str(movie)]
    tracking.main(argv + extra)
    wf.write_settings("tracking", {"extra_args": extra}, movie=str(movie) if movie else None)


def cmd_t1(args, extra) -> None:
    from .core.t1 import run_tm_t1

    wf, _ = _folder(args.target, args.results)
    labels = _need(wf, "tracking", f"tissuekit track {args.target}")
    argv = ["--labels", str(labels), "--output-dir", str(wf.step_dir("t1"))]
    rgb = labels.with_name(labels.name.replace("_tracked_labels", "_tracked_rgb"))
    if rgb.is_file():
        argv += ["--rgb", str(rgb)]
    run_tm_t1.main(argv + extra)
    wf.write_settings("t1", {"extra_args": extra})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tissuekit", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, func, help_):
        sp = sub.add_parser(name, help=help_, description=help_)
        sp.add_argument("target", help="movie file, or an existing results folder")
        sp.add_argument("--results", help="results folder (default: <movie>_results)")
        sp.set_defaults(func=func)
        return sp

    seg = add("segment", cmd_segment, "Cellpose segmentation of a movie")
    seg.add_argument("--preset", choices=("default", "cellflow"), default="default")
    seg.add_argument("--batch-size", type=int, default=8)
    seg.add_argument("--force-cpu", action="store_true")
    add("track", cmd_track, "track the saved segmentation")
    add("t1", cmd_t1, "detect T1 events on the saved tracking")
    return p


def main(argv=None) -> None:
    args, extra = build_parser().parse_known_args(argv)
    args.func(args, extra)


if __name__ == "__main__":
    main(sys.argv[1:])
