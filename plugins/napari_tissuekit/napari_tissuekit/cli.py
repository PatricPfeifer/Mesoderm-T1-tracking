"""``tissuekit``: run the pipeline steps from the command line (laptop or HPC).

Every step reads and writes the same results folder the napari widget uses, so a
job run on the HPC can be opened with "Folder..." / "Load results" afterwards::

    tissuekit segment movie.tif --preset cellflow     # -> movie_results/segmentation/cellflow/
    tissuekit track   movie.tif                       # -> movie_results/tracking/<run>/ (active run)
    tissuekit t1      movie.tif                       # -> movie_results/t1/<run>/

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
    return wf.movie_path() or wf.guess_movie()


def _need(wf: WorkingFolder, step: str, hint: str) -> Path:
    files = wf.found().get(step)  # segmentation: the active run
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
    params = segmentation.resolve_all(args.preset, args.params)  # fails early on a bad preset
    batch = args.batch_size or params["batch_size"]  # --batch-size wins over the preset
    # Each parameter set is its own run folder. Without --run a new name is picked (never
    # overwrites); an explicit --run reuses that folder, so a timed-out cluster job resumes.
    if args.run:
        from .core.seg_params import run_name

        run = run_name(args.run)
    else:
        run = wf.new_run_name(args.preset)
    out = wf.seg_run_dir(run)
    print(f"segmentation run: {run} -> {out}", flush=True)
    argv = ["--input", str(movie), "--output-dir", str(out), "--preset", args.preset,
            "--batch-size", str(batch)]
    if args.params:
        argv += ["--params", args.params]
    if args.force_cpu:
        argv.append("--force-cpu")
    if not args.fill_holes:
        argv.append("--no-fill-holes")
    segmentation.main(argv + extra)
    run_info = {"preset": args.preset, "params": {**params, "batch_size": batch},
                "batch_size": batch, "gpu": not args.force_cpu, "fill_holes": args.fill_holes}
    wf.write_run_params(run, run_info)
    wf.set_active_seg_run(run)
    wf.write_settings("segmentation", {"run": run, **run_info}, movie=str(movie))


def cmd_track(args, extra) -> None:
    from .core import tracking

    wf, movie = _folder(args.target, args.results)
    masks = _need(wf, "segmentation", f"tissuekit segment {args.target}")
    movie = _movie(wf, movie)
    run = wf.active_seg_run()  # each segmentation run keeps its own tracking: tracking/<run>/
    out = wf.run_step_dir("tracking", run)
    print(f"tracking of run {out.name} -> {out}", flush=True)
    argv = ["--masks", str(masks), "--output-dir", str(out)]
    if movie is not None:
        argv += ["--intensity", str(movie)]
    if args.fill_holes:
        argv.append("--fill-holes")
    tracking.main(argv + extra)
    wf.write_step_params("tracking", out.name, {"extra_args": extra, "masks": str(masks), "fill_holes": bool(args.fill_holes)},
                         movie=str(movie) if movie else None)


def cmd_t1(args, extra) -> None:
    from .core.t1 import run_tm_t1

    wf, _ = _folder(args.target, args.results)
    labels = _need(wf, "tracking", f"tissuekit track {args.target}")
    out = wf.run_step_dir("t1")  # T1 of the active run's tracking: t1/<run>/
    argv = ["--labels", str(labels), "--output-dir", str(out)]
    rgb = labels.with_name(labels.name.replace("_tracked_labels", "_tracked_rgb"))
    if rgb.is_file():
        argv += ["--rgb", str(rgb)]
    run_tm_t1.main(argv + extra)
    wf.write_step_params("t1", out.name, {"extra_args": extra, "labels": str(labels)})


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
    seg.add_argument("--preset", default="default",
                     help="default, cellflow or a preset saved in the plugin")
    seg.add_argument("--params", default=None,
                     help="JSON of Cellpose settings over the preset, e.g. {\"cellprob_threshold\": 0.5}")
    seg.add_argument("--run", default=None,
                     help="run folder name; an existing one is reused (default: the preset name, _2, _3 ... if taken)")
    seg.add_argument("--batch-size", type=int, default=None,
                     help="256x256 px tiles per GPU pass (default: the preset's, else 8)")
    seg.add_argument("--force-cpu", action="store_true")
    seg.add_argument("--no-fill-holes", dest="fill_holes", action="store_false",
                     help="leave empty space enclosed by cells empty (default: it becomes a new cell)")
    trk = add("track", cmd_track, "track the saved segmentation")
    trk.add_argument("--no-fill-holes", dest="fill_holes", action="store_false",
                     help="leave empty areas enclosed by cells without an id (default: they become new cells before tracking)")
    add("t1", cmd_t1, "detect T1 events on the saved tracking")
    return p


def main(argv=None) -> None:
    args, extra = build_parser().parse_known_args(argv)
    args.func(args, extra)


if __name__ == "__main__":
    main(sys.argv[1:])
