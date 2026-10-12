"""End-to-end tracking pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from contextlib import contextmanager
from timeit import default_timer as timer
from typing import Any

import numpy as np
import pandas as pd
import tifffile

from .cells import build_all_frames
from .io import Movie, is_blank_frame, load_movie
from .motion import estimate_all_motion, registration_images
from .qc import compute_qc, write_qc
from .tracks import (
    TrackingResult,
    lineage_rows,
    mark_divisions,
    render_tracked_labels,
    render_tracked_rgb,
    track_movie,
)

from ..output_names import input_stem, tagged_path  # noqa: E402
from ..movie_calibration import Calibration, calibration_for, save_tiff  # noqa: E402


@contextmanager
def _stage(timings: dict[str, float], name: str):
    """Add the wall time of the ``with`` body to ``timings[name]``."""
    t0 = timer()
    try:
        yield
    finally:
        timings[name] = round(timings.get(name, 0.0) + timer() - t0, 3)


def movie_from_arrays(
    masks: np.ndarray,
    intensity: np.ndarray | None = None,
    *,
    drop_blank: bool = True,
) -> Movie:
    """Build a Movie from in-memory stacks, dropping blank frames like load_movie."""
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    if masks.ndim != 3:
        raise ValueError(f"Expected T,Y,X masks; got shape {masks.shape}")
    intensity_arr: np.ndarray | None = None
    if intensity is not None:
        intensity_arr = np.asarray(intensity)
        if intensity_arr.ndim == 2:
            intensity_arr = intensity_arr[np.newaxis, ...]
        if intensity_arr.ndim == 4 and intensity_arr.shape[-1] <= 4:
            intensity_arr = intensity_arr[..., 0]
        if intensity_arr.shape != masks.shape:
            raise ValueError(
                f"Intensity shape {intensity_arr.shape} does not match masks {masks.shape}."
            )
        intensity_arr = intensity_arr.astype(np.float32, copy=False)

    keep: list[int] = []
    for t in range(masks.shape[0]):
        inten_t = intensity_arr[t] if intensity_arr is not None else None
        if drop_blank and is_blank_frame(masks[t], inten_t):
            continue
        keep.append(t)
    if len(keep) < 2:
        raise ValueError(
            f"Need at least 2 non-blank frames; kept {len(keep)} of {masks.shape[0]}."
        )
    idx = np.asarray(keep, dtype=np.int32)
    return Movie(
        masks=masks[idx],
        intensity=None if intensity_arr is None else intensity_arr[idx],
        frame_indices=idx,
    )


def track_loaded(
    movie: Movie,
    output_dir: Path | None = None,
    *,
    stem: str = "movie",
    masks_path: Path | None = None,
    intensity_path: Path | None = None,
    expand: int = 3,
    pyramidal_depth: int = 3,
    w_topo: float = 0.75,
    max_iter: int = 3,
    appear_cost: float = 1.5,
    disappear_cost: float = 1.5,
    division_penalty: float = 0.8,
    max_gap: int = 3,
    gap_close: bool = True,
    max_distance: float = 50.0,
    run_backward: bool = False,
    write_rgb: bool = True,
    verbose: bool = True,
    division_mode: str = "legacy",
    division_refractory: int = 10,
    min_daughter_frames: int = 3,
    calibration: Calibration | None = None,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Track a loaded movie. Writes files when ``output_dir`` is set.

    ``calibration`` (pixel size, frame interval) is written into the output
    TIFFs and qc.json; when not given it is read from the intensity / masks
    files, else from the raw movie of the same stem.
    """
    cal = calibration
    if cal is None and (intensity_path or masks_path):
        cal = calibration_for(intensity_path, masks_path, stem=stem)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    start = timer()
    timings: dict[str, float] = {}
    if verbose:
        print("Building cell tables...", flush=True)
    with _stage(timings, "cell_tables"):
        frames = build_all_frames(
            movie.masks, movie.frame_indices, expand=expand, n_jobs=n_jobs
        )

    if verbose:
        print("Estimating motion...", flush=True)
    with _stage(timings, "motion"):
        reg = registration_images(movie.masks, movie.intensity)
        fields = estimate_all_motion(reg, depth=pyramidal_depth, n_jobs=n_jobs)

    cost_kwargs = {"max_distance": max_distance}
    if verbose:
        print("Tracking (forward)...", flush=True)
    with _stage(timings, "tracking_forward"):
        result = track_movie(
            frames,
            fields,
            w_topo=w_topo,
            max_iter=max_iter,
            appear_cost=appear_cost,
            disappear_cost=disappear_cost,
            division_penalty=division_penalty,
            max_gap=max_gap,
            gap_close=gap_close,
            cost_kwargs=cost_kwargs,
            verbose=verbose,
            division_mode=division_mode,
            division_refractory=division_refractory,
            min_daughter_frames=min_daughter_frames,
            timings=timings,
        )

    backward_result = None
    if run_backward:
        if verbose:
            print("Tracking (backward)...", flush=True)
        with _stage(timings, "tracking_backward"):
            rev_masks = movie.masks[::-1]
            rev_idx = movie.frame_indices[::-1]
            rev_frames = build_all_frames(
                rev_masks, rev_idx, expand=expand, n_jobs=n_jobs
            )
            rev_reg = reg[::-1]
            rev_fields = estimate_all_motion(
                rev_reg, depth=pyramidal_depth, n_jobs=n_jobs
            )
            backward_result = track_movie(
                rev_frames,
                rev_fields,
                w_topo=w_topo,
                max_iter=max_iter,
                appear_cost=appear_cost,
                disappear_cost=disappear_cost,
                division_penalty=division_penalty,
                max_gap=max_gap,
                gap_close=gap_close,
                cost_kwargs=cost_kwargs,
                verbose=verbose,
                division_mode=division_mode,
                division_refractory=division_refractory,
                min_daughter_frames=min_daughter_frames,
            )

    with _stage(timings, "render"):
        tracked = render_tracked_labels(
            movie.masks, result.frames, result.label_to_track
        )
        rgb = render_tracked_rgb(tracked) if write_rgb else None
        if rgb is not None and division_mode == "lineage":
            mark_divisions(rgb, result)
    labels_out = None
    rgb_out = None
    if output_dir is not None:
        write_timer = timer()
        labels_out = tagged_path(output_dir, stem, "tracked_labels.tif")
        save_tiff(labels_out, tracked, cal)
        if rgb is not None:
            rgb_out = tagged_path(output_dir, stem, "tracked_rgb.tif")
            save_tiff(rgb_out, rgb, cal)
            if verbose:
                print(f"RGB tracks: {rgb_out}", flush=True)
        _write_tables(result, output_dir, stem)
        timings["write"] = round(timer() - write_timer, 3)

    qc = compute_qc(result, backward=backward_result)
    qc["elapsed_s"] = round(timer() - start, 2)
    qc["timings_s"] = timings
    qc["masks"] = str(Path(masks_path).resolve()) if masks_path else None
    qc["intensity"] = (
        str(Path(intensity_path).resolve()) if intensity_path else None
    )
    qc["frame_indices"] = movie.frame_indices.tolist()
    qc["calibration"] = cal.as_dict() if cal else None
    if output_dir is not None:
        write_qc(qc, tagged_path(output_dir, stem, "qc.json"))

    if verbose:
        print(
            f"Done in {qc['elapsed_s']}s. tracks={result.n_tracks} "
            f"full_length={qc['full_length_tracks']} "
            f"mean_topo={qc['mean_topology_score']:.3f}",
            flush=True,
        )
        print(
            "Timings: "
            + ", ".join(f"{k}={v:.1f}s" for k, v in timings.items()),
            flush=True,
        )
        if labels_out is not None:
            print(f"Labels: {labels_out}", flush=True)
        if rgb_out is not None:
            print(f"RGB:    {rgb_out}", flush=True)

    return {
        "result": result,
        "qc": qc,
        "tracked": tracked,
        "rgb": rgb,
        "movie": movie,
        "rgb_path": rgb_out,
        "labels_path": labels_out,
    }


def track_arrays(
    masks: np.ndarray,
    intensity: np.ndarray | None = None,
    output_dir: Path | None = None,
    *,
    stem: str = "movie",
    expand: int = 3,
    pyramidal_depth: int = 3,
    w_topo: float = 0.75,
    max_iter: int = 3,
    appear_cost: float = 1.5,
    disappear_cost: float = 1.5,
    division_penalty: float = 0.8,
    max_gap: int = 3,
    gap_close: bool = True,
    max_distance: float = 50.0,
    run_backward: bool = False,
    write_rgb: bool = True,
    verbose: bool = True,
    division_mode: str = "legacy",
    division_refractory: int = 10,
    min_daughter_frames: int = 3,
    calibration: Calibration | None = None,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Track in-memory label (and optional intensity) stacks."""
    movie = movie_from_arrays(masks, intensity, drop_blank=True)
    return track_loaded(
        movie,
        output_dir,
        stem=stem,
        expand=expand,
        pyramidal_depth=pyramidal_depth,
        w_topo=w_topo,
        max_iter=max_iter,
        appear_cost=appear_cost,
        disappear_cost=disappear_cost,
        division_penalty=division_penalty,
        max_gap=max_gap,
        gap_close=gap_close,
        max_distance=max_distance,
        run_backward=run_backward,
        write_rgb=write_rgb,
        verbose=verbose,
        division_mode=division_mode,
        division_refractory=division_refractory,
        min_daughter_frames=min_daughter_frames,
        calibration=calibration,
        n_jobs=n_jobs,
    )


def run_tracking(
    masks_path: Path,
    intensity_path: Path | None,
    output_dir: Path,
    *,
    max_frames: int | None = None,
    expand: int = 3,
    pyramidal_depth: int = 3,
    w_topo: float = 0.75,
    max_iter: int = 3,
    appear_cost: float = 1.5,
    disappear_cost: float = 1.5,
    division_penalty: float = 0.8,
    max_gap: int = 3,
    gap_close: bool = True,
    max_distance: float = 50.0,
    run_backward: bool = False,
    write_rgb: bool = True,
    verbose: bool = True,
    division_mode: str = "legacy",
    division_refractory: int = 10,
    min_daughter_frames: int = 3,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Track a Cellpose movie and write all outputs under ``output_dir``."""
    movie = load_movie(
        Path(masks_path),
        Path(intensity_path) if intensity_path else None,
        max_frames=max_frames,
        drop_blank=True,
    )
    stem = input_stem(intensity_path or masks_path)
    return track_loaded(
        movie,
        output_dir,
        stem=stem,
        masks_path=Path(masks_path),
        intensity_path=Path(intensity_path) if intensity_path else None,
        expand=expand,
        pyramidal_depth=pyramidal_depth,
        w_topo=w_topo,
        max_iter=max_iter,
        appear_cost=appear_cost,
        disappear_cost=disappear_cost,
        division_penalty=division_penalty,
        max_gap=max_gap,
        gap_close=gap_close,
        max_distance=max_distance,
        run_backward=run_backward,
        write_rgb=write_rgb,
        verbose=verbose,
        division_mode=division_mode,
        division_refractory=division_refractory,
        min_daughter_frames=min_daughter_frames,
        n_jobs=n_jobs,
    )


def _write_tables(result: TrackingResult, output_dir: Path, stem: str) -> None:
    link_rows = [
        {
            "frame_from": ln.frame_from,
            "label_from": ln.label_from,
            "frame_to": ln.frame_to,
            "label_to": ln.label_to,
            "kind": ln.kind,
            "cost": ln.cost,
            "topology_score": ln.topology_score,
            "track_id": ln.track_id,
        }
        for ln in result.links
    ]
    pd.DataFrame(link_rows).to_csv(
        tagged_path(output_dir, stem, "links.csv"), index=False
    )

    track_rows = []
    for t, fc in enumerate(result.frames):
        mapping = result.label_to_track[t]
        score = (
            result.per_frame_scores[t]
            if t < len(result.per_frame_scores)
            else None
        )
        for i in range(fc.n_cells):
            lab = int(fc.original_labels[i + 1])
            track = mapping.get(lab)
            if track is None:
                continue
            track_rows.append(
                {
                    "frame": fc.original_frame,
                    "local_label": lab,
                    "track_id": track,
                    "area": int(fc.areas[i]),
                    "centroid_y": round(float(fc.centroids[i, 0]), 2),
                    "centroid_x": round(float(fc.centroids[i, 1]), 2),
                    "topology_score": score,
                }
            )
    pd.DataFrame(track_rows).to_csv(
        tagged_path(output_dir, stem, "tracks.csv"), index=False
    )

    event_rows = [
        {
            "kind": e.kind,
            "frame_from": e.frame_from,
            "frame_to": e.frame_to,
            "label_from": e.label_from,
            "label_to": e.label_to,
            "track_id": e.track_id,
            "track_id_b": e.track_id_b,
            "track_id_c": e.track_id_c,
            "cost": e.cost,
            "topology_score": e.topology_score,
            "flagged": e.flagged,
            "note": e.note,
        }
        for e in result.events
    ]
    pd.DataFrame(event_rows).to_csv(
        tagged_path(output_dir, stem, "track_events.csv"), index=False
    )

    if result.division_mode == "lineage":
        pd.DataFrame(
            lineage_rows(result),
            columns=[
                "track_id",
                "start_frame",
                "end_frame",
                "parent_track_id",
                "divides",
                "flagged",
            ],
        ).to_csv(tagged_path(output_dir, stem, "lineage.csv"), index=False)
