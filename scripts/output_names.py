"""Name pipeline outputs after the input movie plus a step tag.

Example: Cri_WT_GAP43_GFP_38sec_Naima_cut_tracked_labels.tif
"""

from __future__ import annotations

from pathlib import Path

# Longest suffixes first so stripping is greedy and order-stable.
STRIP_SUFFIXES = (
    "_junction_contact_lengths_signed_um",
    "_junction_contact_lengths_um",
    "_junction_lengths_signed_um",
    "_junction_lengths_um",
    "_fourway_vertices_um",
    "_t1_events_um",
    "_cp_masks_partial",
    "_selected_traces_length_signed",
    "_selected_traces_length",
    "_selected_traces_contact_signed",
    "_selected_traces_contact",
    "_selected_traces_angle",
    "_selected_length",
    "_t1_neighbour_changes",
    "_junction_length_vs_frame",
    "_junction_contact_lengths_signed",
    "_junction_contact_lengths",
    "_junction_lengths_signed",
    "_t1_half_events",
    "_fourway_vertices",
    "_fourway_by_frame",
    "_fourway_overlay",
    "_junction_lengths",
    "_selected_traces",
    "_selected_overlay",
    "_selected_t1s",
    "_t1_highlights",
    "_t1_fourways",
    "_t1_overlay",
    "_t1_events",
    "_t1_by_frame",
    "_tracked_labels",
    "_tracked_rgb",
    "_track_events",
    "_track_qc",
    "_cp_outlines",
    "_cp_masks",
)


def input_stem(path: Path | str | None) -> str:
    """Movie stem, with known pipeline tags stripped if present."""
    if path is None:
        raise ValueError("Need an input path to derive the output stem")
    stem = Path(path).name
    if stem.endswith(".npy"):
        stem = stem[: -len(".npy")]
    else:
        stem = Path(path).stem
    changed = True
    while changed:
        changed = False
        for suf in STRIP_SUFFIXES:
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                changed = True
                break
    return stem


def tagged(stem: str, role: str) -> str:
    """``role`` is the step tag plus extension, e.g. ``tracked_labels.tif``."""
    return f"{stem}_{role}"


def tagged_path(directory: Path | str, stem: str, role: str) -> Path:
    return Path(directory) / tagged(stem, role)


# Per-movie folder under a stage parent (not a file suffix).
STAGE_FOLDER_TAGS = {
    "cellpose_seg": "cp_seg",
    "cellpose_track": "cp_track",
    "t1": "t1",
}


def step_folder(stem: str, stage: str) -> str:
    """``Cri_WT_…_cp_track`` for stage ``cellpose_track``."""
    try:
        tag = STAGE_FOLDER_TAGS[stage]
    except KeyError as exc:
        raise ValueError(f"Unknown stage {stage!r}") from exc
    return f"{stem}_{tag}"


def step_dir(root: Path | str, stage: str, stem: str) -> Path:
    """``<root>/<stage>/{stem}_{tag}`` e.g. output/cellpose_track/…_cp_track."""
    return Path(root) / stage / step_folder(stem, stage)
