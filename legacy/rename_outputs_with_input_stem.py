#!/usr/bin/env python3
"""Rename existing generic outputs to {input_stem}_{step_tag}.{ext}."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from napari_patric.core.output_names import input_stem, tagged  # noqa: E402

ROOT = Path("/home/ho/ho_proana/ho_luchi75/projects/patric")
OUTPUT = ROOT / "output"

FOLDER_STEM = {
    "naima_cut": "Cri_WT_GAP43_GFP_38sec_Naima_cut",
    "mac_t1_picker": "Cri_WT_GAP43_GFP_38sec_Naima_cut",
    "pc_t1_picker": "Cri_WT_GAP43_GFP_38sec_Naima_cut",
    "masks": "Cri_WT_GAP43_GFP_38sec_Naima_cut",
    "Cri_WT_GAP43_GFP_38sec_Naima_cut": "Cri_WT_GAP43_GFP_38sec_Naima_cut",
    "chironomus_capture3_pos2": (
        "20260123_63X_Chironomus_LA_and_Gap43-GFP.sld - Capture 3 - Position 2-1-1-1"
    ),
    "chironomus_cap3_pos2": (
        "20260123_63X_Chironomus_LA_and_Gap43-GFP.sld - Capture 3 - Position 2-1-1-1"
    ),
    "lateral20250806_slice7": "lateral20250806bCri-Gap43Verena_6_180s-slice7",
    "ventrolateral20231202_90s": "ventrolateral20231202CriGap43Verenapos001-90s",
    "hermetia_ctrl_lateral2": (
        "20260119_40X_Hermetia_Gap43-GFP_cellularization_#2_Control_lateral-2"
    ),
    "hermetia_ctrl_lateral6": (
        "20260119_40X_Hermetia_Gap43-GFP_cellularization_#6_Control_lateral-1"
    ),
}

# Generic basename -> step-tag filename (without stem prefix).
ROLE_FOR = {
    "tracked_labels.tif": "tracked_labels.tif",
    "tracked_rgb.tif": "tracked_rgb.tif",
    "tracks.csv": "tracks.csv",
    "links.csv": "links.csv",
    "events.csv": "track_events.csv",
    "qc.json": "qc.json",
    "t1_events.csv": "t1_events.csv",
    "t1_overlay.tif": "t1_overlay.tif",
    "t1_fourways.tif": "t1_fourways.tif",
    "fourway_vertices.csv": "fourway_vertices.csv",
    "junction_lengths.csv": "junction_lengths.csv",
    "fourway_by_frame.csv": "fourway_by_frame.csv",
    "fourway_overlay.tif": "fourway_overlay.tif",
    "junction_length_vs_frame.png": "junction_length_vs_frame.png",
    "t1_by_frame.csv": "t1_by_frame.csv",
    "t1_half_events.csv": "t1_half_events.csv",
    "t1_neighbour_changes.csv": "t1_neighbour_changes.csv",
    "t1_highlights.tif": "t1_highlights.tif",
    "selected_t1s.csv": "selected_t1s.csv",
    "selected_traces.csv": "selected_traces.csv",
    "selected_traces_length.csv": "selected_traces_length.csv",
    "selected_traces_angle.csv": "selected_traces_angle.csv",
    "selected_overlay.tif": "selected_overlay.tif",
    "selected_length.tif": "selected_length.tif",
    "cp_masks.tif": "cp_masks.tif",
    "cp_outlines.tif": "cp_outlines.tif",
    "naima_cut_cp_masks.tif": "cp_masks.tif",
    "naima_cut_cp_outlines.tif": "cp_outlines.tif",
}

EXTRA_DIRS = [
    ROOT / "pc_t1_picker",
    ROOT / "mac_t1_picker",
]


def stem_for(path: Path) -> str | None:
    for part in path.parts:
        if part in FOLDER_STEM:
            return FOLDER_STEM[part]
    qc = path.parent / "qc.json"
    if qc.is_file():
        try:
            data = json.loads(qc.read_text())
        except json.JSONDecodeError:
            data = {}
        for key in ("intensity", "masks", "labels"):
            val = data.get(key)
            if val:
                return input_stem(val)
    return None


def planned_dest(src: Path) -> Path | None:
    name = src.name
    if name.startswith("t1_overlay_f") and name.endswith(".tif"):
        extra = name[len("t1_overlay") : -len(".tif")]  # _f30_40
        stem = stem_for(src)
        if not stem:
            return None
        return src.with_name(tagged(stem, f"t1_overlay{extra}.tif"))

    role = ROLE_FOR.get(name)
    if role is None:
        return None
    stem = stem_for(src)
    if not stem:
        return None
    dest = src.with_name(tagged(stem, role))
    if dest.resolve() == src.resolve():
        return None
    return dest


def iter_files() -> list[Path]:
    files: list[Path] = []
    for base in (OUTPUT, *EXTRA_DIRS):
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if "archive" in p.parts or "__pycache__" in p.parts:
                continue
            if "frames" in p.parts:
                continue
            files.append(p)
    return files


def main() -> None:
    dry = "--apply" not in sys.argv
    moves: list[tuple[Path, Path]] = []
    skipped: list[str] = []
    for src in sorted(iter_files()):
        dest = planned_dest(src)
        if dest is None:
            continue
        if dest.exists():
            skipped.append(f"EXISTS {dest}  (from {src.name})")
            continue
        moves.append((src, dest))

    for src, dest in moves:
        print(f"{src.relative_to(ROOT)} -> {dest.name}")
    for line in skipped:
        print("SKIP", line)
    print(f"{len(moves)} renames, {len(skipped)} skipped, dry={dry}")
    if dry:
        print("Re-run with --apply to rename.")
        return
    for src, dest in moves:
        src.rename(dest)
    print("Done.")


if __name__ == "__main__":
    main()
