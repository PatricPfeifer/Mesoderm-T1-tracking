#!/usr/bin/env python3
"""Bundle T1 picker inputs next to each T1 run (and under output/t1_picker_ready/)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
_PICKER = _SCRIPTS / "t1" / "run_tm_t1_picker.py"

README = """T1 picker bundle (step: picker_ready) — copy this folder to a Mac or PC.

Original movie stem: {stem}

Sibling step folders on the cluster:
  ../detect/  — T1 detection outputs
  ../picked/  — recommended --output-dir for exports from this picker

Once (conda env t1_pick with numpy tifffile scipy scikit-image PySide6).

Each time:
  conda activate t1_pick
  cd this_folder
  python run_tm_t1_picker.py --output-dir ../picked --window 10

The script auto-finds {stem}_t1_events.csv, {stem}_tracked_labels.tif,
and {stem}_tracked_rgb.tif in this folder.

Also included (for viewing, not required by the GUI):
  {stem}_t1_fourways.tif  grayscale + one type-colored pixel per T1 4-way

calibration.json (pixel size in um, frame interval in s) sits next to the data.
With it, every export also writes a microns-and-seconds twin of each table
(*_um.csv); the pixel/frame tables are always written too. Without it, set
--um-per-px and --s-per-frame on the command line.
"""


def picker_ready_dirs(t1_output_dir: Path, stem: str) -> list[Path]:
    """Destinations for the Mac/PC picker bundle.

    Step layout (preferred):
      .../{stem}_t1/detect/        → detection outputs
      .../{stem}_t1/picker_ready/  → this bundle (sibling of detect)

    Legacy fallback when output-dir is not named ``detect``:
      <t1-output-dir>/<stem>_t1_picker

    Also mirrored under output/t1_picker_ready/<stem>.
    """
    t1_out = Path(t1_output_dir).expanduser().resolve()
    if t1_out.name == "detect":
        local = t1_out.parent / "picker_ready"
    else:
        local = t1_out / f"{stem}_t1_picker"
    return [
        local,
        _ROOT / "output" / "t1_picker_ready" / stem,
    ]


def picker_ready_dir(t1_output_dir: Path, stem: str) -> Path:
    """Primary picker bundle (``picker_ready`` next to ``detect``)."""
    return picker_ready_dirs(t1_output_dir, stem)[0]


def _place_data(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.hardlink_to(src)
    except AttributeError:
        import os

        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _write_one_picker_dir(
    dest: Path,
    *,
    stem: str,
    events: Path,
    labels: Path,
    rgb_path: Path,
    four_path: Path | None,
    calibration: dict | None = None,
) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    _place_data(events, dest / events.name)
    _place_data(labels, dest / labels.name)
    _place_data(rgb_path, dest / rgb_path.name)
    if four_path is not None and four_path.is_file():
        _place_data(four_path, dest / four_path.name)
    shutil.copy2(_PICKER, dest / "run_tm_t1_picker.py")
    (dest / "README.txt").write_text(README.format(stem=stem))
    (dest / "original_input.txt").write_text(f"{stem}\n")
    _write_calibration(dest, calibration)
    return dest


def _write_calibration(dest: Path, calibration: dict | None) -> None:
    """calibration.json next to the data (the picker is standalone on a Mac/PC)."""
    if calibration and (calibration.get("um_per_px") or calibration.get("s_per_frame")):
        (dest / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")


def write_picker_ready(
    *,
    t1_output_dir: Path,
    stem: str,
    events: Path,
    labels: Path,
    rgb: Path | None,
    fourways: Path | None = None,
    calibration: dict | None = None,
) -> Path | None:
    """Hardlink picker data + copy the GUI script. Returns primary dest or None."""
    events = Path(events).expanduser().resolve()
    labels = Path(labels).expanduser().resolve()
    rgb_path = Path(rgb).expanduser().resolve() if rgb is not None else None
    four_path = (
        Path(fourways).expanduser().resolve() if fourways is not None else None
    )
    if not events.is_file() or not labels.is_file():
        print("Picker-ready: skip (missing events or labels)", flush=True)
        return None
    if rgb_path is None or not rgb_path.is_file():
        print("Picker-ready: skip (no tracked RGB)", flush=True)
        return None

    primary: Path | None = None
    for dest in picker_ready_dirs(t1_output_dir, stem):
        written = _write_one_picker_dir(
            dest,
            stem=stem,
            events=events,
            labels=labels,
            rgb_path=rgb_path,
            four_path=four_path,
            calibration=calibration,
        )
        print(f"Picker-ready: {written}", flush=True)
        if primary is None:
            primary = written
    return primary


def backfill_picker_ready() -> None:
    """Rebuild picker_ready next to each movie's detect/ folder."""
    t1_root = _ROOT / "output" / "t1"
    track_root = _ROOT / "output" / "cellpose_track"
    for movie_dir in sorted(t1_root.iterdir()):
        if not movie_dir.is_dir():
            continue
        detect = movie_dir / "detect"
        stable = detect if detect.is_dir() else movie_dir
        events = next(stable.glob("*_t1_events.csv"), None)
        if events is None:
            events = stable / "t1_events.csv"
        if not events.is_file():
            continue
        stem = events.stem.replace("_t1_events", "")
        track = track_root / f"{stem}_cp_track"
        if not track.is_dir():
            track = track_root / movie_dir.name
        labels = (
            next(track.glob("*_tracked_labels.tif"), None)
            if track.is_dir()
            else None
        )
        rgb = (
            next(track.glob("*_tracked_rgb.tif"), None)
            if track.is_dir()
            else None
        )
        if labels is None and track.is_dir():
            labels = track / "tracked_labels.tif"
        if rgb is None and track.is_dir():
            rgb = track / "tracked_rgb.tif"
        fourways = next(stable.glob("*_t1_fourways.tif"), None)
        if fourways is None:
            fourways = stable / "t1_fourways.tif"
        cal = _calibration_for_stem(stem, labels)
        write_picker_ready(
            t1_output_dir=stable,
            stem=stem,
            events=events,
            labels=labels,
            rgb=rgb if rgb is not None and rgb.is_file() else None,
            fourways=fourways if fourways.is_file() else None,
            calibration=cal,
        )


def _calibration_for_stem(stem: str, labels: Path | None = None) -> dict | None:
    """Calibration dict from the labels TIFF, else the raw movie ``input/<stem>.tif``."""
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    from movie_calibration import calibration_for

    cal = calibration_for(labels if labels is not None and Path(labels).is_file() else None, stem=stem)
    return cal.as_dict() if cal else None


def refresh_picker_bundles() -> list[Path]:
    """Update the script, README and calibration.json of every existing picker
    bundle in place. The bundles' events / labels / RGB files are not touched."""
    found: list[Path] = []
    for root in (_ROOT / "output" / "t1", _ROOT / "output" / "t1_picker_ready"):
        if root.is_dir():
            found += [p.parent for p in root.rglob("run_tm_t1_picker.py")]
    done: list[Path] = []
    for dest in sorted(set(found)):
        stem_file = dest / "original_input.txt"
        stem = stem_file.read_text().strip() if stem_file.is_file() else ""
        if not stem:
            continue
        shutil.copy2(_PICKER, dest / "run_tm_t1_picker.py")
        (dest / "README.txt").write_text(README.format(stem=stem))
        labels = next(dest.glob("*_tracked_labels.tif"), None)
        _write_calibration(dest, _calibration_for_stem(stem, labels))
        has_cal = (dest / "calibration.json").is_file()
        print(f"refreshed {dest}  calibration.json={'yes' if has_cal else 'NO'}", flush=True)
        done.append(dest)
    return done


if __name__ == "__main__":
    if "--refresh" in sys.argv[1:]:
        refresh_picker_bundles()
    else:
        backfill_picker_ready()
