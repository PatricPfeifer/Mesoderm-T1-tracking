#!/usr/bin/env python3
"""Write the microns-and-seconds tables for existing T1 result folders.

For every ``*_t1_events.csv`` (and the other T1 tables) in the given folder, a
``*_um.csv`` twin is written next to it. The pixel/frame files are not touched.
The calibration is read from the raw movie ``input/<stem>.tif`` (or from
``--movie``, or from ``--labels`` if that TIFF carries it).

Example:
  python scripts/t1/add_calibration.py \\
    --t1-dir output/t1/Cri_WT_GAP43_GFP_38sec_Naima_cut_t1_divisions/detect
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from napari_patric.core.t1.calibrated_outputs import write_um_files  # noqa: E402
from napari_patric.core.movie_calibration import calibration_for  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--t1-dir", type=Path, required=True, nargs="+")
    p.add_argument("--movie", type=Path, default=None, help="Raw movie TIFF with the calibration.")
    p.add_argument("--labels", type=Path, default=None, help="Tracked labels TIFF, if it carries it.")
    args = p.parse_args()

    failed = 0
    for folder in args.t1_dir:
        folder = folder.expanduser()
        stems = sorted(
            f.name[: -len("_t1_events.csv")] for f in folder.glob("*_t1_events.csv")
        )
        if not stems:
            print(f"{folder}: no *_t1_events.csv found", file=sys.stderr)
            failed += 1
            continue
        for stem in stems:
            cal = calibration_for(args.movie, args.labels, stem=stem)
            print(f"{folder.name}/{stem}: {cal.as_dict() if cal else 'no calibration'}")
            if not write_um_files(folder, stem, cal):
                failed += 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
