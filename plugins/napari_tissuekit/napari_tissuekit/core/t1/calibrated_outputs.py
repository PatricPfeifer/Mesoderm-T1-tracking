"""Write the microns-and-seconds twin of each T1 result table.

The T1 scripts write their tables in pixels and frames. For every such table
this module writes a second file with the same rows in µm and seconds, named
with a ``_um`` suffix (``..._t1_events.csv`` -> ``..._t1_events_um.csv``). The
pixel files are never modified. Used by run_tm_t1.py right after it writes its
tables, and by add_calibration.py for result folders that already exist.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


from ..movie_calibration import Calibration, calibration_for  # noqa: E402
from ..output_names import input_stem  # noqa: E402

# Tables of one T1 run: pixel file role -> kind of conversion.
TABLES = {
    "t1_events.csv": "events",
    "fourway_vertices.csv": "fourway",
    "junction_lengths.csv": "wide",
    "junction_lengths_signed.csv": "wide",
    "junction_contact_lengths.csv": "wide",
    "junction_contact_lengths_signed.csv": "wide",
}


def _f(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") or "0"


def _num(text: str):
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def _um_name(path: Path) -> Path:
    return path.with_name(f"{path.stem}_um{path.suffix}")


def convert_events(src: Path, dst: Path, cal: Calibration) -> int:
    """t1_events.csv: frames -> seconds, y/x -> µm, persist frames -> seconds."""
    um, sec = cal.um_per_px, cal.s_per_frame
    with src.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    def rename(col: str) -> str | None:
        if col.startswith("frame_"):
            return f"time_{col[6:]}_s" if sec is not None else None
        if col in ("y", "x"):
            return f"{col}_um" if um is not None else None
        if col in ("persist_pre", "persist_post"):
            return f"{col}_s" if sec is not None else None
        return col

    out_fields = [rename(c) for c in fields]
    with dst.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([c for c in out_fields if c])
        for row in rows:
            cells = []
            for col, new in zip(fields, out_fields):
                if not new:
                    continue
                val = row[col]
                if new != col and val != "":
                    scale = um if col in ("y", "x") else sec
                    val = _f(float(val) * scale, 3)
                cells.append(val)
            writer.writerow(cells)
    return len(rows)


def convert_fourway(src: Path, dst: Path, cal: Calibration) -> int:
    """fourway_vertices.csv: frame -> time_s, y/x -> µm, length px -> µm."""
    um, sec = cal.um_per_px, cal.s_per_frame
    with src.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    mapping: list[tuple[str, str, float | None]] = []  # (src col, new col, scale)
    for col in fields:
        if col == "frame" and sec is not None:
            mapping.append((col, "time_s", sec))
        elif col == "t_index":
            continue  # index into the non-blank frames; time_s replaces it
        elif col in ("y", "x") and um is not None:
            mapping.append((col, f"{col}_um", um))
        elif col == "length_vertex_px" and um is not None:
            mapping.append((col, "length_vertex_um", um))
        elif col == "frame" or col in ("y", "x") or col == "length_vertex_px":
            continue
        else:
            mapping.append((col, col, None))
    with dst.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([new for _, new, _ in mapping])
        for row in rows:
            out = []
            for col, _new, scale in mapping:
                val = row[col]
                if scale is not None and val != "":
                    val = _f(float(val) * scale, 3)
                out.append(val)
            writer.writerow(out)
    return len(rows)


def convert_wide(src: Path, dst: Path, cal: Calibration) -> int:
    """Wide Excel table (``sep=;``, comma decimals): first column dt in frames
    -> seconds, lengths px -> µm. Flags ("nv", "4way", ...) pass through."""
    um, sec = cal.um_per_px, cal.s_per_frame
    lines = src.read_text(encoding="utf-8").splitlines()
    out = []
    n = 0
    for i, line in enumerate(lines):
        if i < 3:  # sep=;, event_id, event_type
            out.append(line)
            continue
        cells = line.split(";")
        if sec is not None and _num(cells[0]) is not None:
            cells[0] = _f(_num(cells[0]) * sec, 2).replace(".", ",")
        for j in range(1, len(cells)):
            value = _num(cells[j]) if cells[j] else None
            if value is not None and um is not None:
                cells[j] = _f(value * um, 3).replace(".", ",")
        out.append(";".join(cells))
        n += 1
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    return n


_CONVERTERS = {"events": convert_events, "fourway": convert_fourway, "wide": convert_wide}


def write_um_files(
    t1_dir: Path | str,
    stem: str,
    cal: Calibration | None,
    *,
    log=print,
) -> list[Path]:
    """Write ``*_um.csv`` twins for every T1 table present in ``t1_dir``.

    Needs both a pixel size and a frame interval to make a meaningful second
    file; with only one of the two, the converted columns are limited to what
    is known. With no calibration nothing is written.
    """
    t1_dir = Path(t1_dir)
    if cal is None or cal.is_empty:
        log(f"No calibration found for {stem}: only pixel/frame tables written.")
        return []
    written: list[Path] = []
    for role, kind in TABLES.items():
        src = t1_dir / f"{stem}_{role}"
        if not src.is_file():
            continue
        dst = _um_name(src)
        rows = _CONVERTERS[kind](src, dst, cal)
        written.append(dst)
        log(f"Wrote {dst.name} ({rows} rows)")
    qc_path = t1_dir / f"{stem}_qc.json"
    if qc_path.is_file():
        try:
            qc = json.loads(qc_path.read_text())
            qc["calibration"] = cal.as_dict()
            qc_path.write_text(json.dumps(qc, indent=2) + "\n")
        except ValueError:
            pass
    missing = [n for n, v in (("pixel size", cal.has_scale), ("frame interval", cal.has_time)) if not v]
    if missing:
        log(f"Warning: calibration lacks {' and '.join(missing)}; those columns were left out.")
    return written


def write_um_files_for_labels(t1_dir: Path | str, labels: Path | str, *, log=print) -> list[Path]:
    """Convenience for the T1 scripts: calibration from the labels TIFF (else
    the raw movie of the same stem), then :func:`write_um_files`."""
    stem = input_stem(labels)
    cal = calibration_for(labels, stem=stem)
    return write_um_files(t1_dir, stem, cal, log=log)
