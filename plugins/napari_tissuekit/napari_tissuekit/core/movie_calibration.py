"""Pixel size and frame interval of a movie, carried through every pipeline step.

Raw movies are ImageJ TIFFs: ``XResolution``/``YResolution`` = pixels per µm
(unit "micron") and ``finterval`` = seconds per frame. :func:`read_calibration`
reads that (or the same information written by :func:`save_tiff`), and
:func:`save_tiff` writes it into every file a step produces, so the next step
can read it from the file it receives. Missing values stay ``None``; nothing is
ever guessed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import tifffile

_SEARCH_DIRS = ("input", "input files")

_UM_UNITS = {"micron", "microns", "um", "µm", "μm", "micrometer", "micrometre"}
_TIME_TO_S = {
    "": 1.0, "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
    "ms": 1e-3, "min": 60.0, "minute": 60.0, "minutes": 60.0, "h": 3600.0,
}


@dataclass(frozen=True)
class Calibration:
    um_per_px_x: float | None = None
    um_per_px_y: float | None = None
    s_per_frame: float | None = None
    source: str = ""

    @property
    def um_per_px(self) -> float | None:
        """Isotropic pixel size (x and y are checked equal on reading)."""
        return self.um_per_px_x if self.um_per_px_x is not None else self.um_per_px_y

    @property
    def has_scale(self) -> bool:
        return self.um_per_px is not None

    @property
    def has_time(self) -> bool:
        return self.s_per_frame is not None

    @property
    def is_empty(self) -> bool:
        return not (self.has_scale or self.has_time)

    def as_dict(self) -> dict:
        return {
            "um_per_px": self.um_per_px,
            "s_per_frame": self.s_per_frame,
            "source": self.source,
        }


def _res_to_um_per_px(value, unit_name: str, imagej_unit: str) -> float | None:
    """TIFF resolution tag value (num, den) -> µm per pixel, or None."""
    try:
        px_per_unit = float(value[0]) / float(value[1])
    except (TypeError, ZeroDivisionError, IndexError):
        return None
    if px_per_unit <= 0:
        return None
    unit_name = (unit_name or "").upper()
    if imagej_unit.lower() in _UM_UNITS or unit_name == "MICROMETER":
        return 1.0 / px_per_unit
    if unit_name == "CENTIMETER":
        return 1e4 / px_per_unit
    if unit_name == "MILLIMETER":
        return 1e3 / px_per_unit
    if unit_name == "INCH":
        return 25400.0 / px_per_unit
    return None  # unit unknown: do not guess


def read_calibration(path: Path | str) -> Calibration | None:
    """Calibration stored in a TIFF, or ``None`` if the file has none."""
    path = Path(path)
    if path.suffix.lower() not in (".tif", ".tiff") or not path.is_file():
        return None
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            tags = page.tags
            imagej = dict(tif.imagej_metadata or {}) if tif.is_imagej else {}
            description = page.description or ""
            ij_unit = str(imagej.get("unit", ""))
            json_meta: dict = {}
            if description.lstrip().startswith("{"):
                try:
                    json_meta = json.loads(description)
                except ValueError:
                    json_meta = {}
                ij_unit = ij_unit or str(json_meta.get("unit", ""))
            unit_tag = ""
            if "ResolutionUnit" in tags:
                unit_tag = getattr(tags["ResolutionUnit"].value, "name", str(tags["ResolutionUnit"].value))
            ux = uy = None
            if "XResolution" in tags:
                ux = _res_to_um_per_px(tags["XResolution"].value, unit_tag, ij_unit)
            if "YResolution" in tags:
                uy = _res_to_um_per_px(tags["YResolution"].value, unit_tag, ij_unit)
            interval = imagej.get("finterval", json_meta.get("finterval"))
            tunit = str(imagej.get("tunit", json_meta.get("tunit", ""))).lower()
    except Exception:
        return None
    if ux is not None and uy is not None and abs(ux - uy) > 1e-3 * max(ux, uy):
        raise ValueError(
            f"{path.name}: x and y pixel sizes differ ({ux:.5f} vs {uy:.5f} µm/px); "
            "anisotropic pixels are not supported."
        )
    seconds = None
    if interval is not None:
        try:
            seconds = float(interval) * _TIME_TO_S.get(tunit, 1.0)
        except (TypeError, ValueError):
            seconds = None
        if seconds is not None and seconds <= 0:
            seconds = None
    cal = Calibration(ux, uy, seconds, str(path))
    return None if cal.is_empty else cal


def _stem_of(path: Path) -> str:
    from .output_names import input_stem

    return input_stem(path)


def find_source_movie(stem: str, root: Path | None = None) -> Path | None:
    """``input/<stem>.tif`` (or ``input files/``) for a pipeline output stem."""
    base = Path(root) if root else Path.cwd()
    for folder in _SEARCH_DIRS:
        for ext in (".tif", ".tiff"):
            cand = base / folder / f"{stem}{ext}"
            if cand.is_file():
                return cand
    return None


def _merge(primary: Calibration | None, fallback: Calibration | None) -> Calibration | None:
    if primary is None:
        return fallback
    if fallback is None:
        return primary
    return Calibration(
        primary.um_per_px_x if primary.um_per_px_x is not None else fallback.um_per_px_x,
        primary.um_per_px_y if primary.um_per_px_y is not None else fallback.um_per_px_y,
        primary.s_per_frame if primary.s_per_frame is not None else fallback.s_per_frame,
        primary.source or fallback.source,
    )


def calibration_for(*paths: Path | str | None, stem: str | None = None) -> Calibration | None:
    """Calibration from the first given file that has one.

    Files are tried in order. Whatever is still missing afterwards is taken
    from the raw movie ``input/<stem>.tif`` (stem derived from the first path
    when not given), so results produced before calibration was carried along
    can still be converted.
    """
    cal: Calibration | None = None
    first: Path | None = None
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        first = first or p
        cal = _merge(cal, read_calibration(p))
        if cal is not None and cal.has_scale and cal.has_time:
            return cal
    if stem is None and first is not None:
        stem = _stem_of(first)
    if stem:
        source = find_source_movie(stem)
        if source is not None:
            cal = _merge(cal, read_calibration(source))
    return cal


def save_tiff(path: Path | str, data: np.ndarray, cal: Calibration | None = None, **kwargs) -> None:
    """``tifffile.imwrite`` that also stores the calibration.

    uint8 / uint16 / float32 stacks (T,Y,X) and RGB stacks (T,Y,X,3) are saved
    as ImageJ TIFFs so Fiji shows the scale and frame interval. Other types
    (e.g. uint32 track labels, which ImageJ files cannot hold) are saved as
    plain TIFFs with resolution tags in µm and ``finterval`` in a JSON
    description. Without a calibration this is a plain ``imwrite``.
    """
    path = Path(path)
    arr = np.asarray(data)
    if cal is None or cal.is_empty:
        tifffile.imwrite(path, arr, **kwargs)
        return
    meta: dict = {"unit": "micron"}
    if cal.has_time:
        meta["finterval"] = float(cal.s_per_frame)
    extra: dict = {}
    if cal.has_scale:
        ux = cal.um_per_px_x if cal.um_per_px_x is not None else cal.um_per_px
        uy = cal.um_per_px_y if cal.um_per_px_y is not None else cal.um_per_px
        extra["resolution"] = (1.0 / ux, 1.0 / uy)
    rgb = arr.ndim == 4 and arr.shape[-1] == 3 and arr.dtype == np.uint8
    if arr.dtype in (np.uint8, np.uint16, np.float32) and arr.ndim in (2, 3, 4):
        if rgb:
            meta["axes"] = "TYXS"
            extra["photometric"] = "rgb"
        elif arr.ndim == 3:
            meta["axes"] = "TYX"
        tifffile.imwrite(path, arr, imagej=True, metadata=meta, **extra, **kwargs)
    else:
        if cal.has_scale:
            extra["resolutionunit"] = "MICROMETER"
        tifffile.imwrite(path, arr, metadata=meta, **extra, **kwargs)


def scaled(cal: Calibration, **changes) -> Calibration:
    return replace(cal, **changes)
