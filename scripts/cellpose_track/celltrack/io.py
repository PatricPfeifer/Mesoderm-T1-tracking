"""Load and validate Cellpose label + intensity stacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


@dataclass
class Movie:
    """Validated stacks with blank frames removed.

    Attributes:
        masks: (T, Y, X) integer labels, labels independent per frame.
        intensity: (T, Y, X) float32 registration images, or None.
        frame_indices: original frame indices into the on-disk stacks.
    """

    masks: np.ndarray
    intensity: np.ndarray | None
    frame_indices: np.ndarray

def load_stack(path: Path) -> np.ndarray:
    """Load a TIFF/NPY as T,Y,X. Multichannel stacks keep the first channel."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        img = np.asarray(np.load(path))
    else:
        img = np.asarray(tifffile.imread(path))

    if img.ndim == 2:
        img = img[np.newaxis, ...]
    if img.ndim == 4:
        if img.shape[-1] <= 4:
            img = img[..., 0]
        else:
            img = img[:, 0]
    if img.ndim != 3:
        raise ValueError(f"Expected T,Y,X stack; got shape {img.shape} for {path}")
    return img


def is_blank_frame(labels: np.ndarray, intensity: np.ndarray | None = None) -> bool:
    """True if the frame has no cells, or intensity is all zeros."""
    if int(labels.max()) == 0:
        return True
    if intensity is not None and float(np.max(intensity)) == 0.0:
        return True
    return False


def load_movie(
    masks_path: Path,
    intensity_path: Path | None = None,
    *,
    max_frames: int | None = None,
    drop_blank: bool = True,
) -> Movie:
    """Load masks (+ optional intensity), drop blank trailing/interior frames.

    Preserves original frame indices in ``Movie.frame_indices`` so all tables
    stay aligned with the on-disk stacks.
    """
    masks = load_stack(Path(masks_path))
    intensity: np.ndarray | None = None
    if intensity_path is not None:
        intensity = load_stack(Path(intensity_path)).astype(np.float32, copy=False)
        if intensity.shape != masks.shape:
            raise ValueError(
                f"Intensity shape {intensity.shape} does not match masks "
                f"{masks.shape}."
            )

    if max_frames is not None:
        masks = masks[:max_frames]
        if intensity is not None:
            intensity = intensity[:max_frames]

    n = masks.shape[0]
    keep: list[int] = []
    for t in range(n):
        inten_t = intensity[t] if intensity is not None else None
        if drop_blank and is_blank_frame(masks[t], inten_t):
            print(f"Dropping blank frame {t}", flush=True)
            continue
        keep.append(t)

    if len(keep) < 2:
        raise ValueError(
            f"Need at least 2 non-blank frames; kept {len(keep)} of {n}."
        )

    idx = np.asarray(keep, dtype=np.int32)
    masks_kept = masks[idx]
    intensity_kept = intensity[idx] if intensity is not None else None
    print(
        f"Loaded movie: {len(idx)}/{n} frames, shape={masks_kept.shape}, "
        f"dtype={masks_kept.dtype}",
        flush=True,
    )
    return Movie(masks=masks_kept, intensity=intensity_kept, frame_indices=idx)
