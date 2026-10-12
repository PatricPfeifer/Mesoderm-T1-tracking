#!/usr/bin/env python3
"""Render a tracking movie: raw + tracked labels (colour = track ID) with ID text.

  python scripts/cellpose_track/make_track_movie.py \
    --raw raw.tif --labels *_tracked_labels.tif --output track.mp4
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from tifffile import imread


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True, help="T,Y,X tracked labels.")
    p.add_argument("--output", type=Path, required=True, help=".mp4 or .gif")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--scale", type=float, default=3.0)
    p.add_argument("--alpha", type=float, default=0.35)
    a = p.parse_args()

    raw = imread(a.raw).astype(float)
    lab = imread(a.labels)
    n = min(len(raw), len(lab))
    lo, hi = np.percentile(raw, (1, 99.5))
    raw = np.clip((raw - lo) / (hi - lo), 0, 1)

    tmp = Path(tempfile.mkdtemp())
    h, w = lab.shape[1:]
    for t in range(n):
        fig = plt.figure(figsize=(w * a.scale / 100, h * a.scale / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        rgb = label2rgb(lab[t], image=raw[t], alpha=a.alpha, bg_label=0)
        rgb[find_boundaries(lab[t], mode="inner")] = 1
        ax.imshow(rgb, interpolation="nearest")
        for r in regionprops(lab[t]):
            y, x = r.centroid
            ax.text(x, y, str(r.label), color="white", fontsize=7, ha="center",
                    va="center", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.45, lw=0))
        ax.text(4, 12, f"frame {t}", color="yellow", fontsize=10, weight="bold")
        ax.axis("off")
        fig.savefig(tmp / f"f{t:04d}.png")
        plt.close(fig)

    out = a.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = shutil.which("ffmpeg")
    if out.suffix.lower() == ".gif" or not ff:
        import imageio.v2 as imageio
        imageio.mimsave(out.with_suffix(".gif"), [imageio.imread(tmp / f"f{t:04d}.png")
                        for t in range(n)], duration=1000 / a.fps, loop=0)
    else:
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(a.fps),
                        "-i", str(tmp / "f%04d.png"), "-vf",
                        "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", str(out)],
                       check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
