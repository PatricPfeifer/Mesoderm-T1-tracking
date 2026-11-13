"""Motion fields between frames: dense optical flow (default) and the legacy pyramidal phase-correlation tiles (TA registration)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from skimage.registration import phase_cross_correlation

from .parallel import pmap


def _phase_shift(ref: np.ndarray, mov: np.ndarray) -> tuple[float, float]:
    """Sub-pixel translation from ref to mov."""
    out = phase_cross_correlation(ref, mov)
    shift = out[0] if isinstance(out, tuple) else out
    return float(shift[0]), float(shift[1])


def shifted_crop(
    img: np.ndarray, y: int, x: int, height: int, width: int, dy: int, dx: int
) -> np.ndarray:
    """img[y+dy:y+dy+h, x+dx:x+dx+w], zero-padded outside."""
    out = np.zeros((height, width), dtype=img.dtype)
    src_y0, src_x0 = max(0, y + dy), max(0, x + dx)
    src_y1 = min(img.shape[0], y + dy + height)
    src_x1 = min(img.shape[1], x + dx + width)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out
    dst_y0, dst_x0 = max(0, -(y + dy)), max(0, -(x + dx))
    out[
        dst_y0 : dst_y0 + (src_y1 - src_y0),
        dst_x0 : dst_x0 + (src_x1 - src_x0),
    ] = img[src_y0:src_y1, src_x0:src_x1]
    return out


@dataclass
class MotionField:
    """Coarse tile-grid translation field, interpolable to dense H,W,2.

    ``grid_dy``, ``grid_dx`` have shape (n_tiles_y, n_tiles_x).
    ``tile`` is the finest-level tile size used to build the grid.
    """

    grid_dy: np.ndarray
    grid_dx: np.ndarray
    tile: int
    shape_yx: tuple[int, int]

    def dense(self) -> np.ndarray:
        """Return H,W,2 float32 field: [...,0]=dy, [...,1]=dx (t0 -> t1)."""
        height, width = self.shape_yx
        nty, ntx = self.grid_dy.shape
        field = np.zeros((height, width, 2), dtype=np.float32)
        for iy in range(nty):
            y0 = iy * self.tile
            y1 = min(height, y0 + self.tile)
            for ix in range(ntx):
                x0 = ix * self.tile
                x1 = min(width, x0 + self.tile)
                field[y0:y1, x0:x1, 0] = self.grid_dy[iy, ix]
                field[y0:y1, x0:x1, 1] = self.grid_dx[iy, ix]
        return field

    def sample(self, y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample (dy, dx) at float/int coordinates."""
        iy = np.clip(np.asarray(y, dtype=np.int32) // self.tile, 0, self.grid_dy.shape[0] - 1)
        ix = np.clip(np.asarray(x, dtype=np.int32) // self.tile, 0, self.grid_dx.shape[1] - 1)
        return self.grid_dy[iy, ix], self.grid_dx[iy, ix]


def estimate_motion(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    *,
    depth: int = 3,
    block_size: int = 256,
    median_size: int = 3,
) -> MotionField:
    """Estimate a regularised pyramidal translation field from t0 to t1.

    Starts from one global rigid shift, then refines on shrinking tiles
    (256 / 128 / 64 for depth=3). Unlike TA, bad tile shifts are not
    hard-rejected; the tile grid is median-filtered at each level so a
    single outlier cannot create a large regional jump.
    """
    img_t0 = np.asarray(img_t0, dtype=np.float32)
    img_t1 = np.asarray(img_t1, dtype=np.float32)
    height, width = img_t0.shape[:2]

    global_y, global_x = _phase_shift(img_t0, img_t1)
    # TA convention: translation that maps t0 positions onto t1
    base_dy = -global_y
    base_dx = -global_x

    # Accumulate on the finest tile grid.
    finest = max(block_size // (2 ** max(depth - 1, 0)), 16)
    nty = int(np.ceil(height / finest))
    ntx = int(np.ceil(width / finest))
    grid_dy = np.full((nty, ntx), base_dy, dtype=np.float32)
    grid_dx = np.full((nty, ntx), base_dx, dtype=np.float32)

    for level in range(depth):
        tile = max(block_size // (2**level), 16)
        level_nty = int(np.ceil(height / tile))
        level_ntx = int(np.ceil(width / tile))
        level_dy = np.zeros((level_nty, level_ntx), dtype=np.float32)
        level_dx = np.zeros((level_nty, level_ntx), dtype=np.float32)

        for iy in range(level_nty):
            y = iy * tile
            for ix in range(level_ntx):
                x = ix * tile
                # Current estimate at this tile (from coarser / previous).
                fy = min(y // finest, nty - 1)
                fx = min(x // finest, ntx - 1)
                cur_dy = float(grid_dy[fy, fx])
                cur_dx = float(grid_dx[fy, fx])

                th = min(tile, height - y)
                tw = min(tile, width - x)
                tile_t1 = img_t1[y : y + th, x : x + tw]
                tile_t0 = shifted_crop(
                    img_t0,
                    y,
                    x,
                    th,
                    tw,
                    int(round(-cur_dy)),
                    int(round(-cur_dx)),
                )
                if tile_t0.size == 0 or tile_t1.size == 0:
                    continue
                if tile_t0.std() == 0 or tile_t1.std() == 0:
                    continue

                dy, dx = _phase_shift(tile_t0, tile_t1)
                level_dy[iy, ix] = -dy
                level_dx[iy, ix] = -dx

        # Median-filter the residual grid, then add onto the fine grid.
        if median_size > 1 and level_nty >= median_size and level_ntx >= median_size:
            level_dy = median_filter(level_dy, size=median_size)
            level_dx = median_filter(level_dx, size=median_size)

        for iy in range(nty):
            y = iy * finest
            liy = min(y // tile, level_nty - 1)
            for ix in range(ntx):
                x = ix * finest
                lix = min(x // tile, level_ntx - 1)
                grid_dy[iy, ix] += level_dy[liy, lix]
                grid_dx[iy, ix] += level_dx[liy, lix]

        if median_size > 1 and nty >= median_size and ntx >= median_size:
            grid_dy = median_filter(grid_dy, size=median_size)
            grid_dx = median_filter(grid_dx, size=median_size)

    return MotionField(
        grid_dy=grid_dy,
        grid_dx=grid_dx,
        tile=finest,
        shape_yx=(height, width),
    )


class FlowMotionField:
    """Dense optical-flow motion field t0 -> t1 (every pixel its own displacement, not one shift per tile).

    ``fwd`` is the flow at the t0 positions (where a t0 point goes), used to move centroids; ``bwd`` is the flow found from t1 back to t0, so
    the pull-warp of ``costs.warp_labels`` (source = destination - dense) reads exactly the right t0 pixel. Same interface as ``MotionField``.

    Flow is smooth (it comes from 31 px windows), so both fields are kept at half the resolution as float16 and enlarged when used: a
    movie of 100+ full-size frames would otherwise hold several GB of flow (the fields of every frame pair are kept until tracking is done).
    """

    def __init__(self, fwd: np.ndarray, bwd: np.ndarray):
        import cv2

        self.shape_yx = fwd.shape[:2]
        small = (max(1, (self.shape_yx[1] + 1) // 2), max(1, (self.shape_yx[0] + 1) // 2))  # cv2 size = (width, height)
        self._fwd = cv2.resize(np.ascontiguousarray(fwd, dtype=np.float32), small, interpolation=cv2.INTER_AREA).astype(np.float16)
        self._bwd = cv2.resize(np.ascontiguousarray(bwd, dtype=np.float32), small, interpolation=cv2.INTER_AREA).astype(np.float16)
        self.grid_dy, self.grid_dx = self._fwd[..., 0], self._fwd[..., 1]  # for the progress print

    def dense(self) -> np.ndarray:
        import cv2

        h, w = self.shape_yx
        return -cv2.resize(self._bwd.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    def sample(self, y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sh, sw = self._fwd.shape[:2]
        iy = np.clip(np.rint(np.asarray(y) / 2.0 - 0.25).astype(np.int32), 0, sh - 1)
        ix = np.clip(np.rint(np.asarray(x) / 2.0 - 0.25).astype(np.int32), 0, sw - 1)
        return self._fwd[iy, ix, 0].astype(np.float32), self._fwd[iy, ix, 1].astype(np.float32)


def _flow_image(img: np.ndarray) -> np.ndarray:
    """8 bit, lightly smoothed picture for the flow (stretched per frame so a dim frame still has contrast)."""
    import cv2

    img = cv2.GaussianBlur(np.asarray(img, dtype=np.float32), (0, 0), 1.5)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def estimate_flow(img_t0: np.ndarray, img_t1: np.ndarray, levels: int = 5, winsize: int = 31) -> FlowMotionField:
    """Farneback optical flow between two frames (forward and backward) as a ``FlowMotionField``.

    Follows tissue that moves non-uniformly (convergent-extension flow) where one shift per tile fails: the cells of a frame are moved
    with it before they are matched, so they overlap their own next-frame cell again."""
    import cv2

    a, b = _flow_image(img_t0), _flow_image(img_t1)
    kw = dict(pyr_scale=0.5, levels=levels, winsize=winsize, iterations=5, poly_n=7, poly_sigma=1.5, flags=0)
    fwd = cv2.calcOpticalFlowFarneback(a, b, None, **kw)  # (...,0)=dx, (...,1)=dy: pixel (y,x) of a is at (y+dy, x+dx) in b
    bwd = cv2.calcOpticalFlowFarneback(b, a, None, **kw)
    return FlowMotionField(fwd[..., ::-1], bwd[..., ::-1])  # to (dy, dx)


def registration_images(
    masks: np.ndarray, intensity: np.ndarray | None
) -> np.ndarray:
    """Images used for phase correlation.

    Prefer raw intensity; fall back to binary masks if intensity is missing.
    """
    if intensity is not None:
        return intensity.astype(np.float32, copy=False)
    print(
        "No intensity: registering on binary masks (less accurate).",
        flush=True,
    )
    return (masks > 0).astype(np.float32)


def estimate_all_motion(
    reg_images: np.ndarray,
    *,
    depth: int = 3,
    block_size: int = 256,
    median_size: int = 3,
    n_jobs: int = 1,
    on_progress=None,
    motion: str = "flow",
) -> list[MotionField]:
    """Motion fields for consecutive pairs: fields[t] maps frame t -> t+1.

    ``motion``: "flow" = dense optical flow (default; follows non-uniform tissue motion), "phase" = the legacy pyramidal
    phase-correlation tiles (one shift per tile).

    Pairs are independent, so ``n_jobs`` > 1 (or -1 for all cores) runs them
    in parallel with identical results.
    """

    def one(t: int) -> MotionField:
        return estimate_motion(
            reg_images[t],
            reg_images[t + 1],
            depth=depth,
            block_size=block_size,
            median_size=median_size,
        )

    if motion == "flow":
        one = lambda t: estimate_flow(reg_images[t], reg_images[t + 1])  # noqa: E731
    fields = pmap(one, range(reg_images.shape[0] - 1), n_jobs, on_progress)
    for t, field in enumerate(fields):
        if (t + 1) % 20 == 0 or t == 0:
            print(
                f"  motion {t}->{t + 1}: "
                f"mean_shift=({field.grid_dy.mean():.2f}, {field.grid_dx.mean():.2f})",
                flush=True,
            )
    return fields
