"""Shared helpers for Tissue Analyzer (pyTA / EPySeg) segment and track scripts."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.segmentation import find_boundaries

DEFAULT_INPUT = Path(
    "/home/ho/ho_proana/ho_luchi75/projects/patric/input files/"
    "Cri_WT_GAP43_GFP_38sec_Naima_cut.tif"
)
PROJECT_ROOT = Path("/home/ho/ho_proana/ho_luchi75/projects/patric")
BG_COLORS = {0, 0xFFFFFF}


def outlines_rgb_to_labels(rgb: np.ndarray) -> np.ndarray:
    """Convert Cellpose RGB outline stack (T,Y,X,3) to instance labels (T,Y,X).

    Non-black pixels are treated as boundaries. Connected components of the
    remaining pixels become cells; the largest component is background.
    """
    from skimage.measure import label as cc_label

    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        rgb = rgb[np.newaxis, ...]
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected T,Y,X,3 RGB outlines; got {rgb.shape}.")

    out = np.zeros(rgb.shape[:3], dtype=np.uint32)
    for t in range(rgb.shape[0]):
        interior = rgb[t].max(axis=-1) == 0
        lab = cc_label(interior, connectivity=1)
        counts = np.bincount(lab.ravel())
        if counts.size <= 1:
            continue
        counts[0] = 0
        bg = int(counts.argmax())
        lab[lab == bg] = 0
        uniq = np.unique(lab)
        uniq = uniq[uniq != 0]
        remapped = np.zeros_like(lab, dtype=np.uint32)
        for i, u in enumerate(uniq, start=1):
            remapped[lab == u] = i
        out[t] = remapped
    return out


def load_stack(path: Path, *, allow_rgb_outlines: bool = False) -> np.ndarray:
    img = np.asarray(tifffile.imread(path))
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    if allow_rgb_outlines and img.ndim == 4 and img.shape[-1] == 3:
        print(
            f"Detected RGB outline stack {img.shape}; converting to labels...",
            flush=True,
        )
        img = outlines_rgb_to_labels(img)
        print(
            f"Converted outlines → labels shape={img.shape} max_label={int(img.max())}",
            flush=True,
        )
    if img.ndim != 3:
        raise ValueError(
            f"Expected T,Y,X (or Y,X) stack; got shape {img.shape}."
        )
    return img


def write_frames(stack: np.ndarray, frames_dir: Path) -> list[str]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for t, frame in enumerate(stack):
        out = frames_dir / f"t{t:04d}.tif"
        tifffile.imwrite(out, frame)
        paths.append(str(out.resolve()))
    return paths


def labels_to_hand_correction(labels: np.ndarray) -> np.ndarray:
    """Convert instance labels (0=bg) to TA handCorrection (255=bg/bonds)."""
    hand = np.full(labels.shape, 255, dtype=np.uint8)
    hand[labels > 0] = 0
    hand[find_boundaries(labels, mode="outer")] = 255
    return hand


def write_hand_corrections(masks: np.ndarray, frame_paths: list[str]) -> None:
    if masks.shape[0] != len(frame_paths):
        raise ValueError(
            f"Mask frames ({masks.shape[0]}) != intensity frames ({len(frame_paths)})."
        )
    for path, lab in zip(frame_paths, masks):
        stem_dir = Path(path).with_suffix("")
        stem_dir.mkdir(parents=True, exist_ok=True)
        hand = labels_to_hand_correction(lab)
        tifffile.imwrite(stem_dir / "handCorrection.tif", hand)


def collect_frame_paths(frames_dir: Path) -> list[str]:
    """Ordered list of intensity frame TIFFs under frames/ (not handCorrection)."""
    from epyseg.utils.loadlist import create_list

    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    paths = create_list(
        str(frames_dir),
        save=False,
        extensions=["*.tif", "*.tiff"],
        sort_type="natsort",
        recursive=False,
    )
    # Keep only top-level frame files (t0000.tif), not nested masks.
    frame_paths = [
        p
        for p in paths
        if Path(p).parent.resolve() == frames_dir.resolve()
    ]
    if not frame_paths:
        raise FileNotFoundError(
            f"No frame TIFFs found in {frames_dir} (expected t0000.tif, ...)."
        )
    return [str(Path(p).resolve()) for p in frame_paths]


def validate_hand_corrections(frame_paths: list[str]) -> None:
    missing = [
        p
        for p in frame_paths
        if not (Path(p).with_suffix("") / "handCorrection.tif").exists()
    ]
    if missing:
        example = Path(missing[0]).with_suffix("") / "handCorrection.tif"
        raise FileNotFoundError(
            f"Missing handCorrection.tif for {len(missing)}/{len(frame_paths)} "
            f"frames (e.g. {example}). Run run_ta_segment.py first."
        )


def write_list_lst(frame_paths: list[str], frames_dir: Path) -> Path:
    from epyseg.utils.loadlist import save_list_to_file

    list_path = frames_dir / "list.lst"
    save_list_to_file(frame_paths, str(list_path))
    return list_path


# TF 2.15 needs cuDNN 8 (+ CUDA 12 runtime) and cuda_nvcc (ptxas + libdevice) for JIT.
_TA_GPU_PIP_PACKAGES = [
    "nvidia-cublas-cu12",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvcc-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12==8.9.7.29",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-nvtx-cu12",
]


def nvidia_pip_lib_dirs() -> list[Path]:
    """Return site-packages/nvidia/*/lib dirs if the pip CUDA wheels are installed."""
    import site

    dirs: list[Path] = []
    for sp in site.getsitepackages():
        nvidia = Path(sp) / "nvidia"
        if not nvidia.is_dir():
            continue
        for child in sorted(nvidia.iterdir()):
            lib = child / "lib"
            if lib.is_dir():
                dirs.append(lib)
    return dirs


def find_cuda_nvcc_data_dir() -> Path | None:
    """Return CUDA data dir that contains nvvm/libdevice (pip cuda_nvcc or module toolkit)."""
    import os
    import site

    candidates: list[Path] = []
    for sp in site.getsitepackages():
        candidates.append(Path(sp) / "nvidia" / "cuda_nvcc")
    for key in ("CUDA_HOME", "CUDA_PATH", "CUDA_DIR"):
        val = os.environ.get(key)
        if val:
            candidates.append(Path(val))
    for p in (
        Path("/usr/local/cuda-12.2"),
        Path("/usr/local/cuda-12"),
        Path("/usr/local/cuda"),
    ):
        candidates.append(p)

    for root in candidates:
        libdevice = root / "nvvm" / "libdevice"
        if libdevice.is_dir() and any(libdevice.glob("libdevice*.bc")):
            return root
    return None


def ensure_tf_xla_cuda_toolkit() -> Path | None:
    """Put ptxas on PATH and set XLA_FLAGS so TF can JIT PTX (needed on H100/A100).

    Without nvidia-cuda-nvcc-cu12 (or a full CUDA toolkit), predict fails with
    'Failed to launch ptxas' / 'libdevice not found'. Safe to call multiple times.
    """
    import os
    import shutil

    root = find_cuda_nvcc_data_dir()
    if root is None:
        print(
            "GPU setup: cuda_nvcc / libdevice not found. "
            "Run: bash setup_ta_gpu.sh (installs nvidia-cuda-nvcc-cu12).",
            flush=True,
        )
        return None

    bin_dir = root / "bin"
    ptxas = bin_dir / "ptxas"
    if ptxas.is_file():
        path = os.environ.get("PATH", "")
        bin_s = str(bin_dir)
        if bin_s not in path.split(":"):
            os.environ["PATH"] = f"{bin_s}:{path}" if path else bin_s
    elif shutil.which("ptxas") is None:
        print(
            f"GPU setup: ptxas missing under {bin_dir} and not on PATH.",
            flush=True,
        )

    # Drop a broken placeholder if a wrapper set /dev/null earlier.
    xla = os.environ.get("XLA_FLAGS", "")
    if "xla_gpu_cuda_data_dir=/dev/null" in xla:
        parts = [p for p in xla.split() if "xla_gpu_cuda_data_dir=" not in p]
        os.environ["XLA_FLAGS"] = " ".join(parts).strip()
        xla = os.environ["XLA_FLAGS"]

    flag = f"--xla_gpu_cuda_data_dir={root}"
    if "xla_gpu_cuda_data_dir=" not in xla:
        os.environ["XLA_FLAGS"] = f"{xla} {flag}".strip() if xla else flag
    else:
        # Prefer our resolved root over a stale path.
        parts = []
        for p in xla.split():
            if p.startswith("--xla_gpu_cuda_data_dir="):
                parts.append(flag)
            else:
                parts.append(p)
        os.environ["XLA_FLAGS"] = " ".join(parts)

    print(
        f"GPU setup: XLA cuda data dir={root} "
        f"ptxas={'yes' if (bin_dir / 'ptxas').is_file() or shutil.which('ptxas') else 'no'}",
        flush=True,
    )
    return root


def ensure_tf_cuda_lib_path() -> list[Path]:
    """Prepend pip NVIDIA CUDA libs to LD_LIBRARY_PATH before importing TensorFlow.

    On bwHPC, TF often cannot see the GPU unless these wheel libs (esp. cuDNN 8)
    are on LD_LIBRARY_PATH. Also configures XLA/ptxas for JIT. Safe to call again.
    """
    import os

    dirs = nvidia_pip_lib_dirs()
    if not dirs:
        print(
            "GPU setup: no pip nvidia/*/lib dirs found in the TA env. "
            "Run: bash setup_ta_gpu.sh",
            flush=True,
        )
    else:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [str(d) for d in dirs]
        # Keep existing entries after our nvidia paths.
        if existing:
            for p in existing.split(":"):
                if p and p not in parts:
                    parts.append(p)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
        cudnn8 = any((d / "libcudnn.so.8").exists() for d in dirs)
        if not cudnn8:
            print(
                "GPU setup: libcudnn.so.8 missing (TF 2.15 needs cuDNN 8). "
                "Run: bash setup_ta_gpu.sh",
                flush=True,
            )
        else:
            print(
                f"GPU setup: LD_LIBRARY_PATH includes {len(dirs)} nvidia pip lib dirs "
                "(cuDNN 8 present).",
                flush=True,
            )

    ensure_tf_xla_cuda_toolkit()
    return dirs


def detect_use_cpu(force_cpu: bool) -> tuple[bool, list]:
    """Return (use_cpu, gpu_device_list). Import TF here before EZDeepLearning."""
    if not force_cpu:
        ensure_tf_cuda_lib_path()

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    use_cpu = bool(force_cpu) or (len(gpus) == 0)
    if force_cpu:
        print("GPU check: --force-cpu set; using CPU for EPySeg.", flush=True)
    elif gpus:
        names = [getattr(g, "name", str(g)) for g in gpus]
        print(f"GPU available ({len(gpus)}): {names}; using GPU.", flush=True)
    else:
        print(
            "No GPU detected by TensorFlow; using CPU for EPySeg. "
            "If you are on a GPU node, run: bash setup_ta_gpu.sh "
            "and ensure CUDA_VISIBLE_DEVICES is set (do not clear it).",
            flush=True,
        )
    return use_cpu, list(gpus)


def run_epyseg_segmentation(
    frame_paths: list[str],
    *,
    use_cpu: bool,
    channel: int | None,
    tile_size: int,
    tile_overlap: int,
    size_filter: int | None,
) -> None:
    from epyseg.deeplearning.deepl import EZDeepLearning
    from epyseg.ta.deep.fixed_script_yonit import run_segmentation

    deepTA = EZDeepLearning(use_cpu=use_cpu)
    try:
        run_segmentation(
            deepTA=deepTA,
            INPUT_FOLDER=frame_paths,
            IS_TA_OUTPUT_MODE=True,
            input_channel_of_interest=channel,
            TILE_WIDTH=tile_size,
            TILE_HEIGHT=tile_size,
            TILE_OVERLAP=tile_overlap,
            EPYSEG_PRETRAINING="Linknet-vgg16-sigmoid-v2",
            SIZE_FILTER=size_filter,
        )
    finally:
        del deepTA
        gc.collect()


def launch_correction_gui(frame_paths: list[str]) -> int:
    """Open pyTA on the Segmentation tab with frames preloaded for mask edits.

    Returns the Qt application exit code (does not call sys.exit).
    """
    import os

    display = os.environ.get("DISPLAY")
    if not display:
        raise RuntimeError(
            "No DISPLAY set. pyTA GUI needs X11/VNC forwarding "
            "(echo $DISPLAY). Use --skip-gui for headless segmentation only."
        )
    print(
        "Launching pyTA Segmentation GUI.\n"
        "  Edit bonds/masks, then Enter / Ctrl+S to save.\n"
        "  Do not run Tracking in the GUI — close when done, then run "
        "run_ta_track.py.",
        flush=True,
    )
    from epyseg.settings.global_settings import set_UI

    set_UI()
    from epyseg.ta.GUI.pyta import TissueAnalyzer
    from qtpy.QtWidgets import QApplication

    app = QApplication(sys.argv)
    w = TissueAnalyzer()
    w.list.get_list(1).add_to_list(frame_paths)
    w.tabs.setCurrentIndex(1)
    w.list.set_list(1)
    w.list.get_list(1).list.setCurrentRow(0)
    w.selectionChanged()
    w.show()
    return int(app.exec_())


def run_tracking(frame_paths: list[str], channel: int | None) -> None:
    from epyseg.ta.tracking.local_to_track_correspondance import (
        add_localID_to_trackID_correspondance_in_DB,
    )
    from epyseg.ta.tracking.tracking_yet_another_approach_pyramidal_registration_n_neo_swapping_correction import (
        track_cells_dynamic_tissue,
    )

    print("Tracking cells (dynamic tissue)...", flush=True)
    track_cells_dynamic_tissue(frame_paths, channel=channel)
    print("Building local-ID ↔ track-ID correspondence...", flush=True)
    add_localID_to_trackID_correspondance_in_DB(frame_paths)


def rgb24_to_dense_ids(rgb: np.ndarray) -> np.ndarray:
    """Map RGB24 track colors to dense uint32 labels (0 = background)."""
    from epyseg.img import RGB_to_int24

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected H,W,3 RGB track image; got {rgb.shape}")
    codes = RGB_to_int24(rgb)
    unique = np.unique(codes)
    keep = [int(c) for c in unique if int(c) not in BG_COLORS]
    mapping = {c: i + 1 for i, c in enumerate(keep)}
    out = np.zeros(codes.shape, dtype=np.uint32)
    for c, idx in mapping.items():
        out[codes == c] = idx
    return out


def export_track_stacks(
    frame_paths: list[str], output_dir: Path
) -> tuple[Path, Path]:
    rgb_frames: list[np.ndarray] = []
    id_frames: list[np.ndarray] = []

    for path in frame_paths:
        stem_dir = Path(path).with_suffix("")
        track_path = stem_dir / "tracked_cells_resized.tif"
        if not track_path.exists():
            raise FileNotFoundError(
                f"Missing track image after tracking: {track_path}"
            )
        rgb = np.asarray(tifffile.imread(track_path))
        if rgb.ndim == 2:
            raise ValueError(
                f"Expected RGB tracked_cells_resized.tif at {track_path}, "
                f"got shape {rgb.shape}"
            )
        if rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        rgb_frames.append(rgb.astype(np.uint8, copy=False))
        id_frames.append(rgb24_to_dense_ids(rgb))

    rgb_stack = np.stack(rgb_frames, axis=0)
    id_stack = np.stack(id_frames, axis=0)

    rgb_out = output_dir / "tracked_cells_rgb.tif"
    ids_out = output_dir / "tracked_cells_ids.tif"
    tifffile.imwrite(rgb_out, rgb_stack)
    tifffile.imwrite(ids_out, id_stack)
    return rgb_out, ids_out
