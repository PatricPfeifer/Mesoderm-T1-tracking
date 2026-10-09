#!/usr/bin/env bash
# Install / refresh CUDA + cuDNN 8 + NVCC pip libs for the TA env so
# TensorFlow 2.15 can use a GPU on bwHPC (runtime + ptxas/libdevice for JIT).
# Safe to re-run.
set -euo pipefail

ROOT="/home/ho/ho_proana/ho_luchi75/projects/patric"
cd "$ROOT"
source /home/ho/ho_proana/ho_luchi75/miniconda3/etc/profile.d/conda.sh
conda activate TA

echo "Installing NVIDIA CUDA 12 + cuDNN 8.9 + cuda_nvcc wheels into env: $CONDA_PREFIX"
python -m pip install -q \
  "nvidia-cublas-cu12" \
  "nvidia-cuda-cupti-cu12" \
  "nvidia-cuda-nvcc-cu12" \
  "nvidia-cuda-nvrtc-cu12" \
  "nvidia-cuda-runtime-cu12" \
  "nvidia-cudnn-cu12==8.9.7.29" \
  "nvidia-cufft-cu12" \
  "nvidia-curand-cu12" \
  "nvidia-cusolver-cu12" \
  "nvidia-cusparse-cu12" \
  "nvidia-nccl-cu12" \
  "nvidia-nvjitlink-cu12" \
  "nvidia-nvtx-cu12"

# Optional cluster CUDA toolkit (drivers still come from the node).
if command -v module >/dev/null 2>&1; then
  module load devel/cuda/12.8 2>/dev/null || module load devel/cuda/11.8 2>/dev/null || true
fi

python - <<'PY'
from ta_common import (
    ensure_tf_cuda_lib_path,
    find_cuda_nvcc_data_dir,
    nvidia_pip_lib_dirs,
)

dirs = ensure_tf_cuda_lib_path()
print("nvidia lib dirs:", len(dirs))
cudnn = [d / "libcudnn.so.8" for d in dirs if (d / "libcudnn.so.8").exists()]
print("libcudnn.so.8:", cudnn[0] if cudnn else "MISSING")
nvcc = find_cuda_nvcc_data_dir()
print("cuda_nvcc data dir:", nvcc if nvcc else "MISSING")
if nvcc:
    print("libdevice:", list((nvcc / "nvvm" / "libdevice").glob("libdevice*.bc"))[:3])
    print("ptxas:", nvcc / "bin" / "ptxas", "exists" if (nvcc / "bin" / "ptxas").is_file() else "MISSING")

import tensorflow as tf

print("TF", tf.__version__)
print("GPUs", tf.config.list_physical_devices("GPU"))
PY

echo "setup_ta_gpu.sh done. On a GPU allocation, re-run segmentation without --force-cpu"
echo "  and without clearing CUDA_VISIBLE_DEVICES."
