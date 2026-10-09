#!/usr/bin/env bash
# Install Miniforge as ~/miniconda3 and create HPC envs used by project scripts.
# Must run on a Slurm compute allocation (not a login node).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=require_compute.sh
source "$ROOT/scripts/require_compute.sh"

PREFIX="${CONDA_PREFIX_ROOT:-$HOME/miniconda3}"
INSTALLER="${TMPDIR:-/tmp}/Miniforge3-Linux-x86_64.sh"
LOG="${ROOT}/output/setup_conda_envs.log"
mkdir -p "${ROOT}/output"

exec > >(tee -a "$LOG") 2>&1
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-none} $(date -Is)"
echo "PREFIX=$PREFIX ROOT=$ROOT"

if [[ ! -x "${PREFIX}/bin/conda" ]]; then
  echo "Downloading Miniforge..."
  curl -fsSL -o "$INSTALLER" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "$INSTALLER" -b -p "$PREFIX"
  rm -f "$INSTALLER"
else
  echo "Miniforge already at $PREFIX"
fi

# shellcheck disable=SC1091
source "${PREFIX}/etc/profile.d/conda.sh"
conda config --set auto_activate_base false
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

bashrc="${HOME}/.bashrc"
if [[ ! -f "$bashrc" ]] || ! grep -q 'miniconda3/etc/profile.d/conda.sh' "$bashrc" 2>/dev/null; then
  {
    echo ""
    echo "# Miniforge (installed as ~/miniconda3 for project scripts)"
    echo "if [ -f \"\$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then"
    echo "  . \"\$HOME/miniconda3/etc/profile.d/conda.sh\""
    echo "fi"
  } >> "$bashrc"
  echo "Appended conda hook to $bashrc"
fi

create_if_missing() {
  local name="$1"
  if conda env list | awk '{print $1}' | grep -qx "$name"; then
    echo "Env $name already exists"
  else
    echo "Creating env $name (python 3.10)"
    conda create -n "$name" python=3.10 -y
  fi
}

pip_in() {
  local name="$1"
  shift
  conda activate "$name"
  python -m pip install --upgrade pip
  python -m pip install --no-cache-dir "$@"
  conda deactivate
}

echo "=== tissue_miner (T1 detect / fourway / headless picker) ==="
create_if_missing tissue_miner
pip_in tissue_miner numpy tifffile scipy scikit-image PySide6

echo "=== TA (cellpose_track + optional epyseg visualize) ==="
create_if_missing TA
pip_in TA numpy pandas tifffile scipy scikit-image
conda activate TA
python -m pip install --no-cache-dir epyseg || echo "WARN: epyseg failed; cellpose_track still works"
conda deactivate

echo "=== cellpose (segmentation; CPU torch on this node) ==="
create_if_missing cellpose
pip_in cellpose "numpy<2" tifffile cellpose

echo "=== t1_pick (portable picker GUI) ==="
create_if_missing t1_pick
pip_in t1_pick numpy tifffile scipy scikit-image matplotlib PySide6

echo "=== cellflow (napariCellFlow + napari-patric) ==="
create_if_missing cellflow
conda activate cellflow
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -e "${ROOT}/vendor/napariCellFlow"
python -m pip install --no-cache-dir -e "${ROOT}/plugins/napari_patric"
conda deactivate

echo "=== cellpose_ta (PC combined Cellpose+TA; CPU) ==="
create_if_missing cellpose_ta
conda activate cellpose_ta
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r "${ROOT}/pc_cellpose_ta/requirements.txt" || \
  echo "WARN: cellpose_ta requirements failed (cellpose/epyseg conflict is common)"
conda deactivate

echo "=== verify imports ==="
check() {
  local name="$1"
  local expr="$2"
  conda activate "$name"
  echo -n "$name: "
  python -c "$expr" && echo OK || echo FAIL
  conda deactivate
}

check tissue_miner "import numpy,tifffile,scipy,skimage,PySide6"
check TA "import numpy,pandas,tifffile,scipy,skimage"
check cellpose "import numpy,tifffile,cellpose"
check t1_pick "import numpy,tifffile,scipy,skimage,matplotlib,PySide6"
check cellflow "import numpy,tifffile,napari,qtpy,napari_patric"

echo "=== conda env list ==="
conda env list
echo "DONE $(date -Is)"
