#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/ho/ho_proana/ho_praih21/Mesoderm T1 tracking"
cd "$ROOT"
# shellcheck source=../require_compute.sh
source "$ROOT/scripts/require_compute.sh"

activate_t1_python() {
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate tissue_miner
    return
  fi
  set +u
  # shellcheck disable=SC1091
  source /etc/profile
  module load devel/miniforge/25.3.1-python-3.12
  set -u
  local venv="${ROOT}/.venv_t1"
  if [[ ! -x "${venv}/bin/python" ]]; then
    echo "Creating ${venv} (numpy tifffile scipy scikit-image)"
    python -m venv "${venv}"
    "${venv}/bin/pip" install -U pip
    "${venv}/bin/pip" install numpy tifffile scipy scikit-image
  fi
  # shellcheck disable=SC1091
  source "${venv}/bin/activate"
}

activate_t1_python
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"
python -c "import numpy,tifffile,scipy,skimage; print('python', __import__('sys').executable)"

LABELS="output/cellpose_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_cp_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_tracked_labels.tif"
RGB="output/cellpose_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_cp_track/Cri_WT_GAP43_GFP_38sec_Naima_cut_tracked_rgb.tif"
OUT="output/t1/Cri_WT_GAP43_GFP_38sec_Naima_cut_t1/detect"

mkdir -p "$OUT"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-none} $(date -Is)"
echo "ROOT=$ROOT"
echo "Starting T1 naima_cut detect → $OUT"

python -u scripts/t1/run_tm_t1.py \
  --labels "$LABELS" \
  --rgb "$RGB" \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/t1.log"

echo "DONE naima_cut detect exit=$? $(date -Is)"
