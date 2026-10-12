#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/ho/ho_proana/ho_praih21/Mesoderm T1 tracking"
cd "$ROOT"
# shellcheck source=../require_compute.sh
source "$ROOT/scripts/require_compute.sh"
# shellcheck disable=SC1091
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate tissue_miner

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-none} $(date -Is)"

run_one () {
  local stem="$1"
  local run="output/t1/${stem}_t1"
  local pick="${run}/picker_ready"
  local out="${run}/picked"
  # Legacy fallback if only the old flat layout exists
  if [[ ! -d "$pick" ]]; then
    pick="${run}/${stem}_t1_picker"
  fi
  mkdir -p "$out"
  echo "=== HEADLESS PICK $stem $(date -Is) ==="
  echo "  picker_ready=$pick"
  echo "  picked=$out"
  python -u scripts/t1/run_tm_t1_picker.py \
    --headless --select-all --filter all \
    --events "${pick}/${stem}_t1_events.csv" \
    --labels "${pick}/${stem}_tracked_labels.tif" \
    --rgb "${pick}/${stem}_tracked_rgb.tif" \
    --output-dir "$out" \
    --window 10 \
    2>&1 | tee "${out}/headless_pick.log"
  echo "=== DONE PICK $stem $(date -Is) ==="
}

run_one \
  "20260119_40X_Hermetia_Gap43-GFP_cellularization_#2_Control_lateral-2"
run_one \
  "20260119_40X_Hermetia_Gap43-GFP_cellularization_#6_Control_lateral-1"

echo "ALL DONE $(date -Is)"
