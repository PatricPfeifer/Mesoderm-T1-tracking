#!/usr/bin/env bash
# Cellpose (CPU) → track → T1 detect for Capture 12 Position 2 crop.
set -euo pipefail
ROOT="/home/ho/ho_proana/ho_praih21/Mesoderm T1 tracking"
cd "$ROOT"
# shellcheck source=../require_compute.sh
source "$ROOT/scripts/require_compute.sh"
# shellcheck disable=SC1091
source "${HOME}/miniconda3/etc/profile.d/conda.sh"

STEM="cropped z:6:21 t:32:227 - Capture 12 - Position 2"
INP="input files/${STEM}.tif"
SEG="output/cellpose_seg/${STEM}_cp_seg"
TRACK="output/cellpose_track/${STEM}_cp_track"
T1="output/t1/${STEM}_t1/detect"
LOG="${T1}/pipeline.log"

mkdir -p "$SEG" "$TRACK" "$T1"
exec > >(tee -a "$LOG") 2>&1

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-none} $(date -Is)"
echo "INPUT=$INP"

echo "=== CELLPOSE $(date -Is) ==="
conda activate cellpose
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"
python -u scripts/cellpose_seg/run_cellpose_naima.py \
  --input "$INP" \
  --output-dir "$SEG" \
  --force-cpu

echo "=== TRACK $(date -Is) ==="
conda activate TA
python -u scripts/cellpose_track/cellpose_track.py \
  --masks "${SEG}/${STEM}_cp_masks.tif" \
  --intensity "$INP" \
  --output-dir "$TRACK"

echo "=== T1 $(date -Is) ==="
conda activate tissue_miner
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"
python -u scripts/t1/run_tm_t1.py \
  --labels "${TRACK}/${STEM}_tracked_labels.tif" \
  --rgb "${TRACK}/${STEM}_tracked_rgb.tif" \
  --output-dir "$T1"

echo "DONE capture12 pipeline $(date -Is)"
