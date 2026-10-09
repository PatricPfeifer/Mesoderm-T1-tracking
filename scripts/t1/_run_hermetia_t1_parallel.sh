#!/usr/bin/env bash
set -euo pipefail
cd /home/ho/ho_proana/ho_luchi75/projects/patric
# shellcheck source=../require_compute.sh
source "/home/ho/ho_proana/ho_praih21/Mesoderm T1 tracking/scripts/require_compute.sh"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tissue_miner
export PYTHONPATH="/home/ho/ho_proana/ho_luchi75/projects/patric/scripts:${PYTHONPATH:-}"

S2="20260119_40X_Hermetia_Gap43-GFP_cellularization_#2_Control_lateral-2"
S6="20260119_40X_Hermetia_Gap43-GFP_cellularization_#6_Control_lateral-1"
D2="output/t1/${S2}_t1/detect"
D6="output/t1/${S6}_t1/detect"
mkdir -p "$D2" "$D6"

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-none} $(date -Is)"
echo "Starting T1 detect on ${S2} and ${S6} in parallel"

python -u scripts/t1/run_tm_t1.py \
  --labels "output/cellpose_track/${S2}_cp_track/${S2}_tracked_labels.tif" \
  --rgb "output/cellpose_track/${S2}_cp_track/${S2}_tracked_rgb.tif" \
  --output-dir "$D2" \
  > "$D2/t1.log" 2>&1 &
pid2=$!

python -u scripts/t1/run_tm_t1.py \
  --labels "output/cellpose_track/${S6}_cp_track/${S6}_tracked_labels.tif" \
  --rgb "output/cellpose_track/${S6}_cp_track/${S6}_tracked_rgb.tif" \
  --output-dir "$D6" \
  > "$D6/t1.log" 2>&1 &
pid6=$!

echo "PIDs: lateral2=$pid2 lateral6=$pid6"
echo "Logs: $D2/t1.log $D6/t1.log"

wait "$pid2"; ec2=$?
wait "$pid6"; ec6=$?
echo "DONE lateral2 exit=$ec2 lateral6 exit=$ec6 $(date -Is)"
exit $(( ec2 != 0 || ec6 != 0 ? 1 : 0 ))
