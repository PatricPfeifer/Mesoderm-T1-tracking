#!/usr/bin/env bash
set -euo pipefail
cd /home/ho/ho_proana/ho_luchi75/projects/patric
source ~/miniconda3/etc/profile.d/conda.sh
conda activate TA
export PYTHONPATH="/home/ho/ho_proana/ho_luchi75/projects/patric/scripts:${PYTHONPATH:-}"

S2="20260119_40X_Hermetia_Gap43-GFP_cellularization_#2_Control_lateral-2"
S6="20260119_40X_Hermetia_Gap43-GFP_cellularization_#6_Control_lateral-1"
T2="output/cellpose_track/${S2}_cp_track"
T6="output/cellpose_track/${S6}_cp_track"
mkdir -p "$T2" "$T6"

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-none} $(date -Is)"
echo "Starting ${S2} and ${S6} in parallel"

python -u scripts/cellpose_track/cellpose_track.py \
  --masks "output/cellpose_seg/${S2}_cp_seg/${S2}_cp_masks.tif" \
  --intensity "input files/${S2}.tif" \
  --output-dir "$T2" \
  > "$T2/track.log" 2>&1 &
pid2=$!

python -u scripts/cellpose_track/cellpose_track.py \
  --masks "output/cellpose_seg/${S6}_cp_seg/${S6}_cp_masks.tif" \
  --intensity "input files/${S6}.tif" \
  --output-dir "$T6" \
  > "$T6/track.log" 2>&1 &
pid6=$!

echo "PIDs: lateral2=$pid2 lateral6=$pid6"
echo "Logs: $T2/track.log $T6/track.log"

wait "$pid2"; ec2=$?
wait "$pid6"; ec6=$?
echo "DONE lateral2 exit=$ec2 lateral6 exit=$ec6 $(date -Is)"
exit $(( ec2 != 0 || ec6 != 0 ? 1 : 0 ))
