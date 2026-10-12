#!/usr/bin/env bash
# T1 detection on the division-aware tracking of both Hermetia embryos and the
# Naima movie, all three in parallel (~4 GB each; needs >= 16 GB allocation).
# run_tm_t1.py picks up {stem}_lineage.csv next to the labels and excludes
# dividing mothers + daughters. Outputs: output/t1/<stem>_t1_divisions/detect
set -uo pipefail
ROOT=/pfs/data6/home/ho/ho_proana/ho_praih21/Projects/Mesoderm\ T1\ tracking
cd "$ROOT"
source "$ROOT/scripts/require_compute.sh"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tissue_miner
export PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"

H2="20260119_40X_Hermetia_Gap43-GFP_cellularization_#2_Control_lateral-2"
H6="20260119_40X_Hermetia_Gap43-GFP_cellularization_#6_Control_lateral-1"
NM="Cri_WT_GAP43_GFP_38sec_Naima_cut"

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-none} $(date -Is)"
pids=(); names=()
for S in "$H2" "$H6" "$NM"; do
  TRK="output/cellpose_track/${S}_cp_track_divisions"
  OUT="output/t1/${S}_t1_divisions/detect"
  if [ ! -f "$TRK/${S}_tracked_labels.tif" ] || [ ! -f "$TRK/${S}_lineage.csv" ]; then
    echo "=== $S: SKIPPED, tracking output missing in $TRK"; continue
  fi
  mkdir -p "$OUT"
  python -u scripts/t1/run_tm_t1.py \
    --labels "$TRK/${S}_tracked_labels.tif" \
    --rgb "$TRK/${S}_tracked_rgb.tif" \
    --output-dir "$OUT" > "$OUT/t1.log" 2>&1 &
  pids+=($!); names+=("$S")
  echo "started $S pid=$! log=$OUT/t1.log"
done

rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; ec=$?
  echo "=== ${names[$i]} exit=$ec $(date -Is)"
  [ "$ec" -ne 0 ] && rc=1
done
echo "ALL T1 DONE rc=$rc $(date -Is)"
exit $rc
