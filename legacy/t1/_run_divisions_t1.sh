#!/usr/bin/env bash
# T1 detection on the division-aware tracking of the two Hermetia embryos and
# the Naima movie. Waits for a running cellpose_track_divisions.py to finish,
# then runs the three detections one after another (12 GB allocation).
# run_tm_t1.py picks up {stem}_lineage.csv next to the labels and excludes
# dividing mothers + daughters. Outputs: output/t1/<stem>_t1_divisions/detect
set -uo pipefail
ROOT=/pfs/data6/home/ho/ho_proana/ho_praih21/Projects/Mesoderm\ T1\ tracking
cd "$ROOT"
source "$ROOT/scripts/require_compute.sh"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tissue_miner
export PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-none} $(date -Is)"
while pgrep -u "$USER" -f cellpose_track_divisions.py >/dev/null; do
  echo "waiting for tracking to finish... $(date +%H:%M:%S)"
  sleep 20
done

H2="20260119_40X_Hermetia_Gap43-GFP_cellularization_#2_Control_lateral-2"
H6="20260119_40X_Hermetia_Gap43-GFP_cellularization_#6_Control_lateral-1"
NM="Cri_WT_GAP43_GFP_38sec_Naima_cut"

rc=0
for S in "$H2" "$H6" "$NM"; do
  TRK="output/cellpose_track/${S}_cp_track_divisions"
  OUT="output/t1/${S}_t1_divisions/detect"
  LABELS="$TRK/${S}_tracked_labels.tif"
  RGB="$TRK/${S}_tracked_rgb.tif"
  if [ ! -f "$LABELS" ] || [ ! -f "$TRK/${S}_lineage.csv" ]; then
    echo "=== $S: SKIPPED, tracking output missing in $TRK"; rc=1; continue
  fi
  mkdir -p "$OUT"
  echo "=== $S  start $(date -Is)"
  python -u scripts/t1/run_tm_t1.py \
    --labels "$LABELS" --rgb "$RGB" --output-dir "$OUT" 2>&1 | tee "$OUT/t1.log"
  ec=${PIPESTATUS[0]}
  echo "=== $S  exit=$ec  $(date -Is)"
  [ "$ec" -ne 0 ] && rc=1
done
echo "ALL T1 DONE rc=$rc $(date -Is)"
exit $rc
