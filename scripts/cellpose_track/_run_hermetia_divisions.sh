#!/usr/bin/env bash
# Re-track the two Hermetia embryos with cellpose_track_divisions.py.
# Sequential on purpose: each run holds several GB; the interactive
# allocation has 12 GB. Writes to *_cp_track_divisions (existing tracks stay).
set -uo pipefail
ROOT=/pfs/data6/home/ho/ho_proana/ho_praih21/Projects/Mesoderm\ T1\ tracking
cd "$ROOT"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate TA
export PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"

S2="20260119_40X_Hermetia_Gap43-GFP_cellularization_#2_Control_lateral-2"
S6="20260119_40X_Hermetia_Gap43-GFP_cellularization_#6_Control_lateral-1"

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-none} $(date -Is)"
rc=0
for S in "$S2" "$S6"; do
  OUT="output/cellpose_track/${S}_cp_track_divisions"
  mkdir -p "$OUT"
  echo "=== $S  start $(date -Is)"
  python -u scripts/cellpose_track/cellpose_track_divisions.py \
    --masks "output/cellpose_seg/${S}_cp_seg/${S}_cp_masks.tif" \
    --intensity "input/${S}.tif" \
    --output-dir "$OUT" 2>&1 | tee "$OUT/track.log"
  ec=${PIPESTATUS[0]}
  echo "=== $S  exit=$ec  $(date -Is)"
  [ "$ec" -ne 0 ] && rc=1
done
echo "ALL DONE rc=$rc $(date -Is)"
exit $rc
