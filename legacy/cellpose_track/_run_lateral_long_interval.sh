#!/usr/bin/env bash
set -euo pipefail
cd /home/ho/ho_proana/ho_luchi75/projects/patric
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate TA
outdir="output/cellpose_track/lateral20250806_slice7_long_interval"
mkdir -p "$outdir"
python scripts/cellpose_track/cellpose_track_long_interval.py \
  --masks "output/cellpose_seg/lateral20250806bCri-Gap43Verena_6_180s-slice7_cp_seg/lateral20250806bCri-Gap43Verena_6_180s-slice7_cp_masks.tif" \
  --intensity "input files/lateral20250806bCri-Gap43Verena_6_180s-slice7.tif" \
  --output-dir "$outdir" \
  2>&1 | tee -a "$outdir/track.log"
