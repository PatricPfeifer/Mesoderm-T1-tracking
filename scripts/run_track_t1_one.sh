#!/usr/bin/env bash
# Track then T1 for one dataset. Usage: bash scripts/run_track_t1_one.sh <tag>
set -euo pipefail
cd /home/ho/ho_proana/ho_luchi75/projects/patric
source ~/miniconda3/etc/profile.d/conda.sh

tag="${1:?tag}"
case "$tag" in
  ventro)
    stem="ventrolateral20231202CriGap43Verenapos001-90s"
    masks="output/cellpose_seg/${stem}_cp_seg/${stem}_cp_masks.tif"
    intensity="input files/${stem}.tif"
    track_dir="output/cellpose_track/${stem}_cp_track"
    t1_dir="output/t1/${stem}_t1/detect"
    ;;
  chironomus)
    stem="20260123_63X_Chironomus_LA_and_Gap43-GFP.sld - Capture 3 - Position 2-1-1-1"
    masks="output/cellpose_seg/${stem}_cp_seg/${stem}_cp_masks.tif"
    intensity="input files/${stem}.tif"
    track_dir="output/cellpose_track/${stem}_cp_track"
    t1_dir="output/t1/${stem}_t1/detect"
    ;;
  lateral)
    stem="lateral20250806bCri-Gap43Verena_6_180s-slice7"
    masks="output/cellpose_seg/${stem}_cp_seg/${stem}_cp_masks.tif"
    intensity="input files/${stem}.tif"
    track_dir="output/cellpose_track/${stem}_cp_track"
    t1_dir="output/t1/${stem}_t1/detect"
    ;;
  *)
    echo "Unknown tag: $tag" >&2
    exit 1
    ;;
esac

mkdir -p "$track_dir" "$t1_dir"
logfile="$track_dir/track_t1.log"
exec > >(tee -a "$logfile") 2>&1

echo "=== TRACK $tag $(date -Is) ==="
conda activate TA
python scripts/cellpose_track/cellpose_track.py \
  --masks "$masks" \
  --intensity "$intensity" \
  --output-dir "$track_dir"

echo "=== T1 $tag $(date -Is) ==="
conda activate tissue_miner
python scripts/t1/run_tm_t1.py \
  --labels "$track_dir/${stem}_tracked_labels.tif" \
  --rgb "$track_dir/${stem}_tracked_rgb.tif" \
  --output-dir "$t1_dir"

echo "=== DONE $tag $(date -Is) ==="
