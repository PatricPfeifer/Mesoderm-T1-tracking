#!/usr/bin/env bash
# Cheat sheet for Slurm allocations. Prefer the tmux-on-compute launcher below
# so work runs inside session "patric" on the allocation (not login-node tmux).
#
#   bash scripts/hpc_alloc_tmux.sh
#   bash scripts/hpc_alloc_tmux.sh -p cpu -n 8 -t 5:00:00 --mem=8GB
#   bash scripts/hpc_alloc_tmux.sh -p gpu_a100_short -n 12 -t 00:30:00 --mem=32GB --gres=gpu:1
#
# Then tell Cursor the session name is: patric

# Default launcher (cpu_il, 12 cores, 4h, 8GB) — kills old login tmux, starts patric on compute:
# bash scripts/hpc_alloc_tmux.sh
salloc -p dev_gpu_h100 -n 12 -t 0:30:00 --mem=10GB --gres=gpu:1
sinfo_t_idle
salloc -p gpu_a100_short -n 12 -t 00:30:00 --mem=32GB --gres=gpu:1
salloc -p cpu -n 8 -t 2:00:00 --mem=32GB
salloc -p cpu_il -n 12 -t 12:00:00 --mem=12GB

start_vnc_desktop --sw-rendering --queue cpu_il --ppn 4 --walltime 02:00:00
#launch napari on vnc like this:
source /pfs/data6/home/ho/ho_proana/ho_praih21/miniconda3/etc/profile.d/conda.sh
conda activate cellflow
export QT_QPA_PLATFORM=xcb
export QT_X11_NO_MITSHM=1
export QT_AUTO_SCREEN_SCALE_FACTOR=0
export QT_FONT_DPI=96
python -c "
import torch, triton
import napari
napari.Viewer()
napari.run()
"
tmux new -s wed 

# run in a terminal inside the VNC desktop
JOBID=7545314
VNC_DISPLAY=$DISPLAY

(
  # drop the VNC job's Slurm variables so they don't leak into the new step
  for v in $(env | grep -o '^SLURM_[A-Za-z0-9_]*'); do unset "$v"; done

  srun --jobid=$JOBID --overlap -n1 -c12 --cpu-bind=none --pty \
    env DISPLAY=$VNC_DISPLAY XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority} \
    bash -c '
source /pfs/data6/home/ho/ho_proana/ho_praih21/miniconda3/etc/profile.d/conda.sh
conda activate cellflow
export LP_NUM_THREADS=12
export QT_QPA_PLATFORM=xcb
export QT_X11_NO_MITSHM=1
export QT_AUTO_SCREEN_SCALE_FACTOR=0
export QT_FONT_DPI=96
python -c "
import torch, triton
import napari
napari.Viewer()
napari.run()
"
'
)


# Workspace command
# ws_allocate blah 30
# ws_list
# lfs quota -uh $(whoami) $HOME #Home folder space
# lfs quota -uh $USER /pfs/work9

# Old home
#/pfs/data5/home/ho/ho_proana/ho_luchi75/projects
