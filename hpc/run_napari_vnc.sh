#!/usr/bin/env bash
# Start napari with the TissueKit plugin docked, inside the TurboVNC desktop.
# Run it from a terminal INSIDE the VNC desktop (so it runs on the VNC compute node):
#   bash "/pfs/data6/home/ho/ho_proana/ho_praih21/Projects/Mesoderm T1 tracking/hpc/run_napari_vnc.sh"
# Works without pip-installing the plugin: it is loaded straight from plugins/napari_tissuekit.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /pfs/data6/home/ho/ho_proana/ho_praih21/miniconda3/etc/profile.d/conda.sh
conda activate cellflow

export PYTHONPATH="$PROJECT_DIR/plugins/napari_tissuekit${PYTHONPATH:+:$PYTHONPATH}"
export QT_QPA_PLATFORM=xcb
export QT_X11_NO_MITSHM=1
export QT_AUTO_SCREEN_SCALE_FACTOR=0
export QT_FONT_DPI=96

cd "$PROJECT_DIR"
# torch must be imported before napari/Qt, otherwise the VNC software-GL setup segfaults.
python -c "
import torch
try:
    import triton
except Exception:
    pass
import napari
from napari_tissuekit._widget import TissueKitWidget
viewer = napari.Viewer()
viewer.window.add_dock_widget(TissueKitWidget(viewer), area='right', name='TissueKit')
napari.run()
"
