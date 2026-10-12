"""Open napari + Export Cell Props widget. Usage: python open_in_napari.py [labels.tif ...]
With no args, loads the synthetic validation data."""
import sys
from pathlib import Path

import napari
import tifffile

here = Path(__file__).parent
sys.path.insert(0, str(here))
v = napari.Viewer()
if len(sys.argv) > 1:
    for p in sys.argv[1:]:
        v.add_labels(tifffile.imread(p), name=Path(p).stem)
else:
    from test_cell_properties_hexagon import make_stack as hexagon
    v.add_labels(tifffile.imread(here / "validation_labels.tif"), name="rectangles (ids 1,2,3)")
    v.add_labels(hexagon(), name="hexagon (id 1)")
v.window.add_plugin_dock_widget("napari-tissuekit")
napari.run()
