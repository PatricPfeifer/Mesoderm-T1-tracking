# Future improvement ideas

## Problems

- **Outline rays through the background (some frames).** In a few frames the saved `*_cell_outlines.tif` has straight lines running from the outer cells out to the image edge, through the empty space around the embryo. This must never happen.
    - Seen in: `HPC-share/current analysis/segmentation/Custom1/AVG-andbc_lateventral-slightlyturned20261027CriGap43eGFP12-2pmblursmothened_cell_outlines.tif` (cluster run, cell prob. threshold −1.0), frames 14, 37 and 53 (0-based). The other 69 frames have no stray lines; the masks of the bad frames look normal.
    - Cause: `outside_background` (`core/ta_bond_edit.py`) drops every background area that contains a small "enclosed gap" between cells. In each bad frame one 1 px gap at the rim of the tissue (frame 14: y 844, x 631) lies inside the large outside area, so the whole outside (~1.4 M px) stops being background. The watershed (`wshed`) then floods it and draws its basin borders as outline lines.
    - Fix idea: never drop a background area that touches the image edge (the outside of the embryo), or only drop an area when it is mostly gap. Add a test with a 1 px gap on the tissue rim.
    - Affects every place that builds 1 px bonds from masks (`thin_hand`): cluster and local segmentation outlines, Show / Load results, outline editing.

## Ideas

Still open from earlier lists. Already done and removed from here: working/results folder with fixed names and `settings.json`, automatic saving, pixel size and frame interval read from the TIFF, exclude/hide edge tracks, feature help cards.

New ideas are collected in the editable "TissueKit ideas" artifact.

- **Tracking run versions.** Segmentation runs now each get their own folder (`segmentation/<run>/`); tracking still overwrites `tracking/`. Keep earlier tracking runs the same way and record which one is current in `settings.json`.
- **Protect manual corrections.** Re-running tracking replaces the `Tracked` layer. Ask before replacing a layer with unsaved edits, and save corrected tracks as their own version.
- **Intensity field.** Warn, or default to the raw movie, when the Tracking intensity is `(none)`; without it the frame-shift registration falls back to binary masks.
- **Movie export button** (see `legacy/cellpose_track/make_track_movie.py`).
- **"Solo selected layer" checkbox** (hide all other layers on selection). napari's built-in alternative: Alt+click a layer's eye icon.
