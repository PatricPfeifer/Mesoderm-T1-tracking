# Future improvement ideas

Still open from earlier lists. Already done and removed from here: working/results folder with fixed names and `settings.json`, automatic saving, pixel size and frame interval read from the TIFF, exclude/hide edge tracks, feature help cards.

New ideas are collected in the editable "TissueKit ideas" artifact.

- **Run versions.** A re-run into the same results folder overwrites the earlier files. Keep earlier runs (`_v1`, `_v2` or timestamped subfolders) and record which one is current in `settings.json`.
- **Protect manual corrections.** Re-running tracking replaces the `Tracked` layer. Ask before replacing a layer with unsaved edits, and save corrected tracks as their own version.
- **Intensity field.** Warn, or default to the raw movie, when the Tracking intensity is `(none)`; without it the frame-shift registration falls back to binary masks.
- **Movie export button** (see `legacy/cellpose_track/make_track_movie.py`).
- **"Solo selected layer" checkbox** (hide all other layers on selection). napari's built-in alternative: Alt+click a layer's eye icon.
