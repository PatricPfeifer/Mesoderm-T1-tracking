# Future improvement ideas

## Open project folder

A single **Open project folder** button at the top of the Patric widget. The folder holds the raw TIFF, and every step reads its inputs and writes its outputs there by fixed names, so the user never has to pick files.

```
my_movie/
  raw.tif                      <- user's input
  segmentation_masks.tif       <- Segmentation step
  tracked_labels.tif / tracks.csv / track_events.csv   <- Tracking step
  tracked_labels_interior.tif  <- edge-filtered tracks
  t1_events.csv ...            <- T1 step
  settings.json                <- parameters used for each step
```

- Each tab reads what it needs from the folder. Tracking always uses the masks plus `raw.tif` as intensity, so the Intensity field can no longer be left on `(none)` (this makes tracking inaccurate, because the frame-shift registration falls back to binary masks).
- Reopening the folder shows which steps are done; hand-corrected files load automatically.
- `settings.json` stores parameters per step so results are reproducible.
- Layers are named after the files.

Implementation: a button plus a Qt folder dialog in the widget that keeps the folder path as state. napari's own File > Open Folder only loads images as layers, so the project logic is ours.

Open decisions:
1. Raw file: auto-detect the single TIFF, with a dropdown fallback if there are several.
2. Outputs: a `results/` subfolder (suggested) vs. the same folder as the raw data.
3. Corrected masks: version them (`masks_v2.tif`, suggested) vs. overwrite.

## Other ideas noted

- Tracking tab: "Exclude edge tracks" option (track all cells, then drop tracks touching the image edge), as in `cellpose_track.py --exclude-border`.
- Tracking tab: warn or default the Intensity field to the raw image when it is `(none)`.
- Movie export button (see `scripts/cellpose_track/make_track_movie.py`).
- Optional "Solo selected layer" checkbox (hide all other layers on selection). napari's built-in alternative: Alt+click a layer's eye icon.

## Physical units: read pixel size and time interval from the TIFF

**Problem.** All pipeline outputs are in pixels and frames. `tracks.csv` has `area` in px² and `centroid_y/x` in px, `track_events.csv` and `links.csv` use frame numbers, and the output TIFFs (`tracked_labels.tif`, masks) are written without any metadata. A statistics plugin (area, perimeter, shape index, speed) would therefore export numbers that can't be compared between movies with different magnification, and units would have to be entered by hand.

**Where the information already is.** The input TIFFs are ImageJ hyperstacks that carry it. For the test file (`...#1-cut.tif`):
- Pixel size: standard TIFF `XResolution`/`YResolution` tags (9692307/1000000, about 9.69 px per micron, i.e. about 0.103 micron per px) plus `unit: micron` in the ImageJ metadata.
- Time interval: ImageJ metadata `finterval` (about 60.0 s).
- Caveat: the TIFF `ResolutionUnit` tag reads NONE; the unit is only in the ImageJ metadata. Resolution tags may be missing in other files.

**Proposed solution.**
1. When a raw TIFF is loaded (or the project folder is opened), read `px_size_um`, `frame_interval_s` and `unit` with `tifffile` (ImageJ metadata first, then the resolution tags) and store them in `settings.json` / project metadata.
2. Warn when they are missing and let the user type them in once; never guess silently.
3. Write the metadata into every derived TIFF (masks, tracked labels) so each file is self-describing.
4. Keep the pipeline in pixels internally and convert only at the export and statistics layer: add `*_um` and `*_um2` columns and a `time_s` column next to the pixel and frame ones, so nothing existing breaks.
5. Plugins receive the metadata as a parameter, e.g. `analyze(tracked_labels, tracks_df, metadata)`.

## Automatic saving of results (no "Save to" field)

**Problem (current behaviour).** In the GUI, running a step does not save anything unless the user fills in a "Save to" field.
- "Save to" empty (the default): the GUI passes no output folder to the tracker, which only writes files when it has one (`celltrack/pipeline.py`). The result appears in napari as a `Tracked` layer (plus `Tracked RGB`) and the status bar says "Tracking done", but nothing is written to disk. Closing napari loses it.
- "Save to" filled in: it writes `tracked_labels.tif`, `tracked_rgb.tif`, `links.csv`, `tracks.csv`, `track_events.csv` and `qc.json` into that folder. File names come from the Labels layer's name plus a fixed suffix. A second run into the same folder overwrites the earlier files, with no prompt or versioning.
- Re-running tracking replaces the `Tracked` layer, so manual track corrections on that layer are lost.
- The command line scripts always save, because `--output-dir` is required, so the GUI and the scripts behave differently.

**Proposed solution.** Each input file has exactly one working folder, and results are saved there automatically. This is the same idea as "Open project folder" above, and the two should be built together.
1. Derive the working folder from the raw file's location (for example `<raw_dir>/<raw_stem>_results/`, or the project folder). There is no "Save to" field.
2. Every step (segmentation, tracking, T1) writes its outputs to that folder as soon as it finishes, using fixed names, so downstream steps and plugins know where to find them.
3. Never overwrite silently: keep earlier runs as versions (`tracked_labels_v1.tif`, `_v2`, ...) or in timestamped subfolders, and keep a `latest` pointer or `settings.json` entry that says which version is current.
4. Re-running tracking should not destroy manual corrections: save corrected tracks as their own version, and ask before replacing a layer that has unsaved edits.
5. Write the parameters used for each run next to its outputs (`settings.json`) so a result can be reproduced.
6. Reopening the raw file (or the project folder) finds the working folder and reloads the latest results automatically.
