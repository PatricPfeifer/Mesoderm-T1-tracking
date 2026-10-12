# Tests

Run from the repo root (conda env `cellflow`, plus `pip install pytest`):

```bash
pytest -m "not slow"   # fast: cell properties + tracking on stored masks (seconds)
pytest                 # everything, including Cellpose segmentation (about a minute on CPU)
```

- `plugin/`: cell-property maths (`compute_properties`) on synthetic shapes with known answers.
- `pipeline/`: tracking on stored Cellpose masks (fast), and segmentation + tracking on the real movie (slow).
- `data/`: `movie_cut.tif` (real movie, 11 frames, 291x291, blastoderm test crop) and
  `masks_cut_cpsam.tif` (its Cellpose 4 `cpsam` masks). Keep data here small; never commit the full 1 GB movie.
