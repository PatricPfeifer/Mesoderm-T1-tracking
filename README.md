# napari-tissuekit

Cellpose segmentation, cell tracking and T1 detection, as a napari plugin and as a
command line that runs on the HPC. Both use the same code and the same results folder.

## Layout

```
plugins/napari_tissuekit/napari_tissuekit/
  core/      pipeline code, no napari/Qt: segmentation, celltrack/, t1/, project.py (results folder), output_names.py
  cli.py     the `tissuekit` command
  _widget.py, cell_properties.py, track_edit.py   napari widget
hpc/         Slurm helpers (allocation, conda env setup)
legacy/      old per-dataset scripts, TA/epyseg and napariCellFlow code. Not maintained.
tests/
```

## Results folder

Next to each movie: `<movie>_results/` with `segmentation/`, `tracking/`, `t1/`, `cell_props/`
and `settings.json`. "Folder..." / "Load results" in napari open it; `tissuekit` writes it.
File names are listed in `core/project.py` (read) and `core/output_names.py` (written). A test
(`tests/plugin/test_project.py`) fails if the two disagree, so add new output tags to both.

## Command line (laptop or HPC)

```bash
pip install -e plugins/napari_tissuekit              # headless is enough on the HPC
tissuekit segment movie.tif [--preset cellflow] [--force-cpu]   # -> movie_results/segmentation/
tissuekit track   movie.tif                          # -> movie_results/tracking/
tissuekit t1      movie.tif                          # -> movie_results/t1/   (needs a Slurm job, or ALLOW_LOGIN_NODE=1)
```

Extra options are passed to the step (`python -m napari_tissuekit.core.tracking --help`).
Copy `<movie>_results/` back and open it in napari with "Folder...".
Slurm helpers: `hpc/hpc_alloc_tmux.sh`, `hpc/start_working.sh`.

## Tests

```bash
pytest tests -m "not slow"   # fast
pytest tests                 # also runs Cellpose on CPU (about a minute)
```
