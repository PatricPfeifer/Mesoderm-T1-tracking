"""Working-folder logic: location, archiving instead of overwriting, settings.json."""
import json

from napari_tissuekit.core.project import WorkingFolder, find_results_root, movie_stem, results_dir_for


def test_movie_stem_and_default_folder(tmp_path):
    assert movie_stem("a/b/movie.ome.tif") == "movie"
    assert movie_stem("movie.tiff") == "movie"
    assert movie_stem("my.movie.tif") == "my.movie"
    assert results_dir_for(tmp_path / "x" / "movie.tif") == tmp_path / "x" / "movie_results"


def test_resolve_prefers_manual_choice(tmp_path):
    wf = WorkingFolder()
    assert wf.resolve([None, None]) is None
    assert wf.resolve([None, tmp_path / "a.tif"]) == tmp_path / "a_results"
    wf.set_manual(tmp_path / "mine")
    assert wf.resolve([tmp_path / "other.tif"]) == tmp_path / "mine"


def test_settings_accumulate_per_step(tmp_path):
    wf = WorkingFolder()
    wf.set_manual(tmp_path / "res")
    wf.write_settings("segmentation", {"batch": 8}, movie=tmp_path / "m.tif")
    wf.write_settings("tracking", {"expand": 3})
    data = json.loads((tmp_path / "res" / "settings.json").read_text())
    assert data["movie"] == "../m.tif"
    assert data["segmentation"]["params"] == {"batch": 8}
    assert data["tracking"]["params"] == {"expand": 3} and "time" in data["tracking"]


# ---- settings.json in a shared / synced folder --------------------------------
def _results_with_movie(base):
    movie = base / "movie.tif"
    movie.parent.mkdir(parents=True, exist_ok=True)
    movie.write_bytes(b"")
    wf = WorkingFolder()
    wf.set_manual(results_dir_for(movie))
    return wf, movie


def test_settings_store_movie_relative_to_results(tmp_path):
    wf, movie = _results_with_movie(tmp_path / "a")
    wf.write_settings("segmentation", {}, movie=movie)
    data = json.loads(wf.settings_path.read_text())
    assert data["movie"] == "../movie.tif"
    assert str(tmp_path) not in wf.settings_path.read_text()
    assert wf.movie_path().resolve() == movie.resolve()


def test_movie_found_after_folder_is_moved(tmp_path):
    """Same results on another computer (Nextcloud): the movie still resolves."""
    wf, movie = _results_with_movie(tmp_path / "pc1")
    wf.write_settings("segmentation", {}, movie=movie)
    (tmp_path / "pc1").rename(tmp_path / "pc2")
    other = WorkingFolder()
    other.set_manual(tmp_path / "pc2" / "movie_results")
    assert other.movie_path().resolve() == (tmp_path / "pc2" / "movie.tif").resolve()


def test_old_absolute_paths_from_other_computers_fall_back_to_name(tmp_path):
    wf, movie = _results_with_movie(tmp_path)
    wf.root.mkdir()
    for saved in (
        "/Users/verena/Documents/current analysis/movie.tif",
        r"C:\Users\andre\Nextcloud\HPC-share\current analysis\movie.tif",
    ):
        wf.settings_path.write_text(json.dumps({"movie": saved}))
        assert wf.movie_path() == wf.root.parent / "movie.tif"


def test_movie_path_none_when_missing(tmp_path):
    wf, movie = _results_with_movie(tmp_path)
    assert wf.movie_path() is None  # no settings.json
    wf.write_settings("segmentation", {}, movie=movie)
    movie.unlink()
    assert wf.movie_path() is None


def test_found_lists_existing_results(tmp_path):
    wf = WorkingFolder()
    wf.set_manual(tmp_path)
    assert wf.found() == {}
    (wf.step_dir("tracking") / "m_tracked_labels.tif").write_bytes(b"")
    assert list(wf.found()) == ["tracking"]


def test_find_results_root_accepts_movie_folder_or_results_folder(tmp_path):
    res = tmp_path / "m_results"
    (res / "tracking").mkdir(parents=True)
    assert find_results_root(res) == res
    assert find_results_root(tmp_path) == res          # the movie's folder
    (tmp_path / "other_results").mkdir()
    assert find_results_root(tmp_path) == tmp_path     # ambiguous: leave as chosen


# ---- naming contract with the pipeline scripts / HPC layouts -----------------
def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_found_reads_hpc_cellflow_layout_and_guesses_movie(tmp_path):
    """Egg1: movie next to a ``cellflow_seg`` folder holding ``*_cf_masks.tif``, no settings.json."""
    movie = _touch(tmp_path / "egg1_pos3.tif")
    mask = _touch(tmp_path / "cellflow_seg" / "egg1_pos3_cf_masks.tif")
    _touch(tmp_path / "cellflow_seg" / "egg1_pos3_cf_seg_qc.json")
    root = find_results_root(tmp_path)
    assert root == tmp_path
    wf = WorkingFolder()
    wf.set_manual(root)
    assert wf.found() == {"segmentation": [mask]}
    assert wf.guess_movie() == movie


def test_found_reads_cf_tracked_labels_and_legacy_per_movie_folders(tmp_path):
    seg = _touch(tmp_path / "m_cp_seg" / "m_cp_masks.tif")
    trk = _touch(tmp_path / "tracking" / "m_cf_tracked_labels.tif")
    wf = WorkingFolder()
    wf.set_manual(tmp_path)
    assert wf.found() == {"segmentation": [seg], "tracking": [trk]}


def test_guess_movie_in_parent_of_results_folder(tmp_path):
    movie = _touch(tmp_path / "m.ome.tif")
    _touch(tmp_path / "m_results" / "segmentation" / "m_cp_masks.tif")
    wf = WorkingFolder()
    wf.set_manual(tmp_path / "m_results")
    assert wf.guess_movie() == movie


def test_project_matches_every_mask_and_track_tag_in_output_names():
    """Add a new output tag to core/output_names.py -> add it to core/project.py too."""
    from napari_tissuekit.core import output_names, project

    for suffix in project.SEG_MASK_SUFFIXES + project.TRACK_LABEL_SUFFIXES:
        assert suffix[: -len(".tif")] in output_names.STRIP_SUFFIXES, suffix
    scripted = [t for t in output_names.STRIP_SUFFIXES if t.endswith(("_masks", "_tracked_labels"))]
    for tag in scripted:
        assert f"{tag}.tif" in project.SEG_MASK_SUFFIXES + project.TRACK_LABEL_SUFFIXES, tag


def test_found_lists_outlines_in_the_folder_itself_and_in_segmentation(tmp_path):
    wf = WorkingFolder()
    wf.set_manual(tmp_path)
    _touch(tmp_path / "m_cell_outlines.tif")
    _touch(tmp_path / "m_cp_outlines.tif")
    found = wf.found()
    assert [f.name for f in found["outlines"]] == ["m_cell_outlines.tif", "m_cp_outlines.tif"]  # 1 px bonds first
    assert "segmentation" not in found
    _touch(tmp_path / "segmentation" / "n_cp_outlines.tif")
    assert any(f.parent.name == "segmentation" for f in wf.found()["outlines"])


def test_outlines_only_file_gives_cells_between_the_outlines():
    import numpy as np
    from napari_tissuekit.core.outline_io import outline_labels_to_rgb, outlines_rgb_to_masks, rgb_or_labels_to_outline_labels
    ol = np.zeros((1, 21, 21), dtype=np.uint8)
    ol[0, 0, :] = ol[0, -1, :] = ol[0, :, 0] = ol[0, :, -1] = 1   # frame
    ol[0, 10, :] = ol[0, :, 10] = 1                                # cross: four cells
    rgb = outline_labels_to_rgb(ol)
    assert (rgb_or_labels_to_outline_labels(rgb) == ol).all()
    assert (rgb_or_labels_to_outline_labels(ol * 255) == ol).all()  # 0/255 *_cell_outlines.tif
    masks = outlines_rgb_to_masks(rgb, background="auto")
    assert len(set(masks.ravel()) - {0}) == 4          # outlines fill the frame: no region is thrown away as background
    big = np.zeros((1, 41, 41), dtype=np.uint8)         # four small cells in a large empty background
    big[0, 15, 15:26] = big[0, 25, 15:26] = big[0, 15:26, 15] = big[0, 15:26, 25] = 1
    big[0, 20, 15:26] = big[0, 15:26, 20] = 1
    assert len(set(outlines_rgb_to_masks(outline_labels_to_rgb(big), background="auto").ravel()) - {0}) == 4


def test_picking_the_segmentation_folder_still_gives_the_results_folder_with_its_outlines(tmp_path):
    res = tmp_path / "m_results"
    _touch(res / "segmentation" / "m_cell_outlines.tif")
    _touch(res / "segmentation" / "m_cp_masks.tif")
    assert find_results_root(res / "segmentation") == res
    wf = WorkingFolder()
    wf.set_manual(res)
    assert set(wf.found()) == {"segmentation", "outlines"}


def test_outlines_are_edits_only_when_close_to_the_bonds():
    import numpy as np
    from napari_tissuekit.core.outline_io import outlines_are_edits
    base = np.zeros((1, 50, 50), dtype=np.uint8)
    base[0, 10, :] = base[0, 25, :] = base[0, 40, :] = 1
    edited = base.copy()
    edited[0, 10, 10:14] = 1                      # a few painted pixels: an edit
    assert outlines_are_edits(edited, base)
    thick = base.copy()
    thick[0, 24, :] = thick[0, 26, :] = 1         # a different style: not an edit
    assert not outlines_are_edits(thick, base)
