"""Working-folder logic: location, archiving instead of overwriting, settings.json."""
import json

from napari_patric.project import WorkingFolder, find_results_root, movie_stem, results_dir_for


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
    wf.write_settings("segmentation", {"batch": 8}, movie="m.tif")
    wf.write_settings("tracking", {"expand": 3})
    data = json.loads((tmp_path / "res" / "settings.json").read_text())
    assert data["movie"] == "m.tif"
    assert data["segmentation"]["params"] == {"batch": 8}
    assert data["tracking"]["params"] == {"expand": 3} and "time" in data["tracking"]


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


def test_plugin_matches_every_mask_and_track_tag_the_scripts_know():
    """Add a new output tag to scripts/output_names.py -> add it to project.py too."""
    import sys
    from pathlib import Path

    from napari_patric import project

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        import output_names
    finally:
        sys.path.remove(str(scripts))
    for suffix in project.SEG_MASK_SUFFIXES + project.TRACK_LABEL_SUFFIXES:
        assert suffix[: -len(".tif")] in output_names.STRIP_SUFFIXES, suffix
    scripted = [t for t in output_names.STRIP_SUFFIXES if t.endswith(("_masks", "_tracked_labels"))]
    for tag in scripted:
        assert f"{tag}.tif" in project.SEG_MASK_SUFFIXES + project.TRACK_LABEL_SUFFIXES, tag
