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


def test_archive_file_keeps_old_result(tmp_path):
    wf = WorkingFolder()
    wf.set_manual(tmp_path)
    f = wf.step_dir("segmentation") / "m_cp_masks.tif"
    f.write_bytes(b"old")
    moved = wf.archive_file(f)
    assert not f.exists()
    assert moved.read_bytes() == b"old" and moved.parent == tmp_path / "archive"
    assert wf.archive_file(f) is None  # nothing there any more


def test_archive_step_moves_whole_folder_and_never_collides(tmp_path):
    wf = WorkingFolder()
    wf.set_manual(tmp_path)
    for _ in range(2):
        (wf.step_dir("tracking") / "x_tracks.csv").write_text("a")
        assert wf.archive_step("tracking") is not None
    assert len(list((tmp_path / "archive").iterdir())) == 2
    assert wf.archive_step("tracking") is None  # empty or missing folder is skipped


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
