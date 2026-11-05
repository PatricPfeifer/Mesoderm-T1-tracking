"""Batch tab core (core/batch.py) and the batch job script (core/remote.py), no cluster."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from napari_tissuekit.core import batch, remote
from napari_tissuekit.core.project import WorkingFolder
from napari_tissuekit.core.seg_params import full_params

CELLFLOW = {"diameter": 0.0, "flow_threshold": 0.6, "cellprob_threshold": 0.3, "min_size": 25, "normalize": True}


def _movie(folder, name):
    p = folder / name
    p.write_bytes(b"")
    return p


def _run(movie, run, values, preset="x"):
    """A saved run in the movie's results folder, as `tissuekit segment` writes it."""
    wf = WorkingFolder()
    wf.set_manual(movie.parent / f"{movie.name.rsplit('.', 1)[0]}_results")
    d = wf.seg_run_dir(run)
    (d / f"{movie.stem}_cp_masks.tif").write_bytes(b"")
    wf.write_run_params(run, {"preset": preset, "params": full_params(values), "fill_holes": True})
    return wf


def test_find_movies_sorted_and_refuses_shared_stem(tmp_path):
    _movie(tmp_path, "egg2.tif")
    _movie(tmp_path, "Egg1.TIF")
    _movie(tmp_path, "notes.txt")
    (tmp_path / "egg2_results").mkdir()
    assert [m.name for m in batch.find_movies(tmp_path)] == ["Egg1.TIF", "egg2.tif"]
    _movie(tmp_path, "egg2.tiff")
    with pytest.raises(batch.BatchError, match="share the results folder"):
        batch.find_movies(tmp_path)


def test_run_matches_profile_by_values_not_batch_size(tmp_path):
    egg = _movie(tmp_path, "egg1.tif")
    _run(egg, "cellflow", {**CELLFLOW, "batch_size": 64}, preset="cellflow")
    st = batch.egg_status(egg)
    assert st.run_for(CELLFLOW).name == "cellflow"      # batch size only changes speed
    assert st.run_for({}) is None                        # default values: not done
    assert st.active_run == "cellflow"
    assert not st.has_tracking


def test_plan_skips_done_pairs_and_reruns_get_new_names(tmp_path):
    e1, e2 = _movie(tmp_path, "egg1.tif"), _movie(tmp_path, "egg2.tif")
    _run(e1, "default", {})
    eggs = batch.scan_folder(tmp_path)
    profiles = {"default": full_params({}), "cellflow": full_params(CELLFLOW)}
    items, skipped = batch.plan_items(eggs, profiles)
    assert [(it.egg, it.profile, it.run) for it in items] == [
        ("egg1", "cellflow", "cellflow"), ("egg2", "default", "default"), ("egg2", "cellflow", "cellflow")]
    assert skipped == [("egg1", "default", "default")]
    items, skipped = batch.plan_items(eggs, profiles, rerun=True, fill_holes=False)
    assert [it.run for it in items if it.egg == "egg1"] == ["default_2", "cellflow"]
    assert not skipped and not items[0].fill_holes
    assert items[0].params == full_params({})


def test_time_limits():
    assert remote.parse_time("00:05:00") == 300
    assert remote.parse_time("30") == 1800
    assert remote.parse_time("1-02:00:00") == 26 * 3600
    assert remote.batch_time("00:05:00", 7) == "00:35:00"
    assert remote.batch_time("02:00:00", 40) == "80:00:00"
    assert "gpu_h100" in remote.partition_problem("dev_gpu_h100", "00:35:00")
    assert remote.partition_problem("dev_gpu_h100", "00:30:00") == ""
    assert remote.partition_problem("gpu_h100", "80:00:00")
    assert remote.partition_problem("some_queue", "999:00:00") == ""


def test_batch_job_script_one_line_per_item(tmp_path):
    e1, e2 = _movie(tmp_path, "egg 1.tif"), _movie(tmp_path, "egg2.tif")
    items, _ = batch.plan_items(batch.scan_folder(tmp_path),
                                {"default": full_params({}), "My preset": full_params(CELLFLOW)})
    cfg = remote.RemoteConfig(user="u", partition="gpu_h100", time="00:05:00")
    text = remote.batch_job_script(cfg, "/home/u/tissuekit_remote/batch_x", items)
    assert "#SBATCH --time=00:20:00" in text            # 4 items x 5 min
    assert "#SBATCH --job-name=tk_batch_2x4" in text
    assert text.count("\nconda activate ") == 1  # setup once, at the top
    assert text.count("tissuekit segment") == 4
    assert "echo 'TISSUEKIT_ITEM 1/4 egg 1.tif default'" in text
    assert "tissuekit segment 'egg 1.tif' --preset default --batch-size 8 --run default --params" in text
    assert "--preset My_preset" in text and "--run My_preset" in text
    assert "&& echo 'TISSUEKIT_ITEM_DONE 4' || echo 'TISSUEKIT_ITEM_FAILED 4'" in text
    assert text.rstrip().endswith("echo TISSUEKIT_DONE")


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_failed_item_does_not_stop_the_batch(tmp_path):
    """The DONE / FAILED markers under `set -eo pipefail`: a failing item, then the next one runs."""
    script = "\n".join([
        "set -eo pipefail",
        "echo 'TISSUEKIT_ITEM 1/2'", "false && echo 'TISSUEKIT_ITEM_DONE 1' || echo 'TISSUEKIT_ITEM_FAILED 1'",
        "echo 'TISSUEKIT_ITEM 2/2'", "true && echo 'TISSUEKIT_ITEM_DONE 2' || echo 'TISSUEKIT_ITEM_FAILED 2'",
        "echo TISSUEKIT_DONE",
    ])
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    assert out.split()[-1] == "TISSUEKIT_DONE"
    assert "TISSUEKIT_ITEM_FAILED 1" in out and "TISSUEKIT_ITEM_DONE 2" in out


def test_single_job_script_unchanged_by_refactor():
    cfg = remote.RemoteConfig(user="u")
    text = remote.job_script(cfg, "/r", "egg.tif", "t", run="cellflow", params={"a": 1}, fill_holes=False)
    assert "tissuekit segment egg.tif --preset default --batch-size 8 --run cellflow --params '{\"a\":1}' --no-fill-holes" in text
