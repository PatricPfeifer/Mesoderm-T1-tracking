"""Segmentation runs (one folder per parameter set) and presets."""
import json
from pathlib import Path

import pytest

from napari_tissuekit.core import seg_params
from napari_tissuekit.core.project import LEGACY_RUN, WorkingFolder


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


def _wf(root):
    wf = WorkingFolder()
    wf.set_manual(root)
    return wf


def test_runs_listed_and_found_reads_the_active_one(tmp_path):
    wf = _wf(tmp_path)
    a = _touch(tmp_path / "segmentation" / "default" / "m_cp_masks.tif")
    b = _touch(tmp_path / "segmentation" / "cellflow" / "m_cp_masks.tif")
    _touch(tmp_path / "segmentation" / "cellflow" / "m_cell_outlines.tif")
    (tmp_path / "segmentation" / "empty").mkdir()
    assert wf.seg_runs() == ["cellflow", "default"]
    wf.set_active_seg_run("default")
    assert wf.found()["segmentation"] == [a]
    assert "outlines" not in wf.found()
    wf.set_active_seg_run("cellflow")
    assert wf.found()["segmentation"] == [b]
    assert wf.found(run="default")["segmentation"] == [a]


def test_files_directly_in_segmentation_are_the_original_run(tmp_path):
    wf = _wf(tmp_path)
    old = _touch(tmp_path / "segmentation" / "m_cp_masks.tif")
    _touch(tmp_path / "segmentation" / "cellflow" / "m_cp_masks.tif")
    assert wf.seg_runs() == [LEGACY_RUN, "cellflow"]
    wf.set_active_seg_run(LEGACY_RUN)
    assert wf.found()["segmentation"] == [old]
    assert wf.seg_run_dir(LEGACY_RUN) == tmp_path / "segmentation"


def test_active_run_defaults_to_newest_and_ignores_stale_settings(tmp_path):
    import os
    import time

    wf = _wf(tmp_path)
    old = _touch(tmp_path / "segmentation" / "a" / "m_cp_masks.tif")
    new = _touch(tmp_path / "segmentation" / "b" / "m_cp_masks.tif")
    t = time.time()
    os.utime(old, (t - 100, t - 100))
    os.utime(new, (t, t))
    assert wf.active_seg_run() == "b"
    wf.set_active_seg_run("gone")
    assert wf.active_seg_run() == "b"


def test_new_run_name_never_reuses_a_folder(tmp_path):
    wf = _wf(tmp_path)
    assert wf.new_run_name("cellflow") == "cellflow"
    _touch(tmp_path / "segmentation" / "cellflow" / "m_cp_masks.tif")
    (tmp_path / "segmentation" / "cellflow_2").mkdir()  # e.g. a running job, no masks yet
    assert wf.new_run_name("cellflow") == "cellflow_3"
    assert wf.new_run_name("my run / v2") == "my_run_v2"
    assert wf.new_run_name(LEGACY_RUN) == f"{LEGACY_RUN}_2"


def test_run_params_round_trip(tmp_path):
    wf = _wf(tmp_path)
    wf.write_run_params("x", {"preset": "default", "params": {"flow_threshold": 0.5}})
    got = wf.read_run_params("x")
    assert got["run"] == "x" and got["params"] == {"flow_threshold": 0.5} and "time" in got


def test_full_params_and_eval_kwargs():
    p = seg_params.full_params({"flow_threshold": "0.7", "min_size": 3, "bogus": 1})
    assert p["flow_threshold"] == 0.7 and p["min_size"] == 15 and "bogus" not in p
    kw = seg_params.eval_kwargs({})
    assert kw["diameter"] is None and kw["cellprob_threshold"] == 0.0
    assert "batch_size" not in kw  # passed to eval on its own
    assert seg_params.batch_size({}) == 8
    assert seg_params.eval_kwargs({"diameter": 30})["diameter"] == 30.0


def test_user_presets_saved_matched_and_deleted(tmp_path):
    path = tmp_path / "presets.json"
    seg_params.save_user_preset("strict", {"cellprob_threshold": 1.0}, path)
    presets = seg_params.all_presets(path)
    assert list(presets) == ["default", "cellflow", "hpc", "strict"]
    assert seg_params.matching_preset({"cellprob_threshold": 1.0}, presets) == "strict"
    assert seg_params.matching_preset({}, presets) == "default"
    assert seg_params.matching_preset({"flow_threshold": 2.0}, presets) is None
    with pytest.raises(ValueError):
        seg_params.save_user_preset("cellflow", {}, path)
    seg_params.delete_user_preset("strict", path)
    assert "strict" not in seg_params.all_presets(path)
    assert json.loads(path.read_text()) == {}


def test_segment_cli_resolves_preset_and_overrides(monkeypatch):
    from napari_tissuekit.core import segmentation

    monkeypatch.setattr(seg_params, "USER_PRESETS_PATH", seg_params.USER_PRESETS_PATH.parent / "_none_.json")
    kw = segmentation.resolve_params("cellflow", '{"flow_threshold": 1.0}')
    assert kw["flow_threshold"] == 1.0 and kw["cellprob_threshold"] == 0.3 and kw["diameter"] is None
    assert segmentation.resolve_params("from_other_pc", {"min_size": 40})["min_size"] == 40
    with pytest.raises(SystemExit):
        segmentation.resolve_params("nope")


def test_hpc_preset_only_raises_the_batch_size(monkeypatch):
    from napari_tissuekit.core import segmentation

    monkeypatch.setattr(seg_params, "USER_PRESETS_PATH", seg_params.USER_PRESETS_PATH.parent / "_none_.json")
    hpc = segmentation.resolve_all("hpc")
    assert hpc["batch_size"] == 64
    assert segmentation.resolve_params("hpc") == segmentation.resolve_params("default")
    assert segmentation.resolve_all("hpc", '{"batch_size": 32}')["batch_size"] == 32


def test_cli_segment_uses_preset_batch_size_and_records_run(tmp_path, monkeypatch):
    from napari_tissuekit import cli
    from napari_tissuekit.core import segmentation

    monkeypatch.setattr(seg_params, "USER_PRESETS_PATH", tmp_path / "_none_.json")
    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        out = Path(argv[argv.index("--output-dir") + 1])
        (out / "m_cp_masks.tif").write_bytes(b"")

    monkeypatch.setattr(segmentation, "main", fake_main)
    movie = _touch(tmp_path / "m.tif")
    cli.main(["segment", str(movie), "--preset", "hpc"])
    argv = seen["argv"]
    assert argv[argv.index("--batch-size") + 1] == "64"
    wf = _wf(tmp_path / "m_results")
    assert wf.seg_runs() == ["hpc"] and wf.active_seg_run() == "hpc"
    assert wf.read_run_params("hpc")["params"]["batch_size"] == 64
    cli.main(["segment", str(movie), "--preset", "hpc", "--batch-size", "16"])
    assert seen["argv"][seen["argv"].index("--batch-size") + 1] == "16"
    assert wf.seg_runs() == ["hpc", "hpc_2"]
