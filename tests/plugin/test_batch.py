"""Batch tab core (core/batch.py) and the batch job script (core/remote.py), no cluster."""
from __future__ import annotations

import json
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


def test_profiles_with_equal_values_each_show_their_own_run(tmp_path):
    egg = _movie(tmp_path, "egg1.tif")
    _run(egg, "Custom1", CELLFLOW, preset="Custom1")
    _run(egg, "Custom2", CELLFLOW, preset="Custom2").set_active_seg_run("Custom2")
    st = batch.egg_status(egg)
    assert st.active_run == "Custom2"
    assert st.run_for(CELLFLOW, "Custom1").name == "Custom1"
    assert st.run_for(CELLFLOW, "Custom2").name == "Custom2"
    assert st.run_for(CELLFLOW, "Custom3").name == "Custom2"  # no own run: the active one


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


# ---- the batch round trip on a fake cluster ---------------------------------------------

from test_remote import FakeSession  # noqa: E402  (tests/plugin is on sys.path under pytest)


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "CONFIG_PATH", tmp_path / "cfg" / "remote.json")
    monkeypatch.setattr(remote, "JOBS_PATH", tmp_path / "cfg" / "remote_jobs.json")


class BatchCluster(FakeSession):
    """sbatch writes every item's results; each squeue check reveals one more item in the log.

    ``fail``: item numbers whose `tissuekit segment` fails (no results, FAILED marker).
    The full log waits in ``full.log`` so a later session (fetch) sees the same job.
    """

    def __init__(self, disk, states, fail=(), reveal_all=False, **kw):
        super().__init__(disk, states, **kw)
        self.fail = set(fail)
        self.reveal_all = reveal_all
        self.gets = 0
        orig_get = self.sftp.get

        def get(src, dst):
            self.gets += 1
            orig_get(src, dst)
        self.sftp.get = get

    def _fake_results(self, remote_dir):
        import re
        import shlex

        d = self.sftp.local(remote_dir)
        blocks = []
        for line in self.script.splitlines():
            if not line.startswith("tissuekit segment"):
                continue
            words = shlex.split(line.split(" && ")[0])
            i = len(blocks) + 1
            movie, run = words[2], words[words.index("--run") + 1]
            stem = movie.rsplit(".", 1)[0]
            ok = i not in self.fail
            if ok:
                seg = d / f"{stem}_results" / "segmentation" / run
                seg.mkdir(parents=True)
                (seg / f"{stem}_cp_masks.tif").write_bytes(b"masks")
                (seg / "params.json").write_text(json.dumps({"run": run}))
            blocks.append(f"TISSUEKIT_ITEM {i}/N {movie} {run}\nframe 2/2\n"
                          f"TISSUEKIT_ITEM_{'DONE' if ok else 'FAILED'} {i}\n")
        assert re.search(r"#SBATCH --time=", self.script)
        (d / "full.log").write_text("".join(blocks))
        (d / "slurm-4242.log").write_text("")
        self.remote_dir = remote_dir

    def _reveal(self):
        d = self.sftp.local(self.remote_dir)
        blocks = (d / "full.log").read_text().split("TISSUEKIT_ITEM ")[1:]
        k = len(blocks) if self.reveal_all else min(self.polls, len(blocks))
        (d / "slurm-4242.log").write_text("".join("TISSUEKIT_ITEM " + b for b in blocks[:k]))

    def run(self, command, timeout=120, check=True):
        if command.startswith("squeue"):
            out = super().run(command, timeout, check)
            self._reveal()
            return out
        if command.startswith("grep"):
            log = self.sftp.local(command.split()[3].strip("'"))
            return "".join(ln + "\n" for ln in log.read_text().splitlines() if ln.startswith("TISSUEKIT_ITEM"))
        return super().run(command, timeout, check)


def _eggs(tmp_path):
    import numpy as np
    import tifffile

    folder = tmp_path / "eggs"
    folder.mkdir()
    for n in ("egg1", "egg2"):
        tifffile.imwrite(folder / f"{n}.tif", np.zeros((2, 8, 8), np.uint8))
    items, _ = batch.plan_items(batch.scan_folder(folder),
                                {"default": full_params({}), "cellflow": full_params(CELLFLOW)})
    return folder, items


def _connect(monkeypatch, session):
    monkeypatch.setattr(remote.Session, "connect", classmethod(lambda cls, *a, **k: session))


def test_batch_downloads_each_item_as_it_finishes(tmp_path, monkeypatch):
    folder, items = _eggs(tmp_path)
    cluster = BatchCluster(tmp_path / "cluster", ["PENDING"] + ["RUNNING"] * 4, fail={3})
    _connect(monkeypatch, cluster)
    seen = []
    result = remote.run_batch(remote.RemoteConfig(user="u", poll_seconds=0), "pw", "1", items,
                              lambda p, m: seen.append(m), sleep=lambda _s: None)

    assert cluster.script.count("tissuekit segment") == 4
    assert len([c for c in cluster.commands if c.startswith("sbatch")]) == 1  # one job
    # egg1 · default came in while egg1 · cellflow was still to run
    first = seen.index("downloaded item 1/4 egg1 · default")
    assert first < next(i for i, m in enumerate(seen) if m.startswith("downloaded item 2/4"))
    assert "item 3/4 egg2 · default: FAILED (see Log...)" in seen
    assert [it["run"] for it in result.downloaded] == ["default", "cellflow", "cellflow"]
    assert [(it["movie_name"], it["profile"]) for it in result.failed] == [("egg2.tif", "default")]
    assert (folder / "egg1_results" / "segmentation" / "cellflow" / "egg1_cp_masks.tif").is_file()
    assert not (folder / "egg2_results" / "segmentation" / "default").exists()
    assert remote.load_jobs() == []  # collected, so forgotten
    assert result.note == ""
    # each movie uploaded once
    assert sorted(p.name for p in cluster.sftp.local(result.job.remote_dir).glob("*.tif")) == ["egg1.tif", "egg2.tif"]


def test_batch_time_limit_keeps_finished_items(tmp_path, monkeypatch):
    folder, items = _eggs(tmp_path)
    cluster = BatchCluster(tmp_path / "cluster", ["RUNNING", "RUNNING", "TIMEOUT"])
    _connect(monkeypatch, cluster)
    result = remote.run_batch(remote.RemoteConfig(user="u", poll_seconds=0), "pw", "1", items,
                              lambda *_: None, sleep=lambda _s: None)
    assert "TIMEOUT" in result.note
    assert len(result.downloaded) == 3  # the final check still collects item 3
    assert result.failed[0]["note"].startswith("did not run")


def test_batch_lost_connection_then_fetch_downloads_only_the_rest(tmp_path, monkeypatch):
    folder, items = _eggs(tmp_path)
    cluster = BatchCluster(tmp_path / "cluster", ["RUNNING"] * 9, drop_after=2)
    _connect(monkeypatch, cluster)
    with pytest.raises(remote.Disconnected):
        remote.run_batch(remote.RemoteConfig(user="u", poll_seconds=0), "pw", "1", items,
                         lambda *_: None, sleep=lambda _s: None)
    (job,) = remote.load_jobs()
    assert job.is_batch and job.label == "batch (4 items, 2 eggs)"
    assert [it["status"] for it in job.items] == ["downloaded", "downloaded", "pending", "pending"]

    later = BatchCluster(tmp_path / "cluster", [], reveal_all=True, final="COMPLETED")
    later.remote_dir = job.remote_dir
    later._reveal()
    _connect(monkeypatch, later)
    result = remote.fetch_job(remote.RemoteConfig(user="u", poll_seconds=0), job, lambda *_: None, "pw", "2",
                              sleep=lambda _s: None)
    assert len(result.downloaded) == 4 and not result.failed
    assert later.gets == 4  # items 3 and 4: masks + params.json each, nothing twice
    assert remote.load_jobs() == []


def test_batch_refuses_a_too_long_dev_job(tmp_path):
    _folder, items = _eggs(tmp_path)
    cfg = remote.RemoteConfig(user="u", partition="dev_gpu_h100", time="00:10:00")
    with pytest.raises(remote.RemoteError, match="gpu_h100"):
        remote.run_batch(cfg, "pw", "1", items, lambda *_: None)


def test_old_jobs_file_without_items_still_loads(tmp_path):
    remote.JOBS_PATH.parent.mkdir(parents=True)
    remote.JOBS_PATH.write_text(json.dumps([{"job_id": "7", "tag": "t", "remote_dir": "/r",
                                             "movie_name": "egg1.tif", "local_results": "C:/x"}]))
    (job,) = remote.load_jobs()
    assert not job.is_batch and job.label == "egg1.tif"


def test_single_job_script_unchanged_by_refactor():
    cfg = remote.RemoteConfig(user="u")
    text = remote.job_script(cfg, "/r", "egg.tif", "t", run="cellflow", params={"a": 1}, fill_holes=False)
    assert "tissuekit segment egg.tif --preset default --batch-size 8 --run cellflow --params '{\"a\":1}' --no-fill-holes" in text
