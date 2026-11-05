"""Cluster round trip (core/remote.py) without a cluster.

* A fake SSH session (local folder as the cluster disk, scripted Slurm answers) checks
  upload -> sbatch -> poll -> download, the job file, and the lost-connection path.
* A real paramiko SSH server on localhost checks the two-factor login (OTP + password
  keyboard-interactive prompts like bwHPC) and the trust-on-first-use host key.
"""
from __future__ import annotations

import json
import posixpath
import socket
import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_tissuekit.core import remote


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "CONFIG_PATH", tmp_path / "cfg" / "remote.json")
    monkeypatch.setattr(remote, "JOBS_PATH", tmp_path / "cfg" / "remote_jobs.json")
    monkeypatch.setattr(remote, "KNOWN_HOSTS", tmp_path / "cfg" / "known_hosts")


# ---- fake cluster --------------------------------------------------------------

class FakeSftp:
    """SFTP onto a local folder; cluster path /home/u/x -> <disk>/home/u/x."""

    def __init__(self, disk: Path):
        self.disk = disk

    def local(self, path: str) -> Path:
        return self.disk / path.lstrip("/")

    def normalize(self, _path):
        return "/home/u"

    def put(self, src, dst, callback=None):
        data = Path(src).read_bytes()
        self.local(dst).parent.mkdir(parents=True, exist_ok=True)
        self.local(dst).write_bytes(data)
        if callback:
            callback(len(data), len(data))

    def open(self, path, mode="r"):
        self.local(path).parent.mkdir(parents=True, exist_ok=True)
        return open(self.local(path), mode)

    def listdir_attr(self, path):
        d = self.local(path)
        if not d.is_dir():
            raise FileNotFoundError(path)
        return [SimpleNamespace(filename=p.name, st_mode=p.stat().st_mode) for p in d.iterdir()]

    def putfo(self, fo, dst):
        self.local(dst).parent.mkdir(parents=True, exist_ok=True)
        self.local(dst).write_bytes(fo.read())

    def get(self, src, dst):
        Path(dst).write_bytes(self.local(src).read_bytes())

    def close(self):
        pass


class FakeSession(remote.Session):
    """Answers mkdir / sbatch / squeue / sacct / tail like a tiny Slurm."""

    def __init__(self, disk: Path, states: list[str], final: str = "COMPLETED", drop_after: int | None = None):
        super().__init__(transport=None, sftp=FakeSftp(disk))
        self.disk = disk
        self.states = list(states)
        self.final = final
        self.drop_after = drop_after
        self.polls = 0
        self.commands: list[str] = []
        self.script: str | None = None
        self._alive = True
        self.env_ok = True
        self.uploads = 0

    @property
    def alive(self):
        return self._alive

    def close(self):
        pass

    def run(self, command, timeout=120, check=True):
        self.commands.append(command)
        if command.startswith("test -x"):
            return "ENV_OK\n" if self.env_ok else ""
        if command.startswith("cat "):  # the plugin fingerprints on the cluster
            plugin = self.sftp.local("/home/u/tissuekit_remote/plugin")
            read = lambda n: (plugin / n).read_text().strip() if (plugin / n).is_file() else ""
            return f"{read('.code_hash')}\n|\n{read('.deps_hash')}\n"
        if command.startswith("cd ") and "tar xzf" in command:
            import tarfile

            plugin = self.sftp.local("/home/u/tissuekit_remote/plugin")
            with tarfile.open(plugin / ".upload.tgz") as tar:
                tar.extractall(plugin)
            (plugin / ".upload.tgz").unlink()
            (plugin / ".code_hash").write_text(command.split("echo ")[-1].split()[0].strip("'"))
            self.uploads += 1
            return ""
        if command.startswith("mkdir -p"):
            self.sftp.local(command.split()[-1].strip("'")).mkdir(parents=True, exist_ok=True)
            return ""
        if command.startswith("sbatch"):
            script = command.split()[-1].strip("'")
            self.script = self.sftp.local(script).read_text()
            self._fake_results(posixpath.dirname(script))
            return "4242\n"
        if command.startswith("squeue"):
            self.polls += 1
            if self.drop_after is not None and self.polls > self.drop_after:
                self._alive = False
                raise EOFError("connection lost")
            return (self.states.pop(0) + "\n") if self.states else ""
        if command.startswith("sacct"):
            return f"{self.final}\n"
        if command.startswith("tail"):
            log = self.sftp.local(command.split()[3].strip("'"))
            if log.is_file():
                return log.read_text()
            return "loading model cpsam (gpu=True)\nframe 3/10\n  labels=120\n"
        return ""

    def _fake_results(self, remote_dir: str):
        movie = next((p for p in self.sftp.local(remote_dir).iterdir() if p.suffix == ".tif"), None)
        if movie is None:  # the setup job
            (self.sftp.local(remote_dir) / "slurm-4242.log").write_text(
                "step 4/4: installing the plugin and Cellpose\n"
                "torch 2.9.0 cuda 12.8 tissuekit 0.1\nTISSUEKIT_SETUP_DONE\n"
            )
            return
        import re

        run = re.search(r"--run (\S+)", self.script)  # like `tissuekit segment --run`
        seg = movie.parent / f"{movie.stem}_results" / "segmentation" / (run.group(1).strip("'") if run else "")
        seg.mkdir(parents=True)
        (seg / f"{movie.stem}_cp_masks.tif").write_bytes(b"masks")
        (seg / f"{movie.stem}_cell_outlines.tif").write_bytes(b"outlines")


def _cfg(**kw) -> remote.RemoteConfig:
    return remote.RemoteConfig(user="ho_test", poll_seconds=0, **kw)


def _movie(tmp_path) -> Path:
    movie = tmp_path / "local" / "egg1.tif"
    movie.parent.mkdir()
    import numpy as np
    import tifffile

    tifffile.imwrite(movie, np.zeros((3, 100, 200), np.uint16))  # 3 frames, 0.06 Mpx total
    return movie


# ---- pure helpers ------------------------------------------------------------------

def test_answers_otp_and_password_by_prompt():
    prompts = [("Your OTP: ", False), ("Password: ", False)]
    assert remote.answer_prompts(prompts, "pw", "123456") == ["123456", "pw"]
    assert remote.answer_prompts([("Password:", False)], "pw", "1") == ["pw"]


def test_log_progress_reads_last_frame():
    assert remote.log_progress("frame 1/10\nframe 4/10\n") == (40, "frame 4/10")
    assert remote.log_progress("loading model") is None


def test_progress_follows_the_job_phases():
    """The status/progress bar tells what the job does now, read from the real log format."""
    p = remote._seg_progress
    assert p("") is None
    assert p("GPU 0: NVIDIA A100\nloading /x/egg1.tif\nshape (11, 291, 291) uint16\n") == (0, "reading the movie")
    assert p("starting fresh: 11 frames\nloading model cpsam (gpu=True)\n") == (0, "loading the Cellpose model on the GPU")
    assert p("loading model cpsam (gpu=True)\nframe 4/11\n  labels=812\n") == (36, "step 1/2 Cellpose on the GPU: frame 4/11")
    # the outline phase also prints "frame a/b"; it must not look like segmentation
    log = "frame 11/11\n  labels=800\nwrote /x/egg1_cp_masks.tif max_label 830\n  thinning bonds, frame 3/11 (27%)\n"
    assert p(log) == (27, "step 2/2 outlines: thinning bonds, frame 3/11")
    assert p("frame 11/11\nwrote /x/egg1_cp_masks.tif max_label 830\n")[1].startswith("step 2/2 outlines")
    assert p(log + "wrote /x/egg1_cell_outlines.tif\nTISSUEKIT_DONE\n")[0] == 100
    assert remote._setup_progress("step 3/4: installing torch (CUDA build)\nCollecting torch\n") == (
        50, "cluster setup: installing torch (CUDA build)")


def test_job_script_gpu_env_and_run_folder():
    cfg = _cfg(partition="dev_gpu_h100", time="00:20:00")
    text = remote.job_script(cfg, "/home/u/tissuekit_remote/egg 1_x", "egg 1.tif", "egg_1_x", run="default_2")
    assert "#SBATCH --partition=dev_gpu_h100" in text
    assert "#SBATCH --time=00:20:00" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "source $HOME/miniconda3/etc/profile.d/conda.sh" in text
    assert "tissuekit segment 'egg 1.tif' --preset default --batch-size 8 --run default_2" in text
    assert "GPU usable" in text
    assert "rclone" not in text  # no bwSync&Share: results come back over SSH only






def test_config_and_jobs_roundtrip():
    cfg = _cfg(partition="cpu")
    cfg.save()
    assert remote.RemoteConfig.load().partition == "cpu"
    job = remote.RemoteJob("1", "t", "/home/u/r/t", "m.ome.tif", "C:/x")
    remote.remember_job(job)
    assert [j.job_id for j in remote.load_jobs()] == ["1"]
    assert job.remote_seg_dir == "/home/u/r/t/m_results/segmentation"
    remote.forget_job("1")
    assert remote.load_jobs() == []


# ---- the round trip ------------------------------------------------------------------

def test_one_click_round_trip(tmp_path, monkeypatch):
    session = FakeSession(tmp_path / "cluster", ["PENDING", "RUNNING", "RUNNING"])
    monkeypatch.setattr(remote.Session, "connect", classmethod(lambda cls, *a, **k: session))
    movie = _movie(tmp_path)
    results = tmp_path / "local" / "egg1_results"
    seen, logs = [], []

    job, masks = remote.run_segmentation(
        _cfg(), "pw", "123456", movie, results, lambda p, m: seen.append((p, m)),
        sleep=lambda _s: None, on_log=logs.append,
    )
    assert logs and "frame 3/10" in logs[-1]  # the live log window gets the job log

    run_dir = results / "segmentation" / "default"  # a run folder, like a local run
    assert masks == run_dir / "egg1_cp_masks.tif"
    assert masks.read_bytes() == b"masks"
    assert (run_dir / "egg1_cell_outlines.tif").is_file()
    assert "--run default" in session.script
    assert json.loads((results / "settings.json").read_text())["active_segmentation_run"] == "default"
    assert job.job_id == "4242"
    assert (session.sftp.local(job.remote_dir) / "egg1.tif").read_bytes() == movie.read_bytes()
    assert "tissuekit segment egg1.tif" in session.script
    assert any("waiting in the queue" in m for _p, m in seen)
    assert (30, "step 1/2 Cellpose on the GPU: frame 3/10") in seen
    assert remote.load_jobs() == []  # collected, so forgotten

    # a second cluster run never overwrites the first one
    session2 = FakeSession(tmp_path / "cluster2", ["RUNNING"])
    monkeypatch.setattr(remote.Session, "connect", classmethod(lambda cls, *a, **k: session2))
    _job2, masks2 = remote.run_segmentation(_cfg(), "pw", "1", movie, results, lambda *_: None,
                                            sleep=lambda _s: None)
    assert masks2 == results / "segmentation" / "default_2" / "egg1_cp_masks.tif"
    assert masks.is_file()


def test_failed_job_reports_state(tmp_path, monkeypatch):
    session = FakeSession(tmp_path / "cluster", ["RUNNING", "OUT_OF_MEMORY"])
    monkeypatch.setattr(remote.Session, "connect", classmethod(lambda cls, *a, **k: session))
    with pytest.raises(remote.RemoteError, match="OUT_OF_MEMORY"):
        remote.run_segmentation(_cfg(), "pw", "1", _movie(tmp_path), tmp_path / "r",
                                lambda *_: None, sleep=lambda _s: None)


def test_lost_connection_keeps_job_for_fetch(tmp_path, monkeypatch):
    session = FakeSession(tmp_path / "cluster", ["RUNNING"] * 5, drop_after=2)
    monkeypatch.setattr(remote.Session, "connect", classmethod(lambda cls, *a, **k: session))
    results = tmp_path / "local" / "egg1_results"
    with pytest.raises(remote.Disconnected):
        remote.run_segmentation(_cfg(), "pw", "1", _movie(tmp_path), results,
                                lambda *_: None, sleep=lambda _s: None)
    (job,) = remote.load_jobs()

    assert job.run == "default"  # the run name is remembered with the job

    # later (new login): the job is done -> download into the same run folder
    later = FakeSession(tmp_path / "cluster", [], final="COMPLETED")
    monkeypatch.setattr(remote.Session, "connect", classmethod(lambda cls, *a, **k: later))
    masks = remote.fetch_job(_cfg(), job, lambda *_: None, "pw", "654321")
    assert masks == results / "segmentation" / "default" / "egg1_cp_masks.tif"
    assert remote.load_jobs() == []


def test_fetch_keeps_watching_a_running_job(tmp_path, monkeypatch):
    session = FakeSession(tmp_path / "cluster", ["RUNNING"] * 5, drop_after=1)
    monkeypatch.setattr(remote.Session, "connect", classmethod(lambda cls, *a, **k: session))
    results = tmp_path / "local" / "egg1_results"
    with pytest.raises(remote.Disconnected):
        remote.run_segmentation(_cfg(), "pw", "1", _movie(tmp_path), results,
                                lambda *_: None, sleep=lambda _s: None)
    (job,) = remote.load_jobs()

    # Fetch while it still runs: one login, it is watched again (no "try later"), then downloaded
    later = FakeSession(tmp_path / "cluster", ["RUNNING", "RUNNING", "RUNNING"], final="COMPLETED")
    monkeypatch.setattr(remote.Session, "connect", classmethod(lambda cls, *a, **k: later))
    seen = []
    masks = remote.fetch_job(_cfg(), job, lambda p, m: seen.append(m), "pw", "2", sleep=lambda _s: None)
    assert masks.read_bytes() == b"masks"
    assert any("watching it again" in m for m in seen)
    assert later.polls >= 3
    assert remote.load_jobs() == []


def test_fetch_keeps_job_when_results_are_missing(tmp_path, monkeypatch):
    job = remote.RemoteJob("9", "t", "/home/u/tissuekit_remote/t", "egg1.tif", str(tmp_path / "r"), run="default")
    remote.remember_job(job)
    empty = FakeSession(tmp_path / "cluster", [], final="COMPLETED")
    empty.sftp.local(job.remote_dir).mkdir(parents=True)
    monkeypatch.setattr(remote.Session, "connect", classmethod(lambda cls, *a, **k: empty))
    with pytest.raises(remote.RemoteError, match="no \\*_cp_masks.tif"):
        remote.fetch_job(_cfg(), job, lambda *_: None, "pw", "1")
    assert [j.job_id for j in remote.load_jobs()] == ["9"]  # can be fetched again


# ---- keeping the cluster code up to date ---------------------------------------------

@pytest.fixture
def plugin_src(tmp_path, monkeypatch):
    src = tmp_path / "plugin_src"
    (src / "napari_tissuekit" / "__pycache__").mkdir(parents=True)
    (src / "pyproject.toml").write_text('dependencies = ["numpy"]\n')
    (src / "napari_tissuekit" / "__init__.py").write_text("__version__ = '1'\n")
    (src / "napari_tissuekit" / "__pycache__" / "x.pyc").write_bytes(b"cache")
    monkeypatch.setattr(remote, "PLUGIN_DIR", src)
    return src


def _sync(session):
    return remote.sync_plugin(_cfg(), session, lambda *_: None)


def test_sync_uploads_only_changes(tmp_path, plugin_src):
    session = FakeSession(tmp_path / "cluster", [])
    plugin = session.sftp.local("/home/u/tissuekit_remote/plugin")
    (plugin).mkdir(parents=True)
    (plugin / ".deps_hash").write_text(remote.plugin_fingerprint()[1])  # as after setup

    state = _sync(session)
    assert session.uploads == 1 and not state.needs_install
    assert (plugin / "napari_tissuekit" / "__init__.py").read_text() == "__version__ = '1'\n"
    assert not (plugin / "napari_tissuekit" / "__pycache__").exists()

    _sync(session)
    assert session.uploads == 1  # nothing changed: nothing sent

    (plugin_src / "napari_tissuekit" / "__init__.py").write_text("__version__ = '2'\n")
    state = _sync(session)
    assert session.uploads == 2 and not state.needs_install  # code only: no pip
    assert (plugin / "napari_tissuekit" / "__init__.py").read_text() == "__version__ = '2'\n"

    (plugin_src / "pyproject.toml").write_text('dependencies = ["numpy", "zarr"]\n')
    state = _sync(session)
    assert state.needs_install
    script = remote.job_script(_cfg(), "/d", "m.tif", "t", state)
    assert "pip install --no-cache-dir -q -e /home/u/tissuekit_remote/plugin" in script
    assert script.index("pip install") < script.index("tissuekit segment")
    assert "pip install" not in remote.job_script(_cfg(), "/d", "m.tif", "t", None)


def test_run_refuses_without_env(tmp_path, monkeypatch, plugin_src):
    session = FakeSession(tmp_path / "cluster", [])
    session.env_ok = False
    monkeypatch.setattr(remote.Session, "connect", classmethod(lambda cls, *a, **k: session))
    with pytest.raises(remote.RemoteError, match="Set up cluster"):
        remote.run_segmentation(_cfg(), "pw", "1", _movie(tmp_path), tmp_path / "r", lambda *_: None)
    assert not any(c.startswith("sbatch") for c in session.commands)


def test_setup_cluster_submits_env_install(tmp_path, monkeypatch, plugin_src):
    session = FakeSession(tmp_path / "cluster", ["PENDING", "RUNNING"])
    session.env_ok = False
    monkeypatch.setattr(remote.Session, "connect", classmethod(lambda cls, *a, **k: session))
    seen = []
    summary = remote.setup_cluster(_cfg(), "pw", "1", lambda p, m: seen.append((p, m)), sleep=lambda _s: None)
    assert summary.startswith("torch 2.9.0 cuda 12.8")
    assert "#SBATCH --partition=cpu" in session.script
    assert "pip install --no-cache-dir -q -e /home/u/tissuekit_remote/plugin" in session.script
    assert session.uploads == 1
    assert (75, "cluster setup: installing the plugin and Cellpose") in seen
    assert remote.load_jobs() == []  # setup is not a fetchable result


# ---- real SSH login with OTP + password ---------------------------------------------

def _ssh_server(host_key):
    paramiko = pytest.importorskip("paramiko")

    class Server(paramiko.ServerInterface):
        def get_allowed_auths(self, username):
            return "keyboard-interactive"

        def check_auth_interactive(self, username, submethods):
            query = paramiko.InteractiveQuery()
            query.add_prompt("Your OTP: ", False)
            query.add_prompt("Password: ", False)
            return query

        def check_auth_interactive_response(self, responses):
            ok = list(responses) == ["123456", "secret"]
            return paramiko.AUTH_SUCCESSFUL if ok else paramiko.AUTH_FAILED

        def check_channel_request(self, kind, chanid):
            return paramiko.OPEN_SUCCEEDED

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(5)
    transports = []

    def serve():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            t = paramiko.Transport(conn)
            t.add_server_key(host_key)
            t.start_server(server=Server())
            transports.append(t)

    threading.Thread(target=serve, daemon=True).start()
    return listener, transports


def _login(port, password, otp):
    """Session.connect against the local server; True if logged in."""
    cfg = remote.RemoteConfig(host="127.0.0.1", port=port, user="ho_test")
    try:
        session = remote.Session.connect(cfg, password, otp, timeout=10)
    except remote.RemoteError as exc:
        if "Login refused" in str(exc):
            return False
        raise
    ok = session.transport.is_authenticated()
    session.transport.close()
    return ok


def test_two_factor_login_and_host_key(tmp_path, monkeypatch):
    paramiko = pytest.importorskip("paramiko")
    # the test server has no SFTP subsystem; the login is what is tested here
    monkeypatch.setattr(paramiko.SFTPClient, "from_transport", classmethod(lambda cls, t: SimpleNamespace(close=lambda: None)))
    key = paramiko.RSAKey.generate(2048)
    listener, transports = _ssh_server(key)
    port = listener.getsockname()[1]
    try:
        assert _login(port, "secret", "123456") is True
        assert remote.KNOWN_HOSTS.is_file()  # remembered on first use
        assert _login(port, "secret", "000000") is False  # wrong OTP

        other = paramiko.RSAKey.generate(2048)
        listener2, transports2 = _ssh_server(other)
        port2 = listener2.getsockname()[1]
        # pretend the second server is the same host:port as the first one
        name = f"[127.0.0.1]:{port2}"
        known = paramiko.HostKeys(str(remote.KNOWN_HOSTS))
        known.add(name, key.get_name(), key)
        known.save(str(remote.KNOWN_HOSTS))
        with pytest.raises(remote.RemoteError, match="CHANGED"):
            _login(port2, "secret", "123456")
        listener2.close()
    finally:
        listener.close()
        for t in transports:
            t.close()


# ---- queue defaults, time left, reconnecting ------------------------------------





def test_time_left_is_shown_while_running(tmp_path, monkeypatch):
    clock = iter(range(0, 10_000, 30))
    monkeypatch.setattr(remote.time, "time", lambda: float(next(clock)))
    session = FakeSession(tmp_path / "cluster", ["RUNNING", "RUNNING", "RUNNING"])
    logs = iter(["frame 1/10\n", "frame 1/10\nframe 2/10\n", "frame 4/10\n"])
    monkeypatch.setattr(remote, "_tail", lambda *_a, **_k: next(logs, "frame 4/10\n"))
    seen = []
    job = remote.RemoteJob("4242", "t", "/home/u/r/t", "m.tif", str(tmp_path))
    session.final = "COMPLETED"
    remote.wait_for(session, job, lambda p, m: seen.append(m), 0, lambda _s: None)
    assert any("left" in m for m in seen), seen



def test_missing_masks_keeps_job_and_explains(tmp_path, monkeypatch):
    session = FakeSession(tmp_path / "cluster", ["RUNNING"])
    session._fake_results = lambda remote_dir: None  # the job writes nothing
    monkeypatch.setattr(remote.Session, "connect", classmethod(lambda cls, *a, **k: session))
    with pytest.raises(remote.RemoteError, match="no \*_cp_masks.tif"):
        remote.run_segmentation(_cfg(), "pw", "1", _movie(tmp_path), tmp_path / "local" / "r",
                                lambda *_: None, sleep=lambda _s: None)
    assert len(remote.load_jobs()) == 1  # still fetchable


def test_fixed_default_queue_and_time_limit():
    cfg = remote.RemoteConfig()
    assert (cfg.partition, cfg.time) == ("dev_gpu_h100", "00:05:00")
    text = remote.job_script(cfg, "/d", "m.tif", "t")
    assert "#SBATCH --partition=dev_gpu_h100" in text and "#SBATCH --time=00:05:00" in text
    # settings saved by earlier versions of this feature move to the new default
    remote.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    for old in ({"partition": "gpu_a100_short", "time": "00:30:00"}, {"partition": "auto", "time": "auto"}):
        remote.CONFIG_PATH.write_text(json.dumps({"user": "u", "cloud_remote": "bwsync", **old}))
        cfg = remote.RemoteConfig.load()  # unknown old keys (bwSync&Share) are ignored
        assert (cfg.partition, cfg.time, cfg.user) == ("dev_gpu_h100", "00:05:00", "u")
    # a value the user chose in Settings stays
    remote.CONFIG_PATH.write_text(json.dumps({"user": "u", "partition": "gpu_h100", "time": "02:00:00"}))
    assert (remote.RemoteConfig.load().partition, remote.RemoteConfig.load().time) == ("gpu_h100", "02:00:00")


def test_timeout_says_how_to_fix_it(tmp_path):
    session = FakeSession(tmp_path / "cluster", ["RUNNING", "TIMEOUT"])
    job = remote.RemoteJob("4242", "t", "/home/u/r/t", "m.tif", str(tmp_path))
    with pytest.raises(remote.RemoteError, match="Time limit"):
        remote.wait_for(session, job, lambda *_: None, 0, lambda _s: None)
