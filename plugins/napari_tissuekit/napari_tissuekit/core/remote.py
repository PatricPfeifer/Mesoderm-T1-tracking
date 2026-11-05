"""Run segmentation on the bwUniCluster GPU from the desktop (no Qt in here).

One call does the whole round trip::

    log in (password + OTP, once) -> upload the movie (SFTP) -> sbatch a GPU job
    that runs ``tissuekit segment`` -> poll the job -> download ``segmentation/``

The password and OTP are only passed through to the SSH login and never stored.
Non-secret settings (user, partition, conda env, ...) live in
``~/.napari_tissuekit/remote.json``.

If the connection drops while the job runs (laptop asleep, Wi-Fi), the job keeps
running on the cluster. It is remembered in ``~/.napari_tissuekit/remote_jobs.json``
and :func:`fetch_job` (one new SSH login) keeps watching it and downloads the results.

Cluster env: :func:`setup_cluster` (one time) submits a CPU job that installs Miniforge
and a conda env with torch, Cellpose and this plugin (editable, from the uploaded copy).
Every run then calls :func:`sync_plugin`: the desktop's plugin code is sent to the cluster
when its fingerprint changed (a few seconds, only unpacking files), and when
``pyproject.toml`` changed (new dependencies) the GPU job re-runs ``pip install`` first.
So the cluster always runs exactly the code that is on this computer.
"""
from __future__ import annotations

import hashlib
import io
import json
import posixpath
import re
import shlex
import socket
import tarfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .project import movie_stem

CONFIG_DIR = Path.home() / ".napari_tissuekit"
CONFIG_PATH = CONFIG_DIR / "remote.json"
JOBS_PATH = CONFIG_DIR / "remote_jobs.json"
KNOWN_HOSTS = CONFIG_DIR / "known_hosts"
# The plugin source folder (holds pyproject.toml) when running from a checkout.
PLUGIN_DIR = Path(__file__).resolve().parents[2]
MINIFORGE_URL = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"

DONE_STATES = {"COMPLETED"}
FAILED_STATES = {
    "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
    "PREEMPTED", "BOOT_FAIL", "DEADLINE",
}

Progress = Callable[[int, str], None]


class RemoteError(RuntimeError):
    """Something went wrong on the cluster side; the message is meant for the user."""


class Disconnected(RemoteError):
    """The SSH connection was lost while the job keeps running on the cluster."""


@dataclass
class RemoteConfig:
    host: str = "uc3.scc.kit.edu"
    port: int = 22
    user: str = ""                       # e.g. ho_abcd123
    remote_root: str = "tissuekit_remote"  # relative = inside the cluster home
    conda_sh: str = "~/miniconda3/etc/profile.d/conda.sh"
    conda_env: str = "tissuekit"
    # Dev GPU queue: starts within minutes, max 30 min, one running job at a time
    # (bwUniCluster 3.0 batch queue table). 5 min is plenty for one movie; raise it in Settings
    # (or pick gpu_h100, up to 72 h) for very long movies.
    partition: str = "dev_gpu_h100"
    gres: str = "gpu:1"
    cpus: int = 8
    mem: str = "32G"
    time: str = "00:05:00"
    preset: str = "default"
    batch_size: int = 8
    poll_seconds: int = 20
    delete_remote_after: bool = False
    setup_partition: str = "cpu"         # one-time env install runs here
    setup_time: str = "01:00:00"

    @classmethod
    def load(cls, path: Path | None = None) -> "RemoteConfig":
        path = path or CONFIG_PATH
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        names = {f.name for f in fields(cls)}
        if data.get("partition") in ("auto", "gpu_a100_short"):  # earlier defaults of this feature
            data.update(partition=cls.partition, time=cls.time)
        elif data.get("time") == "auto":
            data["time"] = cls.time
        return cls(**{k: v for k, v in data.items() if k in names})

    def save(self, path: Path | None = None) -> None:
        path = path or CONFIG_PATH
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


@dataclass
class RemoteJob:
    """A submitted job, remembered until its results are on this computer."""

    job_id: str
    tag: str                  # folder name on the cluster
    remote_dir: str           # absolute cluster folder holding the movie and the results
    movie_name: str           # e.g. egg1.tif
    local_results: str        # local *_results folder; results go to <it>/segmentation
    host: str = ""
    user: str = ""
    submitted: float = field(default_factory=time.time)
    partition: str = ""
    run: str = ""             # segmentation run folder (segmentation/<run>/), the same here and there

    @property
    def remote_seg_dir(self) -> str:
        stem = movie_stem(self.movie_name)  # same rule as `tissuekit segment`
        base = posixpath.join(self.remote_dir, f"{stem}_results", "segmentation")
        return posixpath.join(base, self.run) if self.run else base

    @property
    def local_seg_dir(self) -> Path:
        base = Path(self.local_results) / "segmentation"
        return base / self.run if self.run else base

    @property
    def log_path(self) -> str:
        return posixpath.join(self.remote_dir, f"slurm-{self.job_id}.log")


# ---- remembered jobs ---------------------------------------------------------

def load_jobs(path: Path | None = None) -> list[RemoteJob]:
    path = path or JOBS_PATH
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    names = {f.name for f in fields(RemoteJob)}
    return [RemoteJob(**{k: v for k, v in r.items() if k in names}) for r in rows]


def _save_jobs(jobs: list[RemoteJob], path: Path | None = None) -> None:
    path = path or JOBS_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(j) for j in jobs], indent=2), encoding="utf-8")


def remember_job(job: RemoteJob, path: Path | None = None) -> None:
    jobs = [j for j in load_jobs(path) if j.job_id != job.job_id]
    _save_jobs([*jobs, job], path)


def forget_job(job_id: str, path: Path | None = None) -> None:
    _save_jobs([j for j in load_jobs(path) if j.job_id != job_id], path)


def fmt_duration(seconds: float) -> str:
    seconds = max(seconds, 0)
    if seconds < 90:
        return f"{int(round(seconds / 10) * 10) or 10} s"
    if seconds < 90 * 60:
        return f"{int(round(seconds / 60))} min"
    return f"{seconds / 3600:.1f} h"


# ---- SSH ---------------------------------------------------------------------

def answer_prompts(prompts, password: str, otp: str) -> list[str]:
    """Answers for one keyboard-interactive round (bwHPC asks "Your OTP:" then "Password:")."""
    answers = []
    for prompt, _echo in prompts:
        p = prompt.lower()
        if any(k in p for k in ("otp", "one-time", "one time", "token", "verification", "code")):
            answers.append(otp)
        else:
            answers.append(password)
    return answers


class Session:
    """One authenticated SSH connection (exec + SFTP) to the cluster."""

    def __init__(self, transport, sftp):
        self.transport = transport
        self.sftp = sftp

    @classmethod
    def connect(cls, cfg: RemoteConfig, password: str, otp: str, timeout: float = 30) -> "Session":
        import paramiko

        if not cfg.user:
            raise RemoteError("Set your cluster user name in the cluster settings first.")
        try:
            sock = socket.create_connection((cfg.host, int(cfg.port)), timeout=timeout)
        except OSError as exc:
            raise RemoteError(f"Cannot reach {cfg.host}: {exc}. VPN needed off campus?") from exc
        transport = paramiko.Transport(sock)
        try:
            transport.start_client(timeout=timeout)
            _check_host_key(cfg.host, int(cfg.port), transport.get_remote_server_key())
            try:
                transport.auth_interactive(
                    cfg.user, lambda _t, _i, prompts: answer_prompts(prompts, password, otp)
                )
            except paramiko.AuthenticationException as exc:
                raise RemoteError(
                    "Login refused: wrong password or OTP (an OTP works only once; wait for the next one)."
                ) from exc
            if not transport.is_authenticated():
                raise RemoteError("Login incomplete: the server asked for more than password + OTP.")
            transport.set_keepalive(30)
            sftp = paramiko.SFTPClient.from_transport(transport)
        except Exception:
            transport.close()
            raise
        return cls(transport, sftp)

    def close(self) -> None:
        try:
            self.sftp.close()
        finally:
            self.transport.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    @property
    def alive(self) -> bool:
        return self.transport.is_active()

    def run(self, command: str, timeout: float = 120, check: bool = True) -> str:
        """Run ``command`` in a login shell on the cluster and return its stdout."""
        chan = self.transport.open_session()
        chan.settimeout(timeout)
        chan.exec_command(f"bash -lc {shlex.quote(command)}")
        out = chan.makefile("rb").read().decode("utf-8", "replace")
        err = chan.makefile_stderr("rb").read().decode("utf-8", "replace")
        code = chan.recv_exit_status()
        chan.close()
        if check and code != 0:
            raise RemoteError(f"`{command.split()[0]}` failed on the cluster (exit {code}): {err.strip() or out.strip()}")
        return out

    def home_path(self, path: str) -> str:
        """Absolute cluster path; relative paths and ``~`` are inside the home folder."""
        home = posixpath.normpath(self.sftp.normalize("."))  # some servers answer "<home>/."
        if path.startswith("~"):
            path = path[1:].lstrip("/")
        return posixpath.normpath(path if path.startswith("/") else posixpath.join(home, path))


def _check_host_key(host: str, port: int, key) -> None:
    """Trust on first use: remember the cluster's key, refuse if it ever changes."""
    import paramiko

    name = host if port == 22 else f"[{host}]:{port}"
    known = paramiko.HostKeys()
    for path in (Path.home() / ".ssh" / "known_hosts", KNOWN_HOSTS):
        if path.is_file():
            try:
                known.load(str(path))
            except Exception:
                pass
    entry = known.lookup(name)
    if entry is not None and key.get_name() in entry:
        if entry[key.get_name()] != key:
            raise RemoteError(
                f"The host key of {host} has CHANGED. Refusing to log in. If the cluster "
                f"announced new keys, remove {host} from {KNOWN_HOSTS} and ~/.ssh/known_hosts."
            )
        return
    own = paramiko.HostKeys()
    if KNOWN_HOSTS.is_file():
        own.load(str(KNOWN_HOSTS))
    own.add(name, key.get_name(), key)
    KNOWN_HOSTS.parent.mkdir(parents=True, exist_ok=True)
    own.save(str(KNOWN_HOSTS))


# ---- the job -----------------------------------------------------------------

def job_script(cfg: RemoteConfig, remote_dir: str, movie_name: str, tag: str,
               install: "PluginState | None" = None, run: str = "",
               params: dict | None = None, fill_holes: bool = True) -> str:
    """Slurm script for one segmentation. ``params`` = every Cellpose value from the plugin's
    Segmentation tab (sent as JSON, so presets saved only on this computer work there too)."""
    q = shlex.quote
    stem = movie_stem(movie_name)
    seg_dir = posixpath.join(remote_dir, f"{stem}_results", "segmentation", *([run] if run else []))
    conda_sh = cfg.conda_sh.replace("~", "$HOME", 1) if cfg.conda_sh.startswith("~") else q(cfg.conda_sh)
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=tk_seg_{re.sub(r'[^A-Za-z0-9_.-]', '_', stem)[:40]}",
        f"#SBATCH --partition={cfg.partition}",
        *([f"#SBATCH --gres={cfg.gres}"] if cfg.gres else []),
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={int(cfg.cpus)}",
        f"#SBATCH --mem={cfg.mem}",
        f"#SBATCH --time={cfg.time}",
        f"#SBATCH --output={remote_dir}/slurm-%j.log",
        "set -eo pipefail  # no -u: conda activate reads unset variables",
        f"source {conda_sh}",
        f"conda activate {q(cfg.conda_env)}",
        f"cd {q(remote_dir)}",
        "nvidia-smi -L || true",
        # shows in the Log window whether Cellpose will really use the GPU (driver vs. torch CUDA)
        "python -c \"import torch; print('torch', torch.__version__, '| GPU usable:', "
        "torch.cuda.is_available())\" || true",
        *(_install_lines(install) if install is not None and install.needs_install else []),
        f"tissuekit segment {q(movie_name)} --preset {q(cfg.preset)} --batch-size {int(cfg.batch_size)}"
        + (f" --run {q(run)}" if run else "")
        + (f" --params {q(json.dumps(params, separators=(',', ':')))}" if params else "")
        + ("" if fill_holes else " --no-fill-holes"),
    ]
    lines.append("echo TISSUEKIT_DONE")
    return "\n".join(lines) + "\n"


def conda_root(cfg: RemoteConfig) -> str:
    """Conda install folder from the conda.sh path (``$HOME`` for ``~``), for use in bash."""
    root = cfg.conda_sh
    if root.endswith("/etc/profile.d/conda.sh"):
        root = root[: -len("/etc/profile.d/conda.sh")]
    return "$HOME" + root[1:] if root.startswith("~") else root


def env_python(cfg: RemoteConfig) -> str:
    return f"{conda_root(cfg)}/envs/{cfg.conda_env}/bin/python"


# ---- keeping the cluster's plugin code in step with this computer -----------------

def _plugin_files(plugin_dir: Path) -> list[tuple[str, Path]]:
    """(archive name, file) for pyproject.toml and the package, sorted, no caches."""
    pkg = plugin_dir / "napari_tissuekit"
    files = [("pyproject.toml", plugin_dir / "pyproject.toml")]
    for f in sorted(pkg.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
            files.append((f.relative_to(plugin_dir).as_posix(), f))
    return files


def plugin_fingerprint(plugin_dir: Path | None = None) -> tuple[str, str]:
    """(hash over all plugin files, hash of pyproject.toml); a change in the latter needs pip."""
    plugin_dir = plugin_dir or PLUGIN_DIR
    code = hashlib.sha256()
    for name, f in _plugin_files(plugin_dir):
        code.update(name.encode() + b"\0" + f.read_bytes() + b"\0")
    deps = hashlib.sha256((plugin_dir / "pyproject.toml").read_bytes()).hexdigest()
    return code.hexdigest()[:16], deps[:16]


def _plugin_tarball(plugin_dir: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, f in _plugin_files(plugin_dir):
            tar.add(str(f), arcname=name)
    return buf.getvalue()


@dataclass
class PluginState:
    remote_dir: str        # cluster folder with the uploaded plugin (the pip install -e target)
    code_hash: str
    deps_hash: str
    needs_install: bool    # pyproject.toml changed since the last pip install


def _install_lines(state: PluginState) -> list[str]:
    q = shlex.quote
    marker = posixpath.join(state.remote_dir, ".deps_hash")
    return [
        "echo 'pyproject.toml changed: updating the env (pip install)'",
        f"python -m pip install --no-cache-dir -q -e {q(state.remote_dir)}",
        f"echo {q(state.deps_hash)} > {q(marker)}",
    ]


def sync_plugin(cfg: RemoteConfig, session: Session, progress: Progress,
                plugin_dir: Path | None = None, need_env: bool = True) -> PluginState:
    """Send this computer's plugin code to the cluster if it changed (usually: nothing to do)."""
    plugin_dir = plugin_dir or PLUGIN_DIR
    if not (plugin_dir / "pyproject.toml").is_file():
        raise RemoteError(
            f"Plugin source not found at {plugin_dir}; install the plugin from a checkout (pip install -e)."
        )
    q = shlex.quote
    remote_dir = posixpath.join(session.home_path(cfg.remote_root), "plugin")
    code, deps = plugin_fingerprint(plugin_dir)
    if need_env and "ENV_OK" not in session.run(f"test -x {env_python(cfg)} && echo ENV_OK", check=False):
        raise RemoteError("The cluster env is not set up yet: press 'Set up cluster...' once.")
    have = session.run(
        f"cat {q(remote_dir)}/.code_hash 2>/dev/null; echo '|'; cat {q(remote_dir)}/.deps_hash 2>/dev/null",
        check=False,
    ).split("|")
    have_code = have[0].strip()
    have_deps = have[1].strip() if len(have) > 1 else ""
    if have_code != code:
        progress(0, "sending the updated plugin code to the cluster ...")
        session.run(f"mkdir -p {q(remote_dir)}")
        session.sftp.putfo(io.BytesIO(_plugin_tarball(plugin_dir)), posixpath.join(remote_dir, ".upload.tgz"))
        session.run(
            f"cd {q(remote_dir)} && rm -rf napari_tissuekit && tar xzf .upload.tgz && rm .upload.tgz"
            f" && echo {q(code)} > .code_hash"
        )
    return PluginState(remote_dir, code, deps, needs_install=have_deps != deps)


def setup_script(cfg: RemoteConfig, state: PluginState, log_dir: str) -> str:
    q = shlex.quote
    env = q(cfg.conda_env)
    return "\n".join([
        "#!/bin/bash",
        "#SBATCH --job-name=tk_setup",
        f"#SBATCH --partition={cfg.setup_partition}",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=16G",
        f"#SBATCH --time={cfg.setup_time}",
        f"#SBATCH --output={log_dir}/slurm-%j.log",
        "set -eo pipefail  # no -u: conda activate reads unset variables",
        f'ROOT="{conda_root(cfg)}"',
        'if [ ! -x "$ROOT/bin/conda" ]; then',
        '  echo "step 1/4: installing Miniforge into $ROOT"',
        f'  curl -fsSL -o "${{TMPDIR:-/tmp}}/miniforge.sh" {MINIFORGE_URL}',
        '  bash "${TMPDIR:-/tmp}/miniforge.sh" -b -p "$ROOT"',
        "else",
        '  echo "step 1/4: Miniforge already in $ROOT"',
        "fi",
        'source "$ROOT/etc/profile.d/conda.sh"',
        f"if ! conda env list | awk '{{print $1}}' | grep -qx {env}; then",
        f"  echo 'step 2/4: creating env {cfg.conda_env}'",
        f"  conda create -y -q -n {env} python=3.11",
        "fi",
        f"conda activate {env}",
        "python -m pip install --no-cache-dir -q --upgrade pip",
        "echo 'step 3/4: installing torch (CUDA build)'",
        "python -m pip install --no-cache-dir -q torch",
        "echo 'step 4/4: installing the plugin and Cellpose'",
        f"python -m pip install --no-cache-dir -q -e {q(state.remote_dir)}",
        "python -c \"import torch, cellpose, napari_tissuekit as t; "
        "print('torch', torch.__version__, 'cuda', torch.version.cuda, 'tissuekit', t.__version__)\"",
        "which tissuekit",
        f"echo {q(state.deps_hash)} > {q(posixpath.join(state.remote_dir, '.deps_hash'))}",
        "echo TISSUEKIT_SETUP_DONE",
    ]) + "\n"


_STEP = re.compile(r"step (\d)/4: (.*)")


def _setup_progress(log: str) -> tuple[int, str] | None:
    hits = _STEP.findall(log)
    if not hits:
        return None
    n, text = hits[-1]
    return 25 * (int(n) - 1), f"cluster setup: {text}"


def setup_cluster(cfg: RemoteConfig, password: str, otp: str, progress: Progress,
                  sleep: Callable[[float], None] = time.sleep, plugin_dir: Path | None = None,
                  on_log: Callable[[str], None] | None = None) -> str:
    """One time: Miniforge + conda env (torch, Cellpose, this plugin) via a CPU Slurm job."""
    q = shlex.quote
    progress(0, f"connecting to {cfg.host} ...")
    with Session.connect(cfg, password, otp) as session:
        state = sync_plugin(cfg, session, progress, plugin_dir, need_env=False)
        log_dir = posixpath.join(session.home_path(cfg.remote_root), "setup")
        session.run(f"mkdir -p {q(log_dir)}")
        script = posixpath.join(log_dir, "setup.sh")
        with session.sftp.open(script, "w") as fh:
            fh.write(setup_script(cfg, state, log_dir))
        job_id = session.run(f"sbatch --parsable {q(script)}").strip().split(";")[0]
        if not job_id.isdigit():
            raise RemoteError(f"sbatch did not return a job id: {job_id!r}")
        job = RemoteJob(job_id, "setup", log_dir, "setup", "", host=cfg.host, user=cfg.user)
        progress(0, f"setup job {job_id} submitted on {cfg.setup_partition} (about 10-20 min)")
        wait_for(session, job, progress, cfg.poll_seconds, sleep, running=_setup_progress, on_log=on_log)
        tail = _tail(session, job.log_path, 400)
    if "TISSUEKIT_SETUP_DONE" not in tail:
        raise RemoteError(f"Setup job {job_id} ended without finishing:\n{tail[-800:]}")
    return next((ln for ln in tail.splitlines() if ln.startswith("torch ")), "cluster env ready")


def job_state(session: Session, job_id: str) -> str:
    """Slurm state such as PENDING, RUNNING, COMPLETED (empty if Slurm forgot the job)."""
    out = session.run(f"squeue -h -j {int(job_id)} -o %T", check=False).strip()
    if out:
        return out.splitlines()[0].strip()
    out = session.run(f"sacct -j {int(job_id)} -X -n -P -o State", check=False).strip()
    return out.splitlines()[0].split()[0].strip() if out else ""


_FRAME = re.compile(r"frame (\d+)/(\d+)")


def log_progress(log_text: str) -> tuple[int, str] | None:
    """(percent, message) from the last ``frame a/b`` line of the job log."""
    hits = _FRAME.findall(log_text)
    if not hits:
        return None
    a, b = (int(x) for x in hits[-1])
    return int(100 * a / max(b, 1)), f"frame {a}/{b}"


_PCT_LINE = re.compile(r"^\s+(.+) \((\d+)%\)$")  # "  thinning bonds, frame 3/11 (27%)"


def _seg_progress(log_text: str) -> tuple[int, str] | None:
    """(percent, what the job is doing now) from the newest telling line of the job log.

    Percent 0 means "no measurable progress in this phase" (the GUI shows a busy bar).
    """
    for line in reversed(log_text.splitlines()):
        text = line.strip()
        if not text:
            continue
        if text == "TISSUEKIT_DONE" or text.startswith("wrote") and "_cell_outlines" in text:
            return 100, "finished on the cluster; downloading ..."
        m = _PCT_LINE.match(line)
        if m:
            return max(int(m.group(2)), 1), f"step 2/2 outlines: {m.group(1)}"
        if text.startswith("wrote") and "_cp_masks" in text:
            return 1, "step 2/2 outlines: masks saved, building 1 px outlines"
        m = _FRAME.match(text)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return max(int(100 * a / max(b, 1)), 1), f"step 1/2 Cellpose on the GPU: frame {a}/{b}"
        if text.startswith("loading model"):
            return 0, "loading the Cellpose model on the GPU"
        if text.startswith(("loading ", "shape ", "calibration:", "starting fresh", "resuming")):
            return 0, "reading the movie"
        if "updating the env" in text:
            return 0, "updating the cluster env (pyproject.toml changed)"
    return None


def _tail(session: Session, path: str, n: int = 40) -> str:
    return session.run(f"tail -n {int(n)} {shlex.quote(path)} 2>/dev/null || true", check=False)


def upload_movie(session: Session, movie: Path, remote_dir: str, progress: Progress) -> str:
    name = movie.name
    size = max(movie.stat().st_size, 1)

    def cb(done, _total):
        progress(int(100 * done / size), f"uploading {name} ({done / 1e6:.0f}/{size / 1e6:.0f} MB)")

    session.sftp.put(str(movie), posixpath.join(remote_dir, name), callback=cb)
    return name


def local_run_name(local_results: Path, preset: str) -> str:
    """The next unused run name in the local results folder (``default``, ``default_2``, ...)."""
    from .project import WorkingFolder

    wf = WorkingFolder()
    wf.set_manual(local_results)
    return wf.new_run_name(preset)


def _mark_active_run(job: "RemoteJob") -> None:
    """Make the downloaded run the one the plugin shows (as a local run would be)."""
    if not job.run:
        return
    from .project import WorkingFolder

    wf = WorkingFolder()
    wf.set_manual(job.local_results)
    try:
        wf.set_active_seg_run(job.run)
    except Exception:
        pass  # the files are there either way; the plugin then shows the newest run


def submit(cfg: RemoteConfig, session: Session, movie: Path, local_results: Path,
           progress: Progress, install: PluginState | None = None,
           run: str | None = None, params: dict | None = None, fill_holes: bool = True) -> RemoteJob:
    """Upload and sbatch. The job is remembered on this computer right away, so a lost
    connection (or a closed napari) never loses it: 'Fetch cluster results' picks it up."""
    run = run or local_run_name(Path(local_results), cfg.preset)  # free here, so nothing local is overwritten
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", movie.stem)
    tag = f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    remote_dir = posixpath.join(session.home_path(cfg.remote_root), tag)
    session.run(f"mkdir -p {shlex.quote(remote_dir)}")
    movie_name = upload_movie(session, movie, remote_dir, progress)
    script = posixpath.join(remote_dir, "job.sh")
    with session.sftp.open(script, "w") as fh:
        fh.write(job_script(cfg, remote_dir, movie_name, tag, install, run, params, fill_holes))
    out = session.run(f"sbatch --parsable {shlex.quote(script)}").strip()
    job_id = out.split(";")[0].strip()
    if not job_id.isdigit():
        raise RemoteError(f"sbatch did not return a job id: {out!r}")
    job = RemoteJob(
        job_id=job_id, tag=tag, remote_dir=remote_dir, movie_name=movie_name,
        local_results=str(local_results), host=cfg.host, user=cfg.user,
        partition=cfg.partition, run=run,
    )
    remember_job(job)
    progress(0, f"submitted job {job_id} on {cfg.partition} (time limit {cfg.time}) as run '{run}'")
    return job


def wait_for(session: Session, job: RemoteJob, progress: Progress, poll_seconds: float = 20,
             sleep: Callable[[float], None] = time.sleep,
             running: Callable[[str], tuple[int, str] | None] | None = None,
             on_log: Callable[[str], None] | None = None) -> str:
    """Poll until the job ends. Returns the final state; raises Disconnected if SSH drops.

    ``on_log`` gets the newest job log lines at every check (for a live log window).
    """
    last = ""
    phase = ("", 0.0, 0)  # (phase name, time it was first seen, percent then) for "time left"
    while True:
        try:
            state = job_state(session, job.job_id)
            log = _tail(session, job.log_path, 400) if state != "PENDING" else ""
            if on_log is not None and log:
                on_log(log)
        except Exception as exc:
            if not session.alive or isinstance(exc, (OSError, EOFError)):
                raise Disconnected(
                    f"Connection to the cluster lost; job {job.job_id} keeps running. "
                    "Use 'Fetch cluster results' later."
                ) from exc
            raise
        if state in DONE_STATES:
            if on_log is not None:
                on_log(_tail(session, job.log_path, 400))
            return state
        if state in FAILED_STATES or (state == "" and last == ""):
            tail = "\n".join(log.strip().splitlines()[-5:])
            hint = (" The job hit its time limit: raise 'Time limit' in Settings... (dev queue max "
                    "00:30:00, or use gpu_h100) and run again." if state == "TIMEOUT" else "")
            raise RemoteError(f"Cluster job {job.job_id} ended with {state or 'unknown state'}.{hint}\n{tail}")
        if state == "":  # finished and gone from squeue/sacct between polls: treat as done
            return "COMPLETED"
        last = state
        hit = (running or _seg_progress)(log)
        if state == "PENDING":
            progress(0, f"job {job.job_id} waiting in the queue ({job_state_hint(job)})")
        elif hit:
            pct, msg = hit
            name, now = msg.split(":")[0], time.time()
            if name != phase[0]:
                phase = (name, now, pct)
            elif 0 < pct < 100 and pct > phase[2] and now - phase[1] > 1:
                left = (100 - pct) * (now - phase[1]) / (pct - phase[2])
                msg = f"{msg} - about {fmt_duration(left)} left"
            progress(pct, msg)
        else:
            progress(0, f"job {job.job_id} {state.lower()}" + ("" if running else " (loading model...)"))
        sleep(poll_seconds)


def job_state_hint(job: RemoteJob) -> str:
    minutes = int((time.time() - job.submitted) / 60)
    return f"{minutes} min" if minutes else "just submitted"


def download_seg(session: Session, job: RemoteJob, progress: Progress) -> list[Path]:
    """Copy the job's run folder into ``<local_results>/segmentation/<run>``."""
    import stat

    dest = job.local_seg_dir
    dest.mkdir(parents=True, exist_ok=True)
    try:
        entries = session.sftp.listdir_attr(job.remote_seg_dir)
    except OSError:
        return []  # the caller reports what the job folder and log show instead
    files = [e for e in entries if not stat.S_ISDIR(e.st_mode or 0)]
    got = []
    for i, entry in enumerate(files):
        progress(int(100 * i / max(len(files), 1)), f"downloading {entry.filename}")
        local = dest / entry.filename
        session.sftp.get(posixpath.join(job.remote_seg_dir, entry.filename), str(local))
        got.append(local)
    return got


def masks_file(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.name.endswith("_cp_masks.tif")), None)


def run_segmentation(cfg: RemoteConfig, password: str, otp: str, movie: Path,
                     local_results: Path, progress: Progress,
                     sleep: Callable[[float], None] = time.sleep,
                     on_log: Callable[[str], None] | None = None,
                     run: str | None = None,
                     params: dict | None = None, fill_holes: bool = True) -> tuple[RemoteJob, Path]:
    """The one-button round trip. Returns the job and the downloaded ``*_cp_masks.tif``.

    ``run``/``params``: run folder name and Cellpose values chosen in the Segmentation tab
    (default: the next free name for ``cfg.preset`` and that preset's values)."""
    progress(0, f"connecting to {cfg.host} ...")
    with Session.connect(cfg, password, otp) as session:
        state = sync_plugin(cfg, session, progress)
        progress(0, "logged in; uploading movie ...")
        job = submit(cfg, session, Path(movie), Path(local_results), progress, state, run, params, fill_holes)
        wait_for(session, job, progress, cfg.poll_seconds, sleep, on_log=on_log)
        files = download_seg(session, job, progress)
        if masks_file(files) is None:  # keep the job remembered: it can be fetched again
            raise RemoteError(_no_masks_report(session, job, files))
        _finish(session, job, cfg.delete_remote_after)
    return job, masks_file(files)


def _no_masks_report(session: Session, job: RemoteJob, files: list[Path]) -> str:
    """What is in the job folder and the end of the job log, for the user and the Log window."""
    listing = session.run(f"cd {shlex.quote(job.remote_dir)} && find . -maxdepth 3 | head -40", check=False)
    tail = "\n".join(_tail(session, job.log_path, 400).strip().splitlines()[-15:])
    got = ", ".join(p.name for p in files) or "nothing"
    return (f"Job {job.job_id} finished but no *_cp_masks.tif was found in {job.remote_seg_dir} "
            f"(downloaded: {got}).\nJob folder:\n{listing.strip()}\nEnd of job log:\n{tail}")


def fetch_job(cfg: RemoteConfig, job: RemoteJob, progress: Progress,
              password: str, otp: str,
              sleep: Callable[[float], None] = time.sleep,
              on_log: Callable[[str], None] | None = None) -> Path:
    """Pick a remembered job up again (after a lost connection, a closed napari, a restart).

    One login: if the job still runs, keep watching it (progress, log), then download.
    """
    progress(0, f"connecting to {cfg.host} ...")
    with Session.connect(cfg, password, otp) as session:
        state = job_state(session, job.job_id)
        if state in FAILED_STATES:
            forget_job(job.job_id)
            tail = "\n".join(_tail(session, job.log_path).strip().splitlines()[-5:])
            raise RemoteError(f"Cluster job {job.job_id} ended with {state}.\n{tail}")
        if state and state not in DONE_STATES:
            progress(0, f"reconnected: job {job.job_id} is {state.lower()}; watching it again")
            wait_for(session, job, progress, cfg.poll_seconds, sleep, on_log=on_log)
        files = download_seg(session, job, progress)
        if masks_file(files) is None:  # keep the job remembered: it can be fetched again
            raise RemoteError(_no_masks_report(session, job, files))
        _finish(session, job, cfg.delete_remote_after)
    return masks_file(files)


def _finish(session: Session, job: RemoteJob, delete_remote: bool) -> None:
    _mark_active_run(job)
    forget_job(job.job_id)
    if delete_remote and job.remote_dir.count("/") >= 3:  # never rm a short path by accident
        session.run(f"rm -rf {shlex.quote(job.remote_dir)}", check=False)
