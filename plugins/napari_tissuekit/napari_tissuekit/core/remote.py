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
    # Batch job: one dict per (egg, profile) item (movie_name, local_results, profile, run,
    # status pending|done|failed|downloaded). Empty for a single-movie job.
    items: list = field(default_factory=list)

    @property
    def is_batch(self) -> bool:
        return bool(self.items)

    @property
    def label(self) -> str:
        """Short name for lists and messages."""
        if not self.items:
            return self.movie_name
        eggs = len({it["movie_name"] for it in self.items})
        return f"batch ({len(self.items)} items, {eggs} egg{'s' * (eggs != 1)})"

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
        _segment_cmd(movie_name, cfg.preset, int(cfg.batch_size), run, params, fill_holes),
    ]
    lines.append("echo TISSUEKIT_DONE")
    return "\n".join(lines) + "\n"


def _segment_cmd(movie_name: str, preset: str, batch_size: int, run: str = "",
                 params: dict | None = None, fill_holes: bool = True) -> str:
    """The ``tissuekit segment`` line of a job script."""
    q = shlex.quote
    params = dict(params or {})
    min_area = params.pop("min_area_um2", None)  # not a Cellpose value: its own command line option
    return (
        f"tissuekit segment {q(movie_name)} --preset {q(preset)} --batch-size {int(batch_size)}"
        + (f" --min-area-um2 {float(min_area):g}" if min_area else "")
        + (f" --run {q(run)}" if run else "")
        + (f" --params {q(json.dumps(params, separators=(',', ':')))}" if params else "")
        + ("" if fill_holes else " --no-fill-holes")
    )


# ---- batch: many (egg, profile) items one after another in one GPU job ----------------

def parse_time(text: str) -> int:
    """Seconds of a Slurm time limit: ``mm``, ``mm:ss``, ``hh:mm:ss`` or ``d-hh[:mm[:ss]]``."""
    text = text.strip()
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
        parts = [int(x) for x in text.split(":")] + [0, 0]
        h, m, s = parts[:3]
    else:
        parts = [int(x) for x in text.split(":")]
        if len(parts) == 1:
            h, m, s = 0, parts[0], 0
        elif len(parts) == 2:
            h, (m, s) = 0, parts
        else:
            h, m, s = parts[-3:]
    return ((days * 24 + h) * 60 + m) * 60 + s


def format_time(seconds: int) -> str:
    """Slurm time limit ``hh:mm:ss`` (hours may exceed 24)."""
    seconds = max(int(seconds), 60)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def batch_time(per_item: str, n_items: int) -> str:
    """Time limit of a batch job: the per-item limit times the number of items."""
    return format_time(parse_time(per_item) * max(int(n_items), 1))


# Longest job a queue accepts (bwUniCluster 3.0 batch queue table): every dev_* queue 30 min,
# gpu_h100 72 h. Other queues are not checked here (Slurm refuses a too long job itself).
DEV_MAX = "00:30:00"
PARTITION_MAX = {"gpu_h100": "72:00:00"}


def partition_problem(partition: str, time_limit: str) -> str:
    """Why ``time_limit`` does not fit ``partition`` (empty if it fits or the queue is unknown)."""
    limit = DEV_MAX if partition.startswith("dev_") else PARTITION_MAX.get(partition)
    if limit is None or parse_time(time_limit) <= parse_time(limit):
        return ""
    hint = " Use gpu_h100 (up to 72 h) in Cluster settings, or fewer items." if partition.startswith("dev_") else ""
    return f"{time_limit} is longer than {partition} allows ({limit}).{hint}"


def batch_job_script(cfg: RemoteConfig, remote_dir: str, items, install: "PluginState | None" = None) -> str:
    """Slurm script that segments every item in turn (``items``: :class:`core.batch.BatchItem`).

    Each item prints ``TISSUEKIT_ITEM i/N``, then ``TISSUEKIT_ITEM_DONE i`` or
    ``TISSUEKIT_ITEM_FAILED i``. A failed item does not stop the others: a command in a
    ``&&`` / ``||`` list does not end the script under ``set -e``.
    """
    q = shlex.quote
    n = len(items)
    conda_sh = cfg.conda_sh.replace("~", "$HOME", 1) if cfg.conda_sh.startswith("~") else q(cfg.conda_sh)
    eggs = len({it.movie_name for it in items})
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=tk_batch_{eggs}x{n}",
        f"#SBATCH --partition={cfg.partition}",
        *([f"#SBATCH --gres={cfg.gres}"] if cfg.gres else []),
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={int(cfg.cpus)}",
        f"#SBATCH --mem={cfg.mem}",
        f"#SBATCH --time={batch_time(cfg.time, n)}",
        f"#SBATCH --output={remote_dir}/slurm-%j.log",
        "set -eo pipefail  # no -u: conda activate reads unset variables",
        f"source {conda_sh}",
        f"conda activate {q(cfg.conda_env)}",
        f"cd {q(remote_dir)}",
        "nvidia-smi -L || true",
        "python -c \"import torch; print('torch', torch.__version__, '| GPU usable:', "
        "torch.cuda.is_available())\" || true",
        *(_install_lines(install) if install is not None and install.needs_install else []),
    ]
    for i, it in enumerate(items, 1):
        cmd = _segment_cmd(it.movie_name, run_name_label(it.profile), int(it.params.get("batch_size", cfg.batch_size)),
                           it.run, it.params, it.fill_holes)
        lines += [
            f"echo {q(f'TISSUEKIT_ITEM {i}/{n} {it.movie_name} {it.run}')}",
            f"{cmd} && echo 'TISSUEKIT_ITEM_DONE {i}' || echo 'TISSUEKIT_ITEM_FAILED {i}'",
        ]
    lines.append("echo TISSUEKIT_DONE")
    return "\n".join(lines) + "\n"


def run_name_label(profile: str) -> str:
    """Preset name as a single shell word (it is only a label when --params is given)."""
    return re.sub(r"\s+", "_", profile.strip()) or "custom"


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
             on_log: Callable[[str], None] | None = None,
             on_poll: Callable[[], None] | None = None) -> str:
    """Poll until the job ends. Returns the final state; raises Disconnected if SSH drops.

    ``on_log`` gets the newest job log lines at every check (for a live log window).
    ``on_poll`` runs at every check once the job runs (a batch downloads finished items there).
    """
    last = ""
    phase = ("", 0.0, 0)  # (phase name, time it was first seen, percent then) for "time left"
    while True:
        try:
            state = job_state(session, job.job_id)
            log = _tail(session, job.log_path, 400) if state != "PENDING" else ""
            if on_log is not None and log:
                on_log(log)
            if on_poll is not None and state != "PENDING":
                on_poll()
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
    return _download_dir(session, job.remote_seg_dir, job.local_seg_dir, progress)


def _download_dir(session: Session, remote_dir: str, dest: Path, progress: Progress,
                  label: str = "") -> list[Path]:
    """Copy the files (not subfolders) of ``remote_dir`` into ``dest``."""
    import stat

    try:
        entries = session.sftp.listdir_attr(remote_dir)
    except OSError:
        return []  # the caller reports what the job folder and log show instead
    dest.mkdir(parents=True, exist_ok=True)
    files = [e for e in entries if not stat.S_ISDIR(e.st_mode or 0)]
    got = []
    for i, entry in enumerate(files):
        progress(int(100 * i / max(len(files), 1)), f"{label}downloading {entry.filename}")
        local = dest / entry.filename
        session.sftp.get(posixpath.join(remote_dir, entry.filename), str(local))
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
    if job.is_batch:
        return _fetch_batch(cfg, job, progress, password, otp, sleep, on_log)
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


def my_jobs(session: Session, user: str) -> list[tuple[str, str, str]]:
    """(job id, job name, state) of every job of ``user`` that Slurm still has queued or running."""
    out = session.run(f"squeue -h -u {shlex.quote(user)} -o '%i|%j|%T'", check=False)
    jobs = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 3 and parts[0]:
            jobs.append((parts[0], parts[1], parts[2]))
    return jobs


def cancel_all_jobs(cfg: RemoteConfig, password: str, otp: str,
                    progress: Progress) -> list[tuple[str, str, str]]:
    """Cancel every waiting and running job of the cluster user (``scancel -u``).

    Returns the jobs that were cancelled. Remembered single-movie jobs among them are
    forgotten (they have no results); batch jobs stay remembered, so 'Fetch cluster
    results' can still download the items that finished before the cancel.
    """
    progress(0, f"connecting to {cfg.host} ...")
    with Session.connect(cfg, password, otp) as session:
        jobs = my_jobs(session, cfg.user)
        if not jobs:
            progress(0, "no waiting or running jobs on the cluster")
            return []
        progress(0, f"cancelling {len(jobs)} job(s) ...")
        session.run(f"scancel -u {shlex.quote(cfg.user)}")
        left = my_jobs(session, cfg.user)
    gone = {j[0] for j in jobs}
    for job in load_jobs():
        if job.job_id in gone and not job.is_batch:
            forget_job(job.job_id)
    if left:
        # scancel only asks Slurm to stop them; a running job may take a moment to end
        progress(0, f"{len(left)} job(s) still ending on the cluster")
    return jobs


def _finish(session: Session, job: RemoteJob, delete_remote: bool) -> None:
    _mark_active_run(job)
    forget_job(job.job_id)
    if delete_remote and job.remote_dir.count("/") >= 3:  # never rm a short path by accident
        session.run(f"rm -rf {shlex.quote(job.remote_dir)}", check=False)


# ---- batch: submit, watch, download item by item --------------------------------------
#
# Progress messages the Batch tab reads (item numbers are 1-based):
#   "item i/N <egg> · <profile>: <what it does>"   item i is running
#   "downloaded item i/N <egg> · <profile>"        its run folder is on this computer
#   "item i/N <egg> · <profile>: FAILED ..."        it ended without results

_ITEM_MARK = re.compile(r"^TISSUEKIT_ITEM(?:_(DONE|FAILED))? (\d+)", re.M)


def parse_markers(text: str) -> tuple[int, set[int], set[int]]:
    """(last started item, finished items, failed items) from the job log's marker lines."""
    started, done, failed = 0, set(), set()
    for kind, i in _ITEM_MARK.findall(text):
        i = int(i)
        if kind == "DONE":
            done.add(i)
        elif kind == "FAILED":
            failed.add(i)
        else:
            started = max(started, i)
    return started, done, failed


def item_text(it: dict) -> str:
    return f"{movie_stem(it['movie_name'])} · {it['profile']}"


@dataclass
class BatchResult:
    job: RemoteJob
    downloaded: list = field(default_factory=list)  # item dicts now on this computer
    failed: list = field(default_factory=list)      # item dicts without results (with "note")
    note: str = ""                                  # why the job ended early, if it did


def submit_batch(cfg: RemoteConfig, session: Session, items, progress: Progress,
                 install: PluginState | None = None) -> RemoteJob:
    """Upload every movie once, sbatch one job for all items, remember it right away."""
    q = shlex.quote
    tag = f"batch_{time.strftime('%Y%m%d-%H%M%S')}"
    remote_dir = posixpath.join(session.home_path(cfg.remote_root), tag)
    session.run(f"mkdir -p {q(remote_dir)}")
    movies = list(dict.fromkeys(it.movie for it in items))
    for k, movie in enumerate(movies, 1):
        upload_movie(session, Path(movie), remote_dir,
                     lambda p, m, k=k: progress(p, f"movie {k}/{len(movies)}: {m}"))
    script = posixpath.join(remote_dir, "job.sh")
    with session.sftp.open(script, "w") as fh:
        fh.write(batch_job_script(cfg, remote_dir, items, install))
    out = session.run(f"sbatch --parsable {q(script)}").strip()
    job_id = out.split(";")[0].strip()
    if not job_id.isdigit():
        raise RemoteError(f"sbatch did not return a job id: {out!r}")
    job = RemoteJob(
        job_id=job_id, tag=tag, remote_dir=remote_dir, movie_name=f"batch of {len(items)}",
        local_results=str(Path(items[0].local_results).parent), host=cfg.host, user=cfg.user,
        partition=cfg.partition,
        items=[{"movie_name": it.movie_name, "local_results": it.local_results, "profile": it.profile,
                "run": it.run, "status": "pending"} for it in items],
    )
    remember_job(job)
    progress(0, f"submitted batch job {job_id} on {cfg.partition} (time limit "
                f"{batch_time(cfg.time, len(items))}): {len(items)} items, {len(movies)} movies")
    return job


class _BatchWatcher:
    """At every check: read the item markers, download finished items, note failed ones."""

    def __init__(self, session: Session, job: RemoteJob, progress: Progress):
        self.session = session
        self.job = job
        self.progress = progress
        self.markers = ""

    def poll(self) -> None:
        self.markers = self.session.run(
            f"grep -a '^TISSUEKIT_ITEM' {shlex.quote(self.job.log_path)} 2>/dev/null || true", check=False)
        _started, done, failed = parse_markers(self.markers)
        n = len(self.job.items)
        changed = False
        for i, it in enumerate(self.job.items, 1):
            if it["status"] in ("downloaded", "failed"):
                continue
            if i in done:
                if self._download(it):
                    it["status"] = "downloaded"
                    self.progress(0, f"downloaded item {i}/{n} {item_text(it)}")
                else:
                    it.update(status="failed", note="finished, but no *_cp_masks.tif in its run folder")
                    self.progress(0, f"item {i}/{n} {item_text(it)}: FAILED ({it['note']})")
                changed = True
            elif i in failed:
                it.update(status="failed", note="tissuekit segment failed (see the job log)")
                self.progress(0, f"item {i}/{n} {item_text(it)}: FAILED (see Log...)")
                changed = True
        if changed:
            remember_job(self.job)  # a lost connection now only misses the rest

    def _download(self, it: dict) -> bool:
        stem = movie_stem(it["movie_name"])
        remote_seg = posixpath.join(self.job.remote_dir, f"{stem}_results", "segmentation", it["run"])
        dest = Path(it["local_results"]) / "segmentation" / it["run"]
        files = _download_dir(self.session, remote_seg, dest, lambda *_: None)
        if masks_file(files) is None:
            return False
        _mark_active_run(RemoteJob("", "", "", it["movie_name"], it["local_results"], run=it["run"]))
        return True

    def running(self, log: str) -> tuple[int, str] | None:
        """Overall percent over all items, and what the current item does."""
        n = len(self.job.items)
        started, done, failed = parse_markers(self.markers)
        if not started:
            return _seg_progress(log)  # before the first item: env update, ...
        finished = len(done | failed)
        it = self.job.items[min(started, n) - 1]
        frac, what = 0, "finished"
        if started not in done | failed:
            mark = f"TISSUEKIT_ITEM {started}/"
            hit = _seg_progress(log.rsplit(mark, 1)[-1] if mark in log else log)
            frac, what = hit if hit else (0, "starting")
            if frac >= 100:
                frac, what = 99, "writing results"
        pct = int(100 * (finished + frac / 100) / n)
        return max(pct, 1), f"item {started}/{n} {item_text(it)}: {what}"


def _watch_batch(cfg: RemoteConfig, session: Session, job: RemoteJob, progress: Progress,
                 sleep: Callable[[float], None], on_log: Callable[[str], None] | None) -> BatchResult:
    """Watch until the job ends, downloading each item as soon as it is finished."""
    watcher = _BatchWatcher(session, job, progress)
    note = ""
    try:
        wait_for(session, job, progress, cfg.poll_seconds, sleep, running=watcher.running,
                 on_log=on_log, on_poll=watcher.poll)
    except Disconnected:
        raise  # the job stays remembered with the items downloaded so far
    except RemoteError as exc:  # time limit, out of memory ...: keep what is finished
        note = str(exc)
    watcher.poll()  # items that finished just before the end
    for it in job.items:
        if it["status"] == "pending":
            it.update(status="failed", note="did not run to the end (the job ended first)")
    _finish(session, job, cfg.delete_remote_after)
    return BatchResult(
        job=job,
        downloaded=[it for it in job.items if it["status"] == "downloaded"],
        failed=[it for it in job.items if it["status"] == "failed"],
        note=note,
    )


def run_batch(cfg: RemoteConfig, password: str, otp: str, items, progress: Progress,
              sleep: Callable[[float], None] = time.sleep,
              on_log: Callable[[str], None] | None = None) -> BatchResult:
    """The Batch tab's round trip: one login, one GPU job for every (egg, profile) item."""
    if not items:
        raise RemoteError("Nothing to run.")
    problem = partition_problem(cfg.partition, batch_time(cfg.time, len(items)))
    if problem:
        raise RemoteError(problem)
    progress(0, f"connecting to {cfg.host} ...")
    with Session.connect(cfg, password, otp) as session:
        state = sync_plugin(cfg, session, progress)
        progress(0, "logged in; uploading the movies ...")
        job = submit_batch(cfg, session, items, progress, state)
        return _watch_batch(cfg, session, job, progress, sleep, on_log)


def _fetch_batch(cfg: RemoteConfig, job: RemoteJob, progress: Progress, password: str, otp: str,
                 sleep: Callable[[float], None], on_log: Callable[[str], None] | None) -> BatchResult:
    """A remembered batch: keep watching it if it still runs, download what is not here yet."""
    progress(0, f"connecting to {cfg.host} ...")
    with Session.connect(cfg, password, otp) as session:
        state = job_state(session, job.job_id)
        if state and state not in DONE_STATES and state not in FAILED_STATES:
            progress(0, f"reconnected: {job.label} is {state.lower()}; watching it again")
        return _watch_batch(cfg, session, job, progress, sleep, on_log)
