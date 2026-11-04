#!/usr/bin/env bash
# Allocate a compute node and start tmux there (not on the login node).
#
# Usage (from project root or anywhere):
#   bash hpc/hpc_alloc_tmux.sh
#   bash hpc/hpc_alloc_tmux.sh -p cpu -n 8 -t 5:00:00 --mem=8GB
#   bash hpc/hpc_alloc_tmux.sh -p gpu_a100_short -n 12 -t 00:30:00 --mem=32GB --gres=gpu:1
#
# Runs as a batch job holding a detached tmux session on the compute node, so it
# survives disconnects. The job ends when the session is closed or time runs out.
# The socket lives in ~/.tmux-sockets because each Slurm step gets a private /tmp.
#   Attach:   srun --jobid=<JOBID> --overlap --pty tmux -S ~/.tmux-sockets/<SESSION>-<JOBID> attach
#   Run cmd:  srun --jobid=<JOBID> --overlap tmux -S ~/.tmux-sockets/<SESSION>-<JOBID> send-keys '<cmd>' Enter

set -euo pipefail

SESSION="${TMUX_SESSION_NAME:-day-$(date +%F)}"
PARTITION="${PARTITION:-cpu_il}"
N="${N:-8}"
TIME="${TIME:-10:00:00}"
MEM="${MEM:-12GB}"
GRES=""

usage() {
  cat <<EOF
Usage: $0 [-p partition] [-n cores (cpus-per-task)] [-t time] [--mem=MEM] [--gres=GRES] [-s session]

Defaults: -p ${PARTITION} -n ${N} -t ${TIME} --mem=${MEM} -s ${SESSION}

Submits a batch job that keeps a detached tmux session '${SESSION}' on the
compute node, then prints the attach command.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p) PARTITION="$2"; shift 2 ;;
    -n) N="$2"; shift 2 ;;
    -t) TIME="$2"; shift 2 ;;
    -s) SESSION="$2"; shift 2 ;;
    --mem=*) MEM="${1#--mem=}"; shift ;;
    --mem) MEM="$2"; shift 2 ;;
    --gres=*) GRES="${1#--gres=}"; shift ;;
    --gres) GRES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Already inside Slurm job ${SLURM_JOB_ID}. Start tmux here instead:" >&2
  echo "  tmux new-session -A -s ${SESSION}" >&2
  exit 1
fi

SBATCH_ARGS=(-p "${PARTITION}" -n 1 -c "${N}" -t "${TIME}" --mem="${MEM}" -J "tmux-${SESSION}")
if [[ -n "${GRES}" ]]; then
  SBATCH_ARGS+=(--gres="${GRES}")
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${PROJECT_DIR}/logs" "${HOME}/.tmux-sockets"

echo "Requesting allocation: ${SBATCH_ARGS[*]}"
JOBID=$(sbatch --parsable "${SBATCH_ARGS[@]}" -o "${PROJECT_DIR}/logs/alloc_%j.out" \
  --wrap "export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK; S=\$HOME/.tmux-sockets/${SESSION}-\$SLURM_JOB_ID; tmux -S \$S new-session -d -s '${SESSION}' -c '${PROJECT_DIR}' && while tmux -S \$S has-session 2>/dev/null; do sleep 60; done; rm -f \$S")
SOCK="${HOME}/.tmux-sockets/${SESSION}-${JOBID}"

echo "Job ${JOBID} submitted; waiting for it to start..."
until [[ "$(squeue -h -j "${JOBID}" -o %T)" == RUNNING ]]; do sleep 5; done
until [[ -S "${SOCK}" ]]; do sleep 2; done
echo "Running on $(squeue -h -j "${JOBID}" -o %N), tmux session '${SESSION}'"
echo "Attach: srun --jobid=${JOBID} --overlap --pty tmux -S ${SOCK} attach"
