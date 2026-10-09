#!/usr/bin/env bash
# Allocate a compute node and start tmux there (not on the login node).
#
# Usage (from project root or anywhere):
#   bash scripts/hpc_alloc_tmux.sh
#   bash scripts/hpc_alloc_tmux.sh -p cpu -n 8 -t 5:00:00 --mem=8GB
#   bash scripts/hpc_alloc_tmux.sh -p gpu_a100_short -n 12 -t 00:30:00 --mem=32GB --gres=gpu:1
#
# After attach, tell Cursor the session name is: patric

set -euo pipefail

SESSION="${TMUX_SESSION_NAME:-patric}"
PARTITION="${PARTITION:-cpu_il}"
N="${N:-12}"
TIME="${TIME:-4:00:00}"
MEM="${MEM:-8GB}"
GRES=""

usage() {
  cat <<EOF
Usage: $0 [-p partition] [-n cores] [-t time] [--mem=MEM] [--gres=GRES] [-s session]

Defaults: -p ${PARTITION} -n ${N} -t ${TIME} --mem=${MEM} -s ${SESSION}

Clears leftover login-node tmux sessions, then salloc + srun --pty tmux on the
compute allocation. Session name (for Cursor): ${SESSION}
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

echo "Clearing login-node tmux sessions..."
if tmux ls >/dev/null 2>&1; then
  tmux kill-server
  echo "Login tmux server killed."
else
  echo "No login tmux sessions."
fi

SALLOC_ARGS=(-p "${PARTITION}" -n "${N}" -t "${TIME}" --mem="${MEM}")
if [[ -n "${GRES}" ]]; then
  SALLOC_ARGS+=(--gres="${GRES}")
fi

echo "Requesting allocation: ${SALLOC_ARGS[*]}"
echo "On grant: tmux session '${SESSION}' on the compute node."
echo "Tell Cursor: session is ${SESSION}"

exec salloc "${SALLOC_ARGS[@]}" srun --pty tmux new-session -A -s "${SESSION}"
