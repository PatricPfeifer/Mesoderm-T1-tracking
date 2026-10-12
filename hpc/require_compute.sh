#!/usr/bin/env bash
# Source from HPC launchers. Heavy jobs must run on a Slurm compute allocation
# (typically tmux after salloc / hpc/hpc_alloc_tmux.sh), not a login node.
#
# Override for local tests: ALLOW_LOGIN_NODE=1

if [[ "${ALLOW_LOGIN_NODE:-}" == "1" ]]; then
  echo "ALLOW_LOGIN_NODE=1 — skipping compute-node check" >&2
  return 0 2>/dev/null || exit 0
fi

host="$(hostname)"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refuse: not inside a Slurm job (host=${host})." >&2
  echo "Start an allocation tmux first, e.g. bash hpc/hpc_alloc_tmux.sh" >&2
  echo "or: salloc ... then tmux, then re-run this script there." >&2
  return 1 2>/dev/null || exit 1
fi

# bwUniCluster login hosts look like uc2 / uc2.localdomain, not uc2n872.
if [[ "${host}" == *login* || "${host}" =~ ^uc[0-9]+(\.|$) ]]; then
  echo "Refuse: login node (${host}, job=${SLURM_JOB_ID})." >&2
  echo "Attach tmux on the compute node of this allocation, then re-run." >&2
  return 1 2>/dev/null || exit 1
fi
