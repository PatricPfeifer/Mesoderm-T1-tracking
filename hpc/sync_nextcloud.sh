#!/usr/bin/env bash
# Sync a local folder with bwSync&Share (Nextcloud) via rclone WebDAV.
#
# Usage:
#   bash hpc/sync_nextcloud.sh push [--dry-run] [FOLDER...]   # project folders -> cloud
#   bash hpc/sync_nextcloud.sh pull [--dry-run] [FOLDER...]   # cloud -> project folders
# FOLDER is a project-relative name (default: input output); each goes to
# bwsync:Patric-master-thesis/<FOLDER>.
#
# One-time setup: see "rclone remote" below. Uses `copy` (never deletes on the target).
set -euo pipefail

REMOTE_NAME="${RCLONE_REMOTE:-bwsync}"          # name given in `rclone config`
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_BASE="${SYNC_REMOTE_BASE:-Patric-master-thesis}"

mode="${1:-}"; shift || true
dry=()
if [[ "${1:-}" == "--dry-run" ]]; then dry=(--dry-run); shift; fi
folders=("$@"); [[ ${#folders[@]} -gt 0 ]] || folders=(input output)

if ! rclone listremotes | grep -qx "${REMOTE_NAME}:"; then
  cat >&2 <<EOF
rclone remote '${REMOTE_NAME}' not found. Create it once with:
  rclone config create ${REMOTE_NAME} webdav \\
    url https://bwsyncandshare.kit.edu/remote.php/dav/files/<USERNAME>/ \\
    vendor nextcloud user <USERNAME>
  rclone config password ${REMOTE_NAME} pass '<APP-PASSWORD>'
(<USERNAME> and an app password come from bwSync&Share -> Settings -> Security.)
EOF
  exit 1
fi

flags=(--progress --transfers 4 --checksum --exclude '.git/**' --exclude '__pycache__/**' "${dry[@]}")

case "$mode" in push|pull) ;; *) sed -n '2,8p' "$0"; exit 2 ;; esac

for f in "${folders[@]}"; do
  local_dir="$PROJECT_DIR/$f"; remote="${REMOTE_NAME}:${REMOTE_BASE}/$f"
  echo "== $mode $f"
  if [[ $mode == push ]]; then rclone copy "${flags[@]}" "$local_dir" "$remote"
  else rclone copy "${flags[@]}" "$remote" "$local_dir"; fi
done
