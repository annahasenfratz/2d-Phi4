#!/usr/bin/env bash
# Keep one current rsync mirror of /Users/anna/Work and retain replaced or
# deleted destination files for 90 days.  This is intentionally not a
# versioned full-snapshot scheme: after the initial mirror, only deltas move.
set -euo pipefail

SOURCE_ROOT="/Users/anna/Work"
DEST_ROOT="/Volumes/Data/Research_Backups/Work"
CURRENT="$DEST_ROOT/current"
ARCHIVE_ROOT="$DEST_ROOT/incremental_archive"
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
  esac
done

[[ -d /Volumes/Data ]] || { echo "Time Capsule is not mounted at /Volumes/Data; skipping." >&2; exit 0; }
[[ -d "$SOURCE_ROOT" ]] || { echo "missing source: $SOURCE_ROOT" >&2; exit 1; }
[[ -e "$CURRENT" ]] || { echo "missing current mirror: $CURRENT" >&2; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$ARCHIVE_ROOT/$STAMP"
RSYNC_ARGS=(
  -rltD
  --no-perms
  --no-owner
  --no-group
  --partial
  --delete
  --backup
  --backup-dir="$ARCHIVE"
  --human-readable
  --itemize-changes
  --exclude='.venv/'
  --exclude='__pycache__/'
  --exclude='.pytest_cache/'
  --exclude='.ipynb_checkpoints/'
  --exclude='.DS_Store'
  --exclude='*.pyc'
  --exclude='*.swp'
  --exclude='.git/index.lock'
)
if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC_ARGS+=(--dry-run)
else
  mkdir -p "$ARCHIVE"
fi

printf 'Source: %s\nCurrent mirror: %s\nRetention archive: %s (90 days)\n' "$SOURCE_ROOT" "$CURRENT" "$ARCHIVE"
rsync "${RSYNC_ARGS[@]}" "$SOURCE_ROOT/" "$CURRENT/"

if [[ "$DRY_RUN" -eq 0 ]]; then
  rmdir "$ARCHIVE" 2>/dev/null || true
  find "$ARCHIVE_ROOT" -mindepth 1 -maxdepth 1 -type d -mtime +90 -exec rm -rf {} +
fi
