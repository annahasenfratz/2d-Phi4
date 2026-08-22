#!/usr/bin/env bash
# Incremental, non-destructive research snapshots on the mounted Time Capsule.
#
# Default source: the Inverse_RG workspace containing phi4, O(3), and XY.
# Default destination: /Volumes/Data/Research_Backups/Inverse_RG
#
# Each completed run creates snapshots/YYYYmmdd_HHMMSS.  Existing snapshots
# are never deleted by this script.  The mounted Time Capsule AFP share does
# not support Unix hard links, so this script deliberately does not request
# rsync hard-link preservation or --link-dest snapshots.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DEST_ROOT="${DEST_ROOT:-/Volumes/Data/Research_Backups/Inverse_RG}"
DRY_RUN=0
RESUME_STAMP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --resume-stamp)
      [[ $# -ge 2 ]] || { echo "--resume-stamp requires YYYYmmdd_HHMMSS" >&2; exit 2; }
      RESUME_STAMP="$2"; shift ;;
    *) echo "usage: $0 [--dry-run] [--resume-stamp YYYYmmdd_HHMMSS]" >&2; exit 2 ;;
  esac
  shift
done

[[ -d "$SOURCE_ROOT" ]] || { echo "missing source: $SOURCE_ROOT" >&2; exit 1; }
[[ -d /Volumes/Data ]] || {
  echo "Time Capsule share is not mounted at /Volumes/Data" >&2
  exit 1
}

SNAPSHOTS="$DEST_ROOT/snapshots"
STAMP="$(date +%Y%m%d_%H%M%S)"
SNAPSHOT_STAMP="${RESUME_STAMP:-$STAMP}"
SNAPSHOT="$SNAPSHOTS/$SNAPSHOT_STAMP"
LATEST="$DEST_ROOT/latest"

if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$SNAPSHOT"
fi
if [[ -n "$RESUME_STAMP" && ! -d "$SNAPSHOT" ]]; then
  echo "cannot resume missing snapshot: $SNAPSHOT" >&2
  exit 1
fi

RSYNC_ARGS=(
  -a
  # AFP does not support the POSIX ownership, mode, and timestamp operations
  # requested by -a.  Retain its recursive/copy semantics while disabling
  # those unsupported metadata writes so a partial snapshot can be resumed.
  --no-perms
  --no-owner
  --no-group
  --no-times
  --partial
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
fi

printf 'Source:      %s\nDestination: %s\n' "$SOURCE_ROOT" "$SNAPSHOT"
if [[ -n "$RESUME_STAMP" ]]; then
  printf 'Mode:        resume partial snapshot %s\n' "$RESUME_STAMP"
else
  printf 'Mode:        new standalone snapshot (AFP has no hard-link support)\n'
fi

rsync "${RSYNC_ARGS[@]}" "$SOURCE_ROOT/" "$SNAPSHOT/"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete; no snapshot was created."
  exit 0
fi

{
  printf 'Created: %s\n' "$(date -Iseconds)"
  printf 'Source: %s\n' "$SOURCE_ROOT"
  printf 'Excluded: caches, virtual environments, editor artifacts, and transient Git locks.\n'
  printf 'Snapshot size:\n'
  du -sh "$SNAPSHOT"
} >"$SNAPSHOT/BACKUP_MANIFEST.txt"

ln -s "$SNAPSHOT_STAMP" "$DEST_ROOT/latest.new"
mv -f "$DEST_ROOT/latest.new" "$LATEST"

echo "Snapshot complete: $SNAPSHOT"
df -h /Volumes/Data
