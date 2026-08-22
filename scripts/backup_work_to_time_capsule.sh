#!/usr/bin/env bash
# Backup every research and support file below /Users/anna/Work.
# See backup_inverse_rg_to_time_capsule.sh for snapshot and exclusion details.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_ROOT="/Users/anna/Work" \
DEST_ROOT="/Volumes/Data/Research_Backups/Work" \
exec bash "$SCRIPT_DIR/backup_inverse_rg_to_time_capsule.sh" "$@"
