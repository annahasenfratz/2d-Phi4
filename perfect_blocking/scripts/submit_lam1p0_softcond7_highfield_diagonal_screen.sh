#!/usr/bin/env bash
# Screen degree-six diagonal interactions after the accepted 13-term action.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PYTHON="$ROOT/../../.venv/bin/python"
OUT="${OUT:-$ROOT/perfect_blocking/perfect_blocking_lam1p0/tests/archive_superseded_kernel_explorations_20260818/softcond7_blocked_action_relative_entropy}"
DRIVER="$ROOT/perfect_blocking/scripts/scan_lam1p0_blocked_action_extensions.py"
BASE_EXTRA="${BASE_EXTRA:-diag_phi2_neighbor,diag_bond_sq,3nn}"
CANDIDATES="${CANDIDATES:-diag_phi4_neighbor,diag_phi3phi3,diag_phi2_bond_sq}"
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1;; --background) BACKGROUND=1;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2;; esac; done
[[ "$BACKGROUND" -eq 0 || "$EXECUTE" -eq 1 ]] || { echo "--background requires --execute" >&2; exit 2; }
CMD=("$PYTHON" -B "$DRIVER" --out "$OUT" --base-extra "$BASE_EXTRA" --candidates "$CANDIDATES" --label highfield_diagonal_after_greedy13)
printf 'out=%s\ncommand=' "$OUT"; printf '%q ' "${CMD[@]}"; printf '\n'
[[ "$EXECUTE" -eq 1 ]] || { echo "Prepared only. Add --execute to start."; exit 0; }
mkdir -p "$OUT/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" > "$OUT/logs/highfield_diagonal_screen.log" 2>&1 < /dev/null & echo "$!" > "$OUT/highfield_diagonal_screen_pid.txt"; echo "background_pid=$!"; else exec "${CMD[@]}"; fi
