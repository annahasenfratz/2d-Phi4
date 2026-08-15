#!/usr/bin/env bash
# Fit the highfield15 blocked action with explicit frozen correlation penalties.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PYTHON="$ROOT/../../.venv/bin/python"
SOURCE="${SOURCE:-$ROOT/perfect_blocking/perfect_blocking_lam1p0/tests/softcond7_blocked_action_relative_entropy_highfield15}"
OUT="${OUT:-$ROOT/perfect_blocking/perfect_blocking_lam1p0/tests/softcond7_blocked_action_correlation_regularized}"
CORRELATION_WEIGHT="${CORRELATION_WEIGHT:-0.01}"; EPOCHS="${EPOCHS:-1000}"; EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1;; --background) BACKGROUND=1;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2;; esac; done
[[ "$BACKGROUND" -eq 0 || "$EXECUTE" -eq 1 ]] || { echo "--background requires --execute" >&2; exit 2; }
CMD=("$PYTHON" -B "$ROOT/perfect_blocking/scripts/fit_lam1p0_blocked_action_correlation_regularized.py" --source "$SOURCE" --out "$OUT" --correlation-weight "$CORRELATION_WEIGHT" --epochs "$EPOCHS")
printf 'out=%s\ncommand=' "$OUT"; printf '%q ' "${CMD[@]}"; printf '\n'
[[ "$EXECUTE" -eq 1 ]] || { echo "Prepared only. Add --execute to start."; exit 0; }
mkdir -p "$OUT/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" > "$OUT/logs/run.log" 2>&1 < /dev/null & echo "$!" > "$OUT/submit_pid.txt"; echo "background_pid=$!"; else exec "${CMD[@]}"; fi
