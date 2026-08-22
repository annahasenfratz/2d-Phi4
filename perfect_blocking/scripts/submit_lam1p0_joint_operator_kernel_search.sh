#!/usr/bin/env bash
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$ROOT/../.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$ROOT/../../.venv/bin/python"
DRIVER="$ROOT/perfect_blocking/scripts/run_lam1p0_joint_operator_kernel_search.py"
RUN_DIR="$ROOT/perfect_blocking/perfect_blocking_lam1p0/tests/intermediate/archive_superseded_kernel_searches_20260818/joint_kurtosis_correlations_5000"
CMD=("$PYTHON" -B "$DRIVER")
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1;; --background) BACKGROUND=1;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2;; esac; done
[[ "$BACKGROUND" -eq 0 || "$EXECUTE" -eq 1 ]] || { echo "--background requires --execute" >&2; exit 2; }
[[ -f "$DRIVER" ]] && [[ -x "$PYTHON" ]] || { echo "missing driver or shared Python" >&2; exit 1; }
printf 'output=%s\ncommand=' "$RUN_DIR"; printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "$EXECUTE" -eq 0 ]]; then echo "Prepared only. Add --execute to start."; exit 0; fi
mkdir -p "$RUN_DIR/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" > "$RUN_DIR/logs/run.log" 2>&1 < /dev/null & echo "$!" > "$RUN_DIR/submit_pid.txt"; printf 'background_pid=%s\n' "$!"; else exec "${CMD[@]}"; fi
