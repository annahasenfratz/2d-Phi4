#!/usr/bin/env bash
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Set PYTHON to the environment with numpy/scipy." >&2; exit 1; }
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1;; --background) BACKGROUND=1;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2;; esac; done
OUT="$ROOT/perfect_blocking/perfect_blocking_lam1p0/tests/intermediate/7x7_twovolume_from5x5"
CMD=("$PYTHON" -B "$ROOT/perfect_blocking/scripts/run_lam1p0_7x7_twovolume_reopt.py" --out "$OUT")
printf 'out=%s\npython=%s\n' "$OUT" "$PYTHON"
[[ "$EXECUTE" -eq 1 ]] || exit 0
mkdir -p "$OUT/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" >"$OUT/logs/run.log" 2>&1 </dev/null & echo $! >"$OUT/submit_pid.txt"; echo "background_pid=$!"; else exec "${CMD[@]}"; fi
