#!/usr/bin/env bash
# Ethan 7x7 objective with a fitted eta normalization. Non-promoting.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$ROOT/../../.venv/bin/python"
DRIVER="$ROOT/perfect_blocking/scripts/run_lam1p0_ethan_7x7_free_eta.py"
OUT="$ROOT/perfect_blocking/perfect_blocking_lam1p0/tests/intermediate/ethan_7x7_free_eta_mu30_N5_train9000_test1000_20260820"
EXECUTE=0
BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done
[[ "$BACKGROUND" -eq 0 || "$EXECUTE" -eq 1 ]] || { echo "--background requires --execute" >&2; exit 2; }
[[ -x "$PYTHON" && -f "$DRIVER" ]] || { echo "missing Python environment or driver" >&2; exit 1; }
CMD=("$PYTHON" -B "$DRIVER" --out "$OUT" --kernel "$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/ethan_7x7_paper_objective_eta_included.json" --starts 4 --maxiter 160 --eta-min 0.0 --eta-max 0.5 --mu 30 --locality-box 5)
printf 'output=%s\neta range=[0, 0.5], starting eta=0.25\n' "$OUT"
printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "$EXECUTE" -eq 0 ]]; then
  echo "Prepared only. Add --execute to start."
  exit 0
fi
mkdir -p "$OUT/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" > "$OUT/logs/run.log" 2>&1 < /dev/null &
  echo "$!" > "$OUT/submit_pid.txt"
  printf 'background_pid=%s\n' "$!"
else
  exec "${CMD[@]}"
fi
