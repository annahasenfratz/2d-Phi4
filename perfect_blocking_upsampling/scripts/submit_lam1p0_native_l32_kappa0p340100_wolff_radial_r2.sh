#!/usr/bin/env bash
# Independent replica r2: 1500 direct-native L32 configs at kappa=0.340100.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/../.venv/bin/python"
[[ -x "$PY" ]] || { echo "shared Python interpreter not found; set PYTHON=/path/to/python" >&2; exit 1; }
GENERATOR="$ROOT/phi4_phase-diagram/src/generate_phi4_embedded_wolff_radial_heatbath.py"
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;; esac; done
OUT="$ROOT/data/configs_phi4_2d/lam1p0_kappa0p340100_L32_N1500_r2"
LOG_DIR="$ROOT/data/configs_phi4_2d/logs"; LOG="$LOG_DIR/generate_wolff_radial_L32_kappa0p340100_N1500_r2.log"
CMD=(env PYTHONUNBUFFERED=1 "$PY" -u -B "$GENERATOR" --lambda 1.0 --kappa 0.340100 --L 32 --n-configs 1500 --thermal-sweeps 500 --skip-sweeps 15 --clusters-per-sweep 1 --seed 2026082530 --output-dir "$OUT")
printf 'output_dir=%s\nalgorithm=embedded Wolff sign cluster + radial heat bath; thermal=500, skip=15, configs=1500\n' "$OUT"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$GENERATOR" ]] || { echo "missing generator: $GENERATOR" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite existing output: $OUT" >&2; exit 1; }
mkdir -p "$LOG_DIR"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" >"$LOG" 2>&1 </dev/null &
  echo "$!" >"$OUT.submit_pid"; echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
