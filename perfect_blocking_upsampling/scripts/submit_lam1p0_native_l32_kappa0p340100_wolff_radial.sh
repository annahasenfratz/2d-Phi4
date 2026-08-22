#!/usr/bin/env bash
# Independent direct-native L32 ensemble at kappa=0.340100.
# Algorithm: one radial heat-bath sweep plus one embedded Wolff sign cluster
# per recorded sweep. Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/../.venv/bin/python"
[[ -x "$PY" ]] || { echo "shared Python interpreter not found; set PYTHON=/path/to/python" >&2; exit 1; }
GENERATOR="$ROOT/phi4_phase-diagram/src/generate_phi4_embedded_wolff_radial_heatbath.py"
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;; esac; done
L=32; KAPPA=0.340100; N_CONFIGS=1500; THERMAL_SWEEPS=500; SKIP_SWEEPS=15; CLUSTERS_PER_SWEEP=1; SEED=2026082521
OUT="$ROOT/data/configs_phi4_2d/lam1p0_kappa0p340100_L32_N1500_r1"
LOG_DIR="$ROOT/data/configs_phi4_2d/logs"; LOG="$LOG_DIR/generate_wolff_radial_L32_kappa0p340100_N1500_r1.log"
CMD=(env PYTHONUNBUFFERED=1 "$PY" -u -B "$GENERATOR" --lambda 1.0 --kappa "$KAPPA" --L "$L" --n-configs "$N_CONFIGS" --thermal-sweeps "$THERMAL_SWEEPS" --skip-sweeps "$SKIP_SWEEPS" --clusters-per-sweep "$CLUSTERS_PER_SWEEP" --seed "$SEED" --output-dir "$OUT")
printf 'output_dir=%s\nalgorithm=embedded Wolff sign cluster + radial heat bath\nthermal=%s, skip=%s, configs=%s\ncommand=' "$OUT" "$THERMAL_SWEEPS" "$SKIP_SWEEPS" "$N_CONFIGS"; printf '%q ' "${CMD[@]}"; printf '\n'
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$GENERATOR" ]] || { echo "missing generator: $GENERATOR" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite existing output: $OUT" >&2; exit 1; }
mkdir -p "$LOG_DIR"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" >"$LOG" 2>&1 </dev/null &
  echo "$!" >"$OUT.submit_pid"
  echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
