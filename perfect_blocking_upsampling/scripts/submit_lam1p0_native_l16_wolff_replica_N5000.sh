#!/usr/bin/env bash
# Independent direct-native L16 replica for the three-flow HMC comparison.
# It intentionally writes a new ensemble and never appends to the original
# L16 training/source ensemble.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
GENERATOR="$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L16/generate_phi4_embedded_wolff_radial_heatbath.py"
[[ -x "$PY" ]] || { echo "Python interpreter not found: $PY" >&2; exit 1; }
[[ -f "$GENERATOR" ]] || { echo "Wolff generator not found: $GENERATOR" >&2; exit 1; }

EXECUTE=0
BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done

N_CONFIGS="${N_CONFIGS:-5000}"
OUT="$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L16_N${N_CONFIGS}_replica2_20260818"
CMD=("$PY" -u -B "$GENERATOR"
  --lambda 1.0 --kappa 0.340301 --L 16 --n-configs "$N_CONFIGS"
  --thermal-sweeps 500 --skip-sweeps 15 --clusters-per-sweep 1
  --seed 2026081818 --output-dir "$OUT")

printf 'output=%s\nN=%s\nembedded Wolff sign-cluster + radial heat-bath; thermal=500, skip=15, seed=2026081818\n' "$OUT" "$N_CONFIGS"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite existing output: $OUT" >&2; exit 1; }
mkdir -p "$OUT/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  { echo "started_at=$(date -Iseconds)"; printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'; } > "$OUT/logs/run.log"
  nohup "${CMD[@]}" >> "$OUT/logs/run.log" 2>&1 </dev/null &
  echo "$!" > "$OUT/submit_pid.txt"
  echo "background_pid=$!"
else
  "${CMD[@]}" 2>&1 | tee "$OUT/logs/run.log"
fi
