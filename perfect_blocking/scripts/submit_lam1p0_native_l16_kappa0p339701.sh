#!/usr/bin/env bash
# Generate an independent direct-native L16 ensemble at kappa=0.339701.
# Usage: bash $0 r1 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$ROOT/../../.venv/bin/python"
GENERATOR="$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L16/generate_phi4_embedded_wolff_radial_heatbath.py"
RUN_NUMBER="${1:-}"
if [[ ! "$RUN_NUMBER" =~ ^r[0-9]+$ ]]; then
  echo "usage: $0 rNUMBER [--execute] [--background]" >&2
  exit 2
fi
shift

KAPPA="${KAPPA:-0.339701}"
N_CONFIGS="${N_CONFIGS:-5000}"
THERMAL_SWEEPS="${THERMAL_SWEEPS:-500}"
SKIP_SWEEPS="${SKIP_SWEEPS:-15}"
CLUSTERS_PER_SWEEP="${CLUSTERS_PER_SWEEP:-1}"
SEED="${SEED:-2026080711}"
KAPPA_TAG="${KAPPA/./p}"
OUT="$ROOT/data/configs_phi4_2d/lam1p0_kappa${KAPPA_TAG}_L16_N${N_CONFIGS}_${RUN_NUMBER}"
LOG_DIR="$ROOT/data/configs_phi4_2d/logs"
LOG="$LOG_DIR/generate_wolff_radial_L16_kappa${KAPPA_TAG}_N${N_CONFIGS}_${RUN_NUMBER}.log"
CMD=("$PYTHON" -u -B "$GENERATOR"
  --lambda 1.0 --kappa "$KAPPA" --L 16 --n-configs "$N_CONFIGS"
  --thermal-sweeps "$THERMAL_SWEEPS" --skip-sweeps "$SKIP_SWEEPS"
  --clusters-per-sweep "$CLUSTERS_PER_SWEEP" --seed "$SEED" --output-dir "$OUT")

EXECUTE=0
BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 rNUMBER [--execute] [--background]" >&2; exit 2 ;;
  esac
done
[[ "$BACKGROUND" -eq 0 || "$EXECUTE" -eq 1 ]] || { echo "--background requires --execute" >&2; exit 2; }
for path in "$PYTHON" "$GENERATOR"; do [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }; done
# A failed background launch may leave only submit metadata.  That is safe to
# retry; never overwrite an ensemble that has begun writing generator files.
if [[ -e "$OUT" ]]; then
  for entry in "$OUT"/*; do
    [[ -e "$entry" ]] || continue
    case "$(basename "$entry")" in
      submit_manifest.txt|submit_pid.txt) ;;
      *) echo "output directory already contains generator output: $OUT" >&2; exit 1 ;;
    esac
  done
fi

printf 'output_dir=%s\n' "$OUT"
printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "$EXECUTE" -eq 0 ]]; then
  echo "Prepared only. Add --execute to start; add --background for nohup."
  exit 0
fi
mkdir -p "$OUT" "$LOG_DIR"
printf 'run_number=%s\n' "$RUN_NUMBER" > "$OUT/submit_manifest.txt"
printf 'command=' >> "$OUT/submit_manifest.txt"; printf '%q ' "${CMD[@]}" >> "$OUT/submit_manifest.txt"; printf '\n' >> "$OUT/submit_manifest.txt"
if [[ "$BACKGROUND" -eq 1 ]]; then
  # Linux systems can fully detach with `setsid`; macOS normally relies on
  # nohup, which is sufficient for this non-interactive generator.
  if command -v setsid >/dev/null 2>&1; then
    setsid "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
  else
    nohup "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
  fi
  printf '%s\n' "$!" > "$OUT/submit_pid.txt"
  printf 'background_pid=%s\nlog=%s\n' "$!" "$LOG"
else
  exec "${CMD[@]}"
fi
