#!/usr/bin/env bash
# Independent canonical L128 ensemble: radial heat bath + embedded Wolff sign clusters.
# Usage: bash $0 r1 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/../../.venv/bin/python"
GENERATOR="$ROOT/phi4_phase-diagram/src/generate_phi4_embedded_wolff_radial_heatbath.py"

RUN_NUMBER="${1:-}"; shift || true
[[ "$RUN_NUMBER" =~ ^r[0-9]+$ ]] || {
  echo "usage: $0 rNUMBER [--execute] [--background]" >&2
  exit 2
}
EXEC=0; BG=0
for flag in "$@"; do
  case "$flag" in
    --execute) EXEC=1 ;;
    --background) BG=1 ;;
    *) echo "unknown argument: $flag" >&2; exit 2 ;;
  esac
done

# These are independent samples, not a continuation of the L8->...->L64
# upscale chain.  The defaults give a 1500-configuration target comparable to
# the L128 proposal ensemble, with a conservative independent thermalization.
N_CONFIGS="${N_CONFIGS:-1500}"
THERMAL_SWEEPS="${THERMAL_SWEEPS:-2000}"
SKIP_SWEEPS="${SKIP_SWEEPS:-20}"
CLUSTERS_PER_SWEEP="${CLUSTERS_PER_SWEEP:-1}"
SEED="${SEED:-20260813128}"
OUT="$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L128"
LOG_DIR="$ROOT/data/configs_phi4_2d/logs"
LOG="$LOG_DIR/generate_wolff_radial_L128_kappa0p340301_N${N_CONFIGS}_${RUN_NUMBER}.log"
PID_FILE="$LOG_DIR/generate_wolff_radial_L128_kappa0p340301_N${N_CONFIGS}_${RUN_NUMBER}.pid"
CMD=(env PYTHONUNBUFFERED=1 "$PY" -u -B "$GENERATOR"
  --lambda 1.0 --kappa 0.340301 --L 128
  --n-configs "$N_CONFIGS" --thermal-sweeps "$THERMAL_SWEEPS"
  --skip-sweeps "$SKIP_SWEEPS" --clusters-per-sweep "$CLUSTERS_PER_SWEEP"
  --seed "$SEED" --output-dir "$OUT")

printf 'output_dir=%s\nconfigs=%s/configs.npz\nstatus=%s/streaming_status.json\ncommand=' "$OUT" "$OUT" "$OUT"
printf '%q ' "${CMD[@]}"
printf '\n'
[[ $EXEC -eq 1 ]] || exit 0
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite existing ensemble directory: $OUT" >&2; exit 1; }
mkdir -p "$LOG_DIR"
{
  echo "submitted_at=$(date -Iseconds)"
  echo "run_number=$RUN_NUMBER"
  echo "algorithm=embedded Wolff sign cluster + exact radial heat bath"
  echo "independent_of_upscaling_chain=true"
  printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
} >"${LOG}.manifest"
if [[ $BG -eq 1 ]]; then
  nohup "${CMD[@]}" >"$LOG" 2>&1 < /dev/null &
  echo "$!" >"$PID_FILE"
  echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
