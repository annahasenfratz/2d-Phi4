#!/usr/bin/env bash
# Fit S_b = S_c + Delta S for the soft-conditioned 7x7 L32->L16 kernel.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$ROOT/../../.venv/bin/python"
DRIVER="$ROOT/perfect_blocking/scripts/fit_lam1p0_blocked_action_relative_entropy.py"
KERNEL="${KERNEL:-$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R3_soft_conditioned_7x7_eta_included.json}"
OUT="${OUT:-$ROOT/perfect_blocking/perfect_blocking_lam1p0/tests/softcond7_blocked_action_relative_entropy}"
N_CONFIGS="${N_CONFIGS:-5000}"
N_TRAIN="${N_TRAIN:-3000}"
N_VALIDATION="${N_VALIDATION:-1000}"
N_BOOTSTRAP="${N_BOOTSTRAP:-250}"
RIDGE="${RIDGE:-1e-4}"
EXTRA_ACTION_OPERATORS="${EXTRA_ACTION_OPERATORS:-}"
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done
[[ "$BACKGROUND" -eq 0 || "$EXECUTE" -eq 1 ]] || { echo "--background requires --execute" >&2; exit 2; }
CMD=("$PYTHON" -B "$DRIVER" --kernel "$KERNEL" --out "$OUT" --n-configs "$N_CONFIGS" --n-train "$N_TRAIN" --n-validation "$N_VALIDATION" --n-bootstrap "$N_BOOTSTRAP" --ridge "$RIDGE")
[[ -z "$EXTRA_ACTION_OPERATORS" ]] || CMD+=(--extra-action-operators "$EXTRA_ACTION_OPERATORS")
printf 'out=%s\ncommand=' "$OUT"; printf '%q ' "${CMD[@]}"; printf '\n'
[[ "$EXECUTE" -eq 1 ]] || { echo "Prepared only. Add --execute to start."; exit 0; }
mkdir -p "$OUT/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" > "$OUT/logs/run.log" 2>&1 < /dev/null &
  echo "$!" > "$OUT/submit_pid.txt"
  echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
