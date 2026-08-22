#!/usr/bin/env bash
# Rethermalize Ethan-7x7 L64->L128 sweep-zero fields at kappa_f=0.340330.
# The input fields were produced by upscaling native L64 configurations.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/../.venv/bin/python"
[[ -x "$PY" ]] || { echo "shared Python interpreter not found; set PYTHON=/path/to/python" >&2; exit 1; }

EXECUTE=0; BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done

N_CHAINS="${N_CHAINS:-1500}"
N_SWEEPS="${N_SWEEPS:-100}"
SAVE_EVERY="${SAVE_EVERY:-5}"
KAPPA_C=0.340301
KAPPA_F=0.340330
INPUT="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L64toL128/L64toL128_ethan7_N1500_S100_global_tau2_n51_eps2over51_r2/levels/L64toL128/checkpoints/checkpoint_sweep_0000.npz"
NATIVE="$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L128/configs.npz"
RUN="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L64toL128_ethan7_sweep0_kf0p34033_N${N_CHAINS}_S${N_SWEEPS}_tau2_n51_eps2over51_r1"

CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py"
  --run-dir "$RUN" --initialization input --input-source "$INPUT" --native-source "$NATIVE"
  --n-chains "$N_CHAINS" --n-sweeps "$N_SWEEPS" --save-every "$SAVE_EVERY"
  --batch-size 25 --hmc-batch-size 25 --measurement-batch-size 25
  --divide 1 --step-size 0.0392156862745098 --leapfrog-steps 51 --seed 2026082503
  --kappa "$KAPPA_F" --kappa-coarse "$KAPPA_C" --level-name L64toL128)

printf 'L64->L128 sweep-zero rethermalization: N=%s, sweeps=%s\n' "$N_CHAINS" "$N_SWEEPS"
printf 'initialization=%s\n' "$INPUT"
printf 'HMC target: lambda=1, kappa_c=%.6f -> kappa_f=%.6f; global volume, tau=2, n=51, eps=2/51\n' "$KAPPA_C" "$KAPPA_F"
printf 'output=%s\n' "$RUN"

[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$INPUT" ]] || { echo "missing sweep-zero input: $INPUT" >&2; exit 1; }
[[ -f "$NATIVE" ]] || { echo "missing L128 native reference: $NATIVE" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "refusing to overwrite existing output: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null &
  echo "$!" >"$RUN/submit_pid.txt"
  echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
