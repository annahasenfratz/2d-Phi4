#!/usr/bin/env bash
# Clean L32->L64 test: direct native L32 -> Ethan-7x7 flow-only sweep zero,
# then full-volume L64 HMC at kappa_f=0.340330.
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
CONFIG="$ROOT/perfect_blocking_upsampling/run_configs/lam1p0_flowonly_L32toL64_ethan7_nativeL32.json"
FLOW_RUN="$ROOT/perfect_blocking_upsampling/outputs/flow_only_lam1p0/L32toL64/L32toL64_ethan7_nativeL32_N${N_CHAINS}_sweep0_r1"
INPUT="$FLOW_RUN/levels/L32toL64/checkpoints/checkpoint_sweep_0000.npz"
NATIVE_L32="$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
NATIVE_L64="$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz"
RUN="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L32toL64_ethan7_nativeL32_sweep0_kf0p34033_N${N_CHAINS}_S${N_SWEEPS}_tau2_n36_eps2over36_r1"

FLOW_CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_hmc_upscale_chain.py"
  --config "$CONFIG" --run-dir "$FLOW_RUN" --n-chains "$N_CHAINS" --start-index 0)
HMC_CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py"
  --run-dir "$RUN" --initialization input --input-source "$INPUT" --native-source "$NATIVE_L64"
  --n-chains "$N_CHAINS" --n-sweeps "$N_SWEEPS" --save-every "$SAVE_EVERY"
  --batch-size 50 --hmc-batch-size 50 --measurement-batch-size 50
  --divide 1 --step-size 0.05555555555555555 --leapfrog-steps 36 --seed 2026082505
  --kappa "$KAPPA_F" --kappa-coarse "$KAPPA_C" --level-name L32toL64)

printf 'clean source: direct native L32: %s\n' "$NATIVE_L32"
printf 'stage 1: Ethan-7x7 flow-only L32->L64, output=%s\n' "$FLOW_RUN"
printf 'stage 2: full-volume L64 HMC, kappa_c=%.6f -> kappa_f=%.6f; tau=2, n=36, eps=2/36\n' "$KAPPA_C" "$KAPPA_F"
printf 'final output=%s\n' "$RUN"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$NATIVE_L32" && -f "$NATIVE_L64" ]] || { echo "missing direct native input/reference" >&2; exit 1; }
[[ ! -e "$FLOW_RUN" ]] || { echo "refusing to overwrite flow-only output: $FLOW_RUN" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "refusing to overwrite HMC output: $RUN" >&2; exit 1; }
mkdir -p "$FLOW_RUN/logs" "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  FLOW_Q=$(printf ' %q' "${FLOW_CMD[@]}")
  HMC_Q=$(printf ' %q' "${HMC_CMD[@]}")
  nohup bash -c "$FLOW_Q && $HMC_Q" >"$RUN/logs/run.log" 2>&1 </dev/null &
  echo "$!" >"$RUN/submit_pid.txt"
  echo "background_pid=$!"
else
  "${FLOW_CMD[@]}"
  "${HMC_CMD[@]}" 2>&1 | tee "$RUN/logs/run.log"
fi
