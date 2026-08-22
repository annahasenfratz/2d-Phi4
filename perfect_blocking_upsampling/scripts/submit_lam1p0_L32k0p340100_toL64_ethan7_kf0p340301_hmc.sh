#!/usr/bin/env bash
# Clean off-critical test: direct L32 at kappa_c=0.340100 -> Ethan-7x7 L64,
# then full-volume L64 HMC at kappa_f=0.340301.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/../.venv/bin/python"
[[ -x "$PY" ]] || { echo "shared Python interpreter not found; set PYTHON=/path/to/python" >&2; exit 1; }
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;; esac; done
N=1500; KAPPA_C=0.340100; KAPPA_F=0.340301
CONFIG="$ROOT/perfect_blocking_upsampling/run_configs/lam1p0_flowonly_L32k0p340100_toL64_ethan7.json"
FLOW_RUN="$ROOT/perfect_blocking_upsampling/outputs/flow_only_lam1p0/L32toL64/L32k0p340100_toL64_ethan7_N1500_sweep0_r1"
INPUT="$FLOW_RUN/levels/L32toL64/checkpoints/checkpoint_sweep_0000.npz"
NATIVE="$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz"
RUN="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L32k0p340100_toL64_ethan7_sweep0_kf0p340301_N1500_S100_tau2_n36_eps2over36_r1"
FLOW_CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_hmc_upscale_chain.py" --config "$CONFIG" --run-dir "$FLOW_RUN" --n-chains "$N" --start-index 0)
HMC_CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py" --run-dir "$RUN" --initialization input --input-source "$INPUT" --native-source "$NATIVE" --n-chains "$N" --n-sweeps 100 --save-every 5 --batch-size 50 --hmc-batch-size 50 --measurement-batch-size 50 --divide 1 --step-size 0.05555555555555555 --leapfrog-steps 36 --seed 2026082524 --kappa "$KAPPA_F" --kappa-coarse "$KAPPA_C" --level-name L32toL64)
printf 'stage 1: direct L32 kappa_c=%.6f -> Ethan-7x7 L64 sweep zero: %s\n' "$KAPPA_C" "$FLOW_RUN"
printf 'stage 2: L64 HMC kappa_f=%.6f, global tau=2, n=36, eps=2/36: %s\n' "$KAPPA_F" "$RUN"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ ! -e "$FLOW_RUN" && ! -e "$RUN" ]] || { echo 'refusing to overwrite existing flow/HMC output' >&2; exit 1; }
mkdir -p "$FLOW_RUN/logs" "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  FLOW_Q=$(printf ' %q' "${FLOW_CMD[@]}"); HMC_Q=$(printf ' %q' "${HMC_CMD[@]}")
  nohup bash -c "$FLOW_Q && $HMC_Q" >"$RUN/logs/run.log" 2>&1 </dev/null &
  echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"
else
  "${FLOW_CMD[@]}"
  "${HMC_CMD[@]}" 2>&1 | tee "$RUN/logs/run.log"
fi
