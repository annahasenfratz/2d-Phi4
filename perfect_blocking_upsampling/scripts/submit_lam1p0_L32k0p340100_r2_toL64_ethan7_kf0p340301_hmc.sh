#!/usr/bin/env bash
# Replica r2 clean test: direct L32(kappa_c=0.340100) -> Ethan-7x7 L64,
# then full-volume L64 HMC at kappa_f=0.340301.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/../.venv/bin/python"
[[ -x "$PY" ]] || { echo "shared Python interpreter not found; set PYTHON=/path/to/python" >&2; exit 1; }
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;; esac; done
SOURCE="$ROOT/data/configs_phi4_2d/lam1p0_kappa0p340100_L32_N1500_r2/configs.npz"
CONFIG="$ROOT/perfect_blocking_upsampling/run_configs/lam1p0_flowonly_L32k0p340100_r2_toL64_ethan7.json"
FLOW_RUN="$ROOT/perfect_blocking_upsampling/outputs/flow_only_lam1p0/L32toL64/L32k0p340100_r2_toL64_ethan7_N1500_sweep0_r1"
INPUT="$FLOW_RUN/levels/L32toL64/checkpoints/checkpoint_sweep_0000.npz"
NATIVE="$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz"
RUN="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L32k0p340100_r2_toL64_ethan7_sweep0_kf0p340301_N1500_S100_tau2_n36_eps2over36_r1"
FLOW_CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_hmc_upscale_chain.py" --config "$CONFIG" --run-dir "$FLOW_RUN" --n-chains 1500 --start-index 0)
HMC_CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py" --run-dir "$RUN" --initialization input --input-source "$INPUT" --native-source "$NATIVE" --n-chains 1500 --n-sweeps 100 --save-every 5 --batch-size 50 --hmc-batch-size 50 --measurement-batch-size 50 --divide 1 --step-size 0.05555555555555555 --leapfrog-steps 36 --seed 2026082532 --kappa 0.340301 --kappa-coarse 0.340100 --level-name L32toL64)
printf 'source=%s\nstage 1 flow-only=%s\nstage 2 HMC=%s\n' "$SOURCE" "$FLOW_RUN" "$RUN"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$SOURCE" ]] || { echo "replica-r2 source is not ready: $SOURCE" >&2; exit 1; }
[[ ! -e "$FLOW_RUN" && ! -e "$RUN" ]] || { echo 'refusing to overwrite existing replica-r2 output' >&2; exit 1; }
mkdir -p "$FLOW_RUN/logs" "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  FLOW_Q=$(printf ' %q' "${FLOW_CMD[@]}"); HMC_Q=$(printf ' %q' "${HMC_CMD[@]}")
  nohup bash -c "$FLOW_Q && $HMC_Q" >"$RUN/logs/run.log" 2>&1 </dev/null &
  echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"
else
  "${FLOW_CMD[@]}"
  "${HMC_CMD[@]}" 2>&1 | tee "$RUN/logs/run.log"
fi
