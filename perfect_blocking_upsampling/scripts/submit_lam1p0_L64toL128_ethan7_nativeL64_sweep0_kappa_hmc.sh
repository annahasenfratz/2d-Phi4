#!/usr/bin/env bash
# Paired clean native-L64 -> Ethan-7x7 L128 kappa scan.
# The shared flow-only sweep-zero ensemble was built from direct L64 fields.
# Usage: bash $0 --kappa VALUE [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/../.venv/bin/python"
[[ -x "$PY" ]] || { echo "shared Python interpreter not found; set PYTHON=/path/to/python" >&2; exit 1; }
EXECUTE=0; BACKGROUND=0; KAPPA_F=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kappa) [[ $# -ge 2 ]] || { echo '--kappa requires a value' >&2; exit 2; }; KAPPA_F="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --background) BACKGROUND=1; shift ;;
    *) echo "usage: $0 --kappa VALUE [--execute] [--background]" >&2; exit 2 ;;
  esac
done
case "$KAPPA_F" in 0.340301|0.340320|0.340340) ;; *) echo 'kappa must be one of 0.340301, 0.340320, 0.340340' >&2; exit 2 ;; esac
KAPPA_TAG="${KAPPA_F/./p}"
N_CHAINS=1500; N_SWEEPS=100; SAVE_EVERY=5; KAPPA_C=0.340301
INPUT="$ROOT/perfect_blocking_upsampling/outputs/flow_only_lam1p0/L64toL128/L64toL128_ethan7_nativeL64_N1500_sweep0_r1/levels/L64toL128/checkpoints/checkpoint_sweep_0000.npz"
NATIVE="$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L128/configs.npz"
RUN="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L64toL128_ethan7_nativeL64_sweep0_kf${KAPPA_TAG}_N1500_S100_tau2_n51_eps2over51_r1"
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py" --run-dir "$RUN" --initialization input --input-source "$INPUT" --native-source "$NATIVE" --n-chains "$N_CHAINS" --n-sweeps "$N_SWEEPS" --save-every "$SAVE_EVERY" --batch-size 25 --hmc-batch-size 25 --measurement-batch-size 25 --divide 1 --step-size 0.0392156862745098 --leapfrog-steps 51 --seed 2026082507 --kappa "$KAPPA_F" --kappa-coarse "$KAPPA_C" --level-name L64toL128)
printf 'shared initialization=%s\npaired clean HMC: N=1500, sweeps=100, kappa_c=%.6f -> kappa_f=%.6f\noutput=%s\n' "$INPUT" "$KAPPA_C" "$KAPPA_F" "$RUN"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$INPUT" ]] || { echo "missing clean native-L64 sweep-zero input: $INPUT" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "refusing to overwrite existing output: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null &
  echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
