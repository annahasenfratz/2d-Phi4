#!/usr/bin/env bash
# Deliberately mis-normalized Ethan-7x7 initial condition: retain the same
# L128->L256 flow sample, but apply 2^(-eta/2) to its physical fine field.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
SOURCE="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L128toL256/L128toL256_ethan7_N1500_S100_global_tau2_n72_eps2over72_streamed_r1/levels/L128toL256/checkpoints/checkpoint_sweep_0000.npz"
RUN="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L128toL256_ethan7_norm1_sameflow_N1500_S100_tau2_n72_eps2over72_r1"
SCALE="0.9170040432046712"  # 2^(-eta/2), eta=0.25
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py"
  --run-dir "$RUN" --initialization input --input-source "$SOURCE" --input-scale "$SCALE"
  --n-chains 1500 --n-sweeps 100 --save-every 5 --batch-size 10 --hmc-batch-size 10 --measurement-batch-size 10
  --divide 1 --step-size 0.0277777777777778 --leapfrog-steps 72 --seed 2026082105 --level-name L128toL256)
printf 'run_dir=%s\nsource=%s\ninitial_field_scale=%s = 2^(-eta/2), eta=0.25\nHMC=global full-volume, tau=2, n=72, eps=2/72\n' "$RUN" "$SOURCE" "$SCALE"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$SOURCE" ]] || { echo "missing L256 sweep-zero source: $SOURCE" >&2; exit 1; }
if [[ -e "$RUN" ]]; then echo "run already exists: $RUN" >&2; exit 1; fi
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null & echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
