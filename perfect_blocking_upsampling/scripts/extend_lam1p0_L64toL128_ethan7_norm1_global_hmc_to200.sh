#!/usr/bin/env bash
# Continue the rescaled L64->L128 normalization control from sweep 100 to 200.
# The input checkpoint is already rescaled/rethermalized: do not apply input-scale again.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
RUN="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L64toL128_ethan7_norm1_sameflow_N1500_S100_tau2_n51_eps2over51_r1"
SOURCE="$RUN/checkpoints/checkpoint_sweep_0100.npz"
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py"
  --run-dir "$RUN" --initialization input --input-source "$SOURCE" --input-scale 1.0 --append --sweep-offset 100
  --n-chains 1500 --n-sweeps 100 --save-every 5 --batch-size 25 --hmc-batch-size 25 --measurement-batch-size 25
  --divide 1 --step-size 0.0392156862745098 --leapfrog-steps 51 --seed 2026082107 --level-name L64toL128)
printf 'run_dir=%s\nsource=%s\ncontinuation=sweeps 101..200; input-scale=1 (the checkpoint is already scaled)\nHMC=global full-volume, tau=2, n=51, eps=2/51\n' "$RUN" "$SOURCE"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$SOURCE" ]] || { echo "missing sweep-100 checkpoint: $SOURCE" >&2; exit 1; }
if [[ -f "$RUN/continuations/continuation_0100_to_0200.json" ]]; then echo "continuation already exists" >&2; exit 1; fi
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" >"$RUN/logs/continuation_0100_to_0200.log" 2>&1 </dev/null & echo "$!" >"$RUN/continuation_0100_to_0200.pid"; echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
