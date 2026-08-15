#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$ROOT/../../.venv/bin/python"
STAMP="$(date +%Y%m%dT%H%M%SZ)"
RUN="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L32toL64_stage_eo_${STAMP}"
CMD=("$PYTHON" -B "$ROOT/perfect_blocking_upsampling/scripts/train_lam1p0_l16to32_rqspline_finetune.py" --run-dir "$RUN" --source-checkpoint "$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_current5x5_tailstratified_proposal_coverage_N5000_20260803T160559Z/checkpoints/checkpoint_epoch002.pt" --fine-config-source "$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz" --kernel-path "$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json" --coarse-lattice 32 --total-count 5000 --train-count 4000 --val-count 500 --test-count 500 --epochs 5 --patience 2 --batch-size 64 --lr 2e-6 --eval-every 1 --raw-eval-count 500 --train-stage eo --device cpu)
EXECUTE=0
BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done
if [[ "$BACKGROUND" -eq 1 && "$EXECUTE" -eq 0 ]]; then echo "--background requires --execute" >&2; exit 2; fi
if [[ "$EXECUTE" -eq 0 ]]; then printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
mkdir -p "$RUN/logs"
printf 'run_dir=%s\n' "$RUN"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" > "$RUN/logs/run.log" 2>&1 < /dev/null &
  echo "$!" > "$RUN/submit_pid.txt"
  printf 'background_pid=%s\nlog=%s\n' "$!" "$RUN/logs/run.log"
else
  exec "${CMD[@]}"
fi
