#!/usr/bin/env bash
# Independent L32 -> L64 conditional-flow baseline.  This deliberately does
# not inherit parameters or normalization from the L16 -> L32 flow.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -x "$ROOT/../../.venv/bin/python" ]]; then
  PYTHON="$ROOT/../../.venv/bin/python"
elif [[ -x "$ROOT/../.venv/bin/python" ]]; then
  PYTHON="$ROOT/../.venv/bin/python"
else
  echo "shared Python not found" >&2
  exit 1
fi

DRIVER="$ROOT/perfect_blocking_upsampling/scripts/train_lam1p0_l16to32_rqspline_finetune.py"
KERNEL="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
FINE="$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz"
STAMP="$(date +%Y%m%dT%H%M%SZ)"
LABEL="lam1p0_L32toL64_fresh_joint_N5000_${STAMP}"
RUN="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/$LABEL"

# This is a full, joint fit: eo, oe, and oo are all trainable.  Epoch-zero is
# retained as a diagnostic only; it is not a transferred proposal.
CMD=(
  "$PYTHON" -B "$DRIVER"
  --run-dir "$RUN"
  --initialization-mode fresh
  --fine-config-source "$FINE"
  --kernel-path "$KERNEL"
  --coarse-lattice 32
  --total-count 5000 --train-count 4000 --val-count 500 --test-count 500
  --layers 8 --hidden-channels 48 --conv-kernel-size 3
  --epochs 40 --patience 10 --batch-size 64 --lr 5e-5
  --eval-every 1 --exact-eval-every 5 --raw-eval-count 500
  --global-chains 4 --global-sweeps 5 --local-chains 4 --local-sweeps 2
  --local-detail-patch-size 32 --local-detail-passes 10
  --random-seed 2026080421 --device cpu
  --obs-weights action_density=0.025,phi2=0.020,phi4=0.030,local_kurtosis_ratio=0.030,NN=0.012,2nn=0.004,diag=0.004,G_pmin_avg=0
  --tail-stratified-train --tail-stratified-quantile 0.10 --tail-stratified-tail-fraction 0.40
  --proposal-action-lowtail-weight 0.15 --proposal-kurtosis-lowtail-weight 0.25
  --local-kurtosis-shape-guard --local-kurtosis-shape-weight 0.03
)

EXECUTE=0
BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done
if [[ "$BACKGROUND" -eq 1 && "$EXECUTE" -eq 0 ]]; then
  echo "--background requires --execute" >&2
  exit 2
fi
for path in "$PYTHON" "$DRIVER" "$KERNEL" "$FINE"; do
  [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }
done

printf 'run_dir=%s\n' "$RUN"
printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "$EXECUTE" -eq 0 ]]; then
  echo "Prepared only. Add --execute to start; add --background for nohup."
  exit 0
fi

mkdir -p "$RUN/logs"
{
  echo "run_id=$LABEL"
  echo "initialization=fresh; train_stage=all; selection=held-out generated-observable diagnostics"
  printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
} > "$RUN/submit_manifest.txt"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" > "$RUN/logs/run.log" 2>&1 < /dev/null &
  echo "$!" > "$RUN/submit_pid.txt"
  printf 'background_pid=%s\nlog=%s\n' "$!" "$RUN/logs/run.log"
else
  exec "${CMD[@]}"
fi
