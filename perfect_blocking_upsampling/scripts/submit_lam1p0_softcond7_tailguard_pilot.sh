#!/usr/bin/env bash
# One-epoch, conservative end-to-end tail/width correction of the softcond7
# NLL flow.  NLL remains active; all original checkpoints stay untouched.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PYTHON="$ROOT/../../.venv/bin/python"
DRIVER="$ROOT/perfect_blocking_upsampling/scripts/train_lam1p0_l16to32_rqspline_finetune.py"
BASE="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_softcond7_pureNLL_N5000_20260808T230212Z"
CHECKPOINT="$BASE/stage_oo/checkpoints/checkpoint_best_nll.pt"; NORMALIZATION="$BASE/stage_eo/normalization_metadata.json"
KERNEL="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R3_soft_conditioned_7x7_eta_included.json"
EPOCHS="${EPOCHS:-1}"; RUN="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_softcond7_tailguardPilot_N5000_$(date +%Y%m%dT%H%M%SZ)"
EXECUTE=0; BACKGROUND=0
for a in "$@"; do case "$a" in --execute) EXECUTE=1;; --background) BACKGROUND=1;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2;; esac; done
[[ "$BACKGROUND" -eq 0 || "$EXECUTE" -eq 1 ]] || { echo "--background requires --execute" >&2; exit 2; }
for p in "$PYTHON" "$DRIVER" "$CHECKPOINT" "$NORMALIZATION" "$KERNEL"; do [[ -e "$p" ]] || { echo "missing: $p" >&2; exit 1; }; done
CMD=("$PYTHON" -B "$DRIVER" --run-dir "$RUN" --source-checkpoint "$CHECKPOINT" --kernel-path "$KERNEL" --normalization-metadata "$NORMALIZATION" --source-start-index 0 --total-count 5000 --train-count 4000 --val-count 500 --test-count 500 --epochs "$EPOCHS" --patience 1 --eval-every 1 --exact-eval-every 1 --raw-eval-count 500 --global-chains 4 --global-sweeps 5 --local-chains 4 --local-sweeps 2 --batch-size 128 --lr 2e-6 --random-seed 2026080914 --device cpu --obs-weights action_density=0.020,phi2=0.010,phi4=0.025,local_kurtosis_ratio=0.035,NN=0,2nn=0,diag=0,G_pmin_avg=0 --two-sided-tail-guard --action-support-weight 0.020 --phi4-support-weight 0.025 --tail-guard-std-weight 0.05 --tail-guard-quantile-weight 0.25 --tail-guard-occupancy-weight 0.5 --tail-guard-low-occupancy-weight 0.75 --tail-guard-high-occupancy-weight 0.5 --action-std-match-weight 0.040 --phi4-std-match-weight 0.080 --local-kurtosis-shape-guard --local-kurtosis-shape-weight 0.015 --stop-after-eval-epoch "$EPOCHS")
printf 'run_dir=%s\n' "$RUN"; printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
[[ "$EXECUTE" -eq 1 ]] || { echo "Prepared only. Add --execute to start."; exit 0; }
mkdir -p "$RUN/logs"; { echo "base_checkpoint=$CHECKPOINT"; echo "objective=NLL + conservative action/phi4/kurtosis tail and width guards"; printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'; } > "$RUN/submit_manifest.txt"
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" > "$RUN/logs/run.log" 2>&1 < /dev/null & echo "$!" > "$RUN/submit_pid.txt"; echo "background_pid=$!"; else exec "${CMD[@]}"; fi
