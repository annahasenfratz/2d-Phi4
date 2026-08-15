#!/usr/bin/env bash
# Train a conditional L16->L32 upscaling flow for the soft-conditioned 7x7
# blocking kernel.  EO -> OE -> OO stages use pure conditional NLL only.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$ROOT/../../.venv/bin/python"
STAGE_DRIVER="$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_l16to32_highcorr5_pure_nll_stages.sh"
SOURCE_RUN="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_current5x5_tailstratified_proposal_coverage_N5000_20260803T160559Z"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$SOURCE_RUN/checkpoints/checkpoint_best_nll.pt}"
KERNEL="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R3_soft_conditioned_7x7_eta_included.json"
TOTAL_COUNT="${TOTAL_COUNT:-5000}"; TRAIN_COUNT="${TRAIN_COUNT:-4000}"; VAL_COUNT="${VAL_COUNT:-500}"; TEST_COUNT="${TEST_COUNT:-500}"
EPOCHS="${EPOCHS:-20}"; PATIENCE="${PATIENCE:-6}"; LR="${LR:-1e-5}"; BATCH_SIZE="${BATCH_SIZE:-128}"; SEED="${SEED:-2026080811}"; DEVICE="${DEVICE:-cpu}"
RUN_LABEL="lam1p0_L16to32_softcond7_pureNLL_N${TOTAL_COUNT}_$(date +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RESUME_RUN:-$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/$RUN_LABEL}"
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;; esac; done
[[ "$BACKGROUND" -eq 0 || "$EXECUTE" -eq 1 ]] || { echo "--background requires --execute" >&2; exit 2; }
for path in "$PYTHON" "$STAGE_DRIVER" "$SOURCE_CHECKPOINT" "$KERNEL"; do [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }; done
CMD=(env "RUN_ROOT=$RUN_DIR" "SOURCE_CHECKPOINT=$SOURCE_CHECKPOINT" "KERNEL=$KERNEL" "TOTAL_COUNT=$TOTAL_COUNT" "TRAIN_COUNT=$TRAIN_COUNT" "VAL_COUNT=$VAL_COUNT" "TEST_COUNT=$TEST_COUNT" "EPOCHS=$EPOCHS" "PATIENCE=$PATIENCE" "LR=$LR" "BATCH_SIZE=$BATCH_SIZE" "SEED=$SEED" "DEVICE=$DEVICE" bash "$STAGE_DRIVER")
printf 'run_dir=%s\n' "$RUN_DIR"; printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "$EXECUTE" -eq 0 ]]; then echo "Prepared only. Add --execute to start; add --background for nohup."; exit 0; fi
mkdir -p "$RUN_DIR/logs"
{
  echo "run_id=$(basename "$RUN_DIR")"; echo "schedule=EO -> OE -> OO; transfer initialization, then pure conditional NLL only"
  echo "kernel=allL16 soft-conditioned extended-local 7x7 candidate; not promoted to production"
  echo "normalization=calculated from this kernel's blocked training pairs in EO and reused by OE/OO"; echo "source_checkpoint=$SOURCE_CHECKPOINT"
  printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
} > "$RUN_DIR/submit_manifest.txt"
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" > "$RUN_DIR/logs/run.log" 2>&1 < /dev/null & echo "$!" > "$RUN_DIR/submit_pid.txt"; printf 'background_pid=%s\nlog=%s\n' "$!" "$RUN_DIR/logs/run.log"; else exec "${CMD[@]}"; fi
