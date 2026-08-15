#!/usr/bin/env bash
# Matched L64 -> L32 fine-tune for the direct L32 -> L64 proposal.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -x "$REPO_ROOT/../../.venv/bin/python" ]]; then PYTHON="$REPO_ROOT/../../.venv/bin/python"; elif [[ -x "$REPO_ROOT/../.venv/bin/python" ]]; then PYTHON="$REPO_ROOT/../.venv/bin/python"; else echo 'shared Python not found' >&2; exit 1; fi
DRIVER="$REPO_ROOT/perfect_blocking_upsampling/scripts/train_lam1p0_l16to32_rqspline_finetune.py"
SOURCE="$REPO_ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_current5x5_tailstratified_proposal_coverage_N5000_20260803T160559Z/checkpoints/checkpoint_epoch002.pt"
KERNEL="$REPO_ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
FINE="$REPO_ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz"
STAMP="$(date +%Y%m%dT%H%M%SZ)"
LABEL="lam1p0_L32toL64_current5x5_tailshape_finetune_N5000_$STAMP"
RUN_DIR="$REPO_ROOT/perfect_blocking_upsampling/runs/lam1p0/training/$LABEL"
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;; esac; done
for path in "$PYTHON" "$DRIVER" "$SOURCE" "$KERNEL" "$FINE"; do [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }; done
CMD=("$PYTHON" -B "$DRIVER" --run-dir "$RUN_DIR" --source-checkpoint "$SOURCE" --kernel-path "$KERNEL" --fine-config-source "$FINE" --coarse-lattice 32 --source-start-index 0 --total-count 5000 --train-count 4000 --val-count 500 --test-count 500 --epochs 5 --patience 2 --eval-every 1 --exact-eval-every 2 --raw-eval-count 500 --global-chains 4 --global-sweeps 4 --local-chains 4 --local-sweeps 2 --local-detail-patch-size 32 --batch-size 64 --lr 2e-6 --random-seed 2026080419 --device cpu --obs-weights action_density=0.025,phi2=0.020,phi4=0.030,local_kurtosis_ratio=0.030,NN=0.012,2nn=0.004,diag=0.004,G_pmin_avg=0 --tail-stratified-train --tail-stratified-quantile 0.10 --tail-stratified-tail-fraction 0.40 --proposal-action-lowtail-weight 0.15 --proposal-kurtosis-lowtail-weight 0.25 --local-kurtosis-shape-guard --local-kurtosis-shape-weight 0.03)
echo "RUN_DIR=$RUN_DIR"; printf 'command: '; printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "$EXECUTE" -eq 0 ]]; then echo 'Prepared only. Re-run with --execute to start; add --background for nohup.'; exit 0; fi
mkdir -p "$RUN_DIR/logs"; { echo "run_id=$LABEL"; echo 'training=matched L64 pairs for direct L32->L64 use; losses=tail-stratified action/kurtosis low coverage plus mild kurtosis shape'; printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'; } > "$RUN_DIR/submit_manifest.txt"
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" > "$RUN_DIR/logs/run.log" 2>&1 & echo "$!" > "$RUN_DIR/submit_pid.txt"; echo "started background PID $!"; else exec "${CMD[@]}"; fi
