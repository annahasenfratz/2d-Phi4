#!/usr/bin/env bash
# MH test of the high-correlation 7x7, pure-NLL L16->L32 flow.
# Usage: bash $0 rNUMBER [--execute] [--background]
# Override N_CHAINS, START_INDEX, N_SWEEPS, etc. as environment variables.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE="$ROOT/perfect_blocking_upsampling/scripts/submit_lam1p0_checkerboard_mh.sh"

# Keep the established checkerboard-MH update parameters; only the
# initialization flow and its corresponding 7x7 blocking kernel differ.
export LC="${LC:-16}"
export LF="${LF:-32}"
export N_CHAINS="${N_CHAINS:-500}"
export N_SWEEPS="${N_SWEEPS:-400}"
export START_INDEX="${START_INDEX:-500}"
export BATCH_SIZE="${BATCH_SIZE:-50}"
export SEED="${SEED:-2026080702}"
export DIVIDE="${DIVIDE:-2}"
export DETAIL_PASSES="${DETAIL_PASSES:-2}"
export COARSE_SIGMA="${COARSE_SIGMA:-0.40}"
export DETAIL_SIGMA="${DETAIL_SIGMA:-0.10}"
export COARSE_UPDATES="${COARSE_UPDATES:-1}"
export SAVE_EVERY="${SAVE_EVERY:-5}"
export INITIALIZATION="${INITIALIZATION:-direct_coarse_flow}"
export FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_highcorr7_pureNLL_N5000_20260806T205454Z/stage_oo/checkpoints/checkpoint_best_nll.pt}"
export KERNEL_PATH="${KERNEL_PATH:-$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R3_corrW5000_highcorr_7x7_eta_included.json}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/perfect_blocking_upsampling/outputs/checkerboard_mh_lam1p0_highcorr7_nll}"

exec bash "$BASE" "$@"
