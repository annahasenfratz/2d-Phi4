#!/usr/bin/env bash
# Held-out N=5000 global-A/R baseline for the matched high-correlation 5x5 pair.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHECKPOINT="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_highcorr5_pureNLL_N5000_20260807T063341Z/stage_oo/checkpoints/checkpoint_best_nll.pt"
KERNEL="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R2_corrW5000_highcorr_5x5_eta_included.json"
OUT="${OUT:-$ROOT/perfect_blocking_upsampling/outputs/global_ar_lam1p0/L16to32_highcorr5_baseline_N5000}"
N="${N:-5000}"; EXEC=0; BACKGROUND=0
for argument in "$@";do case "$argument" in --execute)EXEC=1;;--background)BACKGROUND=1;;*)echo "Usage: $0 [--execute] [--background]" >&2;exit 2;;esac;done
printf 'checkpoint=%s\nkernel=%s\nout=%s\nn=%s\n' "$CHECKPOINT" "$KERNEL" "$OUT" "$N"; [[ $EXEC -eq 1 ]] || exit 0
ARGS=(--execute)
[[ $BACKGROUND -eq 1 ]] && ARGS+=(--background)
CHECKPOINT="$CHECKPOINT" KERNEL="$KERNEL" OUT="$OUT" N="$N" bash "$ROOT/perfect_blocking_upsampling/scripts/submit_lam1p0_softcond7_global_ar_baseline.sh" "${ARGS[@]}"
