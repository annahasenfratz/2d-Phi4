#!/usr/bin/env bash
# Strict protocol launcher. Candidate fitting, validation selection, and the
# untouched test report are handled by the strict driver.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$ROOT/../../.venv/bin/python"
DRIVER="$ROOT/perfect_blocking/scripts/run_lam1p0_allL16_chi2_kernel_search.py"
RADIUS="${KERNEL_RADIUS:-2}"
CORR_WEIGHT="${CORRELATION_WEIGHT:-0}"
POSITIVITY_ONLY="${POSITIVITY_ONLY:-0}"
MAXITER="${MAXITER:-160}"
EXTRA_LOCAL2="${EXTRA_LOCAL2:-0}"
FROZEN_BLOCK_COV="${FROZEN_BLOCK_COV:-0}"
INVERSE_CAP="${INVERSE_CAP:-2.5}"
CONDITION_CAP="${CONDITION_CAP:-3.0}"
MIN_K_FLOOR="${MIN_K_FLOOR:-0.35}"
SOFT_CONDITION_TARGET="${SOFT_CONDITION_TARGET:-2.3}"
SOFT_CONDITION_WIDTH="${SOFT_CONDITION_WIDTH:-0.5}"
SOFT_CONDITION_WEIGHT="${SOFT_CONDITION_WEIGHT:-5.0}"
SOFT_INVERSE_TARGET="${SOFT_INVERSE_TARGET:-2.0}"
SOFT_INVERSE_WIDTH="${SOFT_INVERSE_WIDTH:-0.5}"
SOFT_INVERSE_WEIGHT="${SOFT_INVERSE_WEIGHT:-2.0}"
CAP_TAG="${INVERSE_CAP//./p}"
CONDITION_TAG="${CONDITION_CAP//./p}"
SOFT_CONDITION_TAG="${SOFT_CONDITION_TARGET//./p}"
SOFT_INVERSE_TAG="${SOFT_INVERSE_TARGET//./p}"
MODE_TAG="softCond${SOFT_CONDITION_TAG}_inv${SOFT_INVERSE_TAG}_finalCond${CONDITION_TAG}_inv${CAP_TAG}"; [[ "$POSITIVITY_ONLY" == "1" ]] && MODE_TAG="positiveOnly"
EXTRA_TAG=""; [[ "$EXTRA_LOCAL2" == "1" ]] && EXTRA_TAG="extraLocal2_"
FROZEN_TAG=""; [[ "$FROZEN_BLOCK_COV" == "1" ]] && FROZEN_TAG="frozenBlockCov_"
OUT="$ROOT/perfect_blocking/perfect_blocking_lam1p0/tests/intermediate/allL16_chi2_R${RADIUS}_corrW${CORR_WEIGHT}_${EXTRA_TAG}${FROZEN_TAG}${MODE_TAG}_train3000_val1000_test1000"
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1;; --background) BACKGROUND=1;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2;; esac; done
[[ "$BACKGROUND" -eq 0 || "$EXECUTE" -eq 1 ]] || { echo "--background requires --execute" >&2; exit 2; }
[[ -x "$PYTHON" && -f "$DRIVER" ]] || { echo "missing shared Python or strict driver" >&2; exit 1; }
printf 'output=%s\n' "$OUT"
if [[ "$EXECUTE" -eq 0 ]]; then echo "Prepared only. Add --execute to start."; exit 0; fi
mkdir -p "$OUT/logs"
CMD=(env "KERNEL_RADIUS=$RADIUS" "CORRELATION_WEIGHT=$CORR_WEIGHT" "POSITIVITY_ONLY=$POSITIVITY_ONLY" "MAXITER=$MAXITER" "EXTRA_LOCAL2=$EXTRA_LOCAL2" "FROZEN_BLOCK_COV=$FROZEN_BLOCK_COV" "MIN_K_FLOOR=$MIN_K_FLOOR" "SOFT_CONDITION_TARGET=$SOFT_CONDITION_TARGET" "SOFT_CONDITION_WIDTH=$SOFT_CONDITION_WIDTH" "SOFT_CONDITION_WEIGHT=$SOFT_CONDITION_WEIGHT" "SOFT_INVERSE_TARGET=$SOFT_INVERSE_TARGET" "SOFT_INVERSE_WIDTH=$SOFT_INVERSE_WIDTH" "SOFT_INVERSE_WEIGHT=$SOFT_INVERSE_WEIGHT" "INVERSE_CAP=$INVERSE_CAP" "CONDITION_CAP=$CONDITION_CAP" "$PYTHON" -B "$DRIVER")
printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" > "$OUT/logs/run.log" 2>&1 < /dev/null & echo "$!" > "$OUT/submit_pid.txt"; printf 'background_pid=%s\n' "$!"; else exec "${CMD[@]}"; fi
