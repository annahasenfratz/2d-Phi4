#!/usr/bin/env bash
# Same flow/checkpoint/reference, two coarse inputs: blocked-native and direct L16.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PYTHON="$ROOT/../../.venv/bin/python"
DRIVER="$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_nativeblocked_zero_sweep_observables.py"
FLOW="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_softcond7_pureNLL_N5000_20260808T230212Z/stage_oo/checkpoints/checkpoint_best_nll.pt"
KERNEL="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R3_soft_conditioned_7x7_eta_included.json"
OUT="$ROOT/perfect_blocking_upsampling/outputs/flow_input_audit_lam1p0/softcond7_N1500_20260809"
N_CHAINS="${N_CHAINS:-1500}"; BATCH_SIZE="${BATCH_SIZE:-64}"; SEED="${SEED:-2026080913}"; EXECUTE=0; BACKGROUND=0
for a in "$@"; do case "$a" in --execute) EXECUTE=1;; --background) BACKGROUND=1;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2;; esac; done
[[ "$BACKGROUND" -eq 0 || "$EXECUTE" -eq 1 ]] || { echo "--background requires --execute" >&2; exit 2; }
for p in "$PYTHON" "$DRIVER" "$FLOW" "$KERNEL"; do [[ -e "$p" ]] || { echo "missing: $p" >&2; exit 1; }; done
CMD=(bash -c '"$1" -B "$2" --coarse-mode blocked_native --checkpoint "$3" --kernel "$4" --out-dir "$5/blocked_native" --n-chains "$6" --batch-size "$7" --seed "$8" && "$1" -B "$2" --coarse-mode direct_native --checkpoint "$3" --kernel "$4" --out-dir "$5/direct_l16" --n-chains "$6" --batch-size "$7" --seed "$8"' _ "$PYTHON" "$DRIVER" "$FLOW" "$KERNEL" "$OUT" "$N_CHAINS" "$BATCH_SIZE" "$SEED")
printf 'out_dir=%s\n' "$OUT"; printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
[[ "$EXECUTE" -eq 1 ]] || { echo "Prepared only. Add --execute to start."; exit 0; }
[[ ! -e "$OUT" ]] || { echo "output already exists: $OUT" >&2; exit 1; }; mkdir -p "$OUT/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" > "$OUT/logs/run.log" 2>&1 < /dev/null & echo "$!" > "$OUT/submit_pid.txt"; echo "background_pid=$!"; else exec "${CMD[@]}"; fi
