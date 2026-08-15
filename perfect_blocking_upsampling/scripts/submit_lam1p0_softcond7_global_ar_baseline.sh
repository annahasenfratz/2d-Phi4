#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/../../.venv/bin/python"

# Override these from the command line when auditing another kernel/flow pair.
CK="${CHECKPOINT:-$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_softcond7_pureNLL_N5000_20260808T230212Z/stage_oo/checkpoints/checkpoint_best_nll.pt}"
K="${KERNEL:-$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R3_soft_conditioned_7x7_eta_included.json}"
OUT="${OUT:-$ROOT/perfect_blocking_upsampling/outputs/global_ar_lam1p0/softcond7_N5000_baseline}"
N="${N:-1000}"

EXEC=0
BG=0
for x in "$@"; do
  case "$x" in
    --execute) EXEC=1 ;;
    --background) BG=1 ;;
    *) echo "Usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done

CMD=("$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/audit_lam1p0_softcond7_global_ar.py"
  --checkpoint "$CK" --kernel "$K" --out "$OUT" --n "$N")

printf 'checkpoint=%s\nkernel=%s\nout=%s\nn=%s\n' "$CK" "$K" "$OUT" "$N"
[[ $EXEC -eq 1 ]] || exit 0
mkdir -p "$OUT/logs"
if [[ $BG -eq 1 ]]; then
  nohup "${CMD[@]}" >"$OUT/logs/run.log" 2>&1 < /dev/null &
  echo $! >"$OUT/submit_pid.txt"
  echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
