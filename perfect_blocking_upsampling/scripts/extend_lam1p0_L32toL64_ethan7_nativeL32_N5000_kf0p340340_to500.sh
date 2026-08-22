#!/usr/bin/env bash
# Continue the 5000-root clean L32->L64 kappa_f=0.340340 HMC chain from
# absolute sweep 50 through absolute sweep 500.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/../.venv/bin/python"
[[ -x "$PY" ]] || { echo "shared Python interpreter not found; set PYTHON=/path/to/python" >&2; exit 1; }
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;; esac; done
RUN="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L32toL64_ethan7_nativeL32_sweep0_kf0p340340_N5000_S50_tau2_n36_eps2over36_r1"
INPUT="$RUN/checkpoints/checkpoint_sweep_0050.npz"
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py"
  --run-dir "$RUN" --initialization input --input-source "$INPUT"
  --n-chains 5000 --n-sweeps 450 --save-every 5
  --batch-size 50 --hmc-batch-size 50 --measurement-batch-size 50
  --divide 1 --step-size 0.05555555555555555 --leapfrog-steps 36
  --seed 2026082522 --kappa 0.340340 --kappa-coarse 0.340301
  --sweep-offset 50 --append --level-name L32toL64)
printf 'continuation: %s\ninput=%s\nabsolute sweeps: 50 -> 500; kappa_f=0.340340; global tau=2, n=36, eps=2/36\n' "$RUN" "$INPUT"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$INPUT" ]] || { echo "missing sweep-50 checkpoint: $INPUT" >&2; exit 1; }
[[ ! -e "$RUN/submit_pid_extend_to500.txt" ]] || { echo "continuation submission marker exists: $RUN/submit_pid_extend_to500.txt" >&2; exit 1; }
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" >"$RUN/logs/extend_to500.log" 2>&1 </dev/null &
  echo "$!" >"$RUN/submit_pid_extend_to500.txt"; echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
