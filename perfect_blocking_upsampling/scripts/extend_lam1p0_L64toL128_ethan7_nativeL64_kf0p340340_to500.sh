#!/usr/bin/env bash
# Continue the clean native-L64 -> L128 kappa_f=0.340340 HMC chain from
# absolute sweep 100 through absolute sweep 500.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/../.venv/bin/python"
[[ -x "$PY" ]] || { echo "shared Python interpreter not found; set PYTHON=/path/to/python" >&2; exit 1; }
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;; esac; done
RUN="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L64toL128_ethan7_nativeL64_sweep0_kf0p340340_N1500_S100_tau2_n51_eps2over51_r1"
INPUT="$RUN/checkpoints/checkpoint_sweep_0100.npz"
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py"
  --run-dir "$RUN" --initialization input --input-source "$INPUT"
  --n-chains 1500 --n-sweeps 400 --save-every 5
  --batch-size 25 --hmc-batch-size 25 --measurement-batch-size 25
  --divide 1 --step-size 0.0392156862745098 --leapfrog-steps 51
  --seed 2026082520 --kappa 0.340340 --kappa-coarse 0.340301
  --sweep-offset 100 --append --level-name L64toL128)
printf 'continuation: %s\ninput=%s\nabsolute sweeps: 100 -> 500; kappa_f=0.340340; global tau=2, n=51, eps=2/51\n' "$RUN" "$INPUT"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$INPUT" ]] || { echo "missing sweep-100 checkpoint: $INPUT" >&2; exit 1; }
[[ ! -e "$RUN/submit_pid_extend_to500.txt" ]] || { echo "continuation submission marker exists: $RUN/submit_pid_extend_to500.txt" >&2; exit 1; }
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" >"$RUN/logs/extend_to500.log" 2>&1 </dev/null &
  echo "$!" >"$RUN/submit_pid_extend_to500.txt"; echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
