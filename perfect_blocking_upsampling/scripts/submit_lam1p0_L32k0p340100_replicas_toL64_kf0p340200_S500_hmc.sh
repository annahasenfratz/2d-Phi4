#!/usr/bin/env bash
# Combine the two independent L32(kappa_c=0.340100) Ethan-7x7 sweep-zero
# L64 replicas (1500+1500), then run 500 full-volume HMC sweeps at kappa_f=.340200.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/../.venv/bin/python"
[[ -x "$PY" ]] || { echo "shared Python interpreter not found; set PYTHON=/path/to/python" >&2; exit 1; }
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;; esac; done
SRC1="$ROOT/perfect_blocking_upsampling/outputs/flow_only_lam1p0/L32toL64/L32k0p340100_toL64_ethan7_N1500_sweep0_r1/levels/L32toL64/checkpoints/checkpoint_sweep_0000.npz"
SRC2="$ROOT/perfect_blocking_upsampling/outputs/flow_only_lam1p0/L32toL64/L32k0p340100_r2_toL64_ethan7_N1500_sweep0_r1/levels/L32toL64/checkpoints/checkpoint_sweep_0000.npz"
INPUT="$ROOT/perfect_blocking_upsampling/outputs/combined_initializations_lam1p0/L32k0p340100_replicas_N3000_toL64_ethan7_sweep0.npz"
RUN="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L32k0p340100_replicas_N3000_toL64_ethan7_sweep0_kf0p340200_S500_tau2_n36_eps2over36_r1"
MERGE_CMD=("$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/merge_phi_checkpoints.py" --source "$SRC1" --source "$SRC2" --output "$INPUT")
HMC_CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py" --run-dir "$RUN" --initialization input --input-source "$INPUT" --n-chains 3000 --n-sweeps 500 --save-every 5 --batch-size 50 --hmc-batch-size 50 --measurement-batch-size 50 --divide 1 --step-size 0.05555555555555555 --leapfrog-steps 36 --seed 2026082533 --kappa 0.340200 --kappa-coarse 0.340100 --level-name L32toL64)
printf 'sources:\n  %s\n  %s\nmerged sweep-zero input=%s\nHMC output=%s\n' "$SRC1" "$SRC2" "$INPUT" "$RUN"
printf 'target: kappa_c=0.340100 -> kappa_f=0.340200; N=3000, sweeps=500, global tau=2, n=36, eps=2/36\n'
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$SRC1" && -f "$SRC2" ]] || { echo 'one or both sweep-zero replica checkpoints are missing' >&2; exit 1; }
# A previous failed submission left only RUN/logs and submit_pid.txt.  That is
# not an HMC result and is safe to reuse; never overwrite a started run.
[[ ! -e "$RUN/run_config.json" && ! -e "$RUN/status.json" ]] || { echo 'refusing to overwrite a started HMC output' >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ ! -e "$INPUT" ]]; then
  "${MERGE_CMD[@]}"
fi
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${HMC_CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null &
  echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"
else
  "${HMC_CMD[@]}" 2>&1 | tee "$RUN/logs/run.log"
fi
