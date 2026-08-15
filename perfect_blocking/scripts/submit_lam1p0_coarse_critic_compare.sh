#!/usr/bin/env bash
# Held-out direct-L16 versus blocked-L32 coarse-field critics for the active
# 5x5 and soft-conditioned 7x7 candidates. Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PYTHON="$ROOT/../../.venv/bin/python"
DRIVER="$ROOT/perfect_blocking/scripts/train_lam1p0_coarse_marginal_critic.py"
OUTROOT="$ROOT/perfect_blocking/perfect_blocking_lam1p0/tests/coarse_marginal_critics_20260809"
K5="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R2_corrW5000_highcorr_5x5_eta_included.json"
K7="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R3_soft_conditioned_7x7_eta_included.json"
EPOCHS="${EPOCHS:-30}"; EXECUTE=0; BACKGROUND=0
for a in "$@"; do case "$a" in --execute) EXECUTE=1;; --background) BACKGROUND=1;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2;; esac; done
[[ "$BACKGROUND" -eq 0 || "$EXECUTE" -eq 1 ]] || { echo "--background requires --execute" >&2; exit 2; }
CMD=(bash -c '"$1" -B "$2" --kernel "$3" --out "$4" --epochs "$5" && "$1" -B "$2" --kernel "$6" --out "$7" --epochs "$5"' _ "$PYTHON" "$DRIVER" "$K5" "$OUTROOT/highcorr5" "$EPOCHS" "$K7" "$OUTROOT/softcond7")
printf 'outroot=%s\n' "$OUTROOT"; printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
[[ "$EXECUTE" -eq 1 ]] || { echo "Prepared only. Add --execute to start."; exit 0; }
mkdir -p "$OUTROOT/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" > "$OUTROOT/logs/run.log" 2>&1 < /dev/null & echo "$!" > "$OUTROOT/submit_pid.txt"; echo "background_pid=$!"; else exec "${CMD[@]}"; fi
