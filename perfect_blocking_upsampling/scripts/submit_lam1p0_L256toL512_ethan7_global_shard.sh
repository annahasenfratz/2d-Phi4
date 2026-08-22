#!/usr/bin/env bash
# Ethan 7x7: one memory-bounded L256->L512 shard (250 configurations).
# Usage: bash $0 --shard 0..5 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
CONFIG="$ROOT/perfect_blocking_upsampling/run_configs/lam1p0_hmc_L256toL512_ethan7_global_tau2_n100_sharded.json"
INPUT="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L128toL256/L128toL256_ethan7_N1500_S100_global_tau2_n72_eps2over72_streamed_r1/final_phi.npz"
SHARD=""; EXECUTE=0; BACKGROUND=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --shard) SHARD="${2:-}"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --background) BACKGROUND=1; shift ;;
    *) echo "usage: $0 --shard 0..5 [--execute] [--background]" >&2; exit 2 ;;
  esac
done
[[ "$SHARD" =~ ^[0-5]$ ]] || { echo "--shard must be an integer from 0 through 5" >&2; exit 2; }
COUNT=250; START=$((SHARD * COUNT))
printf -v SHARD_TAG '%02d' "$SHARD"
RUN="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L256toL512/L256toL512_ethan7_N250_shard${SHARD_TAG}_S100_global_tau2_n100_eps0p02_streamed_r1"
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_hmc_upscale_chain.py" --config "$CONFIG" --run-dir "$RUN" --n-chains "$COUNT" --start-index "$START")
printf 'run_dir=%s\nsource_indices=%d..%d\nflow/HMC batch=1; HMC=global full volume, tau=2, n=100, eps=.02\ncheckpoints=0,25,50,75,100\n' "$RUN" "$START" "$((START + COUNT - 1))"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$INPUT" ]] || { echo "L128->L256 stage has not completed: $INPUT" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "run already exists: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null & echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"; else exec "${CMD[@]}"; fi
