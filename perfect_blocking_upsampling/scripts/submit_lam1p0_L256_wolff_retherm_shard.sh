#!/usr/bin/env bash
# Restartable standard embedded-Wolff + radial-heat-bath test from L256 sweep 0.
# Usage: bash $0 --shard 0..5 [--target-sweeps N] [--resume] [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
SOURCE="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L128toL256/L128toL256_ethan7_N1500_S100_global_tau2_n72_eps2over72_streamed_r1/checkpoints/checkpoint_sweep_0000.npz"
SHARD=""; TARGET=100; RESUME=0; EXECUTE=0; BACKGROUND=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --shard) SHARD="${2:-}"; shift 2 ;;
    --target-sweeps) TARGET="${2:-}"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --execute) EXECUTE=1; shift ;;
    --background) BACKGROUND=1; shift ;;
    *) echo "usage: $0 --shard 0..5 [--target-sweeps N] [--resume] [--execute] [--background]" >&2; exit 2 ;;
  esac
done
[[ "$SHARD" =~ ^[0-5]$ ]] || { echo "--shard must be an integer from 0 through 5" >&2; exit 2; }
[[ "$TARGET" =~ ^[0-9]+$ ]] || { echo "--target-sweeps must be non-negative" >&2; exit 2; }
COUNT=250; START=$((SHARD * COUNT)); printf -v TAG '%02d' "$SHARD"
RUN="$ROOT/perfect_blocking_upsampling/outputs/wolff_rethermalization_lam1p0/L256_ethan7_sweep0_N250_shard${TAG}_radial_wolff_fixed4_r2"
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_l256_wolff_rethermalization.py" --source "$SOURCE" --run-dir "$RUN" --L 256 --start-index "$START" --n-chains "$COUNT" --target-sweeps "$TARGET" --checkpoint-every 25 --clusters-per-sweep 4)
[[ "$RESUME" -eq 1 ]] && CMD+=(--resume)
printf 'run_dir=%s\nsource_indices=%d..%d\nalgorithm=one radial heat-bath sweep plus Wolff clusters accumulated to one lattice-volume equivalent per configuration\nrestart: --resume --target-sweeps NEW_TOTAL\n' "$RUN" "$START" "$((START + COUNT - 1))"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$SOURCE" ]] || { echo "missing sweep-0 L256 source: $SOURCE" >&2; exit 1; }
if [[ "$RESUME" -eq 0 && -e "$RUN" ]]; then echo "run already exists: $RUN" >&2; exit 1; fi
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null & echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"; else exec "${CMD[@]}"; fi
