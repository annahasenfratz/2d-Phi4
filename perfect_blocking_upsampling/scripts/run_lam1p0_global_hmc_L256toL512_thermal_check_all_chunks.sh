#!/usr/bin/env bash
# Run all RAM-bounded L512 thermalization-check chunks sequentially.
# Completed chunks are retained and skipped on a later invocation.
# Usage: bash $0 [--execute]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXECUTE=0
for arg in "$@"; do
  case "$arg" in --execute) EXECUTE=1 ;; *) echo "usage: $0 [--execute]" >&2; exit 2 ;; esac
done

TOTAL_CHAINS="${TOTAL_CHAINS:-1500}"; CHUNK_SIZE="${CHUNK_SIZE:-25}"
N_CHUNKS=$(( (TOTAL_CHAINS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
echo "L256->L512 thermalization check: $N_CHUNKS sequential chunks of at most $CHUNK_SIZE fields; only one chunk is resident in RAM."
[[ "$EXECUTE" -eq 1 ]] || { echo "Add --execute to run."; exit 0; }

for ((chunk = 0; chunk < N_CHUNKS; chunk++)); do
  start=$(( chunk * CHUNK_SIZE ))
  n=$(( TOTAL_CHAINS - start )); (( n > CHUNK_SIZE )) && n="$CHUNK_SIZE"
  run="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L256toL512_thermal_check_chunks/L256toL512_N${n}_start${start}_S100_tau2_n100_eps2over100"
  if [[ -f "$run/final_phi.npz" ]]; then
    echo "chunk $chunk already complete; skipping"
    continue
  fi
  [[ ! -e "$run" ]] || { echo "chunk $chunk has an incomplete output directory: $run" >&2; exit 1; }
  bash "$ROOT/perfect_blocking_upsampling/scripts/submit_lam1p0_global_hmc_L256toL512_thermal_check_chunk.sh" "$chunk" --execute
done
