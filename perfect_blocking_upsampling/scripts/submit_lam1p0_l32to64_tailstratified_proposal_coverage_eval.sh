#!/usr/bin/env bash
# Cross-volume raw conditional-proposal diagnostic: direct L64 vs L32 -> L64.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -x "$REPO_ROOT/../../.venv/bin/python" ]]; then PYTHON="$REPO_ROOT/../../.venv/bin/python"; elif [[ -x "$REPO_ROOT/../.venv/bin/python" ]]; then PYTHON="$REPO_ROOT/../.venv/bin/python"; else echo 'shared Python not found' >&2; exit 1; fi
DRIVER="$REPO_ROOT/perfect_blocking_upsampling/scripts/compare_lam1p0_raw_upscaled_vs_native.py"
CKPT="$REPO_ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_current5x5_tailstratified_proposal_coverage_N5000_20260803T160559Z/checkpoints/checkpoint_epoch002.pt"
KERNEL="$REPO_ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
NATIVE="$REPO_ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz"
STAMP="$(date +%Y%m%dT%H%M%SZ)"
LABEL="raw_L32toL64_current5x5_tailstratified_proposal_coverage_epoch002_N5000_$STAMP"
RUN_DIR="$REPO_ROOT/perfect_blocking_upsampling/runs/lam1p0/upscaling/$LABEL"
FIELD="$REPO_ROOT/data/configs_phi4_2d/upscaled/lam1p0_kappac0p340301_L32_to_L64_current5x5_tailstratified_proposal_coverage_epoch002_N5000_$STAMP.npz"
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;; esac; done
for path in "$PYTHON" "$DRIVER" "$CKPT" "$KERNEL" "$NATIVE"; do [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }; done
COARSE="$REPO_ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
[[ -e "$COARSE" ]] || { echo "missing: $COARSE" >&2; exit 1; }
CMD=("$PYTHON" -B "$DRIVER" --checkpoint "$CKPT" --kernel "$KERNEL" --native-source "$NATIVE" --coarse-source "$COARSE" --coarse-lattice 32 --fine-lattice 64 --output-dir "$RUN_DIR" --count 5000 --seed 2026080417 --batch-size 64 --label tailstratified_proposal_coverage_epoch002 --upscaled-config-output "$FIELD")
echo "RUN_DIR=$RUN_DIR"; echo "FIELD=$FIELD"; printf 'command: '; printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "$EXECUTE" -eq 0 ]]; then echo 'Prepared only. Re-run with --execute to start; add --background for nohup.'; exit 0; fi
mkdir -p "$RUN_DIR/logs"; { echo "run_id=$LABEL"; echo 'comparison=direct L64 versus independent direct-L32 upscaled proposal; checkpoint=tailstratified epoch 2'; printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'; } > "$RUN_DIR/submit_manifest.txt"
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" > "$RUN_DIR/logs/run.log" 2>&1 & echo "$!" > "$RUN_DIR/submit_pid.txt"; echo "started background PID $!"; else exec "${CMD[@]}"; fi
