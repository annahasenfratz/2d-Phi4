#!/usr/bin/env bash
set -euo pipefail

# Runs the full bounded L32->L64 kappa_f matching diagnostic.
# This is not a broad native L64 scan; it uses transported/upscaled chains only.
#
# Expected cost is several hours if run sequentially on a laptop-class CPU.

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
OUT_DIR="${OUT_DIR:-perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/L32_to_L64_kappaf_matching/full_8x300}"

"$PYTHON_BIN" -B perfect_blocking_upsampling/scripts/run_L32_to_L64_kappaf_matching_experiment.py \
  --kappa-f 0.27050 0.27075 0.27100 0.27125 \
  --chains 8 \
  --sweeps 300 \
  --save-sweeps 0 10 25 50 100 200 300 \
  --out-dir "$OUT_DIR"
