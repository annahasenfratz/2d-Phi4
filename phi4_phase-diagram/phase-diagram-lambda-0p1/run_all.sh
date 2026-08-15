#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$(pwd)/work/mplconfig}"
export MPLCONFIGDIR

mkdir -p outputs work/mplconfig

"$PYTHON" scripts/phi4_lambda01_cluster_scan.py \
  --lambda 0.1 \
  --Ls 16,24,32 \
  --centers 0.28,0.30,0.32 \
  --samples 8192 \
  --thermal-sweeps 3000 \
  --skip-sweeps 4 \
  --clusters-per-sweep 1 \
  --proposal-width 1.0 \
  --window 0.02 \
  --step 0.001 \
  --min-ess 0.30 \
  --output-csv outputs/phi4_lambda01_cluster_l16_l24_l32_curves.csv \
  --output-json outputs/phi4_lambda01_cluster_l16_l24_l32_summary.json

"$PYTHON" scripts/phi4_lambda01_cluster_scan.py \
  --lambda 0.1 \
  --Ls 16,24,32 \
  --centers 0.295,0.300,0.305 \
  --samples 8192 \
  --thermal-sweeps 3000 \
  --skip-sweeps 4 \
  --clusters-per-sweep 1 \
  --proposal-width 1.0 \
  --window 0.012 \
  --step 0.0005 \
  --min-ess 0.30 \
  --output-csv outputs/phi4_lambda01_cluster_l16_l24_l32_refined_curves.csv \
  --output-json outputs/phi4_lambda01_cluster_l16_l24_l32_refined_summary.json

"$PYTHON" scripts/plot_phi4_lambda01_phase.py \
  --broad outputs/phi4_lambda01_cluster_l16_l24_l32_curves.csv \
  --refined outputs/phi4_lambda01_cluster_l16_l24_l32_refined_curves.csv \
  --output-png outputs/phi4_lambda01_l16_l24_l32_chi_binder.png \
  --output-pdf outputs/phi4_lambda01_l16_l24_l32_chi_binder.pdf \
  --output-json outputs/phi4_lambda01_l16_l24_l32_chi_binder.json
