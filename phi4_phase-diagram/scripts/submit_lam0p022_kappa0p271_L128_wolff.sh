#!/usr/bin/env bash
set -euo pipefail

# Manual launcher for a canonical lambda=0.022, kappa=0.271, L=128
# embedded-Wolff-sign-cluster plus radial-heatbath ensemble.
#
# Run from the project root:
#   bash phi4_phase-diagram/scripts/submit_lam0p022_kappa0p271_L128_wolff.sh
#
# Adjust N_CONFIGS / THERMAL_SWEEPS / SKIP_SWEEPS before submitting if the
# target machine budget differs.

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
N_CONFIGS="${N_CONFIGS:-500}"
THERMAL_SWEEPS="${THERMAL_SWEEPS:-3000}"
SKIP_SWEEPS="${SKIP_SWEEPS:-16}"
SEED="${SEED:-2026271128}"
OUT_DIR="${OUT_DIR:-phi4_phase-diagram/ensembles/lam0p022_kappa0p271_L128_embedded_wolff_sign_cluster_plus_radial_heatbath_N${N_CONFIGS}}"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" -B phi4_phase-diagram/src/generate_phi4_embedded_wolff_radial_heatbath.py \
  --lambda 0.022 \
  --kappa 0.271 \
  --L 128 \
  --n-configs "$N_CONFIGS" \
  --thermal-sweeps "$THERMAL_SWEEPS" \
  --skip-sweeps "$SKIP_SWEEPS" \
  --clusters-per-sweep 1 \
  --seed "$SEED" \
  --output-dir "$OUT_DIR"

"$PYTHON_BIN" -B phi4_phase-diagram/scripts/summarize_lam0p022_kappa0p271_wolff_ensembles.py \
  --ensemble-dir "$OUT_DIR"
