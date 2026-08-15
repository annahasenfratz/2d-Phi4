#!/usr/bin/env bash
set -euo pipefail
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${PKG_DIR}/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PKG_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PROJECT_ROOT}/../.venv/bin/python"
"$PYTHON" "$PKG_DIR/scripts/00_check_environment.py"
"$PYTHON" "$PKG_DIR/scripts/01_preflight.py" --config "$PKG_DIR/configs/default_lam0p022_k02705.yaml"
"$PYTHON" "$PKG_DIR/scripts/03_reproduce_from_frozen.py" --config "$PKG_DIR/configs/default_lam0p022_k02705.yaml"
"$PYTHON" "$PKG_DIR/scripts/04_proposal_only.py" --config "$PKG_DIR/configs/default_lam0p022_k02705.yaml"
"$PYTHON" "$PKG_DIR/scripts/05_independent_ar.py" --config "$PKG_DIR/configs/default_lam0p022_k02705.yaml"
"$PYTHON" "$PKG_DIR/scripts/06_make_report.py" --config "$PKG_DIR/configs/default_lam0p022_k02705.yaml"
