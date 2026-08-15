#!/usr/bin/env bash
# Stage 1: 7x7 kernel, extended operator *means* only.  No explicit
# covariance/correlation loss is included in this first comparison.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export KERNEL_RADIUS="${KERNEL_RADIUS:-3}"
export EXTRA_LOCAL2="${EXTRA_LOCAL2:-1}"
export FROZEN_BLOCK_COV="${FROZEN_BLOCK_COV:-1}"
export CORRELATION_WEIGHT="${CORRELATION_WEIGHT:-0}"
export POSITIVITY_ONLY="${POSITIVITY_ONLY:-0}"
export MIN_K_FLOOR="${MIN_K_FLOOR:-0.35}"
export SOFT_CONDITION_TARGET="${SOFT_CONDITION_TARGET:-2.3}"
export SOFT_CONDITION_WIDTH="${SOFT_CONDITION_WIDTH:-0.5}"
export SOFT_CONDITION_WEIGHT="${SOFT_CONDITION_WEIGHT:-5.0}"
export SOFT_INVERSE_TARGET="${SOFT_INVERSE_TARGET:-2.0}"
export SOFT_INVERSE_WIDTH="${SOFT_INVERSE_WIDTH:-0.5}"
export SOFT_INVERSE_WEIGHT="${SOFT_INVERSE_WEIGHT:-2.0}"
export INVERSE_CAP="${INVERSE_CAP:-2.5}"
export CONDITION_CAP="${CONDITION_CAP:-3.0}"
export MAXITER="${MAXITER:-300}"

exec bash "$ROOT/perfect_blocking/scripts/submit_lam1p0_strict_joint_kernel_search.sh" "$@"
