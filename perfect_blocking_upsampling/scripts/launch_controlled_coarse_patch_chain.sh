#!/usr/bin/env bash
set -euo pipefail

# Launch the controlled coarse patch-chain driver with defaults for the frozen
# same-kappa 32->64 validation. Override any value by passing command-line args.
#
# Usage:
#   bash perfect_blocking_upsampling/scripts/launch_controlled_coarse_patch_chain.sh
#   bash perfect_blocking_upsampling/scripts/launch_controlled_coarse_patch_chain.sh \
#     64 128 0.022 0.2705 0.2705 12 2 12 3 8 1000
#
# Positional arguments:
#   1  coarse_L
#   2  fine_L
#   3  lambda
#   4  kappa_c
#   5  kappa_f
#   6  coarse_patch_size
#   7  coarse_passes
#   8  detail_patch_size
#   9  detail_updates_per_sweep
#   10 chains
#   11 sweeps

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT_DIR}/../.venv/bin/python"
SCRIPT="${ROOT_DIR}/perfect_blocking_upsampling/scripts/run_controlled_coarse_patch_chain_submit.py"

COARSE_L="${1:-32}"
FINE_L="${2:-64}"
LAM="${3:-0.022}"
KAPPA_C="${4:-0.2705}"
KAPPA_F="${5:-0.2705}"
PATCH_SIZE="${6:-12}"
COARSE_PASSES="${7:-2}"
DETAIL_PATCH_SIZE="${8:-12}"
DETAIL_UPDATES="${9:-3}"
CHAINS="${10:-8}"
SWEEPS="${11:-1000}"

if [[ "${COARSE_L}" == "16" && "${FINE_L}" == "32" && "${ALLOW_OLD_SHAPE_PARAMETRIC_OUTPUT:-0}" != "1" ]]; then
  cat >&2 <<EOF
Refusing to launch L16->L32 through the old shape_parametric_sampler_validation workflow.

Use the separated L16->L32-trained-kernel launcher instead:
  ACTION=launch bash perfect_blocking_upsampling/scripts/launch_controlled_coarse_patch_chain_new_l16to32_kernel.sh

To intentionally run the old frozen-kernel workflow anyway, set:
  ALLOW_OLD_SHAPE_PARAMETRIC_OUTPUT=1
EOF
  exit 2
fi

TAG="controlled_coarse_patch_chain_${COARSE_L}to${FINE_L}_P${PATCH_SIZE}_pass${COARSE_PASSES}_detail${DETAIL_UPDATES}_lam${LAM}_kc${KAPPA_C}_kf${KAPPA_F}"
OUT_DIR="${ROOT_DIR}/perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/${TAG}"
LOG_DIR="${OUT_DIR}/logs"
CHECKPOINT_DIR="${OUT_DIR}/checkpoints"
mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${CHECKPOINT_DIR}"

CMD=(
  "${PYTHON}" -u -B "${SCRIPT}"
  --coarse-L "${COARSE_L}"
  --fine-L "${FINE_L}"
  --lambda "${LAM}"
  --kappa-c "${KAPPA_C}"
  --kappa-f "${KAPPA_F}"
  --patch-size "${PATCH_SIZE}"
  --coarse-passes "${COARSE_PASSES}"
  --detail-patch-size "${DETAIL_PATCH_SIZE}"
  --detail-updates-per-sweep "${DETAIL_UPDATES}"
  --chains "${CHAINS}"
  --sweeps "${SWEEPS}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --out-dir "${OUT_DIR}"
)

echo "launching:"
printf '  %q' "${CMD[@]}"
echo
echo "output_dir=${OUT_DIR}"
echo "log_file=${LOG_DIR}/run.out"

nohup "${CMD[@]}" >> "${LOG_DIR}/run.out" 2>&1 &
echo "pid=$!"
