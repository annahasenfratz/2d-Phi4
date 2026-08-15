#!/usr/bin/env bash
set -euo pipefail

# Same controlled coarse patch-chain workflow as launch_controlled_coarse_patch_chain.sh,
# but for L16->L32 with the L16->L32-trained finite-footprint checkpoint bundle.
#
# Usage:
#   ACTION=launch bash perfect_blocking_upsampling/scripts/launch_controlled_coarse_patch_chain_new_l16to32_kernel.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT_DIR}/../.venv/bin/python"
SCRIPT="${ROOT_DIR}/perfect_blocking_upsampling/scripts/run_controlled_coarse_patch_chain_submit.py"

ACTION="${ACTION:-preflight}"

COARSE_L="${1:-${COARSE_L:-16}}"
FINE_L="${2:-${FINE_L:-32}}"
LAM="${3:-${LAMBDA:-0.022}}"
KAPPA_C="${4:-${KAPPA_C:-0.2705}}"
KAPPA_F="${5:-${KAPPA_F:-0.2705}}"
PATCH_SIZE="${6:-${P_COARSE:-12}}"
COARSE_PASSES="${7:-${COARSE_PASSES:-10}}"
DETAIL_PATCH_SIZE="${8:-${P_DETAIL:-12}}"
DETAIL_UPDATES="${9:-${N_DETAIL:-1}}"
CHAINS="${10:-${N_CHAINS:-8}}"
SWEEPS="${11:-${N_SWEEPS:-500}}"
OBS_INTERVAL="${OBS_INTERVAL:-5}"
SEED="${SEED:-2026070411}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/perfect_blocking_upsampling/outputs/l16to32_trained_kernel_controlled_patch_chain_validation}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT_DIR}/perfect_blocking_upsampling/outputs/lam0p022_L16to32_flow_footprint_scan/fp_medium_1}"

TAG="controlled_coarse_patch_chain_${COARSE_L}to${FINE_L}_P${PATCH_SIZE}_pass${COARSE_PASSES}_detail${DETAIL_UPDATES}_lam${LAM}_kc${KAPPA_C}_kf${KAPPA_F}_fp_medium_1_l16to32_trained_kernel"
OUT_DIR="${OUTPUT_ROOT}/${TAG}"
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
  --save-sweeps 0 5 "${SWEEPS}"
  --progress-interval "${OBS_INTERVAL}"
  --seed "${SEED}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --out-dir "${OUT_DIR}"
  --l16to32-footprint-checkpoint-root "${CHECKPOINT_PATH}"
  --l16to32-footprint 11
)

printf 'command:'
printf ' %q' "${CMD[@]}"
printf '\n'
echo "output_dir=${OUT_DIR}"
echo "log_file=${LOG_DIR}/run.out"

if [[ "${ACTION}" == "preflight" ]]; then
  "${CMD[@]}" --dry-run
elif [[ "${ACTION}" == "launch" ]]; then
  nohup "${CMD[@]}" >> "${LOG_DIR}/run.out" 2>&1 &
  echo "pid=$!"
else
  echo "unknown ACTION=${ACTION}; expected preflight or launch" >&2
  exit 2
fi
