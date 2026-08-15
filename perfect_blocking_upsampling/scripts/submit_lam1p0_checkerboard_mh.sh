#!/usr/bin/env bash
# Flow initialization followed by exact checkerboard coarse/detail MH.
# Usage: bash $0 r1 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -x "$ROOT/../../.venv/bin/python" ]]; then
  PYTHON="$ROOT/../../.venv/bin/python"
elif [[ -x "$ROOT/../.venv/bin/python" ]]; then
  PYTHON="$ROOT/../.venv/bin/python"
else
  echo "shared Python not found" >&2
  exit 1
fi

RUN_NUMBER="${1:-}"
if [[ ! "$RUN_NUMBER" =~ ^r[0-9]+$ ]]; then
  echo "usage: $0 rNUMBER [--execute] [--background]" >&2
  exit 2
fi
shift

# The L16->L32 defaults reproduce the complete successful July-24 setup:
# checkerboard geometry, the archived July flow, and its archived kernel.
# The current flow/kernel remain available through FLOW_CHECKPOINT/KERNEL_PATH
# overrides and are not modified by this launcher.
LC="${LC:-16}"
LF="${LF:-32}"
N_CHAINS="${N_CHAINS:-1000}"
N_SWEEPS="${N_SWEEPS:-400}"
START_INDEX="${START_INDEX:-4000}"
SEED="${SEED:-2026080429}"
BATCH_SIZE="${BATCH_SIZE:-50}"
DIVIDE="${DIVIDE:-2}"
DETAIL_PASSES="${DETAIL_PASSES:-2}"
COARSE_SIGMA="${COARSE_SIGMA:-0.40}"
COARSE_PROPOSAL_MODE="${COARSE_PROPOSAL_MODE:-sc_reversible}"
DETAIL_SIGMA="${DETAIL_SIGMA:-0.10}"
COARSE_UPDATES="${COARSE_UPDATES:-1}"
INITIAL_DETAIL_ONLY_SWEEPS="${INITIAL_DETAIL_ONLY_SWEEPS:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
PREVIOUS_SWEEPS="${PREVIOUS_SWEEPS:-0}"
SAVE_EVERY="${SAVE_EVERY:-5}"
DEVICE="${DEVICE:-cpu}"
FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-}"
FLOW_VARIANT="${FLOW_VARIANT:-patch}"
KERNEL_PATH="${KERNEL_PATH:-}"
KERNEL_VARIANT="${KERNEL_VARIANT:-july}"
INITIALIZATION="${INITIALIZATION:-direct_coarse_flow}"
UPDATE_MODE="${UPDATE_MODE:-coarse_detail}"
DRIVER="$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_checkerboard_mh.py"

if [[ "$LF" -ne $((2 * LC)) ]]; then
  echo "require LF=2*LC; got L${LC}->L${LF}" >&2
  exit 2
fi
case "$INITIALIZATION" in
  direct_coarse_flow|direct_coarse_impflow|blocked_native|blocked_native_flow) ;;
  *) echo "INITIALIZATION must be direct_coarse_flow, direct_coarse_impflow, blocked_native, or blocked_native_flow; got $INITIALIZATION" >&2; exit 2 ;;
esac
case "$UPDATE_MODE" in
  coarse_detail|detail_only) ;;
  *) echo "UPDATE_MODE must be coarse_detail or detail_only; got $UPDATE_MODE" >&2; exit 2 ;;
esac
case "${LC}:${LF}" in
  8:16|16:32|32:64)
    COARSE_SOURCE="$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L${LC}/configs.npz"
    NATIVE_SOURCE="$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L${LF}/configs.npz"
    ;;
  *) echo "supported factor-two pairs: L8->L16, L16->L32, L32->L64" >&2; exit 2 ;;
esac

if [[ -z "$FLOW_CHECKPOINT" ]]; then
  if [[ "${LC}:${LF}" == "16:32" ]]; then
    case "$FLOW_VARIANT" in
      patch)
        FLOW_CHECKPOINT="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_newkernel_rqspline_N2000_action_lowtail_from_N2000_bestpatch_20260719T171401Z/checkpoints/checkpoint_best_patch.pt"
        ;;
      nll)
        FLOW_CHECKPOINT="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_newkernel_rqspline_N2000_action_lowtail_from_N2000_bestpatch_20260719T171401Z/checkpoints/checkpoint_best_nll.pt"
        ;;
      pure_nll)
        FLOW_CHECKPOINT="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_current5x5_pureNLL_from_tailstratifiedNLL_N5000_20260805T065543Z/stage_oo/checkpoints/checkpoint_best_nll.pt"
        ;;
      *)
        echo "FLOW_VARIANT must be patch, nll, or pure_nll; got $FLOW_VARIANT" >&2
        exit 2
        ;;
    esac
  else
    echo "Set FLOW_CHECKPOINT explicitly for L${LC}->L${LF}; the restored July flow is L16->L32 only." >&2
    exit 2
  fi
fi

if [[ -z "$KERNEL_PATH" ]]; then
  case "$KERNEL_VARIANT" in
    july)
      KERNEL_PATH="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernel_retune_pc1_radial_mode_20260720/current_kernel.json"
      ;;
    current)
      KERNEL_PATH="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
      ;;
    *)
      echo "KERNEL_VARIANT must be july or current; got $KERNEL_VARIANT" >&2
      exit 2
      ;;
  esac
fi

if (( SAVE_EVERY <= 0 || PREVIOUS_SWEEPS < 0 )); then echo "SAVE_EVERY must be positive and PREVIOUS_SWEEPS nonnegative" >&2; exit 2; fi
if [[ -n "$RESUME_CHECKPOINT" && "$INITIAL_DETAIL_ONLY_SWEEPS" != "0" ]]; then
  echo "INITIAL_DETAIL_ONLY_SWEEPS must be zero when RESUME_CHECKPOINT is set" >&2
  exit 2
fi
if [[ -z "$RESUME_CHECKPOINT" && "$PREVIOUS_SWEEPS" != "0" ]]; then
  echo "PREVIOUS_SWEEPS requires RESUME_CHECKPOINT" >&2
  exit 2
fi
END_SWEEP=$((PREVIOUS_SWEEPS + N_SWEEPS))
SAVE_SWEEPS="${SAVE_SWEEPS:-$(seq -s, "$PREVIOUS_SWEEPS" "$SAVE_EVERY" "$END_SWEEP")}" 
INIT_TAG=""
if [[ "$INITIALIZATION" == "blocked_native" ]]; then INIT_TAG="_blockedNative"; fi
if [[ "$INITIALIZATION" == "blocked_native_flow" ]]; then INIT_TAG="_blockedNativeFlow"; fi
UPDATE_TAG=""
if [[ "$UPDATE_MODE" == "detail_only" ]]; then UPDATE_TAG="_detailOnly"; fi
COARSE_MODE_TAG=""
if [[ "$COARSE_PROPOSAL_MODE" == "symmetric_rw" ]]; then COARSE_MODE_TAG="_crw"; fi
RUN_TAG="L${LC}toL${LF}${INIT_TAG}_N${N_CHAINS}_S${END_SWEEP}_start${START_INDEX}_div${DIVIDE}_D${DETAIL_PASSES}${UPDATE_TAG}${COARSE_MODE_TAG}_sc${COARSE_SIGMA/./p}_sd${DETAIL_SIGMA/./p}_${RUN_NUMBER}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/perfect_blocking_upsampling/outputs/checkerboard_mh_lam1p0}"
RUN="$OUTPUT_ROOT/L${LC}toL${LF}/$RUN_TAG"

CMD=(
  "$PYTHON" -B "$DRIVER"
  --run-dir "$RUN"
  --coarse-proposal-source "$COARSE_SOURCE"
  --native-reference-source "$NATIVE_SOURCE"
  --flow-checkpoint "$FLOW_CHECKPOINT"
  --kernel-path "$KERNEL_PATH"
  --from-L "$LC" --to-L "$LF"
  --n-chains "$N_CHAINS" --n-sweeps "$N_SWEEPS" --save-sweeps "$SAVE_SWEEPS"
  --batch-size "$BATCH_SIZE" --seed "$SEED" --device "$DEVICE"
  --initial-start-index "$START_INDEX"
  --initialization "$INITIALIZATION" --improved-flow-map none
  --update-mode "$UPDATE_MODE"
  --coarse-sigma "$COARSE_SIGMA" --coarse-updates-per-sweep "$COARSE_UPDATES"
  --coarse-proposal-mode "$COARSE_PROPOSAL_MODE"
  --detail-sigma "$DETAIL_SIGMA" --detail-passes-per-sweep "$DETAIL_PASSES"
  --initial-detail-only-sweeps "$INITIAL_DETAIL_ONLY_SWEEPS"
  --sweep-offset "$PREVIOUS_SWEEPS"
  --divide "$DIVIDE"
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  CMD+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

EXECUTE=0
BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 rNUMBER [--execute] [--background]" >&2; exit 2 ;;
  esac
done
if [[ "$BACKGROUND" -eq 1 && "$EXECUTE" -eq 0 ]]; then
  echo "--background requires --execute" >&2
  exit 2
fi
for path in "$PYTHON" "$DRIVER" "$FLOW_CHECKPOINT" "$KERNEL_PATH" "$COARSE_SOURCE" "$NATIVE_SOURCE"; do
  [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }
done
if [[ -n "$RESUME_CHECKPOINT" ]]; then [[ -f "$RESUME_CHECKPOINT" ]] || { echo "missing resume checkpoint: $RESUME_CHECKPOINT" >&2; exit 1; }; fi
if [[ -e "$RUN" ]]; then
  echo "run directory already exists: $RUN" >&2
  exit 1
fi

printf 'run_dir=%s\n' "$RUN"
printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "$EXECUTE" -eq 0 ]]; then
  echo "Prepared only. Add --execute to start; add --background for nohup."
  exit 0
fi

mkdir -p "$RUN/logs"
{
  echo "run_id=$RUN_TAG"
  echo "initialization=$INITIALIZATION"
  echo "algorithm=exact checkerboard coarse/detail MH; flow is used only for flow initializations"
  echo "update_mode=$UPDATE_MODE"
  echo "detail=three sectors x divide^2 residue classes x DETAIL_PASSES complete passes"
  echo "initial_detail_only_sweeps=$INITIAL_DETAIL_ONLY_SWEEPS (not included in the run name)"
  echo "continuation_from_sweep=$PREVIOUS_SWEEPS"
  if [[ "$COARSE_PROPOSAL_MODE" == "symmetric_rw" ]]; then
    echo "coarse=symmetric Gaussian random walk in each checkerboard residue class; accepted with -Delta S_f"
  else
    echo "coarse=per-site S_c transition in each residue class, outer fine correction"
  fi
  printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
} > "$RUN/submit_manifest.txt"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" > "$RUN/logs/run.log" 2>&1 < /dev/null &
  echo "$!" > "$RUN/submit_pid.txt"
  printf 'background_pid=%s\nlog=%s\n' "$!" "$RUN/logs/run.log"
else
  exec "${CMD[@]}"
fi
