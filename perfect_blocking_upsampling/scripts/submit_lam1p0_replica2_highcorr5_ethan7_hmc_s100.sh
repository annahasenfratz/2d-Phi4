#!/usr/bin/env bash
# Independent-replica HMC comparison: highcorr 5x5 versus Ethan 7x7.
# Both flows start from the new disjoint N=5000 direct native L16 Wolff sample
# and then use identical full-field L32 HMC (tau=2) through sweep 100.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PY" ]] || { echo "Python interpreter not found: $PY" >&2; exit 1; }

EXECUTE=0
BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done

N_CHAINS="${N_CHAINS:-5000}"
N_SWEEPS="${N_SWEEPS:-100}"
SAVE_EVERY="${SAVE_EVERY:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
HMC_BATCH_SIZE="${HMC_BATCH_SIZE:-32}"
COARSE="$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L16_N5000_replica2_20260818/configs.npz"
NATIVE="$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
OUTROOT="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global"

HIGHCORR_FLOW="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_base_highcorr5_pureNLL_retrain_N10000_20260818T045952Z/stage_oo/checkpoints/checkpoint_best_nll.pt"
HIGHCORR_KERNEL="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R2_corrW5000_highcorr_5x5_eta_included.json"
HIGHCORR_RUN="$OUTROOT/L16toL32_highcorr5_pureNLL_directnative_replica2_N${N_CHAINS}_S${N_SWEEPS}_tau2_n28_eps2over28_20260818"
ETHAN_FLOW="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_ethan7_fresh_pureNLL_control_N10000_20260818T050235Z/stage_oo/checkpoints/checkpoint_best_nll.pt"
ETHAN_KERNEL="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/ethan_7x7_paper_objective_eta_included.json"
ETHAN_RUN="$OUTROOT/L16toL32_ethan7_fresh_pureNLL_directnative_replica2_N${N_CHAINS}_S${N_SWEEPS}_tau2_n28_eps2over28_20260818"

make_cmd() {
  local tag="$1"
  local flow kernel seed run
  if [[ "$tag" == highcorr5 ]]; then
    flow="$HIGHCORR_FLOW"; kernel="$HIGHCORR_KERNEL"; seed=2026081819; run="$HIGHCORR_RUN"
  else
    flow="$ETHAN_FLOW"; kernel="$ETHAN_KERNEL"; seed=2026081820; run="$ETHAN_RUN"
  fi
  env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py" \
    --run-dir "$run" --native-source "$NATIVE" --coarse-source "$COARSE" \
    --flow-checkpoint "$flow" --kernel-path "$kernel" --initialization direct_coarse_flow \
    --n-chains "$N_CHAINS" --n-sweeps "$N_SWEEPS" --save-every "$SAVE_EVERY" \
    --batch-size "$BATCH_SIZE" --hmc-batch-size "$HMC_BATCH_SIZE" --measurement-batch-size "$HMC_BATCH_SIZE" \
    --step-size 0.07142857142857142 --leapfrog-steps 28 --divide 1 --seed "$seed" --level-name L16toL32
}

printf 'coarse replica=%s\nN=%s sweeps=%s save_every=%s\nfull-field HMC: eps=2/28, n=28, tau=2\n' "$COARSE" "$N_CHAINS" "$N_SWEEPS" "$SAVE_EVERY"
printf 'highcorr5 run=%s\nethan7 run=%s\n' "$HIGHCORR_RUN" "$ETHAN_RUN"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$COARSE" && -f "$NATIVE" ]] || { echo "missing coarse or native source" >&2; exit 1; }
for tag in highcorr5 ethan7; do
  if [[ "$tag" == highcorr5 ]]; then flow="$HIGHCORR_FLOW"; kernel="$HIGHCORR_KERNEL"; run="$HIGHCORR_RUN"; else flow="$ETHAN_FLOW"; kernel="$ETHAN_KERNEL"; run="$ETHAN_RUN"; fi
  [[ -f "$flow" && -f "$kernel" ]] || { echo "missing model assets for $tag" >&2; exit 1; }
  [[ ! -e "$run" ]] || { echo "refusing to overwrite existing run: $run" >&2; exit 1; }
done

launch() {
  local tag="$1"
  local flow kernel seed run
  if [[ "$tag" == highcorr5 ]]; then
    flow="$HIGHCORR_FLOW"; kernel="$HIGHCORR_KERNEL"; seed=2026081819; run="$HIGHCORR_RUN"
  else
    flow="$ETHAN_FLOW"; kernel="$ETHAN_KERNEL"; seed=2026081820; run="$ETHAN_RUN"
  fi
  mkdir -p "$run/logs"
  if [[ "$BACKGROUND" -eq 1 ]]; then
    {
      echo "started_at=$(date -Iseconds)"
      echo "coarse_replica=$COARSE"
      printf 'flow_checkpoint=%s\nkernel=%s\n' "$flow" "$kernel"
      echo 'full-field HMC: eps=2/28, n=28, tau=2'
    } > "$run/logs/run.log"
    nohup env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py" \
      --run-dir "$run" --native-source "$NATIVE" --coarse-source "$COARSE" \
      --flow-checkpoint "$flow" --kernel-path "$kernel" --initialization direct_coarse_flow \
      --n-chains "$N_CHAINS" --n-sweeps "$N_SWEEPS" --save-every "$SAVE_EVERY" \
      --batch-size "$BATCH_SIZE" --hmc-batch-size "$HMC_BATCH_SIZE" --measurement-batch-size "$HMC_BATCH_SIZE" \
      --step-size 0.07142857142857142 --leapfrog-steps 28 --divide 1 --seed "$seed" --level-name L16toL32 \
      >> "$run/logs/run.log" 2>&1 </dev/null &
    echo "$!" > "$run/submit_pid.txt"
    echo "$tag background_pid=$!"
  else
    make_cmd "$tag" 2>&1 | tee "$run/logs/run.log"
  fi
}
launch highcorr5
launch ethan7
