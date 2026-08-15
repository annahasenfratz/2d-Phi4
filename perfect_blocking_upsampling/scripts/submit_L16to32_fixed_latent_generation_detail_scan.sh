#!/usr/bin/env bash
set -euo pipefail

ORIGINAL_ARGV=("$0" "$@")
SUBMIT_COMMAND="$(printf '%q ' "${ORIGINAL_ARGV[@]}")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=""
if [[ -x "$REPO_ROOT/../.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/../.venv/bin/python"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
else
  echo "ERROR: could not find ../.venv/bin/python or .venv/bin/python from $REPO_ROOT" >&2
  exit 1
fi

CONFIGS=128
SWEEPS=1000
START_INDEX=""
LAMBDA=0.2
KAPPA_F=0.323124
KAPPA_C=""
INCLUDE_DETAIL16=0
DRY_RUN=0
PARALLEL=1
RESUME=0
CHECKPOINT_EVERY=50
COARSE_PATCH=4
DETAIL_PASSES="4,8"
DETAIL_PATCH=8
COARSE_SOURCES="blocked_native_L32,direct_native_L16"
CONTINUE_COMPLETED=0

usage() {
  cat <<'USAGE'
Usage:
  perfect_blocking_upsampling/scripts/submit_L16to32_fixed_latent_generation_detail_scan.sh [options]

Options:
  --configs N           Number of chains/configs per scan point (default: 128)
  --sweeps N            Sweeps per scan point (default: 1000)
  --start-index N       Deterministic contiguous coarse-config slice start index
  --lambda X            Lambda for fine target action (default: 0.2)
  --kappa-f X           Fine target kappa for exact chain (default: 0.323124)
  --kappa-c X           Coarse ensemble kappa metadata
  --include-detail16    Also run detail_passes=16 for both coarse sources
  --dry-run             Print commands without executing
  --parallel N          Run up to N scan points concurrently (default: 1, serial)
  --checkpoint-every N  Save resumable checkpoint every N sweeps (default: 50)
  --coarse-patch N      Coarse-lattice patch size for fixed-latent coarse moves (default: 4)
  --detail-passes LIST  Comma- or space-separated detail-pass scan list (default: 4,8)
  --detail-patch N      Fine/detail-coordinate patch size for detail updates (default: 8)
  --coarse-sources LIST Comma- or space-separated sources: blocked_native_L32,direct_native_L16
  --resume              Resume each scan point from its latest compatible checkpoint
  --continue-completed  With --resume, do not skip existing summary JSON; continue from checkpoint
  -h, --help            Show this help

Examples:
  # serial scan
  perfect_blocking_upsampling/scripts/submit_L16to32_fixed_latent_generation_detail_scan.sh

  # dry run
  perfect_blocking_upsampling/scripts/submit_L16to32_fixed_latent_generation_detail_scan.sh --dry-run

  # include detail_passes=16
  perfect_blocking_upsampling/scripts/submit_L16to32_fixed_latent_generation_detail_scan.sh --include-detail16
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --configs)
      CONFIGS="$2"
      shift 2
      ;;
    --sweeps)
      SWEEPS="$2"
      shift 2
      ;;
    --start-index)
      START_INDEX="$2"
      shift 2
      ;;
    --lambda)
      LAMBDA="$2"
      shift 2
      ;;
    --kappa-f)
      KAPPA_F="$2"
      shift 2
      ;;
    --kappa-c)
      KAPPA_C="$2"
      shift 2
      ;;
    --include-detail16)
      INCLUDE_DETAIL16=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --parallel)
      PARALLEL="$2"
      shift 2
      ;;
    --checkpoint-every)
      CHECKPOINT_EVERY="$2"
      shift 2
      ;;
    --coarse-patch)
      COARSE_PATCH="$2"
      shift 2
      ;;
    --detail-passes)
      DETAIL_PASSES="$2"
      shift 2
      ;;
    --detail-patch)
      DETAIL_PATCH="$2"
      shift 2
      ;;
    --coarse-sources)
      COARSE_SOURCES="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --continue-completed)
      CONTINUE_COMPLETED=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "$CONFIGS" =~ ^[0-9]+$ && "$SWEEPS" =~ ^[0-9]+$ && "$PARALLEL" =~ ^[0-9]+$ && "$CHECKPOINT_EVERY" =~ ^[0-9]+$ && "$COARSE_PATCH" =~ ^[0-9]+$ && "$DETAIL_PATCH" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --configs, --sweeps, --parallel, --checkpoint-every, --coarse-patch, and --detail-patch must be nonnegative/positive integers" >&2
  exit 1
fi
if [[ "$PARALLEL" -lt 1 ]]; then
  echo "ERROR: --parallel must be >= 1" >&2
  exit 1
fi
if [[ "$COARSE_PATCH" -lt 1 ]]; then
  echo "ERROR: --coarse-patch must be >= 1" >&2
  exit 1
fi
if [[ "$DETAIL_PATCH" -lt 1 ]]; then
  echo "ERROR: --detail-patch must be >= 1" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1

OUT_ROOT="perfect_blocking_upsampling/outputs/controlled_patch_lam0p2/L16to32_fixed_latent_generation_validation_detail_scan"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

DRIVER="perfect_blocking_upsampling/scripts/run_lam0p2_L8to16_fixed_latent_coarse_patch_chain.py"
SUMMARIZER="perfect_blocking_upsampling/scripts/summarize_L16to32_fixed_latent_generation_detail_scan.py"
NATIVE_L16="perfect_blocking_upsampling/outputs/lam0p2_kappa0p323124/native/L16/configs.npz"
NATIVE_L32="perfect_blocking_upsampling/outputs/lam0p2_kappa0p323124/native/L32/configs.npz"
BLOCKED_COARSE="$OUT_ROOT/blocked_native_L32_to_L16_coarse_rand5x5_0084.npz"

required_paths=(
  "$DRIVER"
  "$SUMMARIZER"
  "$NATIVE_L16"
  "$NATIVE_L32"
  "perfect_blocking_upsampling/outputs/controlled_patch_lam0p2/tail_aware_kernel_search_L16to32/rand5x5_0084_kernel.json"
  "perfect_blocking_upsampling/outputs/controlled_patch_lam0p2/rand5x5_0084_residual_flow_pilot/checkpoints/checkpoint_best_residual_flow.pt"
  "perfect_blocking_upsampling/outputs/controlled_patch_lam0p2/rand5x5_0084_tail_aware_ar_flow_pilot/width0p03_tail0p03/autoregressive_checkpoint.pt"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required path missing: $path" >&2
    exit 1
  fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] $PYTHON -u -B $SUMMARIZER --out-root $OUT_ROOT --submit-command $SUBMIT_COMMAND --prepare-blocked-coarse"
else
  "$PYTHON" -u -B "$SUMMARIZER" --out-root "$OUT_ROOT" --submit-command "$SUBMIT_COMMAND" --prepare-blocked-coarse
fi

DETAIL_PASSES_NORMALIZED="${DETAIL_PASSES// /,}"
IFS=',' read -r -a details <<< "$DETAIL_PASSES_NORMALIZED"
clean_details=()
for detail in "${details[@]}"; do
  if [[ -z "$detail" ]]; then
    continue
  fi
  if ! [[ "$detail" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid detail pass value in --detail-passes: $detail" >&2
    exit 1
  fi
  clean_details+=("$detail")
done
details=("${clean_details[@]}")
if [[ "$INCLUDE_DETAIL16" -eq 1 ]]; then
  has_detail16=0
  for detail in "${details[@]}"; do
    if [[ "$detail" == "16" ]]; then
      has_detail16=1
    fi
  done
  if [[ "$has_detail16" -eq 0 ]]; then
    details+=(16)
  fi
fi
if [[ "${#details[@]}" -eq 0 ]]; then
  echo "ERROR: --detail-passes produced an empty scan list" >&2
  exit 1
fi

COARSE_SOURCES_NORMALIZED="${COARSE_SOURCES// /,}"
IFS=',' read -r -a sources <<< "$COARSE_SOURCES_NORMALIZED"
clean_sources=()
for source in "${sources[@]}"; do
  if [[ -z "$source" ]]; then
    continue
  fi
  case "$source" in
    blocked_native_L32|direct_native_L16)
      clean_sources+=("$source")
      ;;
    *)
      echo "ERROR: invalid coarse source in --coarse-sources: $source" >&2
      echo "Allowed: blocked_native_L32, direct_native_L16" >&2
      exit 1
      ;;
  esac
done
sources=("${clean_sources[@]}")
if [[ "${#sources[@]}" -eq 0 ]]; then
  echo "ERROR: --coarse-sources produced an empty source list" >&2
  exit 1
fi

commands=()
names=()
for source in "${sources[@]}"; do
  for detail in "${details[@]}"; do
    run_name="${source}_patch${COARSE_PATCH}_detailpatch${DETAIL_PATCH}_detail${detail}"
    out_dir="$OUT_ROOT/$run_name"
    coarse="$NATIVE_L16"
    if [[ "$source" == "blocked_native_L32" ]]; then
      coarse="$BLOCKED_COARSE"
    fi
    cmd=("$PYTHON" -u -B "$DRIVER"
      --from-L 16
      --to-L 32
      --fine-reference "$NATIVE_L32"
      --coarse-ensemble "$coarse"
      --configs "$CONFIGS"
      --sweeps "$SWEEPS"
      --lambda "$LAMBDA"
      --kappa-f "$KAPPA_F"
      --patch-sizes "$COARSE_PATCH"
      --detail-passes "$detail"
      --detail-patch-size "$DETAIL_PATCH"
      --out-dir "$out_dir"
      --print-every 100
      --checkpoint-every "$CHECKPOINT_EVERY")
    if [[ -n "$KAPPA_C" ]]; then
      cmd+=(--kappa-c "$KAPPA_C")
    fi
    if [[ -n "$START_INDEX" ]]; then
      cmd+=(--start-index "$START_INDEX")
    fi
    if [[ "$RESUME" -eq 1 ]]; then
      cmd+=(--resume)
    fi
    commands+=("$(printf '%q ' "${cmd[@]}")")
    names+=("$run_name")
  done
done

run_one() {
  local idx="$1"
  local name="${names[$idx]}"
  local out_dir="$OUT_ROOT/$name"
  local log="$LOG_DIR/$name.log"
  local done_marker="$out_dir/fixed_latent_coarse_patch_chain_summary.json"
  if [[ -f "$done_marker" && ! ( "$RESUME" -eq 1 && "$CONTINUE_COMPLETED" -eq 1 ) ]]; then
    echo "SKIP $name: summary exists at $done_marker"
    return 0
  fi
  if [[ -f "$done_marker" && "$RESUME" -eq 1 && "$CONTINUE_COMPLETED" -eq 1 ]]; then
    echo "CONTINUE $name: summary exists, resuming from latest checkpoint because --continue-completed was set"
  fi
  echo "START $name"
  echo "LOG   $log"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] ${commands[$idx]} 2>&1 | tee $log"
  else
    mkdir -p "$out_dir"
    bash -lc "${commands[$idx]} 2>&1 | tee $(printf '%q' "$log")"
  fi
}

if [[ "$PARALLEL" -eq 1 ]]; then
  for i in "${!commands[@]}"; do
    run_one "$i"
  done
else
  active=0
  pids=()
  for i in "${!commands[@]}"; do
    run_one "$i" &
    pids+=("$!")
    active=$((active + 1))
    if [[ "$active" -ge "$PARALLEL" ]]; then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
      active=$((active - 1))
    fi
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] $PYTHON -u -B $SUMMARIZER --out-root $OUT_ROOT --submit-command $SUBMIT_COMMAND --summarize"
else
  "$PYTHON" -u -B "$SUMMARIZER" --out-root "$OUT_ROOT" --submit-command "$SUBMIT_COMMAND" --summarize
fi

cat <<EOF

L16->L32 fixed-latent detail scan launcher finished.

Serial launch:
  perfect_blocking_upsampling/scripts/submit_L16to32_fixed_latent_generation_detail_scan.sh

Dry run:
  perfect_blocking_upsampling/scripts/submit_L16to32_fixed_latent_generation_detail_scan.sh --dry-run

Include detail_passes=16:
  perfect_blocking_upsampling/scripts/submit_L16to32_fixed_latent_generation_detail_scan.sh --include-detail16

Logs:
  $LOG_DIR

Final report:
  $OUT_ROOT/detail_scan_report.md

Checkpoint/resume:
  Checkpoints are written under each run directory in checkpoints/.
  Use --resume to continue from the latest compatible checkpoint.
  Completed runs are skipped if their summary JSON already exists.
EOF
