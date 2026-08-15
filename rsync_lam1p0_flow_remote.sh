#!/usr/bin/env bash
set -euo pipefail

# Sync the curated lambda=1.0 wrapped-flow MIT diagnostic setup to the remote
# machine.  Deliberately excludes all outputs and production run data.
# Usage:
#   ./rsync_lam1p0_flow_remote.sh
#   ./rsync_lam1p0_flow_remote.sh anna@host:/path/to/Inverse_RG/

REMOTE="${1:-anna@10.0.0.96:/Users/anna/Dropbox/Research/Normalizing-flow/Inverse_RG/}"

if [[ ! "$REMOTE" == */ ]]; then
  REMOTE="${REMOTE}/"
fi

# macOS often ships rsync 2.6.x, which does not support --info=progress2.
# Use the portable progress flag by default. Opt into progress2 only on systems
# where you know the local rsync supports it:
#   RSYNC_PROGRESS2=1 ./rsync_lam1p0_flow_remote.sh
if [[ "${RSYNC_PROGRESS2:-0}" == "1" ]]; then
  COMMON_RSYNC_FLAGS=(-avR --info=progress2)
else
  COMMON_RSYNC_FLAGS=(-avR --progress)
fi

paths=(
  "submit_mit_style"
  "submit_mit_four_substep"
  "submit_mit_coordinate_mh"
  "analyze_upscaled"
  "README.md"
  "perfect_blocking_upsampling/requirements.txt"
  "perfect_blocking_upsampling/docs/mit_style_update_evolution_20260722.md"
  "perfect_blocking_upsampling/docs/REQUIRED_ARTIFACTS.md"
  "perfect_blocking_upsampling/scripts/analyze_coarse_detail_run_partial.py"
  "perfect_blocking_upsampling/scripts/run_lam1p0_l16to32_rqspline_zeroshot.py"
  "perfect_blocking_upsampling/scripts/run_lam1p0_mit_style_inverse_blocking_L8to16.py"
  "perfect_blocking_upsampling/scripts/run_lam1p0_mit_four_substep_L8to16.py"
  "perfect_blocking_upsampling/scripts/run_lam1p0_mit_coordinate_mh_L8to16.py"
  "perfect_blocking_upsampling/src/perfect_blocking_upsampling"

  "perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
  "perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.txt"

  "perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_newkernel_rqspline_N2000_action_lowtail_from_N2000_bestpatch_20260719T171401Z/checkpoints/checkpoint_best_patch.pt"
  "perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_newkernel_rqspline_N2000_action_lowtail_from_N2000_bestpatch_20260719T171401Z/run_config.yaml"
  "perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_newkernel_rqspline_N2000_action_lowtail_from_N2000_bestpatch_20260719T171401Z/normalization_metadata.json"
  "perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_newkernel_rqspline_N2000_action_lowtail_from_N2000_bestpatch_20260719T171401Z/kernel_metadata.json"

  # The MIT-style direct and four-substep jobs support L8->L16, L16->L32,
  # and L32->L64.
  "data/configs_phi4_2d/lam1p0_kappac0p340301_L8/configs.npz"
  "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
  "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
  "data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz"
)

# Some checkpoints contain provenance pointing at this older L8->L16 checkpoint.
# Sync it if present so remote checkpoint loading cannot fail on metadata fallback.
optional_paths=(
  "perfect_blocking_upsampling/runs/lam1p0/lam1p0_L8to16_kf0p340301_kc0p340301_7x7_phi2_nn_guarded_autoregressive_detail_localreg_cont_from_ep84_20260717T050248Z/checkpoints/checkpoint_best.pt"
  "perfect_blocking_upsampling/runs/lam1p0/lam1p0_L8to16_kf0p340301_kc0p340301_7x7_phi2_nn_guarded_autoregressive_detail_8layer48_rqspline_localreg_from_affine_ep137_20260717T125835Z/checkpoints/checkpoint_best.pt"
)

echo "Checking required paths..."
for path in "${paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

echo "Remote: $REMOTE"
echo "Syncing required files..."
rsync "${COMMON_RSYNC_FLAGS[@]}" "${paths[@]}" "$REMOTE"

for path in "${optional_paths[@]}"; do
  if [[ -e "$path" ]]; then
    echo "Syncing optional file: $path"
    rsync "${COMMON_RSYNC_FLAGS[@]}" "$path" "$REMOTE"
  else
    echo "Skipping optional missing file: $path"
  fi
done

echo "Done."
