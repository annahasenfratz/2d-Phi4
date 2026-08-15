# Upscaling Workflow

## Directory Roles

- `data/configs_phi4_2d/`: native and generated raw field configurations.
- `perfect_blocking/`: perfect-blocking kernels, kernel diagnostics, and kernel
  selection records.
- `perfect_blocking_upsampling/`: upscaling run configs, logs, checkpoints,
  observables, acceptance histories, plots, and summaries.

Do not store raw input/native/generated configurations under
`perfect_blocking_upsampling/`. Checkpoints may live under a run directory
because they are needed for continuation.

## Configuration

Every new run starts from a fully resolved YAML config. Required fields include:

- `lambda`, `kappa_f`, `kappa_c`
- `eta`, `eta_scale_numeric`, `block_factor`
- `L_f`, `L_c`
- `fine_config_source`, `coarse_config_source`
- `kernel_path`, `kernel_coefficients_include_eta_scale`
- `mode`
- `patch.detail_patch_size`, `patch.coarse_patch_size`,
  `patch.detail_passes`, `patch.coarse_passes`, `patch.random_origin`
- `random_seed`, `n_chains`, `n_sweeps`
- `checkpoint_every`, `measure_every`, `save_every`
- `resume.enabled`, `resume.checkpoint_path`

Final perfect-blocking kernels are eta-included operational kernels. Apply them
as stored and do not multiply by `eta_scale` again.

## Launching

Prepare a lambda=1.0 run with the calibrated empirical initializer:

```bash
perfect_blocking_upsampling/scripts/submit_flow_detail_coarse_detail \
  --config perfect_blocking_upsampling/run_configs/templates/lam1p0_L16to32_flow_detail.yaml
```

This creates:

- `run_config.yaml`
- `submit_manifest.txt`
- `status.json`
- `logs/`, `checkpoints/`, `observables/`, `plots/`, `summaries/`, `debug/`

Use `--execute` only after the config points to a validated runner.

The default lambda=1.0 sweep-zero initializer is
`calibrated_empirical_joint_2x2`. It draws joint 12-dimensional
`D01,D10,D11` details on aligned 2x2 coarse blocks from the fixed 1000-donor
empirical bank, using an exact cKDTree nearest-context lookup (`k=8`,
`tau=q25`, `beta=0.01`). Its only radial calibration is
`D01,D10 <- 0.97 * exp[(0.32/Lc) z] * D01,D10` for one `z ~ N(0,1)` per
configuration; `D11` is unchanged. The empirical density is an initializer
only: neither detail nor coarse patch acceptance includes it. Patch updates
remain the established exact target-action updates.

## Extending

```bash
perfect_blocking_upsampling/scripts/submit_extend_flow_detail \
  --run-dir perfect_blocking_upsampling/runs/lam1p0/<run_id> \
  --target-sweeps 200
```

The extension command reads the existing config and checkpoint. It appends work
to the same run instead of changing run metadata or starting an incompatible
run. It resumes saved chain states; it never regenerates sweep-zero empirical
details.

## Observables

Store ordinary observables in `observables/all_observables_per_config.csv`:

- `action_density`
- `phi2`
- `phi4`
- `local_kurtosis_ratio`
- `NN`
- `2nn`
- `diag`
- `m`
- `m2`
- `m4`
- `Binder_U4_from_averages`
- `xi_over_L`

Store momentum observables separately in `observables/Gk_per_config.csv`:

- `G_00`
- `G_10`
- `G_01`
- `G_pmin_avg`

Keep acceptance and sweep-level summaries in:

- `observables/acceptance_history.csv`
- `observables/sweep_summary.csv`

## Acceptance Modes

Full-detail fixed-coarse independence tests propose a complete detail field at
fixed coarse field. Their acceptance denominator is the number of full-field
proposals, normally one proposal per chain per sweep. This is a diagnostic
MIT-style normalizing-flow acceptance test, not the production patchwise update.

Local patchwise detail-only updates hold the coarse field fixed and perturb only
non-retained fine/detail sites inside random periodic fine-lattice patches. The
validated lambda=0.2 implementation uses symmetric local perturbations with exact
fine-action Metropolis correction, `log_accept = -Delta S_f`; no flow-density
term is needed for this local symmetric detail proposal. The acceptance
denominator is `chains * detail_passes * ceil(2 * L_f^2 / P_d^2)` per sweep.

Local patchwise coarse+detail updates add local coarse proposals before or after
the detail patches. In the validated lambda=0.2 corrected mode, coarse proposals
use fixed AR latents and the exact correction
`log_accept = -Delta S_f + Delta logJ_AR`; detail patches then use the same local
symmetric fixed-coarse update as detail-only mode. Coarse and detail acceptances
have separate denominators and must not be combined.

## Comparing Runs

Compare runs by histogram and support diagnostics against the relevant direct
native ensemble. Keep G(k) diagnostics separate from one-site and local-link
observables so the provenance of each comparison is clear.

## Active Scope

The active tree is focused on recent lambda=0.2, lambda=0.5, and lambda=1.0
5x5/7x7 work. Lambda=0.022 and small3 branches are quarantined, not active.

## Legacy Outputs

The old `outputs/` tree includes the successful lambda=0.2 flow/detail/coarse-detail
studies and several exploratory branches. Flow training runs go under
`runs/lam*/<run_id>/`. Patchwise production and validation chains go under
`outputs/controlled_patch_lam*/...`, for example
`outputs/controlled_patch_lam1p0/coarse_detail_L16to32/`.

There is no active package-local `data/` or `configs/` directory. Historical
snapshots of those directories were moved to `archive/` with quarantine
manifests.
