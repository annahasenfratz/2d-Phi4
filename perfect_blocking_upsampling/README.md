# Perfect Blocking Upsampling

This tree contains shared upscaling scripts, run templates, and run outputs for
perfect-blocking upsampling studies. Native and generated raw field
configurations belong under `data/configs_phi4_2d/`, not in this tree, except
for checkpoint state required to continue a run.

## Layout

- `scripts/`: stable entry points plus reusable helpers.
- `scripts/submit_flow_detail_coarse_detail`: prepare or launch a new
  flow/detail/coarse-detail run.
- `scripts/submit_extend_flow_detail`: extend an existing run from
  `checkpoints/checkpoint_latest.*`.
- `run_configs/templates/`: lambda-specific YAML templates for active 5x5/7x7
  workflows.
- `runs/lam0p2/`, `runs/lam0p5/`, `runs/lam1p0/`: clean run directories for
  new work.
- `outputs/`: legacy and recent output archive. Small3 and lambda=0.022 legacy
  branches have been quarantined under `archive/`.
- `archive/`: quarantined legacy configs, raw-config snapshots, small3 tests,
  and older lambda=0.022 branches.

There is intentionally no active top-level `data/` or `configs/` directory in
this package. Use `../data/configs_phi4_2d/` for raw field configurations and
`run_configs/templates/` for run YAML.

## Run Directory Contract

Each new run directory should contain:

```text
run_config.yaml
submit_manifest.txt
logs/
checkpoints/
observables/
plots/
summaries/
debug/
status.json
```

Observable CSVs are stored in the run directory:

- `observables/all_observables_per_config.csv`
- `observables/Gk_per_config.csv`
- `observables/acceptance_history.csv`
- `observables/sweep_summary.csv`

Raw field configurations are not stored in the run directory unless they are
checkpoint state needed for restart.

## Launch

Prepare a run directory without executing the long simulation:

```bash
perfect_blocking_upsampling/scripts/submit_flow_detail_coarse_detail \
  --config perfect_blocking_upsampling/run_configs/templates/lam1p0_L16to32_flow_detail.yaml
```

The command prints the prepared run directory. Add `--execute` only when the
template has a valid `legacy_runner` or modern runner implementation for that
lambda.

## Extend

```bash
perfect_blocking_upsampling/scripts/submit_extend_flow_detail \
  --run-dir perfect_blocking_upsampling/runs/lam1p0/<run_id> \
  --target-sweeps 100
```

Add `--execute` to run the configured extension implementation. The extender
reads `run_config.yaml`, locates `checkpoints/checkpoint_latest.*`, writes an
`extend_*.log`, and updates `status.json`.

## Legacy Lambda 0.2 Template

The mature lambda=0.2 flow/detail/coarse-detail implementation remains in:

- `scripts/run_lam0p2_flow_detail_rethermalization.py`
- `scripts/extend_lam0p2_flow_detail_rethermalization.py`
- `outputs/controlled_patch_lam0p2/flow_detail_rethermalization_L16to32_latest/`

The new wrappers preserve those scripts as legacy hooks while moving new run
metadata into `runs/`.

## Quarantine Manifests

Cleanup manifests record the non-destructive moves:

- `reorg_manifest_20260716T204037Z.csv`
- `quarantine_manifest_20260716T204830Z.csv`
- `quarantine_manifest_20260716T204904Z.csv`
