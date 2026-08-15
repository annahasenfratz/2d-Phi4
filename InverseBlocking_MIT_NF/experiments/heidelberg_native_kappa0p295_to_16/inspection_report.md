# Heidelberg Native kappa=0.295 -> 16 Inspection

Date: 2026-06-24

This is a read-only inspection of `heidelberg-phi4-reproduction/` for a new Heidelberg-style branch using native 8x8 coarse fields at `lambda_c=1.0`, `kappa_c=0.295`, upscaled to 16x16. No old Heidelberg outputs were overwritten.

## Existing Code Layout

Main package:

- `heidelberg_phi4/cnf_architecture.py`: Eq. 18 naive block-repeat upsampling, simple block averaging, zero-sum Gaussian block noise, and `add_invertible_noise`.
- `heidelberg_phi4/conventions.py`: paper/current phi4 action conventions and gradients. The notes state `lambda_paper=lambda_ours`, `kappa_paper=kappa_ours`, with `S_current = S_paper + lambda*V`.
- `heidelberg_phi4/observables.py`: magnetization, susceptibility, Binder, structure factors, second-moment/Heidelberg-style correlation lengths.
- `heidelberg_phi4/local_cnf.py`: small NumPy local CNF prototype with analytic divergence.
- `heidelberg_phi4/paper_cnf.py`: NumPy Eq. 22-25 paper-style CNF block with factorized weights and analytic divergence.

Main scripts:

- `scripts/reproduce_fig4a_l8_baseline.py`: L=4 -> 8 baseline with internally generated Langevin coarse fields, naive upsampling, constrained zero-sum block noise, a global scale placeholder, and ESS/N diagnostics.
- `scripts/reproduce_fig4b_l8_kappa_scan.py`: baseline kappa scan using the same Eq. 18-21 plus global-scale approach.
- `scripts/train_ir_matching_l8_cnf.py`: NumPy/SPSA local-CNF trainer.
- `scripts/train_ir_matching_l8_full_cnf.py`: NumPy/SPSA Eq. 22-25 paper-style CNF trainer.
- `scripts/train_ir_matching_l8_torch_cnf.py`: Torch/Adam Eq. 22-25 CNF trainer with density accounting and ESS diagnostics.
- `scripts/train_ir_matching_l8_torch_joint_kappa.py`: Torch/Adam trainer with learnable coarse `kappa_L` and Eq. 34/35 logZ-gradient correction.
- `scripts/train_uv_matching_l8_torch_target_kappa.py`: Torch/Adam UV-style trainer with fixed coarse `kappa_L` and trainable fine `kappa_scriptL`.
- Phase/observable support scripts: `scan_finite_volume_phi4.py`, `reweight_*`, `estimate_kappa_cr.py`, `analyze_ensemble.py`, and plotting scripts.

Tests exist for conventions, observables, CNF architecture, local CNF, and paper CNF.

## What Was Previously Reproduced

The repo is a partial reproduction workspace for the Heidelberg/SciPost super-resolution phi4 setup. It reproduced and audited:

- Eq. 18 naive upsampling and Eqs. 19-21 zero-sum constrained block Gaussian noise.
- A first Fig. 4a baseline for `L=4 -> script L=8`, `lambda=0.01`, using a global scale placeholder rather than the full trained CNF.
- Torch/Adam Eq. 22-25 IR-matching CNF smoke, one-point, and kappa-grid runs.
- Joint-kappa IR-style runs with a trainable coarse coupling.
- UV-style target-kappa runs with a trainable fine coupling.
- Finite-volume / phase-diagram checks for several lambda values, including `lambda=1.0`, mostly outside the CNF reproduction branch.

The notes explicitly caution that the visible paper examples use `lambda=0.01`, not our current `lambda=1.0` finite-lambda problem.

## Previously Used Parameters

Representative outputs:

- `outputs/fig4a_l8_symmetric_baseline.json`
  - target: `L=4 -> 8`, `lambda=0.01`, `kappa_fine=0.10`
  - best baseline: `kappa_coarse=0.08`, `sigma=1.3`, `scale=0.55`, `ESS/N≈0.444`
  - not a trained CNF.

- `outputs/fig4_l8_torch_cnf_onepoint.json`
  - Torch Eq. 22-25 CNF
  - `coarse_L=4`, `target_L=8`, `lambda=0.01`
  - `kappa_scriptL=0.10`, `kappa_L=0.08`
  - `init_sigma=1.3`, `init_scale_flow=0.55`, `train_steps=180`
  - reported `ESS/N≈0.579`.

- `outputs/fig4_l8_torch_cnf_kappaL_grid.json`
  - `lambda=0.01`, `kappa_scriptL=0.10`
  - scanned `kappa_L=0.04,0.06,0.08,0.10,0.12`
  - best recorded `kappa_L=0.04`, `ESS/N≈0.707`.

- `outputs/fig4_l8_joint_kappa_k020.json`
  - `lambda=0.01`, target `kappa_scriptL=0.20`, initial coarse `kappa_init=0.16`
  - joint trainable coarse-kappa run with logZ correction.

- `outputs/uv_l8_target_kappa_kL016_k020.json`
  - UV-style fixed coarse `kappa_L=0.16`, initial trainable target `kappa_scriptL=0.20`
  - reported fresh `ESS/N≈0.321`.

The repo also contains `lambda=1.0` cluster phase-scan outputs such as `phi4_lambda1_cluster_l16_l24_l32_summary.json`, but these are phase/observable scans, not Heidelberg CNF runs.

## Existing Outputs and Logs

The `heidelberg-phi4-reproduction/outputs/` directory contains:

- Fig. 4 baseline JSON/CSV outputs.
- Torch CNF smoke/one-point/kappa-grid JSON/CSV outputs and `.pt` checkpoints.
- Joint-kappa JSON/CSV outputs and `.pt` checkpoints.
- UV target-kappa JSON/CSV outputs and `.pt` checkpoints.
- Finite-volume scan CSV/JSON outputs.
- Binder/susceptibility plots and reweighting outputs.
- Cluster phase-scan summaries for `lambda=0.01`, `0.1`, `0.5`, and `1.0`.

These old outputs should remain provenance only for this new branch.

## Support for This New Native-Coarse 8 -> 16 Branch

Supported directly:

- Tunable `lambda` and `kappa` in action functions and trainer CLIs.
- One-stage scale factor `b=2`; scripts check `target_L == 2*coarse_L`, so `--coarse-L 8 --target-L 16` is structurally compatible.
- Simple block-repeat upscaling and zero-sum Gaussian noise.
- Density accounting for constrained block noise, CNF logdet, action, logq, logw, and ESS in the Torch CNF scripts.
- Fine target kappa can be fixed (`train_ir_matching_l8_torch_cnf.py`) or trainable (`train_uv_matching_l8_torch_target_kappa.py`).
- Lambda can be scanned by passing `--lambda`.

Not supported cleanly yet:

- External coarse `.npy` fields are not exposed as a command-line option in the main trainers. The Torch trainer has an internal `train_one(..., coarse_np, ...)` function that can accept an external array, but `main()` currently generates coarse fields using `langevin_samples`.
- Custom precomputed initial fields are not supported by the main trainers. They construct `base = naive_upsample(coarse) + sigma * zero_sum_noise` internally.
- Explicit sigma scan over fixed initial fields is not a first-class path; `sigma` is a learnable model parameter initialized by `--init-sigma`.
- The code conditions on coarse fields through the base construction and base density, but it does not implement extra condition channels such as the symmetric Fourier `phi_back`.
- Existing coarse generation uses Langevin in these scripts. For this branch, the native kappa=0.295 input should be loaded from disk and honestly labeled as existing mixed Wolff sign-cluster plus local Metropolis amplitude data.

## Recommended Adapter Strategy

Reuse the Torch Eq. 22-25 implementation from `train_ir_matching_l8_torch_cnf.py` rather than rewriting the CNF. Add an experiment-local adapter under this branch that:

1. Loads `InverseBlocking_MIT_NF/outputs/coarse_distribution_calibration/generated_native_wolff/native_coarse_lam1_kappa0p295_L8_wolff.npy`.
2. Records the metadata caveat: existing mixed Wolff sign-cluster plus local Metropolis amplitude updates, not Wolff-only.
3. Runs preflight: `8x8 -> 16x16`, block-average preservation, finite fine action at `lambda_f=1.0`, `kappa_f=0.320`, and initial observables for `sigma in {0.10,0.15,0.20,0.30}`.
4. Calls the existing `train_one` function with `coarse_np` instead of generating Langevin coarse fields.
5. Writes all outputs into this isolated experiment directory.

This is a minimal modification path and keeps old Heidelberg reproduction outputs untouched.

## Immediate Parameter Mapping for First Tiny Pilot

Use:

- coarse input: `native_coarse_lam1_kappa0p295_L8_wolff.npy`
- `coarse_L=8`
- `target_L=16`
- `lambda_c=lambda_f=1.0`
- `kappa_L=0.295`
- first fine target: `kappa_scriptL=0.320`
- initial zero-sum noise scan: `sigma=0.10,0.15,0.20,0.30`
- first training pilot: `sigma≈0.15`, small batch, 20-50 epochs/steps equivalent, isolated outputs

Because this is Heidelberg-style simple-block-average initialization, it should not be expected to preserve the symmetric `B_sym` map. It should preserve only the simple 2x2 block average.

