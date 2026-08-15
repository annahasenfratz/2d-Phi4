# Superseded contiguous-patch MH kernel

> **Quarantined on 2026-08-04.** This note documents the contiguous-patch
> kernel retained for provenance only. Do not use it for new thermalization
> runs. The active method is described in
> [`lam1p0_checkerboard_coordinate_mh.md`](lam1p0_checkerboard_coordinate_mh.md).

This note records the MH kernel being tested for the L16 -> L32 and L32 ->
L64 upscaling runs.  The learned RQ-spline is an **initializer only**.  It is
not part of any post-initialization acceptance probability.

## Coordinates

Let `psi = K phi`, where `K` is the promoted eta-included blocking kernel and
`phi = K^{-1} psi`.  The retained coarse coordinate is

```text
psi_ee = psi[:, 0::2, 0::2].
```

The detail coordinates are the other three sublattices:

```text
psi[:, 0::2, 1::2], psi[:, 1::2, 0::2], psi[:, 1::2, 1::2].
```

The RQ-spline samples these detail coordinates once, conditional on the
direct native coarse input, to construct the sweep-zero field.  Thereafter its
latent variables, log density, and Jacobian are not evaluated or updated.

## Coarse update: fixed-detail delayed acceptance

For a symmetric random-walk patch proposal in `psi_ee`, keep all three detail
sublattices fixed.

1. With `S_c(psi_ee)`, perform the coarse MH stage:

   ```text
   A_1 = min(1, exp(-Delta S_c)).
   ```

2. For configurations passing stage 1, construct the proposed full `psi`,
   apply `K^{-1}`, and evaluate the fine action `S_f(phi)`.  Apply the delayed
   fine correction:

   ```text
   A_2 = min(1, exp(-Delta S_f + Delta S_c)).
   ```

There is no spline proposal density or Jacobian term.  The expected diagnostic
formula is:

```text
-DeltaSf + DeltaSc_after_stage1_coarse_MH_fixed_detail
```

## Detail update: fixed-coarse random walk

Keep `psi_ee` fixed.  Random-walk a local patch in only the three detail
sublattices, apply `K^{-1}`, and accept against the fine target:

```text
A_detail = min(1, exp(-Delta S_f)).
```

No random spline latent `z` is refreshed in this update.

## Sweep convention

Patch origins are random, so a sweep gives expected coverage rather than an
exact one-visit tiling.

- coarse patches per pass: `ceil(Lc^2 / Pc^2)`;
- detail patches per pass: `ceil(2 * Lf^2 / Pd^2)`.

For L16 -> L32 with `Pc = Pd = 4` and one pass, this is 16 coarse and 128
detail patch proposals per sweep, each applied to every chain in the run.

## Implementation and launch

- MH runner: `perfect_blocking_upsampling/scripts/run_lam1p0_rqspline_patchwise.py`
- generic launcher: `perfect_blocking_upsampling/scripts/submit_lam1p0_mh.sh`
- L16 -> L32 config: `perfect_blocking_upsampling/run_configs/lam1p0_L16to32_current5x5_epoch002_flowinit_mh.yaml`
- L32 -> L64 config: `perfect_blocking_upsampling/run_configs/lam1p0_L32to64_rqspline_newkernel_coarse_detail_production.yaml`

The run configuration and `logs/coarse_two_stage_history.csv` should be
checked before interpreting a trajectory.  A run whose coarse history records
`Delta_logJ` in the acceptance formula used the superseded spline/Jacobian
coarse update and is not a test of this kernel.
