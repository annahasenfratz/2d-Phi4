# Lambda=1 checkerboard-coordinate MH

## Decision

Contiguous real-space patch proposals are quarantined.  They have a high
acceptance rate but do not efficiently move the long-wavelength degrees of
freedom relevant to L16 -> L32 and L32 -> L64 thermalization.

The active production method is the July-24 checkerboard-coordinate kernel,
with the current flow used only to initialize sweep zero and the current
eta-included blocking kernel used for `psi = K phi`.

## Coordinates and initialization

`psi_ee = psi[:, 0::2, 0::2]` is the retained coarse field.  The other three
parity sectors are `d01`, `d10`, and `d11`.  A conditional RQ-spline samples
the three detail fields once given a direct native coarse configuration.
There is no latent refresh, flow-density, or flow-Jacobian factor after this
initialization.

## One recorded sweep

Set `divide=2`.  The coarse lattice is split into the four residue classes
`(y mod 2, x mod 2)`.  Every class contains spatially dispersed sites, not a
contiguous patch.

1. **Coarse transition.** For each residue class, perform one random-order,
   per-site Gaussian Metropolis transition in `psi_ee` targeting `exp(-S_c)`.
   Apply `K^{-1}` and accept the resulting coarse proposal with

   ```text
   log A_coarse = -Delta S_f + Delta S_c.
   ```

2. **Detail transition.** For every requested detail pass, then every residue
   class, update all active coordinates of `d01`, then `d10`, then `d11`
   simultaneously by symmetric Gaussian random walks.  Apply `K^{-1}` and
   accept each sector block with

   ```text
   log A_detail = -Delta S_f.
   ```

For L16 -> L32, one residue block contains 64 coordinates of one detail
sector.  With `detail_passes=2`, one sweep has 24 such detail MH decisions
and covers every detail coordinate exactly twice.  This was the successful
July-24 geometry.  Its lower detail acceptance (about 0.22) is expected and
is not a defect: accepted moves are spatially distributed and effective for
IR modes.

## Current production defaults

The canonical launcher is:

```bash
bash perfect_blocking_upsampling/scripts/submit_lam1p0_checkerboard_mh.sh r1 --execute --background
```

Defaults are L16 -> L32, 500 chains, 400 sweeps, `divide=2`, two detail
passes, coarse sigma 0.4, detail sigma 0.1, and measurements every five
sweeps.  It uses the current Aug-3 L16 -> L32 flow checkpoint and the current
promoted kernel.  To run L32 -> L64, set `LC=32 LF=64` before the command.

Every run stores full kernel metadata in `run_config.yaml`, so the matrix is
pinned even if the default kernel path changes later.

## Historical reference

Representative July-24 run:

```text
mitcoord_bL32_wrapped_RQS_cfg4000_N512_Dpc2_Rpc1_S400_coordinateMH_coarseDetail_div2_20260724T173710Z
```

It used the older July-19 flow and a prior 5x5 kernel matrix.  The July-24
and current kernels have the same eta convention and sum but are not identical.

## Quarantined method

The `rqspline_patchwise` runner and all output/test artifacts based on
contiguous `Pc x Pc` / `Pd x Pd` patches are retained only for provenance in
the dated contiguous-patch quarantine.  Do not use them for new production
thermalization runs.
