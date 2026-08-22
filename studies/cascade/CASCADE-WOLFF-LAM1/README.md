# CASCADE-WOLFF-LAM1: Wolff+radial rethermalization in the lambda=1 cascade

The formatted per-level comparison tables are in [`tables.md`](tables.md).

- **Status:** ACTIVE
- **Latest substantive update:** 2026-08-21 — completed L64-to-L128 at 50 sweeps and prepared L128-to-L256
- **Scientific question:** Can repeated Ethan-7x7 inverse blocking and
  upscaling be rethermalized reliably at increasing volumes using a
  Wolff+radial update rather than global HMC?
- **Blocking/upscaling method:** Ethan-7x7 kernel and its associated flow.
- **Rethermalization method:** one radial update plus four Wolff sign clusters
  per sweep, with saved sweep-dependent histories and checkpoints.

## Why this study uses Wolff+radial

This is an explicit algorithmic follow-on to
[`THERM-L64-LAM1`](../../thermalization/THERM-L64-LAM1/README.md).  That study
found that global HMC brings the local action density close to the independent
native L64 reference by roughly 20 sweeps, but does not demonstrate
equilibration of `chi`, `U4`, and `xi/L` over 500 sweeps.  Wolff+radial removes
the local mismatch rapidly and is therefore the current preferred
rethermalization method.

The evidence and quantitative L64 measurements justifying this switch are
preserved in the linked study's `runs.csv` and `observables.csv`; this study
does not duplicate them.

## Planned cascade record

Each cascade level will be added to `runs.csv` with stable raw `run_id`s from
the project registry.  `observables.csv` will retain the sweep histories of
local observables, long-distance observables, apparent stationarity, and—when
available—comparison against an independent native equilibrium ensemble.
No plateau alone will be treated as proof of thermalization.

## First completed level

`WOLFF-CASCADE-L32-001` is complete.  It takes the first 1500
fields from the existing fresh pure-NLL Ethan-7x7 direct-native L16 sweep-zero
sample, rethermalizes at L32 for 100 Wolff+radial sweeps, and writes to:

```text
perfect_blocking_upsampling/outputs/cascade_wolff_lam1p0/
  CASCADE-WOLFF-LAM1/
    L16toL32_kc0p340301_kf0p340301/
      N1500_r1/
        run_config.yaml
        observables/
        checkpoints/checkpoint_sweep_0100.npz
```

The next level will be a sibling such as
`L32toL64_kc0p340301_kf0p340301/N1500_r1/`, and takes the preceding level's
final checkpoint as its source.  More statistics are represented by a new,
independent sibling such as `N1500_r2/` rather than silently replacing this
run.  This layout retains the usual `run_config.yaml`, `observables/`, and
`checkpoints/` paths, so existing plotting cells need only change `RUN_DIR`.

`WOLFF-CASCADE-L32-002` is the completed independent `N1500_r2` realization.
It uses the disjoint native-L16 and matching sweep-zero-flow indices 1500--2999
from the verified 5000-field source, with the same kernel, flow, and completed
100-sweep Wolff+radial schedule. It remains separate from r1 unless a later analysis
explicitly pools the two replicas.

## Prepared second level: L32 to L64

`FLOW-CASCADE-L64-001` has applied the unchanged Ethan-7x7 flow to the saved
L32 checkpoint from `WOLFF-CASCADE-L32-001`. Its sweep-zero L64 output is the
sole input to `WOLFF-CASCADE-L64-001`, which is prepared to perform 100
Wolff+radial sweeps. The two stages live under the same level/statistics path as
`initialization/` and the level root, respectively; use the level root as
`RUN_DIR` for the usual thermalization plotting notebooks. This matches the
existing L16-to-L32 level layout.

`FLOW-CASCADE-L64-002` completed the matching second-level `N1500_r2`
flow-only branch in 53.70 s. `WOLFF-CASCADE-L64-002` is ready to begin from
its sweep-zero checkpoint; the branch originated from source indices
1500--2999 and remains separate from r1 until a pooling analysis explicitly
combines the replicas.

The completed L32-to-L64 level is locally equilibrated by sweep 25: its
action-density discrepancy decreases from about 26 standard deviations at
sweep zero to below one. Its `chi`, `U4`, and `xi/L` measurements at sweeps
25, 50, and 100 are compatible with independent native L64. A sweep-75
long-distance fluctuation is not sustained at sweep 100, so there is no
observed persistent drift at the present N=1500 precision. See `tables.md`
and `observables.csv` for the values and uncertainties.

## Completed third level: L64 to L128

`FLOW-CASCADE-L128-001` applied the unchanged Ethan-7x7 flow to the saved
L64 checkpoint from `WOLFF-CASCADE-L64-001`. `WOLFF-CASCADE-L128-001` then
ran 50 Wolff+radial sweeps from that sweep-zero L128 field. Its independent comparison ensemble is
`REF-PHI4-L128-K340301`.

## Prepared fourth level: L128 to L256

`FLOW-CASCADE-L256-001` will use the completed L128
`checkpoint_sweep_0050.npz`, apply the unchanged Ethan-7x7 flow in streamed
batches of 10, and save the L256 sweep-zero configuration under
`L128toL256_kc0p340301_kf0p340301/N1500_r1/initialization/`. Then
`WOLFF-CASCADE-L256-001` will apply 50 Wolff+radial sweeps in the parent
directory. This level has no independent native L256 comparison ensemble; its
full history will nevertheless be retained for stationarity checks and for the
next cascade level.

Against `REF-PHI4-L32-K340301`, the L16-to-L32 run has removed its large
sweep-zero local-action discrepancy by sweep 25.  Its sweep-25--100 values
of `S/V`, `chi`, `U4`, and `xi/L` are compatible with the native reference at
the current N=1500 precision; the full history is retained in
`observables.csv`.  This is encouraging but is not, by itself, a substitute
for the broader thermalization evidence from `THERM-L64-LAM1`.

When a cascade run materially changes this choice or its evidence, update this
README and `observables.csv` without modifying raw output directories.
