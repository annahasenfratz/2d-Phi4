# THERM-L64-LAM1: L32 to L64 rethermalization at lambda=1

This study compares rethermalization of Ethan-7x7 inverse-blocked L32 to L64
fields.  The raw outputs remain in place; `runs.csv` maps the short analysis
identifiers to their immutable registry entries.
The formatted Wolff and HMC comparison tables are in
[`tables.md`](tables.md).
The expanded common-sweep comparison requested on 2026-08-21 is available as
[`requested_l32_to_l64_table.tex`](requested_l32_to_l64_table.tex), with the
human-readable rendering in [`requested_l32_to_l64_table.md`](requested_l32_to_l64_table.md)
and its machine-readable values in
[`requested_l32_to_l64_table_data.csv`](requested_l32_to_l64_table_data.csv).
The native-reference stationarity audit is in
[`native_l64_stationarity.md`](native_l64_stationarity.md), with sequential
block values in [`native_l64_stationarity.csv`](native_l64_stationarity.csv).

- **Status:** ACTIVE
- **Latest substantive update:** 2026-08-21
- **Scientific question:** Does HMC or Wolff+radial rethermalize the local and
  long-distance sectors of L32-to-L64 inverse-blocked fields to the independent
  native L64 equilibrium distribution?
- **Source ensembles:** direct native L32 configurations at
  `kappa_c=0.340301`, and independent L32 replicas at `kappa_c=0.340100`.
- **Fine target and reference:** lambda=1, L64, with the primary target
  `kappa_f=0.340301` and native reference `REF-PHI4-L64-K340301`.
- **Update methods:** global tau=2 HMC (36 leapfrog steps, step size 2/36) and
  one radial update plus four Wolff sign clusters per sweep.

## Current result

Wolff+radial rapidly removes the local mismatch.  In the same-kappa run its
action density moves from about -0.5268 at sweep zero to -0.550 within one
update, and is consistent with the native L64 reference thereafter.  The
offcritical-source Wolff run behaves similarly in local observables.

Global HMC brings `S/V` close to equilibrium by about 20 sweeps.  It does not,
however, demonstrably thermalize the long-distance quantities `chi`, `U4`, and
`xi/L` over 500 sweeps.  In the direct comparison with the common target
`kappa_f=0.340301`, the same-kappa and `kappa_c=0.340100` sources retain
different long-distance values through sweep 500.  This is evidence of slow
infrared HMC evolution, not a basis for selecting a different fine kappa.

The canonical native L64 reference itself has no detectable generation-time
drift: its first- versus second-half differences are only \(0.69\sigma\) in
\(U_4\) and \(0.30\sigma\) in \(\xi/L\). The 500-sweep warm-up is much
longer than the measured 8--12-sweep autocorrelation scales of the saved
long-distance ingredients. Thus its slightly low central \(U_4\) and
\(\xi/L\) do not currently point to incomplete native thermalization.

Wolff+radial is therefore the presently preferred rethermalization method.
For the current Wolff data, sweeps 25--75 are a conservative analysis window:
they avoid the sweep-zero transient and the visibly noisy endpoint of the
N=3000 offcritical-source run at sweep 100.

## Scope and measurement conventions

The native reference is `REF-PHI4-L64-K340301`.  `chi` is the connected
susceptibility, `L^2 (<m^2>-<m>^2)`; `U4` and `xi/L` are derived from ensemble
averages.  Their uncertainties in `observables.csv` are configuration
bootstraps at each saved sweep.  `S/V` uncertainties are configuration
standard errors saved by the run.  The HMC offcritical pair is concatenated
only because its two input ensembles and random streams are independent.

Two alternate-target HMC controls are retained.  The raw configuration files
resolve an earlier name-level ambiguity: `HMC-L64-011` has `kappa_f=0.340340`,
whereas `HMC-L64-003` has `kappa_f=0.340200`; they are distinct and must never
be merged.

When subsequent measurements materially change this conclusion, update this
README and `observables.csv` while leaving raw output directories untouched.

## Expanded common-sweep table (2026-08-21)

The expanded table includes the local operators \(S/V\), \(\phi^2\),
\(\phi^4\), NN, 2NN, and diagonal correlator, together with connected
\(\chi\), \(U_4\), and \(\xi/L\).  It uses Wolff sweeps 0, 10, 20, 50, and
100, and HMC sweeps 0, 50, 100, 250, and 500, for both
\(\kappa_c=0.340301\) and \(0.340100\), always with
\(\kappa_f=0.340301\).  All values were rederived directly from the retained
per-configuration histories; the explicit table uses a common
configuration-bootstrap prescription for every operator.
