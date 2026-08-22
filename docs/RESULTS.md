# Results index

## Active studies

- [`THERM-L64-LAM1`](../studies/thermalization/THERM-L64-LAM1/README.md) —
  L32-to-L64 rethermalization at lambda=1.  Current conclusion: Wolff+radial
  rapidly equilibrates local observables and is preferred to global HMC for
  this rethermalization task; HMC has not demonstrated long-distance
  equilibration over 500 sweeps.
- [`CASCADE-WOLFF-LAM1`](../studies/cascade/CASCADE-WOLFF-LAM1/README.md) —
  active lambda=1 cascade study.  Uses Wolff+radial rethermalization on the
  basis of the evidence in `THERM-L64-LAM1`.
- [`ETA-FREE-ETHAN7-LAM1`](../studies/kernel_optimization/ETA-FREE-ETHAN7-LAM1/README.md) —
  free-eta diagnostic for Ethan's 7x7 kernel. It finds \(\eta=0.2519\), not a
  meaningful departure from 0.25; the candidate is not promoted.

See `registry/runs.csv` for raw-run provenance and each study's
`observables.csv` for the machine-readable measurements underlying its result.
