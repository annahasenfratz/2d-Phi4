# MIT point vs criticality

This note records the basic Wolff + reweighting scan used to place the MIT
training point in the `(\lambda,\kappa)` plane of the 2D lattice `\phi^4`
phase diagram.

## Question

The MIT training runs in our notation used approximately:

```text
lambda = 0.5
kappa  = 0.25
```

The relevant question was how far this point is from the critical region.

## Scan setup

I ran a modest phase-diagram scan in `phi4_phase-diagram/` using:

```text
lambda = 0.5
L      = 16, 32
samples per center = 4096
algorithm = local Metropolis amplitude sweeps + embedded Wolff sign-cluster updates
observables = Binder cumulant U4, chi_|m|, sign-flip diagnostics
```

Broad scan centers:

```text
kappa0 = 0.320, 0.340, 0.360
```

Refined scan centers:

```text
kappa0 = 0.335, 0.340
```

The raw outputs are in:

```text
phi4_phase-diagram/runs/lambda0p5_L16_L32_basic/outputs/phi4_lambda05_L16_L32_broad.json
phi4_phase-diagram/runs/lambda0p5_L16_L32_basic/outputs/phi4_lambda05_L16_L32_refined.json
```

## Main results

### Broad scan

For `L = 16`:

```text
kappa0 = 0.320  -> chi_abs peak at kappa = 0.329
kappa0 = 0.340  -> chi_abs peak at kappa = 0.333923...
kappa0 = 0.360  -> chi_abs peak at kappa = 0.350
```

For `L = 32`:

```text
kappa0 = 0.320  -> chi_abs peak at kappa = 0.326
kappa0 = 0.340  -> chi_abs peak at kappa = 0.337928...
kappa0 = 0.360  -> chi_abs peak at kappa = 0.355
```

Binder crossing estimate from the broad `kappa0 = 0.34` curves:

```text
kappa_c ≈ 0.34194
```

### Refined scan

For `L = 16`:

```text
kappa0 = 0.335  -> chi_abs peak at kappa = 0.334740...
kappa0 = 0.340  -> chi_abs peak at kappa = 0.333928...
```

For `L = 32`:

```text
kappa0 = 0.335  -> chi_abs peak at kappa = 0.338807...
kappa0 = 0.340  -> chi_abs peak at kappa = 0.337940...
```

The refined Binder crossing stays at essentially the same value:

```text
kappa_c ≈ 0.34194
```

## Interpretation

The MIT point `kappa = 0.25` is not close to this critical region.
It sits roughly:

```text
0.092 below the Binder-crossing estimate
0.085 to 0.089 below the susceptibility-peak region
```

So in this convention the MIT point is well on the symmetric side of the
transition, not near criticality.

## Why this matters for training

This explains why the translated `lambda = 0.022, kappa = 0.2705` runs were
harder than the original MIT tutorial point:

- the translated point is much closer to criticality,
- correlation lengths are larger,
- naive proposals have worse overlap,
- `logw` tails are heavier,
- and the flow has to model a much sharper distribution.

The consequence is that training difficulty there is not surprising; it is a
criticality/overlap issue, not just a bug in the notebook trainer.

## Bottom line

The MIT `kappa = 0.25` point is safely away from criticality in the
`lambda = 0.5` scan, while the translated `lambda = 0.022, kappa = 0.2705`
point is much more demanding because it lies near the critical region.
