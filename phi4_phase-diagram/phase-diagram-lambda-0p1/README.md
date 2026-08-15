# Phase Diagram Lambda=0.1

Standalone phi4 `lambda=0.1` finite-volume phase-diagram check, split out from
the Heidelberg phi4 reproduction thread.

## Model

Paper/action convention:

```text
S = sum_x [(1 - 2 lambda) phi_x^2 + lambda phi_x^4
           - 2 kappa sum_mu phi_x phi_{x+mu}]
lambda = 0.1
```

Sampling uses local Metropolis amplitude sweeps plus embedded Wolff sign-cluster
updates at fixed amplitudes. At fixed `lambda`, reweighting in `kappa` uses

```text
w(kappa; kappa0) = exp[2 (kappa - kappa0) B]
B = sum_x,mu phi_x phi_{x+mu}
```

The susceptibility used for finite-size scaling is

```text
chi_|m| = V (<m^2> - <|m|>^2)
```

## Current Results

Volumes: `L = 16, 24, 32`.

Broad scan centers: `kappa0 = 0.280, 0.300, 0.320`.

Refined susceptibility centers: `kappa0 = 0.295, 0.300, 0.305`.

Abs-centered susceptibility peaks:

```text
L    kappa_peak   half-spread   chi_|m| max
16   0.29905      0.00036       16.56
24   0.30001      0.00150       31.51
32   0.30027      0.00250       45.34
```

Important: `lambda=0.1` shows noticeably larger reweighting-center dependence
than `lambda=0.5` or `lambda=1`, especially for `L=32`. This supports the worry
that the Heidelberg `lambda=0.01` regime may require more careful sampling,
larger statistics, narrower/multiple-histogram reweighting, or larger volumes
before fitting finite-size scaling.

Naive peak-height scaling from averaged peaks:

```text
chi_|m|,max ~ L^p
p = 1.462
```

This is not yet Ising-clean; compare `p = 1.753` at `lambda=0.5` and
`p = 1.726` at `lambda=1` in the companion threads.

Binder crossings from refined centers:

```text
kappa0=0.300:
16-24    0.302465
24-32    closest grid 0.3025, no sign change
16-32    closest grid 0.3025, no sign change

kappa0=0.305:
16-24    0.303344
24-32    closest grid 0.3025, no sign change
16-32    0.302571
```

Working read: Binder points to roughly `kappa ~= 0.3025..0.303`, while
susceptibility peaks are around `0.299..0.300` with significant center
dependence. Treat this as a diagnostic scan, not a final phase-boundary
estimate.

## Files

- `scripts/phi4_lambda01_cluster_scan.py`: standalone cluster/reweighting scan.
- `scripts/plot_phi4_lambda01_phase.py`: plot and summary extraction.
- `run_all.sh`: regenerates the broad scan, refined scan, and summary plot.
- `outputs/phi4_lambda01_l16_l24_l32_chi_binder.png`: headline plot.
- `outputs/phi4_lambda01_l16_l24_l32_chi_binder.json`: extracted peak/crossing summary.
