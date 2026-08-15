# Phase Diagram Lambda=0.01

Standalone phi4 `lambda=0.01` finite-volume phase-diagram check, split out from
the Heidelberg phi4 reproduction thread.

## Model

Paper/action convention:

```text
S = sum_x [(1 - 2 lambda) phi_x^2 + lambda phi_x^4
           - 2 kappa sum_mu phi_x phi_{x+mu}]
lambda = 0.01
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

Broad scan centers: `kappa0 = 0.257, 0.260, 0.263`.

Refined susceptibility centers: `kappa0 = 0.260, 0.261, 0.262`.

Abs-centered susceptibility peaks:

```text
L    kappa_peak   half-spread   chi_|m| max
16   0.26079      0.00036       31.43
24   0.26089      0.00005       53.19
32   0.26099      0.00010       86.58
```

The peak locations are tight and move only slightly right with volume. The
height scaling is not yet Ising-clean:

```text
chi_|m|,max ~ L^p
p = 1.451
```

This is well below the 2D Ising value `gamma/nu = 1.75`, and close to the
non-clean value observed for `lambda=0.1`.

Binder crossings are more sensitive to the reweighting center than the
susceptibility peak locations:

```text
kappa0=0.261:
16-24    0.260515
24-32    0.260998
16-32    0.260829

kappa0=0.262:
16-24    0.261565
24-32    0.261519
16-32    0.261539
```

Working read: the finite-volume diagnostics point to `kappa ~= 0.261`, with a
systematic uncertainty at least of order `1e-3`. Treat this as a diagnostic
scan, not a final thermodynamic critical-coupling estimate.

## Files

- `scripts/phi4_lambda001_cluster_scan.py`: standalone cluster/reweighting scan.
- `scripts/plot_phi4_lambda001_phase.py`: plot and summary extraction.
- `run_all.sh`: regenerates the broad scan, refined scan, and summary plot.
- `outputs/phi4_lambda001_l16_l24_l32_chi_binder.png`: headline plot.
- `outputs/phi4_lambda001_l16_l24_l32_chi_binder.json`: extracted peak/crossing summary.
