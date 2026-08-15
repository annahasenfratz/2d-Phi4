# Phase Diagram Lambda=1

Standalone phi4 `lambda=1` finite-volume phase-diagram sanity check, split out
from the Heidelberg phi4 reproduction thread.

## Model

Paper/action convention:

```text
S = sum_x [(1 - 2 lambda) phi_x^2 + lambda phi_x^4
           - 2 kappa sum_mu phi_x phi_{x+mu}]
lambda = 1
```

Sampling uses local Metropolis amplitude sweeps plus embedded Wolff sign-cluster
updates at fixed amplitudes.  At fixed `lambda`, reweighting in `kappa` uses

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

Refined susceptibility centers: `kappa0 = 0.330, 0.335`.

Broad Binder center: `kappa0 = 0.340`.

Abs-centered susceptibility peaks:

```text
L    kappa_peak   half-spread   chi_|m| max
16   0.32854      0.00038       11.25
24   0.33235      0.00021       22.51
32   0.33446      0.00011       37.22
```

Important: the susceptibility peak is still moving right with volume.  Do not
interpret these finite-volume peaks as a final infinite-volume `kappa_c`.

Peak-height scaling:

```text
chi_|m|,max ~ L^p
p = 1.726
```

Binder crossings from the broad `kappa0 = 0.340` curves:

```text
L pair   kappa_cross
16-24    0.339098
24-32    0.340935
16-32    0.340123
```

These crossings bracket the expected `kappa_cr ~ 0.34`.

## Files

- `scripts/phi4_lambda1_cluster_scan.py`: standalone cluster/reweighting scan.
- `scripts/plot_phi4_lambda1_ising_limit.py`: plot and summary extraction.
- `run_all.sh`: regenerates the broad scan, refined scan, and summary plot.
- `outputs/phi4_lambda1_l16_l24_l32_chi_binder.png`: headline plot.
- `outputs/phi4_lambda1_l16_l24_l32_chi_binder.json`: extracted peak/crossing summary.
