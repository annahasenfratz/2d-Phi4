# Phase Diagram Lambda=0.5

Standalone phi4 `lambda=0.5` finite-volume phase-diagram check, split out from
the Heidelberg phi4 reproduction thread.

## Model

Paper/action convention:

```text
S = sum_x [(1 - 2 lambda) phi_x^2 + lambda phi_x^4
           - 2 kappa sum_mu phi_x phi_{x+mu}]
lambda = 0.5
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

Broad scan centers: `kappa0 = 0.320, 0.340, 0.360`.

Refined susceptibility centers: `kappa0 = 0.335, 0.340`.

Abs-centered susceptibility peaks:

```text
L    kappa_peak   half-spread   chi_|m| max
16   0.33423      0.00014       11.79
24   0.33702      0.00001       23.93
32   0.33843      0.00010       39.78
```

Important: the susceptibility peak is still moving right with volume. Do not
interpret these finite-volume peaks as a final infinite-volume `kappa_c`.

Peak-height scaling:

```text
chi_|m|,max ~ L^p
p = 1.753
```

Binder crossings from the broad `kappa0 = 0.340` curves:

```text
L pair   kappa_cross
16-24    0.343278
24-32    0.341952
16-32    0.342482
```

The Binder crossings are around `0.342`, while the susceptibility peaks are
still moving right toward that region.

## Files

- `scripts/phi4_lambda05_cluster_scan.py`: standalone cluster/reweighting scan.
- `scripts/plot_phi4_lambda05_phase.py`: plot and summary extraction.
- `run_all.sh`: regenerates the broad scan, refined scan, and summary plot.
- `outputs/phi4_lambda05_l16_l24_l32_chi_binder.png`: headline plot.
- `outputs/phi4_lambda05_l16_l24_l32_chi_binder.json`: extracted peak/crossing summary.
