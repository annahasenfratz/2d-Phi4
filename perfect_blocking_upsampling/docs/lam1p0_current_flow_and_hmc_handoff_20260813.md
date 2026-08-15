# Lambda=1 current flow selection and fine-HMC thermalization

This is the active method after retiring the Gaussian/coordinate-MH studies.
The retired runs are retained, not deleted, in
`perfect_blocking_upsampling/quarantine_legacy_gaussian_mh_20260813/`.

## Current deployed flow/kernel pair

All present volume-scaling chains use the same volume-independent L16-to-L32
conditional RQ-spline flow and the same eta-included 5x5 blocking kernel:

```text
flow:
runs/lam1p0/training/
lam1p0_L16to32_alternatingKL_highcorr5_r1_iteration5/
flow5/stage_oo/checkpoints/checkpoint_best_nll.pt

kernel:
perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/
alternatingKL_L16to32_highcorr5_r1_iteration5/iteration5.json
```

The kernel is D4-symmetric, has the eta-scaled sum rule, and is well
conditioned: `min |K| = 0.6243`, `max |K| = 1.1827`, condition number `1.894`,
and `max |K^{-1}| = 1.602`.

## How the flow/kernel were chosen

1. The blocking kernel was fitted with a correlated, covariance-weighted
   operator objective and a Fourier-conditioning penalty.  The operator basis
   included local terms and selected correlations; the conditioning penalty
   prevents an apparently excellent-but-nonlocal inverse kernel.
2. For a fixed kernel, the flow is trained by conditional maximum likelihood
   (negative log likelihood, NLL) of the three unresolved detail fields given
   the coarse field.  The EO, OE, OO sublattice stages are trained in sequence.
3. Kernel and flow were alternated.  Candidate pairs were compared by the
   global A/R log-weight diagnostic

   ```text
   log w = S_f(phi) - S_c(coarse) + log|det K| - log q_theta(details | coarse),
   ```

   together with raw generated-observable histograms and support/tail checks.
   NLL alone is not sufficient: it can improve paired reconstruction while
   worsening the direct-coarse generated distribution or subsequent
   thermalization.
4. Iteration 5 was retained; iteration 6 was tested but was not a clear
   improvement.  In the 5000-sample global diagnostic, iteration 4 -> 5 gave
   a modest improvement: log-weight standard deviation `4.333 -> 4.206`
   (2.9% lower), ESS/N `0.00290 -> 0.00341` (18% higher), and independence-MH
   acceptance proxy `0.00861 -> 0.00978` (14% higher).  Iteration 6 reduced
   the width further to `4.103`, but ESS/N fell to `0.00265` and the proxy to
   `0.00891`; it was therefore not promoted.

This is a real but **modest** flow-pair improvement, not a guarantee of
near-unity global independence acceptance.  The old/original flow branches
are kept in quarantine for histogram and A/R comparisons.  The most useful
surviving comparison summaries are in `outputs/global_ar_lam1p0/` and the
raw-upscaling diagnostic archives.

## Active thermalization: direct physical-field HMC

The flow is used **once only**, at sweep zero:

```text
coarse field + sampled details -> psi -> phi = K^{-1} psi.
```

After that, the Markov state is the fine physical field `phi`.  There are no
updates of latent `z`, no flow-density term, no coarse action, and no later
application of `K^{-1}`.  The target is exactly

```text
pi(phi) proportional to exp[-S_f(phi)].
```

For one HMC subtrajectory, choose one uniformly spaced residue class of sites,
draw Gaussian momenta on those active sites only, evolve the conditional
Hamiltonian

```text
H = S_f(phi) + 1/2 sum_active p_x^2
```

by ten leapfrog steps, and accept/reject with the full Hamiltonian difference.
Inactive field sites are held fixed.  Cycling over all residue classes makes
one sweep; hence every fine site is updated once per sweep.  This is exact
conditional HMC, with no blocking approximation in the transition kernel.

### Observed behavior

- L8 -> L16 -> L32: the L16 and L32 stages thermalized well with the direct
  fine-HMC update; the staged L8->L32 test used 50 L16 sweeps and 400 L32
  sweeps.
- L8 -> L16 -> L32 -> L64 with 100 sweeps per level had acceptance `92.20%`,
  `92.28%`, and `92.33%` at L16, L32, and L64 respectively, using fixed
  `16^2` active-site sublattices and `(epsilon, n_leapfrog) = (0.08, 10)`.
- At L128, changing from fixed `16^2` active classes (`divide=8`) to
  `divide=2` made the update practical: four `64^2` active-site trajectories
  per sweep.  Its 0--100 and 100--200 pieces had cumulative acceptance about
  `69.9%` and `69.8%`.  Ordered-chain lag-one correlations at sweep 200 were
  statistically consistent with zero, as expected because the 1500 starting
  chains are independent.

The old Gaussian random walks in inverse-blocking coordinates were much slower
and could show long drift even with apparently reasonable acceptance.  They
are not to be used for new production thermalization.

## Scaling and operational notes

- L128->L256 now streams the one-time flow initialization in batches to an
  on-disk memmap (`checkpoints/initialization_phi.npy`) and reports completed
  batches in `initialization_progress.json`.  HMC is also processed in
  independent chain batches; this is a memory optimization only.
- Direct L128 reference configurations are independently generated by radial
  heat-bath plus embedded Wolff sign clusters.  They are not initialized from
  the upscaling chain.  Until final `configs.npz` is written, live observables
  are in `data/configs_phi4_2d/lam1p0_kappac0p340301_L128/generation_log.csv`.
- Do not compare L128 output to an L64 native ensemble as a physics test.  The
  temporary nearest-volume proxy exists only to keep generic notebooks from
  failing before the direct L128 ensemble finishes.
