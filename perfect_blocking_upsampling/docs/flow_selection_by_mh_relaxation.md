# Selecting Upscaling Flows by Exact-MH Relaxation

## Decision

For stochastic L16 -> L32 upscaling flows, **held-out conditional NLL and short
exact-MH relaxation take priority over raw generated-observable histograms**.
Histogram agreement is retained as a support/coverage guardrail, not as the
principal checkpoint-selection objective.

This rule applies equally when extending the method to L32 -> L64.

## Why

A flow can make one-dimensional raw histograms look excellent--including their
widths and tails--while being wrong in the joint conditional distribution of
the detail fields.  In particular, it can miss correlations among detail
fields, long-distance modes, or correlations between the blocked coarse field
and the details.  Those defects can be nearly invisible in the histograms of
action density, phi2, phi4, etc., but are exposed by an exact Metropolis-Hastings
chain through a long thermalization transient.

Thus

```
good observable marginals != close joint proposal distribution
valid MH kernel          != rapid thermalization
```

At criticality, the problem is amplified by slow long-wavelength modes.  The
error in a flow initializer is a combination of relaxation modes, not a single
``action offset``.  Different modes can have opposite contributions to the
action.  Consequently the action may cross the direct-native value, overshoot
it, and remain displaced for many sweeps even though individual raw histograms
look acceptable.

## Direct evidence from the July L16 -> L32 flow branch

The successful July-24 checkerboard MH geometry was held fixed:

- one flow draw at sweep zero only;
- then exact physical-field checkerboard MH with `divide=2`;
- coarse update: per-site transition reversible with exp(-S_c), followed by
  its S_f - S_c correction;
- detail update: physical d01, d10, d11 random walks, accepted with S_f;
- `coarse_sigma=0.40`, `detail_sigma=0.10`, two detail passes per sweep.

The July training run was

```
runs/lam1p0/training/
lam1p0_L16to32_newkernel_rqspline_N2000_action_lowtail_from_N2000_bestpatch_20260719T171401Z
```

It trained the autoregressive conditional detail factorization

```
q(d01 | c) q(d10 | c,d01) q(d11 | c,d01,d10)
```

on matched pairs made by blocking native L32 configurations to L16.  Its main
loss was conditional NLL, but training/selection also included observable and
tail/width coverage terms.  The two important retained checkpoints are:

| checkpoint | epoch | role in the original run |
|---|---:|---|
| `checkpoints/checkpoint_best_nll.pt` | 1 | lowest validation NLL |
| `checkpoints/checkpoint_best_patch.pt` | 2 | selected for patch/raw-observable diagnostics |

The July production-like runs used `checkpoint_best_patch.pt`.  Subsequent
efforts that emphasized raw histogram width, tails, and observable coverage
produced flows with much longer exact-MH relaxation, despite attractive raw
histograms.  The initial test of the retained `checkpoint_best_nll.pt` in
`L16toL32_N512_S400_start500_div2_D2_sc0p40_sd0p10_r4` looked substantially
better on the observed single initialization.  This is evidence, not yet a
high-statistics final ranking.

The historical July kernel must be used for an apples-to-apples comparison:

```
perfect_blocking/perfect_blocking_lam1p0/
kernel_retune_pc1_radial_mode_20260720/current_kernel.json
```

Its stored 5x5 matrix exactly matches the July successful runs.  It differs
from the later `kernels/final/chosen_kernel.json`, despite the earlier paths
having the same generic name/family in metadata.

## Required candidate-evaluation protocol

For every flow checkpoint candidate:

1. Evaluate held-out conditional NLL on matched blocked-native pairs.
2. Check raw generated histograms and support.  Reject clear support failure,
   pathological tails, or gross operator mismatches, but do not rank primarily
   by width/quantile matching.
3. Initialize the same fixed set of native coarse configurations with one
   independent detail draw per chain.
4. Run 50--100 sweeps of the exact checkerboard MH kernel, with all kernel and
   MH parameters fixed across candidates.
5. Compare time histories to the direct native L32 reference for at least:
   action density, phi2, phi4, NN, diag, 2nn, m2, G(pmin), Binder U4, and
   xi/L.  Include chi when it is part of the standard analysis.
6. Rank candidates by short-run relaxation, for example by the maximum or
   integrated squared standardized displacement from the direct-native mean
   across these observables and sweeps.  Record the random seed, coarse index
   range, kernel matrix, and exact checkpoint path in the run configuration.

Where feasible, add full-density diagnostics: held-out log probability,
importance-weight variability/effective sample size, and acceptance of a
global independence proposal constructed from the flow.  These probe joint
proposal quality more directly than one-dimensional histograms.

## Interpretation of the current MH runs

The long transient in the recent runs does **not** by itself imply an incorrect
MH acceptance formula.  With flow used only at sweep zero, the checkerboard
MH update targets the fine action exactly.  It indicates that the initializer
has appreciable overlap with slow modes of the target distribution.  Continuous
patch updates were separately quarantined because they mixed those modes even
less effectively; switching back to checkerboard updates restores the better
transition geometry but cannot by itself repair a poor initializer.

## Launcher support

`scripts/submit_lam1p0_checkerboard_mh.sh` defaults, for L16 -> L32, to the
historical July `best_patch` flow and the exact archived July kernel.  It also
accepts:

```bash
FLOW_VARIANT=nll bash scripts/submit_lam1p0_checkerboard_mh.sh rN --execute --background
```

which chooses the same branch's `checkpoint_best_nll.pt`.  An explicit
`FLOW_CHECKPOINT=/absolute/or/repo/relative/path` overrides this selector.
