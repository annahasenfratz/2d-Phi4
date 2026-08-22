# Base highcorr 5x5 blocking-promotion audit

## Object under audit

This audit concerns only the frozen base high-correlation 5x5 kernel

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/
allL16_chi2_R2_corrW5000_highcorr_5x5_eta_included.json
SHA-256 d4bb395431c5547a9fa58b7128e4afb37f3612f1a6003d22a7c8566c753469ff
```

It does **not** concern the alternating-KL iteration-5 kernel or the pure-NLL
flow trained with it.  Those are separate artifacts and must never be called
the base highcorr kernel.

The reference is the currently promoted 5x5 kernel
`perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json`
(SHA-256 `424c7a6a65aa843a6685f69024fa76629c273c8e736dbbe1a8f02ccab55a85e3`).

## Audit protocol and proposed promotion gate

All checks use the first 5,000 configurations of each native ensemble at
`lambda=1`, `kappa=0.340301`, with the stored eta normalization applied once.
The kernel sum is checked to be `2^(eta/2)=1.0905077326652577`.

The L32-to-L16 test is an in-sample diagnostic because the highcorr kernel was
optimized using that scale.  The transfer tests are L64-to-L32 and L128-to-L64.
The proposed blocking gate is:

1. no zeros in `K(p)` and condition number below 2.5;
2. on both transfer volumes, KS at most 0.07 for action density, `phi2`,
   `phi4`, local kurtosis, NN, 2nn, diagonal correlator, `m2`, and `m4`;
3. retain the per-configuration CSVs, manifests, and plots needed to rerun
   every quoted statistic.

This gate was documented as part of this audit, rather than pre-registered
before the L128-to-L64 measurement; it therefore needs explicit approval
before being used as a permanent promotion rule.  Passing it makes a kernel
eligible for paired-flow training.  It does not by itself promote a new
production *kernel--flow pair*.

## Results

The base highcorr kernel has `cond(K)=1.89435`, so it passes the numerical
stability gate.

Its transfer KS values are:

| Observable | L64-to-L32 | L128-to-L64 |
|---|---:|---:|
| action density | 0.0430 | 0.0624 |
| phi2 | 0.0406 | 0.0574 |
| phi4 | 0.0498 | 0.0628 |
| local kurtosis | 0.0270 | 0.0442 |
| NN | 0.0176 | 0.0084 |
| 2nn | 0.0286 | 0.0280 |
| diagonal | 0.0230 | 0.0158 |
| m2 | 0.0196 | 0.0224 |
| m4 | 0.0196 | 0.0224 |

Under the proposed gate it passes the blocking check.  At L128-to-L64 it is especially
stronger than the present promoted kernel for local kurtosis (KS 0.0442 versus
0.2952).  Its principal regressions relative to the present kernel occur at
the smaller volumes in action density and `phi4`; both remain within the gate.

## Decision

**Approved as the next frozen kernel candidate for an exact paired-flow
retraining and raw-upscaling audit.  Not yet a production replacement.**

The next promotion decision must use a flow trained from scratch with this
exact kernel hash and must compare raw L16-to-L32 upscaling with the current
tail-stratified 5x5 flow on the same independent native-L16 input set.  The
result must include sweep-zero histograms and the subsequent exact-HMC
rethermalization test.

## Reproducibility artifacts

The L32-to-L16 and L64-to-L32 comparisons, including the Ethan and promoted
references, are retained under
`perfect_blocking/perfect_blocking_lam1p0/tests/intermediate/ethan7_vs_highcorr5_20260818/`.
The independent L128-to-L64 test is retained under
`perfect_blocking/perfect_blocking_lam1p0/tests/intermediate/highcorr5_promotion_audit_20260818/L128_to_L64/`.

The measurements were generated with `perfect_blocking/scripts/block_and_measure.py`
and compared with `perfect_blocking/scripts/test_kernel_observables.py`.
