# Calibrated Empirical Submit Workflow

The lambda=1.0 default sweep-zero initializer is the calibrated empirical
joint-detail upscaler in
`src/perfect_blocking_upsampling/empirical_joint_detail_upscaler.py`, governed
by `configs/lam1p0/calibrated_empirical_upscaler.json`.

It samples aligned joint 2x2 `D01,D10,D11` blocks from a fixed 1000-configuration
native-L32 donor bank using 7x7 coarse contexts, cKDTree exact nearest-context
lookup, `k=8`, `tau=q25`, and `beta=0.01` diagonal 12D kernel noise. It then
uses exactly one radial latent per configuration:

```
D01,D10 <- 0.97 * exp[(0.32/Lc) z] * D01,D10,  z ~ N(0,1)
D11       <- D11
```

The initializer preserves the supplied coarse field to the configured tolerance.
Its empirical density is not part of detail or coarse patch acceptance. The
existing target-action detail and coarse-plus-detail updates are unchanged.

New L16-to-L32 run:

```bash
perfect_blocking_upsampling/scripts/submit_flow_detail_coarse_detail \
  --preset lam1p0_L16to32_balanced --execute \
  --set n_chains=128 --set n_sweeps=100
```

Extend an existing run without generating details again:

```bash
perfect_blocking_upsampling/scripts/submit_extend_flow_detail \
  --run-dir perfect_blocking_upsampling/outputs/controlled_patch_lam1p0/coarse_detail_L16to32/<run_id> \
  --target-sweeps 200 --execute
```

The extension runner reads `checkpoint_latest.npz`, restores its stored RNG
state, and appends sweep rows. Legacy flow-initialized runs remain supported.

## Stage-C Sweep-100 Endpoint

The direct-L16 Stage-C continuation was stopped at sweep 100. Coarse acceptance
remained useful and details stayed near 0.39 acceptance, but the direct-coarse
width modes relaxed only weakly: fine `phi2`/`phi4` width ratios were about
0.85/0.84 and coarse `phi2`/`phi4` about 0.90/0.85. `m2` and `G_pmin` widths
remained about 1.08 and 1.20. This supports a slow coarse mismatch rather than
a failure of the detail updater; no continuation beyond sweep 100 was launched.
