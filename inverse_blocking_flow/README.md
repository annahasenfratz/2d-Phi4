# Conditional inverse-blocking flow prototype

This is a small proof of concept for reconstructing fine 2D scalar phi4
configurations from block-averaged coarse fields plus learned residual detail
variables.

The default action convention is

```text
S(phi) = -2 kappa sum_{x,mu} phi_x phi_{x+mu}
       + sum_x phi_x^2
       + lambda sum_x (phi_x^2 - 1)^2
```

with periodic boundary conditions. The default debugging case is:
`--fine-size 16 --lambda 1.0 --kappa-fine 0.31`, so the default coarse size is
`8`.

## Run

From the repository root, use the shared environment:

```bash
../.venv/bin/python -m pytest -m "not slow and not training and not scan"
make test-fast
make test-slow
make test-all
../.venv/bin/python inverse_blocking_flow/train_conditional_flow.py --mode mle
../.venv/bin/python inverse_blocking_flow/train_conditional_flow.py --mode reverse_kl
../.venv/bin/python inverse_blocking_flow/train_conditional_flow.py --mode mixed --mle-mix-alpha 0.5
../.venv/bin/python inverse_blocking_flow/train_conditional_flow.py --mode tempered_reverse_kl
../.venv/bin/python inverse_blocking_flow/sample_with_ar.py
```

`sample_with_ar.py` defaults to the reverse-KL checkpoint because that is the
proposal intended for the A/R correction. To test another proposal explicitly:

```bash
../.venv/bin/python inverse_blocking_flow/sample_with_ar.py --checkpoint inverse_blocking_flow/outputs_fine16/conditional_detail_flow_mle.pt --tag mle
```

To compare available MLE, reverse-KL, and mixed checkpoints with the same A/R
settings:

```bash
../.venv/bin/python inverse_blocking_flow/sample_with_ar.py --compare-defaults
```

To sanity-check the A/R bookkeeping and histogram code without the flow, propose
full fine configurations from a held-out half of the empirical fine ensemble:

```bash
../.venv/bin/python inverse_blocking_flow/sample_with_ar.py --ar-with-true-details --tag true_details
```

Patchwise detail Metropolis at fixed `phi_c` is available as a first
infrastructure test with independent Gaussian detail-patch proposals:

```bash
../.venv/bin/python inverse_blocking_flow/patchwise_detail_metropolis.py --patch-size 4 --n-sweeps 4
```

To produce a controlled inverse-RG quality diagnostic report without training:

```bash
../.venv/bin/python inverse_blocking_flow/analyze_inverse_rg_quality.py
```

Useful quick smoke settings:

```bash
../.venv/bin/python inverse_blocking_flow/train_conditional_flow.py --mode mle --n-configs 64 --epochs 1 --burn-in 10 --sample-interval 2 --batch-size 16
```

Default debugging outputs are written under `inverse_blocking_flow/outputs_fine16/`:

- `conditional_detail_flow_<mode>.pt`: model checkpoint.
- `diagnostics_<mode>.json`: reconstruction, inverse-flow, action, and observable summaries.
- `action_histograms_<mode>.pdf`: true, Gaussian-detail, and flow action distributions.
- `logq_histograms_<mode>.pdf`: true-detail versus generated-detail log-density histograms.
- `logw_histograms_<mode>.pdf`: true-detail versus generated-detail full log-weight histograms.
- `scatter_action_logq_<mode>.pdf`: `S_f` versus `log q` for true and generated details.
- `scatter_action_logw_proposals_<mode>.pdf`: proposal `S_f` versus `-S_f - log q`.
- `ar_diagnostics_<tag>.json` and `action_histograms_ar_<tag>.pdf`: independence Metropolis diagnostics.
- `ar_comparison_summary.json`: compact A/R comparison for multiple checkpoints.

## Diagnostic Classes

Algebra / code-integrity tests verify only the deterministic blocking and
reconstruction code:

- exact detail reconstruction
- `block_average(reconstruct_from_average_block(phi_c, d)) == phi_c`
- residuals from `detail_to_residual(d)` have zero block average

Learning / generative diagnostics sample fresh Gaussian `eta`, generate
`d = F_theta(eta | phi_c)`, reconstruct `phi_rec`, and compare action, `phi^2`,
Binder, nearest-neighbor correlator, correlators, and power spectra to the true
fine ensemble. These say whether the learned inverse map is useful as an
observable-level inverse RG model.

Exact-sampling diagnostics compute `logw = -S_f - logq`, ESS/N, and global A/R
acceptance. These say whether the flow is a usable exact independence proposal.
Good observable matching does not imply good exact-sampling performance.

## Conditional Scope

The current experiments are in the supervised/conditional setup. Fine
configurations are generated from `S_f`, coarse fields `phi_c` are obtained by
blocking those true fine configurations, and restricted detail MCMC samples
`P_f(d | phi_c)` at fixed `phi_c`.

This makes fixed-`phi_c` restricted MCMC valid for testing the conditional
reconstruction on the fiber `B(phi_f) = phi_c`. It is not yet a full sampler
with independently generated coarse fields.

If `phi_c` is generated from an approximate coarse action `S_c^approx`, then
fixed-`phi_c` detail MCMC corrects only the conditional detail distribution; it
does not correct the coarse marginal. A full algorithm will need a second stage:
match or tune `S_c` so `phi_c` has the correct blocked marginal, add
coarse-field updates/corrections, or include reweighting/global correction for
the coarse marginal.

## Notes

The deterministic kernel is explicit:

- `block_average(phi_f) -> phi_c`
- `prolong_constant(phi_c) -> phi0`
- `detail_to_residual(d) -> chi`, with zero average in every 2x2 fine block
- `reconstruct_from_average_block(phi_c, d) = prolong_constant(phi_c) + chi`
- `average_block(phi_f) -> phi_c, d_true`, where `d_true` comes from the true residual

The older `haar_block` and `haar_unblock` function names are kept as
compatibility aliases, but they now use the average/prolong/residual
parameterization.

The conditional flow is invertible in the three detail channels for fixed
`phi_c`. It uses affine coupling layers whose CNNs see both the frozen detail
channels and the coarse field, with circular padding.

The independence Metropolis step assumes `phi_c` is sampled uniformly from the
empirical blocked fine ensemble, so the marginal `phi_c` proposal factor cancels.
If `phi_c` is later proposed from an independent coarse action `S_coarse`, the
acceptance ratio must include the coarse proposal/target contribution.
