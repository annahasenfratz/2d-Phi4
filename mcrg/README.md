# MCRG side analysis

This directory implements Swendsen's covariance method independently of production inverse-blocking code.  For every paired hierarchy level it measures extensive operators and solves `A T = B` by an SVD pseudoinverse, where `A=Cov(S^(n+1),S^(n+1))` and `B=Cov(S^(n+1),S^(n))`.  No explicit inverse is used.

The 7x7 branch imports `perfect_blocking.scripts.common.blocking.block_configs` and its selected eta-included kernel.  The matched 2x2 average obtains its normalization from that loaded kernel's matrix sum.  Both downsample at even sites; the 2x2 average spans `(2i,2i+1)x(2j,2j+1)`, whereas the odd-width perfect stencil is centered on `(2i,2j)`, an unavoidable half-site geometric distinction.

`analyze.py` bootstraps original configuration indices jointly at both levels, preserving their correlation.  It reports conditioning, singular values, eigenvectors, and nested bases.  The input and output configuration data are never modified.

The JSON also records unbootstrapped leading-eigenvalue sensitivity at several SVD relative cutoffs; results which vary across those cutoffs should be treated as ill-conditioned.  `plot_results.py` creates a PDF and compact Markdown table from one or more analysis JSON files.  Error bars and table uncertainties are paired-bootstrap 1σ standard deviations.

Run tests with:

```bash
../../.venv/bin/python -m pytest mcrg/tests -q
```

Pilot example:

```bash
../../.venv/bin/python -m mcrg.analyze --configs data/configs_phi4_2d/lam1p0_kappac0p340301_L128/configs.npz --kernel perfect7 --max-configs 500 --output mcrg/results/pilot_l128_perfect7.json
```
