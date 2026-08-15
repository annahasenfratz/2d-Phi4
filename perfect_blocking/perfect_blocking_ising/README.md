# Perfect Blocking Ising

This subproject searches for a stochastic 2D Ising blocking rule that maps
fine `16x16` critical configurations to coarse `8x8` configurations whose
observables match the true critical `8x8` ensemble as closely as possible.

## What it does

- locates bundled Ising demo data if available
- generates critical Ising reference ensembles when needed
- optimizes a centered `3x3` stochastic blocking stencil
- validates the optimized kernel against the true coarse ensemble
- writes summary tables and plots to `perfect_blocking_ising/outputs/`

## Run

```bash
../.venv/bin/python -B perfect_blocking_ising/scripts/optimize_perfect_blocking.py
```

## Outputs

- `perfect_blocking_ising/outputs/perfect_blocking_summary.json`
- `perfect_blocking_ising/outputs/perfect_blocking_report.md`
- `perfect_blocking_ising/outputs/perfect_blocking_observables.csv`
- `perfect_blocking_ising/outputs/perfect_blocking_plots.pdf`

