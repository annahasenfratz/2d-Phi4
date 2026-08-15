# Critical Ensemble Summary

This file summarizes phi4 scan outputs that contain at least both `L=16` and `L=32` runs. The machine-readable version is [`critical_ensembles.csv`](critical_ensembles.csv).

`kappa_cr` uses the mean of stored linear Binder crossings when those exist. If a summary has no stored crossings, the script attempts a direct `L=16`/`L=32` Binder crossing from the curve CSV, then falls back to a linear `1/L` extrapolation of the abs-centered susceptibility peak locations. For raw cluster sample sets, the same peak extrapolation is used when no Binder curves are available. Binder cumulants at `kappa_cr` are linear interpolations from available refined curves and averaged over reweighting centers for each volume.

| lambda | kappa_cr | method | Binder U4 L16 | Binder U4 L24 | Binder U4 L32 | chi peak kappa L16 | chi peak kappa L24 | chi peak kappa L32 | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.010000 | 0.261115 | linear_binder_mean | 0.515663 | 0.520531 | 0.511079 | 0.260791 | 0.260894 | 0.260986 | [`2026-06-09-phase-diagram-lambda-0p01/outputs/phi4_lambda001_l16_l24_l32_chi_binder.json`](../2026-06-09-phase-diagram-lambda-0p01/outputs/phi4_lambda001_l16_l24_l32_chi_binder.json) |
| 0.010000 | 0.261115 | linear_binder_mean | 0.515663 | 0.520531 | 0.511079 | 0.260791 | 0.260894 | 0.260986 | [`2026-06-10-phase-diagram-lambda-0p01/outputs/phi4_lambda001_l16_l24_l32_chi_binder.json`](../2026-06-10-phase-diagram-lambda-0p01/outputs/phi4_lambda001_l16_l24_l32_chi_binder.json) |
| 0.022000 | 0.270500 | chi_abs_peak_1_over_L_extrapolation |  |  |  | 0.268500 |  | 0.269500 | 6 raw sample metadata files |
| 0.100000 | 0.302724 | linear_binder_mean | 0.584987 | 0.588265 | 0.595739 | 0.299049 | 0.300012 | 0.300269 | [`2026-06-09-phase-diagram-lambda-0p1-2/outputs/phi4_lambda01_l16_l24_l32_chi_binder.json`](../2026-06-09-phase-diagram-lambda-0p1-2/outputs/phi4_lambda01_l16_l24_l32_chi_binder.json) |
| 0.100000 | 0.302724 | linear_binder_mean | 0.584987 | 0.588265 | 0.595739 | 0.299049 | 0.300012 | 0.300269 | [`2026-06-09-phase-diagram-lambda-0p1/outputs/phi4_lambda01_l16_l24_l32_chi_binder.json`](../2026-06-09-phase-diagram-lambda-0p1/outputs/phi4_lambda01_l16_l24_l32_chi_binder.json) |
| 0.500000 | 0.342570 | linear_binder_mean | 0.595082 | 0.597036 | 0.594390 | 0.334227 | 0.337025 | 0.338434 | [`2026-06-09-phase-diagram-lambda-0p5-2/outputs/phi4_lambda05_l16_l24_l32_chi_binder.json`](../2026-06-09-phase-diagram-lambda-0p5-2/outputs/phi4_lambda05_l16_l24_l32_chi_binder.json) |
| 0.500000 | 0.342570 | linear_binder_mean | 0.595082 | 0.597036 | 0.594390 | 0.334227 | 0.337025 | 0.338434 | [`2026-06-09-phase-diagram-lambda-0p5/outputs/phi4_lambda05_l16_l24_l32_chi_binder.json`](../2026-06-09-phase-diagram-lambda-0p5/outputs/phi4_lambda05_l16_l24_l32_chi_binder.json) |
| 0.500000 | 0.326873 | direct_L16_L32_binder_crossing | 0.368908 |  | 0.160850 | 0.337641 |  | 0.339643 | [`runs/lambda0p5_L16_L32_basic/outputs/phi4_lambda05_L16_L32_broad.json`](../runs/lambda0p5_L16_L32_basic/outputs/phi4_lambda05_L16_L32_broad.json) |
| 0.500000 | 0.342403 | direct_L16_L32_binder_crossing | 0.591720 |  | 0.591192 | 0.334334 |  | 0.338374 | [`runs/lambda0p5_L16_L32_basic/outputs/phi4_lambda05_L16_L32_refined.json`](../runs/lambda0p5_L16_L32_basic/outputs/phi4_lambda05_L16_L32_refined.json) |
| 1.000000 | 0.339628 | linear_binder_mean | 0.600951 | 0.603252 | 0.601444 | 0.328537 | 0.332351 | 0.334460 | [`2026-06-09-phase-diagram-lambda-1-2/outputs/phi4_lambda1_l16_l24_l32_chi_binder.json`](../2026-06-09-phase-diagram-lambda-1-2/outputs/phi4_lambda1_l16_l24_l32_chi_binder.json) |
| 1.000000 | 0.339628 | linear_binder_mean | 0.600951 | 0.603252 | 0.601444 | 0.328537 | 0.332351 | 0.334460 | [`2026-06-09-phase-diagram-lambda-1/outputs/phi4_lambda1_l16_l24_l32_chi_binder.json`](../2026-06-09-phase-diagram-lambda-1/outputs/phi4_lambda1_l16_l24_l32_chi_binder.json) |

Regenerate with:

```bash
../.venv/bin/python -B phi4_phase-diagram/src/summarize_critical_ensembles.py
```
