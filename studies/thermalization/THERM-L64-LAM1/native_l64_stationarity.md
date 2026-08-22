# Native L64 stationarity audit
- **Date:** 2026-08-21
- **Reference:** `REF-PHI4-L64-K340301`; canonical \(L=64\), \(\lambda=1\), \(\kappa=0.340301\), \(N=5000\).
- **Generator:** one embedded Wolff sign cluster plus radial heat-bath sweep per update; 500 warm-up sweeps, followed by saved configurations every 15 sweeps.
- **Scope:** only indices 0--4999 are used. These are exactly the saved reference ensemble; later partial-append log rows are excluded.

## Sequential blocks
| Segment | generator sweeps | \(S/V\) | \(\chi\) | \(U_4\) | \(\xi/L\) |
|---|---:|---:|---:|---:|---:|
| full | 515--75500 | -0.54991(38) | 1220.0(73) | 0.6074(16) | 0.881(11) |
| first_half | 515--38000 | -0.54953(53) | 1216(11) | 0.6063(23) | 0.877(15) |
| second_half | 38015--75500 | -0.55029(54) | 1224(10) | 0.6085(22) | 0.884(15) |
| block_1 | 515--15500 | -0.54992(85) | 1221(16) | 0.6056(36) | 0.886(24) |
| block_2 | 15515--30500 | -0.54918(84) | 1210(16) | 0.6042(37) | 0.872(24) |
| block_3 | 30515--45500 | -0.54953(83) | 1217(16) | 0.6107(33) | 0.876(24) |
| block_4 | 45515--60500 | -0.55103(87) | 1229(16) | 0.6068(35) | 0.878(24) |
| block_5 | 60515--75500 | -0.54988(85) | 1219(17) | 0.6099(34) | 0.891(25) |

## Autocorrelation
The integrated autocorrelation time \(\tau_\mathrm{int}\), in *saved configurations*, is:
- `m2`: 0.689 saved configurations = 10.3 generator sweeps; first lags 0.139, 0.053, -0.003.
- `m4`: 0.689 saved configurations = 10.3 generator sweeps; first lags 0.142, 0.056, 0.002.
- `G(pmin)`: 0.533 saved configurations = 8.0 generator sweeps; first lags 0.027, 0.017, -0.011.
- `S/V`: 0.809 saved configurations = 12.1 generator sweeps; first lags 0.212, 0.066, 0.034.

## Interpretation
The first- and second-half shifts are \(\Delta U_4=+0.00219\) and \(\Delta(\xi/L)=+0.00653\), respectively. Their independent-half uncertainties are \(\sqrt{\sigma_1^2+\sigma_2^2}=0.00317\) and 0.02184, hence 0.69 and 0.30 standard deviations.
The five consecutive blocks fluctuate around the full-sample values without a monotonic drift. Together with the roughly 8--12-sweep autocorrelation scales and 500-sweep warm-up, this gives no evidence that the low central \(U_4\) and \(\xi/L\) values arise from incomplete thermalization. They remain compatible with finite-statistics fluctuation and/or the known small shift of this finite-volume coupling from the infinite-volume critical point.
