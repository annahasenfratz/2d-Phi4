# CASCADE-WOLFF-LAM1 comparison tables

## First level: L16 to L32

`WOLFF-CASCADE-L32-001` applies the fixed Ethan 7x7 flow to the first 1500
direct native L16 configurations, then rethermalizes the resulting L32 fields
with Wolff+radial updates at \(\kappa_c=\kappa_f=0.340301\). One sweep is one
radial heat-bath update of every site plus four embedded Wolff sign-cluster
updates per configuration.

The independent native L32 reference (`REF-PHI4-L32-K340301`, \(N=10000\)) is
\(S/V=-0.554539(501)\), \(\chi=366.818(1.551)\),
\(U_4=0.608085(1105)\), and \(\xi/L=0.891760(7739)\).

| Sweep | \(S/V\) | \(\chi\) | \(U_4\) | \(\xi/L\) |
|---:|---:|---:|---:|---:|
| 0 | -0.532651(1410) | 369.263(4.150) | 0.604276(3070) | 0.892620(20600) |
| 25 | -0.553768(1250) | 362.072(4.110) | 0.606709(2930) | 0.865575(19600) |
| 50 | -0.555493(1250) | 369.698(3.930) | 0.612325(2690) | 0.907828(20700) |
| 75 | -0.554334(1260) | 362.759(3.980) | 0.606950(2850) | 0.881021(19400) |
| 100 | -0.555816(1230) | 367.638(4.010) | 0.607614(2860) | 0.886443(19500) |

The large sweep-zero local-action discrepancy is gone by sweep 25. At the
current \(N=1500\) precision, sweep-25--100 values are compatible with the
independent native L32 reference. The underlying machine-readable values and
uncertainty methods are in [`observables.csv`](observables.csv).

## Second level: L32 to L64

`WOLFF-CASCADE-L64-001` applies the same fixed Ethan 7x7 flow to the completed
L32 cascade field and then rethermalizes the L64 field at
\(\kappa_c=\kappa_f=0.340301\). The native L64 reference
(`REF-PHI4-L64-K340301`, \(N=5000\)) is
\(S/V=-0.549909(410)\), \(\chi=1220.0(7.9)\),
\(U_4=0.60745(17)\), and \(\xi/L=0.88073(112)\).

| Sweep | \(S/V\) | \(\chi\) | \(U_4\) | \(\xi/L\) |
|---:|---:|---:|---:|---:|
| 0 | -0.527771(760) | 1244.67(13.51) | 0.608508(2796) | 0.887188(19468) |
| 25 | -0.550474(692) | 1221.05(13.26) | 0.607667(2897) | 0.877682(20080) |
| 50 | -0.550254(672) | 1237.00(13.01) | 0.611959(2651) | 0.904853(20240) |
| 75 | -0.550940(698) | 1260.65(13.05) | 0.613539(2559) | 0.947528(21089) |
| 100 | -0.550034(648) | 1229.42(13.14) | 0.611316(2797) | 0.896951(20784) |

The local action-density discrepancy falls from roughly \(26\sigma\) at
sweep zero to below \(1\sigma\) by sweep 25. At sweeps 25, 50, and 100, all
four tabulated observables are compatible with the independent native L64
reference at the current precision. Sweep 75 has a simultaneous upward
long-distance fluctuation (about \(2.7\sigma\) in \(\chi\) and
\(2.8\sigma\) in \(\xi/L\)) that returns by sweep 100, so it is not evidence
of a persistent thermalization drift. The mean signed-cluster fraction is
0.338 per cluster, or 1.352 summed over the four clusters in one sweep.
