# THERM-L64-LAM1 comparison tables

These tables are formatted views of the provenance-preserving measurements in
[`observables.csv`](observables.csv). The native L64 reference is
\(\chi=1220.0(7.9)\), \(U_4=0.60745(17)\),
\(\xi/L=0.88073(112)\), and \(S/V=-0.549909(410)\).

For the expanded common-sweep comparison, including local operators at Wolff
sweeps 10 and 20 and HMC sweep 250, see
[`requested_l32_to_l64_table.md`](requested_l32_to_l64_table.md) and its
LaTex source [`requested_l32_to_l64_table.tex`](requested_l32_to_l64_table.tex).

## Wolff+radial rethermalization

The runs are \(\kappa_c=\kappa_f=0.340301\), \(N=5000\), and
\(\kappa_c=0.340100\to\kappa_f=0.340301\), \(N=3000\). One sweep is one
radial heat-bath update plus four embedded Wolff sign-cluster updates per
configuration.

| Run | Sweep | \(\chi\) | \(U_4\) | \(\xi/L\) |
|---|---:|---:|---:|---:|
| same \(\kappa\) | 0 | 1243.7(7.3) | 0.61000(151) | 0.89758(1126) |
|  | 25 | 1234.6(7.4) | 0.60832(156) | 0.89741(1124) |
|  | 50 | 1232.4(7.0) | 0.61011(147) | 0.89483(1072) |
|  | 75 | 1232.3(7.4) | 0.60872(155) | 0.89756(1110) |
|  | 100 | 1231.6(7.2) | 0.61066(151) | 0.90043(1108) |
| \(0.340100\to0.340301\) | 0 | 1211.2(9.4) | 0.60340(210) | 0.86861(1329) |
|  | 25 | 1242.0(9.5) | 0.61043(192) | 0.91429(1471) |
|  | 50 | 1233.1(9.3) | 0.61003(192) | 0.89637(1391) |
|  | 75 | 1226.6(9.0) | 0.61079(186) | 0.89977(1380) |
|  | 100 | 1258.9(9.3) | 0.61361(176) | 0.93184(1460) |

The short-distance mismatch is corrected after approximately one update; by
sweeps 3--5 the local observables agree with native L64. The same-kappa run
is already close in long-distance quantities at sweep zero. For the
offcritical-source run, sweeps 25--75 are the conservative window: the
sweep-100 upward fluctuation is not by itself evidence of a drift.

## Global HMC rethermalization, matched fine target

Both runs target \(\kappa_f=0.340301\) and use the Ethan 7x7 flow. The
\(\kappa_c=0.340100\) values combine independent r1 and r2 inputs, giving
\(N=3000\).

| Source \(\kappa_c\) | Sweep | \(S/V\) | \(\chi\) | \(U_4\) | \(\xi/L\) |
|---|---:|---:|---:|---:|---:|
| 0.340301 | 0 | -0.52685(44) | 1244(7) | 0.60994(151) | 0.8973(114) |
|  | 20 | -0.54990(38) | 1230(7) | 0.61009(146) | 0.8934(111) |
|  | 50 | -0.55007(38) | 1227(7) | 0.61001(151) | 0.8967(112) |
|  | 100 | -0.55034(37) | 1236(7) | 0.60967(150) | 0.8981(112) |
|  | 500 | -0.55034(37) | 1241(7) | 0.61117(148) | 0.9173(116) |
| 0.340100 | 0 | -0.52544(55) | 1211(10) | 0.60332(209) | 0.8676(138) |
|  | 20 | -0.54914(47) | 1200(9) | 0.60465(201) | 0.8642(130) |
|  | 50 | -0.54931(50) | 1206(10) | 0.60411(210) | 0.8682(134) |
|  | 100 | -0.54891(49) | 1205(10) | 0.60527(208) | 0.8663(137) |
|  | 500 | -0.54891(48) | 1201(9) | 0.60367(210) | 0.8595(133) |

HMC brings \(S/V\) close to equilibrium by about 20 sweeps. It does not
demonstrate long-distance thermalization: at sweep 20 the two HMC inputs differ
by approximately \(2.6\sigma\) in \(\chi\), \(2.2\sigma\) in \(U_4\), and
\(1.7\sigma\) in \(\xi/L\); at sweep 500 the corresponding differences are
\(3.4\sigma\), \(2.9\sigma\), and \(3.3\sigma\).
