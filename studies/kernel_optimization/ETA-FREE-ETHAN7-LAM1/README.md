# ETA-FREE-ETHAN7-LAM1: free-normalization refit of Ethan's 7x7 kernel

- **Status:** VALIDATED (non-promoting diagnostic)
- **Latest substantive update:** 2026-08-21
- **Scientific question:** Does allowing the anomalous-dimension exponent
  \(\eta\) to vary materially improve Ethan's 7x7 perfect-blocking kernel?
- **Source:** Ethan's operational 7x7 kernel,
  `kernels/selected_for_upscaling/ethan_7x7_paper_objective_eta_included.json`.
- **Training / validation:** 9000 L32 configurations blocked to L16, with
  4000 direct L16 reference configurations; 1000 held-out L32-to-L16 and 1000
  cross-volume L64-to-L32 checks. The exact indices are in the raw
  [`result.json`](../../../perfect_blocking/perfect_blocking_lam1p0/tests/intermediate/ethan_7x7_free_eta_mu30_N5_train9000_test1000_20260820/result.json).
- **Method:** D4-symmetric 7x7 shape with all ten orbit coefficients and
  \(\eta\) optimized by four Powell starts. The objective is the sum of squared
  standardized residuals for \(\phi^2,\phi^4,\phi^6\), NN, 2NN, diagonal,
  \(m^2\), \(G_{21},G_{22},G_{30},G_{31}\), plus \(30\) times the fraction of
  \(K^{-1}\) outside a centered 5x5 box.

## Result

The best start gives \(\eta=0.2518658\) and
\(2^{\eta/2}=1.0912131\), compared with the supplied \(\eta=0.25\) and
normalization 1.0905077. Thus freeing the normalization changes the scale by
only 0.065%; the other three starts give \(\eta=0.25047\)--0.25150. There is
no evidence here for a materially different eta.

The in-sample objective falls from 7.065 for the supplied kernel to 2.997:
the standardized-residual contribution falls from 1.644 to 0.700 and the
inverse-tail fraction from 0.18069 to 0.07656. This is accompanied by worse
conditioning: \(\min K\) falls from 0.54372 to 0.44446, the condition number
rises from 2.288 to 2.845, and \(\max|K^{-1}|\) rises from 1.839 to 2.250.

Held-out tests show no decisive generalization improvement. On L32-to-L16 the
free fit improves the \(\phi^2\) and \(\phi^4\) means but has comparable or
larger KS distances for several operators. On L64-to-L32, means and KS values
are modestly better for several local operators, but the \(\phi^2\) and
\(\phi^4\) width ratios become less favorable (direct/blocked 0.850 and
0.809, versus 0.854 and 0.816 for the supplied kernel). The change does not
justify replacing the established Ethan kernel or changing its fixed
\(\eta=0.25\) normalization for cascade production.

`observables.csv` preserves both same-split held-out reports, including means,
width ratios, and the histogram KS statistic used by the optimizer driver.
