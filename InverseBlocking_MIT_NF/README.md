# InverseBlocking_MIT_NF

Project scaffold for combining the MIT normalizing-flow \(\phi^4\) tutorial with inverse blocking at finite \(\lambda\).

Core idea:

1. Generate coarse \(8^2\) configurations at \((\kappa_c,\lambda)\).
2. Upscale coarse fields in momentum space to \(16^2\), apply the inverse finite-\(\lambda\) blocking kernel, and transform back to position space.
3. Place the upscaled field on the even-even sublattice and keep those sites fixed.
4. Train a conditional MIT-style normalizing flow to generate the missing sites \(d\sim q_\theta(d\mid \phi_{ee})\).
5. Tune by inverse KL / reverse-KL against the fine \(\phi^4\) action at \((\kappa_f,\lambda)\), with optional reweighting/MH diagnostics.

The actual experiment must remain conditional generative sampling. Regression-only tests are allowed only for coordinate, shape, kernel, and algebra checks.

## MIT-to-our normalization

MIT tutorial action in two dimensions:

\[
S_{MIT}(\varphi)=\sum_x\left[(M^2+4)\varphi_x^2 + \lambda_{MIT}\varphi_x^4\right]
-2\sum_{x,\mu}\varphi_x\varphi_{x+\hat\mu}.
\]

Our finite-\(\lambda\) Ising/\(\phi^4\) normalization:

\[
S_{ours}(\phi)=-2\kappa\sum_{x,\mu}\phi_x\phi_{x+\hat\mu}
+\sum_x\left[\phi_x^2+\lambda(\phi_x^2-1)^2\right].
\]

With \(\varphi=\sqrt{\kappa}\,\phi\), constants ignored:

\[
M^2 = \frac{1-2\lambda}{\kappa}-4,
\qquad
\lambda_{MIT}=\frac{\lambda}{\kappa^2}.
\]

Thus MIT's tutorial point \(M^2=-4,\lambda_{MIT}=8\) corresponds to our \((\lambda,\kappa)=(0.5,0.25)\).

## Starting files

- `src/invblock_mit_nf/actions.py`: \(\phi^4\) action in our normalization plus MIT conversion.
- `src/invblock_mit_nf/blocking.py`: finite-\(\lambda\) momentum-space inverse-kernel machinery.
- `src/invblock_mit_nf/conditional_flow.py`: checkerboard conditional affine coupling layers with fixed even-even sites.
- `src/invblock_mit_nf/train_inverse_kl.py`: reverse-KL training loop skeleton.
- `notebooks/mit_phi4_kappa_lambda_conditional.ipynb`: notebook port restricted to \(\phi^4\) with \((\kappa,\lambda)\) inputs.
- `docs/project_plan.md`: detailed staged plan and diagnostics.
- `kernels/finite_lambda_kernel_template.json`: kernel placeholder to be filled from finite-\(\lambda\) perfect-blocking optimization.

## Finite-lambda kernel provenance

The preferred inverse-blocking kernel is now included as
`kernels/finite_lambda_lam1_L32_to_L16_5x5_KL.json`.  It is the 5x5 KL-optimized finite-lambda kernel shape from the prior blocking work at lambda=1.0, kappa≈0.3401, eta=1/4. The listed coefficients are rounded working values; only the unit-sum normalization is exact. The older 3x3 kernels are also saved for comparison.

See `docs/finite_lambda_kernels.md` for the shell parameters and normalization.
