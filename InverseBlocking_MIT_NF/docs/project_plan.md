# Inverse blocking + MIT NF plan

## Phase 0: lock conventions

Use our action

\[
S=-2\kappa\sum_{x,\mu}\phi_x\phi_{x+\hat\mu}+\sum_x[\phi_x^2+\lambda(\phi_x^2-1)^2].
\]

The MIT flow code is used only as an architecture/training template. Inputs are always \((\kappa,\lambda)\) in our normalization.

Conversion for comparisons:

\[
M^2=(1-2\lambda)/\kappa-4,\quad \lambda_{MIT}=\lambda/\kappa^2.
\]

MIT tutorial point: \(M^2=-4,\lambda_{MIT}=8\Rightarrow \lambda=0.5,\kappa=0.25\).

## Phase 1: identify finite-lambda blocking kernel

Use the finite-\(\lambda\) perfect-blocking output. The prior diagnostic comparison favored the 5x5 KL-optimized kernel over the 3x3 kernel on the same L32->L16 subset:

- 3x3: \(D_{op}\approx 12.07\)
- 5x5: \(D_{op}\approx 6.89\)

Therefore start with the 5x5 KL kernel unless a later lambda-specific run supersedes it. Fill `kernels/finite_lambda_kernel_template.json` with its stencil and record the source output path and normalization.

Required kernel checks:

1. Fourier symbol has no near-zero values on the coarse Brillouin zone.
2. `block(inverse_upscale(phi_c))` reconstructs `phi_c` to numerical precision.
3. Position-space embedding puts only the inferred field on even-even sites.
4. Additive/detail variables do not alter even-even sites.

## Phase 2: generate coarse ensemble

Generate \(8^2\) configurations at \((\kappa_c,\lambda)\). Use an ordinary reference sampler first, not the NF, so the conditional-flow test is not contaminated by coarse-model errors.

Diagnostics:

- plaquette/NN equivalent
- \(m^2,m^4\), Binder cumulant
- \(\xi/L\) if practical
- autocorrelation estimates

## Phase 3: inverse-kernel uplift

For each coarse configuration:

1. FFT on \(8^2\).
2. Divide by finite-\(\lambda\) kernel symbol \(K(q)\).
3. IFFT to infer an even-even field.
4. Embed on \(16^2\) even-even sites.

This is the condition \(c\), not the final generated fine field.

## Phase 4: conditional MIT-style NF

Train a conditional flow for missing sites:

\[
d \sim q_\theta(d\mid c),\quad \phi_f = c_{ee}+d_{rest},\quad \phi_f|_{ee}=c.
\]

The fixed even-even mask must be enforced algebraically after every coupling layer.

Loss:

\[
D_{KL}(q_\theta || p_f) = E_q[\log q_\theta(\phi_f|c)+S_f(\phi_f)] + const.
\]

Use inverse KL first. Track ESS from \(\log w=-S_f-\log q\), but do not use ESS alone as success.

## Phase 5: diagnostics and acceptance

Compare generated fine configurations to direct \(16^2\) reference at \((\kappa_f,\lambda)\):

- fixed even-even reconstruction exactness
- action density and components
- \(m^2,m^4\), Binder
- NN, diagonal, 2NN and squared versions
- \(\xi/L\)
- log-weight variance and ESS/N
- optional independence MH acceptance

## Non-regression rule

This project is conditional generative inverse blocking. Do not replace it by supervised prediction of missing sites. Regression losses may be used only as diagnostics or warm-start checks, never as the final sampling objective.
