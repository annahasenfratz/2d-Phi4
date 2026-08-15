I want to prototype a conditional inverse-blocking normalizing flow for 2D scalar phi4 theory.

Goal:
Given fine configurations generated from a known fine action S_fine on a (2L)x(2L) lattice, block them to LxL coarse configurations. Then train a conditional normalizing flow that reconstructs fine configurations from the coarse field plus Gaussian noise. The reconstructed fine ensemble should be corrected with a Metropolis accept/reject step targeting S_fine.

Please implement this as a small, modular prototype, not a large framework.

Definitions:

* Fine field: phi_f with shape (2L, 2L).
* Coarse field: phi_c with shape (L, L), obtained by 2x2 block averaging of phi_f.
* Detail variables: d with shape (3, L, L), representing the missing three degrees of freedom per 2x2 block.
* Gaussian prior noise: eta ~ N(0,1), same shape as d.
* Conditional flow: d = F_theta(eta; phi_c), invertible in eta for fixed phi_c.
* Reconstructed fine field: phi_rec = inverse_block(phi_c, d).

Use the following simple orthogonal 2x2 Haar-like parameterization inside each block:
Given the four fine fields in a block:
a = phi[2i,   2j]
b = phi[2i+1, 2j]
c = phi[2i,   2j+1]
e = phi[2i+1, 2j+1]

Define:
LL = (a+b+c+e)/2
HL = (a-b+c-e)/2
LH = (a+b-c-e)/2
HH = (a-b-c+e)/2

Then inverse:
a = (LL+HL+LH+HH)/2
b = (LL-HL+LH-HH)/2
c = (LL+HL-LH-HH)/2
e = (LL-HL-LH+HH)/2

For now, use LL as phi_c. Later I may add eta/anomalous-dimension normalization, but do not include it yet unless it is already in the codebase.

Tasks:

1. Find or create a small phi4 action module:
   S_fine(phi) = sum_x [ kinetic + m2 phi_x^2 + lambda phi_x^4 ],
   with periodic boundary conditions.
   Use the same convention already used in this repository if one exists.

2. Implement:

   * haar_block(phi_f) -> phi_c, d_true
   * haar_unblock(phi_c, d) -> phi_f
   * tests showing haar_unblock(*haar_block(phi_f)) reconstructs phi_f to numerical precision.

3. Build a conditional normalizing flow:

   * Input conditioning field: phi_c, shape (batch, 1, L, L).
   * Noise/detail field: eta or d, shape (batch, 3, L, L).
   * The flow should be invertible in d/eta for fixed phi_c.
   * Use affine coupling layers on the 3 detail channels, with CNNs that can see both the frozen detail channels and phi_c.
   * Use periodic/circular padding.
   * Return d, log_q(d | phi_c), and support reverse if convenient.

4. Training:
   Use paired data from fine configurations:
   phi_f -> phi_c, d_true.
   Train the conditional density q_theta(d_true | phi_c) by maximum likelihood first:
   loss = - mean log q_theta(d_true | phi_c).
   This is a supervised sanity check.

5. Then add self-training / reverse-KL option:
   sample eta ~ N(0,1)
   d = F_theta(eta; phi_c)
   phi_rec = haar_unblock(phi_c, d)
   loss = mean[ S_fine(phi_rec) + log_q(d | phi_c) ]
   Here phi_c is drawn from the blocked training set.
   Keep this as a separate training mode.

6. Add independence Metropolis accept/reject:
   Given a sequence of proposed phi_rec generated from phi_c sampled from the blocked set and d sampled from q_theta(d|phi_c), accept/reject with target S_fine.
   For the first prototype, assume phi_c is sampled empirically from the blocked fine ensemble, so its marginal proposal factor cancels if proposals are drawn from the same empirical pool. Include comments explaining that if phi_c later comes from an independent coarse action S_coarse, the acceptance ratio must include S_coarse(phi_c).

   Acceptance log ratio between current and proposal should use:
   log_target = -S_fine(phi)
   log_proposal = log q_theta(d | phi_c)
   if phi_c is drawn from the same empirical proposal distribution.
   So:
   logA = [-S_fine(phi_new) - logq_new] - [-S_fine(phi_old) - logq_old]

7. Diagnostics:

   * Reconstruction test from true details: exact.
   * Compare S_fine distributions for:
     true fine configs,
     raw Haar reconstruction using Gaussian details,
     flow-generated reconstructions before A/R,
     flow-generated reconstructions after A/R.
   * Compare simple observables:
     mean phi,
     mean phi^2,
     Binder cumulant if available,
     nearest-neighbor correlator,
     two-point susceptibility if already implemented.
   * Report acceptance rate.

8. Keep everything small and runnable:

   * default L = 8 or 16, so fine lattice is 16x16 or 32x32.
   * Use PyTorch.
   * Put code in a new subdirectory, e.g. inverse_blocking_flow/
   * Include a README.md with how to run:
     python train_conditional_flow.py --mode mle
     python train_conditional_flow.py --mode reverse_kl
     python sample_with_ar.py

9. Please do not over-engineer. I want a working proof of concept with clear tests and plots.

Use the kappa-lambda action convention:
S(phi) =
  -2*kappa * sum_{x,mu} phi_x phi_{x+mu}
  + sum_x phi_x^2
  + lambda * sum_x (phi_x^2 - 1)^2

with periodic boundary conditions. For the first test use:
  kappa = kappa_fine = 0.31
  lambda = 1.0
  fine_size = 32

Please make these parameters configurable from the command line, with defaults:
  --fine-size 32
  --lambda 1.0
  --kappa-fine 0.31
  --coarse-size inferred as fine_size // 2

  