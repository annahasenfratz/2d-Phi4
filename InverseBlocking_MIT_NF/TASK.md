# Codex task: inverse blocking MIT NF

Implement and run the finite-lambda inverse-blocking MIT normalizing-flow pilot.

Hard constraints:

- Keep the project conditional generative: sample missing variables d ~ q_theta(d | phi_ee).
- Do not replace the experiment by deterministic supervised regression.
- Save checkpoints, partial metrics, and error/fix logs during all long runs.
- Run preflight I/O tests before long jobs.
- Use our (kappa, lambda) normalization everywhere in configs and reports.

Immediate tasks:

1. DONE: finite-lambda kernels recovered from the prior chat and saved under `kernels/`; preferred start is `finite_lambda_lam1_L32_to_L16_5x5_KL.json`. Verify against original output JSON if available.
2. DONE: added `tests/test_kernels.py` for kernel normalization and forward-blocking reconstruction after inverse upscaling.
3. Generate a small 8^2 coarse reference ensemble at (kappa_c, lambda) and save it under `outputs/coarse_8/`.
4. Convert the coarse ensemble to fixed even-even 16^2 conditions using `momentum_inverse_upscale_to_even_even`.
5. Train `ConditionalPhi4Flow` with reverse-KL loss against the fine 16^2 action.
6. Save checkpoints every `checkpoint_every` steps and write a CSV metrics log.
7. Compare generated 16^2 observables to a direct fine 16^2 reference.

Success criteria for the pilot:

- even-even sites are exactly fixed after sampling;
- loss decreases without numerical instability;
- log-weight variance is finite and tracked;
- generated operator table is closer to direct fine reference than the bare inverse-kernel uplift;
- if ESS/N is high but operators disagree, call it a failure, not a success.
