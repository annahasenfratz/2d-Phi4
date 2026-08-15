# 2D $\phi^4$ inverse blocking

Research code for two-dimensional $\phi^4$ simulations, blocking kernels, inverse-blocking proposals, and rethermalization studies.

This repository tracks source code, launch scripts, lightweight configuration, selected kernels, and documentation.  Generated configurations, model checkpoints, plots, and bulk run products are deliberately excluded and should be protected by the separate data-backup workflow.

The main components are:

- `perfect_blocking_upsampling/`: the current upsampling and rethermalization workflow.
- `perfect_blocking/`: blocking-kernel development and validation utilities.
- `InverseBlocking_MIT_NF/`: MIT normalizing-flow inverse-blocking experiments.
- `inverse_blocking_flow/`: earlier inverse-RG flow prototypes.
- `phi4_phase-diagram/`: phase-diagram generation and analysis tools.

Use the shared Python environment supplied by the surrounding research workspace rather than a project-local virtual environment.
