# Required Local Artifacts

The curated repository intentionally excludes generated ensembles, checkpoints,
and run products.  The MIT submit scripts require these local artifacts:

- native lambda=1.0 ensembles at L8, L16, L32, and L64 under
  `data/configs_phi4_2d/lam1p0_kappac0p340301_L*/configs.npz`;
- final kernel
  `perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json`;
- wrapped RQ-spline checkpoint
  `perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_newkernel_rqspline_N2000_action_lowtail_from_N2000_bestpatch_20260719T171401Z/checkpoints/checkpoint_best_patch.pt`;
- its run configuration, normalization metadata, and kernel metadata in the
  same training directory;
- either optional L8->L16 provenance checkpoint listed by
  `rsync_lam1p0_flow_remote.sh`, when checkpoint metadata requires it.

`rsync_lam1p0_flow_remote.sh` validates and transfers these artifacts for a
remote MIT diagnostic installation.  It never transfers `outputs/` or other
production run directories.

Historical local experiments may remain on a developer machine under ignored
paths, but they are not part of this collaboration package.
