# Continuous-patch quarantine — 2026-08-04

The contiguous-patch MH experiments are retained for provenance but are not
valid candidates for future production thermalization.  They update local
real-space squares rather than the checkerboard-coordinate blocks validated
on 2026-07-24.

The following output roots are moved intact to:

```text
perfect_blocking_upsampling/outputs/quarantine_contiguous_patch_20260804/
```

- `controlled_patch_lam1p0/coarse_detail_L16to32`
- `controlled_patch_lam1p0/coarse_detail_L32to64`

The explicit L8 -> L16 continuous-patch tests are moved to:

```text
perfect_blocking/perfect_blocking_lam1p0/tests/quarantine_contiguous_patch_20260804/
```

This is a relocation, not deletion.  The independent checkerboard-coordinate
outputs under `outputs/controlled_patch_lam1p0/mit_coordinate_mh_*` remain
active historical references.
