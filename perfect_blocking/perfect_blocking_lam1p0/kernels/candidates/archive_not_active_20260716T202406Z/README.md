# Archived Lambda 1.0 Candidate Kernels

Created UTC: `20260716T202406Z`

These candidate kernels were moved out of the active candidate area to keep lambda=1.0 upscaling tests focused on the retained best kernels.

Archived contents include:

| item | reason |
|---|---|
| `7x7_from_retrained_5x5/` | Previous unconstrained 7x7 branch; superseded after NN regression was identified. |
| `7x7_full_retraining_controlled/` | Broad retraining branch that did not beat the current final under guardrails. |
| `7x7_full_retraining_phi2_constrained/` | Diagnostic branch; improved phi2 but traded off other local observables. |
| `7x7_full_retraining_phi2_phi4_constrained/` | Useful but superseded by the phi2+NN-guarded candidate for upscaling priority. |
| `7x7_full_retraining_phi2_priority/` | Useful but NN sat on the acceptance edge; superseded by phi2+NN-guarded branch. |
| `7x7_search/` | Initial 7x7 exploration branch; superseded. |
| legacy/top-level 3x3 and 5x5 JSON files | Historical or provisional candidates; not active upscaling targets. |

Active candidate directories retained outside this archive:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/systematic_training/
perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/7x7_no33_nn_constrained/
perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/7x7_full_retraining_phi2_nn_guarded/
```

Stable upscaling paths are under:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/
```
