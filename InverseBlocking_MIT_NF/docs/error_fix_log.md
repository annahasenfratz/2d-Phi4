# Error / Fix Log

## 2026-06-22 smoke test aggregation bug

- `scripts/smoke_test.py` tried to compute `obs.mean(axis=0)` on a Python list of
  observable dictionaries.
- This raised `TypeError: unsupported operand type(s) for +: 'dict' and 'dict'`.
- Fix: aggregate each observable key explicitly across the batch.

## 2026-06-22 conditional preflight dtype alignment

- The preflight scripts use float64 coarse tensors from NumPy.
- The conditional flow must be cast to double precision to avoid dtype mismatch
  against the convolution weights.
- Fix: instantiate `ConditionalPhi4Flow(...).double()` in the preflight path.

## 2026-06-22 fine-to-coarse blocking dimension mismatch

- The initial forward blocking helper tried to FFT the full 16x16 fine field with
  an 8x8 kernel symbol, which raised a size mismatch.
- Fix: block the even-even sublattice first, then apply the coarse kernel symbol
  on the resulting 8x8 field.

## 2026-06-22 diagnostics summary missing finite flag

- The first physics-diagnostics pass reached the summary stage but failed when it
  tried to read `summaries["inverse_kernel"]["finite"]`.
- Fix: include the `finite` field in the per-ensemble diagnostic summary.

## 2026-06-22

- First `pytest -q` was launched from the inverse-RG root and collected unrelated
  workspace tests from other projects. The failures were in `ML_sampling_clean`
  and were not scaffold regressions.
- Action taken: rerun the scaffold test suite from inside
  `InverseBlocking_MIT_NF/` so only the scaffold package is measured.
