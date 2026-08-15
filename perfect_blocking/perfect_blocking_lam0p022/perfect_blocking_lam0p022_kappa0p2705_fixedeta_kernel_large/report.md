# Lambda=0.022 Critical Large-Kernel Fixed-Eta Optimization

Diagnostic only. Kernel ansatz has orbit weights `w00, w01, w11, w20, w21, w22, w30, w31`.
The primary optimization used L32 -> L16 with all four K_eta sublattices as
correlated blocked samples. The L32 ensemble has N=1000, so this is
statistics-limited.

## Selected Kernel

- best start: `old5_embed`
- eta: `0.25`
- sum K: `1`
- min |K_eta(q)|: `0.51093286`
- max |K_eta(q)|: `1.3157793`
- condition number: `2.575249`
- inverse max real roundtrip error: `3.109e-15`
- inverse max imaginary residue: `6.780e-16`

Weights:
- w00 = `0.844848483811`
- w01 = `-0.032432937708`
- w11 = `-0.0245751408915`
- w20 = `0.0740346004744`
- w21 = `0.0214631605575`
- w22 = `-0.0140586709894`
- w30 = `-0.0012257427138`
- w31 = `-0.00294027511973`

## Matching Scores

| kernel | match | normalized RMS | D_op | rms z | max | cond(K_eta) |
|---|---|---:|---:|---:|---:|---:|
| large | L32->L16 | 0.0703823 | 1289 | 5.69656 | 9.37576 | 2.57525 |
| old 5x5 | L32->L16 | 0.0894392 | 2182.08 | 10.2463 | 15.3967 | 2.06737 |
| large | L16->L8 | 0.10784 | 781.289 | 8.89451 | 14.0671 | 2.45002 |
| old 5x5 | L16->L8 | 0.129335 | 830.071 | 11.9507 | 15.6489 | 2.0623 |

Flags: none.
