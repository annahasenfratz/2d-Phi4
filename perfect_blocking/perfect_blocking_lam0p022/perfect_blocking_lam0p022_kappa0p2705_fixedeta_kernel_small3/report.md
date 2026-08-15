# Lambda=0.022 Critical Small 3-Orbit Fixed-Eta Kernel

Diagnostic only. The primary optimization used L32 -> L16 with all four
K_eta sublattices as correlated blocked samples.

## Selected Kernel

- best start: `edge_only`
- eta: `0.25`
- sum K: `1`
- min |K_eta(q)|: `0.19946638`
- max |K_eta(q)|: `1.0905077`
- condition number: `5.4671254`
- inverse max real roundtrip error: `1.776e-15`
- inverse max imaginary residue: `6.944e-16`

Weights:
- w00 = `0.648926023044`
- w10 = `0.102136064941`
- w11 = `-0.0143675707023`

## Matching Scores

| kernel | match | normalized RMS | D_op | rms z | max z | cond(K_eta) |
|---|---|---:|---:|---:|---:|---:|
| small3 | L32->L16 | 0.0964488 | 4538.96 | 11.6293 | 18.805 | 5.46713 |
| old 5x5 projected | L32->L16 | 0.649294 | 58698.6 | 112.664 | 142.446 | 1.6155 |
| large 8-orbit | L32->L16 | 0.0703823 | 1289 | 5.69656 | 9.37576 | 2.57525 |
| small3 | L16->L8 | 0.139587 | 1523.94 | 13.0665 | 17.8443 | 5.46713 |

Flags: ['min |K_eta(q)| below 0.25', 'condition number more than twice old projected 3-orbit'].
