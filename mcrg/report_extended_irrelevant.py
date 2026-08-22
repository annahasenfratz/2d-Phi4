#!/usr/bin/env python3
"""Report the systematic scalar-even extension without forcing an omega claim."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
root=Path('mcrg/results/extended_irrelevant');out=root/'report';out.mkdir(exist_ok=True)
data={(L,k):json.load(open(root/f'L{L}_{k}.json')) for L in (128,64) for k in ('perfect5','perfect7')}
fig,axes=plt.subplots(2,2,figsize=(10,7),sharey=True,constrained_layout=True)
for ax,(L,k) in zip(axes.flat,data):
 d=data[L,k]
 for base,color in [('7','C0'),('9','C1'),('11','C2')]:
  if base not in d['bases']:continue
  for j,r in enumerate(d['bases'][base]['pairs']):
   vals=[x['eigenvalue']['real'] for x in r['roots'] if abs(x['eigenvalue']['imag'])<1e-7 and 0<abs(x['eigenvalue']['real'])<1]
   ax.scatter([j]*len(vals),vals,color=color,marker='o',label=f'E1–E{base}' if j==0 else None)
 ax.axhline(.25,color='black',ls='--');ax.set(title=f'L{L} {k}',xlabel='blocking depth',ylabel='real subunit roots');ax.grid(alpha=.25);ax.legend(fontsize=7)
fig.savefig(out/'extended_real_subunit_spectra.pdf');plt.close(fig)
lines=['# Systematic enlarged scalar-even Swendsen basis','','## Operators and symmetry','','All operators are extensive, translationally invariant Z2-even D4 scalar (A1) sums.  No lattice-anisotropy irrep was mixed into this matrix.','','| stage | additions |','|---|---|','| E1–E7 | original basis |','| E1–E9 | `phi^8`; D4 knight (2,1) bilinear |','| E1–E11 | D4 (2,2) and axial (3,0) bilinears |','| E1–E13 | nearest-neighbor `phi^3 phi`; `phi^2 (grad phi)^2` |','','## Conditioning','','| L / kernel | E1–E7 cond(A), 128→64 or 64→32 | E1–E9 | E1–E11 |','|---|---:|---:|---:|']
for (L,k),d in data.items():
 first={b:d['bases'][b]['pairs'][0]['condition_number_A'] for b in d['bases']};lines.append(f"| L{L} {k} | {first['7']:.3g} | {first['9']:.3g} | {first['11']:.3g} |")
lines += ['','E1–E13 was explicitly tested for L128 5x5: cond(A) rose to 10^15–10^17, so the extension is rejected as near-linearly dependent.  The stable stopping point is E1–E11, although its 10^5 conditioning already weakens subleading-root control.','','## Full spectra','','Machine-readable full spectra, bootstrap root matching fractions, residuals, eigenvectors, singular values, and whitening-invariance checks are in the four JSON files adjacent to this report.  Common fine-covariance whitening reproduced the central spectrum to numerical precision before its use as a conditioning check.','','### Central eigenvalues (all roots)','','Each entry is `real+imag i`; negative and complex roots are retained.']
for (L,k),d in data.items():
 lines += [f'',f'#### L{L} {k}','', '| basis | pair | central eigenvalues |','|---:|---|---|']
 for base,x in d['bases'].items():
  for r in x['pairs']:
   vals=', '.join(f"{z['eigenvalue']['real']:.5g}{z['eigenvalue']['imag']:+.5g}i" for z in r['roots'])
   lines.append(f"| {base} | {r['pair']} | {vals} |")
lines += ['','## Conclusion','','No real subunit branch near 0.25 remains stable across E1–E7 → E1–E9 → E1–E11, depth, L64/L128, and 5x5/7x7.  Candidate roots near 0.23–0.27 occur only in isolated, poorly conditioned enlarged-basis cases.  Therefore the current Swendsen truncation cannot resolve omega=2.']
(out/'extended_irrelevant_report.md').write_text('\n'.join(lines)+'\n')
