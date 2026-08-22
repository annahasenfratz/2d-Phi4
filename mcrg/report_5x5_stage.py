#!/usr/bin/env python3
"""Compact three-kernel thermal and subleading-spectrum side-project report."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
root=Path('mcrg/results');out=root/'stage_5x5_l256';out.mkdir(exist_ok=True)
native={k:json.load(open(root/'native_full'/f'L128_{f}.json')) for k,f in [('2x2','average_matched'),('5x5','perfect5'),('7x7','perfect7')]}
l256={k:json.load(open(root/'upscaled_l256_sweep0'/f'{f}.json')) for k,f in [('2x2','average_matched'),('5x5','perfect5'),('7x7','perfect7')]}
def rows(d,k):return [x for x in d['results'] if x['sector']=='even' and len(x['operators'])==k]
for qty,target,name,ylabel in [('lambda',2,'native_L128_three_kernel_lambda_t',r'$\lambda_t$'),('nu',1,'native_L128_three_kernel_nu',r'$\nu$')]:
 fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
 for lab,d in native.items():
  for k,m in [(5,'o'),(7,'s')]:
   r=rows(d,k);y=[x['lambda'] if qty=='lambda' else x['exponents']['nu'] for x in r];e=[x['bootstrap']['lambda_std'] if qty=='lambda' else x['bootstrap']['exponent_std'] for x in r];ax.errorbar(range(4),y,yerr=e,marker=m,capsize=2,label=f'{lab} E1–E{k}')
 ax.axhline(target,color='black',ls='--');ax.set(xticks=range(4),xticklabels=['128→64','64→32','32→16','16→8'],xlabel='blocking pair',ylabel=ylabel);ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2);fig.savefig(out/f'{name}.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
for lab in ('5x5','7x7'):
 for source,d,style in [('native L64',json.load(open(root/'native_full'/f'L64_perfect{lab[0]}.json')),'--'),('native L128',native[lab],'-'),('L256 sweep0',l256[lab],':')]:
  r=next(x for x in rows(d,5) if x['pair']=='64->32');ax.errorbar([{'native L64':0,'native L128':1,'L256 sweep0':2}[source]],[r['lambda']],yerr=[r['bootstrap']['lambda_std']],marker='o',linestyle='none',label=f'{lab} {source}')
ax.axhline(2,color='black',ls='--');ax.set(xticks=[0,1,2],xticklabels=['L64 direct','L128 after 1','L256 after 2'],ylabel=r'$\lambda_t$ at 64→32',xlabel='prior blocking history');ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.25);fig.savefig(out/'RG_depth_64to32.pdf');plt.close(fig)
# Real subunit roots are displayed as unlabelled candidate traces: no robust branch claim.
fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True,constrained_layout=True)
for ax,kern in zip(axes,['perfect5','perfect7']):
 d=json.load(open(root/'irrelevant_native'/f'L128_{kern}.json'))
 for basis,color in [('5','C0'),('7','C1')]:
  for j,r in enumerate(d['bases'][basis]):
   vals=[x['eigenvalue']['real'] for x in r['real_subunit_candidates']]
   ax.scatter([j]*len(vals),vals,color=color,marker='o' if basis=='5' else 's',label=f'E1–E{basis}' if j==0 else None)
 ax.axhline(.25,color='black',ls='--');ax.set(title=kern,xticks=range(4),xticklabels=['128→64','64→32','32→16','16→8'],xlabel='pair',ylabel='real subunit eigenvalues');ax.legend();ax.grid(alpha=.25)
fig.savefig(out/'irrelevant_real_subunit_candidates.pdf');plt.close(fig)
lines=['# 5x5 + L256 thermal / irrelevant stage','','## Thermal highlight','','| source | kernel | basis | 128→64 λt | 64→32 λt | 32→16 λt |','|---|---|---:|---:|---:|---:|']
for source,data in [('native L128',native['5x5']),('L256 sweep0 5x5',l256['5x5']),('L256 sweep0 7x7',l256['7x7'])]:
 for k in (5,7):
  r=rows(data,k);z={x['pair']:x for x in r};lines.append(f"| {source} | {data['kernel']} | {k} | {z['128->64']['lambda']:.5g} | {z['64->32']['lambda']:.5g} | {z['32->16']['lambda']:.5g} |")
lines+=['','No stable real subunit branch near 0.25 survives pair, basis, volume, and kernel changes; therefore no ω estimate or ω plot is claimed.']
(out/'summary.md').write_text('\n'.join(lines)+'\n')
