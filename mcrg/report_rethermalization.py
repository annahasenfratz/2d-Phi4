#!/usr/bin/env python3
"""Thermal MCRG evolution over saved Wolff+radial checkpoints only."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SWEEPS=(0,25,50); PAIRS=('64->32','32->16')
def row(d,pair,k): return next(x for x in d['results'] if x['sector']=='even' and x['pair']==pair and len(x['operators'])==k)
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--native-average',type=Path,required=True);p.add_argument('--native-perfect7',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 ds={(s,k):json.loads((a.input_dir/f'sweep_{s:04d}_{k}.json').read_text()) for s in SWEEPS for k in ('average','perfect7')}; native={'average':json.loads(a.native_average.read_text()),'perfect7':json.loads(a.native_perfect7.read_text())}
 summary={'available_sweeps':list(SWEEPS),'comparisons':[]}
 for kernel in ('average','perfect7'):
  for basis in (5,7):
   for pair in PAIRS:
    n=row(native[kernel],pair,basis); nb=n['bootstrap']
    for s in SWEEPS:
     u=row(ds[s,kernel],pair,basis);ub=u['bootstrap']; dl=u['lambda']-n['lambda']; dn=u['exponents']['nu']-n['exponents']['nu']; sl=(ub['lambda_std']**2+nb['lambda_std']**2)**.5; sn=(ub['exponent_std']**2+nb['exponent_std']**2)**.5
     summary['comparisons'].append({'sweep':s,'kernel':kernel,'basis':basis,'pair':pair,'lambda_t':u['lambda'],'lambda_error':ub['lambda_std'],'nu':ub['exponent_median'],'nu_error':ub['exponent_std'],'cond_A':u['condition_number'],'singular_values_A':u['singular_values'],'leading_right_eigenvector':u['leading_right_eigenvector'],'delta_lambda':dl,'delta_lambda_error':sl,'z_lambda':dl/sl,'delta_nu':dn,'delta_nu_error':sn,'z_nu':dn/sn})
 # Perfect7 λ and ν: native bands per pair, E5/E7 markers.
 for quantity,target,name,ylabel in (('lambda_t',2.,'perfect7_lambda_t_vs_sweep',r'$\lambda_t$'),('nu',1.,'perfect7_nu_vs_sweep',r'$\nu$')):
  fig,axes=plt.subplots(1,2,figsize=(9.5,3.8),sharey=True,constrained_layout=True)
  for ax,pair in zip(axes,PAIRS):
   for basis,mark in ((5,'o'),(7,'s')):
    q=[x for x in summary['comparisons'] if x['kernel']=='perfect7' and x['basis']==basis and x['pair']==pair]
    error_key='lambda_error' if quantity=='lambda_t' else 'nu_error'
    ax.errorbar(SWEEPS,[x[quantity] for x in q],yerr=[x[error_key] for x in q],marker=mark,capsize=2,label=f'E1–E{basis}')
   n=row(native['perfect7'],pair,5); centre=n['lambda'] if quantity=='lambda_t' else n['exponents']['nu']; err=n['bootstrap']['lambda_std'] if quantity=='lambda_t' else n['bootstrap']['exponent_std'];ax.axhspan(centre-err,centre+err,color='C2',alpha=.2,label='native E1–E5 ±1σ');ax.axhline(target,color='black',ls='--',lw=.8);ax.set(title=pair,xlabel='Wolff+radial sweeps');ax.grid(alpha=.25);ax.legend(fontsize=8)
  axes[0].set_ylabel(ylabel);fig.savefig(a.output_dir/f'{name}.pdf');plt.close(fig)
 # E5 direct kernel control.
 fig,axes=plt.subplots(1,2,figsize=(9.5,3.8),sharey=True,constrained_layout=True)
 for ax,pair in zip(axes,PAIRS):
  for kernel,color in (('average','C0'),('perfect7','C1')):
   q=[x for x in summary['comparisons'] if x['kernel']==kernel and x['basis']==5 and x['pair']==pair];ax.errorbar(SWEEPS,[x['lambda_t'] for x in q],yerr=[x['lambda_error'] for x in q],color=color,marker='o',capsize=2,label=kernel)
  ax.axhline(2,color='black',ls='--',lw=.8);ax.set(title=pair,xlabel='Wolff+radial sweeps');ax.grid(alpha=.25);ax.legend()
 axes[0].set_ylabel(r'$\lambda_t$');fig.savefig(a.output_dir/'E1_E5_kernel_comparison_vs_sweep.pdf');plt.close(fig)
 (a.output_dir/'rethermalization_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['# Thermal MCRG through Wolff+radial rethermalization','','Checkpoints available: 0, 25, 50.  η and derived exponents are intentionally omitted.','','## perfect7 E1–E5','','| sweep | pair | λt ± 1σ | ν ± 1σ | Δλt from native | z(Δλt) | cond(A) |','|---:|---|---|---|---:|---:|---:|']
 for x in summary['comparisons']:
  if x['kernel']=='perfect7' and x['basis']==5:lines.append(f"| {x['sweep']} | {x['pair']} | {x['lambda_t']:.6g} ± {x['lambda_error']:.2g} | {x['nu']:.6g} ± {x['nu_error']:.2g} | {x['delta_lambda']:.4g} ± {x['delta_lambda_error']:.2g} | {x['z_lambda']:.2f} | {x['cond_A']:.3g} |")
 lines += ['','## E1–E7 systematic: perfect7','','| sweep | pair | λt ± 1σ | ν ± 1σ | z(Δλt native) | cond(A) |','|---:|---|---|---|---:|---:|']
 for x in summary['comparisons']:
  if x['kernel']=='perfect7' and x['basis']==7:lines.append(f"| {x['sweep']} | {x['pair']} | {x['lambda_t']:.6g} ± {x['lambda_error']:.2g} | {x['nu']:.6g} ± {x['nu_error']:.2g} | {x['z_lambda']:.2f} | {x['cond_A']:.3g} |")
 lines += ['','## Matched 2×2 E1–E5 control','','| sweep | pair | λt ± 1σ | ν ± 1σ | z(Δλt native) |','|---:|---|---|---|---:|']
 for x in summary['comparisons']:
  if x['kernel']=='average' and x['basis']==5:lines.append(f"| {x['sweep']} | {x['pair']} | {x['lambda_t']:.6g} ± {x['lambda_error']:.2g} | {x['nu']:.6g} ± {x['nu_error']:.2g} | {x['z_lambda']:.2f} |")
 (a.output_dir/'rethermalization_summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
