#!/usr/bin/env python3
"""Cross-volume thermal MCRG report; eta is fixed input, never fitted here."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mcrg.analyze import load_fields
from mcrg.blocking import average_block, perfect_block
from mcrg.operators import names, measure
from mcrg.rg import solve_rg

PAIR_LABELS={128:["128->64","64->32","32->16","16->8"],64:["64->32","32->16","16->8"]}

def fields(phi,kernel,levels):
 out=[phi]
 for _ in range(levels): out.append(average_block(out[-1],"matched") if kernel=="average" else perfect_block(out[-1]))
 return out

def eig_data(T):
 w,v=np.linalg.eig(T); ix=np.argsort(w.real)[::-1]
 # Positive-real directions only; record all values independently in JSON.
 ix=[i for i in ix if abs(w[i].imag)<1e-7 and w[i].real>0][:2]
 vv=[]
 for i in ix:
  x=v[:,i].real/np.linalg.norm(v[:,i].real); vv.append(x if x[0]>=0 else -x)
 return w, np.array(vv).T

def support(width,n):
 # Explicit recursion of the one-dimensional stencil support, then formula.
 s={0}
 offsets=range(0,width) if width==2 else range(-(width//2),width//2+1)
 for _ in range(n): s={2*x+r for x in s for r in offsets}
 return max(s)-min(s)+1, 1+(width-1)*(2**n-1)

def make_plots(data,out):
 for quantity,target,name in (("lambda",2.,"lambda_t"),("nu",1.,"nu")):
  fig,axes=plt.subplots(1,2,figsize=(9.5,3.8),sharey=True,constrained_layout=True)
  for ax,kernel in zip(axes,["average","perfect7"]):
   d=data[(128,kernel)]
   for k in (3,5,7):
    rows=[r for r in d['results'] if r['sector']=='even' and len(r['operators'])==k]
    y=[r['lambda'] if quantity=='lambda' else r['exponents']['nu'] for r in rows]
    e=[r['bootstrap']['lambda_std'] if quantity=='lambda' else r['bootstrap']['exponent_std'] for r in rows]
    ax.errorbar(range(4),y,yerr=e,marker='o',capsize=2,label=f'E1–E{k}')
   ax.axhline(target,color='black',ls='--',lw=.8); ax.set(title=('matched 2×2' if kernel=='average' else 'perfect7'),xlabel='blocking pair',xticks=range(4),xticklabels=PAIR_LABELS[128]);ax.grid(alpha=.25);ax.legend()
  axes[0].set_ylabel(r'$\lambda_t$' if quantity=='lambda' else r'$\nu$');fig.savefig(out/f'{name}_L128_by_kernel.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(9.5,3.8),sharey=True,constrained_layout=True)
 for ax,k in zip(axes,(5,7)):
  for L,style in ((128,'-'),(64,'--')):
   for kernel,color in (("average","C0"),("perfect7","C1")):
    d=data[(L,kernel)]; rows=[r for r in d['results'] if r['sector']=='even' and len(r['operators'])==k]
    y=[r['lambda'] for r in rows];e=[r['bootstrap']['lambda_std'] for r in rows]
    ax.errorbar(range(len(y)),y,yerr=e,color=color,linestyle=style,marker='o',capsize=2,label=f'{kernel}, L{L}')
  ax.axhline(2,color='black',ls='--',lw=.8);ax.set(title=f'E1–E{k}',xlabel='pair index (native hierarchy)');ax.grid(alpha=.25);ax.legend(fontsize=7)
 axes[0].set_ylabel(r'$\lambda_t$');fig.savefig(out/'lambda_t_fixed_basis_kernel_comparison.pdf');plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument('--l128',type=Path,required=True);p.add_argument('--l64',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 data={(128,'average'):json.loads((a.input_dir/'L128_average_matched.json').read_text()),(128,'perfect7'):json.loads((a.input_dir/'L128_perfect7.json').read_text()),(64,'average'):json.loads((a.input_dir/'L64_average_matched.json').read_text()),(64,'perfect7'):json.loads((a.input_dir/'L64_perfect7.json').read_text())}
 summary={'eta_input':.25,'derived_relations':{'beta':'nu/8','gamma':'7 nu/4'},'support':{},'eigenspaces':{},'same_scale':[]}
 for kernel,width in (("average",2),("perfect7",7)):
  summary['support'][kernel]=[{"depth":n,"explicit_width":support(width,n)[0],"formula_width":support(width,n)[1]} for n in range(1,5)]
 for L,path in ((128,a.l128),(64,a.l64)):
  for kernel in ('average','perfect7'):
   fs=fields(load_fields(path,None),kernel,len(PAIR_LABELS[L])); summary['eigenspaces'][f'L{L}_{kernel}']={}
   for k in (3,5,7):
    obs=[measure(x,names('even')[:k]) for x in fs]; rs=[solve_rg(obs[i],obs[i+1],1e-10) for i in range(len(fs)-1)]; ev=[eig_data(r.T) for r in rs]; rows=[]
    for label,r,(vals,vecs) in zip(PAIR_LABELS[L],rs,ev):
     rows.append({'pair':label,'eigenvalues':[float(x.real) for x in vals],'leading':float(vals[np.argsort(vals.real)[::-1][0]].real),'second':float(vals[np.argsort(vals.real)[::-1][1]].real),'separation':float(vals[np.argsort(vals.real)[::-1][0]].real-vals[np.argsort(vals.real)[::-1][1]].real),'right_leading':vecs[:,0].tolist(),'condition_number':r.condition_number,'singular_values':r.singular_values.tolist()})
    transitions=[]
    for i in range(len(ev)-1):
     v1,v2=ev[i][1],ev[i+1][1]; one=float(abs(v1[:,0]@v2[:,0])); s=np.linalg.svd(np.linalg.qr(v1)[0].T@np.linalg.qr(v2)[0],compute_uv=False);angles=np.degrees(np.arccos(np.clip(s,-1,1)))
     transitions.append({'between':f'{PAIR_LABELS[L][i]} / {PAIR_LABELS[L][i+1]}','single_vector_overlap':one,'leading2_singular_overlaps':s.tolist(),'principal_angles_degrees':angles.tolist()})
    summary['eigenspaces'][f'L{L}_{kernel}'][str(k)]={'pairs':rows,'transitions':transitions}
 # Same input/output lattices: L128 after one preceding block vs independent L64.
 for kernel in ('average','perfect7'):
  for k in (3,5,7):
   for pair in ('64->32','32->16','16->8'):
    x=[r for r in data[(128,kernel)]['results'] if r['sector']=='even' and len(r['operators'])==k and r['pair']==pair][0]
    y=[r for r in data[(64,kernel)]['results'] if r['sector']=='even' and len(r['operators'])==k and r['pair']==pair][0]
    summary['same_scale'].append({'kernel':kernel,'basis':k,'pair':pair,'L128_history_lambda':x['lambda'],'L128_history_nu':x['exponents']['nu'],'independent_L64_lambda':y['lambda'],'independent_L64_nu':y['exponents']['nu'],'difference_lambda':x['lambda']-y['lambda']})
 make_plots(data,a.output_dir); (a.output_dir/'native_full_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['# Full native thermal MCRG benchmark','','η=0.25 is an input to the field normalization. β=ν/8 and γ=7ν/4 are derived quantities only.','','## Same-scale comparison','','| kernel | basis | pair | L128 history λt, ν | independent L64 λt, ν | Δλt |','|---|---:|---|---|---|---:|']
 for x in summary['same_scale']: lines.append(f"| {x['kernel']} | {x['basis']} | {x['pair']} | {x['L128_history_lambda']:.5g}, {x['L128_history_nu']:.5g} | {x['independent_L64_lambda']:.5g}, {x['independent_L64_nu']:.5g} | {x['difference_lambda']:.3g} |")
 lines += ['','## Repeated-block support (verified explicit recursion = formula)','','| kernel | depth | width | width / L128 |','|---|---:|---:|---:|']
 for ker,rows in summary['support'].items():
  for x in rows:lines.append(f"| {ker} | {x['depth']} | {x['explicit_width']} | {x['explicit_width']/128:.3g} |")
 lines += ['','## Eigenspace diagnostics','','Full eigenvalues, vectors, singular values, overlaps, and principal angles are in `native_full_summary.json`.']
 (a.output_dir/'native_full_report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
