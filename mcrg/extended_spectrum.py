#!/usr/bin/env python3
"""Nested scalar-even Swendsen spectra with overlap-based bootstrap matching."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from mcrg.analyze import load_fields
from mcrg.blocking import block
from mcrg.operators import names,measure
from mcrg.rg import solve_rg,solve_rg_common_transform

BASES=(7,9,11,13)
def eigvec(v):
 x=v.real;return x/max(np.linalg.norm(x),1e-300)
def enc(z):return {'real':float(z.real),'imag':float(z.imag)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--configs',type=Path,required=True);p.add_argument('--kernel',choices=['perfect5','perfect7'],required=True);p.add_argument('--levels',type=int,required=True);p.add_argument('--bootstrap',type=int,default=300);p.add_argument('--max-base',type=int,default=13);p.add_argument('--seed',type=int,default=1);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 fs=[load_fields(a.configs,None)]
 for _ in range(a.levels):fs.append(block(fs[-1],a.kernel))
 out={'kernel':a.kernel,'input':str(a.configs),'sizes':[int(x.shape[-1]) for x in fs],'bases':{}}
 rng=np.random.default_rng(a.seed)
 for k in (x for x in BASES if x <= a.max_base):
  op=names('even')[:k]; obs=[measure(x,op) for x in fs]; diag={'operators':op,'variance':obs[0].var(0,ddof=1).tolist(),'correlation':np.corrcoef(obs[0],rowvar=False).tolist(),'fine_covariance_singular_values':np.linalg.svd(np.cov(obs[0],rowvar=False),compute_uv=False).tolist(),'pairs':[]}
  for n in range(a.levels):
   fine,coarse=obs[n],obs[n+1]; central=solve_rg(fine,coarse,1e-10); w,vec=central.eigenvalues,central.right_eigenvectors
   # Fixed symmetric whitening from this pair's fine covariance, common at both levels.
   cf=np.cov(fine,rowvar=False);ev,u=np.linalg.eigh(cf);W=(u/np.sqrt(np.maximum(ev,ev.max()*1e-14))).T
   white=solve_rg_common_transform(fine,coarse,W,1e-10)
   white_match=max(abs(np.sort_complex(w)-np.sort_complex(white.eigenvalues)))
   matched=[[] for _ in w]; reliable=[0]*len(w)
   for _ in range(a.bootstrap):
    idx=rng.integers(len(fine),size=len(fine));r=solve_rg(fine[idx],coarse[idx],1e-10);wb,vb=r.eigenvalues,r.right_eigenvectors
    # Match to central roots using eigenvector overlap first, then proximity;
    # this deliberately avoids an eigenvalue-rank convention.
    for i,z in enumerate(w):
     overlaps=np.array([abs(np.vdot(eigvec(vec[:,i]),eigvec(vb[:,j]))) for j in range(len(wb))])
     scale=max(1.,abs(z));score=overlaps/(1.+abs(wb-z)/scale)
     j=int(np.argmax(score));matched[i].append(wb[j]);reliable[i]+=score[j]>.35
   roots=[]
   for i,z in enumerate(w):
    q=np.asarray(matched[i]);roots.append({'eigenvalue':enc(z),'right_eigenvector':eigvec(vec[:,i]).tolist(),'residual_norm':float(np.linalg.norm(central.T@vec[:,i]-z*vec[:,i])),'bootstrap_real_std':float(q.real.std(ddof=1)),'bootstrap_imag_std':float(q.imag.std(ddof=1)),'match_fraction':reliable[i]/a.bootstrap})
   candidates=[{'index':i,'lambda':enc(z),'omega':float(-np.log(abs(z.real))/np.log(2)) if abs(z.imag)<1e-7 and 0<abs(z.real)<1 else None,'match_fraction':roots[i]['match_fraction']} for i,z in enumerate(w) if abs(z.imag)<1e-7 and 0<abs(z.real)<1]
   diag['pairs'].append({'pair':f'{fs[n].shape[-1]}->{fs[n+1].shape[-1]}','condition_number_A':central.condition_number,'singular_values_A':central.singular_values.tolist(),'whitening_max_eigenvalue_difference':float(white_match),'roots':roots,'real_subunit_candidates':candidates})
  out['bases'][str(k)]=diag
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
