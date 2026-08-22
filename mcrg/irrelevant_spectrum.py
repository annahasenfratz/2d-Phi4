#!/usr/bin/env python3
"""Bootstrap full even spectra; candidate labels are continuity aids, not claims."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from mcrg.analyze import load_fields
from mcrg.blocking import block
from mcrg.operators import names,measure
from mcrg.rg import solve_rg,bootstrap_rg
def enc(z):return {'real':float(z.real),'imag':float(z.imag)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--configs',type=Path,required=True);p.add_argument('--kernel',choices=['average','perfect5','perfect7'],required=True);p.add_argument('--levels',type=int,required=True);p.add_argument('--bootstrap',type=int,default=300);p.add_argument('--seed',type=int,default=1);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 fs=[load_fields(a.configs,None)]
 for _ in range(a.levels):fs.append(block(fs[-1],a.kernel))
 out={'input':str(a.configs),'kernel':a.kernel,'sizes':[x.shape[-1] for x in fs],'bases':{}}
 for k in (3,5,7):
  obs=[measure(x,names('even')[:k]) for x in fs];rows=[]
  for n in range(a.levels):
   r=solve_rg(obs[n],obs[n+1],1e-10); central=r.eigenvalues; boots=bootstrap_rg(obs[n],obs[n+1],1e-10,a.bootstrap,a.seed+100*k+n)
   # Match each bootstrap spectrum to each central root by nearest complex value,
   # avoiding a fixed sorted-rank interpretation.
   stats=[]
   for z in central:
    vals=[]
    for b in boots:
     w=b.eigenvalues;vals.append(w[np.argmin(abs(w-z))])
    vals=np.asarray(vals);stats.append({'eigenvalue':enc(z),'bootstrap_real_std':float(vals.real.std(ddof=1)),'bootstrap_imag_std':float(vals.imag.std(ddof=1))})
   candidates=[{'eigenvalue':enc(z),'omega_if_real':float(-np.log(abs(z.real))/np.log(2)) if abs(z.imag)<1e-7 and 0<abs(z.real)<1 else None} for z in central if abs(z.imag)<1e-7 and 0<abs(z.real)<1]
   rows.append({'pair':f'{fs[n].shape[-1]}->{fs[n+1].shape[-1]}','condition_number':r.condition_number,'singular_values':r.singular_values.tolist(),'spectrum':stats,'real_subunit_candidates':candidates})
  out['bases'][str(k)]=rows
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
