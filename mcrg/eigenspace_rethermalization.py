#!/usr/bin/env python3
"""Leading even 2D invariant-subspace comparisons across saved checkpoints."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from mcrg.analyze import load_fields
from mcrg.blocking import perfect_block
from mcrg.operators import names,measure
from mcrg.rg import solve_rg
PAIRS=('64->32','32->16')
def basis(T):
 w,v=np.linalg.eig(T);ii=[i for i in np.argsort(w.real)[::-1] if w[i].real>0 and abs(w[i].imag)<1e-7][:2];return np.linalg.qr(v[:,ii].real)[0]
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out={}
 vectors={}
 for s in (0,25,50):
  fs=[load_fields(a.checkpoints/f'checkpoint_sweep_{s:04d}.npz',None)]
  for _ in range(4):fs.append(perfect_block(fs[-1]))
  for k in (5,7):
   obs=[measure(x,names('even')[:k]) for x in fs]
   for i,pair in enumerate(PAIRS):vectors[(s,k,pair)]=basis(solve_rg(obs[i+1],obs[i+2],1e-10).T)
 for k in (5,7):
  for pair in PAIRS:
   for s0,s1 in ((0,25),(25,50),(0,50)):
    q0,q1=vectors[(s0,k,pair)],vectors[(s1,k,pair)];sv=np.linalg.svd(q0.T@q1,compute_uv=False);out[f'E1-E{k}_{pair}_{s0}_to_{s1}']={'singular_overlaps':sv.tolist(),'principal_angles_degrees':np.degrees(np.arccos(np.clip(sv,-1,1))).tolist()}
 a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
