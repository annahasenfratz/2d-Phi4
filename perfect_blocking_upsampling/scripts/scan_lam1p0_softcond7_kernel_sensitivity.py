#!/usr/bin/env python3
"""Frozen-flow, D4-orbit finite differences of the global-A/R proposal."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R3_soft_conditioned_7x7_eta_included.json'
CK=ROOT/'perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_softcond7_pureNLL_N5000_20260808T230212Z/stage_oo/checkpoints/checkpoint_best_nll.pt'
AUDIT=ROOT/'perfect_blocking_upsampling/scripts/audit_lam1p0_softcond7_global_ar.py'
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--epsilon',type=float,default=5e-4);p.add_argument('--n',type=int,default=1000);p.add_argument('--seed',type=int,default=20260812);a=p.parse_args();out=a.out if a.out.is_absolute() else ROOT/a.out;out.mkdir(parents=True,exist_ok=True);d=json.loads(BASE.read_text());M=np.array(d['matrix'],float);r=3
 orbs=[(i,j) for i in range(1,4) for j in range(i+1)]
 rows=[]
 for ox,oy in orbs:
  mult=sum(1 for x in range(-r,r+1) for y in range(-r,r+1) if tuple(sorted((abs(x),abs(y)),reverse=True))==(ox,oy))
  for sign in (-1,1):
   A=M.copy()
   for i,x in enumerate(range(-r,r+1)):
    for j,y in enumerate(range(-r,r+1)):
     if tuple(sorted((abs(x),abs(y)),reverse=True))==(ox,oy):A[i,j]+=sign*a.epsilon
   A[r,r]-=sign*a.epsilon*mult
   eig=np.abs(np.fft.fft2(A,s=(32,32))); 
   if eig.min()<0.35 or eig.max()/eig.min()>3.0:
    rows.append({'orbit':f'{ox}{oy}','sign':sign,'status':'rejected_guard','epsilon':a.epsilon,'logw_std':'','ess_over_n':'','stationary_acceptance':''});continue
   q=dict(d);q['matrix']=A.tolist();q['name']=f'fd_{ox}{oy}_{sign:+d}';kp=out/f'kernel_{ox}{oy}_{sign:+d}.json';kp.write_text(json.dumps(q,indent=2)+'\n')
   op=out/f'case_{ox}{oy}_{sign:+d}';cmd=[sys.executable,'-B',str(AUDIT),'--checkpoint',str(CK),'--kernel',str(kp),'--out',str(op),'--n',str(a.n),'--seed',str(a.seed)];res=subprocess.run(cmd,capture_output=True,text=True)
   if res.returncode: rows.append({'orbit':f'{ox}{oy}','sign':sign,'status':'failed','stderr':res.stderr[-500:]});continue
   s=json.loads((op/'summary.json').read_text());rows.append({'orbit':f'{ox}{oy}','sign':sign,'status':'ok','epsilon':a.epsilon,'logw_std':s['logw_std'],'ess_over_n':s['ess_over_n'],'stationary_acceptance':s['stationary_independence_MH_acceptance_proxy']})
 import csv
 with (out/'sensitivity.csv').open('w',newline='') as f: csv.DictWriter(f,fieldnames=list(rows[0])).writeheader();csv.DictWriter(f,fieldnames=list(rows[0])).writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
