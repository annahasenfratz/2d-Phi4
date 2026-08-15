#!/usr/bin/env python3
"""Common N=5000 global-A/R comparison for alternating-KL iterations 2, 3, 4."""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PKG=ROOT/'perfect_blocking_upsampling'
AUDIT=PKG/'scripts/audit_lam1p0_l8to16_global_ar.py'
BASE_RUN=PKG/'runs/lam1p0/training/lam1p0_L8to16_alternatingKL_5x5_r1'
CONT_RUN=PKG/'runs/lam1p0/training/lam1p0_L8to16_alternatingKL_5x5_r1_continue'
BASE_K=ROOT/'perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L8to16_5x5_r1/iteration2.json'
CONT_K=ROOT/'perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L8to16_5x5_r1_continue'

def main() -> None:
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=5000);p.add_argument('--seed',type=int,default=20260810);a=p.parse_args()
 out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
 pairs=[('iteration2',BASE_RUN/'flow2/checkpoints/checkpoint_best.pt',BASE_K),('iteration3',CONT_RUN/'flow3/checkpoints/checkpoint_best.pt',CONT_K/'iteration3.json'),('iteration4',CONT_RUN/'flow4/checkpoints/checkpoint_best.pt',CONT_K/'iteration4.json')]
 rows=[]
 for label,ck,kernel in pairs:
  subprocess.run([sys.executable,'-B',str(AUDIT),'--checkpoint',str(ck),'--kernel',str(kernel),'--out',str(out/label),'--n',str(a.n),'--seed',str(a.seed)],check=True)
  row=json.loads((out/label/'summary.json').read_text());row['label']=label;rows.append(row)
 keys=['label','logw_std','logw_q01','logw_q99','ess_over_n','stationary_independence_MH_acceptance_proxy','min_K','max_invK']
 with (out/'comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in rows])
 print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
