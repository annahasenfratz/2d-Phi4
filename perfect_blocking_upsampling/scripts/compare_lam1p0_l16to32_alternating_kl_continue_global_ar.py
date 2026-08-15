#!/usr/bin/env python3
"""Common N=5000 global-A/R audit for L16->L32 KL iterations 2, 3, 4."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PKG=ROOT/'perfect_blocking_upsampling';AUDIT=PKG/'scripts/audit_lam1p0_softcond7_global_ar.py'
BASE=PKG/'runs/lam1p0/training/lam1p0_L16to32_alternatingKL_highcorr5_r1';CONT=PKG/'runs/lam1p0/training/lam1p0_L16to32_alternatingKL_highcorr5_r1_continue';K2=ROOT/'perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L16to32_highcorr5_r1/iteration2.json';KC=ROOT/'perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L16to32_highcorr5_r1_continue'
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=5000);p.add_argument('--seed',type=int,default=20260812);a=p.parse_args();out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
 pairs=[('iteration2',BASE/'flow2/stage_oo/checkpoints/checkpoint_best_nll.pt',K2),('iteration3',CONT/'flow3/stage_oo/checkpoints/checkpoint_best_nll.pt',KC/'iteration3.json'),('iteration4',CONT/'flow4/stage_oo/checkpoints/checkpoint_best_nll.pt',KC/'iteration4.json')];rows=[]
 for label,ck,k in pairs:
  subprocess.run([sys.executable,'-B',str(AUDIT),'--checkpoint',str(ck),'--kernel',str(k),'--out',str(out/label),'--n',str(a.n),'--seed',str(a.seed)],check=True)
  row=json.loads((out/label/'summary.json').read_text());row['label']=label;rows.append(row)
 keys=['label','logw_std','logw_q05','logw_q95','ess_over_n','stationary_independence_MH_acceptance_proxy','logdet_K']
 with (out/'comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in rows])
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
