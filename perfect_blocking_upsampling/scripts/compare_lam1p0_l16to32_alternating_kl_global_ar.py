#!/usr/bin/env python3
"""Common N=5000 global-A/R audit: highcorr5 baseline vs KL iterations 1/2."""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PKG=ROOT/'perfect_blocking_upsampling'
AUDIT=PKG/'scripts/audit_lam1p0_softcond7_global_ar.py'
BASE_CK=PKG/'runs/lam1p0/training/lam1p0_L16to32_highcorr5_pureNLL_N5000_20260807T063341Z/stage_oo/checkpoints/checkpoint_best_nll.pt'
BASE_K=ROOT/'perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R2_corrW5000_highcorr_5x5_eta_included.json'
RUN=PKG/'runs/lam1p0/training/lam1p0_L16to32_alternatingKL_highcorr5_r1'
KDIR=ROOT/'perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L16to32_highcorr5_r1'

def main() -> None:
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=5000);p.add_argument('--seed',type=int,default=20260812);a=p.parse_args()
 out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
 pairs=[('baseline',BASE_CK,BASE_K),('iteration1',RUN/'flow1/stage_oo/checkpoints/checkpoint_best_nll.pt',KDIR/'iteration1.json'),('iteration2',RUN/'flow2/stage_oo/checkpoints/checkpoint_best_nll.pt',KDIR/'iteration2.json')]
 rows=[]
 for label,ck,kernel in pairs:
  subprocess.run([sys.executable,'-B',str(AUDIT),'--checkpoint',str(ck),'--kernel',str(kernel),'--out',str(out/label),'--n',str(a.n),'--seed',str(a.seed)],check=True)
  row=json.loads((out/label/'summary.json').read_text());row['label']=label;rows.append(row)
 keys=['label','logw_std','logw_q05','logw_q95','ess_over_n','stationary_independence_MH_acceptance_proxy','logdet_K']
 with (out/'comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in rows])
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
