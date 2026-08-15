#!/usr/bin/env python3
"""Run a common global-A/R audit for all L8->L16 alternating-KL pairs."""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L8to16_alternatingKL_5x5_r1"
KERNELS = ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L8to16_5x5_r1"
AUDIT = ROOT / "perfect_blocking_upsampling/scripts/audit_lam1p0_l8to16_global_ar.py"
BASE = ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/best_5x5_retrained_full_objective_eta_included.json"

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=5000);p.add_argument('--seed',type=int,default=20260810);a=p.parse_args()
    out=ROOT/a.out; out.mkdir(parents=True,exist_ok=True)
    pairs=[('initial',RUN/'flow0/checkpoints/checkpoint_best.pt',BASE),('iteration1',RUN/'flow1/checkpoints/checkpoint_best.pt',KERNELS/'iteration1.json'),('iteration2',RUN/'flow2/checkpoints/checkpoint_best.pt',KERNELS/'iteration2.json')]
    rows=[]
    for label,checkpoint,kernel in pairs:
        case=out/label
        cmd=[sys.executable,'-B',str(AUDIT),'--checkpoint',str(checkpoint),'--kernel',str(kernel),'--out',str(case),'--n',str(a.n),'--seed',str(a.seed)]
        subprocess.run(cmd,check=True)
        row=json.loads((case/'summary.json').read_text()); row['label']=label;rows.append(row)
    keys=['label','logw_std','logw_q01','logw_q99','ess_over_n','stationary_independence_MH_acceptance_proxy','min_K','max_invK']
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in rows])
    print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
