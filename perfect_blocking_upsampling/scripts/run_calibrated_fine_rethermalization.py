import csv,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'));sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/src'))
from perfect_blocking_upsampling.actions import ActionSpec
from run_native_l32_metropolis import metropolis_sweep,StreamingCsv
from train_lam1p0_local_multistage_rqspline import metrics_rows
RUN=ROOT/'perfect_blocking_upsampling/runs/lam1p0/fresh_disjoint_calibrated_empirical_validation_20260721';OUT=ROOT/'perfect_blocking_upsampling/runs/lam1p0/calibrated_empirical_fine_rethermalization_20260721'
def w(p,r):
 ks=[]
 for x in r:
  for k in x:
   if k not in ks:ks.append(k)
 with p.open('w',newline='') as f:q=csv.DictWriter(f,ks);q.writeheader();q.writerows(r)
def run(L,n,saves):
 z=np.load(RUN/f'paired_fields_L{L}.npz');cal=z['calibrated'][:n].copy();native=z['native'][:n].copy();reference=native.copy();act=ActionSpec('phi4_nn',1.,.340301);rng=np.random.default_rng(998+L);writer=StreamingCsv(OUT/f'updates_L{L}.csv',['sweep','pass','update_order','parity','sites_touched','attempts','accepted','acceptance','DeltaS_mean','DeltaS_std','DeltaS_min','DeltaS_max','log_accept_mean','log_accept_std','elapsed_sec']);rows=[]
 for sw in range(max(saves)+1):
  if sw in saves:
    for label,x in [('calibrated',cal),('native_control',native)]:
     for r in metrics_rows(reference,x,label,'whole'): rows.append({'sweep':sw,'ensemble':label,**r})
  if sw<max(saves):cal,_=metropolis_sweep(cal,act,sw+1,1,.5,'checkerboard',rng,writer);native,_=metropolis_sweep(native,act,sw+1,1,.5,'checkerboard',rng,writer)
 writer.close();w(OUT/f'metrics_L{L}.csv',rows)
def main():
 OUT.mkdir(parents=True,exist_ok=True);run(32,200,[0,1,2,5,10,20,50]);run(64,100,[0,1,2,5,10,20])
if __name__=='__main__':main()
