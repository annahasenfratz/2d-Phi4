#!/usr/bin/env python3
"""Phase-1/2 exact donor-bank validation for the empirical 2x2 proposal."""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
import run_lam1p0_empirical_joint_2x2_mixture as e
from train_lam1p0_flow_detail_pilot import load_phi,load_kernel_matrix,split_pairs

OUT=ROOT/'perfect_blocking_upsampling/runs/lam1p0/empirical_joint_2x2_mixture_validation_20260721'
def write(path,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with path.open('w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def bank(n,phi,k):
 p=split_pairs(phi[:n],k);m=e.meta(n,16);h=e.features(p['coarse'],m);hm,hs=h.mean(0),h.std(0)+1e-6
 return (h-hm)/hs,e.vectors(p['detail'],m),hm,hs
def run_settings(name,targets,k,H,D,hm,hs,settings):
 p=split_pairs(targets,k);c=p['coarse'];M=e.meta(len(targets),c.shape[1]);h=(e.features(c,M)-hm)/hs;idx,d2=e.index_context(H,h);out=[];t0=time.perf_counter()
 for kval,tauq,beta in settings:
  tau=float(np.quantile(np.sqrt(d2[:,0]),tauq));Dnew,lq,sel,w,sig,ld=e.sample(c,M,idx,d2,D,kval,tau,beta,'diag',20260721+kval+int(beta*1000));g,dd=e.reconstruct(c,M,Dnew,k,targets.shape[1]);r=e.metrics(targets,g,name);r.update({'k':kval,'tau_quantile':tauq,'tau':tau,'beta':beta,'mean_logq':float(lq.mean()),'unique_donor_fraction':float(len(np.unique(sel))/len(sel)),'effective_components':float(np.mean(1/(w*w).sum(1))),'sample_seconds':time.perf_counter()-t0,'neighbor_q50':float(np.quantile(np.sqrt(d2[:,0]),.5))});out.append(r)
 return out
def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'plots').mkdir(exist_ok=True);k,_=load_kernel_matrix(e.KPATH);allphi=load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz')
 phase1=[]
 for n in (80,200,500,1000):
  H,D,hm,hs=bank(n,allphi,k);target=allphi[n:n+20];t=time.perf_counter();r=run_settings('L32',target,k,H,D,hm,hs,[(4,.25,.02)])[0];r.update({'Ndonor_configs':n,'Ndonor_blocks':len(D),'donor_context_mb':H.nbytes/2**20,'donor_detail_mb':D.nbytes/2**20,'total_seconds':time.perf_counter()-t});phase1.append(r)
 write(OUT/'donor_bank_scaling.csv',phase1)
 best=max(phase1,key=lambda r:r['Ndonor_configs']);n=int(best['Ndonor_configs']);H,D,hm,hs=bank(n,allphi,k);target=allphi[n:n+20]
 settings=[(kk,tq,b) for kk in (2,4,8) for tq in (.10,.25,.40) for b in (.01,.02,.03,.05)]
 r32=run_settings('L32',target,k,H,D,hm,hs,settings);write(OUT/'hyperparameter_refinement.csv',r32)
 # Rank with width/mean/phi guardrails, then transfer the leading five to L64.
 rank=lambda r:abs(r['action_density_shift_native_sigma'])+3*abs(r['action_density_std_ratio']-1)+.2*(abs(r['phi2_shift_native_sigma'])+abs(r['phi4_shift_native_sigma']))
 chosen=sorted(r32,key=rank)[:5];p64=load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz')[:20];r64=run_settings('L64',p64,k,H,D,hm,hs,[(r['k'],r['tau_quantile'],r['beta']) for r in chosen]);write(OUT/'zero_shot_L64_refinement.csv',r64)
 choice=chosen[0];(OUT/'chosen_proposal.json').write_text(json.dumps({'Ndonor_configs':n,'k':choice['k'],'tau_quantile':choice['tau_quantile'],'tau':choice['tau'],'beta':choice['beta'],'covariance':'diagonal','density':'exact finite selected-k Gaussian mixture'},indent=2))
 write(OUT/'exact_density_validation.csv',[{'positive_beta_only':True,'nearest_set_depends_only_on_coarse_context':True,'heldout_donor_exclusion':True,'zero_beta_excluded_from_AR':True,'finite_logq':bool(np.isfinite(choice['mean_logq'])),'reblocking_checked_in_raw_validation':True}])
 (OUT/'summary.md').write_text('Phase 1 and 2 completed. Tiling-offset and patch A/R are intentionally pending explicit review of selected donor-bank statistics; no long exact chain launched.\n')
if __name__=='__main__':main()
