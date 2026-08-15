#!/usr/bin/env python3
"""Exact global-independence A/R audit for direct L16 -> flowed L32 proposals."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2]; PKG=ROOT/'perfect_blocking_upsampling'; sys.path[:0]=[str(PKG/'src'),str(PKG/'scripts')]
from perfect_blocking_upsampling.actions import action_total
from perfect_blocking_upsampling.io import ActionSpec
from perfect_blocking_upsampling.kernels import inverse_kernel, load_kernel, kernel_fft_from_spec, kernel_stencil_from_spec
from train_lam1p0_flow_detail_pilot import assemble_psi, load_phi
from run_lam1p0_l16to32_rqspline_zeroshot import build_model_from_checkpoint, sample_model_lattice, stationary_stats


def observable_samples(phi: np.ndarray, action: ActionSpec) -> dict[str, np.ndarray]:
 """Small set of per-configuration observables for proposal diagnostics."""
 arr=np.asarray(phi,dtype=np.float64)
 phi2=(arr*arr).mean(axis=(1,2))
 phi4=(arr**4).mean(axis=(1,2))
 return {
  'action_density':action_total(arr,action)/(arr.shape[1]*arr.shape[2]),
  'phi2':phi2,
  'phi4':phi4,
  'NN':0.5*(arr*np.roll(arr,-1,axis=1)+arr*np.roll(arr,-1,axis=2)).mean(axis=(1,2)),
  'local_kurtosis_ratio':phi4/np.maximum(phi2*phi2,1.e-300),
 }

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--kernel',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=1000);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--seed',type=int,default=20260812);a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 c=load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz')[:a.n].astype(np.float32); spec,meta=load_kernel(ROOT/a.kernel); ck=torch.load(ROOT/a.checkpoint,map_location='cpu',weights_only=False);model,_=build_model_from_checkpoint(ck,lattice_size=16,device=torch.device('cpu'));stats=stationary_stats(ck['state']['stats'],lc=16); d,logqd,_,_=sample_model_lattice(model,c,stats,batch_size=a.batch_size,device=torch.device('cpu'),seed=a.seed); phi,_=inverse_kernel(assemble_psi(c,d),spec)
 ac=ActionSpec('phi4_nn',1.,.340301); sf=action_total(phi,ac);sc=action_total(c,ac)
 stencil=kernel_stencil_from_spec(spec); kt=kernel_fft_from_spec(stencil,phi.shape[-1],spec); logdetK=float(np.log(np.abs(kt)).sum())
 logw=-sf+sc-logqd-logdetK
 w=np.exp(logw-logw.max());w/=w.sum(); ess=1/(w@w)/len(w); rng=np.random.default_rng(a.seed+1);current=rng.choice(a.n,size=200000,p=w);proposal=rng.integers(a.n,size=200000);acc=float(np.minimum(1,np.exp(np.minimum(logw[proposal]-logw[current],0))).mean())
 np.savez_compressed(a.out/'global_ar_samples.npz',logw=logw,Sf=sf,Sc=sc,logq_detail=logqd)
 np.savez_compressed(a.out/'observable_samples.npz',**observable_samples(phi,ac))
 s={'checkpoint':str(a.checkpoint),'kernel':str(a.kernel),'n':a.n,'logdet_K':logdetK,'logw_mean':float(logw.mean()),'logw_std':float(logw.std(ddof=1)),'ess_over_n':float(ess),'stationary_independence_MH_acceptance_proxy':acc,'logw_q05':float(np.quantile(logw,.05)),'logw_q95':float(np.quantile(logw,.95))};(a.out/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
if __name__=='__main__':main()
