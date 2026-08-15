#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
from train_lam1p0_flow_detail_pilot import load_kernel_matrix,load_phi,split_pairs
from train_lam1p0_autoregressive_detail_flow import torch_kernel_fft,torch_inverse_kernel
from train_lam1p0_local_multistage_r2_gate import LocalFlow
from train_lam1p0_local_multistage_rqspline import assemble_psi,destandardize,metrics_rows,write_csv
def action(x):
 p2=(x*x).mean((1,2));p4=(x**4).mean((1,2));nn=.5*((x*np.roll(x,-1,1)).mean((1,2))+(x*np.roll(x,-1,2)).mean((1,2)));return 1-2*p2+p4-4*.340301*nn
def main():
 out=Path('perfect_blocking_upsampling/runs/lam1p0/diagnostics/context_matched_residual_transplant_20260721');out.mkdir(parents=True,exist_ok=True);k,_=load_kernel_matrix(Path('perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json'));phi=load_phi(Path('data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz'))[:100];p=split_pairs(phi,k);ck=torch.load(Path('perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_local_residual_AR_R3_newkernel_N100_20260721/checkpoints/checkpoint_best.pt'),map_location='cpu',weights_only=False);m=LocalFlow('C');m.load_state_dict(ck['model_state']);st=ck['stats'];c=torch.from_numpy((p['coarse']-float(np.asarray(st['c']['mean']).reshape(-1)[0]))/float(np.asarray(st['c']['std']).reshape(-1)[0]));d=[torch.from_numpy((p['detail'][:,i]-float(np.asarray(st[x]['mean']).reshape(-1)[0]))/float(np.asarray(st[x]['std']).reshape(-1)[0])) for i,x in enumerate(('e1','e2','body'))]
 with torch.no_grad(): ps=[m.params_full(c,None,None,'edge'),m.params_full(c,d[0],None,'corr'),m.params_full(c,d[0],d[1],'body')];r=[(x-mu)*torch.exp(-s) for x,(_,mu,s) in zip(d,ps)]
 target=np.arange(90,100); donor=np.arange(80); rows=[]
 for level in ('c','c_patch','c_mu_sigma','full_context'):
  rr=[torch.empty_like(x[target]) for x in r]
  for ti,i in enumerate(target):
   # Configuration-level nearest context proxy, then aligned joint residual blocks; progressively add predictor information.
   f=[c[i].mean(),c[i].std()];
   if level!='c': f += [c[i].square().mean(),c[i].abs().mean()]
   if level in ('c_mu_sigma','full_context'): f += [ps[0][1][i].mean(),ps[1][1][i].mean(),ps[2][1][i].mean(),ps[0][2][i].mean(),ps[1][2][i].mean(),ps[2][2][i].mean()]
   if level=='full_context': f += [d[0][i].mean(),d[1][i].mean()]
   dist=[]
   for j in donor:
    q=[c[j].mean(),c[j].std()];
    if level!='c':q += [c[j].square().mean(),c[j].abs().mean()]
    if level in ('c_mu_sigma','full_context'):q += [ps[0][1][j].mean(),ps[1][1][j].mean(),ps[2][1][j].mean(),ps[0][2][j].mean(),ps[1][2][j].mean(),ps[2][2][j].mean()]
    if level=='full_context':q += [d[0][j].mean(),d[1][j].mean()]
    dist.append(float(sum((a-b).square() for a,b in zip(f,q))))
   j=donor[int(np.argmin(dist))]
   for s in range(3): rr[s][ti]=r[s][j]
  ds=[mu[target]+torch.exp(sig[target])*z for z,(_,mu,sig) in zip(rr,ps)];psi=assemble_psi(*[destandardize(v,st,x) for v,x in zip([c[target]]+ds,['c','e1','e2','body'])]);g=torch_inverse_kernel(psi,torch_kernel_fft(k,32,torch.device('cpu'))).detach().numpy();met={x['observable']:x for x in metrics_rows(phi[target],g,level,'whole')};a=action(g);rows.append({'match_level':level,'action_shift':met['action_density']['shift_native_sigma'],'action_std':met['action_density']['std_ratio'],'q05_occupancy':float((a<=np.quantile(action(phi[target]),.05)).mean()),'phi4_shift':met['phi4']['shift_native_sigma'],'kurtosis_shift':met['local_kurtosis_ratio']['shift_native_sigma']})
 write_csv(out/'context_matched_transplant_metrics.csv',rows);(out/'summary.md').write_text('Configuration-level context proxy completed; block-local nearest-neighbour matching remains required before a final conditional-information conclusion.\n')
if __name__=='__main__':main()
