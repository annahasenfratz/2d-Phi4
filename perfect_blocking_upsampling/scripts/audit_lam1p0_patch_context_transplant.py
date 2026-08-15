#!/usr/bin/env python3
"""Leave-one-configuration-out 2x2 joint residual block transplant."""
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
def feat(c,i,y,x,mode):
 if mode=='amplitude': return np.array([c[i,y,x]])
 r=1 if mode=='patch3' else 3; return np.array([c[i,(y+dy)%16,(x+dx)%16] for dy in range(-r,r+1) for dx in range(-r,r+1)])
def main():
 out=Path('perfect_blocking_upsampling/runs/lam1p0/diagnostics/context_matched_residual_transplant_20260721/patch_nn');out.mkdir(parents=True,exist_ok=True);k,_=load_kernel_matrix(Path('perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json'));phi=load_phi(Path('data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz'))[:100];p=split_pairs(phi,k);ck=torch.load(Path('perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_local_residual_AR_R3_newkernel_N100_20260721/checkpoints/checkpoint_best.pt'),map_location='cpu',weights_only=False);m=LocalFlow('C');m.load_state_dict(ck['model_state']);st=ck['stats'];cn=(p['coarse']-float(np.asarray(st['c']['mean']).reshape(-1)[0]))/float(np.asarray(st['c']['std']).reshape(-1)[0]);c=torch.from_numpy(cn);d=[torch.from_numpy((p['detail'][:,i]-float(np.asarray(st[x]['mean']).reshape(-1)[0]))/float(np.asarray(st[x]['std']).reshape(-1)[0])) for i,x in enumerate(('e1','e2','body'))]
 with torch.no_grad(): ps=[m.params_full(c,None,None,'edge'),m.params_full(c,d[0],None,'corr'),m.params_full(c,d[0],d[1],'body')];r=[((x-mu)*torch.exp(-s)).numpy() for x,(_,mu,s) in zip(d,ps)]
 target=np.arange(90,100); donors=np.arange(90);rows=[]
 for mode in ('amplitude','patch3','patch7'):
  rr=[np.empty((10,16,16),np.float32) for _ in range(3)]
  for ti,i in enumerate(target):
   for y in range(0,16,2):
    for x in range(0,16,2):
     f=feat(cn,i,y,x,mode);best=(1e99,0,0,0)
     for j in donors:
      if j==i:continue
      for yy in range(0,16,2):
       for xx in range(0,16,2):
        z=np.mean((f-feat(cn,j,yy,xx,mode))**2)
        if z<best[0]:best=(z,j,yy,xx)
     _,j,yy,xx=best
     for s in range(3):rr[s][ti,y:y+2,x:x+2]=r[s][j,yy:yy+2,xx:xx+2]
  ds=[mu[target]+torch.exp(sig[target])*torch.from_numpy(z) for z,(_,mu,sig) in zip(rr,ps)];psi=assemble_psi(*[destandardize(v,st,x) for v,x in zip([c[target]]+ds,['c','e1','e2','body'])]);g=torch_inverse_kernel(psi,torch_kernel_fft(k,32,torch.device('cpu'))).detach().numpy();met={z['observable']:z for z in metrics_rows(phi[target],g,mode,'whole')};rows.append({'context':mode,'action_shift':met['action_density']['shift_native_sigma'],'action_std':met['action_density']['std_ratio'],'q05_occupancy':float((action(g)<=np.quantile(action(phi[target]),.05)).mean()),'phi4_shift':met['phi4']['shift_native_sigma']})
 write_csv(out/'per_patch_context_transplant.csv',rows)
if __name__=='__main__':main()
