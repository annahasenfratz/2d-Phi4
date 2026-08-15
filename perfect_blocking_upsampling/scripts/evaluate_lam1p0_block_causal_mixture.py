#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
from train_lam1p0_block_causal_mixture import Mix
from train_lam1p0_flow_detail_pilot import load_kernel_matrix,load_phi,split_pairs
from train_lam1p0_autoregressive_detail_flow import torch_kernel_fft,torch_inverse_kernel
from train_lam1p0_local_multistage_rqspline import assemble_psi,metrics_rows,write_csv
def features(c,d,stage,meta):
 X=[]
 for i,y,x in meta:
  f=[c[i,(y+dy)%c.shape[1],(x+dx)%c.shape[2]] for dy in range(-3,4) for dx in range(-3,4)]
  for s in range(stage):f += [d[s][i,(2*((y//2+dy)% (c.shape[1]//2))+u)%c.shape[1],(2*((x//2+dx)%(c.shape[2]//2))+v)%c.shape[2]] for dy in(-1,0,1) for dx in(-1,0,1) for u in range(2) for v in range(2)]
  X.append(f)
 return np.asarray(X,np.float32)
def main():
 run=Path('perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_block_causal_2x2_detail_mixture_20260721');k,_=load_kernel_matrix(Path('perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json'));rows=[]
 for label,path,n in [('L32',Path('data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz'),20),('L64',Path('data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz'),30)]:
  phi=load_phi(path)[80:80+n] if label=='L32' else load_phi(path)[:n];p=split_pairs(phi,k);c=p['coarse'];L=c.shape[1];meta=np.array([(i,y,x) for i in range(n) for y in range(0,L,2) for x in range(0,L,2)]);d=[np.zeros((n,L,L),np.float32) for _ in range(3)]
  for s in range(3):
   ck=torch.load(run/f'stage{s}_K4_full.pt',map_location='cpu',weights_only=False);m=Mix(len(ck['xmean']));m.load_state_dict(ck['model']);X=(features(c,d,s,meta)-ck['xmean'])/ck['xstd'];w,mu,chol=m(torch.tensor(X));cat=torch.distributions.Categorical(logits=w).sample();z=torch.randn(len(X),4);v=mu[torch.arange(len(X)),cat]+torch.bmm(chol[torch.arange(len(X)),cat],z[:,:,None]).squeeze(-1);v=(v.detach().numpy()*ck['ystd']+ck['ymean'])
   for q,(i,y,x) in enumerate(meta):d[s][i,y:y+2,x:x+2]=v[q].reshape(2,2)
  psi=assemble_psi(torch.from_numpy(c),*[torch.from_numpy(x) for x in d]);g=torch_inverse_kernel(psi,torch_kernel_fft(k,phi.shape[1],torch.device('cpu'))).numpy();rows += metrics_rows(phi,g,label,'free_running')
 write_csv(run/'free_running_metrics_L32.csv',[r for r in rows if r['label']=='L32']);write_csv(run/'zero_shot_metrics_L64.csv',[r for r in rows if r['label']=='L64'])
if __name__=='__main__':main()
