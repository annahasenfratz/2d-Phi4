#!/usr/bin/env python3
"""N100 direct-detail 2x2 block-causal K=4 full-Cholesky mixture check."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
from train_lam1p0_flow_detail_pilot import load_kernel_matrix,load_phi,split_pairs
from train_lam1p0_local_multistage_rqspline import write_csv,write_json
class Mix(nn.Module):
 def __init__(self,n):super().__init__();self.net=nn.Sequential(nn.Linear(n,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU(),nn.Linear(96,4+4*4+4*10))
 def forward(self,x):
  o=self.net(x);w=o[:,:4];mu=o[:,4:20].reshape(-1,4,4);z=o[:,20:].reshape(-1,4,10);L=torch.zeros(len(x),4,4,4,device=x.device);ii=torch.tril_indices(4,4);L[:, :, ii[0],ii[1]]=z;L[:, :, range(4),range(4)]=torch.nn.functional.softplus(L[:,:,range(4),range(4)])+.03;return w,mu,L
 def nll(self,x,y):
  w,m,L=self(x);d=y[:,None]-m;sol=torch.linalg.solve_triangular(L,d[...,None],upper=False).squeeze(-1);lp=-.5*(sol.square().sum(-1)+4*np.log(2*np.pi))-torch.log(torch.diagonal(L,dim1=-2,dim2=-1)).sum(-1);return -torch.logsumexp(torch.log_softmax(w,-1)+lp,-1).mean()
def main():
 out=Path('perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_block_causal_2x2_detail_mixture_20260721');out.mkdir(parents=True,exist_ok=True);phi=load_phi(Path('data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz'))[:100];k,_=load_kernel_matrix(Path('perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json'));p=split_pairs(phi,k);c=p['coarse'];d=p['detail'];rows=[]
 # 7x7 patch anchored at lower-left block site (2by,2bx); direct global standard detail blocks.
 for stage in range(3):
  X=[];Y=[]
  for i in range(100):
   for by in range(8):
    for bx in range(8):
     y,x=2*by,2*bx;f=[c[i,(y+dy)%16,(x+dx)%16] for dy in range(-3,4) for dx in range(-3,4)]
     if stage>0:f += [d[i,0,(2*((by+dy)%8)+u)%16,(2*((bx+dx)%8)+v)%16] for dy in(-1,0,1) for dx in(-1,0,1) for u in range(2) for v in range(2)]
     if stage>1:f += [d[i,1,(2*((by+dy)%8)+u)%16,(2*((bx+dx)%8)+v)%16] for dy in(-1,0,1) for dx in(-1,0,1) for u in range(2) for v in range(2)]
     X.append(f);Y.append(d[i,stage,y:y+2,x:x+2].reshape(-1))
  X=np.asarray(X,np.float32);Y=np.asarray(Y,np.float32);tr=np.repeat(np.arange(100),64)<80;xm,xs=X[tr].mean(0),X[tr].std(0)+1e-6;ym,ys=Y[tr].mean(0),Y[tr].std(0)+1e-6;X=(X-xm)/xs;Y=(Y-ym)/ys;m=Mix(X.shape[1]);opt=torch.optim.AdamW(m.parameters(),lr=1e-3);tx=torch.tensor(X[tr]);ty=torch.tensor(Y[tr]);vx=torch.tensor(X[~tr]);vy=torch.tensor(Y[~tr])
  for e in range(10):opt.zero_grad();loss=m.nll(tx,ty);loss.backward();opt.step();rows.append({'stage':stage,'epoch':e+1,'train_nll':float(loss),'validation_nll':float(m.nll(vx,vy))})
  torch.save({'model':m.state_dict(),'xmean':xm,'xstd':xs,'ymean':ym,'ystd':ys,'stage':stage},out/f'stage{stage}_K4_full.pt')
 write_csv(out/'training_history.csv',rows);write_json(out/'dataset_metadata.json',{'N':100,'split':'80/20','block':'2x2','context':'7x7 anchored lower-left offsets -3..3','K':4,'covariance':'full 4x4 Cholesky'});(out/'model_specification.md').write_text('Block-causal direct-detail mixture. q01(c7x7); q10(c7x7,D01 3x3 blocks); q11(c7x7,D01,D10 3x3 blocks).')
if __name__=='__main__':main()
