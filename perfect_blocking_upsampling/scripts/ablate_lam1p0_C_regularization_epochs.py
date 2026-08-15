#!/usr/bin/env python3
"""Three-epoch matched C loss ablation with per-epoch tail coverage checks."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling'/'scripts'))
from train_lam1p0_flow_detail_pilot import load_kernel_matrix,load_phi,split_pairs
from train_lam1p0_autoregressive_detail_flow import torch_kernel_fft
from train_lam1p0_local_multistage_rqspline import make_dataset,write_csv
from train_lam1p0_local_multistage_r2_gate import LocalFlow,run_epoch,sample_phi

def action(x):
 p2=(x*x).mean((1,2));p4=(x**4).mean((1,2));nn=.5*((x*np.roll(x,-1,1)).mean((1,2))+(x*np.roll(x,-1,2)).mean((1,2)));return 1-2*p2+p4-4*.340301*nn
def coverage(native,generated):
 n,g=action(native),action(generated);return {f'q{int(q*100):02d}_occupancy':float((g<=np.quantile(n,q)).mean()) for q in(.01,.05,.10)}
def generated(model,c,stats,kernel,fine,seed):
 fft=torch_kernel_fft(kernel,fine,torch.device('cpu'))
 with torch.no_grad(): x,_=sample_phi(model,c,stats,fft,torch.Generator().manual_seed(seed))
 return x.numpy()
def main():
 root=Path('perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_local_multistage_neighbor_details_R3R4_newkernel_20260721/candidate_C_R3/regularization_ablation');root.mkdir(parents=True,exist_ok=True)
 kernel,_=load_kernel_matrix(Path('perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json'));seed=2026072122
 p32=load_phi(Path('data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz'));idx=np.random.default_rng(seed).permutation(len(p32))[:100]; data=make_dataset(p32[idx],kernel,np.arange(80)); train=np.arange(80);test=np.arange(90,100)
 p64=load_phi(Path('data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz'));idx64=np.random.default_rng(seed).permutation(len(p64))[:30];n64=p64[idx64];c64=split_pairs(n64,kernel)['coarse']; s=data.stats['c'];c64=torch.from_numpy((c64-float(np.asarray(s['mean']).reshape(-1)[0]))/float(np.asarray(s['std']).reshape(-1)[0]))
 fft32=torch_kernel_fft(kernel,32,torch.device('cpu'));torch.manual_seed(seed);initial=LocalFlow('C').state_dict();rows=[]
 for name,scale in (('primary',1.0),('pure_nll',0.0)):
  model=LocalFlow('C');model.load_state_dict(initial);opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
  for epoch in range(1,4):
   tr,_=run_epoch(model,data,train,opt,fft32,epoch,True,reg_scale=scale);g32=generated(model,torch.from_numpy(data.c[test]),data.stats,kernel,32,seed+epoch);g64=generated(model,c64,data.stats,kernel,64,seed+100+epoch)
   row={'variant':name,'epoch':epoch,'reg_scale':scale,'train_nll':tr['nll'],'train_loss':tr['loss'],**{'L32_'+k:v for k,v in coverage(data.phi[test],g32).items()},**{'L64_'+k:v for k,v in coverage(n64,g64).items()}};rows.append(row);print(json.dumps(row),flush=True)
   torch.save({'model_state':model.state_dict(),'stats':data.stats,'epoch':epoch,'variant':name},root/f'{name}_epoch{epoch:02d}.pt')
   if row['L32_q05_occupancy']>0 and row['L64_q05_occupancy']>0:
    torch.save({'model_state':model.state_dict(),'stats':data.stats,'epoch':epoch,'variant':name},root/f'earliest_coverage_{name}.pt')
 write_csv(root/'epoch_tail_coverage.csv',rows);(root/'status.md').write_text('Three epochs completed for primary and pure-NLL variants. Earliest checkpoints with simultaneous nonzero L32/L64 q05 coverage are retained when present.\n')
if __name__=='__main__':main()
