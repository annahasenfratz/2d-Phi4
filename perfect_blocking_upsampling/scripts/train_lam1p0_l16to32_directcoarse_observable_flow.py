#!/usr/bin/env python3
"""Fine-tune q(d|c) on direct-L16 coarse fields by generated L32 observables.

This is deliberately *not* conditional NLL: direct L16 fields are unpaired
with L32 fields.  At each step, c is drawn from direct L16, d~q_theta(d|c),
phi=K^{-1}(c,d), and its observable moments are matched to an independent
direct-L32 ensemble.  The selected checkpoint is the lowest held-out physical
moment loss, not the lowest paired-detail NLL.
"""
from __future__ import annotations

import argparse, copy, csv, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
PKG=ROOT/'perfect_blocking_upsampling';sys.path[:0]=[str(PKG/'src'),str(PKG/'scripts')]
from perfect_blocking_upsampling.io import ActionSpec
from perfect_blocking_upsampling.kernels import load_kernel
from train_lam1p0_flow_detail_pilot import load_phi
from train_lam1p0_flow_detail_localreg import torch_inverse_kernel, torch_kernel_fft
from run_lam1p0_l16to32_rqspline_zeroshot import build_model_from_checkpoint, stationary_stats

NAMES=('action_density','phi2','phi4','NN','local_kurtosis_ratio')

def obs(phi:torch.Tensor)->dict[str,torch.Tensor]:
 p2=(phi*phi).mean((1,2));p4=(phi**4).mean((1,2))
 nn=.5*((phi*torch.roll(phi,-1,1)).mean((1,2))+(phi*torch.roll(phi,-1,2)).mean((1,2)))
 return {'action_density':-p2+p4-4*.340301*nn,'phi2':p2,'phi4':p4,'NN':nn,'local_kurtosis_ratio':p4/torch.clamp(p2*p2,min=1e-12)}

def physical(phi:torch.Tensor, coarse:torch.Tensor, model, stats, kt:torch.Tensor)->dict[str,torch.Tensor]:
 cm,cs=float(stats['coarse_mean']),float(stats['coarse_std'])
 dm=torch.as_tensor(np.asarray(stats['detail_mean']).reshape(1,3,1,1),dtype=phi.dtype,device=phi.device)
 ds=torch.as_tensor(np.asarray(stats['detail_std']).reshape(1,3,1,1),dtype=phi.dtype,device=phi.device)
 d,_,_,_=model.sample((coarse-cm)/cs);d=d*ds+dm
 psi=torch.empty((len(coarse),32,32),dtype=phi.dtype,device=phi.device)
 psi[:,0::2,0::2]=coarse;psi[:,0::2,1::2]=d[:,0];psi[:,1::2,0::2]=d[:,1];psi[:,1::2,1::2]=d[:,2]
 return obs(torch_inverse_kernel(psi,kt))

def moment_loss(g:dict[str,torch.Tensor], target:dict[str,tuple[float,float]], weights:dict[str,float])->tuple[torch.Tensor,dict[str,float]]:
 loss=next(iter(g.values())).new_zeros(());log={}
 for name in NAMES:
  x=g[name];mean=x.mean();std=x.std(unbiased=False);tmean,tstd=target[name]
  zmean=(mean-tmean)/tstd;zstd=(std-tstd)/tstd
  term=weights[name]*(zmean*zmean+.25*zstd*zstd);loss=loss+term
  log.update({f'{name}_mean':float(mean.detach()),f'{name}_std':float(std.detach()),f'{name}_zmean':float(zmean.detach()),f'{name}_zstd':float(zstd.detach())})
 return loss,log

def main()->None:
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--source-checkpoint',type=Path,required=True);p.add_argument('--kernel',type=Path,required=True);p.add_argument('--n',type=int,default=5000);p.add_argument('--train-count',type=int,default=4000);p.add_argument('--val-count',type=int,default=500);p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--lr',type=float,default=2e-6);p.add_argument('--seed',type=int,default=2026081219);a=p.parse_args()
 torch.manual_seed(a.seed);np.random.seed(a.seed);dev=torch.device('cpu');run=ROOT/a.run_dir;run.mkdir(parents=True,exist_ok=True)
 c=load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz')[:a.n].astype(np.float32);f=load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz')[:a.n].astype(np.float32)
 ck=torch.load(ROOT/a.source_checkpoint,map_location=dev,weights_only=False);model,_=build_model_from_checkpoint(ck,16,dev);stats=stationary_stats(ck['state']['stats'],16);spec,_=load_kernel(ROOT/a.kernel);stencil=np.asarray(spec.matrix,dtype=np.float32);pad=np.zeros((32,32),np.float32);pad[:stencil.shape[0],:stencil.shape[1]]=np.fft.ifftshift(stencil);kt=torch.fft.fft2(torch.tensor(pad)).real
 ref=obs(torch.tensor(f));target={k:(float(ref[k].mean()),float(max(ref[k].std(unbiased=True),torch.tensor(1e-6)))) for k in NAMES};weights={'action_density':.10,'phi2':.03,'phi4':.08,'NN':.02,'local_kurtosis_ratio':.04}
 train=np.arange(min(a.train_count,len(c)));val=np.arange(len(train),min(len(train)+a.val_count,len(c)));opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5);rng=np.random.default_rng(a.seed);hist=[];best=(float('inf'),None,None)
 def evaluate(idx,seed):
  model.eval();torch.manual_seed(seed)
  with torch.no_grad():
   g=physical(torch.empty(0),torch.tensor(c[idx]),model,stats,kt);loss,log=moment_loss(g,target,weights)
  return float(loss),log
 for epoch in range(a.epochs+1):
  if epoch:
   model.train();perm=rng.permutation(train);total=0.;count=0
   for start in range(0,len(perm),a.batch_size):
    ix=perm[start:start+a.batch_size];coarse=torch.tensor(c[ix]);opt.zero_grad(set_to_none=True);g=physical(torch.empty(0),coarse,model,stats,kt);loss,_=moment_loss(g,target,weights);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step();total+=float(loss.detach())*len(ix);count+=len(ix)
  v,vlog=evaluate(val,a.seed+1000+epoch);row={'epoch':epoch,'validation_physical_loss':v,**vlog};
  if epoch:row['train_physical_loss']=total/count
  hist.append(row);print(json.dumps(row),flush=True)
  if v<best[0]:best=(v,copy.deepcopy(model.state_dict()),epoch)
 model.load_state_dict(best[1]);out=copy.deepcopy(ck);out['model_state']=model.state_dict();out['optimizer_state']=opt.state_dict();out['epoch']=best[2];out['absolute_epoch']=best[2];out['history']=hist;out['checkpoint_metadata']={'selection':'best_direct_L16_generated_observable_loss','best_epoch':best[2],'best_validation_physical_loss':best[0],'objective':'direct_L16 c; d~q(d|c); K^-1; match direct_L32 observable moments'};out['config']=dict(out.get('config',{}))|{'mode':'direct_L16_generated_observable_finetune','source_checkpoint':str(a.source_checkpoint),'kernel':str(a.kernel)}
 (run/'checkpoints').mkdir();torch.save(out,run/'checkpoints/checkpoint_best_directL16_observables.pt');
 with (run/'training_history.csv').open('w',newline='') as h:
  fields=list(dict.fromkeys(key for row in hist for key in row))
  w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(hist)
 (run/'summary.json').write_text(json.dumps({'best_epoch':best[2],'best_validation_physical_loss':best[0],'target':target,'weights':weights,'checkpoint':str(run/'checkpoints/checkpoint_best_directL16_observables.pt')},indent=2)+'\n')
 print(json.dumps({'status':'completed','best_epoch':best[2],'loss':best[0]},indent=2))
if __name__=='__main__':main()
