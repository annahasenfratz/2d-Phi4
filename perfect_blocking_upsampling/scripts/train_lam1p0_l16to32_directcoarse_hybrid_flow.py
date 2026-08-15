#!/usr/bin/env python3
"""Conservative direct-L16 observable refinement with a conditional-NLL anchor.

Direct L16 fields provide the physical coarse distribution used in production.
Blocked native L32 pairs provide an auxiliary conditional-NLL anchor.  Model
selection uses a fixed large direct-L16 validation set and several independent
latent draws per coarse configuration to prevent one-draw validation overfit.
"""
from __future__ import annotations
import argparse, copy, csv, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];PKG=ROOT/'perfect_blocking_upsampling';sys.path[:0]=[str(PKG/'src'),str(PKG/'scripts')]
from perfect_blocking_upsampling.kernels import load_kernel
from train_lam1p0_flow_detail_pilot import load_phi, split_pairs
from train_lam1p0_flow_detail_localreg import torch_inverse_kernel
from run_lam1p0_l16to32_rqspline_zeroshot import build_model_from_checkpoint, stationary_stats
NAMES=('action_density','phi2','phi4','NN','local_kurtosis_ratio'); ND=3*16*16
def obs(phi):
 p2=(phi*phi).mean((1,2));p4=(phi**4).mean((1,2));nn=.5*((phi*torch.roll(phi,-1,1)).mean((1,2))+(phi*torch.roll(phi,-1,2)).mean((1,2)))
 return {'action_density':-p2+p4-4*.340301*nn,'phi2':p2,'phi4':p4,'NN':nn,'local_kurtosis_ratio':p4/torch.clamp(p2*p2,min=1e-12)}
def sample_obs(coarse,model,stats,kt):
 cm,cs=float(stats['coarse_mean']),float(stats['coarse_std']);dtype=coarse.dtype;dev=coarse.device
 dm=torch.as_tensor(np.asarray(stats['detail_mean']).reshape(1,3,1,1),dtype=dtype,device=dev);ds=torch.as_tensor(np.asarray(stats['detail_std']).reshape(1,3,1,1),dtype=dtype,device=dev)
 d,_,_,_=model.sample((coarse-cm)/cs);d=d*ds+dm;psi=torch.empty((len(coarse),32,32),dtype=dtype,device=dev);psi[:,0::2,0::2]=coarse;psi[:,0::2,1::2]=d[:,0];psi[:,1::2,0::2]=d[:,1];psi[:,1::2,1::2]=d[:,2]
 return obs(torch_inverse_kernel(psi,kt))
def loss_of(g,target,weights):
 out=next(iter(g.values())).new_zeros(());log={}
 for n in NAMES:
  x=g[n];mean=x.mean();std=x.std(unbiased=False);tm,ts=target[n];zm=(mean-tm)/ts;zs=(std-ts)/ts;out+=weights[n]*(zm*zm+.25*zs*zs);log.update({f'{n}_mean':float(mean.detach()),f'{n}_std':float(std.detach()),f'{n}_zmean':float(zm.detach()),f'{n}_zstd':float(zs.detach())})
 return out,log
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--source-checkpoint',type=Path,required=True);p.add_argument('--kernel',type=Path,required=True);p.add_argument('--n',type=int,default=5000);p.add_argument('--train-count',type=int,default=2500);p.add_argument('--val-count',type=int,default=2500);p.add_argument('--validation-draws',type=int,default=4);p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--lr',type=float,default=5e-7);p.add_argument('--nll-anchor-weight',type=float,default=.01);p.add_argument('--seed',type=int,default=2026081223);a=p.parse_args()
 torch.manual_seed(a.seed);np.random.seed(a.seed);run=ROOT/a.run_dir;run.mkdir(parents=True,exist_ok=True)
 c=np.asarray(load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz')[:a.n],np.float32);f=np.asarray(load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz')[:a.n],np.float32)
 ck=torch.load(ROOT/a.source_checkpoint,map_location='cpu',weights_only=False);model,_=build_model_from_checkpoint(ck,16,torch.device('cpu'));stats=stationary_stats(ck['state']['stats'],16);spec,_=load_kernel(ROOT/a.kernel);st=np.asarray(spec.matrix,np.float32);pad=np.zeros((32,32),np.float32);pad[:st.shape[0],:st.shape[1]]=np.fft.ifftshift(st);kt=torch.fft.fft2(torch.tensor(pad)).real
 pairs=split_pairs(f,st);cm,cs=float(stats['coarse_mean']),float(stats['coarse_std']);dm=np.asarray(stats['detail_mean'],np.float32).reshape(1,3,1,1);ds=np.asarray(stats['detail_std'],np.float32).reshape(1,3,1,1);pair_c=((pairs['coarse']-cm)/cs).astype(np.float32);pair_d=((pairs['detail']-dm)/ds).astype(np.float32);jac=-float(16*16*np.log(ds.reshape(3)).sum())
 r=obs(torch.tensor(f));target={k:(float(r[k].mean()),float(max(r[k].std(unbiased=True),torch.tensor(1e-6)))) for k in NAMES};weights={'action_density':.10,'phi2':.03,'phi4':.08,'NN':.02,'local_kurtosis_ratio':.04};train=np.arange(a.train_count);val=np.arange(a.train_count,min(a.train_count+a.val_count,a.n));opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5);rng=np.random.default_rng(a.seed);hist=[];best=(float('inf'),None,None)
 def evaluate(epoch):
  model.eval();chunks=[]
  with torch.no_grad():
   for draw in range(a.validation_draws):
    torch.manual_seed(a.seed+10000+100*epoch+draw);chunks.append(sample_obs(torch.tensor(c[val]),model,stats,kt))
   g={k:torch.cat([q[k] for q in chunks]) for k in NAMES};return loss_of(g,target,weights)
 for epoch in range(a.epochs+1):
  if epoch:
   model.train();perm=rng.permutation(train);tot=0.;nobs=0.;nnll=0.
   for start in range(0,len(perm),a.batch_size):
    ix=perm[start:start+a.batch_size];coarse=torch.tensor(c[ix]);pc=torch.tensor(pair_c[ix]);pd=torch.tensor(pair_d[ix]);opt.zero_grad(set_to_none=True);physical, _=loss_of(sample_obs(coarse,model,stats,kt),target,weights);nll=-(model.log_prob(pc,pd)+jac).mean()/ND;loss=physical+a.nll_anchor_weight*nll;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step();tot+=float(loss.detach())*len(ix);nobs+=float(physical.detach())*len(ix);nnll+=float(nll.detach())*len(ix)
  v,log=evaluate(epoch);row={'epoch':epoch,'validation_physical_loss':float(v.detach()),**log}
  if epoch:row|={'train_total_loss':tot/len(train),'train_physical_loss':nobs/len(train),'train_nll_per_detail':nnll/len(train)}
  hist.append(row);print(json.dumps(row),flush=True)
  if row['validation_physical_loss']<best[0]:best=(row['validation_physical_loss'],copy.deepcopy(model.state_dict()),epoch)
 model.load_state_dict(best[1]);out=copy.deepcopy(ck);out['model_state']=model.state_dict();out['optimizer_state']=opt.state_dict();out['epoch']=best[2];out['absolute_epoch']=best[2];out['history']=hist;out['checkpoint_metadata']={'selection':'large_multidraw_directL16_physical_loss_with_paired_NLL_anchor','best_epoch':best[2],'loss':best[0],'validation_draws':a.validation_draws,'nll_anchor_weight':a.nll_anchor_weight};out['config']=dict(out.get('config',{}))|{'mode':'direct_L16_hybrid_observable_NLL_anchor','source_checkpoint':str(a.source_checkpoint),'kernel':str(a.kernel)};(run/'checkpoints').mkdir(exist_ok=True);torch.save(out,run/'checkpoints/checkpoint_best_directL16_hybrid.pt')
 fields=list(dict.fromkeys(k for row in hist for k in row));
 with (run/'training_history.csv').open('w',newline='') as h:w=csv.DictWriter(h,fields);w.writeheader();w.writerows(hist)
 (run/'summary.json').write_text(json.dumps({'best_epoch':best[2],'best_validation_physical_loss':best[0],'validation_draws':a.validation_draws,'nll_anchor_weight':a.nll_anchor_weight,'target':target,'weights':weights},indent=2)+'\n');print(json.dumps({'status':'completed','best_epoch':best[2],'loss':best[0]},indent=2))
if __name__=='__main__':main()
