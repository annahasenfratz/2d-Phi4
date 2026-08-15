#!/usr/bin/env python3
"""Frozen-flow variational-KL update of a D4 5x5 L16->L32 kernel."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
PKG=ROOT/'perfect_blocking_upsampling'; sys.path[:0]=[str(PKG/'scripts')]
from train_lam1p0_flow_detail_pilot import load_phi
from run_lam1p0_l16to32_rqspline_zeroshot import build_model_from_checkpoint, stationary_stats

ETA=2.0**.125; ORBITS=((1,0),(1,1),(2,0),(2,1),(2,2))

def load_kernel(path: Path):
 d=json.loads(path.read_text());a=np.asarray(d['matrix'],float)
 if a.shape==(7,7) and np.allclose(a[[0,-1],:],0) and np.allclose(a[:,[0,-1]],0): a=a[1:-1,1:-1]
 if a.shape!=(5,5): raise ValueError(f'expected 5x5 kernel, got {a.shape}')
 return a,d

def mask(orbit):
 u=torch.arange(-2,3);x,y=torch.meshgrid(u,u,indexing='ij')
 return (torch.maximum(x.abs(),y.abs())==orbit[0])&(torch.minimum(x.abs(),y.abs())==orbit[1])

def assemble(c,d):
 psi=torch.empty((len(c),32,32),dtype=c.dtype,device=c.device)
 psi[:,0::2,0::2]=c;psi[:,0::2,1::2]=d[:,0];psi[:,1::2,0::2]=d[:,1];psi[:,1::2,1::2]=d[:,2]
 return psi

def inverse(psi,k):
 pad=torch.zeros((32,32),dtype=psi.dtype,device=psi.device);pad[:5,:5]=torch.fft.ifftshift(k)
 khat=torch.fft.fft2(pad).real
 return torch.fft.ifft2(torch.fft.fft2(psi)/khat).real,khat

def action(phi):
 bonds=(phi*torch.roll(phi,-1,1)).sum((1,2))+(phi*torch.roll(phi,-1,2)).sum((1,2))
 return -(phi*phi).sum((1,2))+(phi**4).sum((1,2))-2*.340301*bonds

def main():
 p=argparse.ArgumentParser();p.add_argument('--kernel',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out-kernel',type=Path,required=True);p.add_argument('--out-summary',type=Path,required=True);p.add_argument('--steps',type=int,default=40);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--n-coarse',type=int,default=2000);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=20260810);p.add_argument('--min-k',type=float,default=.5);p.add_argument('--max-inv-k',type=float,default=2.);p.add_argument('--guard-weight',type=float,default=1e5);a=p.parse_args()
 torch.manual_seed(a.seed);np.random.seed(a.seed);dev=torch.device('cpu')
 base,meta=load_kernel(ROOT/a.kernel);base=torch.tensor(base,dtype=torch.float32,device=dev);ms=[mask(o).to(dev) for o in ORBITS];mult=torch.tensor([int(m.sum()) for m in ms],dtype=torch.float32,device=dev)
 ck=torch.load(ROOT/a.checkpoint,map_location=dev,weights_only=False);model,_=build_model_from_checkpoint(ck,16,dev);model.eval()
 for q in model.parameters():q.requires_grad_(False)
 st=stationary_stats(ck['state']['stats'],16)
 cm,cs=float(st['coarse_mean']),float(st['coarse_std']);dm=torch.tensor(np.asarray(st['detail_mean']).reshape(1,3,1,1),dtype=torch.float32,device=dev);ds=torch.tensor(np.asarray(st['detail_std']).reshape(1,3,1,1),dtype=torch.float32,device=dev)
 coarse=torch.tensor(load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz')[:a.n_coarse],dtype=torch.float32,device=dev)
 delta=torch.nn.Parameter(torch.zeros(5,dtype=torch.float32,device=dev));opt=torch.optim.Adam([delta],lr=a.lr);hist=[];best=None
 for step in range(a.steps+1):
  idx=torch.randint(len(coarse),(a.batch_size,),device=dev);c=coarse[idx]
  with torch.no_grad():dn,_,_,_=model.sample((c-cm)/cs)
  k=base.clone()
  for z,m in zip(delta,ms):k=k+z*m
  k[2,2]=k[2,2]-torch.sum(delta*mult)
  phi,kh=inverse(assemble(c,dn*ds+dm),k);mink=kh.min();maxinv=1/mink
  kl=(action(phi)-action(c)).mean()/1024+torch.log(kh).sum()/1024
  guard=a.guard_weight*(torch.relu(a.min_k-mink)**2+torch.relu(maxinv-a.max_inv_k)**2);loss=kl+guard
  row={'step':step,'kl_per_site':float(kl.detach()),'loss':float(loss.detach()),'min_K':float(mink.detach()),'max_invK':float(maxinv.detach())};row.update({f'delta_{o[0]}{o[1]}':float(z.detach()) for o,z in zip(ORBITS,delta)});hist.append(row)
  if float(mink.detach())>=a.min_k and float(maxinv.detach())<=a.max_inv_k and (best is None or row['kl_per_site']<best[0]):best=(row['kl_per_site'],k.detach().cpu().numpy())
  if step==a.steps:break
  opt.zero_grad(set_to_none=True);loss.backward();opt.step()
 if best is None:raise RuntimeError('no guard-valid kernel')
 final=np.asarray(best[1],dtype=np.float64);final[2,2]+=ETA-final.sum();out=dict(meta);out['name']=str(meta.get('name','kernel'))+'_alternating_kl';out['matrix']=final.tolist();out['kernel_coefficients_include_eta_scale']=True;out['alternating_kl_optimization']={'source_kernel':str(a.kernel),'source_checkpoint':str(a.checkpoint),'objective':'E_q[(S_f-S_c+log|det K|)/L_f^2]','orbits':[f'{x}{y}' for x,y in ORBITS],'best_kl_per_site':best[0],'min_K_guard':a.min_k,'max_invK_guard':a.max_inv_k,'history':hist}
 kp=ROOT/a.out_kernel;sp=ROOT/a.out_summary;kp.parent.mkdir(parents=True,exist_ok=True);sp.parent.mkdir(parents=True,exist_ok=True);kp.write_text(json.dumps(out,indent=2)+'\n');sp.write_text(json.dumps(out['alternating_kl_optimization'],indent=2)+'\n');print(json.dumps({'out_kernel':str(kp),**out['alternating_kl_optimization']},indent=2))
if __name__=='__main__':main()
