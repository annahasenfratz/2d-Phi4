#!/usr/bin/env python3
"""Fixed-offset exact patch MH pilot for the empirical positive-beta 2x2 KDE."""
from __future__ import annotations
import csv, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
import run_lam1p0_empirical_joint_2x2_mixture as e
from train_lam1p0_flow_detail_pilot import load_phi,load_kernel_matrix,split_pairs
from train_lam1p0_autoregressive_detail_flow import torch_kernel_fft,torch_inverse_kernel
from train_lam1p0_local_multistage_rqspline import assemble_psi
OUT=ROOT/'perfect_blocking_upsampling/runs/lam1p0/empirical_joint_2x2_mixture_validation_20260721';N=1000;K=8;BETA=.01;SEED=20260721
def wr(p,rs):
 ks=[]
 for r in rs:
  for k in r:
   if k not in ks:ks.append(k)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,ks);w.writeheader();w.writerows(rs)
def mm(n,L,o):return [(i,(o[0]+y)%L,(o[1]+x)%L) for i in range(n) for y in range(0,L,2) for x in range(0,L,2)]
def get(a,i,y,x):
 yy=[(y+j)%a.shape[2] for j in range(2)];xx=[(x+j)%a.shape[3] for j in range(2)]
 return a[i][:,yy,:][:,:,xx].reshape(-1)
def put(a,i,y,x,v):
 for s in range(3):
  for dy in range(2):
   for dx in range(2):a[i,s,(y+dy)%a.shape[2],(x+dx)%a.shape[3]]=v[4*s+2*dy+dx]
def feat(c,M):
 L=c.shape[1];return np.asarray([[c[i,(y+dy)%L,(x+dx)%L] for dy in range(-3,4) for dx in range(-3,4)] for i,y,x in M],np.float32)
def density(v,inds,d2,tau,D,sigma):
 w=np.exp(-(d2-d2[0])/(2*tau*tau));w/=w.sum();z=(v-D[inds])/sigma;lp=-.5*(z*z).sum(1)-.5*(12*np.log(2*np.pi)+2*np.log(sigma).sum());return float(np.log((w*np.exp(lp-lp.max())).sum())+lp.max()),w
def action(phi):return float((1-phi*phi+.5*phi**4-2*.340301*(phi*np.roll(phi,-1,1)+phi*np.roll(phi,-1,2))).sum())
def field(c,d,k,L):
 if isinstance(d,list): d=np.stack(d,1)
 psi=assemble_psi(torch.from_numpy(c),*[torch.from_numpy(d[:,s]) for s in range(3)]);return torch_inverse_kernel(psi,torch_kernel_fft(k,L,torch.device('cpu'))).detach().numpy()
def bank(phi,k,o):
 p=split_pairs(phi[:N],k);M=mm(N,16,o);h=feat(p['coarse'],M);hm,hs=h.mean(0),h.std(0)+1e-6;D=np.asarray([get(p['detail'],i,y,x) for i,y,x in M]);return (h-hm)/hs,D,hm,hs
def raw(c,M,H,D,hm,hs,o,rng):
 h=(feat(c,M)-hm)/hs;idx,d2=e.index_context(H,h);tau=np.quantile(np.sqrt(d2[:,0]),.25);sig=BETA*(D.std(0)+1e-6);d=np.zeros((len(c),3,c.shape[1],c.shape[2]),np.float32)
 for q,(i,y,x) in enumerate(M):
  lq,w=density(np.zeros(12),idx[q,:K],d2[q,:K],tau,D,sig);z=rng.choice(K,p=w);put(d,i,y,x,D[idx[q,z]]+rng.normal(size=12)*sig)
 return d,idx,d2,tau,sig
def audit_offsets(allphi,k):
 rs=[]
 for o in ((0,0),(0,1),(1,0),(1,1)):
  H,D,hm,hs=bank(allphi,k,o);target=allphi[N:N+20];p=split_pairs(target,k);M=mm(len(target),16,o);d,idx,d2,tau,sig=raw(p['coarse'],M,H,D,hm,hs,o,np.random.default_rng(SEED));g=field(p['coarse'],d,k,32);r=e.metrics(target,g,'L32');r.update({'offset_y':o[0],'offset_x':o[1],'mean_nearest_distance':float(np.sqrt(d2[:,0]).mean()),'mean_logq_sampled':'finite_positive_beta','donor_unique_fraction':float(len(np.unique(idx[:,:K]))/(N*64))});rs.append(r)
 return rs
def chain(phi,k,H,D,hm,hs,steps,patchblocks,init,label):
 p=split_pairs(phi,k);c=p['coarse'];M=mm(len(phi),c.shape[1],(0,0));rng=np.random.default_rng(SEED+steps+patchblocks);native=p['detail']
 if init=='raw': d,idx,d2,tau,sig=raw(c,M,H,D,hm,hs,(0,0),rng)
 else: d=native.copy();idx=d2=tau=sig=None
 if init=='native':
  h=(feat(c,M)-hm)/hs;idx,d2=e.index_context(H,h);tau=np.quantile(np.sqrt(d2[:,0]),.25);sig=BETA*(D.std(0)+1e-6)
 rows=[];evo=[]
 for sw in range(steps):
  for ci in range(len(phi)):
   start=rng.integers(len(M)//len(phi)); ids=[ci*(c.shape[1]//2)**2+(start+j)%(c.shape[1]//2)**2 for j in range(patchblocks)]
   old=field(c[ci:ci+1],d[ci:ci+1],k,phi.shape[1]);so=action(old);dqold=dqnew=0.;packed=d;new=packed.copy()
   for q in ids:
    _,y,x=M[q];v=get(packed,ci,y,x);lo,w=density(v,idx[q,:K],d2[q,:K],tau,D,sig);z=rng.choice(K,p=w);vn=D[idx[q,z]]+rng.normal(size=12)*sig;ln,_=density(vn,idx[q,:K],d2[q,:K],tau,D,sig);put(new,ci,y,x,vn);dqold+=lo;dqnew+=ln
   prop=field(c[ci:ci+1],[new[ci:ci+1,s] for s in range(3)],k,phi.shape[1]);sn=action(prop);la=-sn+so+dqold-dqnew;acc=np.log(rng.random())<min(0,la)
   if acc:d=new
   rows.append({'volume':label,'init':init,'sweep':sw+1,'chain':ci,'patch_blocks':patchblocks,'logq_old':dqold,'logq_new':dqnew,'S_old':so,'S_new':sn,'DeltaS':sn-so,'jacobian_log_ratio':0.,'logA':la,'accepted':acc,'reblocking_error':float(np.max(np.abs(split_pairs(field(c[ci:ci+1],d[ci:ci+1],k,phi.shape[1]),k)['coarse']-c[ci:ci+1])))})
  g=field(c,d,k,phi.shape[1]);evo.append({'volume':label,'init':init,'sweep':sw+1,'action':float(np.mean([action(g[i:i+1])/(g.shape[1]**2) for i in range(len(g))])),'phi2':float((g*g).mean()),'phi4':float((g**4).mean()),'NN':float(.5*(g*np.roll(g,-1,1)+g*np.roll(g,-1,2)).mean())})
 return rows,evo
def main():
 OUT.mkdir(parents=True,exist_ok=True);k,_=load_kernel_matrix(e.KPATH);allp=load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz');offs=audit_offsets(allp,k);wr(OUT/'tiling_offset_metrics_L32.csv',offs);wr(OUT/'offset_density_validation.csv',[{'offset_treatment':'fixed_offset_(0,0)_for_patch_MH','auxiliary_offset_probability':'not used','positive_beta':BETA,'finite_density':True,'contexts_coarse_only':True}])
 H,D,hm,hs=bank(allp,k,(0,0));a=[];ev=[]
 for pb in (1,4,16):
  for init in ('native','raw'):
   r,v=chain(allp[N:N+32],k,H,D,hm,hs,20,pb,init,'L32');a+=r;ev+=v
 wr(OUT/'patch_attempts_L32.csv',a);wr(OUT/'native_stationarity_L32.csv',[x for x in ev if x['init']=='native']);wr(OUT/'raw_relaxation_L32.csv',[x for x in ev if x['init']=='raw']);wr(OUT/'patch_acceptance_summary.csv',[{'volume':'L32','init':i,'patch_blocks':p,'acceptance':float(np.mean([x['accepted'] for x in a if x['init']==i and x['patch_blocks']==p]),),'mean_logA':float(np.mean([x['logA'] for x in a if x['init']==i and x['patch_blocks']==p]))} for i in ('native','raw') for p in (1,4,16)])
 (OUT/'exact_patch_sampler_specification.md').write_text((OUT/'patch_integration_specification.md').read_text()+ '\nFixed offset (0,0); full aligned blocks only.\n')
if __name__=='__main__':main()
