#!/usr/bin/env python3
"""Exact finite-k context-weighted empirical mixture for joint 2x2 detail blocks."""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
from train_lam1p0_flow_detail_pilot import load_phi,load_kernel_matrix,split_pairs
from train_lam1p0_autoregressive_detail_flow import torch_kernel_fft,torch_inverse_kernel
from train_lam1p0_local_multistage_rqspline import assemble_psi,metrics_rows

OUT=ROOT/'perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_empirical_joint_2x2_detail_mixture_N100_20260721'
KPATH=ROOT/'perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json'; BLOCK=2; SEED=20260721
def write(path,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with path.open('w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def meta(n,L):return np.asarray([(i,y,x) for i in range(n) for y in range(0,L,2) for x in range(0,L,2)],np.int32)
def features(c,M):
 L=c.shape[1];return np.asarray([[c[i,(y+dy)%L,(x+dx)%L] for dy in range(-3,4) for dx in range(-3,4)] for i,y,x in M],np.float32)
def vectors(detail,M):return np.asarray([detail[i,:,y:y+2,x:x+2].reshape(-1) for i,y,x in M],np.float32)
def index_context(donor_h,target_h,maxk=32,chunk=256):
 out_i=[];out_d=[]
 for s in range(0,len(target_h),chunk):
  dd=((target_h[s:s+chunk,None]-donor_h[None])**2).sum(-1);ii=np.argpartition(dd,maxk-1,axis=1)[:,:maxk];dv=np.take_along_axis(dd,ii,1);order=np.argsort(dv,axis=1);out_i.append(np.take_along_axis(ii,order,1));out_d.append(np.take_along_axis(dv,order,1))
 return np.concatenate(out_i),np.concatenate(out_d)
def sample(c,M,idx,d2,donorD,k,tau,beta,cov,seed):
 rng=np.random.default_rng(seed);D=np.zeros((len(M),12),np.float32);logs=[];selected=[];weights_all=[];scale=donorD.std(0)+1e-6
 if cov=='diag': transform=np.eye(12,dtype=np.float32);kernel_scale=scale
 else:
  _,_,vt=np.linalg.svd((donorD-donorD.mean(0))/scale,full_matrices=False);transform=vt.astype(np.float32);kernel_scale=np.std((donorD-donorD.mean(0))@transform.T,0)+1e-6
 sigma=beta*kernel_scale; logdet=float(2*np.log(sigma).sum()) if beta else float('-inf')
 for q in range(len(M)):
  inds=idx[q,:k];w=np.exp(-(d2[q,:k]-d2[q,0])/(2*tau*tau));w/=w.sum();z=int(rng.choice(k,p=w));eps=rng.normal(size=12).astype(np.float32);disp=(eps*sigma)@transform if cov=='pca' else eps*sigma;D[q]=donorD[inds[z]]+disp if beta else donorD[inds[z]]
  # finite selected donor set is the exact proposal definition.
  if beta:
   diff=(D[q]-donorD[inds]);zv=diff@transform.T if cov=='pca' else diff;lp=-.5*((zv/sigma)**2).sum(1)-.5*(12*np.log(2*np.pi)+logdet);logs.append(float(np.log(np.sum(w*np.exp(lp-lp.max()))+1e-300)+lp.max()))
  else: logs.append(float('nan'))
  selected.append(int(inds[z]));weights_all.append(w)
 return D,np.asarray(logs),np.asarray(selected),np.asarray(weights_all),sigma,logdet
def reconstruct(c,M,D,k,Lfine):
 d=np.zeros((c.shape[0],3,c.shape[1],c.shape[2]),np.float32)
 for q,(i,y,x) in enumerate(M):d[i,:,y:y+2,x:x+2]=D[q].reshape(3,2,2)
 psi=assemble_psi(torch.from_numpy(c),*[torch.from_numpy(d[:,s]) for s in range(3)])
 return torch_inverse_kernel(psi,torch_kernel_fft(k,Lfine,torch.device('cpu'))).detach().numpy(),d
def metrics(native,gen,label):
 a={r['observable']:r for r in metrics_rows(native,gen,label,'whole')};act_n=np.asarray([a['action_density']['native_mean']])
 row={'volume':label}
 for o in ('action_density','phi2','phi4','local_kurtosis_ratio','NN','diag','2nn'):
  for z in ('shift_native_sigma','std_ratio','KS','overlap'):row[f'{o}_{z}']=a[o][z]
 # per-config action tails using project action convention
 def action(x):return (1-x*x+.5*x**4-2*.340301*(x*np.roll(x,-1,1)+x*np.roll(x,-1,2))).mean((1,2))
 an,ag=action(native),action(gen)
 for q in (.01,.05,.10):row[f'action_q{int(q*100):02d}_occupancy']=float((ag<=np.quantile(an,q)).mean())
 row['nonfinite']=int((~np.isfinite(gen)).sum());return row
def adj_cov(d):
 rows=[]
 for s in range(3):
  b=d[:,s].reshape(len(d),d.shape[2]//2,2,d.shape[3]//2,2).transpose(0,1,3,2,4).mean((3,4))
  for ax,n in ((1,'y'),(2,'x')):rows.append({'stage':s,'direction':n,'adjacent_block_mean_covariance':float(np.cov(b.ravel(),np.roll(b,-1,axis=ax).ravel())[0,1])})
 return rows
def eval_grid(name,phi,k,donorH,donorD,hmean,hstd,settings,save_meta=False):
 p=split_pairs(phi,k);c=p['coarse'];M=meta(len(phi),c.shape[1]);H=(features(c,M)-hmean)/hstd;idx,d2=index_context(donorH,H); rows=[];covrows=[];metadata={}
 tau_ref=np.sqrt(d2[:,0]);taus=np.quantile(tau_ref,[.25,.5,.75])
 for kval in (4,8,16,32):
  for ti,tau in enumerate(taus):
   for beta in (0.,.02,.05,.10,.20):
    for cov in ('diag','pca'):
     D,lq,sel,w,sig,ld=sample(c,M,idx,d2,donorD,kval,float(tau),beta,cov,SEED+kval*100+ti*10+int(beta*100));gen,d=reconstruct(c,M,D,k,phi.shape[1]);r=metrics(phi,gen,name);r.update({'k':kval,'tau_quantile':('.25','.50','.75')[ti],'tau':float(tau),'beta':beta,'covariance':cov,'mean_log_q':float(np.nanmean(lq)) if beta else float('nan'),'logdet_kernel':float(ld),'density_type':'continuous_gaussian_mixture' if beta else 'singular_zero_noise_transplant'});rows.append(r)
     for z in adj_cov(d):z.update({'volume':name,'k':kval,'tau_quantile':('.25','.50','.75')[ti],'beta':beta,'covariance':cov});covrows.append(z)
     if save_meta and kval==8 and ti==1 and beta==.05 and cov=='diag':metadata={'donor_indices':sel,'weights':w,'logq':lq,'kernel_sigma':sig,'kernel_logdet':ld,'nearest_distance':np.sqrt(d2[:,0]),'target_meta':M}
 return rows,covrows,metadata
def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'plots').mkdir(exist_ok=True);assert hashlib.sha256(KPATH.read_bytes()).hexdigest()=='84013adc2235f9f89a12bc835c2f1cbe6185f1d50d5b1b86dc05d649990baace'
 k,_=load_kernel_matrix(KPATH);phi=load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz')[:100];p=split_pairs(phi,k);Mtrain=meta(80,16);Hraw=features(p['coarse'][:80],Mtrain);hm,hs=Hraw.mean(0),Hraw.std(0)+1e-6;H=(Hraw-hm)/hs;D=vectors(p['detail'][:80],Mtrain)
 rows32,cov32,info=eval_grid('L32',phi[80:100],k,H,D,hm,hs,None,True);phi64=load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz')[:20];rows64,cov64,_=eval_grid('L64',phi64,k,H,D,hm,hs,None,False)
 write(OUT/'free_running_metrics_L32.csv',rows32);write(OUT/'zero_shot_metrics_L64.csv',rows64);write(OUT/'neighboring_block_covariance.csv',cov32+cov64);np.savez_compressed(OUT/'sampling_metadata_reference_k8_tau50_beta05_diag.npz',**info)
 # zero bandwidth baseline is the selected empirical transplant; density is singular and deliberately not called continuous.
 (OUT/'model_specification.md').write_text('Finite exact selected-k empirical mixture over native 12D joint 2x2 detail blocks. q(D|h)=sum selected normalized context weights times N(D; donor D, fixed kernel covariance). Context is standardized 7x7 coarse patch only.\n')
 (OUT/'dataset_metadata.json').write_text(json.dumps({'N':100,'donor_configs':list(range(80)),'held_out_configs':list(range(80,100)),'donor_blocks':len(D),'context_dimension':49,'detail_dimension':12,'kernel_hash':hashlib.sha256(KPATH.read_bytes()).hexdigest(),'leave_one_configuration_out':'held-out target configs are absent from donor set'},indent=2))
 (OUT/'summary.md').write_text('Completed non-neural N100 empirical joint-2x2 KDE mixture grid: k=4,8,16,32; tau at held-out nearest-distance q25/q50/q75; beta=.02,.05,.10,.20; diagonal and fixed-PCA diagonal kernel covariance. No neural model trained.\n')
if __name__=='__main__':main()
