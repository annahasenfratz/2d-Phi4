import csv,json,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
import run_lam1p0_empirical_joint_2x2_mixture as e
from train_lam1p0_flow_detail_pilot import load_phi,load_kernel_matrix,split_pairs
OUT=ROOT/'perfect_blocking_upsampling/runs/lam1p0/fresh_disjoint_calibrated_empirical_validation_20260721'
def w(n,r):
 ks=[]
 for x in r:
  for k in x:
   if k not in ks:ks.append(k)
 with (OUT/n).open('w',newline='') as f:q=csv.DictWriter(f,ks);q.writeheader();q.writerows(r)
def gen(phi,k,H,D,hm,hs,seed):
 p=split_pairs(phi,k);M=e.meta(len(phi),p['coarse'].shape[1]);T=(e.features(p['coarse'],M)-hm)/hs;dist,idx=cKDTree(H).query(T,k=8);tau=np.quantile(dist[:,0],.25);rng=np.random.default_rng(seed);sig=.01*(D.std(0)+1e-6);v=np.empty((len(M),12),np.float32)
 for i in range(len(M)):
  ww=np.exp(-(dist[i]**2-dist[i,0]**2)/(2*tau*tau));ww/=ww.sum();v[i]=D[idx[i,rng.choice(8,p=ww)]]+rng.normal(size=12)*sig
 raw,_=e.reconstruct(p['coarse'],M,v,k,phi.shape[1]);d=split_pairs(raw,k)['detail'];mean=d.copy();mean[:,:2]*=.97;z=rng.normal(size=len(phi));full=mean.copy();full[:,:2]*=np.exp(.02*z)[:,None,None,None];
 import torch
 from train_lam1p0_autoregressive_detail_flow import torch_kernel_fft,torch_inverse_kernel
 from train_lam1p0_local_multistage_rqspline import assemble_psi
 psi=assemble_psi(torch.from_numpy(p['coarse']),*[torch.from_numpy(full[:,s]) for s in range(3)]);cal=torch_inverse_kernel(psi,torch_kernel_fft(k,phi.shape[1],torch.device('cpu'))).detach().numpy()
 return raw, e.reconstruct(p['coarse'],M,mean.reshape(len(M),12),k,phi.shape[1])[0],cal,z
def rows(native,x,label):
 m=e.metrics(native,x,label);return m
def main():
 OUT.mkdir(parents=True,exist_ok=True);k,_=load_kernel_matrix(e.KPATH);allp=load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz');pd=split_pairs(allp[:1000],k);M=e.meta(1000,16);h=e.features(pd['coarse'],M);hm,hs=h.mean(0),h.std(0)+1e-6;H=(h-hm)/hs;D=e.vectors(pd['detail'],M);tar=allp[2000:3000];raw,mean,cal,z=gen(tar,k,H,D,hm,hs,88);np.savez_compressed(OUT/'paired_fields_L32.npz',native=tar,raw=raw,mean_calibrated=mean,calibrated=cal,z=z,gamma=.97*np.exp(.02*z));
 for fn,x in [('raw_metrics_L32.csv',raw),('mean_calibrated_metrics_L32.csv',mean),('full_calibrated_metrics_L32.csv',cal)]:w(fn,[rows(tar,x,'L32')]);w('quantitative_comparison_L32.csv',[rows(tar,x,n) for n,x in [('raw',raw),('mean',mean),('full',cal)]])
 w('latent_draws.csv',[{'z':z[i],'gamma':.97*np.exp(.02*z[i]),'raw_phi2':(raw[i]**2).mean(),'raw_phi4':(raw[i]**4).mean(),'cal_phi2':(cal[i]**2).mean(),'cal_phi4':(cal[i]**4).mean()} for i in range(len(z))]);w('source_index_inventory.csv',[{'role':'donor','start':0,'stop':999},{'role':'calibration','start':1000,'stop':1999},{'role':'fresh_L32','start':2000,'stop':2999},{'overlap':False}]);(OUT/'calibration_specification.json').write_text(json.dumps({'gamma_mean':.97,'sigma_gamma':.02,'k':8,'beta':.01,'donors':1000},indent=2));(OUT/'validation_summary.md').write_text('Fresh disjoint L32 validation complete; L64 pending separate execution.\n')
if __name__=='__main__':main()
