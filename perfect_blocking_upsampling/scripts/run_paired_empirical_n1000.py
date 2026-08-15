import csv,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
import run_lam1p0_empirical_joint_2x2_mixture as e
from train_lam1p0_flow_detail_pilot import load_phi,load_kernel_matrix,split_pairs
OUT=ROOT/'perfect_blocking_upsampling/runs/lam1p0/paired_nativeL32_blockL16_reupscaleL32_N1000_20260721'
def w(n,r):
 ks=[]
 for x in r:
  for k in x:
   if k not in ks:ks.append(k)
 with (OUT/n).open('w',newline='') as f:q=csv.DictWriter(f,ks);q.writeheader();q.writerows(r)
def main():
 OUT.mkdir(parents=True,exist_ok=True);k,_=load_kernel_matrix(e.KPATH);phi=load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz');don=phi[:1000];tar=phi[1000:2000];pd=split_pairs(don,k);md=e.meta(1000,16);H=e.features(pd['coarse'],md);hm,hs=H.mean(0),H.std(0)+1e-6;H=(H-hm)/hs;D=e.vectors(pd['detail'],md);pt=split_pairs(tar,k);mt=e.meta(1000,16);T=(e.features(pt['coarse'],mt)-hm)/hs;t=time.time();dist,idx=cKDTree(H).query(T,k=8);tau=np.quantile(dist[:,0],.25);rng=np.random.default_rng(20260721);new=np.zeros((len(mt),12),np.float32);lq=[];sig=.01*(D.std(0)+1e-6)
 for q in range(len(mt)):
  ww=np.exp(-(dist[q]**2-dist[q,0]**2)/(2*tau*tau));ww/=ww.sum();z=rng.choice(8,p=ww);new[q]=D[idx[q,z]]+rng.normal(size=12)*sig
  zz=(new[q]-D[idx[q]])/sig;lp=-.5*(zz*zz).sum(1)-.5*(12*np.log(2*np.pi)+2*np.log(sig).sum());lq.append(np.log((ww*np.exp(lp-lp.max())).sum())+lp.max())
 gen,_=e.reconstruct(pt['coarse'],mt,new,k,32);rn=e.metrics(tar,gen,'L32');w('donor_target_index_inventory.csv',[{'role':'donor','start':0,'stop':999,'count':1000},{'role':'target','start':1000,'stop':1999,'count':1000},{'overlap':False}]);w('quantitative_comparison_unpaired.csv',[rn|{'mean_logq':float(np.mean(lq)),'runtime_sec':time.time()-t,'max_reblocking_error':float(np.max(np.abs(split_pairs(gen,k)['coarse']-pt['coarse'])))}]);
 rows=[]
 for name,fn in [('action',lambda x:(1-x*x+.5*x**4-2*.340301*(x*np.roll(x,-1,1)+x*np.roll(x,-1,2))).mean((1,2))),('phi2',lambda x:(x*x).mean((1,2))),('phi4',lambda x:(x**4).mean((1,2))),('NN',lambda x:(x*np.roll(x,-1,1)+x*np.roll(x,-1,2)).mean((1,2)))]:
  a,b=fn(tar),fn(gen);d=b-a;rows.append({'observable':name,'paired_mean':d.mean(),'paired_se':d.std(ddof=1)/np.sqrt(len(d)),'paired_z':d.mean()/(d.std(ddof=1)/np.sqrt(len(d))),'correlation':np.corrcoef(a,b)[0,1],'width_ratio':b.std(ddof=1)/a.std(ddof=1)})
 w('quantitative_comparison_paired.csv',rows);np.savez_compressed(OUT/'paired_fields.npz',native=tar,upscaled=gen);(OUT/'summary.md').write_text('Paired N1000 raw empirical comparison; no rethermalization. Exact cKDTree k=8 donor selection.\n')
if __name__=='__main__':main()
