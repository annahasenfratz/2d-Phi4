#!/usr/bin/env python3
"""Ring-1 conditional empirical-density overlap scan; no Markov chains."""
import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
from audit_lam1p0_boundary_transplant import boundary
from run_lam1p0_empirical_2x2_exact_patch import mm,get,N
from train_lam1p0_flow_detail_pilot import load_phi,load_kernel_matrix,split_pairs
import run_lam1p0_empirical_joint_2x2_mixture as e
OUT=ROOT/'perfect_blocking_upsampling/runs/lam1p0/empirical_joint_2x2_mixture_validation_20260721'
def wr(n,r):
 ks=[]
 for x in r:
  for k in x:
   if k not in ks:ks.append(k)
 with (OUT/n).open('w',newline='') as f:w=csv.DictWriter(f,ks);w.writeheader();w.writerows(r)
def main():
 k,_=load_kernel_matrix(e.KPATH);phi=load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz');pd=split_pairs(phi[:N],k);md=mm(N,16,(0,0));co=np.asarray([e.features(pd['coarse'],[(i,y,x)])[0] for i,y,x in md]);ri=np.asarray([boundary(pd['detail'],i,y,x,1) for i,y,x in md]);D=np.asarray([get(pd['detail'],i,y,x) for i,y,x in md]);cm,cs=co.mean(0),co.std(0)+1e-6;rm,rs=ri.mean(0),ri.std(0)+1e-6
 pt=split_pairs(phi[N:N+20],k);mt=mm(20,16,(0,0));tc=np.asarray([e.features(pt['coarse'],[(i,y,x)])[0] for i,y,x in mt]);tr=np.asarray([boundary(pt['detail'],i,y,x,1) for i,y,x in mt]);td=np.asarray([get(pt['detail'],i,y,x) for i,y,x in mt]);
 rows=[]
 for metric in ('equalized','group_balanced'):
  H=np.concatenate([(co-cm)/cs,(ri-rm)/rs],1);T=np.concatenate([(tc-cm)/cs,(tr-rm)/rs],1)
  if metric=='group_balanced':H=np.concatenate([H[:,:49]/np.sqrt(49),H[:,49:]/np.sqrt(H.shape[1]-49)],1);T=np.concatenate([T[:,:49]/np.sqrt(49),T[:,49:]/np.sqrt(T.shape[1]-49)],1)
  idx,d2=e.index_context(H,T,maxk=16)
  for kk in (4,8,16):
   for tq in (.1,.25,.5):
    tau=np.quantile(np.sqrt(d2[:,0]),tq)
    for beta in (.005,.01,.02,.05):
     sig=beta*(D.std(0)+1e-6);vals=[]
     for q in range(len(T)):
      ii=idx[q,:kk];w=np.exp(-(d2[q,:kk]-d2[q,0])/(2*tau*tau));w/=w.sum();z=(td[q]-D[ii])/sig;lp=-.5*(z*z).sum(1)-.5*(12*np.log(2*np.pi)+2*np.log(sig).sum());lo=np.log(np.sum(w*np.exp(lp-lp.max())))+lp.max();j=np.random.default_rng(q).choice(kk,p=w);new=D[ii[j]]+np.random.default_rng(q+99).normal(size=12)*sig;z2=(new-D[ii])/sig;lp2=-.5*(z2*z2).sum(1)-.5*(12*np.log(2*np.pi)+2*np.log(sig).sum());ln=np.log(np.sum(w*np.exp(lp2-lp2.max())))+lp2.max();vals.append(lo-ln)
     rows.append({'metric':metric,'k':kk,'tau_quantile':tq,'tau':tau,'beta':beta,'mean_logq_old_minus_new':float(np.mean(vals)),'q05':float(np.quantile(vals,.05)),'q50':float(np.quantile(vals,.5)),'q95':float(np.quantile(vals,.95)),'nearest_q50':float(np.quantile(np.sqrt(d2[:,0]),.5))})
 wr('ring1_hyperparameter_scan.csv',rows);wr('ring1_native_density_overlap.csv',rows)
 (OUT/'ring1_context_specification.md').write_text('h=[standardized 7x7 coarse, standardized ring-1 outside-block detail values for all sectors]. Ring excludes dy,dx in {0,1}x{0,1}. Group-balanced divides squared group distances by group dimension.\n')
 (OUT/'ring1_donor_bank_metadata.json').write_text(json.dumps({'Ndonor':N,'blocks':N*64,'k':[4,8,16],'beta':[.005,.01,.02,.05]},indent=2))
if __name__=='__main__':main()
