#!/usr/bin/env python3
"""Boundary-conditioned native joint-block transplant diagnostic; no chain sampling."""
import csv,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
from run_lam1p0_empirical_2x2_exact_patch import get,put,mm,feat,field,action,N,K
from train_lam1p0_flow_detail_pilot import load_phi,load_kernel_matrix,split_pairs
import run_lam1p0_empirical_joint_2x2_mixture as e
OUT=ROOT/'perfect_blocking_upsampling/runs/lam1p0/empirical_joint_2x2_mixture_validation_20260721'
def wr(n,r):
 ks=[]
 for x in r:
  for k in x:
   if k not in ks:ks.append(k)
 with (OUT/n).open('w',newline='') as f:w=csv.DictWriter(f,ks);w.writeheader();w.writerows(r)
def boundary(d,i,y,x,w):
 L=d.shape[2];v=[]
 for s in range(3):
  for dy in range(-w,2+w):
   for dx in range(-w,2+w):
    if not(0<=dy<2 and 0<=dx<2):v.append(d[i,s,(y+dy)%L,(x+dx)%L])
 return v
def main():
 k,_=load_kernel_matrix(e.KPATH);phi=load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz');pd=split_pairs(phi[:N],k);md=mm(N,16,(0,0));base=feat(pd['coarse'],md);rows=[]
 for name,width in [('coarse',0),('ring1',1),('ring2',2),('bond_summary',1)]:
  hd=[]
  for i,y,x in md:
   z=list(base[len(hd)])
   if width:z+=boundary(pd['detail'],i,y,x,width)
   if name=='bond_summary':z+=[np.mean(boundary(pd['detail'],i,y,x,1)),np.std(boundary(pd['detail'],i,y,x,1))]
   hd.append(z)
  hd=np.asarray(hd);hm,hs=hd.mean(0),hd.std(0)+1e-6;hd=(hd-hm)/hs
  pt=split_pairs(phi[N:N+20],k);mt=mm(20,16,(0,0));ht=[]
  for q,(i,y,x) in enumerate(mt):
   z=list(feat(pt['coarse'],[(i,y,x)])[0])
   if width:z+=boundary(pt['detail'],i,y,x,width)
   if name=='bond_summary':z+=[np.mean(boundary(pt['detail'],i,y,x,1)),np.std(boundary(pt['detail'],i,y,x,1))]
   ht.append(z)
  ht=(np.asarray(ht)-hm)/hs;idx,d2=e.index_context(hd,ht,maxk=8);ds=[]
  for q,(i,y,x) in enumerate(mt):
   old=field(pt['coarse'][i:i+1],pt['detail'][i:i+1],k,32);dnew=pt['detail'][i:i+1].copy();put(dnew,0,y,x,get(pd['detail'],int(md[idx[q,0]][0]),int(md[idx[q,0]][1]),int(md[idx[q,0]][2])));new=field(pt['coarse'][i:i+1],dnew,k,32);ds.append(action(new)-action(old))
  a=np.asarray(ds);rows.append({'context':name,'ring_width':width,'mean_DeltaS':a.mean(),'median_DeltaS':np.median(a),'frac_lt0':np.mean(a<0),'frac_lt1':np.mean(a<1),'frac_lt2':np.mean(a<2),'frac_lt5':np.mean(a<5),'frac_lt10':np.mean(a<10),'mean_exp_minus_DeltaS':np.mean(np.exp(np.clip(-a,-700,700))),'nearest_distance_q50':np.quantile(np.sqrt(d2[:,0]),.5),'donor_unique_fraction':len(np.unique(idx[:,0]))/N})
 wr('boundary_conditioned_transplant_metrics.csv',rows)
 a=list(csv.DictReader((OUT/'patch_attempts_L32.csv').open()));wr('failed_logA_decomposition.csv',[{'mean_DeltaS':np.mean([float(x['DeltaS']) for x in a]),'mean_logq_ratio':np.mean([float(x['logq_old'])-float(x['logq_new']) for x in a]),'mean_logA':np.mean([float(x['logA']) for x in a]),'dominant':'target_action_and_proposal_ratio'}])
 (OUT/'boundary_context_specification.md').write_text('Contexts contain coarse 7x7 and only transformed-detail sites outside the 2x2 target block. They are fixed during a block update, so old/new density evaluations use the same J_k.\n')
 (OUT/'boundary_conditioned_summary.md').write_text('Diagnostic only; no boundary-conditioned A/R sampler is enabled.\n')
if __name__=='__main__':main()
