#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
from train_lam1p0_flow_detail_pilot import load_kernel_matrix,load_phi,split_pairs
from train_lam1p0_autoregressive_detail_flow import torch_kernel_fft,torch_inverse_kernel
from train_lam1p0_local_multistage_r2_gate import LocalFlow
from train_lam1p0_local_multistage_rqspline import assemble_psi,destandardize,metrics_rows,write_csv,write_json
def action(x):
 p2=(x*x).mean((1,2));p4=(x**4).mean((1,2));nn=.5*((x*np.roll(x,-1,1)).mean((1,2))+(x*np.roll(x,-1,2)).mean((1,2)));return 1-2*p2+p4-4*.340301*nn
def feats(c,r):
 rows=[];meta=[]
 for i in range(len(c)):
  for y in range(0,16,2):
   for x in range(0,16,2):
    rows.append([c[i,(y+dy)%16,(x+dx)%16] for dy in range(-r,r+1) for dx in range(-r,r+1)]);meta.append((i,y,x))
 return np.asarray(rows,np.float64),np.asarray(meta)
def main():
 out=Path('perfect_blocking_upsampling/runs/lam1p0/diagnostics/per_patch_context_matched_residual_transplant_20260721');out.mkdir(parents=True,exist_ok=True);k,_=load_kernel_matrix(Path('perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json'));phi=load_phi(Path('data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz'))[:100];p=split_pairs(phi,k);ck=torch.load(Path('perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_local_residual_AR_R3_newkernel_N100_20260721/checkpoints/checkpoint_best.pt'),map_location='cpu',weights_only=False);m=LocalFlow('C');m.load_state_dict(ck['model_state']);st=ck['stats'];cn=(p['coarse']-float(np.asarray(st['c']['mean']).reshape(-1)[0]))/float(np.asarray(st['c']['std']).reshape(-1)[0]);c=torch.from_numpy(cn);d=[torch.from_numpy((p['detail'][:,i]-float(np.asarray(st[x]['mean']).reshape(-1)[0]))/float(np.asarray(st[x]['std']).reshape(-1)[0])) for i,x in enumerate(('e1','e2','body'))]
 with torch.no_grad(): ps=[m.params_full(c,None,None,'edge'),m.params_full(c,d[0],None,'corr'),m.params_full(c,d[0],d[1],'body')];res=[((x-mu)*torch.exp(-s)).numpy() for x,(_,mu,s) in zip(d,ps)]
 target_cfg=np.arange(90,100); donor_cfg=np.arange(90);allrows=[];distrows=[]
 for label,radius in [('amplitude',0),('patch3',1),('patch7',3)]:
  F,M=feats(cn,radius);don=np.isin(M[:,0],donor_cfg);tar=np.isin(M[:,0],target_cfg);FD,MD=F[don],M[don];FT,MT=F[tar],M[tar];mean,std=FD.mean(0),FD.std(0);std[std<1e-8]=1;FD=(FD-mean)/std;FT=(FT-mean)/std
  picked=[];dd=[]
  for q,(v,meta) in enumerate(zip(FT,MT)):
   ds=np.sum((FD-v)**2,axis=1);ds[MD[:,0]==meta[0]]=np.inf;j=int(np.argmin(ds));picked.append(j);dd.append(float(ds[j]))
  rr=[np.empty((len(target_cfg),16,16),np.float32) for _ in range(3)]
  for q,(j,tm) in enumerate(zip(picked,MT)):
   ti=int(tm[0]-90);y,x=tm[1:];dj,dy,dx=MD[j]
   for s in range(3):rr[s][ti,y:y+2,x:x+2]=res[s][dj,dy:dy+2,dx:dx+2]
   allrows.append({'context':label,'target_config':int(tm[0]),'target_y':int(y),'target_x':int(x),'donor_config':int(dj),'donor_y':int(dy),'donor_x':int(dx),'distance':dd[-1]})
  ds=[mu[target_cfg]+torch.exp(sig[target_cfg])*torch.from_numpy(z) for z,(_,mu,sig) in zip(rr,ps)];psi=assemble_psi(*[destandardize(v,st,x) for v,x in zip([c[target_cfg]]+ds,['c','e1','e2','body'])]);g=torch_inverse_kernel(psi,torch_kernel_fft(k,32,torch.device('cpu'))).detach().numpy();met={z['observable']:z for z in metrics_rows(phi[target_cfg],g,label,'whole')};aa=action(g);allrows[-1]['result_marker']=True;distrows.append({'context':label,'mean':np.mean(dd),'median':np.median(dd),'q90':np.quantile(dd,.9),'q95':np.quantile(dd,.95),'q99':np.quantile(dd,.99)});allrows.append({'context':label,'metric_action_shift':met['action_density']['shift_native_sigma'],'metric_action_std':met['action_density']['std_ratio'],'metric_q05':float((aa<=np.quantile(action(phi[target_cfg]),.05)).mean()),'metric_phi4_shift':met['phi4']['shift_native_sigma']})
 write_csv(out/'donor_index.csv',allrows);write_csv(out/'nearest_neighbor_distance_metrics.csv',distrows);write_json(out/'context_feature_metadata.json',{'standardization':'donor mean/std per component','LOO':'all same-config donor blocks masked','features':{'amplitude':1,'patch3':9,'patch7':49}});(out/'summary.md').write_text('Exact indexed LOO patch transplant completed. Metrics rows are marked in donor_index.csv.\n')
if __name__=='__main__':main()
