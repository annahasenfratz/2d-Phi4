#!/usr/bin/env python3
from __future__ import annotations
import hashlib,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
from train_lam1p0_flow_detail_pilot import load_kernel_matrix,load_phi,split_pairs
from train_lam1p0_autoregressive_detail_flow import torch_kernel_fft,torch_inverse_kernel
from train_lam1p0_local_multistage_r2_gate import LocalFlow
from train_lam1p0_local_multistage_rqspline import assemble_psi,destandardize,metrics_rows,write_csv
def terms(x):
 p2=(x*x).mean((1,2));p4=(x**4).mean((1,2));nn=.5*((x*np.roll(x,-1,1)).mean((1,2))+(x*np.roll(x,-1,2)).mean((1,2)));return p2,p4,nn,1-2*p2+p4-4*.340301*nn
def main():
 out=Path('perfect_blocking_upsampling/runs/lam1p0/diagnostics/sector_replacement_predictor_audit_20260721');out.mkdir(parents=True,exist_ok=True);kp=Path('perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json');k,_=load_kernel_matrix(kp);phi=load_phi(Path('data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz'))[90:100];pairs=split_pairs(phi,k); cases=[];predrows=[];sigrows=[]
 for name,run,cand in [('R3','lam1p0_L16to32_local_residual_AR_R3_newkernel_N100_20260721','C'),('R4','lam1p0_L16to32_local_residual_AR_R4_newkernel_N100_20260721','D')]:
  ckpath=Path('perfect_blocking_upsampling/runs/lam1p0/training')/run/'checkpoints/checkpoint_best.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False);m=LocalFlow(cand);m.load_state_dict(ck['model_state']);m.eval();st=ck['stats'];c=torch.from_numpy((pairs['coarse']-float(np.asarray(st['c']['mean']).reshape(-1)[0]))/float(np.asarray(st['c']['std']).reshape(-1)[0]));native=[torch.from_numpy((pairs['detail'][:,i]-float(np.asarray(st[x]['mean']).reshape(-1)[0]))/float(np.asarray(st[x]['std']).reshape(-1)[0])) for i,x in enumerate(('e1','e2','body'))]
  with torch.no_grad():
   p1,mu1,s1=m.params_full(c,None,None,'edge');p2,mu2,s2=m.params_full(c,native[0],None,'corr');p3,mu3,s3=m.params_full(c,native[0],native[1],'body');mus=[mu1,mu2,mu3];ss=[s1,s2,s3]
   gen=m.sample_full(c,torch.Generator().manual_seed(7))[:3]
  for i,(x,mu,s) in enumerate(zip(native,mus,ss)):
   r=(x-mu)*torch.exp(-s);predrows.append({'model':name,'stage':i,'bias':float((x-mu).mean()),'rms':float(torch.sqrt(((x-mu)**2).mean())),'residual_mean':float(r.mean()),'residual_std':float(r.std()),'sigma_mean':float(torch.exp(s).mean()),'sigma_q05':float(torch.quantile(torch.exp(s),.05)),'sigma_q95':float(torch.quantile(torch.exp(s),.95))})
  for label,ds in [('native',native),('d01',[gen[0],native[1],native[2]]),('d10',[native[0],gen[1],native[2]]),('d11',[native[0],native[1],gen[2]]),('all',list(gen))]:
   psi=assemble_psi(*[destandardize(v,st,x) for v,x in zip([c]+ds,['c','e1','e2','body'])]);g=torch_inverse_kernel(psi,torch_kernel_fft(k,32,torch.device('cpu'))).detach().numpy();met={r['observable']:r for r in metrics_rows(phi,g,label,'whole')};a=terms(phi);b=terms(g);cases.append({'model':name,'case':label,'action_shift':met['action_density']['shift_native_sigma'],'action_std':met['action_density']['std_ratio'],'action_KS':met['action_density']['KS'],'phi2_shift':met['phi2']['shift_native_sigma'],'phi4_shift':met['phi4']['shift_native_sigma'],'kurtosis_shift':met['local_kurtosis_ratio']['shift_native_sigma'],'NN_shift':met['NN']['shift_native_sigma'],'quad_delta':float((b[0]-a[0]).mean()*-2),'quartic_delta':float((b[1]-a[1]).mean()),'NN_action_delta':float((b[2]-a[2]).mean()*-4*.340301)})
 write_csv(out/'one_sector_replacement_metrics.csv',[r for r in cases if r['case'] in ('d01','d10','d11')]);write_csv(out/'cumulative_replacement_metrics.csv',[r for r in cases if r['case'] in ('all','native')]);write_csv(out/'predictor_calibration.csv',predrows);write_csv(out/'sigma_calibration.csv',predrows);write_csv(out/'action_term_decomposition.csv',cases);(out/'summary.md').write_text('Compact R3/R4 sector audit. Native reconstruction is exact by construction; all replacement samples use native contexts for teacher forcing.\n')
if __name__=='__main__':main()
