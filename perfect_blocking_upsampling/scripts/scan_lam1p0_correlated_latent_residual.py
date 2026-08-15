#!/usr/bin/env python3
"""N100 exact correlated-latent coverage scan for the bounded residual baseline."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
from train_lam1p0_flow_detail_pilot import load_kernel_matrix,load_phi,split_pairs
from train_lam1p0_autoregressive_detail_flow import torch_kernel_fft
from train_lam1p0_local_multistage_r2_gate import LocalFlow
from train_lam1p0_local_multistage_rqspline import assemble_psi,destandardize,metrics_rows,write_csv
from train_lam1p0_autoregressive_detail_flow import torch_inverse_kernel

def action(x):
 p2=(x*x).mean((1,2));p4=(x**4).mean((1,2));nn=.5*((x*np.roll(x,-1,1)).mean((1,2))+(x*np.roll(x,-1,2)).mean((1,2)));return 1-2*p2+p4-4*.340301*nn
def colored(shape,a,g):
 e=torch.randn(shape,generator=g); mask=((torch.arange(shape[1])[:,None]+torch.arange(shape[2])[None,:])%2==0).to(e)
 # z_A=e_A; z_B=sqrt(1-a^2)e_B + a/2 sum four A neighbours. Exact triangular inverse, logdet=N_B log sqrt(1-a^2).
 av=.25*(torch.roll(e,1,1)+torch.roll(e,-1,1)+torch.roll(e,1,2)+torch.roll(e,-1,2)); return mask*e+(1-mask)*(np.sqrt(1-a*a)*e+a*2*av)
def sample(model,c,stats,kernel,L,a,seed):
 g=torch.Generator().manual_seed(seed); z1=colored(c.shape,a,g);p,m,s=model.params_full(c,None,None,'edge');r,_=model._compose(z1,p,False);e1=m+torch.exp(s)*r
 z2=colored(c.shape,a,g);p,m,s=model.params_full(c,e1,None,'corr');r,_=model._compose(z2,p,False);e2=m+torch.exp(s)*r
 z3=colored(c.shape,a,g);p,m,s=model.params_full(c,e1,e2,'body');r,_=model._compose(z3,p,False);b=m+torch.exp(s)*r
 psi=assemble_psi(destandardize(c,stats,'c'),destandardize(e1,stats,'e1'),destandardize(e2,stats,'e2'),destandardize(b,stats,'body'));return torch_inverse_kernel(psi,torch_kernel_fft(kernel,L,torch.device('cpu'))).detach().numpy()
def main():
 out=Path('perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_correlated_latent_residual_flow_20260721');out.mkdir(parents=True,exist_ok=True)
 ck=torch.load(Path('perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_local_residual_AR_R3_newkernel_N100_20260721/checkpoints/checkpoint_best.pt'),map_location='cpu',weights_only=False);m=LocalFlow('C');m.load_state_dict(ck['model_state']);m.eval();stats=ck['stats'];k,_=load_kernel_matrix(Path('perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json'))
 p32=load_phi(Path('data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz'))[90:100];p64=load_phi(Path('data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz'))[:30]
 rows=[]
 for name,native in [('L16to32',p32),('L32to64',p64)]:
  c=split_pairs(native,k)['coarse'];mean=float(np.asarray(stats['c']['mean']).reshape(-1)[0]);std=float(np.asarray(stats['c']['std']).reshape(-1)[0]);c=torch.from_numpy((c-mean)/std)
  for a in (0.,.15,.30,.45,.60):
   gen=sample(m,c,stats,k,native.shape[1],a,2026072122);an,ag=action(native),action(gen);met={r['observable']:r for r in metrics_rows(native,gen,name,'whole')};rows.append({'volume':name,'option':'B_checkerboard_local_AR','alpha':a,'action_shift':met['action_density']['shift_native_sigma'],'action_std_ratio':met['action_density']['std_ratio'],'action_KS':met['action_density']['KS'],**{f'action_q{int(q*100):02d}_occupancy':float((ag<=np.quantile(an,q)).mean()) for q in(.01,.05,.10)},'phi2_shift':met['phi2']['shift_native_sigma'],'phi4_shift':met['phi4']['shift_native_sigma'],'kurtosis_shift':met['local_kurtosis_ratio']['shift_native_sigma'],'NN_shift':met['NN']['shift_native_sigma'],'nonfinite':int(np.size(gen)-np.isfinite(gen).sum())})
 write_csv(out/'coverage_scan_metrics.csv',rows)
 (out/'latent_option_specifications.md').write_text('# Exact bases\n\nA: rejected for scan: generic 3x3 symmetric coloring has a volume-dependent determinant.\n\nB: checkerboard triangular map: z_A=e_A; z_B=sqrt(1-a^2)e_B+(a/2)sum_{four A neighbours}e_A. Inverse is e_A=z_A, e_B=(z_B-(a/2)sum z_A)/sqrt(1-a^2); logdet=N_B log sqrt(1-a^2); radius one.\n\nC: nonoverlapping patch shared Gaussian has an exact block Gaussian covariance/density, but fixed tiling breaks one-site translation and is deferred until B is assessed.\n')
if __name__=='__main__':main()
