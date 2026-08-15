#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
from train_lam1p0_flow_detail_pilot import load_kernel_matrix,load_phi,split_pairs
from train_lam1p0_autoregressive_detail_flow import torch_kernel_fft,torch_inverse_kernel
from train_lam1p0_local_multistage_r2_gate import LocalFlow
from train_lam1p0_local_multistage_rqspline import assemble_psi,destandardize,metrics_rows,write_csv
def act(x):
 p2=(x*x).mean((1,2));p4=(x**4).mean((1,2));nn=.5*((x*np.roll(x,-1,1)).mean((1,2))+(x*np.roll(x,-1,2)).mean((1,2)));return 1-2*p2+p4-4*.340301*nn
def stats(x,label,stage):
 q=np.quantile(x,[.01,.05,.1,.5,.9,.95,.99]);return {'kind':label,'stage':stage,'mean':x.mean(),'std':x.std(),'skew':np.mean((x-x.mean())**3)/x.std()**3,'excess_kurtosis':np.mean((x-x.mean())**4)/x.std()**4-3,'E_r4':np.mean(x**4),'q01':q[0],'q05':q[1],'q10':q[2],'q50':q[3],'q90':q[4],'q95':q[5],'q99':q[6]}
def main():
 out=Path('perfect_blocking_upsampling/runs/lam1p0/diagnostics/residual_shape_independence_ablation_20260721');out.mkdir(parents=True,exist_ok=True);k,_=load_kernel_matrix(Path('perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json'));phi=load_phi(Path('data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz'))[90:100];p=split_pairs(phi,k);ck=torch.load(Path('perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_local_residual_AR_R3_newkernel_N100_20260721/checkpoints/checkpoint_best.pt'),map_location='cpu',weights_only=False);m=LocalFlow('C');m.load_state_dict(ck['model_state']);st=ck['stats'];c=torch.from_numpy((p['coarse']-float(np.asarray(st['c']['mean']).reshape(-1)[0]))/float(np.asarray(st['c']['std']).reshape(-1)[0]));d=[torch.from_numpy((p['detail'][:,i]-float(np.asarray(st[x]['mean']).reshape(-1)[0]))/float(np.asarray(st[x]['std']).reshape(-1)[0])) for i,x in enumerate(('e1','e2','body'))]
 with torch.no_grad():
  ps=[m.params_full(c,None,None,'edge'),m.params_full(c,d[0],None,'corr'),m.params_full(c,d[0],d[1],'body')];rn=[((x-mu)*torch.exp(-s)).numpy() for x,(_,mu,s) in zip(d,ps)];g=m.sample_full(c,torch.Generator().manual_seed(3))[:3];rg=[]
  for z,(pa,mu,s) in zip([torch.randn_like(c) for _ in range(3)],ps): rg.append(m._compose(z,pa,False)[0].numpy())
 rows=[stats(x,'native',i) for i,x in enumerate(rn)]+[stats(x,'flow',i) for i,x in enumerate(rg)];write_csv(out/'native_residual_metrics.csv',[r for r in rows if r['kind']=='native']);write_csv(out/'generated_residual_metrics.csv',[r for r in rows if r['kind']=='flow']);write_csv(out/'residual_quantiles.csv',rows)
 cases=[];rng=np.random.default_rng(1)
 for name,res in [('native',rn),('gaussian',[rng.normal(size=x.shape) for x in rn]),('site_bootstrap',[rng.choice(x.reshape(-1),x.size).reshape(x.shape) for x in rn]),('joint_config_bootstrap',[np.stack([x[rng.integers(len(x))] for _ in range(len(x))]) for x in rn]),('flow',rg)]:
  ds=[mu+torch.exp(s)*torch.from_numpy(r.astype(np.float32)) for r,(_,mu,s) in zip(res,ps)];psi=assemble_psi(*[destandardize(v,st,x) for v,x in zip([c]+ds,['c','e1','e2','body'])]);x=torch_inverse_kernel(psi,torch_kernel_fft(k,32,torch.device('cpu'))).detach().numpy();met={r['observable']:r for r in metrics_rows(phi,x,name,'whole')};a=act(x);cases.append({'case':name,'action_shift':met['action_density']['shift_native_sigma'],'action_std':met['action_density']['std_ratio'],'q05_occupancy':float((a<=np.quantile(act(phi),.05)).mean()),'phi4_shift':met['phi4']['shift_native_sigma'],'kurtosis_shift':met['local_kurtosis_ratio']['shift_native_sigma']})
 write_csv(out/'reconstruction_ablation_metrics.csv',cases);(out/'summary.md').write_text('Residual reconstruction audit; native residual reconstruction must be exact.\n')
if __name__=='__main__':main()
