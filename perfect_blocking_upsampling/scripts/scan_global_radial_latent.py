import csv,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
from train_lam1p0_flow_detail_pilot import load_kernel_matrix,split_pairs
from train_lam1p0_autoregressive_detail_flow import torch_kernel_fft,torch_inverse_kernel
from train_lam1p0_local_multistage_rqspline import assemble_psi
import torch
RUN=ROOT/'perfect_blocking_upsampling/runs/lam1p0/paired_nativeL32_blockL16_reupscaleL32_N1000_20260721'
def main():
 z=np.load(RUN/'paired_fields.npz');a,b=z['native'],z['upscaled'];k,_=load_kernel_matrix(ROOT/'perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json');p=split_pairs(b,k);c=p['coarse'];rows=[];rng=np.random.default_rng(7)
 for sg in (0,.005,.01,.015,.02):
  d=p['detail'].copy();g=.97*np.exp(sg*rng.normal(size=len(d)));d[:,0]*=g[:,None,None];d[:,1]*=g[:,None,None];psi=assemble_psi(torch.from_numpy(c),*[torch.from_numpy(d[:,s]) for s in range(3)]);x=torch_inverse_kernel(psi,torch_kernel_fft(k,32,torch.device('cpu'))).detach().numpy()
  for n,f in [('phi2',lambda q:(q*q).mean((1,2))),('phi4',lambda q:(q**4).mean((1,2))),('action',lambda q:(1-q*q+.5*q**4-2*.340301*(q*np.roll(q,-1,1)+q*np.roll(q,-1,2))).mean((1,2))),('NN',lambda q:(q*np.roll(q,-1,1)+q*np.roll(q,-1,2)).mean((1,2)))]:
   u,v=f(a),f(x);rows.append({'sigma_gamma':sg,'observable':n,'shift_sigma':(v.mean()-u.mean())/u.std(ddof=1),'width_ratio':v.std(ddof=1)/u.std(ddof=1),'reblocking_error':np.max(np.abs(split_pairs(x,k)['coarse']-c))})
 with (RUN/'global_radial_latent_scan.csv').open('w',newline='') as f:w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
