import csv,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
from train_lam1p0_flow_detail_pilot import load_kernel_matrix,split_pairs
from train_lam1p0_autoregressive_detail_flow import torch_kernel_fft,torch_inverse_kernel
from train_lam1p0_local_multistage_rqspline import assemble_psi
from train_lam1p0_local_multistage_rqspline import metrics_rows
RUN=ROOT/'perfect_blocking_upsampling/runs/lam1p0/fresh_disjoint_calibrated_empirical_validation_20260721'
def write(p,r):
 ks=[]
 for x in r:
  for k in x:
   if k not in ks:ks.append(k)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,ks);w.writeheader();w.writerows(r)
def main():
 k,_=load_kernel_matrix(ROOT/'perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json');allrows=[];bd=[]
 for L in (32,64):
  z=np.load(RUN/f'paired_fields_L{L}.npz');native,raw=z['native'],z['raw'];p=split_pairs(raw,k);c=p['coarse'];Lc=c.shape[1];rng=np.random.default_rng(990+L)
  for ps in (2,4,8,Lc):
   for sg in (0,.005,.01,.015,.02):
    d=p['detail'].copy();g=np.ones((len(d),Lc,Lc))*.97
    for y in range(0,Lc,ps):
     for x in range(0,Lc,ps):g[:,y:y+ps,x:x+ps]*=np.exp(sg*rng.normal(size=(len(d),1,1)))
    d[:,:2]*=g[:,None];psi=assemble_psi(torch.from_numpy(c),*[torch.from_numpy(d[:,s]) for s in range(3)]);q=torch_inverse_kernel(psi,torch_kernel_fft(k,L,torch.device('cpu'))).detach().numpy();m={r['observable']:r for r in metrics_rows(native,q,'x','whole')};allrows.append({'L':L,'patch_size':ps,'sigma_gamma':sg,**{f'{o}_{v}':m[o][v] for o in ('phi2','phi4','action_density','NN','diag','2nn','local_kurtosis_ratio') for v in ('shift_native_sigma','std_ratio','KS')},'reblocking_error':float(np.max(np.abs(split_pairs(q,k)['coarse']-c)))})
    # NN patch-boundary product diagnostic on coarse patch labels.
    u=np.arange(Lc*2)//(2*ps);lab=u[:,None]*(Lc//ps)+u[None,:];same=lab==np.roll(lab,-1,1);nn=.5*(q*np.roll(q,-1,1)+q*np.roll(q,-1,2));bd.append({'L':L,'patch_size':ps,'sigma_gamma':sg,'NN_inside':float(nn[:,same].mean()),'NN_across':float(nn[:,~same].mean()),'boundary_difference':float(nn[:,same].mean()-nn[:,~same].mean())})
  write(RUN/f'patch_radial_latent_scan_L{L}.csv',[r for r in allrows if r['L']==L])
 write(RUN/'patch_boundary_diagnostics.csv',bd)
 (RUN/'patch_radial_latent_summary.md').write_text('Patch latents scale D01,D10 only: 0.97 exp(sigma z_patch); D11 and coarse ee fixed.\n')
if __name__=='__main__':main()
