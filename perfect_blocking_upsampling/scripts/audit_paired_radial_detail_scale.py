import csv,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
from train_lam1p0_flow_detail_pilot import load_kernel_matrix,split_pairs
from train_lam1p0_autoregressive_detail_flow import torch_kernel_fft,torch_inverse_kernel
from train_lam1p0_local_multistage_rqspline import assemble_psi,metrics_rows
import torch
RUN=ROOT/'perfect_blocking_upsampling/runs/lam1p0/paired_nativeL32_blockL16_reupscaleL32_N1000_20260721'
def w(n,r):
 ks=[]
 for x in r:
  for k in x:
   if k not in ks:ks.append(k)
 with (RUN/n).open('w',newline='') as f:q=csv.DictWriter(f,ks);q.writeheader();q.writerows(r)
def main():
 z=np.load(RUN/'paired_fields.npz');a,b=z['native'],z['upscaled'];k,_=load_kernel_matrix(ROOT/'perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json');pa,pb=split_pairs(a,k),split_pairs(b,k);p2a=(a*a).mean((1,2));p2b=(b*b).mean((1,2));scale=np.sqrt(p2a/p2b);c=pb['coarse'];co={'coarse_phi2':(c*c).mean((1,2)),'coarse_phi4':(c**4).mean((1,2)),'coarse_NN':(c*np.roll(c,-1,1)+c*np.roll(c,-1,2)).mean((1,2)),'coarse_m':c.mean((1,2))}
 rows=[{'a':x,'source_index':1000+i,**{n:v[i] for n,v in co.items()}} for i,x in enumerate(scale)];w('radial_scale_per_configuration.csv',rows);w('radial_scale_correlations.csv',[{'variable':n,'pearson':np.corrcoef(scale,v)[0,1]} for n,v in co.items()]);
 out=[]
 for label,sectors in [('all',(0,1,2)),('D01',(0,)),('D10',(1,)),('D11',(2,)),('edges',(0,1))]:
  for g in (.95,.97,.99,1.,1.01,1.03,1.05):
   d=pb['detail'].copy();d[:,sectors]*=g;psi=assemble_psi(torch.from_numpy(c),*[torch.from_numpy(d[:,s]) for s in range(3)]);x=torch_inverse_kernel(psi,torch_kernel_fft(k,32,torch.device('cpu'))).detach().numpy();m={r['observable']:r for r in metrics_rows(a,x,'x','whole')};out.append({'sector':label,'gamma':g,'reblocking_error':float(np.max(np.abs(split_pairs(x,k)['coarse']-c))),'phi2_shift':m['phi2']['shift_native_sigma'],'phi2_width':m['phi2']['std_ratio'],'phi4_shift':m['phi4']['shift_native_sigma'],'phi4_width':m['phi4']['std_ratio'],'action_shift':m['action_density']['shift_native_sigma'],'NN_shift':m['NN']['shift_native_sigma']})
 w('detail_sector_scale_scan.csv',out)
 (RUN/'radial_calibration_specification.md').write_text('Diagnostic transformations scale psi detail sectors only with coarse ee held fixed; inverse is D/gamma and the detail-coordinate log-Jacobian is n_scaled log|gamma|.\n')
if __name__=='__main__':main()
