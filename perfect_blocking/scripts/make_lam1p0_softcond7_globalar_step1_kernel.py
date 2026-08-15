import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2]
src=R/'perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R3_soft_conditioned_7x7_eta_included.json'
out=R/'perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/softcond7_globalar_step1_outerminus.json'
d=json.loads(src.read_text());a=np.array(d['matrix'],float);r=3;eps=.0005
for ox,oy in [(3,1),(3,2),(2,1)]:
 m=0
 for i,x in enumerate(range(-r,r+1)):
  for j,y in enumerate(range(-r,r+1)):
   if tuple(sorted((abs(x),abs(y)),reverse=True))==(ox,oy):a[i,j]-=eps;m+=1
 a[r,r]+=eps*m
k=np.abs(np.fft.fft2(a,s=(32,32)))
if k.min()<.35 or k.max()/k.min()>3: raise RuntimeError('guard failed')
d['name']='softcond7_globalar_step1_outerminus';d['matrix']=a.tolist();d['global_ar_step']={'orbits_minus_eps':['31','32','21'],'epsilon':eps,'min_K':float(k.min()),'condition':float(k.max()/k.min())};out.write_text(json.dumps(d,indent=2)+'\n');print(out)
