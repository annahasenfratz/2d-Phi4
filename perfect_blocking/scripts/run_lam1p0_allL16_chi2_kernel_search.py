#!/usr/bin/env python3
"""Strict all-L16-reference chi2 kernel search (train/validation/test in L32)."""
from pathlib import Path
import json, sys, os, numpy as np
R=Path(__file__).resolve().parents[1]; P=R.parent; sys.path.insert(0,str(R))
from scipy.optimize import minimize
from scripts.common.blocking import load_configs
from scripts.run_lam1p0_7x7_kernel_search import block, observable_arrays, ETA_SCALE, momentum_extrema
rad=int(os.environ.get('KERNEL_RADIUS','2')); corr_weight=float(os.environ.get('CORRELATION_WEIGHT','0')); positivity_only=os.environ.get('POSITIVITY_ONLY','0') == '1'; maxiter=int(os.environ.get('MAXITER','160')); extra_local2=os.environ.get('EXTRA_LOCAL2','0') == '1'; frozen_block_cov=os.environ.get('FROZEN_BLOCK_COV','0') == '1'
# Spectral safeguards.  The soft terms guide the search without creating the
# steep, discontinuous valleys that stalled bounded Powell.  The final caps
# are applied only when choosing a usable candidate on the dense grid.
min_k_floor=float(os.environ.get('MIN_K_FLOOR','0.35'))
soft_condition_target=float(os.environ.get('SOFT_CONDITION_TARGET','2.3'))
soft_condition_width=float(os.environ.get('SOFT_CONDITION_WIDTH','0.5'))
soft_condition_weight=float(os.environ.get('SOFT_CONDITION_WEIGHT','5.0'))
soft_inverse_target=float(os.environ.get('SOFT_INVERSE_TARGET','2.0'))
soft_inverse_width=float(os.environ.get('SOFT_INVERSE_WIDTH','0.5'))
soft_inverse_weight=float(os.environ.get('SOFT_INVERSE_WEIGHT','2.0'))
inverse_cap=float(os.environ.get('INVERSE_CAP','2.5'))
condition_cap=float(os.environ.get('CONDITION_CAP','3.0'))
tag=f'allL16_chi2_R{rad}_corrW{corr_weight:g}_{"extraLocal2_" if extra_local2 else ""}{"frozenBlockCov_" if frozen_block_cov else ""}{"positiveOnly" if positivity_only else f"softCond{soft_condition_target}_inv{soft_inverse_target}_finalCond{condition_cap}_inv{inverse_cap}"}_train3000_val1000_test1000'
tag=tag.replace('.', 'p'); O=R/'perfect_blocking_lam1p0/tests/intermediate'/tag; O.mkdir(parents=True,exist_ok=True)
D=load_configs(P/'data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz'); F=load_configs(P/'data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz'); g=np.random.default_rng(20260808); fi=g.permutation(len(F))[:5000]; T,V,E=F[fi[:3000]],F[fi[3000:4000]],F[fi[4000:]]
keys=['action_density','phi2','phi4','local_kurtosis_ratio','NN','diag','2nn','m2','m4','G_pmin_avg'] + (['bond_sq','phi3_neighbor'] if extra_local2 else [])
def operators(x):
 o=observable_arrays(x)
 if extra_local2:
  px,py=np.roll(x,-1,axis=1),np.roll(x,-1,axis=2)
  o['bond_sq']=.5*np.mean((x*px)**2+(x*py)**2,axis=(1,2))
  o['phi3_neighbor']=.5*np.mean(x**3*(px+py),axis=(1,2))
 return o
def feat(x):
 o=operators(x); A=np.column_stack([o[k] for k in keys]); return np.column_stack([A,A[:,1]*A[:,4],A[:,1]*A[:,2],A[:,4]*A[:,5],A[:,0]*A[:,3]])
def shrink_cov(x):
 x=np.asarray(x,float); c=np.cov(x,rowvar=False,ddof=1); alpha=.10
 return (1-alpha)*c + alpha*np.trace(c)/c.shape[0]*np.eye(c.shape[0])
ref=feat(D); mu=ref.mean(0); C=shrink_cov(ref)/len(ref)
pair_idx=[(1,4),(1,2),(1,3),(4,5),(0,3)]
ref_corr=np.corrcoef(np.column_stack([operators(D)[k] for k in keys]),rowvar=False)
orbit=[(a,b) for a in range(1,rad+1) for b in range(a+1)]
def mat(x):
 q=np.zeros((2*rad+1,2*rad+1));
 for iy,y in enumerate(range(-rad,rad+1)):
  for ix,z in enumerate(range(-rad,rad+1)):
   if y or z: q[iy,ix]=x[orbit.index(tuple(sorted((abs(y),abs(z)),reverse=True)))]
 q[rad,rad]=1-q.sum(); return ETA_SCALE*q
if frozen_block_cov:
 reference_kernel=Path(os.environ.get('REFERENCE_KERNEL',str(R/'perfect_blocking_lam1p0/kernels/candidates/colleague_paper_objective_5x5_eta_included.json')))
 if not reference_kernel.exists(): raise RuntimeError(f'missing REFERENCE_KERNEL: {reference_kernel}')
 reference_matrix=np.asarray(json.loads(reference_kernel.read_text())['matrix'],float)
 reference_features=feat(block(T,reference_matrix))
 W_fixed=np.linalg.pinv(C+shrink_cov(reference_features)/len(reference_features))
def chi(x,s):
 M=mat(x); mom=momentum_extrema(M,grid=192)
 # The positive floor is the only hard in-fit safeguard.  The remaining
 # spectral preferences are smooth fourth-power hinges.
 penalty=1e12 if mom['min_K'] <= min_k_floor else 0.0
 if not positivity_only:
  condition_number=mom['max_K']/mom['min_K']
  penalty+=soft_condition_weight*max(0.,(condition_number-soft_condition_target)/soft_condition_width)**4
  penalty+=soft_inverse_weight*max(0.,(mom['max_inverse_K']-soft_inverse_target)/soft_inverse_width)**4
 blocked=block(s,M); z=feat(blocked); d=z.mean(0)-mu; W=W_fixed if frozen_block_cov else np.linalg.pinv(C+shrink_cov(z)/len(z)); r=np.corrcoef(np.column_stack([operators(blocked)[k] for k in keys]),rowvar=False); corr=sum((r[i,j]-ref_corr[i,j])**2 for i,j in pair_idx); return float(d@W@d+penalty+corr_weight*corr)
core=np.array([-.050604837,.002289799,.041616846,.026921096,-.008012467]); starts=[np.zeros(len(orbit)),np.pad(core,(0,len(orbit)-5))]
# For 7x7, explore distinct stable outer-shell directions rather than relying
# only on the padded 5x5 basin.  The momentum guard rejects unsafe proposals.
if rad >= 3:
 rng=np.random.default_rng(20260810)
 for _ in range(5 if rad == 3 else 9):
  trial=np.pad(core,(0,len(orbit)-5)).astype(float)
  trial[5:]+=rng.normal(0.,0.003,size=len(orbit)-5)
  starts.append(trial)
fits=[]
for start_index,a in enumerate(starts):
 r=minimize(lambda x:chi(x,T),a,method='Powell',bounds=[(-.1,.1)]*len(orbit),options={'maxiter':maxiter})
 x_try=r.x; mom_try=momentum_extrema(mat(x_try),grid=1024); condition_try=mom_try['max_K']/mom_try['min_K']
 record={'start_index':start_index,'success':bool(r.success),'message':str(r.message),'nfev':int(r.nfev),'coefficients':x_try.tolist(),'validation_score':chi(x_try,V),'momentum_stability':mom_try,'condition_number':condition_try,'admissible':bool(mom_try['min_K']>0.4 and mom_try['max_inverse_K']<=inverse_cap and condition_try<=condition_cap)}
 fits.append(record)
 (O/'multistart_progress.json').write_text(json.dumps({'settings':{'min_k_floor':min_k_floor,'soft_condition_target':soft_condition_target,'soft_condition_width':soft_condition_width,'soft_condition_weight':soft_condition_weight,'soft_inverse_target':soft_inverse_target,'soft_inverse_width':soft_inverse_width,'soft_inverse_weight':soft_inverse_weight,'final_inverse_cap':inverse_cap,'final_condition_cap':condition_cap},'completed_starts':fits},indent=2))
 print(f"completed start {start_index+1}/{len(starts)}: validation_score={record['validation_score']:.6g}, admissible={record['admissible']}",flush=True)
admissible=[f for f in fits if f['admissible']]
selected=min(admissible or fits,key=lambda f:f['validation_score'])
x=np.asarray(selected['coefficients']); mom=selected['momentum_stability']; out={'radius':rad,'correlation_weight':corr_weight,'extra_local2':extra_local2,'operators':keys,'frozen_block_covariance':frozen_block_cov,'reference_kernel':str(reference_kernel) if frozen_block_cov else None,'positivity_only':positivity_only,'spectral_regularizer':{'min_k_floor':min_k_floor,'soft_condition_target':soft_condition_target,'soft_condition_width':soft_condition_width,'soft_condition_weight':soft_condition_weight,'soft_inverse_target':soft_inverse_target,'soft_inverse_width':soft_inverse_width,'soft_inverse_weight':soft_inverse_weight},'final_inverse_cap':inverse_cap,'final_condition_cap':condition_cap,'selection_status':'admissible' if admissible else 'no_admissible_candidate','selected_start_index':selected['start_index'],'maxiter':maxiter,'orbits':orbit,'coefficients':x.tolist(),'matrix':mat(x).tolist(),'momentum_stability':mom,'condition_number':mom['max_K']/mom['min_K'],'train_chi2':chi(x,T),'validation_chi2':chi(x,V),'test_chi2':chi(x,E),'n_direct_L16':len(D),'n_L32':[3000,1000,1000],'multistarts':fits}; (O/'result.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
