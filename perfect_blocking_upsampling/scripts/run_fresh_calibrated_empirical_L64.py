import sys,json,traceback,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
from run_fresh_calibrated_empirical_validation import gen,rows,w
import run_lam1p0_empirical_joint_2x2_mixture as e
from train_lam1p0_flow_detail_pilot import load_phi,load_kernel_matrix,split_pairs
OUT=ROOT/'perfect_blocking_upsampling/runs/lam1p0/fresh_disjoint_calibrated_empirical_validation_20260721'
def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
 (OUT/'RUNNING').write_text('running\n');(OUT/'00_startup_complete.json').write_text(json.dumps({'cwd':os.getcwd(),'script':str(Path(__file__).resolve()),'output':str(OUT.resolve()),'targets':'L64[200:400]'}));print(f'OUTPUT {OUT.resolve()}',flush=True)
 try:
  k,_=load_kernel_matrix(e.KPATH);l32=load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz');pd=split_pairs(l32[:1000],k);M=e.meta(1000,16);h=e.features(pd['coarse'],M);hm,hs=h.mean(0),h.std(0)+1e-6;H=(h-hm)/hs;D=e.vectors(pd['detail'],M);l64=load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz')[200:400];(OUT/'01_sources_loaded.json').write_text('{}');raw,mean,cal,z=gen(l64,k,H,D,hm,hs,20260764);np.savez_compressed(OUT/'paired_fields_L64.npz',native=l64,raw=raw,mean_calibrated=mean,calibrated=cal,z=z,gamma=.97*np.exp(.02*z));(OUT/'03_L64_generation_complete.json').write_text('{}');
  for fn,x in [('raw_metrics_L64.csv',raw),('mean_calibrated_metrics_L64.csv',mean),('full_calibrated_metrics_L64.csv',cal)]:w(fn,[rows(l64,x,'L64')]);w('quantitative_comparison_L64.csv',[rows(l64,x,n) for n,x in [('raw',raw),('mean',mean),('full',cal)]]);w('latent_draws_L64.csv',[{'source_index':200+i,'z':z[i],'gamma':.97*np.exp(.02*z[i]),'raw_phi2':(raw[i]**2).mean(),'cal_phi2':(cal[i]**2).mean(),'raw_phi4':(raw[i]**4).mean(),'cal_phi4':(cal[i]**4).mean()} for i in range(len(z))]);
  err=np.max(np.abs(split_pairs(cal,k)['coarse']-split_pairs(l64,k)['coarse']));(OUT/'validation_summary_L64.md').write_text(f'Fresh L64 indices 200:399; N=200; max calibrated reblocking error={err:.3e}.\n');(OUT/'COMPLETED.json').write_text('{}');(OUT/'RUNNING').unlink(missing_ok=True)
 except Exception as exc:
  (OUT/'FAILED.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}));raise
if __name__=='__main__':main()
