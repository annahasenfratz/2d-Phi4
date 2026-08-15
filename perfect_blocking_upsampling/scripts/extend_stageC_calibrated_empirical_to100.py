#!/usr/bin/env python3
"""Continue Stage C from saved sweep-20 fields through sweep 100."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]; PKG=ROOT/'perfect_blocking_upsampling'
sys.path.insert(0,str(PKG/'src'));sys.path.insert(0,str(PKG/'scripts'));os.environ.setdefault('MPLCONFIGDIR','/tmp/inverse_rg_mpl')
from perfect_blocking_upsampling.actions import ActionSpec
from perfect_blocking_upsampling.kernels import apply_kernel, load_kernel
from run_calibrated_empirical_blocked_native_detail_only import OBS, comparison, ensemble, obs, write_csv
from run_lam0p2_residual_flow_patch_chain import StreamingCsv, patch_correct
from run_lam1p0_l16to32_rqspline_zeroshot import build_model_from_checkpoint, stationary_stats
from run_lam1p0_rqspline_patchwise import TWO_STAGE_COARSE_FIELDS, rqspline_two_stage_coarse_action_patch_update
from train_lam1p0_flow_detail_pilot import load_phi

OUT=PKG/'runs/lam1p0/calibrated_empirical_patchwise_rethermalization_20260721'; KPATH=ROOT/'perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json'; CFG=PKG/'run_configs/lam1p0_L16to32_rqspline_balanced_coarse_detail_production.yaml'; L32=ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz'; SAVES=(30,50,75,100)

def dargs(s): return argparse.Namespace(disable_coarse_updates=True,detail_passes=10,fine_proposal_sigma=.04,fine_patch_size=16,passes=0,proposal_sigma=.0,coarse_patch_size=16,global_sweep=s,verbose_patch_log=False)
def append(path, old, new): write_csv(path,old+new)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--chains',type=int,default=128);args=ap.parse_args()
 state=np.load(OUT/'stageC_pilot_states.npz');direct=state['direct_sweep20'][:args.chains].copy();control=state['control_sweep20'][:args.chains].copy()
 kernel,_=load_kernel(KPATH);direct_psi=apply_kernel(direct,kernel).astype(np.float32);control_psi=apply_kernel(control,kernel).astype(np.float32)
 native=load_phi(L32)[2000:2000+args.chains];action=ActionSpec('phi4_nn',1.,.340301);coarse_action=ActionSpec('phi4_nn',1.,.340301);reference=obs(native,action);reference_coarse=obs(apply_kernel(native,kernel)[:,0::2,0::2],coarse_action)
 config=yaml.safe_load(CFG.read_text());device=torch.device('cpu');checkpoint=torch.load(ROOT/config['flow_checkpoint'],map_location=device,weights_only=False);model,_=build_model_from_checkpoint(checkpoint,lattice_size=16,device=device);stats=stationary_stats(checkpoint['state']['stats'],lc=16)
 direct_initial=state['direct_start'][:args.chains];control_initial=state['control_start'][:args.chains]
 fine_existing=list(csv.DictReader((OUT/'direct_coarse_coarse_detail_metrics.csv').open()));control_existing=list(csv.DictReader((OUT/'blocked_native_coarse_detail_stationarity.csv').open()));coarse_existing=list(csv.DictReader((OUT/'coarse_field_evolution.csv').open()));coarse_acc=list(csv.DictReader((OUT/'coarse_acceptance.csv').open()));detail_acc=list(csv.DictReader((OUT/'detail_acceptance_stageC.csv').open()))
 fine_new=[];control_new=[];coarse_new=[];ca_new=[];da_new=[];rd=np.random.default_rng(2026072203);rc=np.random.default_rng(2026072204)
 for sweep in range(21,101):
  for label,psi,phi,rng in [('direct',direct_psi,direct,rd),('blocked_control',control_psi,control,rc)]:
   cw=StreamingCsv(OUT/'logs'/f'stageC_{label}_coarse_sweep{sweep:03d}.csv',["sweep","phase","pass","patch_index","patch_x","patch_y","patch_size","attempts","accepted","acceptance","A_over_R","deltaS_mean","deltaS_std","deltaS_min","deltaS_max","delta_logw_mean","delta_logw_std","log_accept_mean","log_accept_std","patch_l2_mean","local_rms","elapsed_sec"])
   dw=StreamingCsv(OUT/'coarse_logA_diagnostics.csv',TWO_STAGE_COARSE_FIELDS,append=True)
   psi,phi,cm=rqspline_two_stage_coarse_action_patch_update(psi,kernel,coarse_action,action,model,stats,batch_size=64,device=device,passes=1,patch_size=16,step_size=.04,sweep=sweep,writer=cw,diagnostics_writer=dw,rng=rng);cw.close();dw.close()
   pw=StreamingCsv(OUT/'logs'/f'stageC_{label}_detail_sweep{sweep:03d}.csv',["sweep","phase","patch_size","pass","patch_index","patch_x","patch_y","attempts","accepted","acceptance","A_over_R","deltaS_mean","deltaS_std","deltaS_min","deltaS_max","delta_logw_mean","delta_logw_std","log_accept_mean","log_accept_std","patch_l2_mean","local_rms","elapsed_sec"])
   phi,psi,dm=patch_correct(psi,kernel,action,dargs(sweep),pw,rng);pw.close();ca_new.append({'ensemble':label,'sweep':sweep,**cm});da_new.append({'ensemble':label,'sweep':sweep,**dm})
   if label=='direct':direct_psi,direct=psi,phi
   else:control_psi,control=psi,phi
  if sweep in SAVES:
   val=obs(direct,action);fine_new.extend(comparison(reference,val,'direct_L16_stageC',sweep,obs(direct_initial,action)));fine_new.extend({'sweep':sweep,'ensemble':'direct_L16_stageC','observable':k,'value_mean':v} for k,v in ensemble(direct).items())
   cv=obs(control,action);control_new.extend(comparison(reference,cv,'blocked_native_stageC_control',sweep,obs(control_initial,action)))
   for label,psi in [('direct_L16_stageC',direct_psi),('blocked_native_stageC_control',control_psi)]:
    co=obs(psi[:,0::2,0::2],coarse_action)
    for name in ('phi2','phi4','NN','m2','m4','G_pmin_avg'):
     st=max(float(np.std(reference_coarse[name],ddof=1)),1e-15);coarse_new.append({'sweep':sweep,'ensemble':label,'observable':name,'blocked_native_mean':float(np.mean(reference_coarse[name])),'value_mean':float(np.mean(co[name])),'shift_blocked_native_sigma':float((np.mean(co[name])-np.mean(reference_coarse[name]))/st),'width_ratio':float(np.std(co[name],ddof=1)/st)})
 for label,psi in [('direct',direct_psi),('blocked_control',control_psi)]:
  err=float(np.max(np.abs(apply_kernel(__import__('perfect_blocking_upsampling.kernels',fromlist=['inverse_kernel']).inverse_kernel(psi,kernel)[0],kernel)-psi)))
  if err>2e-6:raise RuntimeError(f'{label} roundtrip {err}')
 append(OUT/'direct_coarse_coarse_detail_metrics.csv',fine_existing,fine_new);append(OUT/'blocked_native_coarse_detail_stationarity.csv',control_existing,control_new);append(OUT/'coarse_field_evolution.csv',coarse_existing,coarse_new);append(OUT/'coarse_acceptance.csv',coarse_acc,ca_new);append(OUT/'detail_acceptance_stageC.csv',detail_acc,da_new)
 np.savez_compressed(OUT/'stageC_extension_state_sweep100.npz',direct_phi=direct,direct_psi=direct_psi,control_phi=control,control_psi=control,direct_rng_state=np.array(json.dumps(rd.bit_generator.state)),control_rng_state=np.array(json.dumps(rc.bit_generator.state)))
 (OUT/'stageC_extension_manifest.json').write_text(json.dumps({'start_state':'stageC_pilot_states.npz: sweep20','end_sweep':100,'continuation_rng_seeds':{'direct':2026072203,'blocked_control':2026072204},'coarse':{'P':16,'passes':1,'step':.04,'scheme':'two_stage_coarse_action'},'detail':{'P':16,'passes':10,'sigma':.04}},indent=2)+'\n')
if __name__=='__main__':main()
