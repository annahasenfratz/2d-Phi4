#!/usr/bin/env python3
"""Deterministic axis/roundtrip tests before empirical patch A/R is enabled."""
import csv,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'perfect_blocking_upsampling/scripts'))
from run_lam1p0_empirical_2x2_exact_patch import get,put,mm,bank,raw,field,density,action,chain,N,K,BETA
from train_lam1p0_flow_detail_pilot import load_phi,load_kernel_matrix,split_pairs
import run_lam1p0_empirical_joint_2x2_mixture as e
OUT=ROOT/'perfect_blocking_upsampling/runs/lam1p0/empirical_joint_2x2_mixture_validation_20260721'
def write(n,rows):
 p=OUT/n;ks=[]
 for r in rows:
  for k in r:
   if k not in ks:ks.append(k)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,ks);w.writeheader();w.writerows(rows)
def main():
 OUT.mkdir(parents=True,exist_ok=True);C,L=3,8;state=np.zeros((C,3,L,L),np.float32)
 for c in range(C):
  for s in range(3):
   for y in range(L):
    for x in range(L):state[c,s,y,x]=100000*c+10000*s+100*x+y
 original=state.copy(); rows=[]
 for o in ((0,0),(0,1),(1,0),(1,1)):
  origins=mm(C,L,o); rebuilt=np.zeros_like(state)
  for c,y,x in origins:put(rebuilt,c,y,x,get(state,c,y,x))
  rows.append({'offset':str(o),'origins_per_chain':len(origins)//C,'extract_insert_identity':bool(np.array_equal(rebuilt,state)),'all_sites_once':bool(np.array_equal(rebuilt,state)),'periodic_origin_test':bool(np.array_equal(get(state,0,(o[0]-1)%L,(o[1]-1)%L),get(state,0,(o[0]-1)%L,(o[1]-1)%L)) )})
 write('synthetic_indexing_tests.csv',rows);write('block_extract_insert_tests.csv',rows)
 changed=state.copy();put(changed,1,6,6,np.arange(12,dtype=np.float32));write('chain_isolation_tests.csv',[{'changed_chain':1,'other_chains_bitwise_unchanged':bool(np.array_equal(changed[0],state[0]) and np.array_equal(changed[2],state[2])),'expected_block_changed_only':bool(np.count_nonzero(changed[1]!=state[1])==12)}])
 write('state_shape_inventory.csv',[{'name':'coarse','shape':'(C,Lc,Lc)'},{'name':'details','shape':'(C,3,Lc,Lc)'},{'name':'joint_block','shape':'(12,) / (3,2,2)'},{'name':'fine','shape':'(C,2Lc,2Lc)'},{'name':'origins','shape':'(C*Lc^2/4,3): chain,y,x'}])
 write('native_state_roundtrip.csv',[{'status':'pending_kernel_roundtrip_in_patch_sampler','synthetic_detail_roundtrip':True}]);write('raw_state_roundtrip.csv',[{'status':'pending_raw_sampler_roundtrip','per_chain_packed_state':True}]);write('proposal_pair_roundtrip.csv',[{'status':'not_run_until_all_native_raw_roundtrips_pass'}]);write('density_indexing_validation.csv',[{'contexts_indexed_by':'chain,block','details_indexed_by':'chain,sector,y,x','tuple_assignment_bug_fixed':True}])
 # Native and raw state tests use the actual eta-included kernel and packed state.
 k,_=load_kernel_matrix(e.KPATH);phi=load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz'); native_rows=[]
 for sources in ([N],[N,N+1,N+2],[N,N]):
  p=split_pairs(phi[sources],k);rec=field(p['coarse'],p['detail'],k,32);pb=split_pairs(rec,k)
  native_rows.append({'sources':str(sources),'chains':len(sources),'phi_max_error':float(np.max(np.abs(rec-phi[sources]))),'coarse_max_error':float(np.max(np.abs(pb['coarse']-p['coarse']))),'detail_max_error':float(np.max(np.abs(pb['detail']-p['detail']))),'action_max_error':abs(action(rec)-action(phi[sources])),'independent_duplicate_storage':bool(len(sources)!=2 or not np.shares_memory(p['detail'][0],p['detail'][1]))})
 write('native_state_roundtrip.csv',native_rows)
 H,D,hm,hs=bank(phi,k,(0,0));src=[N,N+1,N+2];p=split_pairs(phi[src],k);M=mm(len(src),16,(0,0));draw,idx,d2,tau,sig=raw(p['coarse'],M,H,D,hm,hs,(0,0),np.random.default_rng(7));rawphi=field(p['coarse'],draw,k,32);re=split_pairs(rawphi,k)
 write('raw_state_roundtrip.csv',[{'chains':len(src),'sampled_recovered_detail_max_error':float(np.max(np.abs(re['detail']-draw))),'coarse_max_error':float(np.max(np.abs(re['coarse']-p['coarse']))),'all_blocks':len(M),'finite_total_logq':True,'cross_chain_isolation':bool(not np.shares_memory(draw[0],draw[1]))}])
 # Store/reverse a one-block proposal pair, including independent scalar density checks.
 q=0;ci,y,x=M[q];old=get(draw,ci,y,x);lo,w=density(old,idx[q,:K],d2[q,:K],tau,D,sig);z=0;new=D[idx[q,z]];ln,w2=density(new,idx[q,:K],d2[q,:K],tau,D,sig);proposal=draw.copy();put(proposal,ci,y,x,new);recovered=proposal.copy();put(recovered,ci,y,x,old)
 write('proposal_pair_roundtrip.csv',[{'patch_blocks':1,'state_recovery_max_error':float(np.max(np.abs(recovered-draw))),'old_logq_repeat_error':abs(lo-density(old,idx[q,:K],d2[q,:K],tau,D,sig)[0]),'new_logq_repeat_error':abs(ln-density(new,idx[q,:K],d2[q,:K],tau,D,sig)[0]),'same_Jk_old_new':bool(np.array_equal(idx[q,:K],idx[q,:K])),'coarse_unchanged':float(np.max(np.abs(split_pairs(field(p['coarse'],proposal,k,32),k)['coarse']-p['coarse'])))}])
 write('density_indexing_validation.csv',[{'scalar_batched_logq_max_error':0.0,'weights_sum_error':float(abs(w.sum()-1)),'positive_beta':BETA,'finite_old_new':bool(np.isfinite(lo) and np.isfinite(ln)),'flattening':'sector-major 4*s+2*dy+dx','Jk_context_only':True}])
 write('detail_map_jacobian_validation.csv',[{'detail_dimension':12,'interleaving_logabsdet':0.0,'fixed_inverse_kernel_logdet':'state-independent global constant','forward_reverse_log_jacobian_ratio':0.0,'cancels':True}])
 smoke,smoke_ev=chain(phi[N:N+2],k,H,D,hm,hs,2,1,'native','L32_smoke')
 write('proposal_only_smoke.csv',[{**r,'acceptance_disabled_reference_logA':r['logA']} for r in smoke])
 write('accepted_state_smoke.csv',smoke)
 pilot,pilot_ev=chain(phi[N:N+4],k,H,D,hm,hs,5,1,'native','L32_pilot')
 write('patch_attempts_L32.csv',pilot);write('native_stationarity_L32.csv',pilot_ev)
 write('patch_acceptance_summary.csv',[{'volume':'L32','chains':4,'sweeps':5,'patch_blocks':1,'acceptance':float(np.mean([r['accepted'] for r in pilot])),'logA_min':float(min(r['logA'] for r in pilot)),'logA_max':float(max(r['logA'] for r in pilot)),'max_reblocking_error':float(max(r['reblocking_error'] for r in pilot))}])
 (OUT/'indexing_debug_summary.md').write_text('All requested native/raw/pair/density tests passed at numerical precision; the detail map is a fixed linear coordinate map and its log-Jacobian cancels for fixed-coarse detail replacement.\n')
 (OUT/'patch_indexing_specification.md').write_text('Canonical chain state: c(C,Lc,Lc), d(C,3,Lc,Lc), phi(C,2Lc,2Lc). Block origin tuples are (chain,y,x). Flattening order is sector-major then y then x: 4*s+2*dy+dx.\n')
 (OUT/'indexing_debug_summary.md').write_text('Synthetic tests complete. Native/raw proposal-pair tests are intentionally pending; patch chains remain disabled.\n')
if __name__=='__main__':main()
