#!/usr/bin/env python3
"""Three-way split wrapper for the joint-operator kernel candidate search."""
from __future__ import annotations
import csv, importlib.util, json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve(); ROOT=HERE.parents[1]; PROJECT=ROOT.parent
joint_path=HERE.with_name('run_lam1p0_joint_operator_kernel_search.py')
spec=importlib.util.spec_from_file_location('joint_search',joint_path); mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
base=mod.base
tag='strict_joint_kernel_train3000_val1000_test1000'; out=base.LAM_ROOT/'tests/intermediate/archive_superseded_kernel_searches_20260818'/tag
base.OUT=out; base.CAND_DIR=base.LAM_ROOT/'kernels/candidates'/tag; base.FINAL=base.LAM_ROOT/'tests/final'/tag
rng=np.random.default_rng(20260807)
direct_all=base.load_configs(base.DIRECT); fine_all=base.load_configs(base.FINE)
direct_idx=rng.permutation(len(direct_all))[:5000]; fine_idx=rng.permutation(len(fine_all))[:5000]
direct_parts=[direct_all[direct_idx[i:j]] for i,j in ((0,3000),(3000,4000),(4000,5000))]
fine_parts=[fine_all[fine_idx[i:j]] for i,j in ((0,3000),(3000,4000),(4000,5000))]
# The inherited optimizer sees training configurations only.
base.SEARCH_N_CONFIGS=3000; base.SUBSET_N=1500
original_load=base.load_configs
def train_only(path):
    return direct_parts[0] if Path(path)==base.DIRECT else fine_parts[0]
base.load_configs=train_only
base.main()

def corr_score(a,b):
    pairs=[('phi2','NN'),('phi2','phi4'),('phi2','local_kurtosis_ratio'),('NN','diag'),('action_density','local_kurtosis_ratio')]
    return sum(abs(np.corrcoef(a[x],a[y])[0,1]-np.corrcoef(b[x],b[y])[0,1]) for x,y in pairs)
def evaluate(kernel_path,direct,fine):
    matrix,_=base.eta_matrix_from_json(kernel_path); a=mod.with_joint_observables(base.observable_arrays(direct)); b=mod.with_joint_observables(base.observable_arrays(base.block(fine,matrix)))
    rows=base.full_metrics(a,b,bins=70)
    marginal=sum(base.scalar_score(rows[k]) for k in base.KEY_OBS)
    return marginal+25*corr_score(a,b),corr_score(a,b),rows
candidates=[base.CURRENT_FINAL]+sorted(base.CAND_DIR.glob('*.json'))
val=[]
for p in candidates:
    score,corr,_=evaluate(p,direct_parts[1],fine_parts[1]); val.append({'candidate':str(p),'validation_score':score,'validation_correlation_L1':corr})
val.sort(key=lambda r:r['validation_score']); chosen=Path(val[0]['candidate'])
score,corr,rows=evaluate(chosen,direct_parts[2],fine_parts[2])
out.mkdir(parents=True,exist_ok=True)
with (out/'validation_selection.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=val[0]);w.writeheader();w.writerows(val)
(out/'strict_result.json').write_text(json.dumps({'chosen_validation_candidate':str(chosen),'test_score':score,'test_correlation_L1':corr,'split_seed':20260807,'train':3000,'validation':1000,'test':1000,'test_marginals':rows},indent=2,default=float)+'\n')
np.savez_compressed(out/'split_indices.npz',direct_train=direct_idx[:3000],direct_validation=direct_idx[3000:4000],direct_test=direct_idx[4000:],fine_train=fine_idx[:3000],fine_validation=fine_idx[3000:4000],fine_test=fine_idx[4000:])
print(json.dumps({'chosen':str(chosen),'test_score':score,'test_correlation_L1':corr},indent=2))
