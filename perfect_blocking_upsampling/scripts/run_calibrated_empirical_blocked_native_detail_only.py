#!/usr/bin/env python3
"""Part I: exact fixed-coarse detail correction from frozen empirical starts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "scripts"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/inverse_rg_mpl")

from perfect_blocking_upsampling.actions import ActionSpec, action_total
from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel, load_kernel
from perfect_blocking_upsampling.observables import second_moment_components
from run_lam0p2_residual_flow_patch_chain import StreamingCsv, patch_correct

FRESH = PKG / "runs/lam1p0/fresh_disjoint_calibrated_empirical_validation_20260721"
OUT = PKG / "runs/lam1p0/calibrated_empirical_patchwise_rethermalization_20260721"
KERNEL_PATH = ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
SAVES = (0, 1, 2, 5, 10, 20, 50, 100)
OBS = ("action_density", "phi2", "phi4", "local_kurtosis", "NN", "diag", "2nn", "m2", "m4", "G_pmin_avg")


def write_csv(path: Path, rows: list[dict]) -> None:
    cols = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader(); writer.writerows(rows)


def obs(phi: np.ndarray, action: ActionSpec) -> dict[str, np.ndarray]:
    x = np.asarray(phi, dtype=np.float64)
    phi2 = np.mean(x*x, axis=(1, 2)); phi4 = np.mean(x**4, axis=(1, 2))
    nn = .5*np.mean(x*np.roll(x, -1, 1)+x*np.roll(x, -1, 2), axis=(1, 2))
    ft = np.fft.fft2(x, axes=(1, 2))
    return {"action_density": 1.-2.*phi2+phi4-4.*action.kappa*nn, "phi2": phi2, "phi4": phi4,
            "local_kurtosis": phi4/np.maximum(phi2*phi2, 1.e-15), "NN": nn,
            "diag": np.mean(x*np.roll(np.roll(x,-1,1),-1,2),axis=(1,2)),
            "2nn": .5*np.mean(x*np.roll(x,-2,1)+x*np.roll(x,-2,2),axis=(1,2)),
            "m2": np.mean(x,axis=(1,2))**2, "m4": np.mean(x,axis=(1,2))**4,
            "G_pmin_avg": .5*(np.abs(ft[:,1,0])**2+np.abs(ft[:,0,1])**2)/(x.shape[1]**2)}


def ensemble(phi: np.ndarray) -> dict[str, float]:
    x=np.asarray(phi,dtype=np.float64); m=x.mean((1,2)); m2=float(np.mean(m*m)); m4=float(np.mean(m**4)); sm=second_moment_components(x)
    return {"Binder": float(1.-m4/(3.*m2*m2)) if m2 else float("nan"), "chi": float(x.shape[1]**2*(m2-np.mean(m)**2)), "xi_over_L": float(sm["xi_over_L"])}


def overlap(a: np.ndarray,b: np.ndarray) -> float:
    lo,hi=min(float(a.min()),float(b.min())),max(float(a.max()),float(b.max())); bins=np.linspace(lo,hi,81) if hi>lo else np.array([lo-1,hi+1])
    ha,_=np.histogram(a,bins,density=True); hb,_=np.histogram(b,bins,density=True)
    return float(np.minimum(ha,hb).sum()*(bins[1]-bins[0]))


def comparison(reference: dict[str,np.ndarray], current: dict[str,np.ndarray], label: str, sweep: int, sweep0: dict[str,np.ndarray]) -> list[dict]:
    result=[]
    for name in OBS:
        a,b=reference[name],current[name]; std=max(float(np.std(a,ddof=1)),1.e-15); d=b-a
        result.append({"sweep":sweep,"ensemble":label,"observable":name,"native_mean":float(a.mean()),"value_mean":float(b.mean()),
                       "shift_native_sigma":float((b.mean()-a.mean())/std),"width_ratio":float(np.std(b,ddof=1)/std),"KS":float(ks_2samp(a,b).statistic),"overlap":overlap(a,b),"W1":float(wasserstein_distance(a,b)),
                       "q01_coverage":float(np.mean((b>=np.quantile(a,.01))&(b<=np.quantile(a,.99)))),"q05_coverage":float(np.mean((b>=np.quantile(a,.05))&(b<=np.quantile(a,.95)))),"q10_coverage":float(np.mean((b>=np.quantile(a,.1))&(b<=np.quantile(a,.9)))),
                       "paired_change_sweep0_mean":float(np.mean(b-sweep0[name])),"paired_difference_mean":float(d.mean()),"paired_difference_se":float(d.std(ddof=1)/np.sqrt(len(d)))})
    return result


def run_chain(label: str, psi0: np.ndarray, kernel, action: ActionSpec, rng: np.random.Generator, reference: dict[str,np.ndarray], source_indices: np.ndarray, n_sweeps: int, all_metrics: list[dict], accept_rows: list[dict], states: dict[str,dict[int,np.ndarray]]) -> None:
    psi=psi0.astype(np.float32).copy(); phi,_=inverse_kernel(psi,kernel); initial=obs(phi,action)
    for sweep in range(n_sweeps+1):
        if sweep in SAVES:
            current=obs(phi,action); all_metrics.extend(comparison(reference,current,label,sweep,initial))
            all_metrics.extend({"sweep":sweep,"ensemble":label,"observable":name,"value_mean":value} for name,value in ensemble(phi).items())
            states[label][sweep]=phi.copy()
        if sweep == n_sweeps: break
        writer=StreamingCsv(OUT/"logs"/f"{label}_patches_sweep{sweep+1:03d}.csv", ["sweep","phase","patch_size","pass","patch_index","patch_x","patch_y","attempts","accepted","acceptance","A_over_R","deltaS_mean","deltaS_std","deltaS_min","deltaS_max","delta_logw_mean","delta_logw_std","log_accept_mean","log_accept_std","patch_l2_mean","local_rms","elapsed_sec"])
        args=argparse.Namespace(disable_coarse_updates=True,detail_passes=10,fine_proposal_sigma=.04,fine_patch_size=16,passes=0,proposal_sigma=.0,coarse_patch_size=16,global_sweep=sweep+1,verbose_patch_log=False)
        phi,psi,meta=patch_correct(psi,kernel,action,args,writer,rng); writer.close()
        accept_rows.append({"ensemble":label,"sweep":sweep+1,**meta})
    final_coarse=apply_kernel(phi,kernel)[:,0::2,0::2]; fixed_coarse=psi0[:,0::2,0::2]
    if np.max(np.abs(final_coarse-fixed_coarse)) > 2.e-6: raise RuntimeError(f"{label} fixed-coarse reblocking failure")


def plots(reference: dict[str,np.ndarray], states: dict[str,dict[int,np.ndarray]], action: ActionSpec) -> None:
    directory=OUT/"plots"; directory.mkdir(exist_ok=True)
    for name in ("action_density","phi2","phi4","NN","m2","G_pmin_avg"):
        values=[]
        for label in states:
            for sweep in (0,10,20,50,100): values.append(obs(states[label][sweep],action)[name])
        allv=np.concatenate([reference[name],*values]); lo,hi=np.quantile(allv,[.001,.999]); bins=np.linspace(lo-.05*(hi-lo),hi+.05*(hi-lo),61)
        fig,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
        for axis,label in zip(axes,("native_detail","calibrated_empirical")):
            axis.hist(reference[name],bins=bins,density=True,histtype="step",lw=1.5,label="native target")
            for sweep in (0,10,20,50,100): axis.hist(obs(states[label][sweep],action)[name],bins=bins,density=True,histtype="step",lw=1.,label=f"s={sweep}")
            axis.set_title(label);axis.set_xlabel(name);axis.legend(frameon=False,fontsize=7)
        fig.savefig(directory/f"blocked_native_detail_only_{name}.pdf");fig.savefig(directory/f"blocked_native_detail_only_{name}.png",dpi=180);plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--chains",type=int,default=128); parser.add_argument("--sweeps",type=int,default=100); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"logs").mkdir(exist_ok=True)
    data=np.load(FRESH/"paired_fields_L32.npz"); native=data["native"][:args.chains].copy(); raw=data["raw"][:args.chains].copy(); calibrated=data["calibrated"][:args.chains].copy()
    kernel,_=load_kernel(KERNEL_PATH); action=ActionSpec("phi4_nn",1.,.340301)
    native_psi=apply_kernel(native,kernel).astype(np.float32); raw_psi=apply_kernel(raw,kernel).astype(np.float32); calibrated_psi=apply_kernel(calibrated,kernel).astype(np.float32)
    c=native_psi[:,0::2,0::2]
    for label,psi in (("native_detail",native_psi),("raw_empirical",raw_psi),("calibrated_empirical",calibrated_psi)):
        err=float(np.max(np.abs(psi[:,0::2,0::2]-c)))
        if err>2.e-6: raise RuntimeError(f"{label} initial coarse mismatch: {err}")
    ref=obs(native,action); rows=[]; acceptance=[]; states={"native_detail":{},"raw_empirical":{},"calibrated_empirical":{}}
    for index,(label,psi) in enumerate((("native_detail",native_psi),("raw_empirical",raw_psi),("calibrated_empirical",calibrated_psi))):
        run_chain(label,psi,kernel,action,np.random.default_rng(2026072100+index),ref,np.arange(2000,2000+args.chains),args.sweeps,rows,acceptance,states)
    write_csv(OUT/"blocked_native_detail_only_metrics.csv",rows); write_csv(OUT/"blocked_native_detail_acceptance.csv",acceptance)
    native_stationarity=[r for r in rows if r["ensemble"]=="native_detail"]
    write_csv(OUT/"blocked_native_stationarity.csv",native_stationarity)
    evo=[]
    for label,by_sweep in states.items():
        for sweep,field in by_sweep.items():
            val=obs(field,action)
            evo.extend({"ensemble":label,"sweep":sweep,"chain_id":i,**{key:float(val[key][i]) for key in OBS}} for i in range(args.chains))
    write_csv(OUT/"observable_evolution.csv",evo); plots(ref,states,action)
    spec={"phase":"Part I only: blocked-native fixed-coarse detail updates","initializer":"frozen calibrated empirical joint 2x2 proposal; used at sweep 0 only","empirical":{"Ndonor":1000,"k":8,"tau":"q25","beta":.01,"kernel":"diagonal joint 12D","radial":"D01,D10 *= 0.97*exp((0.32/Lc) z), D11 unchanged"},"patch_algorithm":{"source":"run_lam0p2_residual_flow_patch_chain.patch_correct","mode":"detail_only","detail_patch_size":16,"detail_passes_per_sweep":10,"fine_proposal_sigma":.04,"patches_per_pass":"ceil(2*Lf^2/P^2)=8","proposal":"symmetric Gaussian random walk in non-ee psi details","acceptance":"target action only; empirical q omitted; fixed linear coordinate Jacobian cancels"},"chains":args.chains,"sweeps":args.sweeps,"target_indices":[2000,2000+args.chains-1]}
    (OUT/"initializer_specification.json").write_text(json.dumps(spec,indent=2)+"\n")
    (OUT/"reused_patch_algorithm_specification.md").write_text("# Reused Detail Updater\n\n`patch_correct` from `run_lam0p2_residual_flow_patch_chain.py` performs symmetric Gaussian patches in the three non-even-even transformed-detail sectors. The fixed even-even transformed coarse field is restored after every proposal. Acceptance is `min(1, exp[-Delta S_f])`; no empirical mixture density is evaluated or included. The fixed linear detail map has a state-independent Jacobian, which cancels in forward/reverse A/R.\n")
    (OUT/"summary.md").write_text("# Part I Running\n\nBlocked-native fixed-coarse detail-only test completed. Part II direct-coarse and coarse-plus-detail stages were not run.\n")

if __name__=="__main__": main()
