#!/usr/bin/env python3
"""Stage C pilot: exact two-stage coarse plus detail evolution from Stage-B states."""

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
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "src")); sys.path.insert(0, str(PKG / "scripts"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/inverse_rg_mpl")

from perfect_blocking_upsampling.actions import ActionSpec
from perfect_blocking_upsampling.kernels import apply_kernel, load_kernel
from run_calibrated_empirical_blocked_native_detail_only import OBS, comparison, ensemble, obs, write_csv
from run_calibrated_empirical_direct_L16_detail_only import empirical_initial
from run_lam0p2_residual_flow_patch_chain import StreamingCsv, patch_correct
from run_lam1p0_l16to32_rqspline_zeroshot import build_model_from_checkpoint, stationary_stats
from run_lam1p0_rqspline_patchwise import TWO_STAGE_COARSE_FIELDS, rqspline_two_stage_coarse_action_patch_update
from train_lam1p0_flow_detail_pilot import load_phi

OUT = PKG / "runs/lam1p0/calibrated_empirical_patchwise_rethermalization_20260721"
L16_PATH = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
L32_PATH = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
KERNEL_PATH = ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
CFG_PATH = PKG / "run_configs/lam1p0_L16to32_rqspline_balanced_coarse_detail_production.yaml"
SAVES = (0, 1, 2, 5, 10, 20)


class NullWriter:
    def write(self, _row: dict) -> None:
        return None
    def close(self) -> None:
        return None


def detail_args(sweep: int) -> argparse.Namespace:
    return argparse.Namespace(disable_coarse_updates=True, detail_passes=10, fine_proposal_sigma=.04, fine_patch_size=16, passes=0, proposal_sigma=.0, coarse_patch_size=16, global_sweep=sweep, verbose_patch_log=False)


def replay_detail(psi: np.ndarray, kernel, action: ActionSpec, *, seed: int, sweeps: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed); current = psi.copy()
    for sweep in range(1, sweeps + 1):
        _phi, current, _meta = patch_correct(current, kernel, action, detail_args(sweep), NullWriter(), rng)
    phi, _ = __import__("perfect_blocking_upsampling.kernels", fromlist=["inverse_kernel"]).inverse_kernel(current, kernel)
    return current, phi


def recovery_check(reference_rows: list[dict], field: np.ndarray, action: ActionSpec) -> list[dict]:
    values = obs(field, action); result=[]
    for name in ("action_density", "phi2", "phi4", "NN", "m2", "G_pmin_avg"):
        saved = next(r for r in reference_rows if int(r["sweep"]) == 200 and r["observable"] == name)
        actual = float(np.mean(values[name])); expected = float(saved["value_mean"])
        result.append({"observable": name, "saved_stageB_sweep200_mean": expected, "replayed_mean": actual, "absolute_difference": abs(actual - expected)})
    return result


def coarse_observables(coarse: np.ndarray, action: ActionSpec) -> dict[str, np.ndarray]:
    return obs(coarse, action)


def plot_metrics(reference: dict[str, np.ndarray], states: dict[int, np.ndarray], action: ActionSpec) -> None:
    directory=OUT/"plots/coarse_plus_detail";directory.mkdir(parents=True,exist_ok=True)
    for name in ("action_density","phi2","phi4","NN","m2","G_pmin_avg"):
        values=np.concatenate([reference[name]]+[obs(states[s],action)[name] for s in SAVES]);lo,hi=np.quantile(values,[.001,.999]);bins=np.linspace(lo-.05*(hi-lo),hi+.05*(hi-lo),61)
        fig,axes=plt.subplots(2,3,figsize=(10.5,6.5),constrained_layout=True)
        for axis,sweep in zip(axes.flat,SAVES):
            axis.hist(reference[name],bins=bins,density=True,histtype="step",lw=1.5,label="native L32")
            axis.hist(obs(states[sweep],action)[name],bins=bins,density=True,histtype="step",lw=1.3,label=f"Stage C s={sweep}")
            axis.set_title(f"sweep {sweep}");axis.set_xlabel(name)
            if sweep==0:axis.legend(frameon=False,fontsize=7)
        fig.savefig(directory/f"stageC_{name}.pdf");fig.savefig(directory/f"stageC_{name}.png",dpi=180);plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--chains",type=int,default=128);parser.add_argument("--pilot-sweeps",type=int,default=20);args=parser.parse_args()
    if args.pilot_sweeps != 20: raise ValueError("Stage C first report is fixed to the requested 20-sweep pilot")
    cfg=yaml.safe_load(CFG_PATH.read_text()); kernel,_=load_kernel(KERNEL_PATH); action=ActionSpec("phi4_nn",1.,.340301); coarse_action=ActionSpec("phi4_nn",1.,.340301)
    direct_all=load_phi(L16_PATH); native_all=load_phi(L32_PATH); coarse=direct_all[:args.chains].astype(np.float32); native= native_all[2000:2000+args.chains].astype(np.float32)
    # Rebuild the Stage-B initializer with frozen donor bank and replay its saved deterministic detail trajectory.
    from run_lam1p0_empirical_joint_2x2_mixture import features, meta, vectors
    from train_lam1p0_flow_detail_pilot import split_pairs, load_kernel_matrix
    matrix,_=load_kernel_matrix(KERNEL_PATH); donor_pairs=split_pairs(native_all[:1000],matrix); donor_meta=meta(1000,16); raw_h=features(donor_pairs["coarse"],donor_meta); hm,hs=raw_h.mean(0),raw_h.std(0)+1.e-6
    donor_h=(raw_h-hm)/hs; donor_d=vectors(donor_pairs["detail"],donor_meta)
    initialized, z, gamma, selected = empirical_initial(coarse,matrix,donor_h,donor_d,hm,hs,2026072116)
    direct_psi=apply_kernel(initialized,kernel).astype(np.float32)
    direct_psi, direct_phi=replay_detail(direct_psi,kernel,action,seed=2026072117,sweeps=200)
    stageb_rows=list(csv.DictReader((OUT/"direct_coarse_detail_only_metrics.csv").open()))
    recovered=recovery_check(stageb_rows,direct_phi,action)
    if max(r["absolute_difference"] for r in recovered)>2.e-6: raise RuntimeError(f"Stage-B replay mismatch: {recovered}")
    # The matched blocked-native control is likewise replayed from its saved native detail state at sweep 100.
    control_psi=apply_kernel(native,kernel).astype(np.float32)
    control_psi, control_phi=replay_detail(control_psi,kernel,action,seed=2026072100,sweeps=100)
    reference=obs(native,action); reference_coarse=coarse_observables(apply_kernel(native,kernel)[:,0::2,0::2],coarse_action)
    device=torch.device("cpu"); checkpoint=torch.load(ROOT/cfg["flow_checkpoint"],map_location=device,weights_only=False); model,_=build_model_from_checkpoint(checkpoint,lattice_size=16,device=device); stats=stationary_stats(checkpoint["state"]["stats"],lc=16)
    all_rows=[]; coarse_rows=[]; coarse_acc=[]; detail_acc=[]; control_rows=[]; states={}; control_states={}; rng_direct=np.random.default_rng(2026072201); rng_control=np.random.default_rng(2026072202)
    for sweep in range(args.pilot_sweeps+1):
        if sweep in SAVES:
            direct_values=obs(direct_phi,action); all_rows.extend(comparison(reference,direct_values,"direct_L16_stageC",sweep,obs(states[0],action) if 0 in states else direct_values)); all_rows.extend({"sweep":sweep,"ensemble":"direct_L16_stageC","observable":name,"value_mean":value} for name,value in ensemble(direct_phi).items());states[sweep]=direct_phi.copy()
            control_values=obs(control_phi,action);control_rows.extend(comparison(reference,control_values,"blocked_native_stageC_control",sweep,obs(control_states[0],action) if 0 in control_states else control_values));control_states[sweep]=control_phi.copy()
            for label,psi in (("direct_L16_stageC",direct_psi),("blocked_native_stageC_control",control_psi)):
                vals=coarse_observables(psi[:,0::2,0::2],coarse_action)
                for name in ("phi2","phi4","NN","m2","m4","G_pmin_avg"):
                    std=max(float(np.std(reference_coarse[name],ddof=1)),1.e-15)
                    coarse_rows.append({"sweep":sweep,"ensemble":label,"observable":name,"blocked_native_mean":float(np.mean(reference_coarse[name])),"value_mean":float(np.mean(vals[name])),"shift_blocked_native_sigma":float((np.mean(vals[name])-np.mean(reference_coarse[name]))/std),"width_ratio":float(np.std(vals[name],ddof=1)/std)})
        if sweep==args.pilot_sweeps: break
        for label,psi,phi,rng in (("direct",direct_psi,direct_phi,rng_direct),("blocked_control",control_psi,control_phi,rng_control)):
            cwriter=StreamingCsv(OUT/"logs"/f"stageC_{label}_coarse_sweep{sweep+1:03d}.csv",["sweep","phase","pass","patch_index","patch_x","patch_y","patch_size","attempts","accepted","acceptance","A_over_R","deltaS_mean","deltaS_std","deltaS_min","deltaS_max","delta_logw_mean","delta_logw_std","log_accept_mean","log_accept_std","patch_l2_mean","local_rms","elapsed_sec"])
            dwriter=StreamingCsv(OUT/"coarse_logA_diagnostics.csv",TWO_STAGE_COARSE_FIELDS,append=(sweep>0 or label!="direct"))
            psi,phi,cmeta=rqspline_two_stage_coarse_action_patch_update(psi,kernel,coarse_action,action,model,stats,batch_size=64,device=device,passes=1,patch_size=16,step_size=.04,sweep=sweep+1,writer=cwriter,diagnostics_writer=dwriter,rng=rng);cwriter.close();dwriter.close()
            pwriter=StreamingCsv(OUT/"logs"/f"stageC_{label}_detail_sweep{sweep+1:03d}.csv",["sweep","phase","patch_size","pass","patch_index","patch_x","patch_y","attempts","accepted","acceptance","A_over_R","deltaS_mean","deltaS_std","deltaS_min","deltaS_max","delta_logw_mean","delta_logw_std","log_accept_mean","log_accept_std","patch_l2_mean","local_rms","elapsed_sec"])
            phi,psi,dmeta=patch_correct(psi,kernel,action,detail_args(sweep+1),pwriter,rng);pwriter.close()
            coarse_acc.append({"ensemble":label,"sweep":sweep+1,**cmeta});detail_acc.append({"ensemble":label,"sweep":sweep+1,**dmeta})
            if label=="direct": direct_psi,direct_phi=psi,phi
            else: control_psi,control_phi=psi,phi
    for label,psi in (("direct",direct_psi),("blocked_control",control_psi)):
        err=float(np.max(np.abs(apply_kernel(__import__("perfect_blocking_upsampling.kernels",fromlist=["inverse_kernel"]).inverse_kernel(psi,kernel)[0],kernel)-psi)))
        if err>2.e-6: raise RuntimeError(f"{label} Stage C roundtrip failure {err}")
    write_csv(OUT/"direct_coarse_coarse_detail_metrics.csv",all_rows);write_csv(OUT/"blocked_native_coarse_detail_stationarity.csv",control_rows);write_csv(OUT/"coarse_field_evolution.csv",coarse_rows);write_csv(OUT/"coarse_acceptance.csv",coarse_acc);write_csv(OUT/"detail_acceptance_stageC.csv",detail_acc);write_csv(OUT/"stageC_start_state_inventory.csv",[{"state":"direct Stage-B sweep200 replay","chains":args.chains,"source":"direct native L16 indices 0-127","replay_validated":True},{"state":"blocked-native control Stage-B sweep100 replay","chains":args.chains,"source":"native L32 indices 2000-2127 blocked to L16","replay_validated":True}]+recovered)
    np.savez_compressed(OUT/"stageC_pilot_states.npz",direct_start=states[0],direct_sweep20=states[20],control_start=control_states[0],control_sweep20=control_states[20])
    plot_metrics(reference,states,action)
    spec="""# Stage C Exact Coarse+Detail Algorithm\n\n- Ordering: one coarse update then ten detail passes per combined sweep.\n- Coarse: validated `two_stage_coarse_action` RQ-spline-coordinate kernel, P=16 (one periodic coarse patch/pass), one pass/sweep, step 0.04. Stage 1 targets the coarse phi4_nn action. Stage 2 accepts the propagated fine state with `-Delta S_f + Delta S_c + Delta log J_flow`. Physical transformed details are held fixed during the coarse proposal; the flow is used only to calculate the exact coordinate Jacobian.\n- Detail: unchanged symmetric Gaussian transformed-detail patches, P=16, ten passes/sweep, sigma=0.04, target-action A/R.\n- The empirical initializer/density is not used after Stage-C sweep 0.\n"""
    (OUT/"coarse_detail_algorithm_specification.md").write_text(spec)
    (OUT/"stageC_summary.md").write_text("# Stage C Pilot\n\nCompleted the requested sweep-20 coarse-plus-detail pilot only. Inspect acceptance and controls before extending to sweeps 100--200.\n")


if __name__=="__main__": main()
