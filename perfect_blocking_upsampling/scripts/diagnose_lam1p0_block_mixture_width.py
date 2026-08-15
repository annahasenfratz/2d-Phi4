#!/usr/bin/env python3
"""Frozen-checkpoint sampler instrumentation and width controls for K=4 blocks."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'perfect_blocking_upsampling' / 'scripts'))
from train_lam1p0_block_causal_mixture import Mix
from train_lam1p0_flow_detail_pilot import load_kernel_matrix, load_phi, split_pairs
from train_lam1p0_autoregressive_detail_flow import torch_kernel_fft, torch_inverse_kernel
from train_lam1p0_local_multistage_rqspline import assemble_psi, metrics_rows, write_csv

RUN = ROOT / 'perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_block_causal_2x2_detail_mixture_20260721'


def action_parts(phi):
    p2 = (phi * phi).mean((1, 2)); p4 = (phi**4).mean((1, 2))
    nnx = (phi * np.roll(phi, -1, 2)).mean((1, 2)); nny = (phi * np.roll(phi, -1, 1)).mean((1, 2))
    return p2, p4, .5 * (nnx + nny)


def blocks_meta(n, length):
    return np.array([(i, y, x) for i in range(n) for y in range(0, length, 2) for x in range(0, length, 2)], dtype=np.int32)


def block_features(c, details, stage, meta):
    L = c.shape[1]; rows = []
    for i, y, x in meta:
        row = [c[i, (y+dy) % L, (x+dx) % L] for dy in range(-3, 4) for dx in range(-3, 4)]
        for s in range(stage):
            for by in (-1, 0, 1):
                for bx in (-1, 0, 1):
                    yy, xx = 2 * ((y//2 + by) % (L//2)), 2 * ((x//2 + bx) % (L//2))
                    row.extend(details[s][i, yy:yy+2, xx:xx+2].reshape(-1))
        rows.append(row)
    return np.asarray(rows, np.float32)


def load_stage(stage):
    ck = torch.load(RUN / f'stage{stage}_K4_full.pt', map_location='cpu', weights_only=False)
    model = Mix(len(ck['xmean'])); model.load_state_dict(ck['model']); model.eval()
    return model, ck


def stage_sample(model, ck, features, generator, alpha=1.0, mode='categorical', stochastic=True, mode_scales=(1.,1.,1.,1.)):
    x = torch.from_numpy((features - ck['xmean']) / ck['xstd'])
    with torch.no_grad(): weights, means, chol = model(x)
    probs = torch.softmax(weights, -1)
    if mode in ('argmax', 'argmax_noise'):
        comp = probs.argmax(-1)
    else:
        comp = torch.multinomial(probs, 1, generator=generator).squeeze(1)
    eps = torch.randn((len(x), 4), generator=generator) if stochastic and mode not in ('categorical_mean', 'mixture_mean') else torch.zeros((len(x), 4))
    if mode == 'mixture_mean': sample = (probs[..., None] * means).sum(1); comp = probs.argmax(-1)
    else:
        L = chol[torch.arange(len(x)), comp] * alpha; disp = torch.bmm(L, eps[..., None]).squeeze(-1)
        basis = torch.tensor([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],dtype=disp.dtype) / 2
        disp = (disp @ basis.T * torch.tensor(mode_scales,dtype=disp.dtype)) @ basis
        sample = means[torch.arange(len(x)), comp] + disp
    physical = sample.numpy() * ck['ystd'] + ck['ymean']
    # Exact variance decomposition at each context.
    W = torch.einsum('nk,nkij->nij', probs, chol @ chol.transpose(-1, -2))
    mm = (probs[..., None] * means).sum(1); delta = means - mm[:, None]
    B = torch.einsum('nk,nki,nkj->nij', probs, delta, delta)
    meta = {'context': x.numpy(), 'pi': probs.numpy(), 'mu': means.numpy(), 'L': chol.numpy(), 'ystd': np.asarray(ck['ystd']), 'component': comp.numpy(), 'epsilon': eps.numpy(), 'normalized_sample': sample.numpy(), 'physical_sample': physical, 'trace_W': torch.diagonal(W, dim1=-2, dim2=-1).sum(-1).numpy(), 'trace_B': torch.diagonal(B, dim1=-2, dim2=-1).sum(-1).numpy()}
    return physical, meta


def sample_field(phi, alpha=1., mode='categorical', sector_mask=(1,1,1), teacher=False, seed=20260721, mode_scales=(1.,1.,1.,1.)):
    k, _ = load_kernel_matrix(ROOT / 'perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json')
    p = split_pairs(phi, k); c = p['coarse']; native = [p['detail'][:, i].copy() for i in range(3)]; d = [np.zeros_like(c) for _ in range(3)]
    meta = blocks_meta(len(phi), c.shape[1]); g = torch.Generator().manual_seed(seed); internals = []
    for s in range(3):
        model, ck = load_stage(s); prior = native if teacher else d
        vals, info = stage_sample(model, ck, block_features(c, prior, s, meta), g, alpha, 'mixture_mean' if not sector_mask[s] else mode, bool(sector_mask[s]), mode_scales)
        for q, (i,y,x) in enumerate(meta): d[s][i,y:y+2,x:x+2] = vals[q].reshape(2,2) if sector_mask[s] else (native[s][i,y:y+2,x:x+2] if teacher else vals[q].reshape(2,2))
        info['stage'] = s; internals.append(info)
    psi = assemble_psi(torch.from_numpy(c), *[torch.from_numpy(x) for x in d])
    gen = torch_inverse_kernel(psi, torch_kernel_fft(k, phi.shape[1], torch.device('cpu'))).detach().numpy()
    return gen, internals, meta, d


def metric_row(native, generated, label, control):
    m = {r['observable']: r for r in metrics_rows(native, generated, label, 'whole')}; an = action_parts(native); ag = action_parts(generated)
    act_n = -an[0] + .5 * an[1] - 4 * .340301 * an[2]
    act_g = -ag[0] + .5 * ag[1] - 4 * .340301 * ag[2]
    tails = {f'action_q{int(q*100):02d}_occupancy': float(np.mean(act_g <= np.quantile(act_n, q))) for q in (.01, .05, .10)}
    return {'label': label, 'control': control, 'action_shift': m['action_density']['shift_native_sigma'], 'action_width': m['action_density']['std_ratio'], 'action_KS':m['action_density']['KS'], **tails, 'phi2_shift':m['phi2']['shift_native_sigma'], 'phi2_width':m['phi2']['std_ratio'], 'phi4_shift':m['phi4']['shift_native_sigma'], 'phi4_width':m['phi4']['std_ratio'], 'NN_shift':m['NN']['shift_native_sigma'], 'NN_width':m['NN']['std_ratio'], 'quad_delta':float((-(ag[0]-an[0])).mean()), 'quartic_delta':float((.5*(ag[1]-an[1])).mean()), 'NN_action_delta':float((-4*.340301*(ag[2]-an[2])).mean()), 'nonfinite':int(np.size(generated)-np.isfinite(generated).sum())}


BLOCK_BASIS = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=float) / 2
MODE_NAMES = ('u', 'sx', 'sy', 'cb')


def internal_summary(internals, label, context_mode):
    rows = []
    for s, info in enumerate(internals):
        pi, mu, L = info['pi'], info['mu'], info['L']
        scale = info['ystd']
        cov = (L @ np.swapaxes(L, -1, -2)) * scale[None, None, :, None] * scale[None, None, None, :]
        W = np.einsum('nk,nkij->nij', pi, cov)
        means_physical = mu * scale[None, None, :] 
        mean = np.einsum('nk,nki->ni', pi, means_physical)
        delta = means_physical - mean[:, None]
        B = np.einsum('nk,nki,nkj->nij', pi, delta, delta)
        for j, name in enumerate(MODE_NAMES):
            v = BLOCK_BASIS[j]
            rows.append({'label':label, 'context_mode':context_mode, 'stage':s, 'mode':name,
                         'within_variance':float(np.einsum('i,nij,j->n',v,W,v).mean()),
                         'between_variance':float(np.einsum('i,nij,j->n',v,B,v).mean()),
                         'mean_trace_W':float(np.trace(W,axis1=1,axis2=2).mean()),
                         'mean_trace_B':float(np.trace(B,axis1=1,axis2=2).mean()),
                         'mean_component_entropy':float((-pi*np.log(np.maximum(pi,1e-12))).sum(1).mean()),
                         'eig_W_min':float(np.linalg.eigvalsh(W).min()), 'eig_W_max':float(np.linalg.eigvalsh(W).max())})
    return rows


def native_mode_rows(phi, label):
    k, _ = load_kernel_matrix(ROOT / 'perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json')
    details = split_pairs(phi, k)['detail']
    rows=[]
    for s in range(3):
        blocks=np.asarray([details[i,s,y:y+2,x:x+2].reshape(-1) for i in range(len(phi)) for y in range(0,details.shape[2],2) for x in range(0,details.shape[3],2)])
        cov=np.cov(blocks,rowvar=False)
        for j,name in enumerate(MODE_NAMES): rows.append({'label':label,'context_mode':'native_global_reference','stage':s,'mode':name,'native_variance':float(BLOCK_BASIS[j]@cov@BLOCK_BASIS[j])})
    return rows


def bond_rows(phi, generated, label):
    rows=[]
    # Fine sites inherit their aligned 2x2-coarse block index: floor(fine/4).
    L=phi.shape[1]; bid=(np.arange(L)//4)[:,None]*(L//4)+(np.arange(L)//4)[None,:]
    for axis,name in ((1,'y'),(2,'x')):
        same = bid == np.roll(bid,-1,axis=axis-1)
        for group,mask in (('intra_block',same),('inter_block',~same)):
            for which,a in (('native',phi),('generated',generated)):
                prod=a*np.roll(a,-1,axis=axis)
                vals=prod[:,mask]
                rows.append({'label':label,'field':which,'direction':name,'bond_class':group,'bonds_per_config':int(mask.sum()),'NN_mean':float(vals.mean()),'NN_variance':float(vals.mean(1).var(ddof=1)),'action_contribution_mean':float((-2*.340301*vals).mean()),'action_contribution_variance':float((-2*.340301*vals).mean(1).var(ddof=1))})
    return rows


def neighbor_cov_rows(native_d, generated_d, label):
    rows=[]
    for s in range(3):
        for which,d in (('native',native_d[s]),('generated',generated_d[s])):
            b=np.asarray([[d[i,y:y+2,x:x+2].reshape(-1) for y in range(0,d.shape[1],2) for x in range(0,d.shape[2],2)] for i in range(len(d))]).reshape(len(d),d.shape[1]//2,d.shape[2]//2,4)
            for dy,dx,direction in ((0,1,'x_adjacent'),(1,0,'y_adjacent')):
                a=b; z=np.roll(b,(-dy,-dx),(1,2))
                for j,name in enumerate(MODE_NAMES): rows.append({'label':label,'field':which,'stage':s,'direction':direction,'quantity':name,'covariance':float(np.cov((a@BLOCK_BASIS[j]).ravel(),(z@BLOCK_BASIS[j]).ravel())[0,1])})
                rows.append({'label':label,'field':which,'stage':s,'direction':direction,'quantity':'block_mean','covariance':float(np.cov(a.mean(-1).ravel(),z.mean(-1).ravel())[0,1])})
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    out = RUN; (out/'plots').mkdir(exist_ok=True)
    phi = load_phi(ROOT / 'data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz')[80:100]
    base, internal, meta, base_d = sample_field(phi, seed=77)
    # Metadata schema and checkpoint-frozen sampler smoke.
    (out/'sampler_internal_metadata_schema.md').write_text('Per block: normalized context, pi[4], mu[4,4], L[4,4,4], selected component, epsilon[4], normalized/physical block sample, stage, block origin and source configuration.\n')
    smoke, _, _, _ = sample_field(phi, seed=77)
    write_csv(out/'sampler_refactor_validation.csv', [{'fixed_seed_max_field_difference':float(np.max(np.abs(base-smoke))), 'blocks':len(meta), 'all_blocks_once':bool(len(meta)==len(phi)*64)}])
    vr=[]
    for s, info in enumerate(internal):
        vr.append({'stage':s,'trace_W_mean':float(info['trace_W'].mean()),'trace_B_mean':float(info['trace_B'].mean()),'within_fraction':float(info['trace_W'].mean()/(info['trace_W'].mean()+info['trace_B'].mean()))})
    write_csv(out/'variance_decomposition.csv',vr)
    if args.smoke: return
    rows=[]
    for alpha in (.5,.65,.75,.85,1.): rows.append(metric_row(phi, sample_field(phi, alpha=alpha, seed=33)[0], 'L32', f'alpha_{alpha}'))
    write_csv(out/'covariance_scale_scan_L32.csv',rows)
    ab=[]
    for mode in ('categorical','argmax','categorical_mean','argmax_noise','mixture_mean'): ab.append(metric_row(phi,sample_field(phi,mode=mode,seed=34)[0],'L32',mode))
    write_csv(out/'component_selection_ablation.csv',ab)
    masks={'q01_only':(1,0,0),'q10_only':(0,1,0),'q11_only':(0,0,1),'q01_q10':(1,1,0),'q01_q11':(1,0,1),'q10_q11':(0,1,1),'all':(1,1,1),'means':(0,0,0)}
    write_csv(out/'sector_stochasticity_ablation.csv',[metric_row(phi,sample_field(phi,sector_mask=v,seed=41)[0],'L32',k) for k,v in masks.items()])
    scans=[]
    for mode_i,name,vals in [(0,'u',(.75,1.,1.25)),(1,'sx',(.5,.75,1.)),(2,'sy',(.5,.75,1.)),(3,'cb',(.5,.75,1.))]:
        for value in vals:
            scale=[1.,1.,1.,1.];scale[mode_i]=value;scans.append(metric_row(phi,sample_field(phi,mode_scales=tuple(scale),seed=42)[0],'L32',f'{name}_{value}'))
    write_csv(out/'directional_covariance_scan_L32.csv',scans)
    # Teacher-forced contexts use native prior sectors only for this diagnostic;
    # normal sampling remains fully free running.
    tf_gen, tf_internal, _, tf_d = sample_field(phi, teacher=True, seed=51)
    fr_gen, fr_internal, _, fr_d = sample_field(phi, teacher=False, seed=51)
    tf_rows = internal_summary(tf_internal, 'L32', 'teacher_forced') + internal_summary(fr_internal, 'L32', 'free_running') + native_mode_rows(phi, 'L32')
    write_csv(out/'teacher_forced_free_running_covariance.csv', tf_rows)
    write_csv(out/'teacher_forced_free_running_width.csv', [
        metric_row(phi, tf_gen, 'L32', 'teacher_forced_all_prior_details_native'),
        metric_row(phi, fr_gen, 'L32', 'free_running_all_prior_details_generated')])
    modes = internal_summary(fr_internal, 'L32', 'free_running') + native_mode_rows(phi, 'L32')
    # Merge model and native rows by stage/mode for explicit ratios.
    native_lookup={(r['stage'],r['mode']):r['native_variance'] for r in modes if 'native_variance' in r}
    for r in modes:
        if 'within_variance' in r:
            r['native_variance_reference']=native_lookup.get((r['stage'],r['mode']),np.nan)
            r['within_to_native_ratio']=r['within_variance']/max(r['native_variance_reference'],1e-12)
    write_csv(out/'block_mode_variance_comparison.csv', modes)
    k,_=load_kernel_matrix(ROOT/'perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json')
    native_d=[split_pairs(phi,k)['detail'][:,s] for s in range(3)]
    write_csv(out/'block_boundary_bond_metrics.csv', bond_rows(phi,fr_gen,'L32'))
    write_csv(out/'neighboring_block_covariance.csv', neighbor_cov_rows(native_d,fr_d,'L32'))
    # Limited volume transfer confirmation: baseline, the two sector masks that
    # isolate the earliest and latest conditional stages, and selected mode scans.
    phi64=load_phi(ROOT/'data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz')[:20]
    l64=[]
    for control,kwargs in [('all',{}),('q01_only',{'sector_mask':(1,0,0)}),('q11_only',{'sector_mask':(0,0,1)}),('u_0.75',{'mode_scales':(.75,1,1,1)}),('sx_0.75',{'mode_scales':(1,.75,1,1)})]:
        generated, _, _, _ = sample_field(phi64, seed=61, **kwargs); l64.append(metric_row(phi64,generated,'L64',control))
    write_csv(out/'directional_covariance_scan_L64.csv',l64)
    generated64, _, _, d64 = sample_field(phi64, seed=61)
    write_csv(out/'block_boundary_bond_metrics_L64.csv',bond_rows(phi64,generated64,'L64'))
    (out/'directional_width_audit_summary.md').write_text(
        'Frozen-checkpoint diagnostic only. Sector masks use the exact conditional mixture mean for deterministic sectors. '
        'Teacher-forced contexts use native prior details only to measure exposure shift. Native block covariance is a global held-out reference; '\
        'context-matched covariance requires a larger conditional-neighbor sample and is not inferred here.\n')

if __name__ == '__main__': main()
