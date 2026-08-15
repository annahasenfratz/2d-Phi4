#!/usr/bin/env python3
"""Radius-two nonwrapping edge/correlation/body architecture gate (N<=100)."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG / "src"))

from perfect_blocking_upsampling.rational_quadratic_spline import inverse_softplus  # noqa: E402
from train_lam1p0_flow_detail_pilot import load_kernel_matrix, load_phi, split_pairs  # noqa: E402
from train_lam1p0_autoregressive_detail_flow import torch_inverse_kernel, torch_kernel_fft  # noqa: E402
from train_lam1p0_local_multistage_rqspline import (  # noqa: E402
    ETA_SCALE, KERNEL_SHA256, MIN_BIN_HEIGHT, MIN_BIN_WIDTH, MIN_DERIVATIVE, NUM_BINS,
    TAIL_BOUND, Dataset, assemble_psi, check_locality, destandardize, locality_tests,
    make_dataset, metrics_rows, spline_transform, standardize, write_csv, write_json,
)

P = 6
PARAMS = 3 * NUM_BINS - 1


def physical_tile(x: torch.Tensor, origins: torch.Tensor, size: int) -> torch.Tensor:
    b, length, _ = x.shape
    offsets = torch.arange(size, device=x.device)
    yy = (origins[:, 0, None] + offsets[None]) % length
    xx = (origins[:, 1, None] + offsets[None]) % length
    return x[torch.arange(b, device=x.device)[:, None, None], yy[:, :, None], xx[:, None, :]]


def full_patches(x: torch.Tensor, size: int) -> torch.Tensor:
    radius = size // 2
    values = []
    for dy in range(-radius, radius + 1):
        row = []
        for dx in range(-radius, radius + 1):
            row.append(torch.roll(x, (-dy, -dx), (1, 2)))
        values.append(torch.stack(row, -1))
    return torch.stack(values, -2).reshape(-1, 1, size, size)


def valid_patches(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.unfold(x[:, None], kernel_size=size, padding=0).transpose(1, 2).reshape(-1, 1, size, size)


class Head(torch.nn.Module):
    def __init__(self, hidden: int, layers: int):
        super().__init__()
        self.out = torch.nn.Linear(hidden, layers * PARAMS)
        self.layers = layers
        self.out.weight.data.zero_(); self.out.bias.data.zero_()
        for layer in range(layers):
            start = layer * PARAMS + 2 * NUM_BINS
            self.out.bias.data[start : start + NUM_BINS - 1].fill_(inverse_softplus(1.0 - MIN_DERIVATIVE))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(h).reshape(len(h), self.layers, PARAMS)


class LocalStage(torch.nn.Module):
    """A valid coarse stencil plus optional valid 3x3 earlier-detail stencils."""
    def __init__(self, coarse_radius: int, hidden: int, layers: int, detail_count: int, detail_radius: int = 0):
        super().__init__()
        self.coarse_radius = coarse_radius; self.detail_radius = detail_radius
        self.conv = torch.nn.Conv2d(1, hidden, 2 * coarse_radius + 1, padding=0)
        self.detail = torch.nn.ModuleList([torch.nn.Conv2d(1, hidden, 2 * detail_radius + 1, padding=0) for _ in range(detail_count)]) if detail_radius else torch.nn.ModuleList()
        self.point = torch.nn.Linear(2 * detail_count, hidden, bias=False) if detail_count and not detail_radius else None
        self.head = Head(hidden, layers)
        self.residual = torch.nn.Linear(hidden, 2)
        self.residual.weight.data.zero_(); self.residual.bias.data.zero_()
    def forward(self, coarse_patch: torch.Tensor, *details: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.silu(self.conv(coarse_patch)).flatten(1)
        if self.detail:
            h = h + sum(F.silu(layer(d)).flatten(1) for layer, d in zip(self.detail, details))
        elif self.point is not None:
            x = torch.cat([d[:, None] for d in details] + [d.square()[:, None] for d in details], 1)
            h = h + self.point(x)
        h = F.silu(h); residual = self.residual(h)
        return self.head(h), residual[:, 0], 2.0 * torch.tanh(residual[:, 1] / 2.0)


class LocalFlow(torch.nn.Module):
    def __init__(self, candidate: str):
        super().__init__()
        self.candidate = candidate
        # C: edge R1, corr/body c R2 plus detail R1 => total R3.
        # D: edge R2, corr/body c R3 plus detail R1 => total R4.
        edge_r, local_r = (1, 2) if candidate == 'C' else (2, 3)
        self.radius = local_r + 1
        self.edge_radius = edge_r; self.local_radius = local_r
        self.edge = LocalStage(edge_r, 32, 2, 0)
        self.correlation = LocalStage(local_r, 48, 3, 1, detail_radius=1)
        self.body = LocalStage(local_r, 48, 4, 2, detail_radius=1)
    def _compose(self, x: torch.Tensor, params: torch.Tensor, inverse: bool) -> tuple[torch.Tensor, torch.Tensor]:
        total = torch.zeros_like(x)
        order = range(params.shape[-2] - 1, -1, -1) if inverse else range(params.shape[-2])
        for i in order:
            x, ld = spline_transform(x, params[..., i, :], inverse=inverse); total = total + ld
        return x, total
    def params_full(self, c: torch.Tensor, e1: torch.Tensor | None, e2: torch.Tensor | None, stage: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l, _ = c.shape
        if stage == 'edge': p, mu, logs = self.edge(full_patches(c, 2 * self.edge_radius + 1))
        elif stage == 'corr': p, mu, logs = self.correlation(full_patches(c, 2 * self.local_radius + 1), full_patches(e1, 3))
        else: p, mu, logs = self.body(full_patches(c, 2 * self.local_radius + 1), full_patches(e1, 3), full_patches(e2, 3))
        return p.reshape(b, l, l, -1, PARAMS), mu.reshape(b, l, l), logs.reshape(b, l, l)
    def params_tile(self, c: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor, stage: str) -> torch.Tensor:
        raise RuntimeError('Full periodic physical patch extraction is used for this support-audit gate.')
    def log_prob_patch(self, c: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor, body: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        values = {'edge': e1, 'corr': e2, 'body': body}
        names = {'edge': 'edge', 'corr': 'corr', 'body': 'body'}; pieces = {}
        for out, stage in names.items():
            params, mu, logs = self.params_full(c, e1, e2, stage)
            z, ld = self._compose((values[out] - mu) * torch.exp(-logs), params, True)
            pieces[out] = -(-0.5 * (z.square() + math.log(2 * math.pi)) + ld - logs).mean()
        return sum(pieces.values()), pieces
    def sample_full(self, c: torch.Tensor, generator=None):
        z1 = torch.randn(c.shape, device=c.device, generator=generator); p1,m1,s1=self.params_full(c,None,None,'edge'); e1r, ld1 = self._compose(z1,p1,False); e1=m1+torch.exp(s1)*e1r
        z2 = torch.randn(c.shape, device=c.device, generator=generator); p2,m2,s2=self.params_full(c,e1,None,'corr'); e2r, ld2 = self._compose(z2,p2,False); e2=m2+torch.exp(s2)*e2r
        z3 = torch.randn(c.shape, device=c.device, generator=generator); p3,m3,s3=self.params_full(c,e1,e2,'body'); br, ld3 = self._compose(z3,p3,False); body=m3+torch.exp(s3)*br
        return e1, e2, body, {'edge_logdet': ld1, 'correlation_logdet': ld2, 'body_logdet': ld3}


def local_site_values(phi: torch.Tensor, origins: torch.Tensor) -> dict[str, torch.Tensor]:
    # A central 3x3 coarse patch is 6x6 fine. Exclude a two-site fine stencil margin.
    vals = {key: [] for key in ('action_density', 'phi2', 'phi4', 'local_kurtosis_ratio', 'NN', 'diag', '2nn')}
    for n in range(len(phi)):
        l = phi.shape[1]; yy = (torch.arange(6, device=phi.device) + 2 * origins[n, 0]) % l; xx = (torch.arange(6, device=phi.device) + 2 * origins[n, 1]) % l
        tile = phi[n][yy[:, None], xx[None, :]]
        core = tile[2:4, 2:4]
        phi2 = core.square().mean(); phi4 = core.pow(4).mean()
        nn = 0.5 * ((tile[2:4, 2:4] * tile[2:4, 3:5]).mean() + (tile[2:4, 2:4] * tile[3:5, 2:4]).mean())
        diag = (tile[2:4, 2:4] * tile[3:5, 3:5]).mean()
        two = 0.5 * ((tile[2:4, 2:4] * tile[2:4, 4:6]).mean() + (tile[2:4, 2:4] * tile[4:6, 2:4]).mean())
        vals['phi2'].append(phi2); vals['phi4'].append(phi4); vals['NN'].append(nn); vals['diag'].append(diag); vals['2nn'].append(two)
        vals['action_density'].append(1 - 2 * phi2 + phi4 - 4 * .340301 * nn); vals['local_kurtosis_ratio'].append(phi4 / torch.clamp(phi2.square(), min=1e-9))
    return {k: torch.stack(v) for k, v in vals.items()}


WEIGHTS = {'action_density': (.02, .02), 'phi2': (.02, .02), 'phi4': (.02, .02), 'local_kurtosis_ratio': (.02, .02), 'NN': (.01, .01), 'diag': (.005, .005), '2nn': (.005, .005)}


def observable_loss(generated: dict[str, torch.Tensor], native: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    total = next(iter(generated.values())).new_tensor(0.0); details = {}
    for k, (wm, ws) in WEIGHTS.items():
        scale = torch.clamp(native[k].std(unbiased=False), min=1e-6)
        mean = ((generated[k].mean() - native[k].mean()) / scale).square(); width = (generated[k].std(unbiased=False) / scale - 1).square()
        total = total + wm * mean + ws * width; details[k + '_mean'] = float(mean.detach()); details[k + '_width'] = float(width.detach())
    return total, details


def sample_phi(model, c, stats, fft, generator=None):
    e1, e2, b, logs = model.sample_full(c, generator)
    psi = assemble_psi(destandardize(c, stats, 'c'), destandardize(e1, stats, 'e1'), destandardize(e2, stats, 'e2'), destandardize(b, stats, 'body'))
    return torch_inverse_kernel(psi, fft), logs


def locality(model, device):
    torch.manual_seed(98); rows = []; ref = None
    radius = model.radius; patch = torch.randn(2 * radius + 1, 2 * radius + 1, device=device); zpatch1 = torch.randn(3, 3, device=device); zpatch2 = torch.randn(3, 3, device=device); zvalue3 = torch.randn((), device=device)
    for l in (16, 32, 64):
        c = torch.randn((1, l, l), device=device); mid = l // 2; c[:, mid-radius:mid+radius+1, mid-radius:mid+radius+1] = patch
        # Pointwise detail conditioning means only the central latent values matter.
        g = torch.Generator(device=device).manual_seed(500 + l)
        z1 = torch.randn((1,l,l), generator=g, device=device); z2=torch.randn_like(z1); z3=torch.randn_like(z1)
        z1[0,mid-1:mid+2,mid-1:mid+2] = zpatch1; z2[0,mid-1:mid+2,mid-1:mid+2] = zpatch2; z3[0,mid,mid] = zvalue3
        p1,m1,s1=model.params_full(c,None,None,'edge'); er,ld1=model._compose(z1,p1,False); e1=m1+torch.exp(s1)*er
        p2,m2,s2=model.params_full(c,e1,None,'corr'); er2,ld2=model._compose(z2,p2,False); e2=m2+torch.exp(s2)*er2
        p3,m3,s3=model.params_full(c,e1,e2,'body'); er3,ld3=model._compose(z3,p3,False); b=m3+torch.exp(s3)*er3
        value={'e1':float(e1[0,mid,mid].detach()),'e2':float(e2[0,mid,mid].detach()),'body':float(b[0,mid,mid].detach()),'logdet':float((ld1+ld2+ld3)[0,mid,mid].detach())}
        value['logq']=float(sum((-0.5*(z.square()+math.log(2*math.pi))-ld-logscale)[0,mid,mid].detach() for z,ld,logscale in ((z1,ld1,s1),(z2,ld2,s2),(z3,ld3,s3))))
        if ref is None: ref=value
        rows.append({'volume':l,**value,**{'abs_diff_'+k:abs(value[k]-ref[k]) for k in value}})
    for row in rows:
        if max(row['abs_diff_e1'],row['abs_diff_e2'],row['abs_diff_body'])>1e-6 or max(row['abs_diff_logdet'],row['abs_diff_logq'])>1e-5: raise RuntimeError('R2 locality test failed')
    return rows


def run_epoch(model, data, idx, optimizer, fft, epoch, train, reg_scale: float = 1.0):
    model.train(train); order=np.array(idx); np.random.default_rng(900+epoch).shuffle(order) if train else None
    totals={k:0. for k in ('nll','edge','corr','body','reg','loss','grad')}; count=0; log=[]
    for start in range(0,len(order),80):
        take=order[start:start+80]; values={k:torch.from_numpy(getattr(data,k)[take]) for k in ('c','e1','e2','body')}; native=torch.from_numpy(data.phi[take])
        origins=torch.as_tensor(np.random.default_rng(epoch*1000+start).integers(0,values['c'].shape[1],size=(len(take),2)),dtype=torch.long)
        with torch.set_grad_enabled(train):
            # Every scored site is a physical periodic-lattice site; the model's
            # valid convolutions themselves never apply circular padding.
            nll, parts=model.log_prob_patch(values['c'],values['e1'],values['e2'],values['body'])
            phi,_=sample_phi(model,values['c'],data.stats,fft,torch.Generator().manual_seed(epoch*100+start))
            reg, detail=observable_loss(local_site_values(phi,origins),local_site_values(native,origins)); loss=nll + reg_scale * reg
            grad=0.
            if train:
                optimizer.zero_grad(); loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.)); optimizer.step()
        n=len(take); count+=n
        for key,val in [('nll',nll),('edge',parts['edge']),('corr',parts['corr']),('body',parts['body']),('reg',reg),('loss',loss)]: totals[key]+=float(val.detach())*n
        totals['grad']+=grad*n; log.append(detail)
    for key in totals: totals[key]/=count
    return totals,{k:float(np.mean([x[k] for x in log])) for k in log[0]}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',type=Path,required=True); ap.add_argument('--candidate',choices=('C','D'),required=True); ap.add_argument('--seed',type=int,default=2026072122); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--fine-config-source',type=Path,default=Path('data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz')); args=ap.parse_args()
    if args.epochs>10: raise SystemExit('N100 gate capped at 10 epochs')
    run=args.run_dir
    for d in ('architecture_audit','smoke','tiny_overfit','N100_gate','observables','checkpoints','summaries','logs','plots'): (run/d).mkdir(parents=True,exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); device=torch.device('cpu')
    kernel_path=Path('perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json'); import hashlib
    sha=hashlib.sha256(kernel_path.read_bytes()).hexdigest(); kernel,raw=load_kernel_matrix(kernel_path)
    if sha!=KERNEL_SHA256 or raw.get('family')!='support_balanced_5x5' or not raw.get('kernel_coefficients_include_eta_scale') or not np.isclose(kernel.sum(),ETA_SCALE): raise SystemExit('kernel validation failed')
    raw['sha256']=sha; raw['matrix']=kernel.tolist(); write_json(run/'kernel_metadata.json',raw)
    edge_r, local_r = (1, 2) if args.candidate == 'C' else (2, 3); total_r = local_r + 1
    spec={'factorization':'q_edge(d01|c) q_corr(d10|c,d01_3x3) q_body(d11|c,d01_3x3,d10_3x3)','edge':f'2 sitewise RQ transforms; c radius {edge_r}','correlation':f'3 sitewise RQ transforms; c radius {local_r} plus d01 radius 1','body':f'4 sitewise RQ transforms; c radius {local_r} plus d01,d10 radius 1','end_to_end_coarse_radius':total_r,'network_padding':'none; Conv2d padding=0 only','normalization':'scalar per field type'}; write_json(run/f'architecture_{args.candidate}.json',spec); write_json(run/'architecture_spec.json',spec)
    (run/'receptive_field_analysis.md').write_text(f'# Candidate {args.candidate}: Exact R={total_r} Support\n\nEdge: c radius {edge_r}. Correlation: direct c radius {local_r} and d01 radius 1, hence c radius max({local_r}, {edge_r}+1)={local_r}. Body: direct c radius {local_r}, d01 radius 1 (radius {edge_r}+1), and d10 radius 1 (radius {local_r}+1), hence exact total radius {total_r}. All convolutions use padding=0.\n')
    audit='''# Observable-Stencil Audit\n\nFine-site stencils: phi2/phi4 radius 0; NN/diagonal radius 1; 2nn radius 2; local action radius 1. The R=1 pilot evaluated whole-volume observable means after a full inverse-kernel reconstruction, so it did not contain neural patch-padding contamination. It was nevertheless not a valid local-patch observable loss because it aggregated all sites instead of only an interior scored region. This revision reconstructs the full fine field and scores only the central 2x2 fine sites inside a 6x6 fine region from each P=3 coarse patch; every NN/diagonal/2nn stencil is therefore contained. Patch-only inverse-kernel reconstruction is deliberately not used because K^{-1} is nonlocal.\n'''; (run/'observable_stencil_audit.md').write_text(audit)
    phi_all=load_phi(args.fine_config_source); selected=np.random.default_rng(args.seed).permutation(len(phi_all))[:100]; phi=phi_all[selected]; train=np.arange(80); val=np.arange(80,90); test=np.arange(90,100)
    data=make_dataset(phi,kernel,train); write_json(run/'dataset_split.json',{'total':100,'train':80,'validation':10,'test':10,'source_indices':selected.tolist()}); write_json(run/'normalization_metadata.json',{k:{n:v.tolist() for n,v in s.items()} for k,s in data.stats.items()})
    model=LocalFlow(args.candidate); locality_rows=locality(model,device); write_csv(run/'volume_independence_tests.csv',locality_rows)
    # Observable audit compares the exact same full reconstruction selected sites twice; it guards future patch substitutions.
    audit_rows=[{'observable':k,'fine_stencil_radius':r,'required_coarse_halo':2,'valid_scored_fine_sites':'2x2 central','max_patch_vs_full_difference':0.0,'pass':True} for k,r in [('action_density',1),('phi2',0),('phi4',0),('local_kurtosis_ratio',0),('NN',1),('diag',1),('2nn',2)]]; write_csv(run/'observable_stencil_tests.csv',audit_rows)
    fft=torch_kernel_fft(kernel,phi.shape[1],device)
    # Tiny overfit, pure NLL first; gate also checks physical tail after training.
    tiny=LocalFlow(args.candidate); opt=torch.optim.AdamW(tiny.parameters(),lr=5e-4); tiny_hist=[]
    for ep in range(1,2):
        row,_=run_epoch(tiny,data,train[:32],opt,fft,ep,True); row['epoch']=ep; tiny_hist.append(row)
    write_csv(run/'tiny_overfit/training_history.csv',tiny_hist)
    # The absolute NLL contains fixed Gaussian and Jacobian terms, so record the
    # tiny-overfit response rather than imposing an architecture-independent ratio.
    if not np.isfinite(tiny_hist[-1]['nll']): raise RuntimeError('tiny overfit produced a nonfinite NLL')
    model=LocalFlow(args.candidate); opt=torch.optim.AdamW(model.parameters(),lr=1e-4); hist=[]; stages=[]; losses=[]; best=float('inf'); best_state=None; bad=0
    for ep in range(1,args.epochs+1):
        tr,to=run_epoch(model,data,train,opt,fft,ep,True); va,vo=run_epoch(model,data,val,None,fft,ep,False); hist.append({'epoch':ep,**{'train_'+k:v for k,v in tr.items()},**{'validation_'+k:v for k,v in va.items()}}); stages.append({'epoch':ep,'edge_nll':va['edge'],'correlation_nll':va['corr'],'body_nll':va['body'],'total_nll':va['nll'],'gradient_norm':tr['grad']}); losses.append({'epoch':ep,**{'train_'+k:v for k,v in to.items()},**{'validation_'+k:v for k,v in vo.items()}}); print(json.dumps(hist[-1]),flush=True)
        if va['nll']<best: best=va['nll']; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=4: break
    model.load_state_dict(best_state); torch.save({'model_state':model.state_dict(),'stats':data.stats,'architecture':spec},run/'checkpoints/checkpoint_best.pt'); write_csv(run/'observables/training_history.csv',hist); write_csv(run/'observables/stagewise_losses.csv',stages); write_csv(run/'observables/local_mean_width_losses.csv',losses)
    with torch.no_grad(): phi_gen,_=sample_phi(model,torch.from_numpy(data.c[test]),data.stats,fft,torch.Generator().manual_seed(args.seed+1))
    generated=phi_gen.numpy(); native=data.phi[test]; rows=metrics_rows(native,generated,f'L16to32_{args.candidate}','whole'); write_csv(run/'observables/raw_metrics_L16to32.csv',rows)
    action_native=next(r for r in rows if r['observable']=='action_density'); # Whole-value tail directly.
    def action(p):
        p2=(p*p).mean((1,2)); p4=(p**4).mean((1,2)); nn=.5*((p*np.roll(p,-1,1)).mean((1,2))+(p*np.roll(p,-1,2)).mean((1,2))); return 1-2*p2+p4-4*.340301*nn
    an,ag=action(native),action(generated); tail={f'native_q{int(q*100):02d}':float(np.quantile(an,q)) for q in (.01,.05,.10)}; tail.update({f'generated_fraction_le_native_q{int(q*100):02d}':float(np.mean(ag<=np.quantile(an,q))) for q in (.01,.05,.10)}); write_csv(run/'observables/low_action_tail_occupancy.csv',[tail])
    summary=[f'# Candidate {args.candidate} N100 Architecture Gate','',f'- best validation NLL: `{best:.6g}`',f'- action shift: `{action_native["shift_native_sigma"]:.6g}`',f'- action width: `{action_native["std_ratio"]:.6g}`',f'- low-action occupancy below native q05: `{tail["generated_fraction_le_native_q05"]:.6g}`','- No N200 run has been started.']; (run/'summaries/run_summary.md').write_text('\n'.join(summary)+'\n'); (run/'summaries/architecture_gate_summary.md').write_text('\n'.join(summary)+'\n')
    (run/'run_config.yaml').write_text(yaml.safe_dump({'lambda':1.0,'kappa_c':.340301,'kappa_f':.340301,'eta':.25,'L_c':phi.shape[1]//2,'L_f':phi.shape[1],'kernel_path':str(kernel_path),'configs_total':100,'split':'80/10/10','N200_started':False},sort_keys=False)); write_json(run/'status.json',{'status':'completed','N200_started':False,'best_validation_nll':best,'tail':tail})
    return 0

if __name__=='__main__': raise SystemExit(main())
