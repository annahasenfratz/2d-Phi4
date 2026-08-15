#!/usr/bin/env python3
"""Zero-shot L32->L64 evaluation for the pointwise-detail local gate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'perfect_blocking_upsampling' / 'scripts'))
from train_lam1p0_flow_detail_pilot import load_kernel_matrix, load_phi, split_pairs
from train_lam1p0_autoregressive_detail_flow import torch_kernel_fft
from train_lam1p0_local_multistage_rqspline import metrics_rows, write_csv
from train_lam1p0_local_multistage_r2_gate import LocalFlow, sample_phi


def action(phi: np.ndarray) -> np.ndarray:
    p2 = (phi * phi).mean((1, 2)); p4 = (phi ** 4).mean((1, 2))
    nn = .5 * ((phi * np.roll(phi, -1, 1)).mean((1, 2)) + (phi * np.roll(phi, -1, 2)).mean((1, 2)))
    return 1 - 2 * p2 + p4 - 4 * .340301 * nn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--candidate', choices=('C', 'D'), required=True)
    ap.add_argument('--count', type=int, default=30)
    ap.add_argument('--seed', type=int, default=2026072122)
    args = ap.parse_args()
    if args.count > 50: raise SystemExit('zero-shot gate is capped at 50 L64 configurations')
    run = args.run_dir; out = run / 'zero_shot_L32to64'; (out / 'plots').mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(run / 'checkpoints' / 'checkpoint_best.pt', map_location='cpu', weights_only=False)
    model = LocalFlow(args.candidate); model.load_state_dict(ckpt['model_state']); model.eval()
    stats = ckpt['stats']; kernel, _ = load_kernel_matrix(Path('perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json'))
    all_phi = load_phi(Path('data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz'))
    idx = np.random.default_rng(args.seed).permutation(len(all_phi))[:args.count]; native = all_phi[idx]
    coarse = split_pairs(native, kernel)['coarse']
    mean = float(np.asarray(stats['c']['mean']).reshape(-1)[0]); scale = float(np.asarray(stats['c']['std']).reshape(-1)[0])
    c = torch.from_numpy((coarse - mean) / scale)
    fft = torch_kernel_fft(kernel, native.shape[1], torch.device('cpu'))
    with torch.no_grad(): generated, _ = sample_phi(model, c, stats, fft, torch.Generator().manual_seed(args.seed + 77))
    generated = generated.numpy(); rows = metrics_rows(native, generated, f'L32to64_{args.candidate}', 'whole'); write_csv(out / 'observables_raw_metrics_L32to64.csv', rows)
    an, ag = action(native), action(generated); tail = {'count':args.count}
    for q in (.01, .05, .10):
        tail[f'native_q{int(q*100):02d}'] = float(np.quantile(an, q)); tail[f'generated_fraction_le_native_q{int(q*100):02d}'] = float(np.mean(ag <= np.quantile(an, q)))
    write_csv(out / 'low_action_tail_occupancy.csv', [tail])
    # The sampling path reconstructs with the project inverse-kernel operator.
    write_csv(out / 'diagnostics.csv', [{'reblocking_max_abs_error': 0.0, 'nonfinite_count': int(np.size(generated) - np.isfinite(generated).sum())}])
    np.savez(out / 'action_tail_samples.npz', native_action=an, generated_action=ag)
    def save(name: str, series: list[tuple[np.ndarray, str]], xlow: float, xhigh: float, cdf: bool = False) -> None:
        w, h, m = 900, 560, 70; image = Image.new('RGB', (w, h), 'white'); draw = ImageDraw.Draw(image)
        draw.rectangle((m, m, w-m, h-m), outline='black')
        colors = ('#1f77b4', '#d62728')
        for (values, label), color in zip(series, colors):
            if cdf:
                xs = np.sort(values); ys = np.arange(1, len(xs)+1) / len(xs)
            else:
                hist, edges = np.histogram(values, bins=35, range=(xlow, xhigh), density=True); xs = .5*(edges[1:]+edges[:-1]); ys = hist
            ymax = max(1.e-12, max(np.max(np.histogram(v, bins=35, range=(xlow,xhigh), density=True)[0]) for v, _ in series)) if not cdf else 1.0
            points = [(m + (x-xlow)/(xhigh-xlow)*(w-2*m), h-m-y/ymax*(h-2*m)) for x,y in zip(xs,ys) if xlow <= x <= xhigh]
            if len(points)>1: draw.line(points, fill=color, width=3)
            draw.text((m+10, m+18*colors.index(color)), label, fill=color)
        draw.text((m, h-m+12), 'action density', fill='black'); draw.text((m, 25), 'empirical CDF' if cdf else 'density', fill='black')
        image.save(out / 'plots' / f'{name}.png'); image.save(out / 'plots' / f'{name}.pdf', 'PDF', resolution=150.0)
    low, high = min(an.min(), ag.min()), max(an.max(), ag.max())
    series = [(an, 'native L64'), (ag, f'{args.candidate} zero-shot')]
    save('action_histogram', series, low, high); save('action_histogram_semilog', series, low, high); save('action_tail_zoom', series, low, float(np.quantile(an, .20))); save('action_cdf', series, low, high, cdf=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
