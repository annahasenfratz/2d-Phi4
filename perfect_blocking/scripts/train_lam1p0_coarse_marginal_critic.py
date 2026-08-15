#!/usr/bin/env python3
"""Held-out coarse-field discriminator: direct L16 versus blocked native L32.

The CNN is translation-equivariant (circular convolutions plus global average
pooling), so its held-out AUC is a compact test for mismatches not visible in
a hand-chosen list of observables.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]; PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
from scripts.common.blocking import load_configs


class Critic(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular"), nn.GELU(),
                                 nn.Conv2d(cout, cout, 3, padding=1, padding_mode="circular"), nn.GELU())
        self.net = nn.Sequential(block(1, channels), block(channels, channels), block(channels, channels))
        self.head = nn.Linear(channels, 1)
    def features(self, x): return self.net(x).mean(dim=(-2, -1))
    def forward(self, x): return self.head(self.features(x)).squeeze(-1)


def block(phi: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    out = np.zeros_like(phi, dtype=np.float32); r = kernel.shape[0] // 2
    for iy, dy in enumerate(range(-r, r + 1)):
        for ix, dx in enumerate(range(-r, r + 1)):
            if kernel[iy, ix]: out += kernel[iy, ix] * np.roll(np.roll(phi, -dy, 1), -dx, 2)
    return out[:, ::2, ::2]


def auc(y: np.ndarray, score: np.ndarray) -> float:
    # Mann--Whitney AUC; class 1 is blocked.
    ranks = rankdata(score); n1 = int(y.sum()); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--direct", type=Path, default=PROJECT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz")
    ap.add_argument("--fine", type=Path, default=PROJECT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz")
    ap.add_argument("--n", type=int, default=5000); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128); ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--seed", type=int, default=20260809)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    kernel = np.asarray(json.loads(args.kernel.read_text())["matrix"], dtype=np.float32)
    direct = load_configs(args.direct)[:args.n].astype(np.float32); blocked = block(load_configs(args.fine)[:args.n].astype(np.float32), kernel)
    assert direct.shape == blocked.shape and len(direct) >= 1000
    n = len(direct); order = rng.permutation(n); split = (int(.70*n), int(.85*n)); train_i, val_i, test_i = order[:split[0]], order[split[0]:split[1]], order[split[1]:]
    mean, std = float(np.mean(direct[train_i])), float(np.std(direct[train_i])); std = max(std, 1e-6)
    x = np.concatenate([direct, blocked])[:, None] / std - mean / std; y = np.r_[np.zeros(n), np.ones(n)].astype(np.float32)
    def ids(ix): return np.r_[ix, ix + n]
    train = TensorDataset(torch.from_numpy(x[ids(train_i)]), torch.from_numpy(y[ids(train_i)]))
    model = Critic(); opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4); lossfn = nn.BCEWithLogitsLoss()
    history=[]; best=(1e9, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in DataLoader(train, batch_size=args.batch_size, shuffle=True):
            opt.zero_grad(); loss=lossfn(model(xb),yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            xv=torch.from_numpy(x[ids(val_i)]); yv=y[ids(val_i)]; logits=model(xv).numpy(); vl=float(lossfn(torch.from_numpy(logits),torch.from_numpy(yv)))
        row={"epoch":epoch,"validation_loss":vl,"validation_auc":auc(yv, logits)}; history.append(row)
        if vl < best[0]: best=(vl,{k:v.detach().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        xt=torch.from_numpy(x[ids(test_i)]); yt=y[ids(test_i)]; score=model(xt).numpy(); feat=model.features(xt).numpy()
    summary={"kernel":str(args.kernel),"n_per_ensemble":n,"split":{"train":len(train_i),"validation":len(val_i),"test":len(test_i)},"standardization":{"mean":mean,"std":std},"test_auc":auc(yt,score),"test_logistic_loss":float(lossfn(torch.from_numpy(score),torch.from_numpy(yt))),"best_validation_loss":best[0],"history":history}
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2)); np.savez_compressed(args.out/"heldout_predictions.npz", labels=yt, logits=score, features=feat, direct_indices=test_i)
    torch.save({"model_state":model.state_dict(),"mean":mean,"std":std,"kernel_path":str(args.kernel),"channels":32},args.out/"critic_best.pt")
    print(json.dumps(summary,indent=2))

if __name__ == "__main__": main()
