"""Dataset helpers for paired fine/coarse/detail training data."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from inverse_blocking_flow.haar import average_block
from inverse_blocking_flow.phi4 import Phi4Params, generate_phi4_configs


def load_or_generate_fine_configs(
    path: Path | None,
    *,
    n_configs: int,
    fine_size: int,
    params: Phi4Params,
    burn_in: int,
    interval: int,
    batch_size: int,
    proposal_width: float,
    seed: int,
    device: str,
) -> torch.Tensor:
    if path is not None and path.exists():
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            data = data["phi"]
        if data.shape[-2:] != (fine_size, fine_size):
            raise ValueError(f"loaded data has shape {data.shape[-2:]}, expected {(fine_size, fine_size)}")
        return data[:n_configs].float()

    configs = generate_phi4_configs(
        n_configs,
        fine_size,
        params,
        burn_in=burn_in,
        interval=interval,
        batch_size=batch_size,
        proposal_width=proposal_width,
        seed=seed,
        device=device,
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"phi": configs, "kappa": params.kappa, "lambda": params.lam}, path)
    return configs.float()


def make_paired_dataset(phi_f: torch.Tensor) -> TensorDataset:
    phi_c, d = average_block(phi_f)
    return TensorDataset(phi_c.unsqueeze(1).float(), d.float(), phi_f.float())
