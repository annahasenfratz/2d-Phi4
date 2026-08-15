#!/usr/bin/env python3
"""Local, volume-scalable L8->L16 conditional inverse-blocking flow pilot.

The neural networks are intentionally shallow valid-convolution modules.  Periodic
boundary conditions are used only to extract physical lattice patches; no neural
module performs circular padding.  Every generated detail site depends on a
radius-one coarse neighbourhood, including through the edge/correlation/body
factorization, so the end-to-end coarse receptive radius is one.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from scipy.stats import ks_2samp, wasserstein_distance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "scripts"))

from perfect_blocking_upsampling.rational_quadratic_spline import (  # noqa: E402
    inverse_softplus,
    unconstrained_rational_quadratic_spline,
)
from perfect_blocking_upsampling.io import ActionSpec  # noqa: E402
from train_lam1p0_flow_detail_pilot import (  # noqa: E402
    apply_kernel,
    inverse_kernel,
    load_kernel_matrix,
    load_phi,
    split_pairs,
)
from train_lam1p0_autoregressive_detail_flow import (  # noqa: E402
    torch_inverse_kernel,
    torch_kernel_fft,
)


KERNEL_SHA256 = "84013adc2235f9f89a12bc835c2f1cbe6185f1d50d5b1b86dc05d649990baace"
ETA_SCALE = 1.0905077326652577
RADIUS = 1
PATCH_SIZE = 4
HALO = 2
NUM_BINS = 8
TAIL_BOUND = 6.0
MIN_BIN_WIDTH = 1.0e-3
MIN_BIN_HEIGHT = 1.0e-3
MIN_DERIVATIVE = 1.0e-3


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def standardize(train: np.ndarray, full: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # Scalar field-type normalization is necessary for the same local map to run
    # on L8, L16, and L32 coarse lattices.  Position-dependent arrays would
    # themselves encode the training volume.
    mean = np.asarray(train.mean(), dtype=np.float32).reshape(1, 1, 1)
    std = np.asarray(max(float(train.std()), 1.0e-6), dtype=np.float32).reshape(1, 1, 1)
    return ((full - mean) / std).astype(np.float32), {"mean": mean, "std": std}


def physical_tile(field: torch.Tensor, origins: torch.Tensor, size: int) -> torch.Tensor:
    """Extract periodic physical tiles.  This is lattice extraction, not NN padding."""
    batch, length, _ = field.shape
    offsets = torch.arange(size, device=field.device)
    yy = (origins[:, 0, None] + offsets[None, :]) % length
    xx = (origins[:, 1, None] + offsets[None, :]) % length
    b = torch.arange(batch, device=field.device)[:, None, None]
    return field[b, yy[:, :, None], xx[:, None, :]]


def periodic_site_patches(field: torch.Tensor) -> torch.Tensor:
    """Return one physical 3x3 neighbourhood for every lattice site."""
    values = []
    for dy in (-1, 0, 1):
        row = []
        for dx in (-1, 0, 1):
            row.append(torch.roll(field, shifts=(-dy, -dx), dims=(1, 2)))
        values.append(torch.stack(row, dim=-1))
    patches = torch.stack(values, dim=-2)  # B, L, L, 3, 3
    return patches.reshape(-1, 1, 3, 3)


def valid_site_patches(tile: torch.Tensor) -> torch.Tensor:
    """Extract valid 3x3 patches without padding from a patch-plus-halo tile."""
    batch, height, width = tile.shape
    patches = F.unfold(tile[:, None], kernel_size=3, padding=0).transpose(1, 2)
    return patches.reshape(batch * (height - 2) * (width - 2), 1, 3, 3)


def center_from_tile(field: torch.Tensor, central_start: int, patch: int) -> torch.Tensor:
    return field[:, central_start : central_start + patch, central_start : central_start + patch]


def spline_transform(x: torch.Tensor, params: torch.Tensor, *, inverse: bool) -> tuple[torch.Tensor, torch.Tensor]:
    widths = params[..., :NUM_BINS]
    heights = params[..., NUM_BINS : 2 * NUM_BINS]
    derivatives = params[..., 2 * NUM_BINS :]
    return unconstrained_rational_quadratic_spline(
        x,
        widths,
        heights,
        derivatives,
        inverse=inverse,
        tail_bound=TAIL_BOUND,
        min_bin_width=MIN_BIN_WIDTH,
        min_bin_height=MIN_BIN_HEIGHT,
        min_derivative=MIN_DERIVATIVE,
    )


class EdgeStage(torch.nn.Module):
    """Directional coarse interpolation: 1x3 then 3x1 valid convolutions, R=1."""

    def __init__(self, hidden: int):
        super().__init__()
        self.horizontal = torch.nn.Conv2d(1, hidden, kernel_size=(1, 3), padding=0)
        self.vertical = torch.nn.Conv2d(hidden, hidden, kernel_size=(3, 1), padding=0)
        self.head = torch.nn.Conv2d(hidden, 3 * NUM_BINS - 1, kernel_size=1, padding=0)
        self._identity_head()

    def _identity_head(self) -> None:
        self.head.weight.data.zero_()
        self.head.bias.data.zero_()
        self.head.bias.data[2 * NUM_BINS :].fill_(inverse_softplus(1.0 - MIN_DERIVATIVE))

    def forward(self, c_patch: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.horizontal(c_patch))
        h = F.silu(self.vertical(h))
        return self.head(h).flatten(1)


class CorrelationStage(torch.nn.Module):
    """Cross-edge stage: local coarse 3x3 feature plus pointwise first-edge feature, R=1."""

    def __init__(self, hidden: int):
        super().__init__()
        self.coarse = torch.nn.Conv2d(1, hidden, kernel_size=3, padding=0)
        self.edge_point = torch.nn.Linear(1, hidden, bias=False)
        self.head = torch.nn.Linear(hidden, 3 * NUM_BINS - 1)
        self._identity_head()

    def _identity_head(self) -> None:
        self.head.weight.data.zero_()
        self.head.bias.data.zero_()
        self.head.bias.data[2 * NUM_BINS :].fill_(inverse_softplus(1.0 - MIN_DERIVATIVE))

    def forward(self, c_patch: torch.Tensor, edge_center: torch.Tensor) -> torch.Tensor:
        h = self.coarse(c_patch).flatten(1)
        h = F.silu(h + self.edge_point(edge_center[:, None]))
        return self.head(h)


class BodyStage(torch.nn.Module):
    """Radial/body stage: coarse 3x3 feature plus pointwise amplitude and edge-product features, R=1."""

    def __init__(self, hidden: int):
        super().__init__()
        self.coarse = torch.nn.Conv2d(1, hidden, kernel_size=3, padding=0)
        self.radial_point = torch.nn.Linear(6, hidden, bias=False)
        self.head = torch.nn.Linear(hidden, 3 * NUM_BINS - 1)
        self._identity_head()

    def _identity_head(self) -> None:
        self.head.weight.data.zero_()
        self.head.bias.data.zero_()
        self.head.bias.data[2 * NUM_BINS :].fill_(inverse_softplus(1.0 - MIN_DERIVATIVE))

    def forward(self, c_patch: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        c0 = c_patch[:, :, 1, 1].flatten(1)
        features = torch.cat([c0, e1[:, None], e2[:, None], c0.square(), e1[:, None].square(), e2[:, None].square()], dim=1)
        h = self.coarse(c_patch).flatten(1)
        h = F.silu(h + self.radial_point(features))
        return self.head(h)


class LocalMultistageFlow(torch.nn.Module):
    """Strictly local edge/correlation/body RQ-spline conditional flow."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.edge = EdgeStage(hidden)
        self.correlation = CorrelationStage(hidden)
        self.body = BodyStage(hidden)

    @staticmethod
    def receptive_field() -> dict[str, Any]:
        return {
            "edge_stage_coarse_radius": 1,
            "correlation_stage_direct_coarse_radius": 1,
            "body_stage_direct_coarse_radius": 1,
            "end_to_end_coarse_radius": 1,
            "network_padding": "none (all convolution padding=0)",
            "physical_periodicity": "used only by patch extraction on complete lattice fields",
        }

    def _params_full(self, c: torch.Tensor, e1: torch.Tensor | None = None, e2: torch.Tensor | None = None, stage: str = "edge") -> torch.Tensor:
        cpatch = periodic_site_patches(c)
        batch, length, _ = c.shape
        if stage == "edge":
            params = self.edge(cpatch)
        elif stage == "correlation":
            assert e1 is not None
            params = self.correlation(cpatch, e1.reshape(-1))
        else:
            assert e1 is not None and e2 is not None
            params = self.body(cpatch, e1.reshape(-1), e2.reshape(-1))
        return params.reshape(batch, length, length, -1)

    def _params_tile(self, c_tile: torch.Tensor, e1_tile: torch.Tensor | None, e2_tile: torch.Tensor | None, patch: int, halo: int, stage: str) -> torch.Tensor:
        cpatch = valid_site_patches(c_tile)
        valid = c_tile.shape[1] - 2
        start = halo - RADIUS
        if stage == "edge":
            params = self.edge(cpatch).reshape(c_tile.shape[0], valid, valid, -1)
        elif stage == "correlation":
            assert e1_tile is not None
            e1_valid = center_from_tile(e1_tile, RADIUS, valid).reshape(-1)
            params = self.correlation(cpatch, e1_valid).reshape(c_tile.shape[0], valid, valid, -1)
        else:
            assert e1_tile is not None and e2_tile is not None
            e1_valid = center_from_tile(e1_tile, RADIUS, valid).reshape(-1)
            e2_valid = center_from_tile(e2_tile, RADIUS, valid).reshape(-1)
            params = self.body(cpatch, e1_valid, e2_valid).reshape(c_tile.shape[0], valid, valid, -1)
        return params[:, start : start + patch, start : start + patch]

    def log_prob_patch(self, c_tile: torch.Tensor, e1_tile: torch.Tensor, e2_tile: torch.Tensor, body_tile: torch.Tensor, patch: int = PATCH_SIZE, halo: int = HALO) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        start = halo
        values = {
            "edge": center_from_tile(e1_tile, start, patch),
            "correlation": center_from_tile(e2_tile, start, patch),
            "body": center_from_tile(body_tile, start, patch),
        }
        params = {
            "edge": self._params_tile(c_tile, None, None, patch, halo, "edge"),
            "correlation": self._params_tile(c_tile, e1_tile, None, patch, halo, "correlation"),
            "body": self._params_tile(c_tile, e1_tile, e2_tile, patch, halo, "body"),
        }
        out: dict[str, torch.Tensor] = {}
        for name in ("edge", "correlation", "body"):
            z, logdet = spline_transform(values[name], params[name], inverse=True)
            base = -0.5 * (z.square() + math.log(2.0 * math.pi))
            out[name] = -(base + logdet).mean()
        return out["edge"] + out["correlation"] + out["body"], out

    def sample_full(self, c: torch.Tensor, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        z1 = torch.randn(c.shape, device=c.device, dtype=c.dtype, generator=generator)
        p1 = self._params_full(c, stage="edge")
        e1, ld1 = spline_transform(z1, p1, inverse=False)
        z2 = torch.randn(c.shape, device=c.device, dtype=c.dtype, generator=generator)
        p2 = self._params_full(c, e1, stage="correlation")
        e2, ld2 = spline_transform(z2, p2, inverse=False)
        z3 = torch.randn(c.shape, device=c.device, dtype=c.dtype, generator=generator)
        p3 = self._params_full(c, e1, e2, stage="body")
        body, ld3 = spline_transform(z3, p3, inverse=False)
        logs = {
            "edge": (-0.5 * (z1.square() + math.log(2.0 * math.pi)) - ld1),
            "correlation": (-0.5 * (z2.square() + math.log(2.0 * math.pi)) - ld2),
            "body": (-0.5 * (z3.square() + math.log(2.0 * math.pi)) - ld3),
            "edge_logdet": ld1,
            "correlation_logdet": ld2,
            "body_logdet": ld3,
        }
        return e1, e2, body, logs

    def log_prob_full(self, c: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor, body: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        params = {
            "edge": self._params_full(c, stage="edge"),
            "correlation": self._params_full(c, e1, stage="correlation"),
            "body": self._params_full(c, e1, e2, stage="body"),
        }
        values = {"edge": e1, "correlation": e2, "body": body}
        logs: dict[str, torch.Tensor] = {}
        total = c.new_zeros(c.shape)
        for name in ("edge", "correlation", "body"):
            z, ld = spline_transform(values[name], params[name], inverse=True)
            logs[name] = -0.5 * (z.square() + math.log(2.0 * math.pi)) + ld
            logs[f"{name}_logdet"] = ld
            total = total + logs[name]
        return total, logs


def assemble_psi(c: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
    batch, length, _ = c.shape
    psi = torch.empty((batch, 2 * length, 2 * length), dtype=c.dtype, device=c.device)
    psi[:, 0::2, 0::2] = c
    psi[:, 0::2, 1::2] = e1
    psi[:, 1::2, 0::2] = e2
    psi[:, 1::2, 1::2] = body
    return psi


def local_observables(phi: torch.Tensor) -> dict[str, torch.Tensor]:
    phi2 = phi.square().mean(dim=(1, 2))
    phi4 = phi.pow(4).mean(dim=(1, 2))
    nn = 0.5 * ((phi * torch.roll(phi, -1, dims=1)).mean(dim=(1, 2)) + (phi * torch.roll(phi, -1, dims=2)).mean(dim=(1, 2)))
    two = 0.5 * ((phi * torch.roll(phi, -2, dims=1)).mean(dim=(1, 2)) + (phi * torch.roll(phi, -2, dims=2)).mean(dim=(1, 2)))
    diag = (phi * torch.roll(torch.roll(phi, -1, dims=1), -1, dims=2)).mean(dim=(1, 2))
    return {
        "action_density": 1.0 - 2.0 * phi2 + phi4 - 4.0 * 0.340301 * nn,
        "phi2": phi2,
        "phi4": phi4,
        "local_kurtosis_ratio": phi4 / torch.clamp(phi2.square(), min=1.0e-12),
        "NN": nn,
        "diag": diag,
        "2nn": two,
    }


OBS_WEIGHTS = {
    "action_density": (0.03, 0.03), "phi2": (0.03, 0.03), "phi4": (0.03, 0.03),
    "local_kurtosis_ratio": (0.03, 0.03), "NN": (0.015, 0.015), "diag": (0.005, 0.005), "2nn": (0.005, 0.005),
}


def mean_width_loss(generated: dict[str, torch.Tensor], native: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    total = next(iter(generated.values())).new_tensor(0.0)
    details: dict[str, float] = {}
    for key, (wmean, wwidth) in OBS_WEIGHTS.items():
        ns = torch.clamp(native[key].std(unbiased=False), min=1.0e-6)
        mean = ((generated[key].mean() - native[key].mean()) / ns).square()
        width = (generated[key].std(unbiased=False) / ns - 1.0).square()
        total = total + wmean * mean + wwidth * width
        details[f"{key}_mean"] = float(mean.detach().cpu())
        details[f"{key}_width"] = float(width.detach().cpu())
    return total, details


@dataclass
class Dataset:
    c: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    body: np.ndarray
    phi: np.ndarray
    stats: dict[str, dict[str, np.ndarray]]


def make_dataset(phi: np.ndarray, kernel: np.ndarray, train_idx: np.ndarray) -> Dataset:
    pairs = split_pairs(phi, kernel)
    values = {"c": pairs["coarse"], "e1": pairs["detail"][:, 0], "e2": pairs["detail"][:, 1], "body": pairs["detail"][:, 2]}
    out: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, np.ndarray]] = {}
    for key, value in values.items():
        normalized, item_stats = standardize(value[train_idx], value)
        out[key] = normalized
        stats[key] = item_stats
    return Dataset(c=out["c"], e1=out["e1"], e2=out["e2"], body=out["body"], phi=phi.astype(np.float32), stats=stats)


def destandardize(x: torch.Tensor, stats: dict[str, np.ndarray], key: str) -> torch.Tensor:
    mean = torch.as_tensor(stats[key]["mean"], dtype=x.dtype, device=x.device)
    std = torch.as_tensor(stats[key]["std"], dtype=x.dtype, device=x.device)
    return x * std + mean


def batch_tiles(batch: dict[str, torch.Tensor], origins: torch.Tensor) -> dict[str, torch.Tensor]:
    size = PATCH_SIZE + 2 * HALO
    return {key: physical_tile(value, origins, size) for key, value in batch.items() if key in {"c", "e1", "e2", "body"}}


def model_physical_sample(model: LocalMultistageFlow, c_norm: torch.Tensor, stats: dict[str, dict[str, np.ndarray]], kernel_fft: torch.Tensor, generator: torch.Generator | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
    e1, e2, body, logs = model.sample_full(c_norm, generator)
    psi = assemble_psi(destandardize(c_norm, stats, "c"), destandardize(e1, stats, "e1"), destandardize(e2, stats, "e2"), destandardize(body, stats, "body"))
    return torch_inverse_kernel(psi, kernel_fft), (e1, e2, body), logs


def metrics_rows(native: np.ndarray, generated: np.ndarray, label: str, scope: str = "whole") -> list[dict[str, Any]]:
    if scope == "whole":
        a = {k: v.detach().cpu().numpy() for k, v in local_observables(torch.from_numpy(native)).items()}
        b = {k: v.detach().cpu().numpy() for k, v in local_observables(torch.from_numpy(generated)).items()}
        a.update({"m2": native.mean(axis=(1, 2)) ** 2, "m4": native.mean(axis=(1, 2)) ** 4})
        b.update({"m2": generated.mean(axis=(1, 2)) ** 2, "m4": generated.mean(axis=(1, 2)) ** 4})
        # Lowest nonzero momentum component is validation-only.
        for values, phi in ((a, native), (b, generated)):
            ft = np.fft.fft2(phi, axes=(1, 2))
            values["G_pmin_avg"] = 0.5 * (np.abs(ft[:, 1, 0]) ** 2 + np.abs(ft[:, 0, 1]) ** 2) / (phi.shape[1] ** 2)
    else:
        a = {"phi2": native.reshape(-1) ** 2, "phi4": native.reshape(-1) ** 4}
        b = {"phi2": generated.reshape(-1) ** 2, "phi4": generated.reshape(-1) ** 4}
        for values, phi in ((a, native), (b, generated)):
            nn = 0.5 * (phi * np.roll(phi, -1, axis=1) + phi * np.roll(phi, -1, axis=2))
            values["NN"] = nn.reshape(-1)
            values["diag"] = (phi * np.roll(np.roll(phi, -1, axis=1), -1, axis=2)).reshape(-1)
            values["2nn"] = (0.5 * (phi * np.roll(phi, -2, axis=1) + phi * np.roll(phi, -2, axis=2))).reshape(-1)
            values["action_density"] = (1.0 - phi**2 + 0.5 * phi**4 - 2.0 * 0.340301 * (phi * np.roll(phi, -1, axis=1) + phi * np.roll(phi, -1, axis=2))).reshape(-1)
            values["local_kurtosis_ratio"] = (phi**4 / np.maximum(phi**2, 1.0e-8) ** 2).reshape(-1)
    rows = []
    for key in a:
        std = max(float(np.std(a[key], ddof=1)), 1.0e-12)
        lo = min(float(np.min(a[key])), float(np.min(b[key])))
        hi = max(float(np.max(a[key])), float(np.max(b[key])))
        bins = np.linspace(lo, hi, 81) if hi > lo else np.array([lo - 1.0, hi + 1.0])
        ha, _ = np.histogram(a[key], bins=bins, density=True)
        hb, _ = np.histogram(b[key], bins=bins, density=True)
        width = bins[1] - bins[0]
        rows.append({"label": label, "scope": scope, "observable": key, "native_mean": float(np.mean(a[key])), "generated_mean": float(np.mean(b[key])), "shift_native_sigma": float((np.mean(b[key]) - np.mean(a[key])) / std), "std_ratio": float(np.std(b[key], ddof=1) / std), "KS": float(ks_2samp(a[key], b[key]).statistic), "overlap": float(np.sum(np.minimum(ha, hb)) * width), "W1": float(wasserstein_distance(a[key], b[key]))})
    return rows


def action_tail_plot(native: np.ndarray, generated: np.ndarray, path: Path, title: str) -> None:
    a = local_observables(torch.from_numpy(native))["action_density"].numpy()
    b = local_observables(torch.from_numpy(generated))["action_density"].numpy()
    # Pillow avoids the host's broken Matplotlib PDF backend while producing both
    # requested raster PNG and portable PDF figures.
    width, height = 2400, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((35, 20), title, fill="black")
    panels = [(35, 75, 775, 655, "linear"), (830, 75, 1570, 655, "semilog-y"), (1625, 75, 2365, 655, "low-action tail")]
    def plot_hist(box, x, y, xbins, log_y, label):
        left, top, right, bottom = box
        hx, _ = np.histogram(x, bins=xbins, density=True)
        hy, _ = np.histogram(y, bins=xbins, density=True)
        ymax = max(float(hx.max()), float(hy.max()), 1.0e-12)
        if log_y:
            ymin = max(min(float(v) for v in np.concatenate([hx[hx > 0], hy[hy > 0]]) if v > 0), 1.0e-8)
            transform = lambda v: (math.log(max(v, ymin)) - math.log(ymin)) / max(math.log(ymax) - math.log(ymin), 1.0e-12)
        else:
            transform = lambda v: v / ymax
        draw.rectangle(box, outline="black", width=2)
        for hist, color in ((hx, (30, 80, 180)), (hy, (200, 65, 60))):
            pts = []
            for i, value in enumerate(hist):
                px = left + (right - left) * i / max(len(hist) - 1, 1)
                py = bottom - (bottom - top) * transform(float(value))
                pts.append((px, py))
            if len(pts) > 1:
                draw.line(pts, fill=color, width=3)
        draw.text((left + 8, top + 8), label, fill="black")
        draw.line((left + 16, top + 32, left + 48, top + 32), fill=(30, 80, 180), width=3); draw.text((left + 54, top + 23), "native", fill=(30, 80, 180))
        draw.line((left + 16, top + 54, left + 48, top + 54), fill=(200, 65, 60), width=3); draw.text((left + 54, top + 45), "local flow", fill=(200, 65, 60))
    bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 70)
    plot_hist(panels[0][:4], a, b, bins, False, panels[0][4])
    plot_hist(panels[1][:4], a, b, bins, True, panels[1][4])
    q = np.quantile(a, 0.10)
    low_a, low_b = a[a <= q], b[b <= q]
    low_floor = min(float(low_a.min()), float(low_b.min())) if len(low_b) else float(low_a.min())
    low_bins = np.linspace(low_floor, q, 40)
    plot_hist(panels[2][:4], low_a, low_b, low_bins, False, panels[2][4])
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path.with_suffix(".png"))
    image.save(path.with_suffix(".pdf"), "PDF", resolution=200.0)


def locality_tests(model: LocalMultistageFlow, device: torch.device) -> list[dict[str, Any]]:
    torch.manual_seed(111)
    results = []
    shared_c = torch.randn(3, 3, device=device)
    shared_z = [torch.randn((), device=device) for _ in range(3)]
    reference: dict[str, float] | None = None
    for length in (8, 16, 32, 64):
        c = torch.randn((1, length, length), device=device)
        center = length // 2
        c[:, center - 1 : center + 2, center - 1 : center + 2] = shared_c
        gen = torch.Generator(device=device).manual_seed(9100 + length)
        # Make the three central latent variables identical across volumes.
        z1 = torch.randn((1, length, length), generator=gen, device=device)
        z2 = torch.randn((1, length, length), generator=gen, device=device)
        z3 = torch.randn((1, length, length), generator=gen, device=device)
        z1[0, center, center], z2[0, center, center], z3[0, center, center] = shared_z
        p1 = model._params_full(c, stage="edge")
        e1, ld1 = spline_transform(z1, p1, inverse=False)
        p2 = model._params_full(c, e1, stage="correlation")
        e2, ld2 = spline_transform(z2, p2, inverse=False)
        p3 = model._params_full(c, e1, e2, stage="body")
        body, ld3 = spline_transform(z3, p3, inverse=False)
        values = {"e1": float(e1[0, center, center]), "e2": float(e2[0, center, center]), "body": float(body[0, center, center]), "logdet": float((ld1 + ld2 + ld3)[0, center, center])}
        local_logq = sum(float((-0.5 * (z.square() + math.log(2.0 * math.pi)) - ld)[0, center, center]) for z, ld in ((z1, ld1), (z2, ld2), (z3, ld3)))
        values["logq"] = local_logq
        if reference is None:
            reference = values
        row = {"volume": length, **values}
        row.update({f"abs_diff_{key}": abs(values[key] - reference[key]) for key in values})
        results.append(row)
    # External-field perturbation outside R=1 must leave central values unchanged.
    length = 16
    center = length // 2
    c = torch.randn((1, length, length), device=device)
    c_alt = c.clone()
    c_alt[:, 0, 0] += 100.0
    gen = torch.Generator(device=device).manual_seed(773)
    z = torch.randn((1, length, length), generator=gen, device=device)
    def central(field: torch.Tensor) -> tuple[float, float]:
        p1 = model._params_full(field, stage="edge"); e1, ld1 = spline_transform(z, p1, inverse=False)
        p2 = model._params_full(field, e1, stage="correlation"); e2, ld2 = spline_transform(z, p2, inverse=False)
        p3 = model._params_full(field, e1, e2, stage="body"); b, ld3 = spline_transform(z, p3, inverse=False)
        return float(b[0, center, center]), float((ld1 + ld2 + ld3)[0, center, center])
    base, base_ld = central(c)
    changed, changed_ld = central(c_alt)
    results.append({"volume": 16, "test": "outside_radius_perturbation", "abs_diff_body": abs(base - changed), "abs_diff_logdet": abs(base_ld - changed_ld)})
    return results


def check_locality(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row.get("test") == "outside_radius_perturbation":
            if row["abs_diff_body"] > 1.0e-6 or row["abs_diff_logdet"] > 1.0e-5:
                raise RuntimeError("outside-radius locality test failed")
        elif row["abs_diff_e1"] > 1.0e-6 or row["abs_diff_e2"] > 1.0e-6 or row["abs_diff_body"] > 1.0e-6 or row["abs_diff_logdet"] > 1.0e-5 or row["abs_diff_logq"] > 1.0e-5:
            raise RuntimeError(f"volume locality test failed at L={row['volume']}")


def smoke(model: LocalMultistageFlow, dataset: Dataset, kernel: np.ndarray, device: torch.device, out: Path) -> list[dict[str, Any]]:
    model.train()
    c = torch.from_numpy(dataset.c[:8]).to(device)
    e1 = torch.from_numpy(dataset.e1[:8]).to(device)
    e2 = torch.from_numpy(dataset.e2[:8]).to(device)
    body = torch.from_numpy(dataset.body[:8]).to(device)
    origins = torch.zeros((len(c), 2), dtype=torch.long, device=device)
    tiles = batch_tiles({"c": c, "e1": e1, "e2": e2, "body": body}, origins)
    nll, parts = model.log_prob_patch(tiles["c"], tiles["e1"], tiles["e2"], tiles["body"])
    nll.backward()
    grad_ok = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad(set_to_none=True)
    cphys = destandardize(c, dataset.stats, "c")
    e1phys = destandardize(e1, dataset.stats, "e1")
    e2phys = destandardize(e2, dataset.stats, "e2")
    bphys = destandardize(body, dataset.stats, "body")
    psi = assemble_psi(cphys, e1phys, e2phys, bphys).detach().cpu().numpy()
    phi, _ = inverse_kernel(psi, kernel)
    reb = apply_kernel(phi, kernel)[:, 0::2, 0::2]
    rows = [{"test": "patch_forward_nll_finite", "value": float(nll.detach().cpu()), "pass": bool(torch.isfinite(nll))}, {"test": "stage_gradients_finite", "value": float(grad_ok), "pass": bool(grad_ok)}, {"test": "reblocking_max_abs_error", "value": float(np.max(np.abs(reb - cphys.detach().cpu().numpy()))), "pass": bool(np.max(np.abs(reb - cphys.detach().cpu().numpy())) < 2.0e-5)}]
    write_csv(out / "smoke" / "smoke_tests.csv", rows)
    if not all(bool(row["pass"]) for row in rows):
        raise RuntimeError("smoke test failure")
    return rows


def run_epoch(model: LocalMultistageFlow, data: Dataset, indices: np.ndarray, optimizer: torch.optim.Optimizer | None, batch_size: int, kernel_fft: torch.Tensor, device: torch.device, epoch: int, train: bool, observable_scale: float = 1.0) -> tuple[dict[str, float], list[dict[str, float]]]:
    order = np.array(indices, copy=True)
    if train:
        np.random.default_rng(10000 + epoch).shuffle(order)
    model.train(train)
    totals: dict[str, float] = {key: 0.0 for key in ("nll", "edge_nll", "correlation_nll", "body_nll", "observable_loss", "loss", "grad_norm", "clip_fraction", "runtime_s")}
    obs_rows: list[dict[str, float]] = []
    start_time = time.perf_counter()
    count = 0
    for offset in range(0, len(order), batch_size):
        idx = order[offset : offset + batch_size]
        batch = {key: torch.from_numpy(getattr(data, key)[idx]).to(device) for key in ("c", "e1", "e2", "body")}
        native = torch.from_numpy(data.phi[idx]).to(device)
        rng = np.random.default_rng(20000 + epoch * 1000 + offset)
        origins = torch.as_tensor(rng.integers(0, batch["c"].shape[1], size=(len(idx), 2)), dtype=torch.long, device=device)
        tiles = batch_tiles(batch, origins)
        with torch.set_grad_enabled(train):
            nll, pieces = model.log_prob_patch(tiles["c"], tiles["e1"], tiles["e2"], tiles["body"])
            if observable_scale:
                # Physical local-observable matching remains secondary to patch NLL.
                generated, _details, _logs = model_physical_sample(model, batch["c"], data.stats, kernel_fft)
                obs_loss, obs_detail = mean_width_loss(local_observables(generated), local_observables(native))
                obs_loss = obs_loss * observable_scale
            else:
                obs_loss = nll.new_tensor(0.0)
                obs_detail = {f"{key}_{part}": 0.0 for key in OBS_WEIGHTS for part in ("mean", "width")}
            loss = nll + obs_loss
            grad_norm = 0.0
            clipped = 0.0
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0).detach().cpu())
                clipped = float(grad_norm > 10.0)
                optimizer.step()
        size = len(idx)
        count += size
        totals["nll"] += float(nll.detach().cpu()) * size
        totals["edge_nll"] += float(pieces["edge"].detach().cpu()) * size
        totals["correlation_nll"] += float(pieces["correlation"].detach().cpu()) * size
        totals["body_nll"] += float(pieces["body"].detach().cpu()) * size
        totals["observable_loss"] += float(obs_loss.detach().cpu()) * size
        totals["loss"] += float(loss.detach().cpu()) * size
        totals["grad_norm"] += grad_norm * size
        totals["clip_fraction"] += clipped * size
        obs_rows.append(obs_detail)
    totals["runtime_s"] = time.perf_counter() - start_time
    for key in list(totals):
        if key != "runtime_s":
            totals[key] /= max(count, 1)
    mean_obs = {key: float(np.mean([row[key] for row in obs_rows])) for key in obs_rows[0]} if obs_rows else {}
    return totals, [mean_obs]


def save_checkpoint(path: Path, model: LocalMultistageFlow, optimizer: torch.optim.Optimizer, dataset: Dataset, config: dict[str, Any], epoch: int) -> None:
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "stats": dataset.stats, "config": config, "epoch": epoch, "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state()}, path)


def sample_evaluate(model: LocalMultistageFlow, dataset: Dataset, indices: np.ndarray, device: torch.device, max_count: int, label: str, out: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    idx = indices[:max_count]
    c = torch.from_numpy(dataset.c[idx]).to(device)
    fft = torch_kernel_fft(np.asarray(json.loads((out / "kernel_metadata.json").read_text())["matrix"], dtype=np.float64), 2 * c.shape[1], device)
    model.eval()
    with torch.no_grad():
        gen = torch.Generator(device=device).manual_seed(50000 + c.shape[1])
        phi, _details, _logs = model_physical_sample(model, c, dataset.stats, fft, gen)
    native = dataset.phi[idx]
    generated = phi.detach().cpu().numpy().astype(np.float32)
    rows = metrics_rows(native, generated, label, "whole") + metrics_rows(native, generated, label, "site")
    write_csv(out / "observables" / f"raw_metrics_{label}.csv", rows)
    action_tail_plot(native, generated, out / "plots" / f"action_tail_{label}", f"{label}: native vs local flow")
    return native, generated, rows


def local_patch_mh(model: LocalMultistageFlow, dataset: Dataset, indices: np.ndarray, kernel: np.ndarray, device: torch.device, chains: int, sweeps: int, seed: int) -> dict[str, Any]:
    """Exact local detail-block MH using a P=4 proposal, for lightweight diagnostics only."""
    idx = indices[:chains]
    c = torch.from_numpy(dataset.c[idx]).to(device)
    fft = torch_kernel_fft(kernel, 2 * c.shape[1], device)
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    model.eval()
    with torch.no_grad():
        gen = torch.Generator(device=device).manual_seed(seed)
        e1, e2, body, _ = model.sample_full(c, gen)
    current = [e1, e2, body]
    def physical(values: list[torch.Tensor]) -> np.ndarray:
        psi = assemble_psi(destandardize(c, dataset.stats, "c"), destandardize(values[0], dataset.stats, "e1"), destandardize(values[1], dataset.stats, "e2"), destandardize(values[2], dataset.stats, "body"))
        return torch_inverse_kernel(psi, fft).detach().cpu().numpy()
    phi = physical(current)
    from train_lam1p0_autoregressive_detail_flow import action_total  # noqa: E402
    current_s = np.asarray(action_total(phi, action), dtype=np.float64)
    rng = np.random.default_rng(seed + 1)
    rows = []
    for sweep in range(1, sweeps + 1):
        origins = torch.as_tensor(rng.integers(0, c.shape[1], size=(len(c), 2)), dtype=torch.long, device=device)
        with torch.no_grad():
            pe1, pe2, pb, _ = model.sample_full(c, torch.Generator(device=device).manual_seed(seed + 200 + sweep))
        candidate = [x.clone() for x in current]
        selected = torch.zeros_like(c, dtype=torch.bool)
        for b in range(len(c)):
            yy = (torch.arange(PATCH_SIZE, device=device) + origins[b, 0]) % c.shape[1]
            xx = (torch.arange(PATCH_SIZE, device=device) + origins[b, 1]) % c.shape[1]
            selected[b][yy[:, None], xx[None, :]] = True
            candidate[0][b][yy[:, None], xx[None, :]] = pe1[b][yy[:, None], xx[None, :]]
            candidate[1][b][yy[:, None], xx[None, :]] = pe2[b][yy[:, None], xx[None, :]]
            candidate[2][b][yy[:, None], xx[None, :]] = pb[b][yy[:, None], xx[None, :]]
        prop_phi = physical(candidate)
        prop_s = np.asarray(action_total(prop_phi, action), dtype=np.float64)
        with torch.no_grad():
            old_logq, _ = model.log_prob_full(c, current[0], current[1], current[2])
            new_logq, _ = model.log_prob_full(c, candidate[0], candidate[1], candidate[2])
        old_patch_logq = (old_logq * selected).sum(dim=(1, 2)).cpu().numpy()
        new_patch_logq = (new_logq * selected).sum(dim=(1, 2)).cpu().numpy()
        delta_s = prop_s - current_s
        loga = -delta_s + old_patch_logq - new_patch_logq
        accept = np.log(rng.random(len(c))) < np.minimum(loga, 0.0)
        for stage in range(3):
            current[stage][accept] = candidate[stage][accept]
        phi[accept] = prop_phi[accept]
        current_s[accept] = prop_s[accept]
        rows.append({"sweep": sweep, "attempts": int(len(c)), "accepted": int(np.sum(accept)), "acceptance": float(np.mean(accept)), "DeltaS_mean": float(np.mean(delta_s)), "logA_mean": float(np.mean(loga))})
    return {"acceptance": float(np.mean([row["acceptance"] for row in rows])), "rows": rows, "nonfinite_count": int(np.sum(~np.isfinite(phi)))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--kernel-path", type=Path, default=Path("perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--tiny-epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.0e-4)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026072116)
    args = ap.parse_args()
    if args.epochs > 10:
        raise SystemExit("N1000 pilot is capped at 10 epochs")
    run = args.run_dir
    for name in ("architecture", "smoke", "tiny_overfit", "N1000", "zero_shot_L16to32", "zero_shot_L32to64", "wrapped_baseline_comparison", "plots", "logs", "checkpoints", "observables", "summaries"):
        (run / name).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cpu")
    sha = hashlib.sha256(args.kernel_path.read_bytes()).hexdigest()
    if sha != KERNEL_SHA256:
        raise SystemExit(f"unexpected kernel hash: {sha}")
    kernel, raw = load_kernel_matrix(args.kernel_path)
    if raw.get("family") != "support_balanced_5x5" or not raw.get("kernel_coefficients_include_eta_scale") or not np.isclose(kernel.sum(), ETA_SCALE):
        raise SystemExit("final kernel metadata/normalization check failed")
    raw["sha256"] = sha
    raw["matrix"] = np.asarray(kernel).tolist()
    write_json(run / "kernel_metadata.json", raw)
    architecture = {"factorization": "q_edge(d01|c) q_corr(d10|c,d01) q_body(d11|c,d01,d10)", "edge": "directional valid 1x3 -> 3x1 -> RQ head", "correlation": "valid coarse 3x3 plus pointwise d01 -> RQ head", "body": "valid coarse 3x3 plus pointwise radial/cross-edge features -> RQ head", "spline": {"num_bins": NUM_BINS, "tail_bound": TAIL_BOUND, "tails": "linear", "min_bin_width": MIN_BIN_WIDTH, "min_bin_height": MIN_BIN_HEIGHT, "min_derivative": MIN_DERIVATIVE}, **LocalMultistageFlow.receptive_field()}
    write_json(run / "architecture_spec.json", architecture)
    (run / "architecture" / "receptive_field_analysis.md").write_text("# Receptive Field\n\nEach stage reads only a physical `3x3` coarse patch (radius 1). Correlation and body stages read prior details only at the same site. Therefore the end-to-end coarse receptive radius is exactly 1. All neural convolutions use `padding=0`; periodicity appears only in full-lattice patch extraction.\n", encoding="utf-8")
    phi16_all = load_phi(Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"))
    if len(phi16_all) < 1000:
        raise SystemExit("need at least 1000 native L16 configurations")
    selection = np.random.default_rng(args.seed).permutation(len(phi16_all))[:1000]
    phi16 = phi16_all[selection]
    train_idx = np.arange(0, 800); val_idx = np.arange(800, 900); test_idx = np.arange(900, 1000)
    write_json(run / "dataset_split.json", {"source": "native_L16", "total_used": 1000, "selection_seed": args.seed, "train_count": 800, "validation_count": 100, "test_count": 100, "source_indices": selection.tolist(), "train_local_indices": train_idx.tolist(), "validation_local_indices": val_idx.tolist(), "test_local_indices": test_idx.tolist()})
    data = make_dataset(phi16, kernel, train_idx)
    write_json(run / "normalization_metadata.json", {key: {name: value.tolist() for name, value in item.items()} for key, item in data.stats.items()})
    model = LocalMultistageFlow(hidden=args.hidden).to(device)
    locality = locality_tests(model, device); check_locality(locality); write_csv(run / "volume_independence_tests.csv", locality)
    smoke(model, data, kernel, device, run)
    # Tiny overfit uses only 32 examples and is required before the pilot.
    tiny = LocalMultistageFlow(hidden=args.hidden).to(device)
    tiny_opt = torch.optim.AdamW(tiny.parameters(), lr=5.0e-4, weight_decay=1.0e-5)
    fft16 = torch_kernel_fft(kernel, 16, device)
    tiny_history = []
    for epoch in range(1, args.tiny_epochs + 1):
        metrics, _ = run_epoch(tiny, data, train_idx[:32], tiny_opt, 32, fft16, device, epoch, True, observable_scale=0.0)
        metrics["epoch"] = epoch; tiny_history.append(metrics)
    write_csv(run / "tiny_overfit" / "training_history.csv", tiny_history)
    if tiny_history[-1]["nll"] >= tiny_history[0]["nll"] * 0.70:
        raise RuntimeError("tiny overfit did not reduce NLL sufficiently")
    tiny_locality = locality_tests(tiny, device); check_locality(tiny_locality); write_csv(run / "tiny_overfit" / "volume_independence_tests.csv", tiny_locality)
    # Fresh model for N1000 pilot; no wrapped-model weights are reused.
    model = LocalMultistageFlow(hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-5)
    history: list[dict[str, Any]] = []; stage_rows: list[dict[str, Any]] = []; mean_width_rows: list[dict[str, Any]] = []
    best = float("inf"); best_state: dict[str, torch.Tensor] | None = None; bad = 0
    config = {"lambda": 1.0, "kappa_c": 0.340301, "kappa_f": 0.340301, "eta": 0.25, "eta_scale": ETA_SCALE, "L_c": 8, "L_f": 16, "kernel_path": str(args.kernel_path), "kernel_coefficients_include_eta_scale": True, "training": {"total_configs": 1000, "train": 800, "validation": 100, "test": 100, "epochs_max": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "fresh_initialization": True, "global_observables_in_loss": False}}
    for epoch in range(1, args.epochs + 1):
        tr, tr_obs = run_epoch(model, data, train_idx, optimizer, args.batch_size, fft16, device, epoch, True)
        va, va_obs = run_epoch(model, data, val_idx, None, args.batch_size, fft16, device, epoch, False)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"validation_{k}": v for k, v in va.items()}}
        history.append(row); stage_rows.append({"epoch": epoch, "edge_nll": va["edge_nll"], "correlation_nll": va["correlation_nll"], "body_nll": va["body_nll"], "total_nll": va["nll"], "grad_norm": tr["grad_norm"], "clip_fraction": tr["clip_fraction"]}); mean_width_rows.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr_obs[0].items()}, **{f"validation_{k}": v for k, v in va_obs[0].items()}})
        print(json.dumps(row), flush=True)
        if va["nll"] < best:
            best = va["nll"]; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; bad = 0
            save_checkpoint(run / "checkpoints" / "checkpoint_best.pt", model, optimizer, data, config, epoch)
        else:
            bad += 1
        save_checkpoint(run / "checkpoints" / "checkpoint_latest.pt", model, optimizer, data, config, epoch)
        if bad >= 4:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    write_csv(run / "observables" / "training_history.csv", history); write_csv(run / "observables" / "stagewise_losses.csv", stage_rows); write_csv(run / "observables" / "local_mean_width_losses.csv", mean_width_rows)
    native8, generated8, rows8 = sample_evaluate(model, data, test_idx, device, 100, "L8to16", run)
    # Zero-shot datasets use the L8 normalization unchanged, as required.
    transfer_rows = list(rows8)
    for label, path, max_count in (("L16to32", Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"), 100), ("L32to64", Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz"), 100)):
        if not path.exists():
            continue
        native = load_phi(path)[:max_count]
        pairs = split_pairs(native, kernel)
        c = pairs["coarse"]
        c_norm = ((c - data.stats["c"]["mean"]) / data.stats["c"]["std"]).astype(np.float32)
        class TransferData: pass
        td = TransferData(); td.c = c_norm; td.stats = data.stats; td.phi = native
        fft = torch_kernel_fft(kernel, native.shape[1], device)
        model.eval()
        with torch.no_grad():
            phi_gen, _d, _l = model_physical_sample(model, torch.from_numpy(c_norm).to(device), data.stats, fft, torch.Generator(device=device).manual_seed(args.seed + native.shape[1]))
        generated = phi_gen.cpu().numpy().astype(np.float32)
        rows = metrics_rows(native, generated, label, "whole") + metrics_rows(native, generated, label, "site")
        write_csv(run / "observables" / f"raw_metrics_{label}.csv", rows); action_tail_plot(native, generated, run / "plots" / f"action_tail_{label}", f"{label}: zero-shot local flow")
        transfer_rows.extend(rows)
        chains, sweeps = ((16, 10) if label == "L16to32" else (8, 5))
        diag_data = TransferData(); diag_data.c = c_norm; diag_data.stats = data.stats; diag_data.phi = native
        diag = local_patch_mh(model, diag_data, np.arange(len(native)), kernel, device, chains, sweeps, args.seed + native.shape[1])
        write_csv(run / "observables" / f"local_patch_diagnostics_{label}.csv", diag["rows"])
        write_json(run / ("zero_shot_" + label) / "local_patch_summary.json", {k: v for k, v in diag.items() if k != "rows"})
    diag8 = local_patch_mh(model, data, test_idx, kernel, device, 16, 10, args.seed + 16)
    write_csv(run / "observables" / "local_patch_diagnostics_L8to16.csv", diag8["rows"])
    locality_after = locality_tests(model, device); check_locality(locality_after); write_csv(run / "N1000" / "volume_independence_tests_after_training.csv", locality_after)
    write_csv(run / "observables" / "all_volume_metrics.csv", transfer_rows)
    (run / "run_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(run / "submit_manifest.txt", {"command": " ".join(sys.argv), "git_commit": git_commit(), "host": socket.gethostname(), "platform": platform.platform(), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    summary = ["# Lambda 1.0 Local Multistage RQ-Spline Pilot", "", "- Fresh local edge/correlation/body initialization; no wrapped-model weights.", "- N1000 split: 800 train / 100 validation / 100 test.", f"- Best validation local patch NLL: `{best:.6g}`.", f"- End-to-end coarse receptive radius: `1`.", f"- L8->L16 local patch acceptance: `{diag8['acceptance']:.6g}`.", "- No global observable is used in the training loss."]
    (run / "summaries" / "run_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (run / "summaries" / "volume_transfer_summary.md").write_text("# Volume Transfer\n\nSee `observables/raw_metrics_L8to16.csv`, `raw_metrics_L16to32.csv`, and `raw_metrics_L32to64.csv`. The architecture locality test passes before and after training.\n", encoding="utf-8")
    write_json(run / "status.json", {"status": "completed", "best_validation_nll": best, "epochs_completed": len(history), "N2000_started": False, "locality_passed": True})
    print(json.dumps({"status": "completed", "run_dir": str(run), "best_validation_nll": best, "epochs": len(history)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
