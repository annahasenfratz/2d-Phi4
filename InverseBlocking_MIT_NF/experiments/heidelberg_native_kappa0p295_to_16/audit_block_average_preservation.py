#!/usr/bin/env python3
"""Audit whether the Heidelberg CNF preserves simple 2x2 block averages."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
BRANCH = Path(__file__).resolve().parent
OUT = BRANCH / "outputs/block_average_preservation_audit"
HEIDELBERG = ROOT / "heidelberg-phi4-reproduction"
HEIDELBERG_SCRIPTS = HEIDELBERG / "scripts"
sys.path.insert(0, str(HEIDELBERG))
sys.path.insert(0, str(HEIDELBERG_SCRIPTS))

from train_ir_matching_l8_torch_cnf import PaperCNF, make_unit_noise, naive_upsample_torch  # noqa: E402


COARSE_NPY = (
    ROOT
    / "InverseBlocking_MIT_NF/outputs/coarse_distribution_calibration/generated_native_wolff/"
    / "native_coarse_lam1_kappa0p295_L8_wolff.npy"
)
TINY_CKPT = BRANCH / "outputs/tiny_pilot_kappaf0p320_sigma0p15/checkpoints/model_step_0019.pt"


def block_average_torch(phi: torch.Tensor) -> torch.Tensor:
    b, lf, _ = phi.shape
    return phi.reshape(b, lf // 2, 2, lf // 2, 2).mean(dim=(2, 4))


def residual_stats(field: torch.Tensor, coarse: torch.Tensor) -> dict[str, float]:
    diff = block_average_torch(field) - coarse
    return {
        "max_abs_residual": float(torch.max(torch.abs(diff)).detach().cpu()),
        "rms_residual": float(torch.sqrt(torch.mean(diff * diff)).detach().cpu()),
        "mean_abs_residual": float(torch.mean(torch.abs(diff)).detach().cpu()),
    }


def make_model(seed: int, init_sigma: float = 0.15) -> PaperCNF:
    torch.manual_seed(seed)
    return PaperCNF(
        kernel_radius=1,
        n_field_features=5,
        n_time_features=5,
        field_bond_dim=6,
        time_bond_dim=6,
        init_sigma=init_sigma,
        init_weight_scale=1.0e-3,
        init_scale_flow=1.0,
        block_period=2,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)

    dtype = torch.float32
    coarse_np = np.load(COARSE_NPY)[:128].astype(np.float32)
    coarse = torch.tensor(coarse_np, dtype=dtype)
    torch.manual_seed(20260624)
    unit_noise = make_unit_noise(coarse.shape[0], 16, torch.device("cpu"), dtype)
    sigma = torch.tensor(0.15, dtype=dtype)
    init = naive_upsample_torch(coarse) + sigma * unit_noise

    rows: list[dict] = []
    init_stats = residual_stats(init, coarse)
    rows.append(
        {
            "case": "initial_zero_sum_base",
            "checkpoint": "",
            "cnf_steps": 0,
            "logdet_mean": 0.0,
            **init_stats,
        }
    )

    random_model = make_model(seed=777)
    with torch.no_grad():
        random_out, random_logdet = random_model(init, n_steps=4)
    rows.append(
        {
            "case": "random_initialized_cnf",
            "checkpoint": "",
            "cnf_steps": 4,
            "logdet_mean": float(torch.mean(random_logdet).detach().cpu()),
            **residual_stats(random_out, coarse),
        }
    )

    trained_model = make_model(seed=202)
    state = torch.load(TINY_CKPT, map_location="cpu")
    trained_model.load_state_dict(state)
    with torch.no_grad():
        trained_out, trained_logdet = trained_model(init, n_steps=4)
    rows.append(
        {
            "case": "tiny_pilot_checkpoint_step_0019",
            "checkpoint": str(TINY_CKPT),
            "cnf_steps": 4,
            "logdet_mean": float(torch.mean(trained_logdet).detach().cpu()),
            **residual_stats(trained_out, coarse),
        }
    )

    write_csv(OUT / "residual_checks.csv", rows)
    summary = {
        "coarse_npy": str(COARSE_NPY),
        "coarse_shape_used": list(coarse_np.shape),
        "sigma": 0.15,
        "tiny_checkpoint": str(TINY_CKPT),
        "conclusion": "The Heidelberg CNF implementation does not preserve simple 2x2 block averages by construction; preservation holds only for the initial zero-sum-noise base.",
        "rows": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    code_notes = f"""# Code Path Notes

## Initial constrained variable

`heidelberg_phi4/cnf_architecture.py` and `train_ir_matching_l8_torch_cnf.py` construct

```text
base = naive_upsample(coarse) + sigma * zero_sum_block_noise
```

The helper `make_unit_noise` subtracts each 2x2 block mean, so `block_average(base) = coarse` at initialization.

## CNF update

The Torch `PaperCNF` class evolves the full fine field:

```text
phi <- phi + dt * vector_field(phi, t)
```

`vector_field` returns one value per 16x16 lattice site. It uses 2x2 phase masks to allow different parameters for the four sublattices, but it does not subtract the 2x2 block mean of the vector field and does not evolve only three zero-sum detail coordinates per block.

Therefore the implementation updates all 16x16 sites freely after initialization.

## Missing ingredient if exact block-average preservation was intended

To preserve the simple block average exactly, the ODE vector field would need to satisfy

```text
sum_{{a,b in 2x2 block}} G_{{2i+a,2j+b}}(phi,t) = 0
```

for every block and every integration time, or the flow would need to be expressed in coordinates `(coarse block average, zero-sum detail)` and update only the detail coordinates. The current code does neither.

"""
    (OUT / "code_path_notes.md").write_text(code_notes)

    report = ["# Heidelberg CNF Block-Average Preservation Audit", ""]
    report.append("## Question")
    report.append("")
    report.append("Does the Heidelberg CNF architecture preserve the simple 2x2 block average by construction, or only at initialization?")
    report.append("")
    report.append("## Answer")
    report.append("")
    report.append("It preserves the simple 2x2 block average only at initialization. The CNF itself updates all fine sites through a full-lattice vector field and is not constrained to have zero block-sum velocity.")
    report.append("")
    report.append("## Residual Checks")
    report.append("")
    report.append("| case | RMS residual | max abs residual | mean abs residual | logdet mean |")
    report.append("| --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        report.append(
            f"| {row['case']} | {row['rms_residual']:.6g} | {row['max_abs_residual']:.6g} | "
            f"{row['mean_abs_residual']:.6g} | {row['logdet_mean']:.6g} |"
        )
    report.append("")
    report.append("## Interpretation")
    report.append("")
    report.append("- Initial zero-sum block noise preserves `block_average(phi)=phi_c` to roundoff.")
    report.append("- Random initialized CNF weights already create a nonzero block-average residual, although small for tiny random weights.")
    report.append("- The tiny-pilot checkpoint creates a much larger residual, confirming that training used the freedom to move block averages.")
    report.append("- This matches the code: the phase masks distinguish sublattices but do not impose a zero-sum constraint within each 2x2 block.")
    report.append("")
    report.append("## If Paper-Style Preservation Was Intended")
    report.append("")
    report.append("The missing piece is a constrained ODE parameterization. The vector field would need to be projected to zero block average at every CNF step, or the model would need to evolve only the three zero-sum detail coordinates per block while holding the coarse average fixed.")
    report.append("")
    (OUT / "report.md").write_text("\n".join(report))
    print(json.dumps({"output": str(OUT), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()

