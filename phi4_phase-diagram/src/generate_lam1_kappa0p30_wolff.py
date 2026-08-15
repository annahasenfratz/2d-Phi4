#!/usr/bin/env python3
"""Generation request for the canonical lambda=1, kappa=0.30 ensemble.

This script deliberately refuses to create production configs with the older
mixed embedded-Wolff-plus-local-Metropolis generator. The standing project rule
is Wolff-only generation for new ensembles.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "phi4_phase-diagram"
ACTION = "finite-lambda phi4 kappa/lambda convention"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--n-configs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20263030)
    args = parser.parse_args()

    request_dir = OUT_ROOT / "generation_requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "status": "blocked_missing_wolff_only_generator",
        "lambda": 1.0,
        "kappa": 0.30,
        "L": args.L,
        "n_configs": args.n_configs,
        "seed": args.seed,
        "required_output_directory": str(
            OUT_ROOT / "ensembles" / f"lam1p000_kappa0p300_L{args.L}_wolff"
        ),
        "required_configs_file": "configs.npz",
        "required_keys": ["phi", "lambda", "kappa", "L", "n_configs", "generator", "seed", "action_convention"],
        "action_convention": ACTION,
        "reason": (
            "A pure Wolff-only finite-lambda phi4 generator is not implemented in this repository. "
            "The available diagnostic generator used embedded Wolff sign-cluster updates plus local "
            "Metropolis amplitude updates, which is superseded and must not be used for new production/training."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = request_dir / f"lam1p000_kappa0p300_L{args.L}_wolff_request.json"
    path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    print(f"Wrote generation request: {path}")
    print(request["reason"])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
