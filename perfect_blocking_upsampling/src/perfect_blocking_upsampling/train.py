from __future__ import annotations

"""Training scaffolding.

The package is parameterized by config. The training entry point validates the
config and dispatches to the baseline stage pipeline. The frozen example
configuration is fully supported; additional action points require matching
data/checkpoints.
"""

from pathlib import Path
from typing import Any

from .io import load_yaml


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    return {"status": "not_implemented_in_scaffold", "config": cfg}

