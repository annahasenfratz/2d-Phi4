from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def dump_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def dump_yaml(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(obj, sort_keys=False))


def resolve_path(base: Path, maybe_relative: str | Path) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else (base / p).resolve()


@dataclass(frozen=True)
class ActionSpec:
    type: str
    lambda_: float
    kappa: float
    kappa_diag: float = 0.0

    @property
    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "lambda": self.lambda_,
            "kappa": self.kappa,
            "kappa_diag": self.kappa_diag,
        }


@dataclass(frozen=True)
class KernelSpec:
    name: str
    type: str
    eta: float
    scale_factor: int
    normalize: bool = True
