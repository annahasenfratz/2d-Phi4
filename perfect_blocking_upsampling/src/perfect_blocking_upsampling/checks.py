from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .io import ActionSpec


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256_manifest(manifest_path: str | Path, root: str | Path | None = None) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_path)
    root = manifest_path.parent.parent if root is None else Path(root)
    rows = []
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, rel = line.split(maxsplit=1)
        path = (root / rel).resolve()
        actual = sha256(path)
        rows.append({"path": str(path), "expected_sha256": digest, "actual_sha256": actual, "matches": int(actual == digest)})
    return rows


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def validate_ensemble_manifest(manifest: dict[str, Any], action: ActionSpec, lattice_L: int) -> list[str]:
    errors = []
    if int(manifest.get("L", -1)) != lattice_L:
        errors.append(f"L mismatch: {manifest.get('L')} != {lattice_L}")
    if abs(float(manifest.get("lambda", action.lambda_)) - action.lambda_) > 1.0e-12:
        errors.append(f"lambda mismatch: {manifest.get('lambda')} != {action.lambda_}")
    if abs(float(manifest.get("kappa", action.kappa)) - action.kappa) > 1.0e-12:
        errors.append(f"kappa mismatch: {manifest.get('kappa')} != {action.kappa}")
    if action.type == "phi4_nn_plus_diag" and float(manifest.get("kappa_diag", action.kappa_diag)) != action.kappa_diag:
        # Optional metadata field; only warn at call site.
        pass
    return errors

