#!/usr/bin/env python3
"""Create a provenance-preserving concatenation of fine-field checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if len(a.source) < 2:
        raise ValueError("at least two --source checkpoints are required")
    fields: list[np.ndarray] = []
    source_indices: list[np.ndarray] = []
    for path in a.source:
        with np.load(path) as data:
            if "phi" not in data:
                raise ValueError(f"checkpoint has no phi array: {path}")
            fields.append(np.asarray(data["phi"], dtype=np.float32))
            source_indices.append(
                np.asarray(data["source_indices"], dtype=np.int64)
                if "source_indices" in data else np.arange(len(fields[-1]), dtype=np.int64)
            )
    shape = fields[0].shape[1:]
    if any(x.shape[1:] != shape for x in fields):
        raise ValueError(f"incompatible field shapes: {[x.shape for x in fields]}")
    total = sum(len(x) for x in fields)
    print(json.dumps({"sources": [str(x) for x in a.source], "field_shape": shape, "n_chains": total}))
    if a.dry_run:
        return 0
    if a.output.exists():
        raise FileExistsError(f"refusing to overwrite existing merged input: {a.output}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    phi = np.concatenate(fields, axis=0)
    replica = np.concatenate([np.full(len(x), i, dtype=np.int16) for i, x in enumerate(fields)])
    np.savez_compressed(a.output, phi=phi, source_indices=np.concatenate(source_indices), replica=replica)
    manifest = {
        "sources": [str(x.resolve()) for x in a.source], "n_chains": int(total),
        "field_shape": list(shape), "ordering": "all chains from source 0, then source 1, etc.",
    }
    a.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
