#!/usr/bin/env python3
"""Inventory and centralize finite-lambda phi4 configuration arrays.

The central repository is intentionally conservative: copied arrays keep
provenance, but only unambiguous Wolff ensembles may become canonical training
inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "phi4_phase-diagram"
SCAN_ROOTS = [
    ROOT / "MIT_NN_test",
    ROOT / "ML_sampling_clean",
    ROOT / "testing_mlneuralsampler_multilevel",
    ROOT / "InverseBlocking_MIT_NF",
    ROOT / "inverse_blocking_flow",
    ROOT / "perfect_blocking",
    ROOT / "data",
]
EXTENSIONS = {".npy", ".npz", ".pt", ".pth", ".h5", ".hdf5"}
ACTION_KAPPA_LAMBDA = (
    "S=sum_x[(1-2*lambda)*phi_x^2 + lambda*phi_x^4] "
    "- 2*kappa*sum_x,mu phi_x phi_{x+mu}; constant lambda omitted"
)
CANONICAL_GENERATORS = {"embedded_wolff_sign_cluster_plus_radial_heatbath"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def file_id(path: Path) -> str:
    h = hashlib.sha1(str(path.relative_to(ROOT)).encode()).hexdigest()
    return h[:10]


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def sidecar_jsons(path: Path) -> list[Path]:
    names = [
        path.with_suffix(".json"),
        path.parent / f"{path.stem}_manifest.json",
        path.parent / f"{path.stem}_metadata.json",
        path.parent / f"{path.stem}_summary.json",
        path.parent / "manifest.json",
        path.parent / "metadata.json",
        path.parent / "generation_metadata.json",
        path.parent / "summary.json",
        path.parent / "config.json",
    ]
    return [p for p in names if p.exists() and p.is_file()]


def flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, val in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_json(val, next_prefix))
    elif isinstance(obj, list):
        for idx, val in enumerate(obj[:20]):
            out.update(flatten_json(val, f"{prefix}.{idx}"))
    else:
        out[prefix] = obj
    return out


def first_key(flat: dict[str, Any], names: list[str]) -> Any:
    lowered = {k.lower(): v for k, v in flat.items()}
    for name in names:
        target = name.lower()
        for key, val in lowered.items():
            if key.endswith(target):
                return val
    return None


def as_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def as_int(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def classify_generator(text: str, metadata_value: Any) -> str:
    if metadata_value is not None:
        s = str(metadata_value).lower()
    else:
        s = ""
    if s in CANONICAL_GENERATORS:
        return s
    s = f"{s}\n{text.lower()}"
    if "embedded_wolff_sign_cluster_plus_radial_heatbath" in s:
        return "embedded_wolff_sign_cluster_plus_radial_heatbath"
    has_wolff = "wolff" in s or "cluster" in s
    has_metropolis = "metropolis" in s or "local_acceptance" in s or "proposal_width" in s
    has_hmc = "hmc" in s or "hamiltonian" in s
    if has_wolff and has_metropolis:
        return "mixed"
    if has_wolff:
        return "wolff"
    if has_metropolis:
        return "metropolis"
    if has_hmc:
        return "hmc"
    return "unknown"


def normalize_field_shape(shape: tuple[int, ...]) -> tuple[int | None, int | None, bool]:
    if len(shape) == 2 and shape[0] == shape[1]:
        return 1, shape[0], shape[0] % 2 == 0
    if len(shape) == 3 and shape[1] == shape[2]:
        return shape[0], shape[1], shape[1] % 2 == 0
    if len(shape) == 4 and shape[1] == 1 and shape[2] == shape[3]:
        return shape[0], shape[2], shape[2] % 2 == 0
    if len(shape) == 4 and shape[-1] == 1 and shape[1] == shape[2]:
        return shape[0], shape[1], shape[1] % 2 == 0
    return None, None, False


def load_np_array_info(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        if path.suffix == ".npy":
            arr = np.load(path, mmap_mode="r")
            n, l, usable = normalize_field_shape(tuple(arr.shape))
            rows.append(
                {
                    "array_key": "phi",
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "n_configs": n,
                    "L": l,
                    "field_like": usable,
                    "load_error": None,
                }
            )
        elif path.suffix == ".npz":
            with np.load(path, mmap_mode="r") as data:
                for key in data.files:
                    arr = data[key]
                    n, l, usable = normalize_field_shape(tuple(arr.shape))
                    rows.append(
                        {
                            "array_key": key,
                            "shape": list(arr.shape),
                            "dtype": str(arr.dtype),
                            "n_configs": n,
                            "L": l,
                            "field_like": usable,
                            "load_error": None,
                        }
                    )
    except Exception as exc:
        rows.append(
            {
                "array_key": None,
                "shape": None,
                "dtype": None,
                "n_configs": None,
                "L": None,
                "field_like": False,
                "load_error": repr(exc),
            }
        )
    return rows


def inspect_hdf5(path: Path) -> list[dict[str, Any]]:
    try:
        import h5py  # type: ignore
    except Exception as exc:
        return [{"array_key": None, "field_like": False, "load_error": f"h5py unavailable: {exc!r}"}]
    rows: list[dict[str, Any]] = []
    try:
        with h5py.File(path, "r") as h5:
            def visit(name: str, obj: Any) -> None:
                if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                    n, l, usable = normalize_field_shape(tuple(obj.shape))
                    rows.append(
                        {
                            "array_key": name,
                            "shape": list(obj.shape),
                            "dtype": str(obj.dtype),
                            "n_configs": n,
                            "L": l,
                            "field_like": usable,
                            "load_error": None,
                        }
                    )

            h5.visititems(visit)
    except Exception as exc:
        rows.append({"array_key": None, "field_like": False, "load_error": repr(exc)})
    return rows


def inspect_torch(path: Path) -> list[dict[str, Any]]:
    lower = str(path).lower()
    if any(token in lower for token in ["checkpoint", "model", "optimizer", "best_by"]):
        return [{"array_key": None, "field_like": False, "load_error": "skipped model/checkpoint-like torch file"}]
    try:
        import torch  # type: ignore
    except Exception as exc:
        return [{"array_key": None, "field_like": False, "load_error": f"torch unavailable: {exc!r}"}]
    rows: list[dict[str, Any]] = []
    try:
        obj = torch.load(path, map_location="cpu")
        candidates: list[tuple[str, Any]] = []
        if hasattr(obj, "shape"):
            candidates.append(("tensor", obj))
        elif isinstance(obj, dict):
            for key, val in obj.items():
                if hasattr(val, "shape"):
                    candidates.append((str(key), val))
        for key, val in candidates:
            shape = tuple(int(x) for x in val.shape)
            n, l, usable = normalize_field_shape(shape)
            rows.append(
                {
                    "array_key": key,
                    "shape": list(shape),
                    "dtype": str(getattr(val, "dtype", "unknown")),
                    "n_configs": n,
                    "L": l,
                    "field_like": usable,
                    "load_error": None,
                }
            )
        if not rows:
            rows.append({"array_key": None, "field_like": False, "load_error": "no tensor-shaped payload found"})
    except Exception as exc:
        rows.append({"array_key": None, "field_like": False, "load_error": repr(exc)})
    return rows


def text_context(path: Path, sidecars: list[Path]) -> str:
    chunks = [str(path.relative_to(ROOT)), path.name]
    for p in sidecars:
        try:
            chunks.append(p.read_text(errors="ignore")[:20000])
        except Exception:
            pass
    return "\n".join(chunks)


def metadata_from_sidecars(sidecars: list[Path]) -> tuple[dict[str, Any], list[str]]:
    merged: dict[str, Any] = {}
    used: list[str] = []
    for p in sidecars:
        data = read_json(p)
        if data is None:
            continue
        used.append(str(p.relative_to(ROOT)))
        flat = flatten_json(data)
        merged.update(flat)
    return merged, used


def infer_path_param(patterns: list[str], text: str) -> float | None:
    low = text.lower()
    for pat in patterns:
        match = re.search(pat, low)
        if match:
            raw = match.group(1).replace("p", ".")
            try:
                return float(raw)
            except Exception:
                pass
    return None


def inspect_candidate(path: Path) -> list[dict[str, Any]]:
    sidecars = sidecar_jsons(path)
    flat, sidecar_used = metadata_from_sidecars(sidecars)
    context = text_context(path, sidecars)
    lam = as_float(first_key(flat, ["lambda", "lambda_", "lambda_f", "lam"]))
    kappa = as_float(first_key(flat, ["kappa", "kappa_f"]))
    m2 = as_float(first_key(flat, ["M2", "m2", "mass2"]))
    seed = first_key(flat, ["seed"])
    action = first_key(flat, ["action_convention"])
    generator_value = first_key(flat, ["generator", "generator_algorithm"])
    generator = classify_generator(context, generator_value)
    parameter_status = "metadata"
    if lam is None:
        path_lam = infer_path_param([r"lambda[_=-]?([0-9]+p?[0-9]*)", r"lam([0-9]+p[0-9]+)", r"lam([0-9]+)"], context)
        if path_lam is not None:
            lam = path_lam
            parameter_status = "path_only_unverified"
    if kappa is None:
        path_kappa = infer_path_param([r"kappa[_=-]?([0-9]+p?[0-9]*)", r"kappa([0-9]+p[0-9]+)"], context)
        if path_kappa is not None:
            kappa = path_kappa
            parameter_status = "path_only_unverified"
    if sidecar_used == [] and (lam is not None or kappa is not None):
        parameter_status = "path_only_unverified"
    if lam is None and kappa is None and m2 is None:
        parameter_status = "ambiguous"

    if path.suffix in {".npy", ".npz"}:
        arrays = load_np_array_info(path)
    elif path.suffix in {".h5", ".hdf5"}:
        arrays = inspect_hdf5(path)
    else:
        arrays = inspect_torch(path)

    rows = []
    for arr in arrays:
        production_grade = "unknown"
        if any(token in str(path).lower() for token in ["smoke", "tiny", "diagnostic", "generated_samples", "sample_examples", "initializer"]):
            production_grade = "diagnostic-only"
        is_superseded = generator in {"metropolis", "mixed"}
        canonical = bool(
            arr.get("field_like")
            and generator in CANONICAL_GENERATORS
            and parameter_status == "metadata"
            and lam is not None
            and kappa is not None
            and not is_superseded
            and "local_metropolis_used\": true" not in context.lower()
        )
        row = {
            "original_path": str(path.relative_to(ROOT)),
            "absolute_path": str(path),
            "file_type": path.suffix,
            "array_key": arr.get("array_key"),
            "shape": json.dumps(arr.get("shape")),
            "dtype": arr.get("dtype"),
            "number_of_configurations": arr.get("n_configs"),
            "lattice_size_L": arr.get("L"),
            "lambda": lam,
            "kappa": kappa,
            "M2": m2,
            "action_convention": action or (ACTION_KAPPA_LAMBDA if lam is not None or kappa is not None else None),
            "generator_algorithm": generator,
            "seed": seed,
            "parameter_status": parameter_status,
            "metadata_sources": json.dumps(sidecar_used),
            "production_grade": production_grade,
            "is_superseded_metropolis": is_superseded,
            "is_canonical": canonical,
            "field_like": bool(arr.get("field_like")),
            "load_error": arr.get("load_error"),
            "copy_status": "not_copied",
            "central_path": None,
            "notes": None,
        }
        if generator == "mixed":
            row["notes"] = "Superseded for new production/training because it includes local Metropolis amplitude updates."
        elif generator == "metropolis":
            row["notes"] = "Superseded for new production/training because it is Metropolis-generated."
        elif parameter_status != "metadata":
            row["notes"] = "Parameters are not unambiguously verified by metadata; not canonical."
        rows.append(row)
    return rows


def inspect_existing_central_manifest(manifest_path: Path) -> dict[str, Any] | None:
    manifest = read_json(manifest_path)
    if manifest is None:
        return None
    configs = manifest_path.parent / manifest.get("configs_file", "configs.npz")
    if not configs.exists():
        return None
    try:
        with np.load(configs, mmap_mode="r") as data:
            arr = data["phi"]
            n, l_val, field_like = normalize_field_shape(tuple(arr.shape))
            dtype = str(arr.dtype)
            shape = list(arr.shape)
    except Exception as exc:
        n, l_val, field_like, dtype, shape = None, manifest.get("L"), False, None, None
        load_error = repr(exc)
    else:
        load_error = None
    generator = classify_generator(json.dumps(manifest), manifest.get("generator"))
    lam = as_float(manifest.get("lambda"))
    kappa = as_float(manifest.get("kappa"))
    parameter_status = str(manifest.get("parameter_status") or "metadata")
    is_superseded = bool(manifest.get("is_superseded")) or generator in {"metropolis", "mixed"}
    canonical = bool(
        field_like
        and generator in CANONICAL_GENERATORS
        and parameter_status == "metadata"
        and lam is not None
        and kappa is not None
        and not is_superseded
        and manifest.get("local_metropolis_used") is False
        and manifest.get("sign_update") == "embedded_wolff_cluster"
        and manifest.get("amplitude_update") == "radial_heatbath"
    )
    return {
        "original_path": str(configs.relative_to(ROOT)),
        "absolute_path": str(configs),
        "file_type": ".npz",
        "array_key": "phi",
        "shape": json.dumps(shape),
        "dtype": dtype,
        "number_of_configurations": n if n is not None else manifest.get("n_configs"),
        "lattice_size_L": l_val if l_val is not None else manifest.get("L"),
        "lambda": lam,
        "kappa": kappa,
        "M2": manifest.get("M2"),
        "action_convention": manifest.get("action_convention"),
        "generator_algorithm": generator,
        "seed": manifest.get("seed"),
        "parameter_status": parameter_status,
        "metadata_sources": json.dumps([str(manifest_path.relative_to(ROOT))]),
        "production_grade": "production" if manifest.get("production_use") else "unknown",
        "is_superseded_metropolis": is_superseded,
        "is_canonical": canonical,
        "field_like": field_like,
        "load_error": load_error,
        "copy_status": "central_existing",
        "central_path": str(manifest_path.parent.relative_to(ROOT)),
        "notes": manifest.get("notes"),
    }


def fmt_float_tag(name: str, value: float | None) -> str:
    if value is None:
        return f"{name}unknown"
    return f"{name}{value:0.3f}".replace(".", "p")


def copy_array_to_repository(row: dict[str, Any], copied_at: str) -> dict[str, Any]:
    if not row.get("field_like"):
        return row
    src = ROOT / row["original_path"]
    if not src.exists():
        return row
    generator = row["generator_algorithm"]
    base = REPO / ("superseded_metropolis" if generator in {"metropolis", "mixed"} else "ensembles")
    lam = as_float(row.get("lambda"))
    kappa = as_float(row.get("kappa"))
    l_val = as_int(row.get("lattice_size_L"))
    tag = "_".join(
        [
            fmt_float_tag("lam", lam),
            fmt_float_tag("kappa", kappa),
            f"L{l_val}" if l_val is not None else "Lunknown",
            generator,
            file_id(src),
        ]
    )
    dest_dir = base / tag
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_npz = dest_dir / "configs.npz"
    key = row.get("array_key")
    try:
        if src.suffix == ".npy":
            arr = np.asarray(np.load(src))
        elif src.suffix == ".npz":
            with np.load(src) as data:
                arr = np.asarray(data[key])
        else:
            row["copy_status"] = "not_copied_non_numpy_source"
            return row
        if arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        elif arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        np.savez_compressed(
            dest_npz,
            phi=arr,
            **{
                "lambda": np.array(np.nan if lam is None else lam),
                "kappa": np.array(np.nan if kappa is None else kappa),
                "L": np.array(-1 if l_val is None else l_val),
                "n_configs": np.array(arr.shape[0] if arr.ndim == 3 else -1),
                "generator": np.array(generator),
                "seed": np.array("" if row.get("seed") is None else str(row.get("seed"))),
                "action_convention": np.array("" if row.get("action_convention") is None else str(row.get("action_convention"))),
            },
        )
    except Exception as exc:
        row["copy_status"] = f"copy_failed: {exc!r}"
        return row

    manifest = {
        "lambda": lam,
        "kappa": kappa,
        "M2": row.get("M2"),
        "action_convention": row.get("action_convention"),
        "L": l_val,
        "n_configs": int(arr.shape[0]) if arr.ndim == 3 else row.get("number_of_configurations"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "generator": generator,
        "seed": row.get("seed"),
        "source_path": row["absolute_path"],
        "source_array_key": key,
        "date_copied": copied_at,
        "is_canonical": bool(row.get("is_canonical")),
        "is_superseded": bool(row.get("is_superseded_metropolis")),
        "parameter_status": row.get("parameter_status"),
        "notes": row.get("notes") or "",
        "configs_file": "configs.npz",
    }
    provenance = {
        "source_path": row["absolute_path"],
        "source_relative_path": row["original_path"],
        "source_array_key": key,
        "copied_at": copied_at,
        "inventory_row": row,
        "policy": "Original files were not deleted. Metropolis/mixed-generator ensembles are superseded for new production/training.",
    }
    write_json(dest_dir / "manifest.json", manifest)
    write_json(dest_dir / "provenance.json", provenance)
    row["copy_status"] = "copied"
    row["central_path"] = str(dest_dir.relative_to(ROOT))
    return row


def discover_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in EXTENSIONS:
                continue
            # Avoid self-inventory copies; originals are represented by provenance.
            try:
                rel = path.relative_to(REPO)
                if rel.parts and rel.parts[0] in {"ensembles", "superseded_metropolis", "quarantine"}:
                    continue
            except Exception:
                pass
            files.append(path)
    return sorted(set(files))


def write_report(rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    field_like = sum(1 for r in rows if r.get("field_like"))
    canonical = sum(1 for r in rows if r.get("is_canonical"))
    superseded = sum(1 for r in rows if r.get("is_superseded_metropolis"))
    copied = sum(1 for r in rows if r.get("copy_status") == "copied")
    lam1_k030_canonical = [
        r
        for r in rows
        if r.get("field_like")
        and r.get("generator_algorithm") in CANONICAL_GENERATORS
        and r.get("parameter_status") == "metadata"
        and as_float(r.get("lambda")) == 1.0
        and as_float(r.get("kappa")) == 0.3
        and as_int(r.get("lattice_size_L")) == 16
    ]
    lines = [
        "# Phi4 Configuration Inventory",
        "",
        f"Inventory timestamp: `{now_iso()}`",
        "",
        "## Summary",
        "",
        f"- Inventory rows: {total}",
        f"- Field-like arrays: {field_like}",
        f"- Copied field-like arrays: {copied}",
        f"- Canonical radial-heatbath candidates: {canonical}",
        f"- Superseded Metropolis/mixed rows: {superseded}",
        f"- Canonical lambda=1, kappa=0.30, L=16 radial-heatbath rows found: {len(lam1_k030_canonical)}",
        "",
        "## Policy",
        "",
        "- `phi4_phase-diagram/` is the canonical configuration repository.",
        "- New production/training ensembles must use `embedded_wolff_sign_cluster_plus_radial_heatbath` and be stored under `phi4_phase-diagram/ensembles/`.",
        "- Mixed or Metropolis-generated ensembles are copied only for provenance under `phi4_phase-diagram/superseded_metropolis/`.",
        "- Parameter values inferred only from paths are marked non-canonical.",
        "- The decimated fill-in experiment must not substitute lambda/kappa points or unknown-generator files.",
        "",
        "## Lambda=1, Kappa=0.30 Status",
        "",
    ]
    if lam1_k030_canonical:
        lines.append("At least one metadata-verified canonical radial-heatbath candidate exists:")
        for row in lam1_k030_canonical:
            lines.append(f"- `{row['original_path']}` copied to `{row.get('central_path')}`")
    else:
        lines.append(
            "No metadata-verified canonical radial-heatbath lambda=1, kappa=0.30, L=16 ensemble was found. "
            "The decimated conditional fill-in experiment should remain blocked until one is generated under "
            "`phi4_phase-diagram/ensembles/lam1p000_kappa0p300_L16_embedded_wolff_sign_cluster_plus_radial_heatbath/`."
        )
    lines += [
        "",
        "## Superseded Mixed/Metropolis Examples",
        "",
    ]
    for row in [r for r in rows if r.get("is_superseded_metropolis") and r.get("field_like")][:20]:
        lines.append(
            f"- `{row['original_path']}`: generator `{row['generator_algorithm']}`, "
            f"lambda `{row.get('lambda')}`, kappa `{row.get('kappa')}`, copied `{row.get('central_path')}`"
        )
    (REPO / "config_inventory_report.md").write_text("\n".join(lines) + "\n")
    reports = REPO / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "config_inventory_report.md", reports / "config_inventory_report.md")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-copy", action="store_true", help="only write inventory; do not copy arrays")
    args = parser.parse_args()
    REPO.mkdir(parents=True, exist_ok=True)
    copied_at = now_iso()
    rows: list[dict[str, Any]] = []
    for path in discover_files():
        rows.extend(inspect_candidate(path))
    for manifest_path in sorted((REPO / "ensembles").glob("*/manifest.json")) if (REPO / "ensembles").exists() else []:
        row = inspect_existing_central_manifest(manifest_path)
        if row is not None:
            rows.append(row)
    if not args.no_copy:
        rows = [
            row if row.get("copy_status") == "central_existing" else copy_array_to_repository(row, copied_at)
            for row in rows
        ]
    write_csv(REPO / "config_inventory.csv", rows)
    write_json(REPO / "config_inventory.json", rows)
    write_report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
