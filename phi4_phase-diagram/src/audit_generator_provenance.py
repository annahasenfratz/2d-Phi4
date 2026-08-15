#!/usr/bin/env python3
"""Audit finite-lambda phi4 generator provenance and downstream use."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "phi4_phase-diagram"
REPORTS = REPO / "reports"
CANONICAL_GENERATOR = "embedded_wolff_sign_cluster_plus_radial_heatbath"
DIAGNOSTIC_MIXED_GENERATOR = "embedded_wolff_sign_cluster_plus_local_metropolis_amplitude"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


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
        writer.writerows(rows)


def classify_manifest(manifest: dict[str, Any]) -> tuple[str, str]:
    generator = str(manifest.get("generator") or "").lower()
    sign_update = str(manifest.get("sign_update") or "").lower()
    amplitude_update = str(manifest.get("amplitude_update") or "").lower()
    local_metropolis = manifest.get("local_metropolis_used")
    radial_heatbath = amplitude_update == "radial_heatbath" or manifest.get("radial_heatbath_used") is True
    parameter_status = str(manifest.get("parameter_status") or "").lower()

    if (
        generator == CANONICAL_GENERATOR
        and sign_update == "embedded_wolff_cluster"
        and amplitude_update == "radial_heatbath"
        and local_metropolis is False
    ):
        return "canonical", "embedded Wolff sign clusters plus radial heat-bath amplitudes; local Metropolis false"
    if generator == DIAGNOSTIC_MIXED_GENERATOR or (
        "wolff" in generator and "metropolis" in generator
    ):
        return "diagnostic_noncanonical", "embedded Wolff sign update with local Metropolis amplitude updates"
    if generator in {"metropolis", "local_metropolis_only"}:
        return "superseded", "local Metropolis-only or Metropolis-labeled ensemble"
    if generator == "mixed":
        return "superseded", "mixed generator label without canonical radial heat-bath metadata"
    if generator in {"unknown", "", "none"}:
        if parameter_status in {"ambiguous", "", "path_only_unverified"}:
            return "requires_manual_inspection", "generator and/or parameters are not determined by metadata"
        return "superseded", "unknown generator under current canonical rules"
    if radial_heatbath and "wolff" in generator and local_metropolis is False:
        return "requires_manual_inspection", "mentions radial heat bath but generator label/sign-update metadata are nonstandard"
    return "requires_manual_inspection", "generator metadata is insufficient or nonstandard"


def manifest_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dirs = []
    for base_name in ["ensembles", "superseded_metropolis", "superseded_or_diagnostic"]:
        base = REPO / base_name
        if base.exists():
            dirs.extend(sorted(base.glob("*/manifest.json")))
    for manifest_path in dirs:
        manifest = read_json(manifest_path) or {}
        provenance = read_json(manifest_path.parent / "provenance.json") or {}
        classification, reason = classify_manifest(manifest)
        source_path = manifest.get("source_path") or provenance.get("source_path")
        row = {
            "central_manifest": str(manifest_path.relative_to(ROOT)),
            "central_directory": str(manifest_path.parent.relative_to(ROOT)),
            "original_source_path": source_path,
            "lambda": manifest.get("lambda"),
            "kappa": manifest.get("kappa"),
            "L": manifest.get("L"),
            "n_configs": manifest.get("n_configs"),
            "shape": json.dumps(manifest.get("shape")),
            "generator_label": manifest.get("generator"),
            "sign_update_method": manifest.get("sign_update"),
            "amplitude_radial_update_method": manifest.get("amplitude_update"),
            "local_metropolis_used": manifest.get("local_metropolis_used"),
            "radial_heatbath_used": (manifest.get("amplitude_update") == "radial_heatbath") or manifest.get("radial_heatbath_used"),
            "parameter_status": manifest.get("parameter_status"),
            "is_canonical_manifest": manifest.get("is_canonical", manifest.get("canonical")),
            "production_use": manifest.get("production_use"),
            "classification": classification,
            "classification_reason": reason,
            "downstream_output_dirs": "",
            "downstream_use_count": 0,
        }
        rows.append(row)
    return rows


def text_files_under(root: Path) -> list[Path]:
    if not root.exists():
        return []
    exts = {".json", ".md", ".txt", ".csv", ".yaml", ".yml", ".py"}
    out: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        if any(part in {".git", "__pycache__", "checkpoints"} for part in path.parts):
            continue
        out.append(path)
    return out


def run_dirs_to_scan() -> list[Path]:
    roots = [
        ROOT / "ML_sampling_clean",
        ROOT / "MIT_NN_test",
        ROOT / "InverseBlocking_MIT_NF",
        ROOT / "testing_mlneuralsampler_multilevel",
        ROOT / "perfect_blocking",
        ROOT / "inverse_blocking_flow",
    ]
    run_dirs: set[Path] = set()
    markers = {"summary.json", "config.json", "input_manifest.json", "report.md", "generation_metadata.json", "manifest.json"}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name in markers:
                run_dirs.add(path.parent)
    return sorted(run_dirs)


def build_source_patterns(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    patterns: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        for key in ["original_source_path", "central_directory"]:
            val = row.get(key)
            if not val:
                continue
            sval = str(val)
            patterns.append((sval, row))
            try:
                patterns.append((str(Path(sval).relative_to(ROOT)), row))
            except Exception:
                pass
        central_configs = ROOT / str(row["central_directory"]) / "configs.npz"
        patterns.append((str(central_configs), row))
        patterns.append((str(central_configs.relative_to(ROOT)), row))
    # Longest first reduces accidental partial path matches.
    return sorted(set((p, id(r)) for p, r in patterns), key=lambda x: len(x[0]), reverse=True)  # type: ignore[arg-type]


def downstream_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    id_to_row = {id(r): r for r in rows}
    patterns = [(p, id_to_row[rid]) for p, rid in build_source_patterns(rows)]
    downstream: list[dict[str, Any]] = []
    uses: dict[int, set[str]] = defaultdict(set)
    for run_dir in run_dirs_to_scan():
        files = text_files_under(run_dir)
        text_chunks = []
        for p in files:
            try:
                text_chunks.append(p.read_text(errors="ignore")[:200000])
            except Exception:
                pass
        text = "\n".join(text_chunks)
        matched_rows: list[dict[str, Any]] = []
        for pat, row in patterns:
            if pat and pat in text:
                matched_rows.append(row)
        # Fallback heuristics for major runs whose reports omit exact file paths.
        low_text = text.lower()
        heuristic_class = None
        heuristic_reason = None
        if not matched_rows:
            if "embedded_wolff_sign_cluster_plus_local_metropolis_amplitude" in low_text:
                heuristic_class = "diagnostic_noncanonical"
                heuristic_reason = "run text mentions embedded Wolff plus local Metropolis amplitude generator"
            elif "local metropolis" in low_text or "metropolis" in low_text:
                heuristic_class = "superseded"
                heuristic_reason = "run text mentions Metropolis and no canonical source path was found"
            elif "lambda=0.022" in low_text or "lam0p022" in low_text or "lambda0p022" in low_text:
                heuristic_class = "requires_manual_inspection"
                heuristic_reason = "lambda=0.022 run with no exact source/generator path in report"
            elif "lambda=1" in low_text or "lam1" in low_text:
                heuristic_class = "requires_manual_inspection"
                heuristic_reason = "lambda=1 run with no exact source/generator path in report"
        if matched_rows:
            seen: set[str] = set()
            for row in matched_rows:
                key = row["central_directory"]
                if key in seen:
                    continue
                seen.add(key)
                uses[id(row)].add(str(run_dir.relative_to(ROOT)))
                cls = row["classification"]
                downstream.append(
                    {
                        "downstream_output_dir": str(run_dir.relative_to(ROOT)),
                        "input_ensemble_path_if_known": row.get("original_source_path") or row.get("central_directory"),
                        "central_ensemble": row["central_directory"],
                        "ensemble_classification": cls,
                        "conclusion_status": conclusion_status(cls),
                        "independent_observable_agreement": infer_observable_agreement(text),
                        "notes": row["classification_reason"],
                    }
                )
        elif heuristic_class is not None:
            downstream.append(
                {
                    "downstream_output_dir": str(run_dir.relative_to(ROOT)),
                    "input_ensemble_path_if_known": None,
                    "central_ensemble": None,
                    "ensemble_classification": heuristic_class,
                    "conclusion_status": conclusion_status(heuristic_class),
                    "independent_observable_agreement": infer_observable_agreement(text),
                    "notes": heuristic_reason,
                }
            )
    for row in rows:
        dirs = sorted(uses.get(id(row), set()))
        row["downstream_output_dirs"] = ";".join(dirs)
        row["downstream_use_count"] = len(dirs)
    return downstream


def conclusion_status(classification: str) -> str:
    if classification == "canonical":
        return "safe under current generator rule, subject to ordinary statistical checks"
    if classification == "diagnostic_noncanonical":
        return "qualitative/diagnostic only; mark noncanonical pending radial-heatbath regeneration"
    if classification == "superseded":
        return "superseded for production/training conclusions under current rule"
    return "requires manual inspection before physics conclusions"


def infer_observable_agreement(text: str) -> str:
    low = text.lower()
    tokens = ["binder", "xi/l", "phi2", "phi4", "nn2", "action", "observable", "comparison"]
    found = [t for t in tokens if t in low]
    if len(found) >= 3:
        return "report contains observable comparisons; inspect before promoting conclusions"
    if found:
        return "limited observable mentions"
    return "not identifiable from text scan"


def write_reports(rows: list[dict[str, Any]], downstream: list[dict[str, Any]]) -> None:
    counts = defaultdict(int)
    for row in rows:
        counts[row["classification"]] += 1
    lines = [
        "# Generator Provenance Audit",
        "",
        "Canonical finite-lambda generator rule:",
        "",
        "`embedded_wolff_sign_cluster_plus_radial_heatbath` is canonical only when sign updates are embedded Wolff clusters, amplitude updates are radial heat bath, and `local_metropolis_used=false`.",
        "",
        "Embedded Wolff sign clusters alone are incomplete for continuous finite-lambda phi4 because amplitudes must update. Embedded Wolff plus local Metropolis amplitude updates is diagnostic/noncanonical and superseded for new production/training.",
        "",
        "## Ensemble Classification Counts",
        "",
    ]
    for key in ["canonical", "diagnostic_noncanonical", "superseded", "requires_manual_inspection"]:
        lines.append(f"- {key}: {counts[key]}")
    lines += [
        "",
        "## Lambda=1 Kappa=0.30 Notes",
        "",
    ]
    lam1 = [
        r
        for r in rows
        if str(r.get("lambda")) in {"1.0", "1"} and str(r.get("kappa")) in {"0.3", "0.30"}
    ]
    if lam1:
        for row in lam1[:40]:
            lines.append(
                f"- `{row['central_directory']}`: {row['classification']}; generator `{row.get('generator_label')}`; "
                f"L={row.get('L')}; n={row.get('n_configs')}; source `{row.get('original_source_path')}`"
            )
    else:
        lines.append("No lambda=1, kappa=0.30 ensemble manifests found.")
    lines += [
        "",
        "## Policy Consequences",
        "",
        "- Existing local-Metropolis or mixed local-Metropolis amplitude ensembles should remain available for provenance only.",
        "- Downstream results based on noncanonical ensembles can remain useful as qualitative diagnostics, but should be marked noncanonical pending regeneration.",
        "- No run should be promoted to production physics solely from an ambiguous or unknown-generator ensemble.",
    ]
    (REPORTS / "generator_provenance_audit.md").write_text("\n".join(lines) + "\n")

    dlines = [
        "# Downstream Run Provenance Audit",
        "",
        "This scan is heuristic: it searches report/config/manifest text for copied central paths, original source paths, and generator labels. Runs without exact source paths are marked for manual inspection.",
        "",
        "## Summary",
        "",
    ]
    dcounts = defaultdict(int)
    for row in downstream:
        dcounts[row["ensemble_classification"]] += 1
    for key in ["canonical", "diagnostic_noncanonical", "superseded", "requires_manual_inspection"]:
        dlines.append(f"- {key}: {dcounts[key]}")
    dlines += ["", "## Major Matched Runs", ""]
    major_terms = ["decimated", "lambda0p022", "lambda_0p022", "lambda_1p0", "perfect", "inverse", "MIT_NN_test", "finite_lambda"]
    shown = 0
    for row in downstream:
        path = row["downstream_output_dir"]
        if any(t.lower() in path.lower() for t in major_terms):
            dlines.append(
                f"- `{path}`: {row['ensemble_classification']}; {row['conclusion_status']}; "
                f"input `{row['input_ensemble_path_if_known']}`; {row['notes']}"
            )
            shown += 1
            if shown >= 160:
                break
    (REPORTS / "downstream_run_provenance_audit.md").write_text("\n".join(dlines) + "\n")


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = manifest_rows()
    downstream = downstream_audit(rows)
    write_csv(REPORTS / "generator_provenance_audit.csv", rows)
    write_json(REPORTS / "generator_provenance_audit.json", {"ensembles": rows, "downstream_runs": downstream})
    write_reports(rows, downstream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
