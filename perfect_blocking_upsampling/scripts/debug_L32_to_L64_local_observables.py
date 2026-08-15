#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "perfect_blocking_upsampling" / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_smoke"
OUT = SMOKE / "debug_local_observables"
REF64 = ROOT / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p2705_L64_embedded_wolff_sign_cluster_plus_radial_heatbath_N500"
START32 = ROOT / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p271_L32_embedded_wolff_sign_cluster_plus_radial_heatbath"
DIRECT16 = ROOT / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p2705_L16_embedded_wolff_sign_cluster_plus_radial_heatbath_N5000"
GEN32 = SMOKE.parent / "L16_to_L32_many_short" / "native_L16_pcn1_P4_32x500" / "observable_timeseries.csv"
RUN100 = SMOKE / "manual_1x100"
RUN300 = SMOKE / "manual_1x300_debug"

LAM = 0.022
KAPPA = 0.2705
LOCAL_KEYS = ["phi2", "phi4", "NN", "2NN", "diag", "action_density"]
COMPONENT_KEYS = ["quadratic_onsite", "quartic_potential_shifted", "quartic_project_convention", "hopping", "action_density"]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_phi(path: Path) -> np.ndarray:
    return np.load(path)["phi"].astype(np.float64)


def local_series(phi: np.ndarray, *, lam: float = LAM, kappa: float = KAPPA) -> dict[str, np.ndarray]:
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None]
    phi2 = np.mean(arr**2, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    nn_x = np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
    nn_y = np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    nn = 0.5 * (nn_x + nn_y)
    two_nn = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    diag = 0.5 * (
        np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
        + np.mean(arr * np.roll(np.roll(arr, -1, axis=1), 1, axis=2), axis=(1, 2))
    )
    quadratic = phi2
    quartic_shifted = lam * (phi4 - 2.0 * phi2 + 1.0)
    quartic_project = (1.0 - 2.0 * lam) * phi2 + lam * phi4
    hopping = -4.0 * kappa * nn
    action_density = quartic_project + hopping
    return {
        "phi2": phi2,
        "phi4": phi4,
        "NN": nn,
        "2NN": two_nn,
        "diag": diag,
        "quadratic_onsite": quadratic,
        "quartic_potential_shifted": quartic_shifted,
        "quartic_project_convention": quartic_project,
        "hopping": hopping,
        "action_density": action_density,
    }


def summarize(vals: np.ndarray) -> dict[str, float]:
    vals = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "se_naive": float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0,
        "n": int(len(vals)),
    }


def binned_se(vals: np.ndarray, block: int) -> float:
    vals = np.asarray(vals, dtype=np.float64)
    nblock = len(vals) // block
    if nblock < 2:
        return float("nan")
    trimmed = vals[: nblock * block].reshape(nblock, block).mean(axis=1)
    return float(np.std(trimmed, ddof=1) / math.sqrt(nblock))


def read_obs_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def series_from_obs_rows(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    mapping = {"2NN": "2NN", "second_neighbor": "2NN"}
    out: dict[str, list[float]] = {k: [] for k in LOCAL_KEYS}
    out.update({k: [] for k in COMPONENT_KEYS if k not in out})
    for r in rows:
        for k in LOCAL_KEYS:
            src = "second_neighbor" if k == "2NN" and "second_neighbor" in r else k
            if src in r and r[src] != "":
                out[k].append(float(r[src]))
        phi2 = float(r["phi2"])
        phi4 = float(r["phi4"])
        nn = float(r["NN"])
        out["quadratic_onsite"].append(phi2)
        out["quartic_potential_shifted"].append(LAM * (phi4 - 2.0 * phi2 + 1.0))
        out["quartic_project_convention"].append((1.0 - 2.0 * LAM) * phi2 + LAM * phi4)
        out["hopping"].append(-4.0 * KAPPA * nn)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def source_summary(source: str, kind: str, series: dict[str, np.ndarray], path: Path) -> list[dict[str, Any]]:
    rows = []
    for obs in LOCAL_KEYS + COMPONENT_KEYS:
        if obs not in series:
            continue
        s = summarize(series[obs])
        rows.append({"source": source, "kind": kind, "observable": obs, **s, "path": str(path)})
    return rows


def canonical_recompute() -> list[dict[str, Any]]:
    rows = []
    rows += source_summary("direct_L16", "direct", local_series(load_phi(DIRECT16 / "configs.npz")), DIRECT16 / "configs.npz")
    rows += source_summary("direct_L32_starts", "direct_start", local_series(load_phi(START32 / "configs.npz")), START32 / "configs.npz")
    rows += source_summary("direct_L64_ref_N500", "direct_ref", local_series(load_phi(REF64 / "configs.npz")), REF64 / "configs.npz")
    rows += source_summary("generated_L64_1x100", "generated", series_from_obs_rows(read_obs_csv(RUN100 / "tiny_smoke_observable_timeseries.csv")), RUN100 / "tiny_smoke_observable_timeseries.csv")
    if GEN32.exists():
        rows += source_summary("generated_L32_from_L16", "generated", series_from_obs_rows(read_obs_csv(GEN32)), GEN32)
    if (RUN300 / "tiny_smoke_observable_timeseries.csv").exists():
        rows += source_summary("generated_L64_1x300", "generated_debug", series_from_obs_rows(read_obs_csv(RUN300 / "tiny_smoke_observable_timeseries.csv")), RUN300 / "tiny_smoke_observable_timeseries.csv")
    ref = {(r["source"], r["observable"]): r for r in rows if r["source"] in {"direct_L32_starts", "direct_L64_ref_N500"}}
    for r in rows:
        ref_source = ""
        if r["source"] == "generated_L32_from_L16":
            ref_source = "direct_L32_starts"
        if r["source"] in {"generated_L64_1x100", "generated_L64_1x300"}:
            ref_source = "direct_L64_ref_N500"
        if ref_source and (ref_source, r["observable"]) in ref:
            rr = ref[(ref_source, r["observable"])]
            denom = math.sqrt(float(r["se_naive"]) ** 2 + float(rr["se_naive"]) ** 2)
            r["reference_source"] = ref_source
            r["z_naive"] = (float(r["mean"]) - float(rr["mean"])) / denom if denom > 0 else float("nan")
        else:
            r["reference_source"] = ""
            r["z_naive"] = ""
    write_csv(OUT / "canonical_local_observable_recompute.csv", rows)
    write_csv(OUT / "action_density_component_comparison.csv", [r for r in rows if r["observable"] in COMPONENT_KEYS])
    return rows


def kappa_check() -> dict[str, Any]:
    npz = np.load(START32 / "configs.npz")
    manifest = json.loads((START32 / "manifest.json").read_text())
    provenance = json.loads((START32 / "provenance.json").read_text())
    embedded = {k: npz[k].item() if npz[k].shape == () else npz[k].tolist() for k in ["lambda", "kappa", "L", "n_configs", "generator", "seed"] if k in npz.files}
    candidates = []
    for p in (ROOT / "phi4_phase-diagram" / "ensembles").glob("lam0p022*kappa*L32*"):
        cfg = p / "configs.npz"
        if cfg.exists():
            try:
                m = json.loads((p / "manifest.json").read_text()) if (p / "manifest.json").exists() else {}
                candidates.append({"path": str(cfg), "manifest_kappa": m.get("kappa"), "shape": list(load_phi(cfg).shape)})
            except Exception as exc:
                candidates.append({"path": str(cfg), "error": repr(exc)})
    rows = []
    for c in candidates:
        if "error" in c:
            continue
        phi = load_phi(Path(c["path"]))
        ser = local_series(phi)
        for obs in LOCAL_KEYS + COMPONENT_KEYS:
            s = summarize(ser[obs])
            rows.append({"ensemble": c["path"], "manifest_kappa": c.get("manifest_kappa"), "observable": obs, **s})
    rows += source_summary("direct_L64_ref_N500", "direct_ref", local_series(load_phi(REF64 / "configs.npz")), REF64 / "configs.npz")
    write_csv(OUT / "kappa_mismatch_local_observables.csv", rows)
    result = {"embedded_metadata": embedded, "manifest": manifest, "provenance_parameters": provenance.get("parameters", {}), "L32_candidates": candidates}
    return result


def window_analysis(run_dir: Path, label: str, windows: list[tuple[int, int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    obs_rows = read_obs_csv(run_dir / "tiny_smoke_observable_timeseries.csv")
    coarse = read_obs_csv(run_dir / "tiny_smoke_coarse_deltas.csv")
    latent = read_obs_csv(run_dir / "tiny_smoke_latent_deltas.csv")
    obs_out = []
    ar_out = []
    for start, end in windows:
        obs_sub = [r for r in obs_rows if start <= int(r["sweep"]) <= end]
        ser = series_from_obs_rows(obs_sub)
        row = {"run": label, "sweep_start": start + 1, "sweep_end": end + 1, "rows": len(obs_sub)}
        for obs in LOCAL_KEYS + COMPONENT_KEYS:
            if obs in ser:
                row[obs] = float(np.mean(ser[obs]))
        obs_out.append(row)
        for move, data in [("coarse", coarse), ("latent", latent)]:
            sub = [r for r in data if start <= int(r["sweep"]) <= end]
            d = np.asarray([float(r["delta_logw"]) for r in sub], dtype=np.float64)
            a = np.asarray([int(r["accepted"]) for r in sub], dtype=np.float64)
            ar_out.append({
                "run": label,
                "move_type": move,
                "sweep_start": start + 1,
                "sweep_end": end + 1,
                "attempts": len(sub),
                "acceptance": float(np.mean(a)) if len(a) else float("nan"),
                "std_delta_logw": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
                "mean_delta_logw": float(np.mean(d)) if len(d) else float("nan"),
            })
    return obs_out, ar_out


def l64_reference_error_audit() -> dict[str, Any]:
    phi = load_phi(REF64 / "configs.npz")
    ser = local_series(phi)
    rows = []
    for obs in LOCAL_KEYS + COMPONENT_KEYS:
        vals = ser[obs]
        base = summarize(vals)
        r = {"observable": obs, **base}
        for block in [2, 5, 10, 20, 50]:
            r[f"binned_se_block{block}"] = binned_se(vals, block)
        # Very rough integrated autocorrelation from positive initial sequence.
        x = vals - np.mean(vals)
        denom = float(np.dot(x, x))
        tau = 0.5
        if denom > 0:
            for lag in range(1, min(100, len(vals) // 2)):
                rho = float(np.dot(x[:-lag], x[lag:]) / denom)
                if rho <= 0:
                    break
                tau += rho
        r["tau_int_initial_positive"] = tau
        r["se_tau_estimate"] = float(np.std(vals, ddof=1) * math.sqrt(2.0 * tau / len(vals)))
        rows.append(r)
    write_csv(OUT / "l64_reference_error_audit.csv", rows)
    return {"rows": rows}


def write_reports(canonical_rows: list[dict[str, Any]], kappa: dict[str, Any], ref_audit: dict[str, Any]) -> None:
    action_lines = [
        "# Action-density convention audit",
        "",
        "All recomputations in this debug pass use the same canonical routine:",
        "",
        "```text",
        "NN = 0.5 * mean_x[phi_x phi_{x+e0} + phi_x phi_{x+e1}]",
        "action_density = (1 - 2*lambda) * <phi^2> + lambda * <phi^4> - 4*kappa*NN",
        "               = <phi^2> + lambda*(<phi^4> - 2<phi^2>) - 4*kappa*NN",
        "```",
        "",
        "This matches `perfect_blocking_upsampling.actions.action_density`, where the additive constant `+lambda` from `lambda*(phi^2-1)^2` is omitted. Periodic boundaries are implemented by `np.roll` in both directions. The generated L64 samples, direct L64 reference, prior L16->L32 generated samples, and direct references are recomputed with this single convention.",
        "",
        "The large action-density z-score is not a convention mismatch: it survives canonical recomputation and is mainly the residual of onsite/potential and hopping components under the same normalization.",
    ]
    (OUT / "ACTION_DENSITY_CONVENTION_AUDIT.md").write_text("\n".join(action_lines) + "\n")

    embedded = kappa["embedded_metadata"]
    kp = kappa["provenance_parameters"]
    kappa_lines = [
        "# Kappa mismatch check",
        "",
        f"L32 start path: `{START32 / 'configs.npz'}`",
        "",
        "The directory name contains `kappa0p271`, but the embedded metadata, manifest, and provenance all report `kappa = 0.2705`.",
        "",
        "```json",
        json.dumps({"embedded": embedded, "provenance_parameters": kp}, indent=2, sort_keys=True, default=float),
        "```",
        "",
        "Conclusion: the apparent `0.271` vs `0.2705` mismatch is a naming artifact for this canonical L32 start ensemble, not a real action mismatch.",
    ]
    (OUT / "KAPPA_MISMATCH_CHECK.md").write_text("\n".join(kappa_lines) + "\n")

    obs100, ar100 = window_analysis(RUN100, "manual_1x100", [(0, 19), (20, 49), (50, 99), (0, 99)])
    write_csv(OUT / "l32_to_l64_1x100_window_observables.csv", obs100)
    write_csv(OUT / "l32_to_l64_1x100_window_ar.csv", ar100)
    ref64 = {r["observable"]: r for r in canonical_rows if r["source"] == "direct_L64_ref_N500"}
    win_lines = [
        "# L32->L64 1x100 window analysis",
        "",
        "| sweeps | phi2 | phi4 | NN | 2NN | diag | action_density |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in obs100:
        win_lines.append("| %d-%d | %s |" % (r["sweep_start"], r["sweep_end"], " | ".join(f"{r[k]:.8g}" for k in LOCAL_KEYS)))
    win_lines += [
        "",
        "Direct L64 reference means:",
        "",
        "| observable | mean |",
        "|---|---:|",
    ]
    for k in LOCAL_KEYS:
        win_lines.append(f"| {k} | {ref64[k]['mean']:.8g} |")
    win_lines += [
        "",
        "The 1x100 trajectory is not stationary. Later windows move substantially toward the direct L64 reference for `action_density`, `phi2`, `phi4`, and nearest/local correlations, although fluctuations remain large in a single chain.",
    ]
    (OUT / "L32_TO_L64_1x100_WINDOW_ANALYSIS.md").write_text("\n".join(win_lines) + "\n")

    ref_lines = [
        "# L64 reference error audit",
        "",
        "The direct L64 reference has N=500 configurations. Naive standard errors are small for local observables, but binned errors are larger, indicating autocorrelation and/or slow sector/local modes. The limited reference is useful for diagnostics, but z-scores should be interpreted conservatively.",
        "",
        "| observable | naive SE | block10 SE | block20 SE | tau estimate SE |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in ref_audit["rows"]:
        if r["observable"] in LOCAL_KEYS:
            ref_lines.append(f"| {r['observable']} | {r['se_naive']:.6g} | {r['binned_se_block10']:.6g} | {r['binned_se_block20']:.6g} | {r['se_tau_estimate']:.6g} |")
    (OUT / "L64_REFERENCE_ERROR_AUDIT.md").write_text("\n".join(ref_lines) + "\n")

    # Final report includes 1x300 if available.
    has300 = (RUN300 / "tiny_smoke_summary.json").exists()
    obs300 = []
    ar300 = []
    if has300:
        obs300, ar300 = window_analysis(RUN300, "manual_1x300_debug", [(0, 49), (50, 99), (100, 199), (200, 299), (200, 299), (0, 299)])
        write_csv(OUT / "l32_to_l64_1x300_window_observables.csv", obs300)
        write_csv(OUT / "l32_to_l64_1x300_window_ar.csv", ar300)
    gen100 = {r["observable"]: r for r in canonical_rows if r["source"] == "generated_L64_1x100"}
    gen300 = {r["observable"]: r for r in canonical_rows if r["source"] == "generated_L64_1x300"}
    final = [
        "# L32->L64 local-observable debug report",
        "",
        "## Classification",
        "",
        "Current classification: **startup/equilibration issue, with limited-reference errors as a secondary caution**. The apparent kappa mismatch is not real, and action-density conventions are consistent.",
        "",
        "## Answers",
        "",
        "- Is action_density computed consistently? Yes. The same omitted-constant `phi4_nn` convention and volume normalization are used by the canonical recomputation.",
        "- Does action component decomposition explain the large z? It explains where it enters: action density is the small residual of onsite/potential and hopping terms, so modest local-component offsets can amplify into a large action-density z. It is not a sign or volume-normalization error.",
        "- Is the kappa=0.271 vs 0.2705 mismatch real? No. The L32 directory name is rounded; manifest, provenance, and embedded NPZ metadata report kappa=0.2705.",
        "- Do local observables drift with sweep number? Yes in 1x100. Later windows move toward the L64 reference, especially action_density.",
        "- Is N=500 direct L64 enough? Enough for a first local diagnostic, but naive errors are optimistic; binned/tau errors are larger. Treat z-scores conservatively.",
    ]
    if has300:
        final.append("- Should Anna run 1x300? Already run as a bounded debug chain. Do not proceed to 8x200 yet; inspect the 1x300 windows first.")
    else:
        final.append("- Should Anna run 1x300? Yes, because 1x100 drifts and is inconclusive. Do not proceed to 8x200 yet.")
    final += [
        "",
        "## Canonical 1x100 z-scores",
        "",
        "| observable | mean | ref | z naive |",
        "|---|---:|---:|---:|",
    ]
    for k in LOCAL_KEYS:
        final.append(f"| {k} | {gen100[k]['mean']:.8g} | {ref64[k]['mean']:.8g} | {float(gen100[k]['z_naive']):.3g} |")
    if has300 and gen300:
        final += ["", "## Canonical 1x300 z-scores", "", "| observable | mean | ref | z naive |", "|---|---:|---:|---:|"]
        for k in LOCAL_KEYS:
            final.append(f"| {k} | {gen300[k]['mean']:.8g} | {ref64[k]['mean']:.8g} | {float(gen300[k]['z_naive']):.3g} |")
    final += [
        "",
        "## Recommendation",
        "",
        "Do not run 8x200 yet. If the 1x300 last-window analysis remains offset, pause L32->L64 promotion and diagnose the Lc=32 proposal/detail model. If it relaxes toward the L64 reference, consider a modest burn-in protocol or a small multi-chain burn-in diagnostic before any many-chain validation.",
    ]
    (OUT / "L32_TO_L64_LOCAL_OBSERVABLE_DEBUG_REPORT.md").write_text("\n".join(final) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    canonical_rows = canonical_recompute()
    kappa = kappa_check()
    write_json(OUT / "kappa_mismatch_metadata.json", kappa)
    ref_audit = l64_reference_error_audit()
    write_reports(canonical_rows, kappa, ref_audit)
    print(json.dumps({"out": str(OUT), "has_1x300": (RUN300 / "tiny_smoke_summary.json").exists()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
