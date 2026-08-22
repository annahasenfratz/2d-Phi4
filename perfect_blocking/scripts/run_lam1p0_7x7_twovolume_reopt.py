#!/usr/bin/env python3
"""Two-volume D4-symmetric 7x7 reoptimization, seeded by the final 5x5 kernel.

The 5x5 kernel is embedded in 7x7 support with K30=K31=K32=K33=0, but all
ten orbit classes are free during the search.  L32->L16 screens cheaply;
L64->L32 participates in the second-stage and full-sample ranking.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "perfect_blocking") not in sys.path:
    sys.path.insert(0, str(ROOT / "perfect_blocking"))
from scripts.common.blocking import load_configs
from scripts.run_lam1p0_7x7_kernel_search import (  # noqa: E402
    CLASS_MULT, ETA_SCALE, block, full_metrics, matrix_from_classes,
    momentum_extrema, observable_arrays,
)

LAM = ROOT / "perfect_blocking/perfect_blocking_lam1p0"
ORBIT_KEYS = ("00", "10", "11", "20", "21", "22", "30", "31", "32", "33")
KEYS = ("action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "G_pmin_avg")


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=LAM / "tests/intermediate/7x7_twovolume_from5x5")
    p.add_argument("--kernel", type=Path, default=LAM / "kernels/final/chosen_kernel.json")
    p.add_argument("--seed", type=int, default=20260817)
    p.add_argument("--fast-n", type=int, default=750)
    p.add_argument("--cross-n", type=int, default=1500)
    p.add_argument("--stage1", type=int, default=640)
    p.add_argument("--local-per-center", type=int, default=16)
    p.add_argument("--centers", type=int, default=32)
    p.add_argument("--full-count", type=int, default=20)
    p.add_argument("--finalize-existing", action="store_true",
                   help="write the best-kernel JSON from an existing ranked CSV; do not rerun")
    return p.parse_args()


def normalise(c: dict[str, float]) -> dict[str, float]:
    out = {k: float(c.get(k, 0.0)) for k in ORBIT_KEYS}
    out["00"] = 1.0 - sum(CLASS_MULT[k] * out[k] for k in ORBIT_KEYS if k != "00")
    return out


def seed_from_5x5(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return normalise(data["base_orbit_classes_before_eta_scale"])


def perturb(rng: np.random.Generator, center: dict[str, float], scale: float) -> dict[str, float]:
    out = dict(center)
    for key in ORBIT_KEYS[1:]:
        # Smaller outer-shell perturbations retain a useful 5x5 starting point,
        # while keeping every coefficient free.
        width = scale * (0.35 if key in {"30", "31", "32", "33"} else 1.0)
        out[key] += float(rng.normal(0.0, width))
    return normalise(out)


def evaluate(classes: dict[str, float], direct16, fine32, direct32=None, fine64=None):
    matrix = ETA_SCALE * matrix_from_classes(classes)
    mom = momentum_extrema(matrix, grid=384)
    rows16 = full_metrics(direct16, observable_arrays(block(fine32, matrix)))
    rows32 = None if direct32 is None else full_metrics(direct32, observable_arrays(block(fine64, matrix)))
    return rows16, rows32, mom, matrix


def level_score(rows: dict[str, dict[str, float]], mom: dict[str, float]) -> float:
    score = (8 * rows["phi2"]["ks_statistic"] + 2 * rows["phi4"]["ks_statistic"]
             + 3 * rows["local_kurtosis_ratio"]["ks_statistic"] + 2 * rows["action_density"]["ks_statistic"]
             + 2 * rows["NN"]["ks_statistic"] + rows["G_pmin_avg"]["ks_statistic"])
    for obs, limit in {"action_density": .05, "phi2": .08, "phi4": .05, "NN": .05,
                       "m2": .04, "m4": .04, "G_pmin_avg": .04}.items():
        score += 2000 * max(0.0, rows[obs]["ks_statistic"] - limit) ** 2
    score += 5000 * max(0.0, 0.45 - mom["min_K"]) ** 2
    score += 1000 * max(0.0, mom["max_inverse_K"] - 1.60) ** 2
    return float(score)


def record(name, stage, classes, rows16, rows32, mom, score):
    r = {"candidate": name, "stage": stage, "score": score,
         **{f"K{k}": classes[k] for k in ORBIT_KEYS}, **mom}
    r["condition_number"] = mom["max_K"] / mom["min_K"]
    for level, rows in (("L32toL16", rows16), ("L64toL32", rows32)):
        if rows is None:
            continue
        for obs in KEYS:
            r[f"{level}_{obs}_KS"] = rows[obs]["ks_statistic"]
            r[f"{level}_{obs}_std_ratio"] = rows[obs]["std_ratio_a_over_b"]
    return r


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def finalize_existing(out: Path) -> Path:
    """Recover the JSON artifact from a completed ranking without recomputation."""
    ranked = out / "two_volume_ranked.csv"
    with ranked.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No candidates in {ranked}")
    best = min(rows, key=lambda row: float(row["score"]))
    classes = {k: float(best[f"K{k}"]) for k in ORBIT_KEYS}
    matrix = ETA_SCALE * matrix_from_classes(classes)
    stability = {k: float(best[k]) for k in ("min_K", "max_K", "min_inverse_K", "max_inverse_K")}
    stability["condition_number"] = stability["max_K"] / stability["min_K"]
    artifact = {
        "name": "best_7x7_two_volume_eta_included",
        "family": "7x7_D4_from_zero_padded_5x5",
        "lambda": 1.0, "kappa_f": .340301, "kappa_c": .340301,
        "eta": .25, "eta_scale_numeric": ETA_SCALE,
        "kernel_coefficients_include_eta_scale": True,
        "base_orbit_classes_before_eta_scale": classes,
        "base_matrix_before_eta_scale": (matrix / ETA_SCALE).tolist(),
        "matrix": matrix.tolist(), "momentum_stability": stability,
        "selection_record": best,
    }
    target = out / "best_7x7_two_volume_eta_included.json"
    target.write_text(json.dumps(artifact, indent=2) + "\n")
    return target


def main() -> None:
    a = cli(); a.out.mkdir(parents=True, exist_ok=True)
    if a.finalize_existing:
        print(finalize_existing(a.out))
        return
    rng, base = np.random.default_rng(a.seed), seed_from_5x5(a.kernel)
    d16 = observable_arrays(load_configs(ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"))
    f32 = load_configs(ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz")
    f64 = load_configs(ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz")
    d16_fast, f32_fast = {k:v[:a.fast_n] for k,v in d16.items()}, f32[:a.fast_n]
    pool = [("seed_5x5_padded", base)] + [(f"stage1_{i:04d}", perturb(rng, base, float(rng.choice([.0005,.001,.002,.004])))) for i in range(a.stage1)]
    fast = []
    for name, c in pool:
        r16, _, mom, _ = evaluate(c, d16_fast, f32_fast)
        fast.append((level_score(r16, mom), name, c))
    fast.sort(key=lambda x:x[0])
    for j, (_, _, center) in enumerate(fast[:a.centers]):
        for i in range(a.local_per_center):
            pool.append((f"stage2_{j:02d}_{i:03d}", perturb(rng, center, .0008 if i < 12 else .0016)))
    refined = []
    for name, c in pool[len(fast):]:
        r16, _, mom, _ = evaluate(c, d16_fast, f32_fast)
        refined.append((level_score(r16, mom), name, c))
    # Cross-volume ranking of the best L16 candidates; this prevents another
    # kernel that only succeeds at L32->L16 from being promoted.
    cross = []
    for _, name, c in sorted(fast + refined, key=lambda x:x[0])[:120]:
        r16, r32, mom, _ = evaluate(c, {k:v[:a.cross_n] for k,v in d16.items()}, f32[:a.cross_n], observable_arrays(f32[:a.cross_n]), f64[:a.cross_n])
        cross.append((level_score(r16, mom) + level_score(r32, mom), name, c))
    cross.sort(key=lambda x:x[0]); records=[]
    for score, name, c in cross[:a.full_count]:
        r16, r32, mom, matrix = evaluate(c, d16, f32, observable_arrays(f32), f64)
        total = level_score(r16, mom) + level_score(r32, mom)
        records.append(record(name, "full", c, r16, r32, mom, total))
    records.sort(key=lambda r:r["score"]); write_csv(a.out / "two_volume_ranked.csv", records)
    print(finalize_existing(a.out))

if __name__ == "__main__": main()
