from __future__ import annotations

from typing import Any


def weighted_operator_score(rows: list[dict[str, Any]], weights: dict[str, float] | None = None) -> float:
    weights = weights or {}
    total = 0.0
    for row in rows:
        obs = str(row.get("observable", ""))
        w = float(weights.get(obs, 1.0))
        total += w * abs(float(row.get("standardized_mean_shift", 0.0)))
    return total
