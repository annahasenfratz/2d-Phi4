from __future__ import annotations

from pathlib import Path


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoint_dir = run_dir / "checkpoints"
    candidates = sorted(checkpoint_dir.glob("checkpoint_latest.*"))
    if candidates:
        return candidates[-1]
    sweep_candidates = sorted(checkpoint_dir.glob("checkpoint_sweep_*.*"))
    return sweep_candidates[-1] if sweep_candidates else None


def require_latest_checkpoint(run_dir: Path) -> Path:
    path = latest_checkpoint(run_dir)
    if path is None:
        raise FileNotFoundError(f"no checkpoint_latest.* or checkpoint_sweep_*.* under {run_dir / 'checkpoints'}")
    return path

