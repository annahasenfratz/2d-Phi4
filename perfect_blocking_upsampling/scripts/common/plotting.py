from __future__ import annotations

from pathlib import Path


def save_pdf(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)

