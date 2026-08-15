#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perfect_blocking_upsampling import __version__


def main() -> int:
    print(json.dumps({
        "package_version": __version__,
        "python": sys.version,
        "root": str(ROOT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

