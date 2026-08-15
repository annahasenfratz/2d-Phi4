#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import load_config
from perfect_blocking_upsampling.train import train_from_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    result = train_from_config(args.config)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

