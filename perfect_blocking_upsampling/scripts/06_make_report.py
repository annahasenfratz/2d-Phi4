#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT, load_config, resolve_run_paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve_run_paths(cfg)["output_dir"]
    summary = json.loads((out / "summary.json").read_text())
    report = f"""# Sample Reproduction Report\n\n- proposal std(logw): {summary['proposal_std_logw']}\n- proposal ESS/N: {summary['proposal_ess_over_n']}\n- proposal phi4: {summary['proposal_phi4']}\n- proposal NN: {summary['proposal_NN']}\n- proposal action density: {summary['proposal_action_density']}\n- proposal Binder_U4: {summary['proposal_Binder_U4']}\n- proposal xi/L: {summary['proposal_xi_over_L']}\n"""
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "sample_reproduction_report.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
