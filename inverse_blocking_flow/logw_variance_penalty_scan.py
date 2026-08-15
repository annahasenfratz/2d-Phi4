"""Train reverse-KL flows with log-weight variance penalties and scan kappa."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", type=str, default="0.01,0.05,0.1,0.5")
    parser.add_argument("--fine-size", type=int, default=32)
    parser.add_argument("--kappa-true", type=float, default=0.31)
    parser.add_argument("--lambda-f", dest="lam", type=float, default=1.0)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-eval", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs/fine_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs/logw_var_penalty"))
    parser.add_argument("--init-checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs/conditional_detail_flow_mle.pt"))
    parser.add_argument("--seed", type=int, default=515151)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--reuse", action="store_true", help="reuse existing tagged checkpoints when present")
    return parser


def alpha_tag(alpha: float) -> str:
    return f"logwvar_alpha_{alpha:g}".replace(".", "p")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def fmt(x: object) -> str:
    if x is None:
        return "missing"
    try:
        return f"{float(x):.6g}"
    except (TypeError, ValueError):
        return str(x)


def write_report(path: Path, summary: dict[str, object]) -> None:
    baseline = summary["baseline"]
    rows = summary["alpha_results"]
    lines = [
        "# Logw Variance Penalty Scan",
        "",
        "All kappa scans use empirically blocked coarse fields `phi_c = B(phi_f_true)`. No approximate coarse action is used.",
        "",
        "Training loss for each alpha is `reverse_KL_loss + alpha * mean((logw - mean(logw))^2)` with `kappa_true=0.31`, `lambda=1.0`.",
        "",
        "## Baseline",
        "",
        f"- checkpoint: `{baseline['checkpoint']}`",
        f"- kappa_min by logw width: `{fmt(baseline['preferred_kappa_by_logw_width'])}`",
        f"- std(logw) at minimum: `{fmt(baseline['min_logw_std'])}`",
        f"- ESS/N at minimum: `{fmt(baseline['min_ess_over_n'])}`",
        f"- aggregate observable error at minimum: `{fmt(baseline['min_aggregate_error'])}`",
        "",
        "## Alpha Results",
        "",
        "| alpha | kappa_min width | logw std min | ESS/N min | A/R proxy min | kappa_min obs | agg err obs min | agg err at true | S mean at true | phi2 at true | NN at true | high-p at true | checkpoint |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        true_row = row["row_at_kappa_true"]
        obs = true_row["observables"] if true_row else {}
        lines.append(
            f"| {fmt(row['alpha'])} | {fmt(row['preferred_kappa_by_logw_width'])} | {fmt(row['min_logw_std'])} | "
            f"{fmt(row['min_ess_over_n'])} | {fmt(row['min_ar_proxy'])} | {fmt(row['preferred_kappa_by_observables'])} | "
            f"{fmt(row['best_observable_aggregate_error'])} | {fmt(row['aggregate_error_at_kappa_true'])} | "
            f"{fmt(obs.get('S_mean'))} | {fmt(obs.get('phi2'))} | {fmt(obs.get('NN_corr'))} | {fmt(obs.get('high_p_power'))} | "
            f"`{row['checkpoint']}` |"
        )
    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            f"- Kappa minimum moves toward 0.31: {summary['answers']['kappa_min_moves_toward_true']}",
            f"- Best logw std improvement: {summary['answers']['best_logw_std_improvement']}",
            f"- Best ESS/N improvement: {summary['answers']['best_ess_improvement']}",
            f"- Observables remain reasonable: {summary['answers']['observables_remain_reasonable']}",
            f"- Recommended checkpoint from this scan: `{summary['answers']['recommended_checkpoint']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    python = sys.executable
    baseline_scan_path = Path("inverse_blocking_flow/outputs/logw_kappa_scan_blocked_summary.json")
    if not baseline_scan_path.exists():
        run([python, "inverse_blocking_flow/logw_kappa_scan_blocked_coarse.py"])
    baseline_scan = json.loads(baseline_scan_path.read_text())
    baseline_best = baseline_scan["best_by_logw_width"]
    baseline_obs_best = baseline_scan["best_by_observable_error"]
    baseline = {
        "checkpoint": baseline_scan["setup"]["checkpoint"],
        "preferred_kappa_by_logw_width": baseline_best["kappa_f"],
        "min_logw_std": baseline_best["logw"]["std_logw_centered"],
        "min_ess_over_n": baseline_best["logw"]["ess_over_n"],
        "min_aggregate_error": baseline_obs_best["aggregate_abs_rel_error_vs_true_kappa_true"],
    }

    rows = []
    for alpha in alphas:
        tag = alpha_tag(alpha)
        checkpoint = args.output_dir / f"conditional_detail_flow_{tag}.pt"
        if not (args.reuse and checkpoint.exists()):
            run(
                [
                    python,
                    "inverse_blocking_flow/train_conditional_flow.py",
                    "--mode",
                    "reverse_kl",
                    "--fine-size",
                    str(args.fine_size),
                    "--kappa-fine",
                    str(args.kappa_true),
                    "--lambda",
                    str(args.lam),
                    "--n-configs",
                    str(args.n_configs),
                    "--data-path",
                    str(args.data_path),
                    "--output-dir",
                    str(args.output_dir),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--lr",
                    str(args.lr),
                    "--layers",
                    str(args.layers),
                    "--hidden-channels",
                    str(args.hidden_channels),
                    "--cnn-depth",
                    str(args.cnn_depth),
                    "--logw-var-alpha",
                    str(alpha),
                    "--checkpoint-tag",
                    tag,
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                ]
                + (["--checkpoint", str(args.init_checkpoint)] if args.init_checkpoint.exists() else [])
            )
        scan_dir = args.output_dir / f"scan_{tag}"
        scan_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                python,
                "inverse_blocking_flow/logw_kappa_scan_blocked_coarse.py",
                "--checkpoint",
                str(checkpoint),
                "--data-path",
                str(args.data_path),
                "--output-dir",
                str(scan_dir),
                "--fine-size",
                str(args.fine_size),
                "--kappa-true",
                str(args.kappa_true),
                "--lambda-f",
                str(args.lam),
                "--n-configs",
                str(args.n_configs),
                "--n-eval",
                str(args.n_eval),
                "--layers",
                str(args.layers),
                "--hidden-channels",
                str(args.hidden_channels),
                "--cnn-depth",
                str(args.cnn_depth),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
            ]
        )
        scan = json.loads((scan_dir / "logw_kappa_scan_blocked_summary.json").read_text())
        best_width = scan["best_by_logw_width"]
        best_obs = scan["best_by_observable_error"]
        true_rows = [row for row in scan["scan"] if abs(row["kappa_f"] - args.kappa_true) < 1e-12]
        row = {
            "alpha": alpha,
            "checkpoint": str(checkpoint),
            "scan_dir": str(scan_dir),
            "preferred_kappa_by_logw_width": best_width["kappa_f"],
            "min_logw_std": best_width["logw"]["std_logw_centered"],
            "min_ess_over_n": best_width["logw"]["ess_over_n"],
            "min_ar_proxy": best_width["logw"]["independence_acceptance_proxy"],
            "preferred_kappa_by_observables": best_obs["kappa_f"],
            "best_observable_aggregate_error": best_obs["aggregate_abs_rel_error_vs_true_kappa_true"],
            "row_at_kappa_true": true_rows[0] if true_rows else None,
            "aggregate_error_at_kappa_true": true_rows[0]["aggregate_abs_rel_error_vs_true_kappa_true"] if true_rows else None,
        }
        rows.append(row)

    best_std = min(rows, key=lambda row: row["min_logw_std"])
    best_ess = max(rows, key=lambda row: row["min_ess_over_n"])
    best_obs = min(rows, key=lambda row: row["best_observable_aggregate_error"])
    baseline_distance = abs(float(baseline["preferred_kappa_by_logw_width"]) - args.kappa_true)
    best_distance = abs(float(best_std["preferred_kappa_by_logw_width"]) - args.kappa_true)
    reasonable = [
        row for row in rows
        if row["best_observable_aggregate_error"] is not None and row["best_observable_aggregate_error"] < 0.15
    ]
    recommended = min(reasonable or rows, key=lambda row: (row["min_logw_std"], row["best_observable_aggregate_error"]))
    summary = {
        "setup": {
            "alphas": alphas,
            "epochs": args.epochs,
            "fine_size": args.fine_size,
            "kappa_true": args.kappa_true,
            "lambda": args.lam,
            "uses_approximate_coarse_action": False,
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint.exists() else None,
        },
        "baseline": baseline,
        "alpha_results": rows,
        "answers": {
            "kappa_min_moves_toward_true": (
                f"{best_distance < baseline_distance}; baseline distance {baseline_distance:.6g}, best penalized distance {best_distance:.6g} from alpha {best_std['alpha']:.6g}."
            ),
            "best_logw_std_improvement": (
                f"alpha {best_std['alpha']:.6g}: std {best_std['min_logw_std']:.6g} vs baseline {baseline['min_logw_std']:.6g}."
            ),
            "best_ess_improvement": (
                f"alpha {best_ess['alpha']:.6g}: ESS/N {best_ess['min_ess_over_n']:.6g} vs baseline {baseline['min_ess_over_n']:.6g}."
            ),
            "observables_remain_reasonable": (
                f"best observable alpha {best_obs['alpha']:.6g}: aggregate error {best_obs['best_observable_aggregate_error']:.6g}."
            ),
            "recommended_checkpoint": recommended["checkpoint"],
        },
    }
    (args.output_dir / "logw_variance_penalty_scan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(args.output_dir / "logw_variance_penalty_scan_report.md", summary)
    print(f"wrote {args.output_dir / 'logw_variance_penalty_scan_summary.json'}")
    print(f"wrote {args.output_dir / 'logw_variance_penalty_scan_report.md'}")
    print("recommended", recommended["checkpoint"])


if __name__ == "__main__":
    main()
