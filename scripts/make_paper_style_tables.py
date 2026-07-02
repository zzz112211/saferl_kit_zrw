#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


SAFE_ALGOS = [
    ("qpsl", "Safety Layer"),
    ("recovery", "Recovery RL"),
    ("lag", "Lagrangian"),
    ("fac", "FAC"),
    ("epo", "EPO"),
]
BASELINE_ALGOS = [("td3", "TD3")]

METRICS = [
    ("EpRet", "SuccessRate", "max"),
    ("EpCost", "Ep-Cost", "min"),
    ("CostRate", "CostRate", "min"),
]


def read_summary(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        return {(row["algo"], int(row["total_steps"])): row for row in rows}


def ci95(row: dict[str, str], metric: str) -> float:
    n = int(row["seed_count"])
    return 1.96 * float(row[f"{metric}_std"]) / math.sqrt(n)


def format_value(row: dict[str, str], metric: str, precision: int) -> str:
    mean = float(row[f"{metric}_mean"])
    ci = ci95(row, metric)
    return f"{mean:.{precision}f} +/- {ci:.{precision}f}"


def format_md_value(row: dict[str, str], metric: str, precision: int, bold: bool) -> str:
    value = format_value(row, metric, precision).replace("+/-", "±")
    return f"**{value}**" if bold else value


def format_tex_value(row: dict[str, str], metric: str, precision: int, bold: bool) -> str:
    value = format_value(row, metric, precision).replace("+/-", r"\pm")
    return rf"\textbf{{{value}}}" if bold else value


def best_algos(summary: dict[tuple[str, int], dict[str, str]], algos, step: int, metric: str, mode: str) -> set[str]:
    values = {
        algo: float(summary[(algo, step)][f"{metric}_mean"])
        for algo, _ in algos
        if (algo, step) in summary
    }
    if not values:
        return set()
    target = max(values.values()) if mode == "max" else min(values.values())
    return {algo for algo, value in values.items() if abs(value - target) < 1e-12}


def write_csv(summary, algos, step: int, out_path: Path) -> None:
    fieldnames = ["Environment", "Metric"] + [label for _, label in algos]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric, label, _ in METRICS:
            precision = 2 if metric != "CostRate" else 3
            row = {"Environment": "MetaDrive", "Metric": label}
            for algo, column in algos:
                row[column] = format_value(summary[(algo, step)], metric, precision)
            writer.writerow(row)


def write_markdown(summary, algos, step: int, out_path: Path, title: str) -> None:
    lines = [
        f"# {title}",
        "",
        f"Step: {step:,}. Values are mean ± normal 95% confidence over 5 seeds.",
        "",
        "| Environment | Metric | " + " | ".join(label for _, label in algos) + " |",
        "|---|---|" + "|".join("---" for _ in algos) + "|",
    ]
    for metric, label, mode in METRICS:
        precision = 2 if metric != "CostRate" else 3
        best = best_algos(summary, algos, step, metric, mode)
        values = [
            format_md_value(summary[(algo, step)], metric, precision, algo in best)
            for algo, _ in algos
        ]
        lines.append("| MetaDrive | " + label + " | " + " | ".join(values) + " |")
    lines.append("")
    lines.append("Notes: Safety Layer corresponds to the repository's `--use_qpsl`; `SuccessRate` is logged as `EpRet` in `train_metadrive.py` for MetaDrive.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(summary, algos, step: int, out_path: Path, caption: str, label: str) -> None:
    columns = "ll" + "c" * len(algos)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        "Environment & Metric & " + " & ".join(name for _, name in algos) + r" \\",
        r"\midrule",
    ]
    for index, (metric, label_text, mode) in enumerate(METRICS):
        precision = 2 if metric != "CostRate" else 3
        best = best_algos(summary, algos, step, metric, mode)
        env = "MetaDrive" if index == 0 else ""
        values = [
            format_tex_value(summary[(algo, step)], metric, precision, algo in best)
            for algo, _ in algos
        ]
        lines.append(env + " & " + label_text + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("runs/tables/paper_1m_20260629_summary.csv"))
    parser.add_argument("--step", type=int, default=1_000_000)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/tables"))
    args = parser.parse_args()

    summary = read_summary(args.summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(summary, SAFE_ALGOS, args.step, args.out_dir / "paper_1m_20260629_table3_metadrive.csv")
    write_markdown(
        summary,
        SAFE_ALGOS,
        args.step,
        args.out_dir / "paper_1m_20260629_table3_metadrive.md",
        "Paper-Style MetaDrive Results",
    )
    write_latex(
        summary,
        SAFE_ALGOS,
        args.step,
        args.out_dir / "paper_1m_20260629_table3_metadrive.tex",
        "Mean performance with normal 95\\% confidence for safety-aware algorithms on MetaDrive.",
        "tab:saferlkit_metadrive",
    )

    write_csv(summary, BASELINE_ALGOS, args.step, args.out_dir / "paper_1m_20260629_td3_reference.csv")
    write_markdown(
        summary,
        BASELINE_ALGOS,
        args.step,
        args.out_dir / "paper_1m_20260629_td3_reference.md",
        "TD3 Reference MetaDrive Results",
    )
    print(f"wrote paper-style tables to {args.out_dir}")


if __name__ == "__main__":
    main()
