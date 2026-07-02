#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


METRICS = [
    "success_rate",
    "clean_success_rate",
    "mean_cost_rate",
    "crash_rate",
    "out_of_road_rate",
]

SELECTED_FIELDS = [
    "algo",
    "selected_candidate",
    "baseline_success_rate",
    "nocicim_success_rate",
    "success_delta",
    "baseline_clean_success_rate",
    "nocicim_clean_success_rate",
    "clean_success_delta",
    "baseline_mean_cost_rate",
    "nocicim_mean_cost_rate",
    "cost_rate_delta",
    "baseline_crash_rate",
    "nocicim_crash_rate",
    "crash_delta",
    "baseline_out_of_road_rate",
    "nocicim_out_of_road_rate",
    "out_of_road_delta",
]

GATE_FIELDS = [
    "algo_count",
    "expected_algo_count",
    "failure_count",
    "all_success_non_regression",
    "mean_crash_reduced",
    "mean_clean_success_not_lower",
    "baseline_success_rate_mean",
    "nocicim_success_rate_mean",
    "baseline_clean_success_rate_mean",
    "nocicim_clean_success_rate_mean",
    "baseline_crash_rate_mean",
    "nocicim_crash_rate_mean",
    "baseline_mean_cost_rate_mean",
    "nocicim_mean_cost_rate_mean",
    "baseline_out_of_road_rate_mean",
    "nocicim_out_of_road_rate_mean",
]

FAILURE_FIELDS = ["algo", "reason", "baseline_success_rate", "candidate_count"]


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_algos(value):
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def metric_value(row, metric):
    if metric in row and row[metric] not in ("", None):
        return float(row[metric])
    mean_key = "{}_mean".format(metric)
    if mean_key in row and row[mean_key] not in ("", None):
        return float(row[mean_key])
    raise KeyError("missing metric {} in row for algo={} candidate={}".format(
        metric,
        row.get("algo", ""),
        row.get("candidate", ""),
    ))


def normalized_row(row):
    out = {
        "algo": str(row.get("algo", "")).strip().lower(),
        "candidate": str(row.get("candidate", "")).strip(),
    }
    for metric in METRICS:
        out[metric] = metric_value(row, metric)
    return out


def mean(values):
    values = list(values)
    return sum(values) / float(len(values)) if values else 0.0


def row_value(row, key, default=0.0):
    return float(row[key]) if key in row and row[key] not in ("", None) else float(default)


def select_portfolio_rows(rows, algos):
    normalized = [normalized_row(row) for row in rows]
    selected = []
    failures = []

    for algo in algos:
        algo_rows = [row for row in normalized if row["algo"] == algo]
        baseline_rows = [row for row in algo_rows if row["candidate"] == "none"]
        candidate_rows = [row for row in algo_rows if row["candidate"] != "none"]
        if not baseline_rows:
            failures.append({
                "algo": algo,
                "reason": "missing_baseline",
                "baseline_success_rate": "",
                "candidate_count": len(candidate_rows),
            })
            continue

        baseline = baseline_rows[-1]
        eligible = [
            row
            for row in candidate_rows
            if row["success_rate"] >= baseline["success_rate"]
        ]
        if not eligible:
            failures.append({
                "algo": algo,
                "reason": "no_success_non_regressing_candidate",
                "baseline_success_rate": baseline["success_rate"],
                "candidate_count": len(candidate_rows),
            })
            continue

        eligible.sort(
            key=lambda row: (
                row["crash_rate"],
                -row["clean_success_rate"],
                row["mean_cost_rate"],
                row["out_of_road_rate"],
                row["candidate"],
            )
        )
        best = eligible[0]
        selected.append({
            "algo": algo,
            "selected_candidate": best["candidate"],
            "baseline_success_rate": baseline["success_rate"],
            "nocicim_success_rate": best["success_rate"],
            "success_delta": best["success_rate"] - baseline["success_rate"],
            "baseline_clean_success_rate": baseline["clean_success_rate"],
            "nocicim_clean_success_rate": best["clean_success_rate"],
            "clean_success_delta": best["clean_success_rate"] - baseline["clean_success_rate"],
            "baseline_mean_cost_rate": baseline["mean_cost_rate"],
            "nocicim_mean_cost_rate": best["mean_cost_rate"],
            "cost_rate_delta": best["mean_cost_rate"] - baseline["mean_cost_rate"],
            "baseline_crash_rate": baseline["crash_rate"],
            "nocicim_crash_rate": best["crash_rate"],
            "crash_delta": best["crash_rate"] - baseline["crash_rate"],
            "baseline_out_of_road_rate": baseline["out_of_road_rate"],
            "nocicim_out_of_road_rate": best["out_of_road_rate"],
            "out_of_road_delta": best["out_of_road_rate"] - baseline["out_of_road_rate"],
        })

    return selected, failures


def compute_gates(selected_rows, algos, failures):
    selected_algos = {row["algo"] for row in selected_rows}
    expected_algos = set(algos)
    all_algos_present = selected_algos == expected_algos
    success_non_regression = all(
        row["nocicim_success_rate"] >= row["baseline_success_rate"]
        for row in selected_rows
    )
    baseline_crash = mean(row_value(row, "baseline_crash_rate") for row in selected_rows)
    nocicim_crash = mean(row_value(row, "nocicim_crash_rate") for row in selected_rows)
    baseline_clean = mean(row_value(row, "baseline_clean_success_rate") for row in selected_rows)
    nocicim_clean = mean(row_value(row, "nocicim_clean_success_rate") for row in selected_rows)

    return {
        "algo_count": len(selected_rows),
        "expected_algo_count": len(expected_algos),
        "failure_count": len(failures),
        "all_success_non_regression": int(bool(all_algos_present and not failures and success_non_regression)),
        "mean_crash_reduced": int(bool(selected_rows and nocicim_crash < baseline_crash)),
        "mean_clean_success_not_lower": int(bool(selected_rows and nocicim_clean >= baseline_clean)),
        "baseline_success_rate_mean": mean(row_value(row, "baseline_success_rate") for row in selected_rows),
        "nocicim_success_rate_mean": mean(row_value(row, "nocicim_success_rate") for row in selected_rows),
        "baseline_clean_success_rate_mean": baseline_clean,
        "nocicim_clean_success_rate_mean": nocicim_clean,
        "baseline_crash_rate_mean": baseline_crash,
        "nocicim_crash_rate_mean": nocicim_crash,
        "baseline_mean_cost_rate_mean": mean(row_value(row, "baseline_mean_cost_rate") for row in selected_rows),
        "nocicim_mean_cost_rate_mean": mean(row_value(row, "nocicim_mean_cost_rate") for row in selected_rows),
        "baseline_out_of_road_rate_mean": mean(row_value(row, "baseline_out_of_road_rate") for row in selected_rows),
        "nocicim_out_of_road_rate_mean": mean(row_value(row, "nocicim_out_of_road_rate") for row in selected_rows),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(out_dir, selected_rows, gates, failures):
    out_dir = Path(out_dir)
    write_csv(out_dir / "portfolio_selected.csv", selected_rows, SELECTED_FIELDS)
    write_csv(out_dir / "portfolio_gates.csv", [gates], GATE_FIELDS)
    write_csv(out_dir / "portfolio_failures.csv", failures, FAILURE_FIELDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--algos", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    algos = parse_algos(args.algos)
    rows = read_rows(args.input)
    selected, failures = select_portfolio_rows(rows, algos)
    gates = compute_gates(selected, algos, failures)
    write_outputs(args.outdir, selected, gates, failures)

    print("wrote {}".format(args.outdir))
    print("selected_algos={}/{}".format(len(selected), len(algos)))
    print("failures={}".format(len(failures)))
    print("all_success_non_regression={}".format(gates["all_success_non_regression"]))
    print("mean_crash_reduced={}".format(gates["mean_crash_reduced"]))
    print("mean_clean_success_not_lower={}".format(gates["mean_clean_success_not_lower"]))


if __name__ == "__main__":
    main()
