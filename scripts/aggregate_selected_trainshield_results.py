#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


TRAIN_METRICS = ["EpRet", "EpCost", "CostRate"]
PAIRED_METRICS = [
    "success_rate",
    "clean_success_rate",
    "mean_reward",
    "mean_episode_cost",
    "mean_cost_rate",
    "crash_rate",
    "out_of_road_rate",
    "out_of_time_rate",
    "intervention_ratio",
    "mean_action_distortion",
]


def maybe_stdev(values):
    return stdev(values) if len(values) > 1 else 0.0


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_algos(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def latest_logger(log_root, tag, algo, seed):
    pattern = "{}_{}_seed{}_SafeMetaDriveEnv-seed{}-*/logger.csv".format(tag, algo, seed, seed)
    matches = sorted(log_root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def matched_eval_dir(log_root, tag, algo, seed, suffix):
    return log_root / "eval_{}_{}_seed{}_{}".format(tag, algo, seed, suffix)


def summarize_training(project, tag, algos, seeds):
    log_root = project / "logs"
    grouped = defaultdict(lambda: defaultdict(list))
    sources = []
    missing = []

    for algo in algos:
        for seed in range(seeds):
            logger = latest_logger(log_root, tag, algo, seed)
            if logger is None:
                missing.append({"kind": "logger", "algo": algo, "seed": seed, "path": ""})
                continue
            rows = read_csv(logger)
            sources.append({"kind": "logger", "algo": algo, "seed": seed, "path": str(logger), "rows": len(rows)})
            for row in rows:
                step = int(float(row["total_steps"]))
                key = (algo, step)
                grouped[key]["seed"].append(seed)
                for metric in TRAIN_METRICS:
                    grouped[key][metric].append(float(row[metric]))

    out = []
    for algo, step in sorted(grouped):
        values = grouped[(algo, step)]
        row = {
            "algo": algo,
            "total_steps": step,
            "seed_count": len(set(values["seed"])),
        }
        for metric in TRAIN_METRICS:
            row["{}_mean".format(metric)] = mean(values[metric])
            row["{}_std".format(metric)] = maybe_stdev(values[metric])
        out.append(row)
    return out, sources, missing


def summarize_matched_eval(project, tag, algos, seeds, suffix):
    log_root = project / "logs"
    per_seed_rows = []
    sources = []
    missing = []

    for algo in algos:
        for seed in range(seeds):
            eval_dir = matched_eval_dir(log_root, tag, algo, seed, suffix)
            summary = eval_dir / "sweep_summary.csv"
            if not summary.exists():
                missing.append({"kind": "matched_eval", "algo": algo, "seed": seed, "path": str(summary)})
                continue
            rows = read_csv(summary)
            sources.append({"kind": "matched_eval", "algo": algo, "seed": seed, "path": str(summary), "rows": len(rows)})
            for row in rows:
                copied = dict(row)
                copied["algo"] = algo
                copied["train_seed"] = seed
                per_seed_rows.append(copied)

    grouped = defaultdict(lambda: defaultdict(list))
    for row in per_seed_rows:
        key = (row["algo"], row["candidate"])
        grouped[key]["seed"].append(int(row["train_seed"]))
        grouped[key]["frontend"].append(row.get("frontend", ""))
        for metric in PAIRED_METRICS:
            grouped[key][metric].append(float(row[metric]))

    summary_rows = []
    for algo, candidate in sorted(grouped):
        values = grouped[(algo, candidate)]
        row = {
            "algo": algo,
            "candidate": candidate,
            "frontend": values["frontend"][0] if values["frontend"] else "",
            "train_seed_count": len(set(values["seed"])),
        }
        for metric in PAIRED_METRICS:
            row["{}_mean".format(metric)] = mean(values[metric])
            row["{}_std".format(metric)] = maybe_stdev(values[metric])
        summary_rows.append(row)

    return per_seed_rows, summary_rows, sources, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--algos", required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--matched-eval-suffix", default="matched_20ep")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    project = args.project.resolve()
    algos = parse_algos(args.algos)
    out_dir = project / "runs" / "tables" / args.tag

    training_rows, train_sources, train_missing = summarize_training(project, args.tag, algos, args.seeds)
    per_seed_rows, matched_rows, matched_sources, matched_missing = summarize_matched_eval(
        project, args.tag, algos, args.seeds, args.matched_eval_suffix
    )
    missing = train_missing + matched_missing
    sources = train_sources + matched_sources

    if args.require_complete and missing:
        raise SystemExit("missing required inputs: {}".format(len(missing)))

    write_csv(
        out_dir / "selected_train_eval_summary.csv",
        training_rows,
        ["algo", "total_steps", "seed_count"]
        + ["{}_mean".format(metric) for metric in TRAIN_METRICS]
        + ["{}_std".format(metric) for metric in TRAIN_METRICS],
    )
    write_csv(
        out_dir / "selected_paired_eval_per_seed.csv",
        per_seed_rows,
        ["algo", "train_seed", "candidate", "frontend"] + PAIRED_METRICS,
    )
    write_csv(
        out_dir / "selected_paired_eval_summary.csv",
        matched_rows,
        ["algo", "candidate", "frontend", "train_seed_count"]
        + ["{}_mean".format(metric) for metric in PAIRED_METRICS]
        + ["{}_std".format(metric) for metric in PAIRED_METRICS],
    )
    write_csv(out_dir / "missing.csv", missing, ["kind", "algo", "seed", "path"])
    write_csv(out_dir / "sources.csv", sources, ["kind", "algo", "seed", "path", "rows"])

    print("wrote {}".format(out_dir))
    print("training_rows={}".format(len(training_rows)))
    print("paired_train_seed_rows={}".format(len(per_seed_rows)))
    print("missing={}".format(len(missing)))


if __name__ == "__main__":
    main()
