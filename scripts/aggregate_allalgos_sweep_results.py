#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = [
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


def parse_algos(value):
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def maybe_stdev(values):
    return stdev(values) if len(values) > 1 else 0.0


def summary_path(log_root, tag, algo, seed):
    return log_root / "{}_{}_seed{}".format(tag, algo, seed) / "sweep_summary.csv"


def collect_rows(log_root, tag, algos, seeds):
    rows = []
    sources = []
    missing = []

    for algo in algos:
        for seed in range(int(seeds)):
            path = summary_path(log_root, tag, algo, seed)
            if not path.exists():
                missing.append({"algo": algo, "train_seed": seed, "path": str(path)})
                continue
            source_rows = read_csv(path)
            sources.append({"algo": algo, "train_seed": seed, "path": str(path), "rows": len(source_rows)})
            for row in source_rows:
                copied = dict(row)
                copied["algo"] = algo
                copied["train_seed"] = seed
                rows.append(copied)

    return rows, sources, missing


def summarize(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    frontends = defaultdict(list)

    for row in rows:
        key = (row["algo"], row["candidate"])
        frontends[key].append(row.get("frontend", ""))
        grouped[key]["train_seed"].append(int(row["train_seed"]))
        for metric in METRICS:
            grouped[key][metric].append(float(row[metric]))

    out = []
    for algo, candidate in sorted(grouped):
        values = grouped[(algo, candidate)]
        row = {
            "algo": algo,
            "candidate": candidate,
            "frontend": frontends[(algo, candidate)][0] if frontends[(algo, candidate)] else "",
            "seed_count": len(set(values["train_seed"])),
        }
        for metric in METRICS:
            row[metric] = mean(values[metric])
            row["{}_std".format(metric)] = maybe_stdev(values[metric])
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--algos", default="td3,epo,rec,qpsl,lag,fac")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    project = args.project.resolve()
    log_root = args.input_root if args.input_root is not None else project / "logs"
    if not log_root.is_absolute():
        log_root = project / log_root
    algos = parse_algos(args.algos)
    rows, sources, missing = collect_rows(log_root, args.tag, algos, args.seeds)

    if args.require_complete and missing:
        raise SystemExit("missing required sweep summaries: {}".format(len(missing)))

    summary_rows = summarize(rows)
    metric_fields = []
    for metric in METRICS:
        metric_fields.extend([metric, "{}_std".format(metric)])

    write_csv(
        args.outdir / "allalgos_sweep_per_seed_summary.csv",
        rows,
        ["algo", "train_seed", "candidate", "frontend"] + METRICS,
    )
    write_csv(
        args.outdir / "allalgos_sweep_5seed_summary.csv",
        summary_rows,
        ["algo", "candidate", "frontend", "seed_count"] + metric_fields,
    )
    write_csv(args.outdir / "sources.csv", sources, ["algo", "train_seed", "path", "rows"])
    write_csv(args.outdir / "missing.csv", missing, ["algo", "train_seed", "path"])

    print("wrote {}".format(args.outdir))
    print("per_seed_rows={}".format(len(rows)))
    print("summary_rows={}".format(len(summary_rows)))
    print("missing={}".format(len(missing)))


if __name__ == "__main__":
    main()
