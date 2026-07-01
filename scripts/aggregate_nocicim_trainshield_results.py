#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRIC_KEYS = [
    "EpRet",
    "EpCost",
    "CostRate",
]

PAIRED_METRIC_KEYS = [
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

EPISODE_KEYS = [
    "success",
    "clean_success",
    "reward",
    "episode_cost",
    "cost_rate",
    "any_crash",
    "out_of_road",
    "out_of_time",
    "intervention_ratio",
    "mean_action_distortion",
]

EPISODE_OUTPUT_NAMES = {
    "success": "success_rate",
    "clean_success": "clean_success_rate",
    "reward": "mean_reward",
    "episode_cost": "mean_episode_cost",
    "cost_rate": "mean_cost_rate",
    "any_crash": "crash_rate",
    "out_of_road": "out_of_road_rate",
    "out_of_time": "out_of_time_rate",
    "intervention_ratio": "intervention_ratio",
    "mean_action_distortion": "mean_action_distortion",
}


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


def latest_logger(log_root, tag, seed):
    pattern = "{}_qpsl_seed{}_SafeMetaDriveEnv-seed{}-*/logger.csv".format(tag, seed, seed)
    matches = sorted(log_root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def paired_eval_dir(log_root, tag, seed, episodes):
    return log_root / "eval_{}_qpsl_seed{}_{}ep".format(tag, seed, episodes)


def summarize_training(project, tag, seeds):
    log_root = project / "logs"
    grouped = defaultdict(lambda: defaultdict(list))
    sources = []
    missing = []

    for seed in range(seeds):
        logger = latest_logger(log_root, tag, seed)
        if logger is None:
            missing.append("seed{} logger".format(seed))
            continue
        rows = read_csv(logger)
        sources.append({"kind": "logger", "seed": seed, "path": str(logger), "rows": len(rows)})
        for row in rows:
            step = int(float(row["total_steps"]))
            for key in METRIC_KEYS:
                grouped[step][key].append(float(row[key]))

    out = []
    for step in sorted(grouped):
        values = grouped[step]
        row = {"total_steps": step, "seed_count": len(values["EpRet"])}
        for key in METRIC_KEYS:
            row["{}_mean".format(key)] = mean(values[key])
            row["{}_std".format(key)] = maybe_stdev(values[key])
        out.append(row)
    return out, sources, missing


def summarize_paired(project, tag, seeds, episodes):
    log_root = project / "logs"
    per_seed_rows = []
    episode_rows = []
    sources = []
    missing = []

    for seed in range(seeds):
        eval_dir = paired_eval_dir(log_root, tag, seed, episodes)
        aggregate = eval_dir / "aggregate_matrix.csv"
        per_episode = eval_dir / "per_episode_matrix.csv"
        if not aggregate.exists():
            missing.append("seed{} aggregate_matrix".format(seed))
            continue
        aggregate_rows = read_csv(aggregate)
        sources.append({"kind": "paired_aggregate", "seed": seed, "path": str(aggregate), "rows": len(aggregate_rows)})
        for row in aggregate_rows:
            row = dict(row)
            row["train_seed"] = seed
            per_seed_rows.append(row)
        if per_episode.exists():
            rows = read_csv(per_episode)
            sources.append({"kind": "paired_episode", "seed": seed, "path": str(per_episode), "rows": len(rows)})
            for row in rows:
                row = dict(row)
                row["train_seed"] = seed
                episode_rows.append(row)
        else:
            missing.append("seed{} per_episode_matrix".format(seed))

    seed_grouped = defaultdict(lambda: defaultdict(list))
    for row in per_seed_rows:
        mode = row["mode"]
        seed_grouped[mode]["train_seed"].append(float(row["train_seed"]))
        for key in PAIRED_METRIC_KEYS:
            seed_grouped[mode][key].append(float(row[key]))

    per_train_seed_summary = []
    for mode in sorted(seed_grouped):
        values = seed_grouped[mode]
        row = {
            "mode": mode,
            "train_seed_count": len(values["train_seed"]),
        }
        for key in PAIRED_METRIC_KEYS:
            row["{}_mean_across_train_seeds".format(key)] = mean(values[key])
            row["{}_std_across_train_seeds".format(key)] = maybe_stdev(values[key])
        per_train_seed_summary.append(row)

    episode_grouped = defaultdict(lambda: defaultdict(list))
    for row in episode_rows:
        mode = row["mode"]
        episode_grouped[mode]["train_seed"].append(float(row["train_seed"]))
        for key in EPISODE_KEYS:
            episode_grouped[mode][key].append(float(row[key]))

    episode_summary = []
    for mode in sorted(episode_grouped):
        values = episode_grouped[mode]
        row = {
            "mode": mode,
            "train_seed_count": len(set(int(v) for v in values["train_seed"])),
            "episodes_total": len(values["success"]),
        }
        for key in EPISODE_KEYS:
            row[EPISODE_OUTPUT_NAMES[key]] = mean(values[key])
        episode_summary.append(row)

    return per_seed_rows, per_train_seed_summary, episode_summary, sources, missing


def latest_baseline_qpsl(project, baseline_summary, step):
    path = baseline_summary
    if not path.is_absolute():
        path = project / path
    if not path.exists():
        return None
    rows = read_csv(path)
    candidates = [
        row
        for row in rows
        if row.get("algo") == "qpsl" and int(float(row.get("total_steps", -1))) == int(step)
    ]
    return candidates[-1] if candidates else None


def write_comparison(out_dir, training_summary, paired_episode_summary, baseline_row):
    rows = []
    final_train = training_summary[-1] if training_summary else None
    if final_train:
        row = {
            "comparison": "train_eval_nocicim_vs_original_qpsl",
            "nocicim_success": final_train["EpRet_mean"],
            "nocicim_cost": final_train["EpCost_mean"],
            "baseline_success": "",
            "baseline_cost": "",
            "delta_success": "",
            "delta_cost": "",
            "gate_success_not_lower_and_cost_lower": "",
        }
        if baseline_row:
            baseline_success = float(baseline_row["EpRet_mean"])
            baseline_cost = float(baseline_row["EpCost_mean"])
            row["baseline_success"] = baseline_success
            row["baseline_cost"] = baseline_cost
            row["delta_success"] = final_train["EpRet_mean"] - baseline_success
            row["delta_cost"] = final_train["EpCost_mean"] - baseline_cost
            row["gate_success_not_lower_and_cost_lower"] = int(
                row["delta_success"] >= 0.0 and row["delta_cost"] < 0.0
            )
        rows.append(row)

    by_mode = {row["mode"]: row for row in paired_episode_summary}
    none_row = by_mode.get("qpsl+none")
    nocicim_row = by_mode.get("qpsl+nocicim_risk_field")
    if none_row and nocicim_row:
        delta_success = float(nocicim_row["success_rate"]) - float(none_row["success_rate"])
        delta_cost = float(nocicim_row["mean_episode_cost"]) - float(none_row["mean_episode_cost"])
        rows.append(
            {
                "comparison": "paired_eval_nocicim_vs_same_policy_none",
                "nocicim_success": nocicim_row["success_rate"],
                "nocicim_cost": nocicim_row["mean_episode_cost"],
                "baseline_success": none_row["success_rate"],
                "baseline_cost": none_row["mean_episode_cost"],
                "delta_success": delta_success,
                "delta_cost": delta_cost,
                "gate_success_not_lower_and_cost_lower": int(delta_success >= 0.0 and delta_cost < 0.0),
            }
        )

    write_csv(
        out_dir / "nocicim_qpsl_comparison.csv",
        rows,
        [
            "comparison",
            "nocicim_success",
            "nocicim_cost",
            "baseline_success",
            "baseline_cost",
            "delta_success",
            "delta_cost",
            "gate_success_not_lower_and_cost_lower",
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--tag", default="nocicim_qpsl_trainshield_1m_20260701")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--paired-eval-episodes", type=int, default=20)
    parser.add_argument("--baseline-summary", type=Path, default=Path("runs/tables/paper_1m_20260629_summary.csv"))
    parser.add_argument("--baseline-step", type=int, default=1000000)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    project = args.project
    out_dir = project / "runs" / "tables" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    training_summary, training_sources, training_missing = summarize_training(project, args.tag, args.seeds)
    paired_rows, paired_seed_summary, paired_episode_summary, paired_sources, paired_missing = summarize_paired(
        project, args.tag, args.seeds, args.paired_eval_episodes
    )
    missing = training_missing + paired_missing

    write_csv(
        out_dir / "train_eval_summary.csv",
        training_summary,
        [
            "total_steps",
            "seed_count",
            "EpRet_mean",
            "EpRet_std",
            "EpCost_mean",
            "EpCost_std",
            "CostRate_mean",
            "CostRate_std",
        ],
    )
    write_csv(
        out_dir / "paired_eval_per_train_seed.csv",
        paired_rows,
        ["train_seed", "mode", "episodes"] + PAIRED_METRIC_KEYS,
    )
    write_csv(
        out_dir / "paired_eval_train_seed_summary.csv",
        paired_seed_summary,
        ["mode", "train_seed_count"]
        + [
            "{}_{}".format(key, suffix)
            for key in PAIRED_METRIC_KEYS
            for suffix in ("mean_across_train_seeds", "std_across_train_seeds")
        ],
    )
    write_csv(
        out_dir / "paired_eval_episode_summary.csv",
        paired_episode_summary,
        ["mode", "train_seed_count", "episodes_total"]
        + [EPISODE_OUTPUT_NAMES[key] for key in EPISODE_KEYS],
    )
    write_csv(
        out_dir / "sources.csv",
        training_sources + paired_sources,
        ["kind", "seed", "path", "rows"],
    )
    write_csv(out_dir / "missing.csv", [{"missing": item} for item in missing], ["missing"])

    baseline = latest_baseline_qpsl(project, args.baseline_summary, args.baseline_step)
    write_comparison(out_dir, training_summary, paired_episode_summary, baseline)

    print("wrote {}".format(out_dir))
    print("training_rows={}".format(len(training_summary)))
    print("paired_train_seed_rows={}".format(len(paired_rows)))
    print("missing={}".format(len(missing)))
    for item in missing:
        print("missing: {}".format(item))
    if args.require_complete and missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
