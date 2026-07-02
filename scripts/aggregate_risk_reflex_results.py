#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


GROUP_KEYS = [
    "variant",
    "alpha",
    "beta",
    "gamma",
    "theta",
    "k",
    "lambda_brake",
    "lambda_throttle",
    "steering_limit_gain",
    "min_steering_scale",
    "trigger_threshold",
    "eval_episodes",
]

METRICS = [
    "SuccessRate",
    "EpCost",
    "CostRate",
    "mean_pain",
    "trigger_rate",
    "mean_longitudinal_delta",
    "steering_limit_rate",
]

REQUIRED_COLUMNS = set(GROUP_KEYS + ["seed"] + METRICS)


def normalize_value(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        return format(float(text), ".12g")
    except ValueError:
        return text


def validate_columns(path, fieldnames):
    field_set = set(fieldnames or [])
    missing = sorted(REQUIRED_COLUMNS - field_set)
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(missing)}")


def read_summary_rows(input_dir, skip_invalid=False):
    rows = []
    invalid_count = 0
    for path in sorted(Path(input_dir).rglob("*_summary.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            try:
                validate_columns(path, reader.fieldnames)
            except ValueError:
                if not skip_invalid:
                    raise
                invalid_count += 1
                continue
            for row in reader:
                item = dict(row)
                item["__source_path"] = str(path)
                rows.append(item)
    return rows, invalid_count


def ci95(values):
    if len(values) <= 1:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def seed_value(row):
    seed = row.get("seed", "")
    if str(seed).strip() != "":
        return normalize_value(seed)
    model_tag = row.get("model_tag", "")
    marker = "seed"
    if marker in model_tag:
        return model_tag.rsplit(marker, 1)[-1]
    return ""


def row_sort_key(row):
    path = Path(row["__source_path"])
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (mtime, str(path))


def dedupe_rows(rows, dedupe):
    seen = {}
    duplicates = defaultdict(list)
    for row in rows:
        key = tuple(normalize_value(row.get(name, "")) for name in GROUP_KEYS)
        seed = seed_value(row)
        dedupe_key = (key, seed)
        if dedupe_key in seen:
            duplicates[dedupe_key].extend([seen[dedupe_key], row])
            if dedupe == "latest":
                seen[dedupe_key] = max([seen[dedupe_key], row], key=row_sort_key)
        else:
            seen[dedupe_key] = row

    if duplicates and dedupe != "latest":
        messages = []
        for (key, seed), duplicate_rows in sorted(duplicates.items(), key=lambda item: (item[0][0], item[0][1])):
            sources = sorted({row["__source_path"] for row in duplicate_rows})
            group = ", ".join(f"{name}={value}" for name, value in zip(GROUP_KEYS, key))
            messages.append(f"group ({group}), seed={seed}: {'; '.join(sources)}")
        raise ValueError("duplicate rows for group/seed: " + " | ".join(messages))

    return list(seen.values())


def aggregate(rows, expected_seeds=0):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(normalize_value(row.get(name, "")) for name in GROUP_KEYS)
        grouped[key].append(row)

    output_rows = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        out = {name: value for name, value in zip(GROUP_KEYS, key)}
        seeds = {seed_value(row) for row in group_rows if seed_value(row) != ""}
        out["seed_count"] = len(seeds)
        out["source_count"] = len(group_rows)
        out["complete"] = "true" if expected_seeds > 0 and len(seeds) == expected_seeds else "false"
        for metric in METRICS:
            try:
                values = [float(row[metric]) for row in group_rows if row.get(metric, "") != ""]
            except ValueError as exc:
                sources = sorted({row["__source_path"] for row in group_rows})
                raise ValueError(f"invalid numeric value for {metric} in {'; '.join(sources)}") from exc
            out[f"{metric}_mean"] = mean(values) if values else ""
            out[f"{metric}_ci95"] = ci95(values) if values else ""
        output_rows.append(out)
    return output_rows


def main():
    parser = argparse.ArgumentParser(description="Aggregate QPSL risk-reflex screening summaries.")
    parser.add_argument("--input_dir", default="runs/risk_reflex_screen")
    parser.add_argument("--out", default="runs/tables/risk_reflex_screen_summary.csv")
    parser.add_argument("--skip_invalid", action="store_true")
    parser.add_argument("--dedupe", choices=["error", "latest"], default="error")
    parser.add_argument("--expected_seeds", default=0, type=int)
    args = parser.parse_args()

    rows, invalid_count = read_summary_rows(args.input_dir, skip_invalid=args.skip_invalid)
    rows = dedupe_rows(rows, args.dedupe)
    output_rows = aggregate(rows, expected_seeds=args.expected_seeds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = GROUP_KEYS + ["seed_count", "source_count", "complete"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_ci95"])

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"read {len(rows)} summary rows from {args.input_dir}")
    if invalid_count:
        print(f"skipped {invalid_count} invalid summary files")
    print(f"wrote {len(output_rows)} aggregate rows to {out_path}")


if __name__ == "__main__":
    main()
