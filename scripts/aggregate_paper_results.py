#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ALGOS = ("td3", "epo", "lag", "qpsl", "recovery", "fac")


def latest_logger(log_root: Path, tag: str, algo: str, seed: int) -> Path | None:
    pattern = f"{tag}_{algo}_seed{seed}_SafeMetaDriveEnv-seed{seed}-*/logger.csv"
    matches = sorted(log_root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(
                {
                    "total_steps": int(float(row["total_steps"])),
                    "EpRet": float(row["EpRet"]),
                    "EpCost": float(row["EpCost"]),
                    "CostRate": float(row["CostRate"]),
                }
            )
        return rows


def maybe_stdev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="paper_1m_20260629")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    args = parser.parse_args()

    log_root = args.project / "logs"
    out_dir = args.project / "runs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}_summary.csv"

    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sources = []

    for algo in ALGOS:
        for seed in range(args.seeds):
            logger = latest_logger(log_root, args.tag, algo, seed)
            if logger is None:
                continue
            rows = read_rows(logger)
            sources.append((algo, seed, str(logger), len(rows)))
            for row in rows:
                key = (algo, row["total_steps"])
                grouped[key]["EpRet"].append(row["EpRet"])
                grouped[key]["EpCost"].append(row["EpCost"])
                grouped[key]["CostRate"].append(row["CostRate"])

    fieldnames = [
        "algo",
        "total_steps",
        "seed_count",
        "EpRet_mean",
        "EpRet_std",
        "EpCost_mean",
        "EpCost_std",
        "CostRate_mean",
        "CostRate_std",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for algo, step in sorted(grouped, key=lambda x: (ALGOS.index(x[0]), x[1])):
            values = grouped[(algo, step)]
            writer.writerow(
                {
                    "algo": algo,
                    "total_steps": step,
                    "seed_count": len(values["EpRet"]),
                    "EpRet_mean": mean(values["EpRet"]),
                    "EpRet_std": maybe_stdev(values["EpRet"]),
                    "EpCost_mean": mean(values["EpCost"]),
                    "EpCost_std": maybe_stdev(values["EpCost"]),
                    "CostRate_mean": mean(values["CostRate"]),
                    "CostRate_std": maybe_stdev(values["CostRate"]),
                }
            )

    print(f"wrote {out_path}")
    print("sources:")
    for algo, seed, path, rows in sources:
        print(f"{algo},seed={seed},rows={rows},{path}")


if __name__ == "__main__":
    main()
