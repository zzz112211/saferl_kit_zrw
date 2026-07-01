import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AGGREGATE_SCRIPT = REPO_ROOT / "scripts" / "aggregate_nocicim_trainshield_results.py"


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class AggregateNocicimTrainShieldResultsTest(unittest.TestCase):
    def test_comparison_uses_latest_complete_step_and_matching_baseline_by_default(self):
        tag = "unit_nocicim_qpsl"
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            baseline_path = project / "runs" / "tables" / "paper_1m_20260629_summary.csv"
            write_csv(
                baseline_path,
                [
                    "algo",
                    "total_steps",
                    "seed_count",
                    "EpRet_mean",
                    "EpRet_std",
                    "EpCost_mean",
                    "EpCost_std",
                    "CostRate_mean",
                    "CostRate_std",
                ],
                [
                    {
                        "algo": "qpsl",
                        "total_steps": "50000",
                        "seed_count": "5",
                        "EpRet_mean": "0.45",
                        "EpRet_std": "0",
                        "EpCost_mean": "19.5",
                        "EpCost_std": "0",
                        "CostRate_mean": "0.04",
                        "CostRate_std": "0",
                    },
                    {
                        "algo": "qpsl",
                        "total_steps": "60000",
                        "seed_count": "5",
                        "EpRet_mean": "0.48",
                        "EpRet_std": "0",
                        "EpCost_mean": "17.75",
                        "EpCost_std": "0",
                        "CostRate_mean": "0.04",
                        "CostRate_std": "0",
                    },
                ],
            )

            for seed in range(5):
                rows = [
                    {"total_steps": "50000", "EpRet": "0.33", "EpCost": "5.28", "CostRate": "0.0085"}
                ]
                if seed < 2:
                    rows.append(
                        {"total_steps": "60000", "EpRet": "0.50", "EpCost": "6.0", "CostRate": "0.009"}
                    )
                logger = (
                    project
                    / "logs"
                    / "{}_qpsl_seed{}_SafeMetaDriveEnv-seed{}-0".format(tag, seed, seed)
                    / "logger.csv"
                )
                write_csv(logger, ["total_steps", "EpRet", "EpCost", "CostRate"], rows)

            subprocess.run(
                [
                    sys.executable,
                    str(AGGREGATE_SCRIPT),
                    "--project",
                    str(project),
                    "--tag",
                    tag,
                    "--seeds",
                    "5",
                    "--baseline-summary",
                    str(baseline_path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            comparison_path = project / "runs" / "tables" / tag / "nocicim_qpsl_comparison.csv"
            with comparison_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["comparison"], "train_eval_nocicim_vs_original_qpsl")
        self.assertEqual(float(rows[0]["nocicim_success"]), 0.33)
        self.assertEqual(float(rows[0]["baseline_success"]), 0.45)
        self.assertEqual(float(rows[0]["nocicim_cost"]), 5.28)
        self.assertEqual(float(rows[0]["baseline_cost"]), 19.5)


if __name__ == "__main__":
    unittest.main()
