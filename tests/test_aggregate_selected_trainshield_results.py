import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AGGREGATE_SCRIPT = REPO_ROOT / "scripts" / "aggregate_selected_trainshield_results.py"


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class AggregateSelectedTrainShieldResultsTest(unittest.TestCase):
    def test_aggregates_training_and_matched_eval_by_algorithm(self):
        tag = "unit_selected"
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            for algo in ("td3", "qpsl"):
                for seed in range(2):
                    logger = (
                        project
                        / "logs"
                        / "{}_{}_seed{}_SafeMetaDriveEnv-seed{}-0".format(tag, algo, seed, seed)
                        / "logger.csv"
                    )
                    write_csv(
                        logger,
                        ["total_steps", "EpRet", "EpCost", "CostRate"],
                        [
                            {
                                "total_steps": "10000",
                                "EpRet": str(0.2 + seed),
                                "EpCost": str(10 + seed),
                                "CostRate": "0.01",
                            },
                            {
                                "total_steps": "20000",
                                "EpRet": str(0.4 + seed),
                                "EpCost": str(8 + seed),
                                "CostRate": "0.02",
                            },
                        ],
                    )
                    eval_dir = project / "logs" / "eval_{}_{}_seed{}_matched_20ep".format(tag, algo, seed)
                    write_csv(
                        eval_dir / "sweep_summary.csv",
                        [
                            "candidate",
                            "frontend",
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
                        ],
                        [
                            {
                                "candidate": "none",
                                "frontend": "none",
                                "success_rate": "0.4",
                                "clean_success_rate": "0.1",
                                "mean_reward": "100",
                                "mean_episode_cost": "9",
                                "mean_cost_rate": "0.03",
                                "crash_rate": "0.8",
                                "out_of_road_rate": "0.5",
                                "out_of_time_rate": "0",
                                "intervention_ratio": "0",
                                "mean_action_distortion": "0",
                            },
                            {
                                "candidate": "nocicim_close_range",
                                "frontend": "nocicim_risk_field",
                                "success_rate": "0.5",
                                "clean_success_rate": "0.2",
                                "mean_reward": "120",
                                "mean_episode_cost": "6",
                                "mean_cost_rate": "0.02",
                                "crash_rate": "0.7",
                                "out_of_road_rate": "0.3",
                                "out_of_time_rate": "0",
                                "intervention_ratio": "0.1",
                                "mean_action_distortion": "0.2",
                            },
                        ],
                    )

            subprocess.run(
                [
                    sys.executable,
                    str(AGGREGATE_SCRIPT),
                    "--project",
                    str(project),
                    "--tag",
                    tag,
                    "--algos",
                    "td3,qpsl",
                    "--seeds",
                    "2",
                    "--matched-eval-suffix",
                    "matched_20ep",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            train_summary = (
                project / "runs" / "tables" / tag / "selected_train_eval_summary.csv"
            )
            matched_summary = (
                project / "runs" / "tables" / tag / "selected_paired_eval_summary.csv"
            )
            with train_summary.open(newline="", encoding="utf-8") as f:
                train_rows = list(csv.DictReader(f))
            with matched_summary.open(newline="", encoding="utf-8") as f:
                matched_rows = list(csv.DictReader(f))

        td3_20k = [r for r in train_rows if r["algo"] == "td3" and r["total_steps"] == "20000"][0]
        self.assertEqual(td3_20k["seed_count"], "2")
        self.assertAlmostEqual(float(td3_20k["EpRet_mean"]), 0.9)
        self.assertAlmostEqual(float(td3_20k["EpCost_mean"]), 8.5)

        qpsl_noci = [
            r
            for r in matched_rows
            if r["algo"] == "qpsl" and r["candidate"] == "nocicim_close_range"
        ][0]
        self.assertEqual(qpsl_noci["train_seed_count"], "2")
        self.assertAlmostEqual(float(qpsl_noci["success_rate_mean"]), 0.5)
        self.assertAlmostEqual(float(qpsl_noci["mean_episode_cost_mean"]), 6.0)


if __name__ == "__main__":
    unittest.main()
