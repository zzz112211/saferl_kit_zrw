import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.select_crash_aware_portfolio import (  # noqa: E402
    compute_gates,
    select_portfolio_rows,
    write_outputs,
)


class CrashAwarePortfolioTest(unittest.TestCase):
    def test_rejects_success_regression_and_prefers_lower_crash(self):
        rows = [
            {
                "algo": "td3",
                "candidate": "none",
                "success_rate": "0.80",
                "clean_success_rate": "0.10",
                "mean_cost_rate": "0.05",
                "crash_rate": "0.70",
                "out_of_road_rate": "0.20",
            },
            {
                "algo": "td3",
                "candidate": "nocicim_regressing_low_crash",
                "success_rate": "0.79",
                "clean_success_rate": "0.20",
                "mean_cost_rate": "0.03",
                "crash_rate": "0.40",
                "out_of_road_rate": "0.10",
            },
            {
                "algo": "td3",
                "candidate": "nocicim_safe",
                "success_rate": "0.80",
                "clean_success_rate": "0.12",
                "mean_cost_rate": "0.04",
                "crash_rate": "0.60",
                "out_of_road_rate": "0.18",
            },
        ]

        selected, failures = select_portfolio_rows(rows, ["td3"])

        self.assertEqual(failures, [])
        self.assertEqual(selected[0]["selected_candidate"], "nocicim_safe")
        self.assertEqual(selected[0]["success_delta"], 0.0)
        self.assertAlmostEqual(selected[0]["crash_delta"], -0.10)

    def test_reports_algo_without_non_regressing_candidate(self):
        rows = [
            {
                "algo": "epo",
                "candidate": "none",
                "success_rate": "0.77",
                "clean_success_rate": "0.29",
                "mean_cost_rate": "0.013",
                "crash_rate": "0.61",
                "out_of_road_rate": "0.22",
            },
            {
                "algo": "epo",
                "candidate": "nocicim_candidate",
                "success_rate": "0.76",
                "clean_success_rate": "0.35",
                "mean_cost_rate": "0.010",
                "crash_rate": "0.50",
                "out_of_road_rate": "0.20",
            },
        ]

        selected, failures = select_portfolio_rows(rows, ["epo"])

        self.assertEqual(selected, [])
        self.assertEqual(failures[0]["algo"], "epo")
        self.assertEqual(failures[0]["reason"], "no_success_non_regressing_candidate")

    def test_computes_portfolio_gates(self):
        selected = [
            {
                "algo": "td3",
                "baseline_success_rate": 0.80,
                "nocicim_success_rate": 0.80,
                "baseline_clean_success_rate": 0.10,
                "nocicim_clean_success_rate": 0.12,
                "baseline_crash_rate": 0.70,
                "nocicim_crash_rate": 0.60,
            },
            {
                "algo": "qpsl",
                "baseline_success_rate": 0.85,
                "nocicim_success_rate": 0.86,
                "baseline_clean_success_rate": 0.08,
                "nocicim_clean_success_rate": 0.09,
                "baseline_crash_rate": 0.80,
                "nocicim_crash_rate": 0.75,
            },
        ]

        gates = compute_gates(selected, ["td3", "qpsl"], [])

        self.assertEqual(gates["all_success_non_regression"], 1)
        self.assertEqual(gates["mean_crash_reduced"], 1)
        self.assertEqual(gates["mean_clean_success_not_lower"], 1)

    def test_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            selected = [
                {
                    "algo": "td3",
                    "selected_candidate": "nocicim_safe",
                    "baseline_success_rate": 0.80,
                    "nocicim_success_rate": 0.80,
                    "success_delta": 0.0,
                    "baseline_clean_success_rate": 0.10,
                    "nocicim_clean_success_rate": 0.12,
                    "clean_success_delta": 0.02,
                    "baseline_mean_cost_rate": 0.05,
                    "nocicim_mean_cost_rate": 0.04,
                    "cost_rate_delta": -0.01,
                    "baseline_crash_rate": 0.70,
                    "nocicim_crash_rate": 0.60,
                    "crash_delta": -0.10,
                    "baseline_out_of_road_rate": 0.20,
                    "nocicim_out_of_road_rate": 0.18,
                    "out_of_road_delta": -0.02,
                }
            ]
            gates = {"all_success_non_regression": 1, "mean_crash_reduced": 1}
            write_outputs(out_dir, selected, gates, [])

            with (out_dir / "portfolio_selected.csv").open(newline="", encoding="utf-8") as f:
                selected_rows = list(csv.DictReader(f))
            with (out_dir / "portfolio_gates.csv").open(newline="", encoding="utf-8") as f:
                gate_rows = list(csv.DictReader(f))

            self.assertEqual(selected_rows[0]["selected_candidate"], "nocicim_safe")
            self.assertEqual(gate_rows[0]["all_success_non_regression"], "1")


if __name__ == "__main__":
    unittest.main()
