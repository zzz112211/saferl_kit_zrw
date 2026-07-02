# Crash-Aware NociCIM Portfolio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select and validate a crash-aware NociCIM portfolio across TD3, EPO, REC, QPSL, LAG, and FAC so paired success does not regress and mean crash rate decreases.

**Architecture:** Keep the runtime front-end in `eval_memristive_frontend.py` unchanged unless tests prove a behavior gap. Add candidate profiles in `sweep_memristive_frontend.py` and a standalone portfolio selector that reads sweep CSVs, rejects success-regressing candidates, and reports portfolio-level gates. Validate first with unit tests and existing pulled sweep artifacts, then use the same selector on fresh 6-algorithm evaluation outputs.

**Tech Stack:** Python 3.7-compatible standard library, `unittest`, existing saferl_kit sweep CSV outputs, Slurm/HPC for long MetaDrive evaluation.

---

### Task 1: Add Portfolio Selection Unit Tests

**Files:**
- Create: `tests/test_select_crash_aware_portfolio.py`
- Modify: none

- [ ] **Step 1: Write tests for success filtering and crash-aware ranking**

```python
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.select_crash_aware_portfolio import (
    compute_gates,
    read_rows,
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
```

- [ ] **Step 2: Run the test and verify it fails before implementation**

Run: `python -m unittest tests.test_select_crash_aware_portfolio`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.select_crash_aware_portfolio'`.

### Task 2: Implement Portfolio Selector

**Files:**
- Create: `scripts/select_crash_aware_portfolio.py`
- Test: `tests/test_select_crash_aware_portfolio.py`

- [ ] **Step 1: Add selector implementation**

Implement these functions:

```python
read_rows(path)
parse_algos(value)
select_portfolio_rows(rows, algos)
compute_gates(selected_rows, algos, failures)
write_outputs(out_dir, selected_rows, gates, failures)
main()
```

The candidate ordering is:

```python
(
    crash_rate,
    -clean_success_rate,
    mean_cost_rate,
    out_of_road_rate,
    candidate_name,
)
```

Rows with `candidate == "none"` are baselines and never selected as NociCIM candidates.

- [ ] **Step 2: Run selector tests**

Run: `python -m unittest tests.test_select_crash_aware_portfolio`

Expected: OK.

### Task 3: Add Crash-Aware Candidate Tests

**Files:**
- Modify: `tests/test_sweep_memristive_frontend.py`
- Modify: `sweep_memristive_frontend.py`

- [ ] **Step 1: Add tests for crash-aware candidates**

Add tests that require these candidates:

```python
nocicim_soft_crash_guard
nocicim_crash_aware_route
nocicim_conservative_clean
```

Expected key properties:

```python
nocicim_soft_crash_guard:
  frontend == "nocicim_risk_field"
  allow_direct_stop == False
  slow_front_risk_threshold >= 0.80
  lateral_throttle_cap >= 0.70

nocicim_crash_aware_route:
  frontend == "nocicim_risk_field"
  require_risk_for_lateral_guard == True
  min_lateral_risk_for_guard >= 0.20
  front_slow_throttle_cap >= 0.92

nocicim_conservative_clean:
  frontend == "nocicim_risk_field"
  slow_front_risk_threshold >= 0.95
  stop_front_risk_threshold >= 0.99
  lateral_guard_threshold >= 0.98
```

- [ ] **Step 2: Run the candidate test and verify it fails**

Run: `python -m unittest tests.test_sweep_memristive_frontend`

Expected: FAIL because candidate names are missing.

- [ ] **Step 3: Add candidates to `CANDIDATES`**

Add three conservative profiles to `sweep_memristive_frontend.py` using only
existing `ReflexCfg` fields. Keep them compatible with `eval_memristive_frontend.py`.

- [ ] **Step 4: Run candidate tests**

Run: `python -m unittest tests.test_sweep_memristive_frontend`

Expected: OK.

### Task 4: Validate Selector on Existing All-Algorithm Artifacts

**Files:**
- Create outputs under `runs/tables/crash_aware_portfolio_existing_20260702/`

- [ ] **Step 1: Run selector on existing profile sweep summary**

Run:

```powershell
python scripts/select_crash_aware_portfolio.py `
  --input runs/tables/hpc_allalgos_profile_sweep_paper1m_20260702/allalgos_profile_sweep_5seed_summary_20260702.csv `
  --algos td3,epo,rec,qpsl,lag,fac `
  --outdir runs/tables/crash_aware_portfolio_existing_20260702
```

Expected: writes `portfolio_selected.csv`, `portfolio_gates.csv`, and
`portfolio_failures.csv`. If gates fail, use the output to decide which
algorithm needs fresh candidate evaluation.

### Task 5: Prepare Fresh 6-Algorithm Evaluation

**Files:**
- Create: `scripts/slurm_allalgos_crash_aware_portfolio_sweep.sh`

- [ ] **Step 1: Add Slurm array script**

The script runs six algorithms x five train seeds with focused candidates:

```text
nocicim_soft_crash_guard,nocicim_crash_aware_route,nocicim_conservative_clean,nocicim_route_preserve,nocicim_micro_guard,nocicim_cost_guard,nocicim_soft_front_trim
```

It calls `sweep_memristive_frontend.py` with paired `none` baseline rows and
stores one `sweep_summary.csv` per algorithm/seed.

- [ ] **Step 2: Submit 20-episode screening array on HPC**

Run through the existing `Host HPC` configuration, from
`/datapool/home/1001900174/home/DYJ/saferl_kit`.

Expected: Slurm job id and 30 array tasks.

### Task 6: Aggregate Fresh Validation

**Files:**
- Create outputs under `runs/tables/hpc_allalgos_crash_aware_portfolio_20260702/`

- [ ] **Step 1: Pull or inspect HPC outputs**

Confirm all 30 `sweep_summary.csv` files exist.

- [ ] **Step 2: Run portfolio selector**

Run selector on the aggregated 20-episode summary. If all gates pass, launch
the 100-episode confirmation with selected candidates only.

- [ ] **Step 3: Completion audit**

Mark the goal complete only if the final 100-episode artifacts prove all six
success non-regression and mean crash reduction.
