# Paper Style Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Saferl Kit SafeMetaDrive baselines in the repository's paper-style setting: 1,000,000 training steps, 5 seeds, 20 evaluation episodes, and all six algorithms.

**Architecture:** Keep algorithm code unchanged. Add Slurm array scripts that map array task IDs to algorithm/seed pairs and use unique experiment names so logs and model files do not collide. Add a lightweight CSV aggregator for mean/std reporting after jobs finish.

**Tech Stack:** Bash, Slurm, Python 3.10 venv at `/datapool/home/1001900174/home/DYJ/saferl_kit/.venv-saferlkit`, SafeMetaDrive, TD3-family algorithms in `train_metadrive.py`.

---

### Task 1: Add Slurm Array Scripts

**Files:**
- Create: `scripts/slurm_all_algos_smoke.sh`
- Create: `scripts/slurm_paper_1m_all_algos_5seeds.sh`

- [ ] **Step 1: Create a smoke array script**

Run one short job per algorithm with 300 training steps, 50 random-start steps, and one evaluation episode.

- [ ] **Step 2: Create the full paper-style array script**

Run 30 tasks: 6 algorithms x 5 seeds. Use 1,000,000 steps, 10,000-step evaluation frequency, and 20 evaluation episodes.

- [ ] **Step 3: Submit the smoke job**

Run: `sbatch scripts/slurm_all_algos_smoke.sh`

Expected: Slurm accepts a six-task array.

- [ ] **Step 4: Verify smoke completion**

Run: `sacct -j <smoke_job_id> --format=JobID,JobName,State,Elapsed,ExitCode -P`

Expected: all array tasks show `COMPLETED` and `0:0`.

- [ ] **Step 5: Submit the full job**

Run: `sbatch scripts/slurm_paper_1m_all_algos_5seeds.sh`

Expected: Slurm accepts a 30-task array.

### Task 2: Add Result Aggregation

**Files:**
- Create: `scripts/aggregate_paper_results.py`

- [ ] **Step 1: Add a CSV aggregator**

The script scans `logs/<tag>_<algo>_seed<seed>_SafeMetaDriveEnv-seed<seed>-*/logger.csv`, keeps the latest run directory for each algorithm/seed, and writes per-step mean/std/count rows.

- [ ] **Step 2: Run aggregation after jobs complete**

Run: `.venv-saferlkit/bin/python scripts/aggregate_paper_results.py --tag paper_1m_20260629`

Expected: writes `runs/tables/paper_1m_20260629_summary.csv` with algorithm, total_steps, mean/std metrics, and seed counts.
