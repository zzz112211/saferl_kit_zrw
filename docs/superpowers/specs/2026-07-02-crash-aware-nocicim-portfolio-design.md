# Crash-Aware NociCIM Portfolio Design

## Goal

Improve the NociCIM evaluation front-end so the six saferl_kit algorithms
(`td3`, `epo`, `rec`, `qpsl`, `lag`, and `fac`) keep paired success rate at
or above their no-front-end baselines while reducing mean crash rate. If a
candidate trades metrics off, crash rate and clean success take priority over
raw success rate.

## Current Evidence

Existing sweeps show that a single profile is not reliable across algorithms:

- `nocicim_close_range` improves TD3 and QPSL success/cost but can increase
  crash rate, especially for TD3 and QPSL.
- `nocicim_route_preserve` is gentler and helps REC/QPSL success, but still
  does not solve every algorithm.
- EPO and FAC are sensitive to stronger intervention; they need conservative
  policies that avoid suppressing success.
- LAG already responds well to a light guard profile.

The design therefore must not force one profile onto all six algorithms.

## Approach

Add an explicit crash-aware portfolio layer that maps each algorithm to a
candidate profile selected for its failure mode:

- TD3 and QPSL: favor lower-crash profiles over aggressive close-range
  intervention, with a soft crash guard that reduces high-risk throttle without
  forcing hard stops too early.
- REC and LAG: preserve existing route-preserving or micro-guard behavior when
  it keeps success and improves safety.
- EPO and FAC: use conservative profiles that only intervene under high risk,
  because these algorithms lose success under broader intervention.

The portfolio is an evaluation-time front-end. It does not retrain the
underlying saferl_kit checkpoints.

## Components

### Candidate Definitions

Extend `sweep_memristive_frontend.py` with a small set of crash-aware
candidates. These candidates should reuse `MemristiveRiskReflex` and override
only `ReflexCfg` fields. No new RL algorithm class is introduced.

### Portfolio Selection

Add an aggregation script that reads per-algorithm sweep summaries and selects
one candidate per algorithm using this ordering:

1. Reject candidates with `success_rate < baseline_success_rate`.
2. Among remaining candidates, prefer lower `crash_rate`.
3. Break ties with higher `clean_success_rate`.
4. Break remaining ties with lower `mean_cost_rate`, then lower
   `out_of_road_rate`.

If an algorithm has no candidate satisfying success non-regression, the script
must report the failure rather than silently choosing a regressing candidate.

### Validation Gates

The final portfolio table must expose these gates:

- `all_success_non_regression`: every algorithm has
  `nocicim_success_rate >= baseline_success_rate`.
- `mean_crash_reduced`: average NociCIM crash rate across the six algorithms is
  lower than the average baseline crash rate.
- `mean_clean_success_not_lower`: average clean success is not lower than the
  baseline average. This is a priority gate for model selection and a
  supplementary-paper metric, not necessarily a main-table metric.

## Experiment Flow

Run the work in two phases:

1. Fast screen: six algorithms, five checkpoint seeds, twenty evaluation
   episodes per seed.
2. Confirmation: the selected portfolio from phase 1, six algorithms, five
   checkpoint seeds, one hundred evaluation episodes per seed.

Both phases compare against the paired no-front-end baseline on the same
algorithm checkpoints and evaluation seeds.

## Reporting

The main paper table should report:

- success rate
- cost rate
- crash rate
- out-of-road rate

Clean success should be kept in the generated CSV and reported in supplementary
or diagnostic material. This avoids overloading the main table while preserving
the stricter safety evidence needed to defend the claim.

## Out of Scope

- Retraining the saferl_kit checkpoints.
- Claiming collision-free driving.
- Replacing the saferl_kit algorithms.
- Hardware energy modeling for this validation pass.

## Acceptance Criteria

The work is complete only when the current local or pulled HPC artifacts prove:

1. Results exist for all six algorithms.
2. Each algorithm's selected NociCIM portfolio row has success rate greater
   than or equal to its paired no-front-end baseline.
3. Mean crash rate across the selected six NociCIM rows is lower than the mean
   crash rate across paired baselines.
4. The generated table includes clean success so metric trade-offs remain
   auditable.
5. Unit tests cover candidate definitions and portfolio selection gates.
