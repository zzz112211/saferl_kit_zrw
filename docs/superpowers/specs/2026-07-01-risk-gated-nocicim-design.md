# Risk-Gated NociCIM Design

## Goal

Improve the SaferlKit NociCIM front-end under a strict paper-baseline gate:
`success_rate` must not drop below the paired baseline. Safety-cost reductions
are useful only when this gate holds.

The immediate target is to recover TD3 and QPSL from current NociCIM
regressions while preserving the positive REC, LAG, EPO, and FAC evidence.

## Current Evidence

The native SaferlKit evaluator must remain the source of truth because the
checkpoints are tied to the vendored MetaDrive v0.2.4 runtime.

Current 20-episode paired results show:

- TD3: cost can be reduced, but success drops.
- QPSL: cost can be reduced, but success drops.
- FAC: `soft_front_trim_nocicim` improves success and cost.
- REC: `close_range_nocicim` improves success and cost.
- LAG: `light_memory_nocicim` keeps success tied and improves safety metrics.
- EPO: success and cost improve, with a small clean-success tradeoff.

Trace inspection found two actionable failure mechanisms:

- TD3 is disturbed by `high_speed_lateral` intervention even when
  `risk_front_risk == 0` and `risk_lateral_risk == 0`.
- QPSL can be stopped by the direct emergency-stop branch before NociCIM memory
  confirms persistent risk, causing long stalled episodes and lost success.

## Design

Add optional risk-confirmation gates to `MemristiveRiskReflex`.

The existing behavior remains the default so older result folders and existing
commands stay reproducible. New sweep candidates opt into the stricter gated
mode.

The gated mode changes only when an intervention is allowed, not how risk is
estimated:

- Direct emergency stop is allowed only when front risk is sufficiently high,
  or when stop distance is reached and the configuration explicitly allows the
  direct stop path.
- Lateral high-speed action limiting is allowed only when lateral/front risk is
  present or when NociCIM lateral memory has accumulated enough evidence.
- NociCIM memory-based throttle trimming remains available, but its hard clamps
  are tied to memory thresholds instead of raw high-speed steering alone.

This keeps the circuit interpretation narrow: raw LiDAR risk feeds a memory
state, and the comparator only activates a reflex after risk is confirmed.

## Configuration

Extend `ReflexCfg` with opt-in gates:

- `require_risk_for_lateral_guard`: block high-speed lateral limiting when no
  risk is present.
- `min_lateral_risk_for_guard`: minimum lateral or front risk required for the
  direct lateral guard.
- `require_memory_for_stop`: block direct emergency stop unless front memory
  has also accumulated.
- `min_front_risk_for_stop`: minimum front risk required for direct stop.
- `allow_direct_stop`: preserve the current emergency stop path by default.

Add new sweep candidates for TD3 and QPSL that enable these gates. Do not remove
the existing candidates because they document earlier tradeoffs.

## Validation

Unit-level self-test coverage:

- A clear observation with high steering/throttle must not be altered when
  `require_risk_for_lateral_guard` is enabled.
- A risky front observation must still be able to reduce throttle.
- A stop-distance observation must not hard-stop when `allow_direct_stop` is
  disabled and memory has not confirmed risk.

Experiment validation:

- First run the known failure windows:
  - TD3 seeds 112-114.
  - QPSL seed 114.
- Then run 20-episode paired evaluation for TD3 and QPSL on seeds 100-119.
- A candidate can update the main audit table only if:
  - `nocicim_success >= baseline_success`.
  - `nocicim_mean_cost < baseline_mean_cost`.
  - The exact aggregate CSV is referenced in the audit row.

## Non-Goals

- Do not retrain TD3 or QPSL in this change.
- Do not change the environment, traffic density, or checkpoint runtime.
- Do not claim multi-seed paper readiness from single-seed 20-episode evidence.
