# Train-Time NociCIM Action Shield Design

## Goal

Improve metrics after connecting NociCIM by moving the front-end from
evaluation-only post-processing into the training closed loop. The target is to
make TD3 and QPSL learn under the same NociCIM action filter used at evaluation
time, instead of surprising a policy with a filter it never saw during training.

The strict acceptance gate remains: a NociCIM-connected result is useful only
when paired `success_rate` does not fall below the unshielded baseline and safety
cost improves.

## Current Evidence

Evaluation-only NociCIM improves REC and FAC and helps LAG safety metrics, but
TD3 and QPSL remain fragile. For TD3 and QPSL, small post-training action
changes can reduce cost on some seeds while causing out-of-road or timeout on
others. This points to policy/filter mismatch rather than a simple threshold
tuning problem.

`train_metadrive.py` is the authoritative training entrypoint for this legacy
SaferlKit stack. It selects an action, calls `env.step(action)`, then writes the
executed action into the replay buffer. This means a NociCIM shield inserted
between action selection and `env.step()` will naturally train critics and actors
against the action actually applied to MetaDrive.

## Design

Add an opt-in train-time NociCIM shield to `train_metadrive.py`.

The shield will reuse `MemristiveRiskReflex` and `ReflexCfg` from
`eval_memristive_frontend.py`. This keeps risk estimation, memory state, and
action clipping identical between training and evaluation.

Action flow:

1. The selected SafeRL algorithm produces a nominal action.
2. If `--train_frontend none`, training remains unchanged.
3. If `--train_frontend nocicim_risk_field`, the shield transforms nominal
   action into executed action before `env.step()`.
4. Replay buffers store the executed action. Recovery RL still stores its raw
   policy action separately, but the executed recovery action is shielded.
5. The shield memory resets whenever the environment resets.

Evaluation flow:

1. `eval_policy()` receives the same frontend mode and `ReflexCfg`.
2. Evaluation applies the same shield before `eval_env.step()`.
3. Evaluation memory resets per episode.

## CLI

Add train/eval switches to `train_metadrive.py`:

- `--train_frontend`, default `none`.
- `--eval_frontend`, default `none`.
- `--nocicim_profile`, default `default`, with at least:
  - `default`: current `ReflexCfg`.
  - `risk_gated_micro_guard`: candidate already tested in evaluation.
  - `front_only_gated`: conservative no-lateral profile for fragile policies.

The first implementation should prefer named profiles over dozens of CLI flags
inside training. The existing evaluator remains the place for broad parameter
sweeps.

## Validation

Code-level validation:

- `eval_memristive_frontend.py --self-test` must keep passing.
- A new train helper self-test must prove:
  - `none` returns the nominal action and no intervention.
  - `nocicim_risk_field` changes a risky action.
  - memory reset makes repeated episodes independent.

Smoke training validation:

- Run TD3 and QPSL smoke fine-tunes with small step counts first:
  - `--max_timesteps 10000`
  - `--eval_freq 5000`
  - `--save_model`
  - `--train_frontend nocicim_risk_field`
  - `--eval_frontend nocicim_risk_field`
- Confirm the run exits 0, writes logger CSV, and saves checkpoints.
- Reload the checkpoint with `eval_memristive_frontend.py` inside SaferlKit's
  native runtime.

Metric validation:

- A smoke run can only prove wiring and compatibility.
- A result can update the main improvement audit only after paired evaluation on
  the same seed window shows:
  - `nocicim_success >= baseline_success`
  - `nocicim_mean_cost < baseline_mean_cost`

## Non-Goals

- Do not change algorithm network architectures.
- Do not change MetaDrive runtime, traffic density, or observation shape.
- Do not claim paper-grade multi-seed results from a smoke fine-tune.
- Do not replace the existing evaluator; reuse it for final paired metrics.
