# Train-Time NociCIM Shield Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in NociCIM action shield to `train_metadrive.py` so policies can be trained and evaluated under the same NociCIM-connected closed loop.

**Architecture:** Reuse `MemristiveRiskReflex` and `ReflexCfg` from `eval_memristive_frontend.py`. `train_metadrive.py` will build a profile-based shield, apply it between policy action selection and `env.step()`, store executed actions in replay buffers, and reset shield memory on episode boundaries.

**Tech Stack:** Python 3.7 in conda env `asaprl`, NumPy, PyTorch, SaferlKit training loop, vendored MetaDrive v0.2.4.

---

### Task 1: Add Shield Helpers to Training Entrypoint

**Files:**
- Modify: `train_metadrive.py`

- [ ] **Step 1: Write the failing helper self-test call**

Add `--self_test_nocicim_shield` to the parser and call `run_nocicim_shield_self_test()` before environment construction. Initially this should fail because the function does not exist.

Run:

```powershell
& C:\Users\Admin\miniconda3\envs\asaprl\python.exe train_metadrive.py --self_test_nocicim_shield --use_td3
```

Expected: `NameError` for `run_nocicim_shield_self_test`.

- [ ] **Step 2: Import evaluator shield primitives**

Add:

```python
from eval_memristive_frontend import MemristiveRiskReflex, ReflexCfg
```

- [ ] **Step 3: Implement profile construction**

Add:

```python
def build_nocicim_reflex_cfg(profile):
    if profile == "default":
        return ReflexCfg()
    if profile == "risk_gated_micro_guard":
        return ReflexCfg(
            front_distance_threshold_m=8.0,
            stop_distance_m=2.0,
            lateral_distance_threshold_m=5.0,
            slow_front_risk_threshold=0.99,
            stop_front_risk_threshold=0.99,
            slow_throttle_cap=0.95,
            lateral_steer_threshold=0.95,
            lateral_throttle_threshold=0.95,
            lateral_throttle_cap=0.85,
            lateral_steering_cap=1.0,
            front_decay=0.5,
            front_gain=0.02,
            front_slow_threshold=0.98,
            front_slow_throttle_cap=0.95,
            lateral_decay=0.5,
            lateral_gain=0.03,
            lateral_guard_threshold=0.98,
            allow_direct_stop=True,
            require_memory_for_stop=False,
            min_front_risk_for_stop=0.95,
            require_risk_for_lateral_guard=True,
            min_lateral_risk_for_guard=0.2,
        )
    if profile == "front_only_gated":
        return ReflexCfg(
            front_distance_threshold_m=8.0,
            stop_distance_m=2.0,
            lateral_distance_threshold_m=1.0,
            slow_front_risk_threshold=0.99,
            stop_front_risk_threshold=0.99,
            slow_throttle_cap=0.95,
            lateral_steer_threshold=1.01,
            lateral_throttle_threshold=1.01,
            lateral_throttle_cap=0.95,
            lateral_steering_cap=1.0,
            front_decay=0.5,
            front_gain=0.02,
            front_slow_threshold=0.98,
            front_slow_throttle_cap=0.95,
            lateral_decay=0.5,
            lateral_gain=0.0,
            lateral_guard_threshold=1.01,
            allow_direct_stop=True,
            require_memory_for_stop=False,
            min_front_risk_for_stop=0.99,
            require_risk_for_lateral_guard=True,
            min_lateral_risk_for_guard=1.01,
        )
    raise ValueError("unknown nocicim profile: {}".format(profile))
```

- [ ] **Step 4: Implement shield wrappers**

Add:

```python
def build_action_shield(frontend, profile):
    if frontend == "none":
        return None
    if frontend not in {"risk_field", "nocicim_risk_field"}:
        raise ValueError("unsupported frontend: {}".format(frontend))
    return MemristiveRiskReflex(frontend, build_nocicim_reflex_cfg(profile))


def apply_action_shield(shield, action, state):
    if shield is None:
        return action, 0, 0.0
    safe_action, _risk, debug = shield.apply(action, state)
    return safe_action, int(debug["intervention"]), float(debug["action_distortion"])
```

- [ ] **Step 5: Implement self-test**

Add:

```python
def run_nocicim_shield_self_test():
    clear_obs = np.ones(49, dtype=np.float32)
    risky_obs = np.ones(49, dtype=np.float32)
    risky_obs[-30] = 0.01
    action = np.array([0.9, 0.9], dtype=np.float32)
    no_shield_action, intervention, distortion = apply_action_shield(None, action, clear_obs)
    assert np.allclose(no_shield_action, action)
    assert intervention == 0
    assert distortion == 0.0
    shield = build_action_shield("nocicim_risk_field", "default")
    safe_action, intervention, distortion = apply_action_shield(shield, action, risky_obs)
    assert safe_action[1] <= 0.0
    assert intervention == 1
    assert distortion > 0.0
    shield = build_action_shield("nocicim_risk_field", "front_only_gated")
    first, _intervention, _distortion = apply_action_shield(shield, action, clear_obs)
    shield = build_action_shield("nocicim_risk_field", "front_only_gated")
    second, _intervention, _distortion = apply_action_shield(shield, action, clear_obs)
    assert np.allclose(first, second)
    print("nocicim shield self-test passed")
```

- [ ] **Step 6: Verify self-test passes**

Run:

```powershell
& C:\Users\Admin\miniconda3\envs\asaprl\python.exe train_metadrive.py --self_test_nocicim_shield --use_td3
```

Expected: `nocicim shield self-test passed`.

### Task 2: Wire Shield into Train and Eval Loops

**Files:**
- Modify: `train_metadrive.py`

- [ ] **Step 1: Add CLI arguments**

Add parser args:

```python
parser.add_argument("--train_frontend", default="none", choices=["none", "risk_field", "nocicim_risk_field"])
parser.add_argument("--eval_frontend", default="none", choices=["none", "risk_field", "nocicim_risk_field"])
parser.add_argument("--nocicim_profile", default="default", choices=["default", "risk_gated_micro_guard", "front_only_gated"])
parser.add_argument("--self_test_nocicim_shield", action="store_true")
```

- [ ] **Step 2: Extend `eval_policy` signature**

Change:

```python
def eval_policy(policy, policy_type, eval_env, seed, eval_episodes=20):
```

to:

```python
def eval_policy(policy, policy_type, eval_env, seed, eval_episodes=20, frontend="none", profile="default"):
```

Inside each episode create:

```python
shield = build_action_shield(frontend, profile)
```

Apply shield before `eval_env.step(action)`.

- [ ] **Step 3: Add training shield state**

After initial `env.reset()`, create:

```python
train_shield = build_action_shield(args.train_frontend, args.nocicim_profile)
train_interventions = 0
train_distortion_sum = 0.0
```

- [ ] **Step 4: Apply shield before environment step**

Before `next_state, reward, done, info = env.step(action)`, add:

```python
executed_action, intervention, distortion = apply_action_shield(train_shield, action, state)
train_interventions += intervention
train_distortion_sum += distortion
```

Then step with `executed_action`.

- [ ] **Step 5: Store executed actions**

Use `executed_action` in replay buffers except for recovery raw action:

```python
if args.use_td3:
    replay_buffer.add(state, executed_action, next_state, reward, done_bool)
elif args.use_recovery:
    replay_buffer.add(state, raw_action, executed_action, next_state, reward, cost, done_bool)
elif args.use_qpsl:
    replay_buffer.add(state, executed_action, next_state, reward, cost, prev_cost, done_bool)
else:
    replay_buffer.add(state, executed_action, next_state, reward, cost, done_bool)
```

- [ ] **Step 6: Reset shield on env reset**

After `state, done = env.reset(), False`, rebuild:

```python
train_shield = build_action_shield(args.train_frontend, args.nocicim_profile)
```

- [ ] **Step 7: Pass eval shield arguments**

Change eval call to:

```python
evalEpRet, evalEpCost = eval_policy(
    policy,
    run_policy_type,
    eval_env,
    args.seed,
    frontend=args.eval_frontend,
    profile=args.nocicim_profile,
)
```

### Task 3: Smoke Verification

**Files:**
- Result dirs under `logs/`
- Checkpoints under `models/`

- [ ] **Step 1: Run code checks**

Run:

```powershell
& C:\Users\Admin\miniconda3\envs\asaprl\python.exe eval_memristive_frontend.py --self-test
& C:\Users\Admin\miniconda3\envs\asaprl\python.exe train_metadrive.py --self_test_nocicim_shield --use_td3
git diff --check
```

Expected: both self-tests pass and diff check passes.

- [ ] **Step 2: Run TD3 10k smoke fine-tune**

Run:

```powershell
& C:\Users\Admin\miniconda3\envs\asaprl\python.exe train_metadrive.py --use_td3 --seed 0 --exp_name nocicim_trainshield_td3_10k_seed0_20260701 --max_timesteps 10000 --eval_freq 5000 --start_timesteps 1000 --save_model --train_frontend nocicim_risk_field --eval_frontend nocicim_risk_field --nocicim_profile front_only_gated
```

Expected: exits 0, writes logger CSV, and saves `models/nocicim_trainshield_td3_10k_seed0_20260701_*`.

- [ ] **Step 3: Run QPSL 10k smoke fine-tune**

Run:

```powershell
& C:\Users\Admin\miniconda3\envs\asaprl\python.exe train_metadrive.py --use_qpsl --seed 0 --exp_name nocicim_trainshield_qpsl_10k_seed0_20260701 --max_timesteps 10000 --eval_freq 5000 --start_timesteps 1000 --save_model --train_frontend nocicim_risk_field --eval_frontend nocicim_risk_field --nocicim_profile front_only_gated
```

Expected: exits 0, writes logger CSV, and saves `models/nocicim_trainshield_qpsl_10k_seed0_20260701_*`.

- [ ] **Step 4: Reload smoke checkpoints with evaluator**

Run paired 5-episode evaluator checks for each smoke checkpoint using
`eval_memristive_frontend.py --frontends none,nocicim_risk_field`.

Expected: checkpoint loads successfully. Metrics are smoke evidence only.

### Task 4: Commit and Report

**Files:**
- Modify: `train_metadrive.py`

- [ ] **Step 1: Final status checks**

Run:

```powershell
git status --short
git diff --stat
```

- [ ] **Step 2: Commit source changes**

Run:

```powershell
git add -- train_metadrive.py
git commit -m "feat: add train-time nocicim shield"
```

- [ ] **Step 3: Report evidence**

Report:

- commits created
- self-test status
- smoke command exits
- logger/checkpoint paths
- smoke evaluator aggregate paths
- whether goal remains active for longer fine-tune/paper-grade runs
