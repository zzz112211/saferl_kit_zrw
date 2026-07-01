# Risk-Gated NociCIM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in risk-confirmation gates to the NociCIM evaluator so TD3 and QPSL can reduce cost without dropping paired baseline success.

**Architecture:** Keep all runtime behavior inside `eval_memristive_frontend.py`, where `MemristiveRiskReflex` already owns action filtering. Preserve existing defaults for reproducibility, then add gated sweep candidates in `sweep_memristive_frontend.py` for TD3/QPSL validation.

**Tech Stack:** Python 3.7 in conda env `asaprl`, NumPy, PyTorch, vendored MetaDrive v0.2.4, CSV result artifacts.

---

### Task 1: Add Risk Gate Configuration and Self-Tests

**Files:**
- Modify: `eval_memristive_frontend.py`

- [ ] **Step 1: Extend `ReflexCfg` with opt-in gates**

Add these dataclass fields with backward-compatible defaults:

```python
    allow_direct_stop: bool = True
    require_memory_for_stop: bool = False
    min_front_risk_for_stop: float = 0.0
    require_risk_for_lateral_guard: bool = False
    min_lateral_risk_for_guard: float = 0.0
```

- [ ] **Step 2: Add helper predicates**

Add methods on `MemristiveRiskReflex`:

```python
    def _front_stop_allowed(self, risk: Risk) -> bool:
        if not bool(self.cfg.allow_direct_stop):
            return False
        if float(risk.front_risk) < float(self.cfg.min_front_risk_for_stop):
            return False
        if bool(self.cfg.require_memory_for_stop) and self.x_front < float(self.cfg.front_slow_threshold):
            return False
        return True

    def _lateral_guard_allowed(self, risk: Risk) -> bool:
        if not bool(self.cfg.require_risk_for_lateral_guard):
            return True
        risk_level = max(float(risk.lateral_risk), float(risk.front_risk))
        if risk_level >= float(self.cfg.min_lateral_risk_for_guard):
            return True
        return self.x_lateral >= float(self.cfg.lateral_guard_threshold)
```

- [ ] **Step 3: Gate direct stop and high-speed lateral intervention**

Change the direct stop branch to require `_front_stop_allowed(risk)`, and change the `high_speed_lateral` branch to require `_lateral_guard_allowed(risk)`.

- [ ] **Step 4: Add self-test assertions**

Add assertions to `run_self_test()`:

```python
    gated_cfg = ReflexCfg(
        num_lasers=30,
        require_risk_for_lateral_guard=True,
        min_lateral_risk_for_guard=0.2,
        allow_direct_stop=False,
        stop_distance_m=3.0,
    )
    gated_reflex = MemristiveRiskReflex("nocicim_risk_field", gated_cfg)
    clear_safe, _risk, clear_debug = gated_reflex.apply(np.array([0.99, 1.0], dtype=np.float32), clear_obs)
    assert clear_safe[1] == 1.0
    assert clear_debug["intervention"] == 0
    stop_obs = np.ones(49, dtype=np.float32)
    stop_obs[-30] = 0.01
    no_stop_reflex = MemristiveRiskReflex("nocicim_risk_field", gated_cfg)
    no_stop_safe, _risk, _debug = no_stop_reflex.apply(np.array([0.0, 1.0], dtype=np.float32), stop_obs)
    assert no_stop_safe[1] > -1.0
```

- [ ] **Step 5: Verify self-test**

Run:

```powershell
& C:\Users\Admin\miniconda3\envs\asaprl\python.exe eval_memristive_frontend.py --self-test
```

Expected: `self-test passed`.

### Task 2: Add Gated Sweep Candidates

**Files:**
- Modify: `sweep_memristive_frontend.py`

- [ ] **Step 1: Add `nocicim_risk_gated_micro_guard`**

Add a candidate that starts from `nocicim_micro_guard`, then enables:

```python
            "allow_direct_stop": True,
            "require_memory_for_stop": False,
            "min_front_risk_for_stop": 0.95,
            "require_risk_for_lateral_guard": True,
            "min_lateral_risk_for_guard": 0.20,
```

- [ ] **Step 2: Add `nocicim_memory_confirmed_stop`**

Add a candidate for QPSL stop-stall diagnosis:

```python
            "allow_direct_stop": False,
            "require_memory_for_stop": True,
            "min_front_risk_for_stop": 0.95,
            "require_risk_for_lateral_guard": True,
            "min_lateral_risk_for_guard": 0.20,
```

- [ ] **Step 3: Verify candidate parsing**

Run:

```powershell
& C:\Users\Admin\miniconda3\envs\asaprl\python.exe -c "from sweep_memristive_frontend import parse_candidate_names; print(parse_candidate_names('nocicim_risk_gated_micro_guard,nocicim_memory_confirmed_stop'))"
```

Expected output includes both candidate names.

### Task 3: Focused Failure-Window Validation

**Files:**
- Result dirs under `logs/`

- [ ] **Step 1: Run TD3 seeds 112-114 sweep**

Run:

```powershell
& C:\Users\Admin\miniconda3\envs\asaprl\python.exe sweep_memristive_frontend.py --algo td3 --checkpoint models/paper_td3_1m_seed0_20260629 --episodes 3 --start-seed 112 --include-risk-field --candidates nocicim_risk_gated_micro_guard,nocicim_memory_confirmed_stop --outdir logs/memristive_sweep_td3_risk_gated_focus_20260701
```

Expected: best candidate has `success_delta >= 0`.

- [ ] **Step 2: Run QPSL seed 114 sweep**

Run:

```powershell
& C:\Users\Admin\miniconda3\envs\asaprl\python.exe sweep_memristive_frontend.py --algo qpsl --checkpoint models/paper_qpsl_1m_seed0_20260629 --episodes 1 --start-seed 114 --include-risk-field --candidates nocicim_risk_gated_micro_guard,nocicim_memory_confirmed_stop --outdir logs/memristive_sweep_qpsl_risk_gated_focus_20260701
```

Expected: best candidate has `success_delta >= 0`.

### Task 4: Full 20-Episode Verification and Audit

**Files:**
- Modify: `logs/nocicim_saferl_improvement_audit_20ep_20260630.csv` only if the A-gate passes.

- [ ] **Step 1: Run 20-episode TD3 eval for the best focused candidate**

Use `eval_memristive_frontend.py` with `--frontends none,nocicim_risk_field`, `--episodes 20`, and `--start-seed 100`.

- [ ] **Step 2: Run 20-episode QPSL eval for the best focused candidate**

Use the same paired-eval structure as TD3.

- [ ] **Step 3: Update audit only for success-preserving wins**

Add or replace rows only when:

```text
nocicim_success >= baseline_success
nocicim_mean_cost < baseline_mean_cost
```

- [ ] **Step 4: Final verification**

Run:

```powershell
& C:\Users\Admin\miniconda3\envs\asaprl\python.exe eval_memristive_frontend.py --self-test
git diff --check
git status --short
```

Expected: self-test passes, diff check passes, and only intended source/docs changes are shown.
