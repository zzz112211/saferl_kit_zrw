import numpy as np

from risk_reflex.configs import RiskReflexConfig
from risk_reflex.risk_frontend import MemristiveRiskFrontEnd
from risk_reflex.reflex_layer import PainReflexLayer
from risk_reflex.wrappers import RiskReflexController


def make_obs(front_lidar_value=1.0, speed=0.5, left=0.5, right=0.5):
    obs = np.ones(49, dtype=np.float32)
    obs[0] = left
    obs[1] = right
    obs[3] = speed
    obs[-30:] = 1.0
    obs[-30] = front_lidar_value
    obs[-1] = front_lidar_value
    obs[-2] = front_lidar_value
    obs[-29] = front_lidar_value
    return obs


def test_memristive_state_accumulates_and_decays():
    cfg = RiskReflexConfig(alpha=0.8, beta=0.7, gamma=0.2, theta=0.3, k=10.0)
    frontend = MemristiveRiskFrontEnd(cfg)

    risky = make_obs(front_lidar_value=0.2, speed=0.9)
    safe = make_obs(front_lidar_value=1.0, speed=0.2)

    first = frontend.step(risky)
    second = frontend.step(risky)
    assert second.memristive_state > first.memristive_state
    assert second.pain > first.pain

    decayed = frontend.step(safe)
    assert decayed.memristive_state < second.memristive_state
    assert 0.0 <= decayed.pain <= 1.0


def test_fresh_safe_observation_stays_low_and_bounded():
    cfg = RiskReflexConfig(alpha=0.8, beta=0.7, gamma=0.2, theta=0.3, k=10.0)
    frontend = MemristiveRiskFrontEnd(cfg)

    safe = make_obs(front_lidar_value=1.0, speed=0.2)

    output = frontend.step(safe)

    assert 0.0 <= output.memristive_state <= cfg.theta
    assert 0.0 <= output.pain <= 0.1


def test_longitudinal_reflex_preserves_steering():
    cfg = RiskReflexConfig(lambda_brake=0.4, enable_steering_limit=False)
    layer = PainReflexLayer(cfg)
    action = np.array([0.25, 0.6], dtype=np.float32)

    adjusted, diag = layer.apply(action, pain=1.0)

    assert np.isclose(adjusted[0], action[0])
    assert adjusted[1] < action[1]
    assert diag["reflex_active"] == 1.0


def test_zero_pain_returns_original_action():
    cfg = RiskReflexConfig(lambda_brake=0.4)
    layer = PainReflexLayer(cfg)
    action = np.array([-0.2, 0.3], dtype=np.float32)

    adjusted, diag = layer.apply(action, pain=0.0)

    assert np.allclose(adjusted, action)
    assert diag["reflex_active"] == 0.0


def test_subthreshold_pain_does_not_adjust_action_or_activate_reflex():
    cfg = RiskReflexConfig(
        lambda_brake=0.4,
        lambda_throttle=0.7,
        trigger_threshold=0.5,
        enable_steering_limit=True,
        steering_limit_gain=0.75,
    )
    layer = PainReflexLayer(cfg)
    action = np.array([0.8, 0.6], dtype=np.float32)

    adjusted, diag = layer.apply(action, pain=0.49)

    assert np.allclose(adjusted, action)
    assert diag["reflex_active"] == 0.0
    assert diag["steering_limited"] == 0.0
    assert np.isclose(diag["longitudinal_delta"], 0.0)


def test_subthreshold_pain_preserves_out_of_range_action_without_clipping():
    cfg = RiskReflexConfig(trigger_threshold=0.5)
    layer = PainReflexLayer(cfg)
    action = np.array([1.5, 1.2], dtype=np.float32)

    adjusted, diag = layer.apply(action, pain=0.49)

    assert np.allclose(adjusted, action)
    assert diag["raw_steer"] == float(action[0])
    assert diag["raw_long"] == float(action[1])
    assert diag["adjusted_steer"] == float(action[0])
    assert diag["adjusted_long"] == float(action[1])
    assert diag["reflex_active"] == 0.0
    assert diag["steering_limited"] == 0.0
    assert np.isclose(diag["longitudinal_delta"], 0.0)


def test_active_positive_throttle_uses_throttle_suppression_and_brake():
    cfg = RiskReflexConfig(
        lambda_brake=0.4,
        lambda_throttle=0.5,
        trigger_threshold=0.5,
        enable_steering_limit=False,
    )
    layer = PainReflexLayer(cfg)
    action = np.array([0.1, 0.8], dtype=np.float32)

    adjusted, diag = layer.apply(action, pain=0.5)

    expected_long = 0.8 * (1.0 - 0.5 * 0.5) - 0.4 * 0.5
    assert np.isclose(adjusted[0], action[0])
    assert np.isclose(adjusted[1], expected_long)
    assert diag["reflex_active"] == 1.0
    assert np.isclose(diag["longitudinal_delta"], action[1] - expected_long)


def test_steering_limit_ablation_clips_steering_symmetrically():
    cfg = RiskReflexConfig(
        lambda_brake=0.0,
        lambda_throttle=0.0,
        enable_steering_limit=True,
        steering_limit_gain=0.75,
        min_steering_scale=0.25,
    )
    layer = PainReflexLayer(cfg)

    for steering in (0.9, -0.9):
        action = np.array([steering, 0.2], dtype=np.float32)

        adjusted, diag = layer.apply(action, pain=1.0)

        assert abs(adjusted[0]) <= 0.25
        assert np.sign(adjusted[0]) == np.sign(action[0])
        assert np.isclose(adjusted[1], action[1])
        assert diag["steering_limited"] == 1.0


def test_controller_applies_reflex_and_resets_episode_diagnostics():
    cfg = RiskReflexConfig(theta=0.2, beta=0.8)
    controller = RiskReflexController(cfg)
    risky = make_obs(front_lidar_value=0.1, speed=0.9)
    action = np.array([0.0, 0.5], dtype=np.float32)

    adjusted, diag = controller.apply(risky, action)

    assert adjusted[1] < action[1]
    assert diag["reflex_active"] == 1.0
    assert controller.summary()["trigger_rate"] > 0.0

    controller.reset()
    summary = controller.summary()
    assert summary["trigger_rate"] == 0.0
    assert summary["mean_pain"] == 0.0
