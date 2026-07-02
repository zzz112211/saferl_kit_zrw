import numpy as np

from .configs import RiskReflexConfig


class PainReflexLayer:
    def __init__(self, config=None):
        self.config = config or RiskReflexConfig()

    def apply(self, action, pain):
        cfg = self.config
        adjusted = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if adjusted.size < 2:
            raise ValueError(f"action must contain at least 2 values, got {adjusted.size}")

        raw_steer = float(adjusted[0])
        raw_long = float(adjusted[1])
        pain_value = float(np.clip(pain, 0.0, 1.0))
        active = pain_value >= float(cfg.trigger_threshold)
        effective_pain = pain_value if active else 0.0

        if not active:
            diag = {
                "raw_steer": raw_steer,
                "raw_long": raw_long,
                "adjusted_steer": raw_steer,
                "adjusted_long": raw_long,
                "pain": pain_value,
                "reflex_active": 0.0,
                "steering_limited": 0.0,
                "longitudinal_delta": 0.0,
            }
            return adjusted, diag

        if raw_long > 0.0:
            throttle_scale = 1.0 - float(cfg.lambda_throttle) * effective_pain
            adjusted[1] = adjusted[1] * throttle_scale
        adjusted[1] = adjusted[1] - float(cfg.lambda_brake) * effective_pain

        steering_limited = 0.0
        if cfg.enable_steering_limit:
            scale = max(float(cfg.min_steering_scale), 1.0 - float(cfg.steering_limit_gain) * effective_pain)
            steering_limit = float(cfg.max_action) * scale
            limited = float(np.clip(adjusted[0], -steering_limit, steering_limit))
            steering_limited = 1.0 if not np.isclose(limited, raw_steer) else 0.0
            adjusted[0] = limited

        adjusted = np.clip(adjusted, -float(cfg.max_action), float(cfg.max_action)).astype(np.float32)

        diag = {
            "raw_steer": raw_steer,
            "raw_long": raw_long,
            "adjusted_steer": float(adjusted[0]),
            "adjusted_long": float(adjusted[1]),
            "pain": pain_value,
            "reflex_active": 1.0 if active else 0.0,
            "steering_limited": steering_limited,
            "longitudinal_delta": raw_long - float(adjusted[1]),
        }
        return adjusted, diag
