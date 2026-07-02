from dataclasses import dataclass, fields


@dataclass
class RiskReflexConfig:
    obs_dim: int = 49
    state_nav_dim: int = 19
    num_lasers: int = 30
    alpha: float = 0.92
    beta: float = 0.45
    gamma: float = 0.08
    theta: float = 0.45
    k: float = 12.0
    proximity_weight: float = 0.55
    closing_speed_weight: float = 0.30
    lane_edge_weight: float = 0.15
    lane_edge_margin: float = 0.18
    front_lidar_rays: int = 2
    lambda_brake: float = 0.35
    lambda_throttle: float = 0.35
    max_action: float = 1.0
    enable_steering_limit: bool = False
    steering_limit_gain: float = 0.5
    min_steering_scale: float = 0.35
    trigger_threshold: float = 0.5


def config_from_args(args):
    values = {}
    for field in fields(RiskReflexConfig):
        arg_name = "risk_reflex_" + field.name
        if hasattr(args, arg_name):
            values[field.name] = getattr(args, arg_name)

    if getattr(args, "risk_reflex_variant", None) == "steering_limit":
        values["enable_steering_limit"] = True

    return RiskReflexConfig(**values)
