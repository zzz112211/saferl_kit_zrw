from dataclasses import dataclass

import numpy as np

from .configs import RiskReflexConfig


@dataclass
class RiskState:
    instantaneous_risk: float
    proximity_risk: float
    closing_speed_risk: float
    lane_edge_risk: float
    memristive_state: float
    pain: float


class MemristiveRiskFrontEnd:
    def __init__(self, config=None):
        self.config = config or RiskReflexConfig()
        self.reset()

    def reset(self):
        self.memristive_state = 0.0
        self.prev_front_clearance = None

    def step(self, observation):
        cfg = self.config
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        required = cfg.state_nav_dim + cfg.num_lasers
        if obs.size < required:
            raise ValueError(
                "observation must contain at least "
                f"{required} values, got {obs.size}"
            )

        lidar = obs[-cfg.num_lasers :]
        rays = max(1, min(int(cfg.front_lidar_rays), cfg.num_lasers))
        front_values = np.concatenate([lidar[:rays], lidar[-rays:]])
        front_clearance = float(np.clip(np.min(front_values), 0.0, 1.0))
        speed = float(np.clip(obs[3], 0.0, 1.0)) if obs.size > 3 else 0.0

        proximity_risk = float(np.clip(1.0 - front_clearance, 0.0, 1.0))
        if self.prev_front_clearance is None:
            closing_speed_risk = 0.0
        else:
            clearance_drop = max(0.0, self.prev_front_clearance - front_clearance)
            closing_speed_risk = float(np.clip(clearance_drop * speed, 0.0, 1.0))

        lane_clearance = float(np.clip(min(float(obs[0]), float(obs[1])), 0.0, 1.0))
        if cfg.lane_edge_margin <= 0.0:
            lane_edge_risk = 0.0
        else:
            lane_edge_risk = float(
                np.clip((cfg.lane_edge_margin - lane_clearance) / cfg.lane_edge_margin, 0.0, 1.0)
            )

        risk = float(
            np.clip(
                cfg.proximity_weight * proximity_risk
                + cfg.closing_speed_weight * closing_speed_risk
                + cfg.lane_edge_weight * lane_edge_risk,
                0.0,
                1.0,
            )
        )
        recovery = max(0.0, 1.0 - risk)
        self.memristive_state = float(
            np.clip(
                cfg.alpha * self.memristive_state + cfg.beta * risk - cfg.gamma * recovery,
                0.0,
                1.0,
            )
        )
        pain = float(1.0 / (1.0 + np.exp(-cfg.k * (self.memristive_state - cfg.theta))))
        self.prev_front_clearance = front_clearance

        return RiskState(
            instantaneous_risk=risk,
            proximity_risk=proximity_risk,
            closing_speed_risk=closing_speed_risk,
            lane_edge_risk=lane_edge_risk,
            memristive_state=self.memristive_state,
            pain=pain,
        )
