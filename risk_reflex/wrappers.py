from collections import defaultdict

from .configs import RiskReflexConfig
from .reflex_layer import PainReflexLayer
from .risk_frontend import MemristiveRiskFrontEnd


class RiskReflexController:
    def __init__(self, config=None):
        self.config = config or RiskReflexConfig()
        self.frontend = MemristiveRiskFrontEnd(self.config)
        self.layer = PainReflexLayer(self.config)
        self.reset()

    def reset(self):
        self.frontend.reset()
        self.steps = 0
        self.totals = defaultdict(float)

    def apply(self, observation, action):
        risk_state = self.frontend.step(observation)
        adjusted, diag = self.layer.apply(action, risk_state.pain)
        diag.update(
            {
                "instantaneous_risk": risk_state.instantaneous_risk,
                "proximity_risk": risk_state.proximity_risk,
                "closing_speed_risk": risk_state.closing_speed_risk,
                "lane_edge_risk": risk_state.lane_edge_risk,
                "memristive_state": risk_state.memristive_state,
            }
        )

        self.steps += 1
        self.totals["risk"] += risk_state.instantaneous_risk
        self.totals["pain"] += risk_state.pain
        self.totals["trigger"] += diag["reflex_active"]
        self.totals["longitudinal_delta"] += diag["longitudinal_delta"]
        self.totals["steering_limited"] += diag["steering_limited"]
        return adjusted, diag

    def summary(self):
        if self.steps == 0:
            return {
                "mean_risk": 0.0,
                "mean_pain": 0.0,
                "trigger_rate": 0.0,
                "mean_longitudinal_delta": 0.0,
                "steering_limit_rate": 0.0,
            }

        return {
            "mean_risk": self.totals["risk"] / self.steps,
            "mean_pain": self.totals["pain"] / self.steps,
            "trigger_rate": self.totals["trigger"] / self.steps,
            "mean_longitudinal_delta": self.totals["longitudinal_delta"] / self.steps,
            "steering_limit_rate": self.totals["steering_limited"] / self.steps,
        }
