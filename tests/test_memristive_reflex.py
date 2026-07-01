import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_memristive_frontend import MemristiveRiskReflex, ReflexCfg


class MemristiveRiskReflexTest(unittest.TestCase):
    def test_stop_brake_value_controls_front_stop_clamp(self):
        cfg = ReflexCfg(num_lasers=30, stop_brake_value=0.0)
        reflex = MemristiveRiskReflex("nocicim_risk_field", cfg)
        obs = np.ones(49, dtype=np.float32)
        obs[-30] = 0.01

        safe, _risk, debug = reflex.apply(np.array([0.0, 1.0], dtype=np.float32), obs)

        self.assertEqual(float(safe[1]), 0.0)
        self.assertEqual(debug["intervention"], 1)

    def test_default_front_stop_remains_hard_brake(self):
        reflex = MemristiveRiskReflex("nocicim_risk_field", ReflexCfg(num_lasers=30))
        obs = np.ones(49, dtype=np.float32)
        obs[-30] = 0.01

        safe, _risk, debug = reflex.apply(np.array([0.0, 1.0], dtype=np.float32), obs)

        self.assertEqual(float(safe[1]), -1.0)
        self.assertEqual(debug["intervention"], 1)


if __name__ == "__main__":
    unittest.main()
