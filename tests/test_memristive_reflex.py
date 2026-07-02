import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_memristive_frontend import MemristiveRiskReflex, ReflexCfg


class MemristiveRiskReflexTest(unittest.TestCase):
    def _obs_with_front_distance(self, distance_m, num_lasers=30, lidar_range_m=50.0):
        obs = np.ones(49, dtype=np.float32)
        lidar_value = np.float32(distance_m / lidar_range_m)
        for idx in (0, 1, 2, 3, 27, 28, 29):
            obs[-num_lasers + idx] = lidar_value
        return obs

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

    def test_stable_following_risk_does_not_accumulate_into_memory_slowdown(self):
        cfg = ReflexCfg(
            num_lasers=30,
            lidar_range_m=50.0,
            front_distance_threshold_m=10.0,
            stop_distance_m=1.0,
            slow_front_risk_threshold=1.01,
            stop_front_risk_threshold=1.01,
            allow_direct_stop=False,
            front_decay=0.90,
            front_gain=0.50,
            front_risk_soft=0.15,
            front_risk_saturate=0.55,
            front_slow_threshold=0.60,
            front_slow_throttle_cap=0.0,
            lateral_steer_threshold=1.01,
            lateral_throttle_threshold=1.01,
            lateral_gain=0.0,
            lateral_guard_threshold=1.01,
        )
        reflex = MemristiveRiskReflex("nocicim_risk_field", cfg)
        obs = self._obs_with_front_distance(6.0)
        action = np.array([0.0, 0.8], dtype=np.float32)

        for _ in range(8):
            safe, _risk, debug = reflex.apply(action, obs)

        self.assertAlmostEqual(float(safe[1]), 0.8, places=6)
        self.assertEqual(debug["nocicim_front_slow_active"], 0)
        self.assertLess(debug["nocicim_x_front"], cfg.front_slow_threshold)


if __name__ == "__main__":
    unittest.main()
