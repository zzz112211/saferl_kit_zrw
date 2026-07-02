import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_metadrive


class NocicimTrainProfileTest(unittest.TestCase):
    def test_close_range_profile_matches_sweep_candidate(self):
        cfg = train_metadrive.build_nocicim_reflex_cfg("close_range")

        self.assertEqual(cfg.front_distance_threshold_m, 6.0)
        self.assertEqual(cfg.stop_distance_m, 2.5)
        self.assertEqual(cfg.lateral_distance_threshold_m, 3.5)
        self.assertEqual(cfg.slow_front_risk_threshold, 0.70)
        self.assertEqual(cfg.stop_front_risk_threshold, 0.92)
        self.assertEqual(cfg.lateral_steer_threshold, 0.70)
        self.assertEqual(cfg.lateral_throttle_threshold, 0.80)
        self.assertEqual(cfg.lateral_throttle_cap, 0.55)
        self.assertEqual(cfg.lateral_steering_cap, 0.90)
        self.assertEqual(cfg.front_gain, 0.06)
        self.assertEqual(cfg.front_slow_threshold, 0.88)
        self.assertEqual(cfg.front_slow_throttle_cap, 0.90)
        self.assertEqual(cfg.lateral_gain, 0.10)
        self.assertEqual(cfg.lateral_guard_threshold, 0.90)

    def test_panic_front_only_profile_matches_low_intervention_sweep_candidate(self):
        cfg = train_metadrive.build_nocicim_reflex_cfg("panic_front_only")

        self.assertEqual(cfg.front_distance_threshold_m, 4.5)
        self.assertEqual(cfg.stop_distance_m, 1.8)
        self.assertEqual(cfg.lateral_distance_threshold_m, 1.0)
        self.assertEqual(cfg.slow_front_risk_threshold, 1.01)
        self.assertEqual(cfg.stop_front_risk_threshold, 0.98)
        self.assertEqual(cfg.lateral_steer_threshold, 1.01)
        self.assertEqual(cfg.lateral_throttle_threshold, 1.01)
        self.assertEqual(cfg.front_slow_threshold, 1.01)
        self.assertEqual(cfg.lateral_guard_threshold, 1.01)

    def test_panic_soft_neutral_profile_uses_neutral_stop_clamp(self):
        cfg = train_metadrive.build_nocicim_reflex_cfg("panic_soft_neutral")

        self.assertEqual(cfg.front_distance_threshold_m, 4.5)
        self.assertEqual(cfg.stop_distance_m, 1.8)
        self.assertEqual(cfg.stop_brake_value, 0.0)
        self.assertEqual(cfg.slow_front_risk_threshold, 1.01)
        self.assertEqual(cfg.lateral_guard_threshold, 1.01)

    def test_panic_soft_front_only_profile_uses_soft_throttle_clamp(self):
        cfg = train_metadrive.build_nocicim_reflex_cfg("panic_soft_front_only")

        self.assertEqual(cfg.front_distance_threshold_m, 4.5)
        self.assertEqual(cfg.stop_distance_m, 1.8)
        self.assertEqual(cfg.stop_brake_value, 0.6)
        self.assertEqual(cfg.slow_front_risk_threshold, 1.01)
        self.assertEqual(cfg.lateral_steer_threshold, 1.01)
        self.assertEqual(cfg.lateral_throttle_threshold, 1.01)
        self.assertEqual(cfg.lateral_gain, 0.0)
        self.assertEqual(cfg.lateral_guard_threshold, 1.01)


if __name__ == "__main__":
    unittest.main()
