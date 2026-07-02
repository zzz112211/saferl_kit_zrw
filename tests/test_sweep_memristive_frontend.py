import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sweep_memristive_frontend import candidate_by_name


class SweepMemristiveFrontendTest(unittest.TestCase):
    def test_includes_panic_soft_neutral_candidate(self):
        candidate = candidate_by_name()["nocicim_panic_soft_neutral"]
        overrides = candidate["overrides"]

        self.assertEqual(candidate["frontend"], "nocicim_risk_field")
        self.assertEqual(overrides["front_distance_threshold_m"], 4.5)
        self.assertEqual(overrides["stop_distance_m"], 1.8)
        self.assertEqual(overrides["stop_brake_value"], 0.0)
        self.assertEqual(overrides["slow_front_risk_threshold"], 1.01)
        self.assertEqual(overrides["lateral_guard_threshold"], 1.01)

    def test_includes_panic_soft_front_only_candidate(self):
        candidate = candidate_by_name()["nocicim_panic_soft_front_only"]
        overrides = candidate["overrides"]

        self.assertEqual(candidate["frontend"], "nocicim_risk_field")
        self.assertEqual(overrides["front_distance_threshold_m"], 4.5)
        self.assertEqual(overrides["stop_distance_m"], 1.8)
        self.assertEqual(overrides["stop_brake_value"], 0.6)
        self.assertEqual(overrides["slow_front_risk_threshold"], 1.01)
        self.assertEqual(overrides["lateral_steer_threshold"], 1.01)
        self.assertEqual(overrides["lateral_throttle_threshold"], 1.01)
        self.assertEqual(overrides["lateral_gain"], 0.0)
        self.assertEqual(overrides["lateral_guard_threshold"], 1.01)

    def test_includes_soft_crash_guard_candidate(self):
        candidate = candidate_by_name()["nocicim_soft_crash_guard"]
        overrides = candidate["overrides"]

        self.assertEqual(candidate["frontend"], "nocicim_risk_field")
        self.assertFalse(overrides["allow_direct_stop"])
        self.assertGreaterEqual(overrides["slow_front_risk_threshold"], 0.80)
        self.assertGreaterEqual(overrides["lateral_throttle_cap"], 0.70)

    def test_includes_crash_aware_route_candidate(self):
        candidate = candidate_by_name()["nocicim_crash_aware_route"]
        overrides = candidate["overrides"]

        self.assertEqual(candidate["frontend"], "nocicim_risk_field")
        self.assertTrue(overrides["require_risk_for_lateral_guard"])
        self.assertGreaterEqual(overrides["min_lateral_risk_for_guard"], 0.20)
        self.assertGreaterEqual(overrides["front_slow_throttle_cap"], 0.92)

    def test_includes_conservative_clean_candidate(self):
        candidate = candidate_by_name()["nocicim_conservative_clean"]
        overrides = candidate["overrides"]

        self.assertEqual(candidate["frontend"], "nocicim_risk_field")
        self.assertGreaterEqual(overrides["slow_front_risk_threshold"], 0.95)
        self.assertGreaterEqual(overrides["stop_front_risk_threshold"], 0.99)
        self.assertGreaterEqual(overrides["lateral_guard_threshold"], 0.98)

    def test_includes_monitor_only_candidate(self):
        candidate = candidate_by_name()["nocicim_monitor_only"]
        overrides = candidate["overrides"]

        self.assertEqual(candidate["frontend"], "nocicim_risk_field")
        self.assertFalse(overrides["allow_direct_stop"])
        self.assertEqual(overrides["front_gain"], 0.0)
        self.assertEqual(overrides["lateral_gain"], 0.0)
        self.assertGreater(overrides["slow_front_risk_threshold"], 1.0)
        self.assertGreater(overrides["lateral_steer_threshold"], 1.0)


if __name__ == "__main__":
    unittest.main()
