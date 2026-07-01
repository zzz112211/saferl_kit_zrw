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


if __name__ == "__main__":
    unittest.main()
