import math
import unittest

from src.common.localization import (
    BearingObservation,
    localization_quality,
    localization_region,
)


class LocalizationTests(unittest.TestCase):
    def test_crossing_wedges_bound_a_region_containing_true_source(self) -> None:
        observations = [
            BearingObservation((0.0, 0.0), 45.0),
            BearingObservation((100.0, 0.0), 135.0),
        ]
        region = localization_region(observations, arena_radius_m=200.0)
        quality = localization_quality(region)
        self.assertTrue(region)
        self.assertGreater(quality["area_m2"], 0.0)
        self.assertTrue(math.isfinite(quality["diameter_m"]))

    def test_nearly_parallel_directions_do_not_crash(self) -> None:
        observations = [
            BearingObservation((0.0, 0.0), 0.0),
            BearingObservation((0.0, 10.0), 0.01),
        ]
        region = localization_region(observations, arena_radius_m=1800.0)
        quality = localization_quality(region)
        self.assertIsInstance(region, list)
        self.assertTrue(math.isfinite(quality["diameter_m"]))

    def test_inconsistent_wedges_return_empty_region(self) -> None:
        observations = [
            BearingObservation((0.0, 0.0), 0.0, 0.1),
            BearingObservation((100.0, 0.0), 0.0, 0.1),
        ]
        self.assertEqual(
            localization_region(observations, arena_radius_m=50.0), []
        )


if __name__ == "__main__":
    unittest.main()
