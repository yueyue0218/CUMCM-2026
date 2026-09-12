import math
import unittest

from src.common.geometry import polygon_diameter
from src.common.localization import BearingObservation, localization_region


class Q1ProductionConsistencyTests(unittest.TestCase):
    def test_symmetric_bearings_match_analytic_diameter(self) -> None:
        observations = [
            BearingObservation((0.0, 0.0), 45.0, 1.0),
            BearingObservation((1000.0, 0.0), 135.0, 1.0),
        ]

        region = localization_region(observations)
        expected_diameter_m = 500.0 * (
            math.tan(math.radians(46.0)) - math.tan(math.radians(44.0))
        )

        self.assertTrue(region)
        self.assertTrue(
            all(math.dist(point, (0.0, 0.0)) < 1800.0 for point in region)
        )
        for observation in observations:
            self.assertTrue(
                all(
                    math.dist(point, observation.station) < 1500.0
                    for point in region
                )
            )
        # At a 1 km coordinate scale, 1e-9 m safely covers accumulated binary64
        # clipping/intersection roundoff without masking a geometric discrepancy.
        self.assertAlmostEqual(
            polygon_diameter(region), expected_diameter_m, delta=1e-9
        )


if __name__ == "__main__":
    unittest.main()
