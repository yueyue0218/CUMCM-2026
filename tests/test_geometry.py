import math
import unittest

from src.common.geometry import (
    candidate_second_points,
    clip_polygon_to_bearing_wedge,
    normalize_angle_deg,
    polygon_diameter,
)


class GeometryTests(unittest.TestCase):
    def test_angle_wraparound(self) -> None:
        self.assertEqual(normalize_angle_deg(360.0), 0.0)
        self.assertEqual(normalize_angle_deg(-1.0), 359.0)

    def test_one_degree_wedge_crosses_zero(self) -> None:
        square = [(-20.0, -20.0), (20.0, -20.0), (20.0, 20.0), (-20.0, 20.0)]
        region = clip_polygon_to_bearing_wedge(square, (0.0, 0.0), 0.0, 1.0)
        self.assertTrue(region)
        self.assertTrue(all(point[0] >= -1e-8 for point in region))
        self.assertTrue(any(point[0] > 10 for point in region))

    def test_polygon_diameter(self) -> None:
        square = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        self.assertAlmostEqual(polygon_diameter(square), math.sqrt(8.0))

    def test_second_points_are_perpendicular_and_symmetric(self) -> None:
        candidates = candidate_second_points((3.0, 4.0), 0.0, [10.0], [5.0])
        self.assertCountEqual(candidates, [(8.0, -6.0), (8.0, 14.0)])


if __name__ == "__main__":
    unittest.main()
