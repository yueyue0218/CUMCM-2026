import unittest

from src.q1.geometry import (
    HalfPlane,
    Observation,
    bearing_halfplanes,
    intersect_halfplanes,
    point_satisfies,
)


class BearingHalfPlaneTests(unittest.TestCase):
    def test_cross_zero_wedge_keeps_east_point(self):
        planes = bearing_halfplanes(Observation((0.0, 0.0), 359.5, 1.0))
        self.assertTrue(point_satisfies((100.0, 0.0), planes))
        self.assertFalse(point_satisfies((-100.0, 0.0), planes))

    def test_axis_aligned_box_is_classified_as_polygon(self):
        planes = [HalfPlane(1, 0, 2), HalfPlane(-1, 0, 0),
                  HalfPlane(0, 1, 1), HalfPlane(0, -1, 0)]
        region = intersect_halfplanes(planes)
        self.assertEqual(region.status, "polygon")
        self.assertEqual(len(region.vertices), 4)
        self.assertLessEqual(region.max_violation, 1e-9)

    def test_contradictory_bounds_are_empty(self):
        region = intersect_halfplanes([HalfPlane(1, 0, 0), HalfPlane(-1, 0, -1)])
        self.assertEqual(region.status, "empty")

    def test_one_wedge_is_unbounded(self):
        region = intersect_halfplanes(bearing_halfplanes(Observation((0, 0), 45)))
        self.assertEqual(region.status, "unbounded")

    def test_equalities_can_reduce_region_to_segment(self):
        planes = [HalfPlane(0, 1, 0), HalfPlane(0, -1, 0),
                  HalfPlane(1, 0, 2), HalfPlane(-1, 0, 0)]
        region = intersect_halfplanes(planes)
        self.assertEqual(region.status, "segment")
        self.assertEqual(region.vertices, ((0.0, 0.0), (2.0, 0.0)))

    def test_non_finite_observation_is_rejected(self):
        with self.assertRaises(ValueError):
            Observation((float("nan"), 0.0), 10.0)


if __name__ == "__main__":
    unittest.main()
