import math
import random
import unittest

from src.q1.geometry import convex_hull
from src.q1.measures import (
    CircularArc,
    LineSegment,
    diameter_circle_coverage,
    diameter_exhaustive,
    diameter_rotating_calipers,
    green_area_centroid,
    polygon_area_centroid,
)


class DiameterTests(unittest.TestCase):
    def test_rectangle_calipers_matches_literal_diagonal(self):
        vertices = [(0, 0), (4, 0), (4, 3), (0, 3)]
        result = diameter_rotating_calipers(vertices)
        self.assertAlmostEqual(result.distance, 5.0, places=12)
        self.assertEqual(result.squared_distance, 25.0)
        self.assertEqual(result.witness, ((0, 0), (4, 3)))

    def test_equilateral_triangle_diameter_circle_does_not_cover(self):
        vertices = [(0, 0), (36, 0), (18, 18 * math.sqrt(3))]
        result = diameter_rotating_calipers(vertices)
        coverage = diameter_circle_coverage(vertices, result)
        self.assertFalse(coverage.covers)
        self.assertGreater(coverage.max_distance, 18.0)

    def test_degenerate_vertex_counts_have_deterministic_witnesses(self):
        cases = [
            ([], 0.0, None),
            ([(2, -1)], 0.0, ((2, -1), (2, -1))),
            ([(3, 4), (-1, 1)], 25.0, ((-1, 1), (3, 4))),
        ]
        for vertices, squared_distance, witness in cases:
            with self.subTest(vertices=vertices):
                result = diameter_rotating_calipers(vertices)
                self.assertEqual(result.squared_distance, squared_distance)
                self.assertEqual(result.witness, witness)

    def test_calipers_matches_exhaustive_for_fixed_seed_integer_hulls(self):
        generator = random.Random(20260911)
        for case_number in range(100):
            points = [
                (generator.randint(-50, 50), generator.randint(-50, 50))
                for _ in range(generator.randint(0, 30))
            ]
            vertices = convex_hull(points, 0.0)
            with self.subTest(case_number=case_number, vertices=vertices):
                exhaustive = diameter_exhaustive(vertices)
                calipers = diameter_rotating_calipers(vertices)
                self.assertEqual(calipers.squared_distance,
                                 exhaustive.squared_distance)
                self.assertEqual(calipers.witness, exhaustive.witness)


class GreenMeasureTests(unittest.TestCase):
    def test_rectangle_green_measure_has_literal_centroid(self):
        area, centroid = polygon_area_centroid(
            [(0, 0), (4, 0), (4, 2), (0, 2)]
        )
        self.assertAlmostEqual(area, 8.0, places=12)
        self.assertAlmostEqual(centroid[0], 2.0, places=12)
        self.assertAlmostEqual(centroid[1], 1.0, places=12)

    def test_full_circle_arc_has_exact_area_and_center(self):
        area, centroid = green_area_centroid([
            CircularArc(
                center=(3, -2), radius=5, start_angle=0,
                sweep_angle=2 * math.pi,
            )
        ])
        self.assertAlmostEqual(area, 25 * math.pi, places=10)
        self.assertAlmostEqual(centroid[0], 3.0, places=10)
        self.assertAlmostEqual(centroid[1], -2.0, places=10)

    def test_upper_semicircle_combines_arc_and_diameter(self):
        area, centroid = green_area_centroid([
            CircularArc((0, 0), 3, 0, math.pi),
            LineSegment((-3, 0), (3, 0)),
        ])
        self.assertAlmostEqual(area, 4.5 * math.pi, places=12)
        self.assertAlmostEqual(centroid[0], 0.0, places=12)
        self.assertAlmostEqual(centroid[1], 4.0 / math.pi, places=12)

    def test_clockwise_polygon_returns_signed_area_and_same_centroid(self):
        area, centroid = polygon_area_centroid(
            [(0, 2), (4, 2), (4, 0), (0, 0)]
        )
        self.assertAlmostEqual(area, -8.0, places=12)
        self.assertEqual(centroid, (2.0, 1.0))

    def test_invalid_arc_radius_and_sweep_are_rejected(self):
        invalid_values = [
            (0, 0), (-1, 0), (1, -0.1),
            (1, 2 * math.pi + 0.1), (float("inf"), 0),
        ]
        for radius, sweep in invalid_values:
            with (self.subTest(radius=radius, sweep=sweep),
                  self.assertRaises(ValueError)):
                CircularArc((0, 0), radius, 0, sweep)

    def test_open_boundary_is_rejected(self):
        with self.assertRaises(ValueError):
            green_area_centroid([
                LineSegment((0, 0), (1, 0)),
                LineSegment((1, 0), (1, 1)),
            ])

    def test_zero_sweep_arc_is_distinct_from_full_circle(self):
        with self.assertRaises(ValueError):
            green_area_centroid([CircularArc((0, 0), 2, 0, 0)])

    def test_translated_unit_square_retains_area_and_centroid(self):
        area, centroid = polygon_area_centroid([
            (1_000_000, 0), (1_000_001, 0),
            (1_000_001, 1), (1_000_000, 1),
        ])
        self.assertAlmostEqual(area, 1.0, places=12)
        self.assertAlmostEqual(centroid[0], 1_000_000.5, places=12)
        self.assertAlmostEqual(centroid[1], 0.5, places=12)

    def test_translated_full_circle_retains_area_and_center(self):
        area, centroid = green_area_centroid([
            CircularArc((1e12, -1e12), 2, 0, 2 * math.pi)
        ])
        self.assertAlmostEqual(area, 4 * math.pi, places=10)
        self.assertAlmostEqual(centroid[0], 1e12, places=10)
        self.assertAlmostEqual(centroid[1], -1e12, places=10)

    def test_large_coordinate_boundary_with_half_metre_gap_is_open(self):
        origin = 1e12
        side = 2_000_000
        with self.assertRaises(ValueError):
            green_area_centroid([
                LineSegment((origin, origin), (origin + side, origin)),
                LineSegment(
                    (origin + side, origin), (origin + side, origin + side)
                ),
                LineSegment(
                    (origin + side, origin + side), (origin, origin + side)
                ),
                LineSegment((origin, origin + side), (origin + 0.5, origin)),
            ])


if __name__ == "__main__":
    unittest.main()
