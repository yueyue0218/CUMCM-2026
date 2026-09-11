import math
import unittest

from src.q1.enclosing_circle import minimum_enclosing_circle


class MinimumEnclosingCircleTests(unittest.TestCase):
    def test_equilateral_triangle_reaches_jung_radius(self):
        points = [(0, 0), (36, 0), (18, 18 * math.sqrt(3))]
        circle = minimum_enclosing_circle(points)
        self.assertAlmostEqual(circle.center[0], 18.0, places=10)
        self.assertAlmostEqual(circle.center[1], 6 * math.sqrt(3), places=10)
        self.assertAlmostEqual(circle.radius, 12 * math.sqrt(3), places=10)
        self.assertLessEqual(circle.max_residual, 1e-9)

    def test_collinear_points_use_longest_pair(self):
        circle = minimum_enclosing_circle([(-2, 0), (5, 0), (1, 0), (5, 0)])
        self.assertEqual(circle.center, (1.5, 0.0))
        self.assertEqual(circle.radius, 3.5)
        self.assertEqual(circle.max_residual, 0.0)

    def test_duplicate_points_match_the_distinct_input(self):
        distinct = [(0, 0), (4, 0), (0, 3)]
        duplicated = [(0, 0), (4, 0), (0, 3), (4, 0), (0, 0)]
        self.assertEqual(
            minimum_enclosing_circle(duplicated),
            minimum_enclosing_circle(distinct),
        )

    def test_fixed_seed_is_reproducible_without_mutating_input(self):
        points = [(0, 0), (4, 0), (4, 2), (0, 2), (1, 1)]
        original = list(points)
        first = minimum_enclosing_circle(points, seed=12345)
        second = minimum_enclosing_circle(points, seed=12345)
        self.assertEqual(first, second)
        self.assertEqual(points, original)

    def test_empty_and_singleton_inputs_have_zero_radius(self):
        empty = minimum_enclosing_circle([])
        singleton = minimum_enclosing_circle([(2, -3)])
        self.assertEqual(empty.center, (0.0, 0.0))
        self.assertEqual(empty.radius, 0.0)
        self.assertEqual(empty.max_residual, 0.0)
        self.assertEqual(singleton.center, (2.0, -3.0))
        self.assertEqual(singleton.radius, 0.0)
        self.assertEqual(singleton.max_residual, 0.0)

    def test_non_finite_point_is_rejected(self):
        with self.assertRaises(ValueError):
            minimum_enclosing_circle([(0, 0), (math.inf, 1)])


if __name__ == "__main__":
    unittest.main()
