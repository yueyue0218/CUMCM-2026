import math
import itertools
import random
import unittest

from src.q1 import enclosing_circle
from src.q1.enclosing_circle import minimum_enclosing_circle


def brute_force_minimum_radius(points):
    candidates = []
    for point in points:
        candidates.append(((float(point[0]), float(point[1])), 0.0))
    for first, second in itertools.combinations(points, 2):
        center = (
            (first[0] + second[0]) / 2.0,
            (first[1] + second[1]) / 2.0,
        )
        candidates.append((center, math.dist(first, second) / 2.0))
    for first, second, third in itertools.combinations(points, 3):
        ax, ay = first
        bx, by = second
        cx, cy = third
        denominator = 2.0 * (
            ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
        )
        if denominator == 0.0:
            continue
        first_squared = ax * ax + ay * ay
        second_squared = bx * bx + by * by
        third_squared = cx * cx + cy * cy
        center = (
            (
                first_squared * (by - cy)
                + second_squared * (cy - ay)
                + third_squared * (ay - by)
            ) / denominator,
            (
                first_squared * (cx - bx)
                + second_squared * (ax - cx)
                + third_squared * (bx - ax)
            ) / denominator,
        )
        candidates.append((center, math.dist(center, first)))

    covering_radii = [
        radius for center, radius in candidates
        if all(math.dist(center, point) <= radius + 1e-9 for point in points)
    ]
    return min(covering_radii, default=0.0)


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

    def test_six_point_forced_boundary_regression(self):
        points = [
            (17, -11), (-3, -20), (-10, 8),
            (-17, -6), (-4, 11), (16, -13),
        ]
        circle = minimum_enclosing_circle(points)
        self.assertAlmostEqual(
            circle.radius, brute_force_minimum_radius(points), places=10
        )
        self.assertLessEqual(circle.max_residual, 1e-9)

    def test_matches_independent_brute_force_oracle_on_fixed_seed_sets(self):
        generator = random.Random(20260911)
        for case_number in range(100):
            points = [
                (generator.randint(-30, 30), generator.randint(-30, 30))
                for _ in range(generator.randint(1, 9))
            ]
            with self.subTest(case_number=case_number, points=points):
                circle = minimum_enclosing_circle(points)
                self.assertAlmostEqual(
                    circle.radius,
                    brute_force_minimum_radius(points),
                    places=9,
                )
                self.assertLessEqual(circle.max_residual, 1e-9)

    def test_collinear_forced_boundary_fallback_keeps_prefix_points(self):
        circle = enclosing_circle._collinear_boundary_fallback(
            prefix=[(0.0, 10.0)],
            boundary=((-2.0, 0.0), (0.0, 0.0), (2.0, 0.0)),
        )
        self.assertAlmostEqual(circle.center[0], 0.0, places=12)
        self.assertAlmostEqual(circle.center[1], 4.8, places=12)
        self.assertAlmostEqual(circle.radius, 5.2, places=12)
        for point in [(0.0, 10.0), (-2.0, 0.0), (0.0, 0.0), (2.0, 0.0)]:
            self.assertLessEqual(math.dist(circle.center, point), circle.radius)


if __name__ == "__main__":
    unittest.main()
