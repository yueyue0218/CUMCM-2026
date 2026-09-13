import math
import random
import unittest
from itertools import combinations

from src.common.geometry import Circle, minimum_enclosing_circle

Point = tuple[float, float]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _diameter_circle(a: Point, b: Point) -> Circle:
    center = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    return Circle(center, _distance(a, b) / 2.0)


def _circumcircle(a: Point, b: Point, c: Point) -> Circle | None:
    bx, by = b[0] - a[0], b[1] - a[1]
    cx, cy = c[0] - a[0], c[1] - a[1]
    denominator = 2.0 * (bx * cy - by * cx)
    if denominator == 0.0:
        return None

    b_squared = bx * bx + by * by
    c_squared = cx * cx + cy * cy
    center = (
        a[0] + (cy * b_squared - by * c_squared) / denominator,
        a[1] + (bx * c_squared - cx * b_squared) / denominator,
    )
    radius = max(_distance(center, point) for point in (a, b, c))
    if not all(math.isfinite(value) for value in (*center, radius)):
        return None
    return Circle(center, radius)


def _contains_all(circle: Circle, points: list[Point], tolerance: float) -> bool:
    return all(
        _distance(circle.center, point) <= circle.radius + tolerance
        for point in points
    )


def minimum_enclosing_circle_exhaustive_oracle(
    points: list[Point],
) -> Circle:
    """Find the MEC by enumerating every one-, two-, and three-point support."""

    if not points:
        raise ValueError("MEC oracle requires at least one point")
    if not all(math.isfinite(coordinate) for point in points for coordinate in point):
        raise ValueError("MEC oracle requires finite coordinates")

    unique_points = sorted(set(points))
    origin = unique_points[0]
    local_points = [
        (point[0] - origin[0], point[1] - origin[1])
        for point in unique_points
    ]
    scale = max(
        1.0,
        max(abs(coordinate) for point in local_points for coordinate in point),
    )
    coverage_tolerance = 1e-12 * scale
    best: Circle | None = None

    def consider(candidate: Circle | None) -> None:
        nonlocal best
        if candidate is None or not _contains_all(
            candidate, local_points, coverage_tolerance
        ):
            return
        if best is None or candidate.radius < best.radius:
            best = candidate

    for point in local_points:
        consider(Circle(point, 0.0))
    for a, b in combinations(local_points, 2):
        consider(_diameter_circle(a, b))
    for a, b, c in combinations(local_points, 3):
        consider(_circumcircle(a, b, c))

    if best is None:
        raise AssertionError("support enumeration found no enclosing circle")
    return Circle(
        (best.center[0] + origin[0], best.center[1] + origin[1]),
        best.radius,
    )


class MinimumEnclosingCircleOracleTests(unittest.TestCase):
    @staticmethod
    def _coordinate_tolerance(points: list[Point], radius: float) -> float:
        coordinate_scale = max(
            1.0,
            max(abs(coordinate) for point in points for coordinate in point),
        )
        return max(1e-9, 8.0 * math.ulp(coordinate_scale), 1e-10 * radius)

    def assert_matches_oracle(self, points: list[Point]) -> tuple[Circle, Circle]:
        production = minimum_enclosing_circle(points)
        oracle = minimum_enclosing_circle_exhaustive_oracle(points)
        tolerance = self._coordinate_tolerance(
            points, max(production.radius, oracle.radius)
        )
        production_covers = _contains_all(production, points, tolerance)
        oracle_covers = _contains_all(oracle, points, tolerance)
        details = (
            f"points={points}, production={production}, oracle={oracle}, "
            f"radius_difference={production.radius - oracle.radius}, "
            f"production_covers={production_covers}, "
            f"oracle_covers={oracle_covers}, tolerance={tolerance}"
        )
        self.assertTrue(production_covers, details)
        self.assertTrue(oracle_covers, details)
        self.assertTrue(
            math.isclose(
                production.radius,
                oracle.radius,
                rel_tol=1e-10,
                abs_tol=tolerance,
            ),
            details,
        )
        return production, oracle

    def test_manual_geometric_cases(self) -> None:
        regular_hexagon = [
            (10.0 * math.cos(k * math.pi / 3.0),
             10.0 * math.sin(k * math.pi / 3.0))
            for k in range(6)
        ]
        many_interior_points = [
            (-10.0, -10.0),
            (10.0, -10.0),
            (10.0, 10.0),
            (-10.0, 10.0),
            *[
                (float(x), float(y))
                for x in range(-4, 5, 2)
                for y in range(-4, 5, 2)
            ],
        ]
        cases = {
            "single_point": [(3.0, 4.0)],
            "two_points": [(0.0, 0.0), (6.0, 8.0)],
            "three_collinear": [(0.0, 0.0), (4.0, 0.0), (10.0, 0.0)],
            "acute_triangle": [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)],
            "right_triangle": [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)],
            "obtuse_triangle": [(0.0, 0.0), (6.0, 0.0), (1.0, 1.0)],
            "square": [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)],
            "rectangle": [(0.0, 0.0), (12.0, 0.0), (12.0, 3.0), (0.0, 3.0)],
            "regular_hexagon": regular_hexagon,
            "many_interior_points": many_interior_points,
            "duplicate_points": [(1.0, 2.0), (1.0, 2.0), (5.0, 2.0), (5.0, 2.0)],
            "unordered_input": [(3.0, 4.0), (-2.0, 1.0), (5.0, -3.0), (0.0, 0.0)],
            "near_collinear": [
                (0.0, 0.0),
                (2.0, 2e-11),
                (5.0, -1e-11),
                (7.0, 3e-11),
                (10.0, 0.0),
            ],
            "asymmetric": [
                (-8.0, 1.0),
                (-1.5, 7.25),
                (6.0, 3.0),
                (9.0, -4.0),
                (0.5, -6.5),
                (2.0, 1.0),
            ],
        }
        for name, points in cases.items():
            with self.subTest(name=name):
                self.assert_matches_oracle(points)

    def test_fixed_seed_random_sets_match_exhaustive_oracle(self) -> None:
        for seed in (20260910, 20260911, 20260912):
            rng = random.Random(seed)
            for case_index in range(100):
                points = [
                    (
                        rng.uniform(-1000.0, 1000.0),
                        rng.uniform(-1000.0, 1000.0),
                    )
                    for _ in range(rng.randint(3, 15))
                ]
                rng.shuffle(points)
                with self.subTest(seed=seed, case_index=case_index):
                    self.assert_matches_oracle(points)

    def test_large_coordinate_translations_match_oracle(self) -> None:
        point_sets = (
            [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)],
            [(-8.0, 1.0), (-1.5, 7.25), (9.0, -4.0), (0.5, -6.5)],
            [(0.0, 0.0), (5.0, 1e-10), (10.0, 0.0)],
        )
        shifts = ((1_000_000.0, -1_000_000.0), (1_000_000_000.0, 1_000_000_000.0))
        for points in point_sets:
            base = minimum_enclosing_circle(points)
            for shift in shifts:
                shifted = [(x + shift[0], y + shift[1]) for x, y in points]
                with self.subTest(points=points, shift=shift):
                    production, _ = self.assert_matches_oracle(shifted)
                    tolerance = self._coordinate_tolerance(shifted, base.radius)
                    self.assertTrue(
                        math.isclose(
                            production.radius,
                            base.radius,
                            rel_tol=1e-10,
                            abs_tol=tolerance,
                        )
                    )

    def test_clear_threshold_radii_match_oracle(self) -> None:
        for expected_radius in (19.99, 20.00, 20.01):
            points = [
                (-expected_radius, 0.0),
                (expected_radius, 0.0),
                (0.0, expected_radius / 3.0),
                (0.0, -expected_radius / 4.0),
            ]
            with self.subTest(expected_radius=expected_radius):
                production, oracle = self.assert_matches_oracle(points)
                self.assertAlmostEqual(production.radius, expected_radius, delta=1e-12)
                self.assertAlmostEqual(oracle.radius, expected_radius, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
