import math
import random
import unittest

from src.common.geometry import (
    polygon_diameter,
    polygon_diameter_calipers,
    polygon_diameter_exhaustive,
)


class PolygonDiameterCalipersTests(unittest.TestCase):
    def assert_matches_oracle(
        self,
        points: list[tuple[float, float]],
        *,
        rel_tol: float = 1e-12,
        abs_tol: float = 1e-12,
    ) -> float:
        expected = polygon_diameter_exhaustive(points)
        actual = polygon_diameter_calipers(points)
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol),
            f"calipers={actual}, exhaustive={expected}, points={points}",
        )
        return actual

    def test_degenerate_inputs(self) -> None:
        cases = (
            ([], 0.0),
            ([(3.0, -4.0)], 0.0),
            ([(0.0, 0.0), (3.0, 4.0)], 5.0),
        )
        for points, expected in cases:
            with self.subTest(points=points):
                self.assertEqual(self.assert_matches_oracle(points), expected)

    def test_acute_right_and_obtuse_triangles(self) -> None:
        cases = {
            "acute": [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)],
            "right": [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)],
            "obtuse": [(0.0, 0.0), (6.0, 0.0), (1.0, 1.0)],
        }
        for name, points in cases.items():
            with self.subTest(name=name):
                self.assert_matches_oracle(points)

    def test_parallel_edges_and_multiple_farthest_pairs(self) -> None:
        cases = {
            "square": [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)],
            "rectangle": [(0.0, 0.0), (8.0, 0.0), (8.0, 3.0), (0.0, 3.0)],
            "regular_hexagon": [
                (math.cos(k * math.pi / 3.0), math.sin(k * math.pi / 3.0))
                for k in range(6)
            ],
            "thin_parallelogram": [
                (0.0, 0.0),
                (1000.0, 0.001),
                (1003.0, 1.001),
                (3.0, 1.0),
            ],
        }
        for name, points in cases.items():
            with self.subTest(name=name):
                self.assert_matches_oracle(points)

    def test_near_tie_perturbed_rectangle_matches_exhaustive_oracle(self) -> None:
        points = [
            (0.0, 0.0),
            (1_000_000.0, 1e-6),
            (1_000_000.000001, 1.000001000002),
            (1e-6, 1.0),
        ]

        self.assert_matches_oracle(points)

    def test_production_api_delegates_to_calipers(self) -> None:
        cases = {
            "empty": [],
            "two_points": [(0.0, 0.0), (3.0, 4.0)],
            "triangle": [(0.0, 0.0), (4.0, 0.0), (1.0, 3.0)],
            "rectangle": [(0.0, 0.0), (8.0, 0.0), (8.0, 3.0), (0.0, 3.0)],
            "interior_and_duplicates": [
                (0.0, 0.0),
                (4.0, 0.0),
                (4.0, 2.0),
                (0.0, 2.0),
                (2.0, 1.0),
                (4.0, 2.0),
                (2.0, 1.0),
            ],
        }
        for name, points in cases.items():
            with self.subTest(name=name):
                self.assertEqual(
                    polygon_diameter(points),
                    polygon_diameter_calipers(points),
                )

    def test_unordered_interior_duplicate_and_collinear_points(self) -> None:
        points = [
            (4.0, 0.0),
            (0.0, 2.0),
            (2.0, 1.0),
            (0.0, 0.0),
            (4.0, 2.0),
            (2.0, 0.0),
            (4.0, 1.0),
            (0.0, 1.0),
            (2.0, 2.0),
            (4.0, 2.0),
            (2.0, 1.0),
        ]

        self.assert_matches_oracle(points)

    def test_fixed_seed_random_sets_match_exhaustive_oracle(self) -> None:
        for seed in (20260910, 20260911, 20260912):
            rng = random.Random(seed)
            for case_index in range(100):
                points = [
                    (float(rng.randrange(-10_000, 10_001)),
                     float(rng.randrange(-10_000, 10_001)))
                    for _ in range(rng.randrange(3, 81))
                ]
                rng.shuffle(points)
                with self.subTest(seed=seed, case_index=case_index):
                    self.assert_matches_oracle(points, abs_tol=1e-9)

    def test_translation_invariance_and_oracle_agreement(self) -> None:
        point_sets = (
            [(0.0, 0.0), (7.5, 0.25), (5.0, 4.0), (-2.0, 3.0)],
            [(0.0, 0.0), (1000.0, 0.001), (1003.0, 1.001), (3.0, 1.0)],
        )
        shifts_and_tolerances = (
            ((1_000_000.0, -1_000_000.0), 1e-9),
            ((1_000_000_000.0, 1_000_000_000.0), 1e-6),
        )
        for points in point_sets:
            base = self.assert_matches_oracle(points)
            for shift, abs_tol in shifts_and_tolerances:
                shifted = [(x + shift[0], y + shift[1]) for x, y in points]
                with self.subTest(points=points, shift=shift):
                    shifted_diameter = self.assert_matches_oracle(
                        shifted,
                        abs_tol=abs_tol,
                    )
                    self.assertTrue(
                        math.isclose(
                            shifted_diameter,
                            base,
                            rel_tol=1e-12,
                            abs_tol=abs_tol,
                        ),
                        (
                            f"translated={shifted_diameter}, base={base}, "
                            f"shift={shift}"
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
