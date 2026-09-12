import math
import unittest

from src.common.geometry import (
    Circle,
    candidate_second_points,
    clip_polygon_to_bearing_wedge,
    clip_polygon_to_circle_outer,
    is_clear_point_certified,
    max_distance_to_region,
    minimum_enclosing_circle,
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


class CircleOuterClippingTests(unittest.TestCase):
    def test_true_circle_cardinal_points_are_not_excluded(self) -> None:
        for point in ((10.0, 0.0), (0.0, 10.0), (-10.0, 0.0), (0.0, -10.0)):
            with self.subTest(point=point):
                self.assertEqual(
                    clip_polygon_to_circle_outer([point], (0.0, 0.0), 10.0, 8),
                    [point],
                )

    def test_outer_circle_is_translated_to_requested_center(self) -> None:
        center = (100.0, -50.0)
        polygon = [(99.0, -51.0), (101.0, -51.0), (101.0, -49.0), (99.0, -49.0)]

        self.assertEqual(
            clip_polygon_to_circle_outer(polygon, center, 10.0, 8),
            polygon,
        )
        self.assertEqual(
            clip_polygon_to_circle_outer(polygon, (0.0, 0.0), 10.0, 8),
            [],
        )

    def test_clipped_vertices_stay_within_outer_polygon_vertex_radius(self) -> None:
        center = (3.0, -4.0)
        radius = 10.0
        vertex_count = 12
        polygon = [(-20.0, -20.0), (30.0, -20.0), (30.0, 20.0), (-20.0, 20.0)]

        clipped = clip_polygon_to_circle_outer(
            polygon,
            center,
            radius,
            vertex_count,
        )
        maximum_radius = radius / math.cos(math.pi / vertex_count)

        self.assertTrue(clipped)
        self.assertTrue(
            all(math.dist(point, center) <= maximum_radius + 1e-9 for point in clipped)
        )

    def test_polygon_completely_outside_outer_circle_becomes_empty(self) -> None:
        polygon = [(20.0, -1.0), (22.0, -1.0), (22.0, 1.0), (20.0, 1.0)]

        self.assertEqual(
            clip_polygon_to_circle_outer(polygon, (0.0, 0.0), 10.0, 16),
            [],
        )

    def test_polygon_inside_true_circle_is_unchanged(self) -> None:
        polygon = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

        self.assertEqual(
            clip_polygon_to_circle_outer(polygon, (0.0, 0.0), 10.0, 16),
            polygon,
        )


class ClearPointCertificationTests(unittest.TestCase):
    def test_square_with_36_m_diagonal_is_certified_at_its_center(self) -> None:
        half_side = 9.0 * math.sqrt(2.0)
        square = [
            (-half_side, -half_side),
            (half_side, -half_side),
            (half_side, half_side),
            (-half_side, half_side),
        ]

        self.assertAlmostEqual(max_distance_to_region(square, (0.0, 0.0)), 18.0)
        self.assertTrue(is_clear_point_certified(square, (0.0, 0.0)))

    def test_equilateral_triangle_blocks_diameter_only_certification(self) -> None:
        triangle = [(0.0, 0.0), (36.0, 0.0), (18.0, 18.0 * math.sqrt(3.0))]
        optimal_center = (18.0, 6.0 * math.sqrt(3.0))

        self.assertAlmostEqual(polygon_diameter(triangle), 36.0)
        self.assertAlmostEqual(
            max_distance_to_region(triangle, optimal_center),
            12.0 * math.sqrt(3.0),
        )
        self.assertFalse(is_clear_point_certified(triangle, optimal_center))

    def test_single_point_region_uses_its_distance_to_center(self) -> None:
        region = [(3.0, 4.0)]

        self.assertEqual(max_distance_to_region(region, (0.0, 0.0)), 5.0)
        self.assertTrue(is_clear_point_certified(region, (0.0, 0.0)))

    def test_line_segment_region_uses_farther_endpoint(self) -> None:
        region = [(-7.0, 0.0), (13.0, 0.0)]

        self.assertEqual(max_distance_to_region(region, (1.0, 0.0)), 12.0)
        self.assertTrue(is_clear_point_certified(region, (1.0, 0.0)))

    def test_empty_region_cannot_be_certified(self) -> None:
        self.assertEqual(max_distance_to_region([], (0.0, 0.0)), math.inf)
        self.assertFalse(is_clear_point_certified([], (0.0, 0.0)))

    def test_exactly_20_m_boundary_is_certified(self) -> None:
        region = [(-20.0, 0.0), (20.0, 0.0)]

        self.assertEqual(max_distance_to_region(region, (0.0, 0.0)), 20.0)
        self.assertTrue(is_clear_point_certified(region, (0.0, 0.0)))


class MinimumEnclosingCircleTests(unittest.TestCase):
    def assert_circle(
        self,
        points: list[tuple[float, float]],
        expected_center: tuple[float, float],
        expected_radius: float,
    ) -> Circle:
        circle = minimum_enclosing_circle(points)
        self.assertAlmostEqual(circle.center[0], expected_center[0])
        self.assertAlmostEqual(circle.center[1], expected_center[1])
        self.assertAlmostEqual(circle.radius, expected_radius)
        for point in points:
            self.assertLessEqual(
                math.dist(point, circle.center),
                circle.radius + 1e-10,
            )
        return circle

    def test_empty_set_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            minimum_enclosing_circle([])

    def test_single_point(self) -> None:
        self.assert_circle([(3.0, 4.0)], (3.0, 4.0), 0.0)

    def test_two_points(self) -> None:
        self.assert_circle([(0.0, 0.0), (10.0, 0.0)], (5.0, 0.0), 5.0)

    def test_duplicate_points(self) -> None:
        self.assert_circle([(1.0, 2.0)] * 3, (1.0, 2.0), 0.0)

    def test_three_collinear_points_use_farthest_pair(self) -> None:
        self.assert_circle(
            [(0.0, 0.0), (4.0, 0.0), (10.0, 0.0)],
            (5.0, 0.0),
            5.0,
        )

    def test_nearly_collinear_points_do_not_create_a_huge_circle(self) -> None:
        self.assert_circle(
            [(0.0, 0.0), (5.0, 1e-10), (10.0, 0.0)],
            (5.0, 0.0),
            5.0,
        )

    def test_three_four_five_right_triangle_uses_hypotenuse(self) -> None:
        self.assert_circle(
            [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)],
            (2.0, 1.5),
            2.5,
        )

    def test_obtuse_triangle_uses_longest_side(self) -> None:
        self.assert_circle(
            [(0.0, 0.0), (6.0, 0.0), (1.0, 1.0)],
            (3.0, 0.0),
            3.0,
        )

    def test_acute_triangle_uses_circumcircle(self) -> None:
        self.assert_circle(
            [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)],
            (2.0, 5.0 / 6.0),
            13.0 / 6.0,
        )

    def test_square_with_36_m_diagonal_has_18_m_radius(self) -> None:
        half_side = 9.0 * math.sqrt(2.0)
        points = [
            (-half_side, -half_side),
            (half_side, -half_side),
            (half_side, half_side),
            (-half_side, half_side),
        ]
        self.assert_circle(points, (0.0, 0.0), 18.0)

    def test_equilateral_triangle_with_36_m_sides(self) -> None:
        points = [(0.0, 0.0), (36.0, 0.0), (18.0, 18.0 * math.sqrt(3.0))]
        self.assert_circle(
            points,
            (18.0, 6.0 * math.sqrt(3.0)),
            12.0 * math.sqrt(3.0),
        )

    def test_general_multi_point_set(self) -> None:
        points = [
            (7.0, -3.0),
            (-3.0, -3.0),
            (2.0, 2.0),
            (2.0, -8.0),
            (2.0, -3.0),
            (3.0, -2.0),
        ]
        self.assert_circle(points, (2.0, -3.0), 5.0)

    def test_input_order_does_not_change_result(self) -> None:
        points = [(-2.0, 1.0), (5.0, 1.0), (1.0, 6.0), (0.0, 2.0)]
        forward = minimum_enclosing_circle(points)
        reverse = minimum_enclosing_circle(list(reversed(points)))

        self.assertEqual(forward, reverse)
        for point in points:
            self.assertLessEqual(
                math.dist(point, forward.center),
                forward.radius + 1e-10,
            )

    def assert_translation_invariant(
        self,
        points: list[tuple[float, float]],
        shift: tuple[float, float],
        *,
        abs_tol: float,
    ) -> None:
        base = minimum_enclosing_circle(points)
        shifted_points = [
            (x + shift[0], y + shift[1])
            for x, y in points
        ]

        shifted = minimum_enclosing_circle(shifted_points)

        self.assertTrue(
            math.isclose(shifted.radius, base.radius, rel_tol=1e-12, abs_tol=abs_tol)
        )
        self.assertTrue(
            math.isclose(
                shifted.center[0] - base.center[0],
                shift[0],
                rel_tol=1e-12,
                abs_tol=abs_tol,
            )
        )
        self.assertTrue(
            math.isclose(
                shifted.center[1] - base.center[1],
                shift[1],
                rel_tol=1e-12,
                abs_tol=abs_tol,
            )
        )

    def test_translation_invariance_at_one_million_scale(self) -> None:
        points = [(-2.5, 1.25), (4.75, -0.5), (1.5, 6.0), (0.25, 2.0)]

        self.assert_translation_invariant(
            points,
            (1_000_000.0, -1_000_000.0),
            abs_tol=1e-9,
        )

    def test_translation_invariance_at_one_billion_scale(self) -> None:
        points = [(-2.5, 1.25), (4.75, -0.5), (1.5, 6.0), (0.25, 2.0)]

        self.assert_translation_invariant(
            points,
            (1_000_000_000.0, -1_000_000_000.0),
            abs_tol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
