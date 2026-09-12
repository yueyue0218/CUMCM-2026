import math
import unittest

from src.common.geometry import (
    circle_polygon,
    clip_polygon_to_bearing_wedge,
    max_distance_to_region,
)
from src.common.localization import (
    BearingObservation,
    LocalizationAssessment,
    LocalizationStatus,
    assess_observations_for_clear,
    assess_region_for_clear,
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

    def test_single_direction_is_limited_by_reception_upper_bound(self) -> None:
        vertex_count = 72
        region = localization_region(
            [BearingObservation((0.0, 0.0), 0.0)],
            arena_radius_m=1800.0,
            circle_vertices=vertex_count,
        )
        maximum_outer_radius = 1500.0 / math.cos(math.pi / vertex_count)

        self.assertTrue(region)
        self.assertTrue(
            all(
                math.dist(point, (0.0, 0.0)) <= maximum_outer_radius + 1e-8
                for point in region
            )
        )

    def test_reception_bound_strictly_reduces_wedge_only_region(self) -> None:
        vertex_count = 72
        observation = BearingObservation((0.0, 0.0), 0.0)
        arena = circle_polygon(1800.0, vertex_count)
        wedge_only_region = clip_polygon_to_bearing_wedge(
            arena,
            observation.station,
            observation.bearing_deg,
            observation.error_deg,
        )
        region_with_range = localization_region(
            [observation],
            arena_radius_m=1800.0,
            circle_vertices=vertex_count,
        )

        wedge_only_area = localization_quality(wedge_only_region)["area_m2"]
        range_limited_area = localization_quality(region_with_range)["area_m2"]
        self.assertLess(range_limited_area, wedge_only_area)

    def test_multiple_directions_satisfy_every_reception_upper_bound(self) -> None:
        vertex_count = 72
        observations = [
            BearingObservation((0.0, 0.0), 45.0),
            BearingObservation((100.0, 0.0), 135.0),
        ]
        region = localization_region(
            observations,
            arena_radius_m=1800.0,
            circle_vertices=vertex_count,
        )
        maximum_outer_radius = 1500.0 / math.cos(math.pi / vertex_count)

        self.assertTrue(region)
        for observation in observations:
            self.assertTrue(
                all(
                    math.dist(point, observation.station)
                    <= maximum_outer_radius + 1e-8
                    for point in region
                )
            )

    def test_empty_observations_return_only_full_arena_outer_polygon(self) -> None:
        vertex_count = 24

        self.assertEqual(
            localization_region([], circle_vertices=vertex_count),
            circle_polygon(1800.0, vertex_count),
        )


class ClearAssessmentTests(unittest.TestCase):
    def assert_status_consistent(self, assessment: LocalizationAssessment) -> None:
        self.assertEqual(
            assessment.clear_ready,
            assessment.status is LocalizationStatus.CLEAR_READY,
        )
        if assessment.clear_position is not None:
            self.assertAlmostEqual(
                assessment.r_max_m,
                max_distance_to_region(
                    assessment.outer_region,
                    assessment.clear_position,
                ),
            )

    def test_empty_region_is_model_conflict(self) -> None:
        assessment = assess_region_for_clear([])

        self.assertIs(assessment.status, LocalizationStatus.MODEL_CONFLICT)
        self.assertFalse(assessment.clear_ready)
        self.assertIsNone(assessment.clear_position)
        self.assertIsNone(assessment.r_max_m)
        self.assertEqual(assessment.diameter_m, math.inf)
        self.assertIn("empty", assessment.reason)
        self.assert_status_consistent(assessment)

    def test_single_point_is_clear_ready_at_that_point(self) -> None:
        assessment = assess_region_for_clear([(3.0, 4.0)])

        self.assertIs(assessment.status, LocalizationStatus.CLEAR_READY)
        self.assertEqual(assessment.clear_position, (3.0, 4.0))
        self.assertEqual(assessment.r_max_m, 0.0)
        self.assertEqual(assessment.diameter_m, 0.0)
        self.assert_status_consistent(assessment)

    def test_square_with_36_m_diagonal_is_clear_ready(self) -> None:
        half_side = 9.0 * math.sqrt(2.0)
        region = [
            (-half_side, -half_side),
            (half_side, -half_side),
            (half_side, half_side),
            (-half_side, half_side),
        ]

        assessment = assess_region_for_clear(region)

        self.assertAlmostEqual(assessment.diameter_m, 36.0)
        self.assertAlmostEqual(assessment.clear_position[0], 0.0)  # type: ignore[index]
        self.assertAlmostEqual(assessment.clear_position[1], 0.0)  # type: ignore[index]
        self.assertAlmostEqual(assessment.r_max_m, 18.0)
        self.assertIs(assessment.status, LocalizationStatus.CLEAR_READY)
        self.assert_status_consistent(assessment)

    def test_equilateral_triangle_with_same_diameter_is_uncertain(self) -> None:
        region = [(0.0, 0.0), (36.0, 0.0), (18.0, 18.0 * math.sqrt(3.0))]

        assessment = assess_region_for_clear(region)

        self.assertAlmostEqual(assessment.diameter_m, 36.0)
        self.assertLess(assessment.diameter_m, 40.0)
        self.assertAlmostEqual(assessment.r_max_m, 12.0 * math.sqrt(3.0))
        self.assertGreater(assessment.r_max_m, 20.0)  # type: ignore[operator]
        self.assertIs(assessment.status, LocalizationStatus.COVERAGE_UNCERTAIN)
        self.assertFalse(assessment.clear_ready)
        self.assert_status_consistent(assessment)

    def test_40_m_segment_is_uncertain_with_default_numerical_margin(self) -> None:
        assessment = assess_region_for_clear([(-20.0, 0.0), (20.0, 0.0)])

        self.assertEqual(assessment.clear_position, (0.0, 0.0))
        self.assertEqual(assessment.diameter_m, 40.0)
        self.assertEqual(assessment.r_max_m, 20.0)
        self.assertIs(assessment.status, LocalizationStatus.COVERAGE_UNCERTAIN)
        self.assertFalse(assessment.clear_ready)
        self.assertIn("numerical guard band", assessment.reason)
        self.assert_status_consistent(assessment)

    def test_default_margin_boundary_cases(self) -> None:
        cases = (
            (19.998, LocalizationStatus.CLEAR_READY),
            (19.999, LocalizationStatus.CLEAR_READY),
            (19.9995, LocalizationStatus.COVERAGE_UNCERTAIN),
            (20.0, LocalizationStatus.COVERAGE_UNCERTAIN),
            (20.0001, LocalizationStatus.COVERAGE_UNCERTAIN),
        )

        for r_max_m, expected_status in cases:
            with self.subTest(r_max_m=r_max_m):
                assessment = assess_region_for_clear(
                    [(-r_max_m, 0.0), (r_max_m, 0.0)]
                )

                self.assertAlmostEqual(assessment.r_max_m, r_max_m)
                self.assertIs(assessment.status, expected_status)
                self.assertEqual(
                    assessment.clear_ready,
                    expected_status is LocalizationStatus.CLEAR_READY,
                )
                self.assert_status_consistent(assessment)

    def test_numerical_guard_reason_is_distinct_from_radius_failure(self) -> None:
        guard_band = assess_region_for_clear([(-19.9995, 0.0), (19.9995, 0.0)])
        radius_failure = assess_region_for_clear([(-20.0001, 0.0), (20.0001, 0.0)])

        self.assertIn("numerical guard band", guard_band.reason)
        self.assertIn("exceeds the physical clear radius", radius_failure.reason)

    def test_zero_margin_restores_closed_radius_boundary(self) -> None:
        assessment = assess_region_for_clear(
            [(-20.0, 0.0), (20.0, 0.0)],
            clear_certification_margin_m=0.0,
        )

        self.assertEqual(assessment.r_max_m, 20.0)
        self.assertIs(assessment.status, LocalizationStatus.CLEAR_READY)
        self.assertTrue(assessment.clear_ready)
        self.assert_status_consistent(assessment)

    def test_invalid_clear_certification_margins_raise_value_error(self) -> None:
        for margin in (-0.001, math.nan, math.inf, 20.0, 20.001):
            with self.subTest(margin=margin):
                with self.assertRaises(ValueError):
                    assess_region_for_clear(
                        [(0.0, 0.0)],
                        clear_certification_margin_m=margin,
                    )

    def test_invalid_clear_radii_raise_value_error(self) -> None:
        for clear_radius_m in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(clear_radius_m=clear_radius_m):
                with self.assertRaises(ValueError):
                    assess_region_for_clear(
                        [(0.0, 0.0)],
                        clear_radius_m=clear_radius_m,
                    )

    def test_segment_slightly_longer_than_40_m_is_uncertain(self) -> None:
        assessment = assess_region_for_clear([(-20.001, 0.0), (20.001, 0.0)])

        self.assertGreater(assessment.diameter_m, 40.0)
        self.assertGreater(assessment.r_max_m, 20.0)  # type: ignore[operator]
        self.assertIs(assessment.status, LocalizationStatus.COVERAGE_UNCERTAIN)
        self.assertFalse(assessment.clear_ready)
        self.assert_status_consistent(assessment)


class ObservationClearAssessmentTests(unittest.TestCase):
    def test_empty_observations_keep_full_arena_as_uncertain(self) -> None:
        assessment = assess_observations_for_clear([])

        self.assertTrue(assessment.outer_region)
        self.assertIs(assessment.status, LocalizationStatus.COVERAGE_UNCERTAIN)
        self.assertIsNot(assessment.status, LocalizationStatus.MODEL_CONFLICT)
        self.assertFalse(assessment.clear_ready)

    def test_crossing_directions_produce_a_finite_assessment(self) -> None:
        observations = [
            BearingObservation((0.0, 0.0), 45.0),
            BearingObservation((100.0, 0.0), 135.0),
        ]

        assessment = assess_observations_for_clear(
            observations,
            arena_radius_m=200.0,
            circle_vertices=72,
        )

        self.assertIsInstance(assessment, LocalizationAssessment)
        self.assertTrue(assessment.outer_region)
        self.assertTrue(math.isfinite(assessment.diameter_m))
        self.assertIsNotNone(assessment.clear_position)
        self.assertIsNotNone(assessment.r_max_m)
        self.assertGreaterEqual(assessment.r_max_m, 0.0)  # type: ignore[operator]

    def test_conflicting_directions_return_model_conflict(self) -> None:
        observations = [
            BearingObservation((0.0, 0.0), 0.0, 0.1),
            BearingObservation((100.0, 0.0), 0.0, 0.1),
        ]

        assessment = assess_observations_for_clear(
            observations,
            arena_radius_m=50.0,
        )

        self.assertIs(assessment.status, LocalizationStatus.MODEL_CONFLICT)
        self.assertFalse(assessment.clear_ready)
        self.assertIsNone(assessment.clear_position)
        self.assertFalse(assessment.outer_region)

    def test_high_level_api_equals_manual_composition(self) -> None:
        observations = [
            BearingObservation((0.0, 0.0), 45.0),
            BearingObservation((100.0, 0.0), 135.0),
        ]
        parameters = {
            "arena_radius_m": 200.0,
            "circle_vertices": 72,
            "reception_radius_upper_m": 120.0,
        }
        region = localization_region(observations, **parameters)
        direct = assess_region_for_clear(region, clear_radius_m=25.0)

        combined = assess_observations_for_clear(
            observations,
            clear_radius_m=25.0,
            **parameters,
        )

        self.assertEqual(combined, direct)

    def test_high_level_api_passes_through_certification_margin(self) -> None:
        observations = [BearingObservation((0.0, 0.0), 0.0)]
        parameters = {
            "arena_radius_m": 30.0,
            "circle_vertices": 24,
            "reception_radius_upper_m": 25.0,
        }
        region = localization_region(observations, **parameters)
        direct = assess_region_for_clear(
            region,
            clear_radius_m=20.0,
            clear_certification_margin_m=0.25,
        )

        combined = assess_observations_for_clear(
            observations,
            clear_radius_m=20.0,
            clear_certification_margin_m=0.25,
            **parameters,
        )

        self.assertEqual(combined, direct)


if __name__ == "__main__":
    unittest.main()
