import math
import unittest
from unittest.mock import patch

from src.q2 import model
from src.q2.model import (
    CandidateDomainFlags,
    FirstDirectionObservation,
    FirstState,
    JointSample,
    Q2Config,
    SecondResponse,
    SecondResponseKind,
    build_first_state,
    evaluate_candidate_domains,
)


class Q2ConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        config = Q2Config()

        self.assertEqual(config.arena_radius_m, 1800.0)
        self.assertEqual(config.reception_radius_min_m, 1000.0)
        self.assertEqual(config.reception_radius_max_m, 1500.0)
        self.assertEqual(config.near_radius_m, 5.0)
        self.assertEqual(config.bearing_error_deg, 1.0)
        self.assertEqual(config.circle_vertices, 720)
        self.assertEqual(config.direction_angle_bins, 720)

    def test_invalid_radius_relationships_raise_value_error(self) -> None:
        cases = (
            {"near_radius_m": 0.0},
            {"near_radius_m": 1000.0},
            {"reception_radius_min_m": 1501.0},
            {"reception_radius_max_m": 4.0},
        )

        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    Q2Config(**kwargs)

    def test_count_parameters_must_be_real_integers(self) -> None:
        cases = (
            {"circle_vertices": 720.5},
            {"direction_angle_bins": 360.5},
            {"circle_vertices": True},
            {"direction_angle_bins": False},
        )

        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    Q2Config(**kwargs)

    def test_non_finite_parameters_raise_value_error(self) -> None:
        cases = (
            {"arena_radius_m": math.inf},
            {"reception_radius_min_m": math.nan},
            {"reception_radius_max_m": math.inf},
            {"near_radius_m": math.nan},
            {"bearing_error_deg": math.inf},
            {"circle_vertices": math.inf},
            {"direction_angle_bins": math.nan},
        )

        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    Q2Config(**kwargs)


class SecondResponseTests(unittest.TestCase):
    def test_direction_requires_bearing(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires bearing_deg"):
            SecondResponse(SecondResponseKind.DIRECTION)

    def test_near_and_no_signal_reject_bearing(self) -> None:
        for kind in (SecondResponseKind.NEAR, SecondResponseKind.NO_SIGNAL):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "must not include"):
                    SecondResponse(kind, bearing_deg=12.0)

    def test_direction_bearing_is_normalized(self) -> None:
        self.assertEqual(
            SecondResponse(SecondResponseKind.DIRECTION, bearing_deg=361.25).bearing_deg,
            1.25,
        )
        self.assertEqual(
            SecondResponse(SecondResponseKind.DIRECTION, bearing_deg=-1.0).bearing_deg,
            359.0,
        )


class FirstStateTests(unittest.TestCase):
    def test_build_first_state_calls_q1_localization(self) -> None:
        observation = FirstDirectionObservation((0.0, 0.0), 0.0)
        samples = [JointSample((100.0, 0.0), 1200.0, 1.0)]

        with patch(
            "src.q2.model.localization_region",
            return_value=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
        ) as localization:
            state = build_first_state(observation, samples=samples)

        localization.assert_called_once()
        self.assertEqual(state.outer_region, ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)))

    def test_build_first_state_filters_incompatible_samples_and_keeps_r(self) -> None:
        observation = FirstDirectionObservation((0.0, 0.0), 0.0)
        valid = JointSample((100.0, 0.0), 1234.0, 2.0)
        near = JointSample((5.0, 0.0), 1200.0, 1.0)
        out_of_reception_radius = JointSample((1200.0, 0.0), 1000.0, 1.0)
        wrong_bearing = JointSample((100.0, 10.0), 1200.0, 1.0)
        outside_arena = JointSample((1801.0, 0.0), 1900.0, 1.0)

        state = build_first_state(
            observation,
            samples=[
                valid,
                near,
                out_of_reception_radius,
                wrong_bearing,
                outside_arena,
            ],
            config=Q2Config(circle_vertices=36),
        )

        self.assertEqual(state.joint_samples, (valid,))
        self.assertEqual(state.joint_samples[0].reception_radius_m, 1234.0)
        self.assertIn("closure(F1)", state.exact_support_label)
        self.assertTrue(state.outer_region)

    def test_build_first_state_drops_zero_weight_compatible_samples(self) -> None:
        observation = FirstDirectionObservation((0.0, 0.0), 0.0)
        positive = JointSample((100.0, 0.0), 1200.0, 2.0)
        zero_weight = JointSample((200.0, 0.0), 1200.0, 0.0)

        state = build_first_state(
            observation,
            samples=[positive, zero_weight],
            config=Q2Config(circle_vertices=36),
        )

        self.assertEqual(state.joint_samples, (positive,))

    def test_build_first_state_rejects_empty_filtered_samples(self) -> None:
        observation = FirstDirectionObservation((0.0, 0.0), 0.0)

        with self.assertRaisesRegex(ValueError, "no samples remain"):
            build_first_state(
                observation,
                samples=[JointSample((100.0, 10.0), 1200.0, 1.0)],
                config=Q2Config(circle_vertices=36),
            )

    def test_build_first_state_rejects_empty_outer_region(self) -> None:
        observation = FirstDirectionObservation((0.0, 0.0), 0.0)
        samples = [JointSample((100.0, 0.0), 1200.0, 1.0)]

        with patch("src.q2.model.localization_region", return_value=[]):
            with self.assertRaisesRegex(ValueError, "outer region is empty"):
                build_first_state(observation, samples=samples)

    def test_build_first_state_rejects_samples_without_positive_weight(self) -> None:
        observation = FirstDirectionObservation((0.0, 0.0), 0.0)

        with self.assertRaisesRegex(ValueError, "positive weight"):
            build_first_state(
                observation,
                samples=[JointSample((100.0, 0.0), 1200.0, 0.0)],
                config=Q2Config(circle_vertices=36),
            )

    def test_non_finite_first_observation_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            FirstDirectionObservation((math.inf, 0.0), 0.0)
        with self.assertRaises(ValueError):
            FirstDirectionObservation((0.0, 0.0), math.nan)


class CandidateDomainTests(unittest.TestCase):
    def make_state(
        self,
        outer_region: tuple[tuple[float, float], ...],
    ) -> FirstState:
        return FirstState(
            observation=FirstDirectionObservation((0.0, 0.0), 0.0),
            exact_support_label="test outer, not exact F1",
            outer_region=outer_region,
            joint_samples=(JointSample((1.0, 0.0), 1000.0, 1.0),),
        )

    def test_q_may_lie_outside_1800m_target_circle(self) -> None:
        state = self.make_state(
            ((1900.0, 0.0), (1910.0, 0.0), (1910.0, 10.0), (1900.0, 10.0))
        )

        flags = evaluate_candidate_domains((1905.0, 5.0), state)

        self.assertTrue(flags.in_c_poss_proxy)
        self.assertEqual(flags.min_distance_to_outer_m, 0.0)

    def test_polygon_inside_and_boundary_distance_are_zero(self) -> None:
        state = self.make_state(
            ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
        )

        inside = evaluate_candidate_domains((5.0, 5.0), state)
        boundary = evaluate_candidate_domains((10.0, 5.0), state)

        self.assertEqual(inside.min_distance_to_outer_m, 0.0)
        self.assertEqual(boundary.min_distance_to_outer_m, 0.0)

    def test_polygon_outside_distance_is_to_nearest_segment(self) -> None:
        state = self.make_state(
            ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
        )

        flags = evaluate_candidate_domains((13.0, 4.0), state)

        self.assertIsInstance(flags, CandidateDomainFlags)
        self.assertAlmostEqual(flags.min_distance_to_outer_m, 3.0)

    def test_c_rec_certifies_boundary_and_rejects_above_1000m(self) -> None:
        state = self.make_state(((0.0, 0.0), (10.0, 0.0)))

        certified = evaluate_candidate_domains((1000.0, 0.0), state)
        rejected = evaluate_candidate_domains((-1000.1, 0.0), state)

        self.assertTrue(certified.in_c_rec_certified)
        self.assertAlmostEqual(certified.max_distance_to_outer_m, 1000.0)
        self.assertFalse(rejected.in_c_rec_certified)
        self.assertGreater(rejected.max_distance_to_outer_m, 1000.0)

    def test_point_to_polygon_distance_handles_single_point_and_segment(self) -> None:
        self.assertEqual(
            model._point_to_polygon_distance((3.0, 4.0), [(0.0, 0.0)]),
            5.0,
        )
        self.assertEqual(
            model._point_to_polygon_distance((5.0, 3.0), [(0.0, 0.0), (10.0, 0.0)]),
            3.0,
        )
        self.assertEqual(
            model._point_to_polygon_distance((5.0, 0.0), [(0.0, 0.0), (10.0, 0.0)]),
            0.0,
        )

    def test_point_to_polygon_distance_handles_convex_polygon_outside_corner(self) -> None:
        polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

        self.assertAlmostEqual(
            model._point_to_polygon_distance((13.0, 14.0), polygon),
            5.0,
        )


if __name__ == "__main__":
    unittest.main()
