import math
import unittest
from unittest.mock import patch

from src.q2 import model
from src.q2.model import (
    BayesianEvaluation,
    BearingErrorAtom,
    BearingErrorBin,
    CandidateDomainFlags,
    FirstDirectionObservation,
    FirstPosteriorSamples,
    FirstState,
    JointSample,
    Q2Config,
    ResponseMetric,
    SecondResponse,
    SecondResponseKind,
    SecondSupport,
    build_first_state,
    evaluate_bayesian,
    evaluate_candidate_domains,
    sample_first_direction_posterior,
    second_support,
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



class FirstPosteriorSamplingTests(unittest.TestCase):
    def uniform_error_bins(self) -> tuple[BearingErrorBin, ...]:
        return (BearingErrorBin(-1.0, 1.0, 1.0),)

    def test_sampler_is_reproducible_and_returns_normalized_joint_samples(self) -> None:
        observation = FirstDirectionObservation((0.0, 0.0), 0.0)
        kwargs = dict(
            prior_draws=20000,
            bearing_error_bins=self.uniform_error_bins(),
            seed=20260913,
            config=Q2Config(circle_vertices=36),
        )

        first = sample_first_direction_posterior(observation, **kwargs)
        second = sample_first_direction_posterior(observation, **kwargs)

        self.assertIsInstance(first, FirstPosteriorSamples)
        self.assertEqual(first, second)
        self.assertGreater(first.retained_draws, 0)
        self.assertAlmostEqual(sum(sample.weight for sample in first.samples), 1.0)
        self.assertGreater(first.effective_sample_size, 0.0)
        self.assertAlmostEqual(
            first.acceptance_rate,
            first.retained_draws / first.prior_draws,
        )
        for sample in first.samples:
            self.assertGreater(sample.weight, 0.0)
            self.assertGreaterEqual(sample.reception_radius_m, 1000.0)
            self.assertLessEqual(sample.reception_radius_m, 1500.0)
            self.assertLessEqual(math.dist(sample.position, (0.0, 0.0)), 1800.0 + 1e-9)

    def test_samples_satisfy_first_direction_physics_and_feed_build_first_state(self) -> None:
        observation = FirstDirectionObservation((0.0, 0.0), 0.0)
        result = sample_first_direction_posterior(
            observation,
            prior_draws=20000,
            bearing_error_bins=self.uniform_error_bins(),
            seed=7,
            config=Q2Config(circle_vertices=36),
        )

        for sample in result.samples:
            source_distance = math.dist(sample.position, observation.station)
            self.assertGreater(source_distance, 5.0)
            self.assertLessEqual(source_distance, sample.reception_radius_m + 1e-9)
            bearing = math.degrees(
                math.atan2(
                    sample.position[1] - observation.station[1],
                    sample.position[0] - observation.station[0],
                )
            ) % 360.0
            self.assertLessEqual(
                abs(model.signed_angle_difference_deg(observation.bearing_deg, bearing)),
                1.0 + 1e-12,
            )

        state = build_first_state(
            observation,
            samples=result.samples,
            config=Q2Config(circle_vertices=36),
        )
        self.assertEqual(state.joint_samples, result.samples)

    def test_explicit_error_density_changes_importance_weights(self) -> None:
        observation = FirstDirectionObservation((0.0, 0.0), 0.0)
        uniform = sample_first_direction_posterior(
            observation,
            prior_draws=30000,
            bearing_error_bins=(BearingErrorBin(-1.0, 1.0, 1.0),),
            seed=99,
            config=Q2Config(circle_vertices=36),
        )
        center_peaked = sample_first_direction_posterior(
            observation,
            prior_draws=30000,
            bearing_error_bins=(
                BearingErrorBin(-1.0, -0.25, 0.2),
                BearingErrorBin(-0.25, 0.25, 0.6),
                BearingErrorBin(0.25, 1.0, 0.2),
            ),
            seed=99,
            config=Q2Config(circle_vertices=36),
        )

        # Same prior draws and the same hard support are used; only the explicit
        # likelihood changes the normalized posterior weights.
        self.assertEqual(
            tuple((s.position, s.reception_radius_m) for s in uniform.samples),
            tuple((s.position, s.reception_radius_m) for s in center_peaked.samples),
        )
        self.assertNotEqual(
            tuple(round(s.weight, 14) for s in uniform.samples),
            tuple(round(s.weight, 14) for s in center_peaked.samples),
        )
        self.assertLess(
            center_peaked.effective_sample_size,
            uniform.effective_sample_size + 1e-9,
        )

    def test_error_bins_are_explicit_and_validated(self) -> None:
        observation = FirstDirectionObservation((0.0, 0.0), 0.0)
        base = dict(
            observation=observation,
            prior_draws=100,
            seed=1,
            config=Q2Config(circle_vertices=36),
        )

        with self.assertRaisesRegex(ValueError, "required"):
            sample_first_direction_posterior(
                bearing_error_bins=(),
                **base,
            )
        with self.assertRaisesRegex(ValueError, "sum to one"):
            sample_first_direction_posterior(
                bearing_error_bins=(BearingErrorBin(-1.0, 1.0, 0.9),),
                **base,
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            sample_first_direction_posterior(
                bearing_error_bins=(BearingErrorBin(-1.1, 1.0, 1.0),),
                **base,
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            sample_first_direction_posterior(
                bearing_error_bins=(
                    BearingErrorBin(-1.0, 0.5, 0.5),
                    BearingErrorBin(0.0, 1.0, 0.5),
                ),
                **base,
            )

    def test_sampler_rejects_bad_controls_and_can_enforce_ess_threshold(self) -> None:
        observation = FirstDirectionObservation((0.0, 0.0), 0.0)
        bins = self.uniform_error_bins()

        for bad_draws in (0, -1, 10.5, True):
            with self.subTest(prior_draws=bad_draws):
                with self.assertRaises(ValueError):
                    sample_first_direction_posterior(
                        observation,
                        prior_draws=bad_draws,  # type: ignore[arg-type]
                        bearing_error_bins=bins,
                        seed=1,
                    )

        with self.assertRaises(ValueError):
            sample_first_direction_posterior(
                observation,
                prior_draws=10000,
                bearing_error_bins=bins,
                seed=123,
                min_effective_sample_size=1e9,
                config=Q2Config(circle_vertices=36),
            )



class SecondSupportTests(unittest.TestCase):
    def make_state(self) -> FirstState:
        samples = (
            JointSample((3.0, 4.0), 1000.0, 1.0),
            JointSample((100.0, 0.0), 1000.0, 2.0),
            JointSample((0.0, 100.0), 1000.0, 3.0),
            JointSample((1200.0, 0.0), 1000.0, 4.0),
        )
        return FirstState(
            observation=FirstDirectionObservation((-500.0, 0.0), 0.0),
            exact_support_label="test support",
            outer_region=(
                (-1300.0, -1300.0),
                (1300.0, -1300.0),
                (1300.0, 1300.0),
                (-1300.0, 1300.0),
            ),
            joint_samples=samples,
        )

    def test_near_filters_samples_and_uses_five_meter_outer_disk(self) -> None:
        config = Q2Config(circle_vertices=36)
        state = self.make_state()

        support = second_support(
            (0.0, 0.0),
            SecondResponse(SecondResponseKind.NEAR),
            state,
            config=config,
        )

        self.assertIsInstance(support, SecondSupport)
        self.assertEqual(
            tuple(sample.position for sample in support.sample_support),
            ((3.0, 4.0),),
        )
        self.assertTrue(support.conservative_outer_region)
        maximum_outer_radius = config.near_radius_m / math.cos(
            math.pi / config.circle_vertices
        )
        self.assertTrue(
            all(
                math.dist(point, (0.0, 0.0)) <= maximum_outer_radius + 1e-8
                for point in support.conservative_outer_region
            )
        )

    def test_direction_filters_by_near_radius_fixed_r_and_bearing(self) -> None:
        state = self.make_state()

        support = second_support(
            (0.0, 0.0),
            SecondResponse(SecondResponseKind.DIRECTION, 0.0),
            state,
            config=Q2Config(circle_vertices=36),
        )

        self.assertEqual(
            tuple(sample.position for sample in support.sample_support),
            ((100.0, 0.0),),
        )
        self.assertEqual(support.sample_support[0].reception_radius_m, 1000.0)

    def test_direction_outer_region_reuses_wedge_and_1500m_outer_disk(self) -> None:
        config = Q2Config(circle_vertices=72)
        state = self.make_state()

        support = second_support(
            (0.0, 0.0),
            SecondResponse(SecondResponseKind.DIRECTION, 0.0),
            state,
            config=config,
        )

        maximum_outer_radius = config.reception_radius_max_m / math.cos(
            math.pi / config.circle_vertices
        )
        self.assertTrue(support.conservative_outer_region)
        for point in support.conservative_outer_region:
            self.assertLessEqual(
                math.dist(point, (0.0, 0.0)),
                maximum_outer_radius + 1e-8,
            )
            if math.dist(point, (0.0, 0.0)) > 1e-8:
                bearing = math.degrees(math.atan2(point[1], point[0])) % 360.0
                error = abs(
                    model.signed_angle_difference_deg(bearing, 0.0)
                )
                self.assertLessEqual(error, config.bearing_error_deg + 1e-8)

    def test_no_signal_filters_by_each_samples_fixed_r_and_keeps_k1_outer(self) -> None:
        state = self.make_state()

        support = second_support(
            (0.0, 0.0),
            SecondResponse(SecondResponseKind.NO_SIGNAL),
            state,
            config=Q2Config(circle_vertices=36),
        )

        self.assertEqual(
            tuple(sample.position for sample in support.sample_support),
            ((1200.0, 0.0),),
        )
        self.assertEqual(support.conservative_outer_region, state.outer_region)

    def test_second_response_boundary_rules_match_interface(self) -> None:
        state = FirstState(
            observation=FirstDirectionObservation((-500.0, 0.0), 0.0),
            exact_support_label="test support",
            outer_region=(
                (-10.0, -10.0),
                (1100.0, -10.0),
                (1100.0, 10.0),
                (-10.0, 10.0),
            ),
            joint_samples=(
                JointSample((5.0, 0.0), 1000.0, 1.0),
                JointSample((1000.0, 0.0), 1000.0, 1.0),
                JointSample((1000.001, 0.0), 1000.0, 1.0),
            ),
        )
        config = Q2Config(circle_vertices=36)

        near = second_support(
            (0.0, 0.0),
            SecondResponse(SecondResponseKind.NEAR),
            state,
            config=config,
        )
        direction = second_support(
            (0.0, 0.0),
            SecondResponse(SecondResponseKind.DIRECTION, 0.0),
            state,
            config=config,
        )
        no_signal = second_support(
            (0.0, 0.0),
            SecondResponse(SecondResponseKind.NO_SIGNAL),
            state,
            config=config,
        )

        self.assertEqual(
            tuple(sample.position for sample in near.sample_support),
            ((5.0, 0.0),),
        )
        self.assertEqual(
            tuple(sample.position for sample in direction.sample_support),
            ((1000.0, 0.0),),
        )
        self.assertEqual(
            tuple(sample.position for sample in no_signal.sample_support),
            ((1000.001, 0.0),),
        )

    def test_empty_nominal_branch_is_returned_without_false_certificate(self) -> None:
        state = self.make_state()

        support = second_support(
            (0.0, 0.0),
            SecondResponse(SecondResponseKind.DIRECTION, 180.0),
            state,
            config=Q2Config(circle_vertices=36),
        )

        self.assertEqual(support.sample_support, ())
        self.assertIsInstance(support.conservative_outer_region, tuple)

    def test_second_support_rejects_non_finite_candidate(self) -> None:
        with self.assertRaises(ValueError):
            second_support(
                (math.nan, 0.0),
                SecondResponse(SecondResponseKind.NEAR),
                self.make_state(),
            )


class BayesianEvaluatorTests(unittest.TestCase):
    def make_state(
        self,
        *,
        station: tuple[float, float] = (-500.0, 0.0),
        bearing_deg: float = 0.0,
        samples: tuple[JointSample, ...] | None = None,
    ) -> FirstState:
        if samples is None:
            samples = (
                JointSample((3.0, 0.0), 1000.0, 1.0),
                JointSample((100.0, 0.0), 1000.0, 2.0),
                JointSample((1200.0, 0.0), 1000.0, 1.0),
            )
        return FirstState(
            observation=FirstDirectionObservation(station, bearing_deg),
            exact_support_label="synthetic posterior support for evaluator tests",
            outer_region=(
                (-20.0, -20.0),
                (1300.0, -20.0),
                (1300.0, 20.0),
                (-20.0, 20.0),
            ),
            joint_samples=samples,
        )

    def test_bearing_error_model_is_explicit_and_validated(self) -> None:
        state = self.make_state()
        grid = (0.0, 90.0, 180.0, 270.0)

        with self.assertRaisesRegex(ValueError, "required"):
            evaluate_bayesian(
                (0.0, 0.0),
                state,
                direction_grid_deg=grid,
                bearing_error_atoms=(),
                config=Q2Config(circle_vertices=36),
            )

        with self.assertRaisesRegex(ValueError, "sum to one"):
            evaluate_bayesian(
                (0.0, 0.0),
                state,
                direction_grid_deg=grid,
                bearing_error_atoms=(
                    BearingErrorAtom(0.0, 0.4),
                    BearingErrorAtom(1.0, 0.4),
                ),
                config=Q2Config(circle_vertices=36),
            )

        with self.assertRaisesRegex(ValueError, "outside"):
            evaluate_bayesian(
                (0.0, 0.0),
                state,
                direction_grid_deg=grid,
                bearing_error_atoms=(BearingErrorAtom(1.1, 1.0),),
                config=Q2Config(circle_vertices=36),
            )

    def test_response_probabilities_follow_joint_sample_weights(self) -> None:
        result = evaluate_bayesian(
            (0.0, 0.0),
            self.make_state(),
            direction_grid_deg=(0.0, 90.0, 180.0, 270.0),
            bearing_error_atoms=(BearingErrorAtom(0.0, 1.0),),
            config=Q2Config(circle_vertices=36),
        )

        self.assertIsInstance(result, BayesianEvaluation)
        self.assertAlmostEqual(
            result.response_probabilities[SecondResponseKind.NEAR],
            0.25,
        )
        self.assertAlmostEqual(
            result.response_probabilities[SecondResponseKind.DIRECTION],
            0.50,
        )
        self.assertAlmostEqual(
            result.response_probabilities[SecondResponseKind.NO_SIGNAL],
            0.25,
        )
        self.assertAlmostEqual(sum(metric.probability for metric in result.metrics), 1.0)

    def test_expected_diameter_proxy_is_probability_weighted(self) -> None:
        samples = (
            JointSample((1.0, 0.0), 1000.0, 1.0),
            JointSample((5.0, 0.0), 1000.0, 1.0),
            JointSample((1200.0, 0.0), 1000.0, 2.0),
        )
        result = evaluate_bayesian(
            (0.0, 0.0),
            self.make_state(samples=samples),
            direction_grid_deg=(0.0, 90.0, 180.0, 270.0),
            bearing_error_atoms=(BearingErrorAtom(0.0, 1.0),),
            config=Q2Config(circle_vertices=36),
        )

        # near has probability 1/2 and support diameter 4 m; no_signal has
        # probability 1/2 and one sampled position, so its proxy diameter is 0.
        self.assertAlmostEqual(result.psi_d_m, 2.0)

    def test_direction_grid_wraparound_maps_to_zero_center(self) -> None:
        angle = math.radians(-0.2)
        sample = JointSample(
            (100.0 * math.cos(angle), 100.0 * math.sin(angle)),
            1000.0,
            1.0,
        )
        result = evaluate_bayesian(
            (0.0, 0.0),
            self.make_state(samples=(sample,)),
            direction_grid_deg=(0.0, 90.0, 180.0, 270.0),
            bearing_error_atoms=(BearingErrorAtom(0.0, 1.0),),
            config=Q2Config(circle_vertices=36),
        )

        direction_metrics = [
            metric
            for metric in result.metrics
            if metric.response.kind is SecondResponseKind.DIRECTION
        ]
        self.assertEqual(len(direction_metrics), 1)
        self.assertEqual(direction_metrics[0].response.bearing_deg, 0.0)

    def test_same_station_repeat_reuses_fixed_direction_and_gives_no_new_split(self) -> None:
        samples = (
            JointSample((100.0, 0.0), 1000.0, 1.0),
            JointSample((200.0, 1.0), 1000.0, 1.0),
        )
        state = self.make_state(
            station=(0.0, 0.0),
            bearing_deg=0.0,
            samples=samples,
        )
        expected_diameter = math.dist(samples[0].position, samples[1].position)

        result = evaluate_bayesian(
            (0.0, 0.0),
            state,
            direction_grid_deg=(0.0, 90.0, 180.0, 270.0),
            bearing_error_atoms=(
                BearingErrorAtom(-1.0, 0.25),
                BearingErrorAtom(0.0, 0.50),
                BearingErrorAtom(1.0, 0.25),
            ),
            config=Q2Config(circle_vertices=36),
        )

        self.assertEqual(len(result.metrics), 1)
        self.assertIs(result.metrics[0].response.kind, SecondResponseKind.DIRECTION)
        self.assertEqual(result.metrics[0].response.bearing_deg, 0.0)
        self.assertAlmostEqual(result.psi_d_m, expected_diameter)
        self.assertEqual(result.movement_m, 0.0)

    def test_direction_bin_outer_uses_widened_wedge(self) -> None:
        sample = JointSample((100.0, 0.0), 1000.0, 1.0)
        result = evaluate_bayesian(
            (0.0, 0.0),
            self.make_state(samples=(sample,)),
            direction_grid_deg=tuple(float(value) for value in range(0, 360, 10)),
            bearing_error_atoms=(BearingErrorAtom(1.0, 1.0),),
            config=Q2Config(circle_vertices=72),
        )

        metric = next(
            metric
            for metric in result.metrics
            if metric.response.kind is SecondResponseKind.DIRECTION
        )
        self.assertGreaterEqual(metric.diameter_outer_m + 1e-9, metric.diameter_true_proxy_m)
        self.assertIsNotNone(metric.clear_radius_outer_m)

    def test_direction_grid_must_be_full_equally_spaced_partition(self) -> None:
        state = self.make_state()
        atoms = (BearingErrorAtom(0.0, 1.0),)

        with self.assertRaisesRegex(ValueError, "equally spaced"):
            evaluate_bayesian(
                (0.0, 0.0),
                state,
                direction_grid_deg=(0.0, 80.0, 180.0, 270.0),
                bearing_error_atoms=atoms,
                config=Q2Config(circle_vertices=36),
            )

        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_bayesian(
                (0.0, 0.0),
                state,
                direction_grid_deg=(0.0, 120.0, 360.0),
                bearing_error_atoms=atoms,
                config=Q2Config(circle_vertices=36),
            )

    def test_non_finite_candidate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_bayesian(
                (math.nan, 0.0),
                self.make_state(),
                direction_grid_deg=(0.0, 90.0, 180.0, 270.0),
                bearing_error_atoms=(BearingErrorAtom(0.0, 1.0),),
                config=Q2Config(circle_vertices=36),
            )



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
