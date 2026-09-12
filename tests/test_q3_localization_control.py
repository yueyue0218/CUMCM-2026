import unittest
from unittest.mock import patch

from src.common.localization import (
    BearingObservation,
    LocalizationAssessment,
    LocalizationStatus,
)
from src.q3.baseline_scan import ChannelDiscovery, MeasureObservation
from src.q3.localization_control import (
    ChannelLocalizationDecision,
    assess_channel_localization,
    direction_observations,
    evaluate_channel,
    latest_near_observation,
)


def saved_observation(
    measure_result: str,
    *,
    position: tuple[float, float] = (10.0, 20.0),
    svd_deg: float | None = None,
    virtual_time_s: float = 5.0,
) -> MeasureObservation:
    return MeasureObservation(
        position=position,
        channel=3,
        measure_result=measure_result,
        svd_deg=svd_deg,
        virtual_time_s=virtual_time_s,
    )


class DirectionObservationTests(unittest.TestCase):
    def test_single_direction_is_converted(self) -> None:
        discovery = ChannelDiscovery(
            status="detected",
            observations=[saved_observation("direction", svd_deg=37.5)],
        )

        self.assertEqual(
            direction_observations(discovery),
            [BearingObservation(station=(10.0, 20.0), bearing_deg=37.5)],
        )

    def test_no_signal_results_before_direction_are_ignored(self) -> None:
        discovery = ChannelDiscovery(
            status="detected",
            observations=[
                saved_observation("no_signal", position=(0.0, 0.0)),
                saved_observation("no_signal", position=(1.0, 0.0)),
                saved_observation(
                    "direction",
                    position=(2.0, 0.0),
                    svd_deg=91.25,
                ),
            ],
        )

        self.assertEqual(
            direction_observations(discovery),
            [BearingObservation(station=(2.0, 0.0), bearing_deg=91.25)],
        )

    def test_multiple_directions_preserve_original_order(self) -> None:
        discovery = ChannelDiscovery(
            status="detected",
            observations=[
                saved_observation("direction", position=(3.0, 4.0), svd_deg=10.0),
                saved_observation("near", position=(5.0, 6.0)),
                saved_observation("direction", position=(7.0, 8.0), svd_deg=220.0),
            ],
        )

        self.assertEqual(
            direction_observations(discovery),
            [
                BearingObservation(station=(3.0, 4.0), bearing_deg=10.0),
                BearingObservation(station=(7.0, 8.0), bearing_deg=220.0),
            ],
        )

    def test_direction_without_svd_raises_value_error(self) -> None:
        discovery = ChannelDiscovery(
            status="detected",
            observations=[saved_observation("direction", svd_deg=None)],
        )

        with self.assertRaisesRegex(ValueError, "direction observation requires svd_deg"):
            direction_observations(discovery)


class ChannelAssessmentTests(unittest.TestCase):
    def test_near_only_channel_is_rejected(self) -> None:
        discovery = ChannelDiscovery(
            status="detected",
            observations=[saved_observation("near")],
        )

        with self.assertRaisesRegex(
            ValueError,
            "channel localization requires at least one direction observation",
        ):
            assess_channel_localization(discovery)

    def test_channel_without_direction_is_rejected(self) -> None:
        discovery = ChannelDiscovery(
            observations=[saved_observation("no_signal")],
        )

        with self.assertRaisesRegex(
            ValueError,
            "channel localization requires at least one direction observation",
        ):
            assess_channel_localization(discovery)


class NearObservationTests(unittest.TestCase):
    def test_latest_near_returns_last_near_observation(self) -> None:
        first_near = saved_observation("near", position=(1.0, 2.0))
        last_near = saved_observation("near", position=(5.0, 6.0))
        discovery = ChannelDiscovery(
            status="detected",
            observations=[
                first_near,
                saved_observation("direction", svd_deg=45.0),
                last_near,
                saved_observation("no_signal"),
            ],
        )

        self.assertIs(latest_near_observation(discovery), last_near)


class ChannelEvaluationTests(unittest.TestCase):
    def test_near_only_is_ready_at_near_position_without_assessment(self) -> None:
        discovery = ChannelDiscovery(
            status="detected",
            observations=[saved_observation("near", position=(12.0, -8.0))],
        )

        evaluation = evaluate_channel(discovery)

        self.assertIs(
            evaluation.decision,
            ChannelLocalizationDecision.READY_TO_CLEAR,
        )
        self.assertEqual(evaluation.clear_position, (12.0, -8.0))
        self.assertIsNone(evaluation.assessment)

    def test_later_near_takes_priority_without_q1_assessment(self) -> None:
        discovery = ChannelDiscovery(
            status="detected",
            observations=[
                saved_observation("direction", svd_deg=30.0),
                saved_observation("near", position=(7.0, 9.0)),
            ],
        )

        with patch(
            "src.q3.localization_control.assess_channel_localization"
        ) as assess:
            evaluation = evaluate_channel(discovery)

        assess.assert_not_called()
        self.assertIs(
            evaluation.decision,
            ChannelLocalizationDecision.READY_TO_CLEAR,
        )
        self.assertEqual(evaluation.clear_position, (7.0, 9.0))
        self.assertIsNone(evaluation.assessment)

    def test_multiple_near_results_use_last_position(self) -> None:
        discovery = ChannelDiscovery(
            status="detected",
            observations=[
                saved_observation("near", position=(1.0, 2.0)),
                saved_observation("near", position=(3.0, 4.0)),
            ],
        )

        evaluation = evaluate_channel(discovery)

        self.assertEqual(evaluation.clear_position, (3.0, 4.0))
        self.assertIsNone(evaluation.assessment)

    def test_q1_statuses_map_directly_without_geometry_reassessment(self) -> None:
        cases = (
            (
                LocalizationStatus.CLEAR_READY,
                ChannelLocalizationDecision.READY_TO_CLEAR,
                9876.0,
                (4.0, 5.0),
            ),
            (
                LocalizationStatus.COVERAGE_UNCERTAIN,
                ChannelLocalizationDecision.NEEDS_MORE_MEASUREMENT,
                0.0,
                None,
            ),
            (
                LocalizationStatus.MODEL_CONFLICT,
                ChannelLocalizationDecision.MODEL_CONFLICT,
                -123.0,
                None,
            ),
        )

        for status, expected_decision, diameter_m, expected_position in cases:
            with self.subTest(status=status):
                assessment = LocalizationAssessment(
                    status=status,
                    outer_region=(),
                    diameter_m=diameter_m,
                    clear_position=(4.0, 5.0),
                    r_max_m=None,
                    clear_ready=status is LocalizationStatus.CLEAR_READY,
                    reason="patched Q1 assessment",
                )
                with patch(
                    "src.q3.localization_control.assess_channel_localization",
                    return_value=assessment,
                ):
                    evaluation = evaluate_channel(ChannelDiscovery())

                self.assertIs(evaluation.decision, expected_decision)
                self.assertEqual(evaluation.clear_position, expected_position)
                self.assertIs(evaluation.assessment, assessment)

    def test_no_signal_only_evaluation_is_rejected(self) -> None:
        discovery = ChannelDiscovery(
            observations=[saved_observation("no_signal")],
        )

        with self.assertRaisesRegex(
            ValueError,
            "channel localization requires at least one direction observation",
        ):
            evaluate_channel(discovery)

    def test_real_direction_flows_through_q1_to_q3_evaluation(self) -> None:
        discovery = ChannelDiscovery(
            status="detected",
            observations=[
                saved_observation(
                    "direction",
                    position=(100.0, -50.0),
                    svd_deg=42.0,
                )
            ],
        )

        evaluation = evaluate_channel(discovery)

        self.assertIsInstance(evaluation.assessment, LocalizationAssessment)
        self.assertIn(
            evaluation.decision,
            {
                ChannelLocalizationDecision.READY_TO_CLEAR,
                ChannelLocalizationDecision.NEEDS_MORE_MEASUREMENT,
            },
        )
        if evaluation.decision is ChannelLocalizationDecision.READY_TO_CLEAR:
            self.assertEqual(
                evaluation.clear_position,
                evaluation.assessment.clear_position,
            )
        else:
            self.assertIsNone(evaluation.clear_position)


if __name__ == "__main__":
    unittest.main()
