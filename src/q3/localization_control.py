"""Pure Q3-v1 adaptation of saved discovery data to the Q1 localization API."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.common.geometry import Point
from src.common.localization import (
    BearingObservation,
    LocalizationAssessment,
    LocalizationStatus,
    assess_observations_for_clear,
)
from src.q3.baseline_scan import ChannelDiscovery, MeasureObservation


class ChannelLocalizationDecision(str, Enum):
    READY_TO_CLEAR = "READY_TO_CLEAR"
    NEEDS_MORE_MEASUREMENT = "NEEDS_MORE_MEASUREMENT"
    MODEL_CONFLICT = "MODEL_CONFLICT"


@dataclass(frozen=True)
class ChannelLocalizationEvaluation:
    decision: ChannelLocalizationDecision
    clear_position: Point | None
    assessment: LocalizationAssessment | None


def direction_observations(
    discovery: ChannelDiscovery,
) -> list[BearingObservation]:
    """Convert saved direction results to Q1 observations in original order."""

    converted: list[BearingObservation] = []
    for observation in discovery.observations:
        if observation.measure_result != "direction":
            continue
        if observation.svd_deg is None:
            raise ValueError("direction observation requires svd_deg")
        converted.append(
            BearingObservation(
                station=observation.position,
                bearing_deg=observation.svd_deg,
            )
        )
    return converted


def latest_near_observation(
    discovery: ChannelDiscovery,
) -> MeasureObservation | None:
    """Return the last saved near observation, if one exists."""

    return next(
        (
            observation
            for observation in reversed(discovery.observations)
            if observation.measure_result == "near"
        ),
        None,
    )


def assess_channel_localization(
    discovery: ChannelDiscovery,
) -> LocalizationAssessment:
    """Assess a channel having at least one saved direction observation."""

    observations = direction_observations(discovery)
    if not observations:
        raise ValueError(
            "channel localization requires at least one direction observation"
        )
    return assess_observations_for_clear(observations)


def evaluate_channel(
    discovery: ChannelDiscovery,
) -> ChannelLocalizationEvaluation:
    """Use a near shortcut or map the Q1 status to the Q3 decision."""

    near_observation = latest_near_observation(discovery)
    if near_observation is not None:
        return ChannelLocalizationEvaluation(
            decision=ChannelLocalizationDecision.READY_TO_CLEAR,
            clear_position=near_observation.position,
            assessment=None,
        )

    assessment = assess_channel_localization(discovery)
    decisions = {
        LocalizationStatus.CLEAR_READY: ChannelLocalizationDecision.READY_TO_CLEAR,
        LocalizationStatus.COVERAGE_UNCERTAIN: (
            ChannelLocalizationDecision.NEEDS_MORE_MEASUREMENT
        ),
        LocalizationStatus.MODEL_CONFLICT: ChannelLocalizationDecision.MODEL_CONFLICT,
    }
    clear_position = (
        assessment.clear_position
        if assessment.status is LocalizationStatus.CLEAR_READY
        else None
    )
    return ChannelLocalizationEvaluation(
        decision=decisions[assessment.status],
        clear_position=clear_position,
        assessment=assessment,
    )
