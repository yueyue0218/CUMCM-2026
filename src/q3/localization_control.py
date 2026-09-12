"""Pure Q3-v1 adaptation of saved discovery data to the Q1 localization API."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.common.localization import (
    BearingObservation,
    LocalizationAssessment,
    LocalizationStatus,
    assess_observations_for_clear,
)
from src.q3.baseline_scan import ChannelDiscovery


class ChannelLocalizationDecision(str, Enum):
    READY_TO_CLEAR = "READY_TO_CLEAR"
    NEEDS_MORE_MEASUREMENT = "NEEDS_MORE_MEASUREMENT"
    MODEL_CONFLICT = "MODEL_CONFLICT"


@dataclass(frozen=True)
class ChannelLocalizationEvaluation:
    decision: ChannelLocalizationDecision
    assessment: LocalizationAssessment


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
    """Map the Q1 assessment status to the corresponding Q3 decision."""

    assessment = assess_channel_localization(discovery)
    decisions = {
        LocalizationStatus.CLEAR_READY: ChannelLocalizationDecision.READY_TO_CLEAR,
        LocalizationStatus.COVERAGE_UNCERTAIN: (
            ChannelLocalizationDecision.NEEDS_MORE_MEASUREMENT
        ),
        LocalizationStatus.MODEL_CONFLICT: ChannelLocalizationDecision.MODEL_CONFLICT,
    }
    return ChannelLocalizationEvaluation(
        decision=decisions[assessment.status],
        assessment=assessment,
    )
