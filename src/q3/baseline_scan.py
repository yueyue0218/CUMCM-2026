"""Q3-v0 deterministic full-channel discovery baseline.

This module deliberately stops after discovery.  It does not localize, clear,
or optimize the travel path.  The seven-point layout is a simple conservative
coverage baseline, not a claim of an optimal covering layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from src.common.simulator_client import MeasureResult, SimulatorClient

CHANNELS = tuple(range(1, 21))
BASELINE_NAME = "q3-v0-fixed-coverage-scan"


def coverage_points() -> list[tuple[float, float]]:
    """Return the origin and the six vertices of a radius-1500 hexagon."""

    points = [(0.0, 0.0)]
    for angle_deg in range(0, 360, 60):
        angle_rad = math.radians(angle_deg)
        points.append(
            (1500.0 * math.cos(angle_rad), 1500.0 * math.sin(angle_rad))
        )
    return points


def snake_channel_order(point_index: int, unknown_channels: Iterable[int]) -> list[int]:
    """Order remaining channels ascending at even points and descending at odd ones."""

    if point_index < 0:
        raise ValueError("point_index must be non-negative")
    channels = set(unknown_channels)
    if any(
        not isinstance(channel, int)
        or isinstance(channel, bool)
        or channel not in CHANNELS
        for channel in channels
    ):
        raise ValueError("channels must be integers in 1..20")
    return sorted(channels, reverse=point_index % 2 == 1)


@dataclass(frozen=True)
class MeasureObservation:
    position: tuple[float, float]
    channel: int
    measure_result: str
    svd_deg: float | None
    virtual_time_s: float


@dataclass
class ChannelDiscovery:
    status: str = "unknown"
    observations: list[MeasureObservation] = field(default_factory=list)


@dataclass
class DiscoveryState:
    channels: dict[int, ChannelDiscovery] = field(
        default_factory=lambda: {channel: ChannelDiscovery() for channel in CHANNELS}
    )
    planned_coverage_points: list[tuple[float, float]] = field(default_factory=list)
    completed_coverage_points: list[tuple[float, float]] = field(default_factory=list)
    completed_full_cover: bool = False

    @property
    def unknown_channels(self) -> set[int]:
        return {
            channel
            for channel, discovery in self.channels.items()
            if discovery.status == "unknown"
        }

    @property
    def detected_channels(self) -> set[int]:
        return {
            channel
            for channel, discovery in self.channels.items()
            if discovery.status == "detected"
        }

    @property
    def excluded_channels(self) -> set[int]:
        return {
            channel
            for channel, discovery in self.channels.items()
            if discovery.status == "excluded_after_full_cover"
        }

    def record_measure(
        self,
        position: tuple[float, float],
        channel: int,
        result: MeasureResult,
    ) -> None:
        if channel not in self.channels:
            raise ValueError("channel must be in 1..20")
        if result.result not in {"direction", "near", "no_signal"}:
            raise ValueError(f"unknown measure result: {result.result!r}")
        virtual_time = result.response.get("virtual_time_s")
        if not isinstance(virtual_time, (int, float)) or isinstance(virtual_time, bool):
            raise ValueError("measure result lacks numeric virtual_time_s")

        observation = MeasureObservation(
            position=(float(position[0]), float(position[1])),
            channel=channel,
            measure_result=result.result,
            svd_deg=result.svd_deg if result.result == "direction" else None,
            virtual_time_s=float(virtual_time),
        )
        discovery = self.channels[channel]
        discovery.observations.append(observation)
        if result.result in {"direction", "near"}:
            discovery.status = "detected"

    def finish_full_cover(self, expected_point_count: int) -> None:
        """Exclude unseen channels only after every planned point completed."""

        if len(self.completed_coverage_points) != expected_point_count:
            raise ValueError("cannot finish before every coverage point is complete")
        self.completed_full_cover = True
        for discovery in self.channels.values():
            if discovery.status == "unknown":
                discovery.status = "excluded_after_full_cover"


def run_fixed_coverage_scan(
    client: SimulatorClient,
    *,
    state: DiscoveryState | None = None,
    points: list[tuple[float, float]] | None = None,
) -> DiscoveryState:
    """Scan unknown channels at all fixed points, propagating client failures.

    Only the built-in layout is coverage-certified.  Custom layouts are scanned
    and recorded but never treated as a full cover.  A point is marked complete
    only after every channel that was unknown upon arrival has produced a
    successful measurement.  Therefore a raised client exception leaves
    ``completed_full_cover`` false and cannot exclude channels.
    """

    scan_state = state if state is not None else DiscoveryState()
    using_default_layout = points is None
    planned_points = list(points) if points is not None else coverage_points()
    scan_state.planned_coverage_points = planned_points

    for point_index, position in enumerate(planned_points):
        for channel in snake_channel_order(point_index, scan_state.unknown_channels):
            result = client.measure(position, channel)
            scan_state.record_measure(position, channel, result)
        scan_state.completed_coverage_points.append(position)

    if using_default_layout:
        scan_state.finish_full_cover(len(planned_points))
    return scan_state


def build_summary(state: DiscoveryState, final_virtual_time_s: float) -> dict[str, object]:
    observations = [
        observation
        for discovery in state.channels.values()
        for observation in discovery.observations
    ]
    return {
        "baseline_name": BASELINE_NAME,
        "coverage_points": state.planned_coverage_points,
        "completed_coverage_points": state.completed_coverage_points,
        "detected_channels": sorted(state.detected_channels),
        "excluded_channels": sorted(state.excluded_channels),
        "detected_count": len(state.detected_channels),
        "measure_count": len(observations),
        "direction_count": sum(
            observation.measure_result == "direction" for observation in observations
        ),
        "near_count": sum(
            observation.measure_result == "near" for observation in observations
        ),
        "no_signal_count": sum(
            observation.measure_result == "no_signal" for observation in observations
        ),
        "final_virtual_time_s": float(final_virtual_time_s),
        "completed_full_cover": state.completed_full_cover,
    }
