"""Q3 deterministic search/localize/clear loop using the existing Q1 adapter.

The paper's section 7 fallback provides finite progress without a learned model.
Only protocol observations reach this controller; offline truth never does.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from src.common.geometry import Point
from src.common.metrics import run_summary
from src.common.simulator_client import SimulatorClient
from src.q3.baseline_scan import CHANNELS, DiscoveryState, coverage_points, snake_channel_order
from src.q3.localization_control import ChannelLocalizationDecision, evaluate_channel

STRATEGY_NAME = "q3-v1-search-localize-clear"
# A small engineering guard for coordinates/bounds, not interval arithmetic.
BOUND_GUARD_M = 1e-6
CLEAR_THRESHOLD_M = 20.0 - 1e-3


def fallback_step(reference: Point, bearing_deg: float, upper_m: float) -> tuple[Point, float]:
    """Section 7: move U/(2 cos(1 degree)) from the actual bearing station.

    The returned radius is centered at the returned point, never at the robot's
    previous position. The guard is carried forward with each new observation.
    """
    if (len(reference) != 2 or not all(math.isfinite(v) for v in reference)
            or not math.isfinite(bearing_deg)
            or not math.isfinite(upper_m) or not 0 < upper_m <= 1500.0):
        raise ValueError("fallback requires finite coordinates/bearing and 0 < U <= 1500")
    step = upper_m / (2.0 * math.cos(math.radians(1.0)))
    angle = math.radians(bearing_deg % 360.0)
    point = (reference[0] + step * math.cos(angle), reference[1] + step * math.sin(angle))
    return point, step + BOUND_GUARD_M


@dataclass(frozen=True)
class ClearAttempt:
    channel: int
    position: Point
    result: str
    virtual_time_s: float
    certificate: str
    radius_upper_m: float | None


@dataclass
class SearchClearState(DiscoveryState):
    """Keep successful receipts and failures separately from inferred absence."""
    clear_attempts: list[ClearAttempt] = field(default_factory=list)
    coverage_receipts: dict[int, set[int]] = field(
        default_factory=lambda: {channel: set() for channel in CHANNELS})
    termination_reason: str = "not_started"
    failure_detail: str | None = None

    @property
    def detected_channels(self) -> set[int]:
        return {channel for channel, discovery in self.channels.items()
                if any(o.measure_result in {"near", "direction"}
                       for o in discovery.observations)}

    @property
    def cleared_channels(self) -> set[int]:
        return {channel for channel, discovery in self.channels.items()
                if discovery.status == "cleared"}

    @property
    def active_channels(self) -> set[int]:
        return self.detected_channels - self.cleared_channels

    @property
    def all_cleared(self) -> bool:
        return (not self.unknown_channels and not self.active_channels
                and self.cleared_channels | self.excluded_channels == set(CHANNELS))


class _StopRun(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason


class SearchClearController:
    """Serial policy. No network calls occur until run() and no /enter is sent."""

    def __init__(self, client: SimulatorClient, state: SearchClearState, *,
                 virtual_limit_s: float = 360000.0, exit_reserve_s: float = 20.0):
        if (not math.isfinite(virtual_limit_s) or not 0 < virtual_limit_s <= 360000
                or not math.isfinite(exit_reserve_s) or exit_reserve_s < 0):
            raise ValueError("require 0 < virtual limit <= 360000 and nonnegative exit reserve")
        self.client = client
        self.state = state
        self.virtual_limit_s = virtual_limit_s
        self.exit_reserve_s = exit_reserve_s

    def check_budget(self, position: Point, channel: int, action: str) -> None:
        """Guard a complete protocol request; also shared by the scan comparison."""
        if (len(position) != 2 or not all(math.isfinite(v) and abs(v) <= 2_000_000
                                        for v in position)):
            raise _StopRun("model_conflict", "nonfinite or out-of-range action coordinates")
        remaining = self.client.remaining_real_time_s()
        retries = self.client.max_network_retries
        request_allowance = ((retries + 1) * self.client.timeout_s
                             + self.client.retry_backoff_s * (2**retries - 1))
        # Reserve a complete retry allowance for /exit as well as this action.
        reserve = max(self.exit_reserve_s, request_allowance + 1.0)
        if remaining is None or remaining <= request_allowance + reserve:
            raise _StopRun("real_time_budget", "insufficient time for an action and safe exit")
        cost = math.dist(self.client.state.position, position) / 5.0 + 5.0
        if action == "measure" and channel != self.client.state.current_channel:
            cost += 1.0
        if self.client.state.virtual_time_s + cost >= self.virtual_limit_s:
            raise _StopRun("virtual_time_budget", "next action would reach virtual-time limit")

    def _measure(self, position: Point, channel: int):
        self.check_budget(position, channel, "measure")
        result = self.client.measure(position, channel)
        self.state.record_measure(position, channel, result)
        return result

    def _conflict(self, channel: int, detail: str) -> None:
        self.state.channels[channel].status = "model_conflict"
        raise _StopRun("model_conflict", f"channel {channel}: {detail}")

    def _clear(self, position: Point, channel: int, certificate: str,
               radius_upper_m: float | None) -> None:
        self.check_budget(position, channel, "clear")
        result = self.client.clear(position, channel)
        self.state.clear_attempts.append(ClearAttempt(
            channel, position, result.result, self.client.state.virtual_time_s,
            certificate, radius_upper_m))
        if result.result != "success":
            self._conflict(channel, "certified clear returned no_target_in_range; do not retry")
        self.state.channels[channel].status = "cleared"

    def _localize_and_clear(self, channel: int) -> None:
        discovery = self.state.channels[channel]
        upper_m = 1500.0
        # Six further direction measurements and a seventh move suffice in the
        # ideal model. The finite cap also guards implementation/protocol errors.
        for _ in range(8):
            latest = discovery.observations[-1]
            if latest.measure_result == "near":
                self._clear(latest.position, channel, "near", 5.0)
                return
            if latest.measure_result != "direction":
                self._conflict(channel, "no signal inside guaranteed reception region")
            try:
                evaluation = evaluate_channel(discovery)
            except ArithmeticError:
                # Numerical optimization failure cannot block analytic progress.
                evaluation = None
            if evaluation is not None:
                if evaluation.decision is ChannelLocalizationDecision.MODEL_CONFLICT:
                    self._conflict(channel, "Q1 conservative region is empty")
                if evaluation.decision is ChannelLocalizationDecision.READY_TO_CLEAR:
                    if evaluation.clear_position is None:
                        self._conflict(channel, "Q1 ready assessment lacks a clear position")
                    radius = evaluation.assessment.r_max_m if evaluation.assessment else 5.0
                    self._clear(evaluation.clear_position, channel, "q1", radius)
                    return
            point, new_upper = fallback_step(latest.position, latest.svd_deg, upper_m)
            if new_upper <= CLEAR_THRESHOLD_M:
                self._clear(point, channel, "analytic_fallback", new_upper)
                return
            self._measure(point, channel)
            upper_m = new_upper
        self._conflict(channel, "finite fallback action bound exceeded")

    def run(self) -> SearchClearState:
        state = self.state
        if state.termination_reason != "not_started" or any(
                c.observations for c in state.channels.values()):
            raise ValueError("use a fresh state for each run; implicit resume is unsupported")
        state.termination_reason = "running"
        state.planned_coverage_points = coverage_points()
        try:
            for index, point in enumerate(state.planned_coverage_points):
                for channel in snake_channel_order(index, state.unknown_channels):
                    result = self._measure(point, channel)
                    state.coverage_receipts[channel].add(index)
                    if result.result in {"direction", "near"}:
                        self._localize_and_clear(channel)
                state.completed_coverage_points.append(point)
                if state.all_cleared:
                    # All 20 channel slots have successful clear receipts. Later
                    # waypoints were not visited and must not appear as covered.
                    state.termination_reason = "all_cleared"
                    state.completed_full_cover = (
                        len(state.completed_coverage_points) == len(state.planned_coverage_points))
                    return state
            # Per-channel receipt proof: seven negative observations at the seven
            # fixed points, not merely a counter of visited waypoints.
            for channel in state.unknown_channels:
                if state.coverage_receipts[channel] != set(range(7)):
                    raise _StopRun("incomplete_coverage", f"channel {channel} has missing receipts")
                state.channels[channel].status = "excluded_after_full_cover"
            state.completed_full_cover = True
            state.termination_reason = "all_cleared" if state.all_cleared else "incomplete"
        except _StopRun as stopped:
            state.termination_reason = stopped.reason
            state.failure_detail = str(stopped)
        except Exception as error:
            state.termination_reason = "error"
            state.failure_detail = f"{type(error).__name__}: {error}"
            raise
        return state


def run_search_and_clear(client: SimulatorClient, *, state: SearchClearState | None = None,
                         virtual_limit_s: float = 360000.0,
                         exit_reserve_s: float = 20.0) -> SearchClearState:
    return SearchClearController(client, state if state is not None else SearchClearState(),
                                 virtual_limit_s=virtual_limit_s,
                                 exit_reserve_s=exit_reserve_s).run()


def build_completion_summary(state: SearchClearState, final_virtual_time_s: float,
                             program_runtime_s: float = 0.0) -> dict[str, object]:
    from src.q3.baseline_scan import build_summary

    summary = build_summary(state, final_virtual_time_s)
    # Total population is unknown until the full absence/clearance proof holds.
    total_count = len(state.detected_channels) if state.all_cleared else None
    summary.update(run_summary(
        cleared_count=len(state.cleared_channels), total_count=total_count,
        localization_clearance_time_s=final_virtual_time_s,
        program_runtime_s=program_runtime_s, all_cleared=state.all_cleared,
        failure_reason=state.failure_detail))
    summary.update({
        "baseline_name": STRATEGY_NAME,
        "cleared_channels": sorted(state.cleared_channels),
        "active_channels": sorted(state.active_channels),
        "unknown_channels": sorted(state.unknown_channels),
        "channel_statuses": {str(k): v.status for k, v in state.channels.items()},
        "coverage_receipts": {str(k): sorted(v) for k, v in state.coverage_receipts.items()},
        "clear_attempts": [asdict(attempt) for attempt in state.clear_attempts],
        "clear_count": len(state.clear_attempts),
        "termination_reason": state.termination_reason,
        "total_count_basis": (
            "all_channel_slots_cleared" if len(state.cleared_channels) == len(CHANNELS)
            else "coverage_and_clear_receipts" if state.all_cleared else "unknown"),
    })
    return summary
