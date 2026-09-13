"""History-only action space shared by training, planning and real inference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.common.geometry import candidate_second_points
from src.common.simulator_client import SimulatorClient
from src.q3.baseline_scan import CHANNELS, coverage_points, snake_channel_order
from src.q3.localization_control import ChannelLocalizationDecision, evaluate_channel
from src.q3.search_clear import (
    CLEAR_THRESHOLD_M, SearchClearController, SearchClearState, _StopRun, fallback_step,
)

FEATURE_VERSION = 'q3-candidates-v1'
GLOBAL_DIM, CHANNEL_DIM, CANDIDATE_DIM = 12, 22, 18


@dataclass(frozen=True)
class Candidate:
    kind: str
    channel: int
    position: tuple[float, float]
    source: str
    coverage_index: int | None = None
    radius_upper_m: float | None = None


class AdaptiveController:
    """Finite free-action prefix followed by an uninterrupted certified fallback.

    Selectors receive arrays only. This object holds protocol history, not the
    simulator's source coordinates, radius, seed, or source count.
    """
    def __init__(self, client: SimulatorClient, *, state: SearchClearState | None = None,
                 max_free_actions: int = 128, max_free_time_s: float = 6000.0,
                 virtual_limit_s: float = 360000.0, exit_reserve_s: float = 20.0,
                 prediction_limit: int = 4, particle_count: int = 48,
                 reward_scale: float = 1000.0, fallback_real_reserve_s: float = 120.0):
        if (max_free_actions < 0 or not math.isfinite(max_free_time_s) or max_free_time_s < 0
                or prediction_limit < 0 or particle_count < 3
                or not math.isfinite(reward_scale) or reward_scale <= 0
                or not math.isfinite(fallback_real_reserve_s) or fallback_real_reserve_s < 0):
            raise ValueError('invalid adaptive-control configuration')
        self.client = client
        self.state = state if state is not None else SearchClearState()
        if any(c.observations for c in self.state.channels.values()):
            raise ValueError('adaptive controller requires a fresh history')
        self.state.planned_coverage_points = coverage_points()
        self.state.termination_reason = 'running'
        self.guard = SearchClearController(client, self.state,
                                          virtual_limit_s=virtual_limit_s,
                                          exit_reserve_s=exit_reserve_s)
        self.max_free_actions = max_free_actions
        self.max_free_time_s = max_free_time_s
        self.free_actions = 0
        self.free_time_s = 0.0
        self.fallback_mode = max_free_actions == 0 or max_free_time_s == 0
        self.fallback_channel: int | None = None
        self.prediction_limit = prediction_limit
        self.particle_count = particle_count
        self.reward_scale = reward_scale
        self.fallback_real_reserve_s = fallback_real_reserve_s
        self._evaluations = {}
        self._bounds: dict[int, float] = {}
        self._offered: list[Candidate] = []
        self._predictions = {}
        self._route_rank = {}

    @property
    def done(self) -> bool:
        return self.state.termination_reason != 'running'

    def cost(self, action: Candidate) -> float:
        return (math.dist(self.client.state.position, action.position) / 5.0 + 5.0
                + int(action.kind == 'measure' and action.channel != self.client.state.current_channel))

    def evaluation(self, channel):
        discovery = self.state.channels[channel]
        if channel not in self.state.active_channels:
            return None
        key = (channel, len(discovery.observations))
        if key not in self._evaluations:
            self._evaluations[key] = evaluate_channel(discovery)
        evaluation = self._evaluations[key]
        if evaluation.decision is ChannelLocalizationDecision.MODEL_CONFLICT:
            self.guard._conflict(channel, 'Q1 region is inconsistent')
        return evaluation

    def _channel_fallback(self, channel: int) -> Candidate:
        discovery = self.state.channels[channel]
        near = next((o for o in reversed(discovery.observations) if o.measure_result == 'near'), None)
        if near:
            return Candidate('clear', channel, near.position, 'near', radius_upper_m=5.0)
        try:
            evaluation = self.evaluation(channel)
        except ArithmeticError:
            evaluation = None
        if evaluation and evaluation.decision is ChannelLocalizationDecision.READY_TO_CLEAR:
            return Candidate('clear', channel, evaluation.clear_position, 'q1',
                             radius_upper_m=evaluation.assessment.r_max_m)
        direction = next((o for o in reversed(discovery.observations)
                          if o.measure_result == 'direction'), None)
        if direction is None:
            self.guard._conflict(channel, 'active channel has no direction or near evidence')
        point, upper = fallback_step(direction.position, direction.svd_deg, self._bounds.get(channel, 1500.0))
        return Candidate('clear' if upper <= CLEAR_THRESHOLD_M else 'measure', channel,
                         point, 'analytic_fallback', radius_upper_m=upper)

    def _default_action(self) -> Candidate | None:
        active = self.state.active_channels
        if active:
            if self.fallback_channel not in active:
                self.fallback_channel = min(active)
            return self._channel_fallback(self.fallback_channel)
        for index, point in enumerate(self.state.planned_coverage_points):
            for channel in snake_channel_order(index, self.state.unknown_channels):
                if index not in self.state.coverage_receipts[channel]:
                    return Candidate('measure', channel, point, 'coverage', index)
        return None

    def _finish_evidence(self):
        for channel in list(self.state.unknown_channels):
            if self.state.coverage_receipts[channel] == set(range(7)):
                self.state.channels[channel].status = 'excluded_after_full_cover'
        # Only claim point coverage when every still-unknown channel has a receipt;
        # do not fill unvisited points after a count-based early completion.
        complete = []
        for index, point in enumerate(self.state.planned_coverage_points):
            received_here = any(index in marks for marks in self.state.coverage_receipts.values())
            if received_here and all(index in self.state.coverage_receipts[c]
                                     for c in self.state.unknown_channels):
                complete.append(point)
        self.state.completed_coverage_points = complete
        self.state.completed_full_cover = len(complete) == 7
        if self.state.all_cleared:
            self.state.termination_reason = 'all_cleared'

    def fallback_cost_bound(self) -> float:
        """Loose analytic completion reserve, including all possibly occupied slots.

        Each active station is <=1500 m from a source in the 1800 m disk; after
        successful clear the robot is <=1820 m from the origin. The scan bound
        includes seven <=3000 m legs, all 140 measurements/switches, plus a
        source's fallback and return-to-scan allowance for every uncleared slot.
        It intentionally does not read nominal training source count (10..16).
        """
        radius = math.hypot(*self.client.state.position)
        slots = 20 - len(self.state.cleared_channels) - len(self.state.excluded_channels)
        return ((radius + 21000.0) / 5 + 840.0
                + slots * (336.2 + (3300.0 + max(radius, 1820.0)) / 5))

    def candidates(self) -> list[Candidate]:
        self._offered = []
        self._predictions = {}
        if self.done:
            return []
        self._finish_evidence()
        if self.done:
            return []
        if self.free_actions >= self.max_free_actions or self.free_time_s >= self.max_free_time_s:
            self.fallback_mode = True
        remaining_real = self.client.remaining_real_time_s()
        if remaining_real is not None and remaining_real <= self.fallback_real_reserve_s:
            self.fallback_mode = True
        if self.client.state.virtual_time_s + self.fallback_cost_bound() >= self.guard.virtual_limit_s:
            self.fallback_mode = True
        try:
            default = self._default_action()
            if default is None:
                raise _StopRun('incomplete', 'no legal completion action')
            self.guard.check_budget(default.position, default.channel, default.kind)
            if self.free_time_s+self.cost(default) > self.max_free_time_s:
                self.fallback_mode = True
            actions = [default]
            if not self.fallback_mode:
                for channel in sorted(self.state.active_channels):
                    base = self._channel_fallback(channel)
                    actions.append(base)
                    if base.kind == 'clear':
                        continue
                    direction = next(o for o in reversed(self.state.channels[channel].observations)
                                     if o.measure_result == 'direction')
                    upper = self._bounds.get(channel, 1500.0)
                    # Reuse Q2/shared transverse and forward candidate geometry.
                    points = candidate_second_points(direction.position, direction.svd_deg,
                                                     (min(250.0, upper / 3),),
                                                     (upper / 3, upper / 2))
                    for point in points:
                        actions.append(Candidate('measure', channel, point, 'q2'))
                for index, point in enumerate(self.state.planned_coverage_points):
                    for channel in snake_channel_order(index, self.state.unknown_channels):
                        if index not in self.state.coverage_receipts[channel]:
                            actions.append(Candidate('measure', channel, point, 'coverage', index))
            seen = set()
            for action in actions:
                key = (action.kind, action.channel, action.position)
                if key in seen:
                    continue
                seen.add(key)
                if action != default and action.kind == 'measure' and any(
                        o.position == action.position for o in self.state.channels[action.channel].observations):
                    continue
                if action != default:
                    if self.free_time_s + self.cost(action) > self.max_free_time_s:
                        continue
                    if (self.client.state.virtual_time_s + self.cost(action) + self.fallback_cost_bound()
                            >= self.guard.virtual_limit_s):
                        continue
                self._offered.append(action)
        except _StopRun as stopped:
            self.state.termination_reason = stopped.reason
            self.state.failure_detail = str(stopped)
        return list(self._offered)

    def features(self, actions: list[Candidate]) -> dict[str, np.ndarray]:
        if not actions:
            raise ValueError('terminal/empty action set has no policy features')
        from src.q3.world_model import open_clear_route, predict_measurement
        ready = {a.channel: a.position for a in actions if a.kind == 'clear'}
        route = open_clear_route(self.client.state.position, ready)
        self._route_rank = {channel: i for i, channel in enumerate(route)}
        known_measurements = sorted((a for a in actions if a.kind == 'measure'
                                    and a.channel in self.state.active_channels), key=self.cost)
        limit = 0 if self.fallback_mode else self.prediction_limit
        for action in known_measurements[:limit]:
            self._predictions[action] = predict_measurement(
                self.state.channels[action.channel], action.position,
                particle_count=self.particle_count, seed=0)
        s = self.state
        global_features = [*np.asarray(self.client.state.position) / 1800,
                           len(s.unknown_channels)/20, len(s.active_channels)/20,
                           len(s.cleared_channels)/20, len(s.excluded_channels)/20,
                           self.client.state.virtual_time_s/360000,
                           self.free_actions/max(self.max_free_actions, 1),
                           self.free_time_s/max(self.max_free_time_s, 1),
                           float(self.fallback_mode), len(s.clear_attempts)/20,
                           sum(len(v) for v in s.coverage_receipts.values())/140]
        channel_rows = []
        diameters = {}
        for channel in CHANNELS:
            discovery = s.channels[channel]
            direction = next((o for o in reversed(discovery.observations)
                              if o.measure_result == 'direction'), None)
            near = any(o.measure_result == 'near' for o in discovery.observations)
            try:
                evaluation = self.evaluation(channel)
            except ArithmeticError:
                evaluation = None
            assessment = evaluation.assessment if evaluation else None
            diameter = assessment.diameter_m if assessment and math.isfinite(assessment.diameter_m) else 3600.0
            diameters[channel] = diameter
            channel_rows.append([
                float(channel in s.unknown_channels), float(channel in s.active_channels),
                float(channel in s.cleared_channels), float(channel in s.excluded_channels),
                float(discovery.status == 'model_conflict'),
                *[float(i in s.coverage_receipts[channel]) for i in range(7)],
                *(np.asarray(direction.position)/1800 if direction else [0.0, 0.0]),
                math.sin(math.radians(direction.svd_deg)) if direction else 0.0,
                math.cos(math.radians(direction.svd_deg)) if direction else 0.0,
                float(near), len(discovery.observations)/16,
                diameter/3600, (assessment.r_max_m or 0)/1500 if assessment else 0.0,
                float(channel in ready), float(assessment is not None)])
        rows = []
        for action in actions:
            pred = self._predictions.get(action)
            available = pred is not None and pred.available
            diameter = pred.expected_diameter_m if available else diameters[action.channel]
            rows.append([
                float(action.kind == 'measure'), float(action.kind == 'clear'),
                *((np.asarray(action.position)-np.asarray(self.client.state.position))/1800),
                math.dist(self.client.state.position, action.position)/5000,
                self.cost(action)/1000, float(action.channel == self.client.state.current_channel),
                float(action.coverage_index is not None), float(action.source == 'analytic_fallback'),
                (action.radius_upper_m or 0)/1500, float(action.radius_upper_m is not None),
                self._route_rank.get(action.channel, 20)/20,
                float(available), pred.p_no_signal if available else 0.0,
                pred.p_near if available else 0.0, pred.p_direction if available else 0.0,
                diameter/3600, (diameters[action.channel]-diameter)/3600])
        features = {'global': np.asarray(global_features, dtype=np.float32),
                    'channels': np.asarray(channel_rows, dtype=np.float32),
                    'candidates': np.asarray(rows, dtype=np.float32),
                    'candidate_channels': np.asarray([a.channel-1 for a in actions], dtype=np.int64),
                    'mask': np.ones(len(actions), dtype=np.bool_)}
        if not all(np.isfinite(v).all() for v in features.values()):
            raise ValueError('nonfinite observable features')
        return features

    def planner_scores(self, actions, features):
        """One-step nominal model cost plus a bounded-task heuristic tail.

        Only fully computed particle predictions are used. Unknown occupancy is
        not assigned independent Bernoulli probabilities. These scores are
        approximate priorities, not certified expected total completion times.
        """
        scores = []
        for i, action in enumerate(actions):
            current_diameter = float(features['channels'][action.channel-1, 18])*3600
            if action.kind == 'clear':
                tail_change = -current_diameter/5 - 40 + self._route_rank.get(action.channel, 0)
            elif action.channel in self.state.unknown_channels:
                tail_change = -36.0
            else:
                pred = self._predictions.get(action)
                predicted = (pred.expected_diameter_m if pred and pred.available else
                             min(current_diameter, 2*action.radius_upper_m)
                             if action.radius_upper_m is not None else current_diameter)
                tail_change = (predicted-current_diameter)/5
            scores.append(self.cost(action)+tail_change)
        return np.asarray(scores, dtype=np.float64)

    def step(self, action: Candidate) -> float:
        if self.done or action not in self._offered:
            raise ValueError('action was not supplied by the current candidate snapshot')
        self._offered = []
        before = self.client.state.virtual_time_s
        was_free = not self.fallback_mode
        try:
            self.guard.check_budget(action.position, action.channel, action.kind)
            if action.kind == 'clear':
                self.guard._clear(action.position, action.channel, action.source, action.radius_upper_m)
            else:
                result = self.guard._measure(action.position, action.channel)
                if action.coverage_index is not None:
                    self.state.coverage_receipts[action.channel].add(action.coverage_index)
                if result.result == 'direction':
                    self._bounds[action.channel] = (action.radius_upper_m
                                                   if action.source == 'analytic_fallback' else 1500.0)
                elif result.result == 'no_signal' and action.radius_upper_m is not None and action.radius_upper_m <= 1000:
                    self.guard._conflict(action.channel, 'no signal inside certified reception bound')
            self._finish_evidence()
        except _StopRun as stopped:
            self.state.termination_reason, self.state.failure_detail = stopped.reason, str(stopped)
        except Exception as error:
            self.state.termination_reason = 'error'
            self.state.failure_detail = f'{type(error).__name__}: {error}'
            raise
        elapsed = self.client.state.virtual_time_s-before
        if was_free:
            self.free_actions += 1
            self.free_time_s += elapsed
            if self.free_actions >= self.max_free_actions or self.free_time_s >= self.max_free_time_s:
                self.fallback_mode = True
        return -elapsed/self.reward_scale
