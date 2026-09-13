"""Batched Q4 controller: positive-bearing outer bounds and mixed coverage."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from src.q3.baseline_scan import coverage_points
from src.q4.q3_adapter import EfficientController, EfficientState, optimized_route
from src.q3.localization_control import ChannelLocalizationDecision
from src.q3.search_clear import ClearAttempt, SearchClearController, _StopRun, build_completion_summary
from src.q4.coverage import covers_mixed, coverage_route, fallback_grid


def optical_points(station, bearing):
    """54 x 2 centers covering the first positive bearing's 1500 m rectangle."""
    a = math.radians(bearing)
    u, v = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
    half_width = 1500*math.sin(math.radians(1))
    return [(station[0]+(i+.5)*1500/54*u[0]+j*half_width/2*v[0],
             station[1]+(i+.5)*1500/54*u[1]+j*half_width/2*v[1])
            for row, j in enumerate((-1, 1))
            for i in (range(54) if row == 0 else range(53, -1, -1))]


@dataclass
class MixedState(EfficientState):
    actions_started: int = 0
    network_requests: int = 0
    planning_time_s: float = 0.
    coverage_basis: str | None = None


class MixedGuard(SearchClearController):
    def __init__(self, *args, max_actions, max_compute_s, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_actions = max_actions
        self.max_compute_s = max_compute_s
        self.cpu_started = time.process_time()

    def check_budget(self, position, channel, action):
        super().check_budget(position, channel, action)
        self.state.planning_time_s = time.process_time()-self.cpu_started
        if self.state.planning_time_s >= self.max_compute_s:
            raise _StopRun('compute_budget', 'cumulative computation budget exhausted')
        if self.state.actions_started >= self.max_actions:
            raise _StopRun('action_budget', 'new action budget exhausted')
        if self.state.network_requests >= self.max_requests-1:
            raise _StopRun('request_budget', 'reserve last network request for exit')
        self.state.actions_started += 1


class MixedController(EfficientController):
    def __init__(self, client, state=None, *, virtual_limit_s=360000., exit_reserve_s=20.,
                 max_actions=4096, max_requests=12300, max_compute_s=900.,
                 active_steps=8, scan_spacing_m=550., strategy='adaptive'):
        if (any(isinstance(x, bool) or not isinstance(x, int) or x < 1
                for x in (max_actions, max_requests)) or max_requests < 3
                or not math.isfinite(max_compute_s) or max_compute_s <= 0
                or isinstance(active_steps, bool) or not isinstance(active_steps, int)
                or not 0 <= active_steps <= 32 or strategy not in {'adaptive', 'seven_grid'}):
            raise ValueError('invalid Q4 budget or strategy')
        super().__init__(client, state if state is not None else MixedState(),
                         virtual_limit_s=virtual_limit_s, exit_reserve_s=exit_reserve_s,
                         scan_spacing_m=scan_spacing_m)
        if not isinstance(self.state, MixedState):
            raise TypeError('MixedState required')
        if self.state.source_count_upper != 16:
            raise ValueError('Q4 uses the public bound 16, never the hidden scene count')
        self.guard = MixedGuard(client, self.state, max_actions=max_actions,
                                max_compute_s=max_compute_s, virtual_limit_s=virtual_limit_s,
                                exit_reserve_s=exit_reserve_s)
        self.max_requests = max_requests
        self.guard.max_requests = max_requests
        self.active_steps = active_steps
        self.strategy = strategy
        self.anchors = coverage_points()
        self.state.planned_coverage_points = self.anchors+fallback_grid()
        self.state.controller_class=f'{type(self).__module__}:{type(self).__name__}'
        self._original_transport = client.transport
        self.state.network_requests = int(client.state.entered)
        client.transport = self._transport
        self.joint_cache = {}

    def estimate(self, c):
        from src.q4.joint_model import sampled_center
        e = self.evaluation(c)
        if e.decision is ChannelLocalizationDecision.READY_TO_CLEAR:
            return e.clear_position
        history = self.state.channels[c].observations
        key = c, len(history)
        if key not in self.joint_cache:
            self.joint_cache[key] = sampled_center(e.assessment.outer_region, history)
        return self.joint_cache[key] or e.assessment.clear_position

    def _transport(self, url, body, timeout):
        # Leave one request for /exit. Retries count, including rejected requests.
        ceiling = self.max_requests if url.endswith('/exit') else self.max_requests-1
        if self.state.network_requests >= ceiling:
            raise _StopRun('request_budget', 'network request budget exhausted')
        self.state.network_requests += 1
        return self._original_transport(url, body, timeout)

    def batch(self, position, anchor=None, target_channel=None, force_scan=False):
        s = self.state
        full = self.should_scan(position, force_scan)
        unknown = s.unknown_channels if full else set()
        active = set()
        for c in s.active_channels:
            e = self.evaluation(c)
            if e.decision is ChannelLocalizationDecision.READY_TO_CLEAR:
                continue
            history = s.channels[c].observations
            if any(math.dist(position, o.position) < 1e-6 for o in history):
                continue
            center = self.estimate(c)
            last = next(o for o in reversed(history) if o.measure_result == 'direction')
            angle = math.degrees(math.atan2(center[1]-position[1], center[0]-position[0]))
            turn = abs((angle-last.svd_deg+180) % 360-180)
            if c == target_channel or self.should_measure_auxiliary(c,position,math.dist(center, position),turn):
                active.add(c)
        for c in sorted(unknown | active, reverse=self.client.state.current_channel > 10):
            self.guard._measure(position, c)
        if full:
            s.full_scan_stations.append(tuple(position))
            s.completed_coverage_points.append(tuple(position))
        if len(s.detected_channels) > s.source_count_upper:
            raise _StopRun('model_conflict', 'source count exceeds problem upper bound')
        if len(s.detected_channels) == s.source_count_upper:
            for c in s.unknown_channels:
                s.channels[c].status = 'excluded_by_source_count'
            s.coverage_basis = 'source_count_upper_bound'
        elif s.unknown_channels and covers_mixed(s.full_scan_stations):
            for c in s.unknown_channels:
                negatives = {o.position for o in s.channels[c].observations if o.measure_result == 'no_signal'}
                if not set(s.full_scan_stations).issubset(negatives):
                    raise _StopRun('incomplete_coverage', 'missing negative channel receipts')
                s.channels[c].status = 'excluded_after_full_cover'
            s.completed_full_cover = True
            s.coverage_basis = ('analytic_600m_grid' if set(fallback_grid()).issubset(s.full_scan_stations)
                                else 'local_convex_hull_box_check_float_guard')

    def should_scan(self, position, force_scan=False):
        return (force_scan or not self.state.full_scan_stations or
                min(math.dist(position,p) for p in self.state.full_scan_stations) >= self.scan_spacing_m)

    def auxiliary_measurement_worthwhile(self, distance, turn):
        return distance < 150 or (distance < 1400 and turn >= 12)

    def should_measure_auxiliary(self, channel, position, distance, turn):
        return self.auxiliary_measurement_worthwhile(distance,turn)

    def initial_baseline_position(self):
        return (self.initial_baseline_m, 0.)

    def target(self, c):
        history = self.state.channels[c].observations
        e = self.evaluation(c)
        if e.decision is ChannelLocalizationDecision.READY_TO_CLEAR:
            return e.clear_position
        center = self.estimate(c)
        directions = [o for o in history if o.measure_result == 'direction']
        if history[-1].measure_result == 'no_signal':
            # Lost radio visibility does not invalidate the position outer bound.
            # Approach on the side of a known positive observation.
            last = directions[-1]
            dx, dy = last.position[0]-center[0], last.position[1]-center[1]
            norm = max(math.hypot(dx, dy), 1)
            offset = max(30., min(100., e.assessment.r_max_m*.25))
            side = -1 if self.visits[c] % 2 else 1
            return (center[0]+offset*(dx/norm+.5*side*dy/norm),
                    center[1]+offset*(dy/norm-.5*side*dx/norm))
        return super().target(c)

    def optical_fallback(self, c):
        self.state.fallback_channels.append(c)
        first = next(o for o in self.state.channels[c].observations if o.measure_result == 'direction')
        points = optical_points(first.position, first.svd_deg)
        # Cover the current positive-bearing outer bound first. Keep the entire
        # analytic 108-point rectangle behind it as a finite safety fallback.
        e = self.evaluation(c)
        a = math.radians(first.svd_deg)
        u, v = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
        region = e.assessment.outer_region
        xs = [p[0]*u[0]+p[1]*u[1] for p in region]
        ys = [p[0]*v[0]+p[1]*v[1] for p in region]
        left, right, bottom, top = min(xs)-.001, max(xs)+.001, min(ys)-.001, max(ys)+.001
        nx, ny = max(1, math.ceil((right-left)/28)), max(1, math.ceil((top-bottom)/28))
        local = []
        for i in range(nx):
            for j in range(ny):
                x, y = left+(i+.5)*(right-left)/nx, bottom+(j+.5)*(top-bottom)/ny
                local.append((x*u[0]+y*v[0], x*u[1]+y*v[1]))
        ordered = []
        current = self.client.state.position
        for pool in (local, points):
            while pool:
                p = min(pool, key=lambda q: math.dist(current, q))
                pool.remove(p)
                ordered.append(p)
                current = p
        for p in ordered:
            self.guard.check_budget(p, c, 'clear')
            result = self.client.clear(p, c)
            self.state.clear_attempts.append(ClearAttempt(c, p, result.result,
                self.client.state.virtual_time_s, 'optical_rectangle_probe', None))
            if result.result == 'success':
                self.state.channels[c].status = 'cleared'
                return
        raise _StopRun('model_conflict', f'channel {c}: optical rectangle exhausted')

    def run(self):
        s = self.state
        if s.termination_reason != 'not_started' or any(d.observations for d in s.channels.values()):
            raise ValueError('fresh state required')
        s.termination_reason = 'running'
        self.client.transport = self._transport
        try:
            self.batch((0., 0.), force_scan=True)
            if self.strategy == 'seven_grid':
                for p in self.anchors[1:]:
                    self.batch(p, force_scan=True)
            elif s.active_channels:
                if self.initial_baseline_m > 0:
                    self.batch(self.initial_baseline_position(),
                               force_scan=getattr(self,'baseline_full_scan',False))
            while not s.all_cleared:
                s.routing_steps += 1
                if s.routing_steps > 128 or self.strategy == 'seven_grid':
                    for p in fallback_grid():
                        if s.unknown_channels and p not in s.full_scan_stations:
                            self.batch(p, force_scan=True)
                    for c in sorted(s.active_channels):
                        self.resolve(c)
                    if not s.all_cleared:
                        raise _StopRun('incomplete', 'finite fallback did not finish')
                    break
                tasks = {c: self.estimate(c) for c in sorted(s.active_channels)}
                if s.unknown_channels:
                    stops = self.plan_coverage(tasks)
                    tasks.update({21+i: p for i, p in enumerate(stops)})
                if not tasks:
                    raise _StopRun('incomplete_coverage', 'no remaining mixed coverage route')
                k = self.choose_task(tasks)
                if k >= 21:
                    self.batch(tasks[k], force_scan=True)
                else:
                    self.resolve(k, one_step=True)
            s.termination_reason = 'all_cleared'
        except _StopRun as error:
            s.termination_reason, s.failure_detail = error.reason, str(error)
        except Exception as error:
            s.termination_reason, s.failure_detail = 'error', f'{type(error).__name__}: {error}'
            raise
        finally:
            s.planning_time_s = time.process_time()-self.guard.cpu_started
            # Keep the wrapper for /exit so management requests count as well.
        return s

    def plan_coverage(self, tasks):
        return coverage_route(self.state.full_scan_stations, tasks, self.client.state.position,
                              len(self.state.unknown_channels))

    def choose_task(self, tasks):
        return optimized_route(self.client.state.position, tasks)[0]

    def resolve(self, c, one_step=False):
        while c in self.state.active_channels:
            e = self.evaluation(c)
            if e.decision is ChannelLocalizationDecision.READY_TO_CLEAR:
                self.guard._clear(e.clear_position, c, 'q1_float_outer' if e.assessment else 'near',
                                  e.assessment.r_max_m if e.assessment else 5.)
                self.batch(self.client.state.position)
                return
            if self.visits[c] >= self.active_steps:
                self.optical_fallback(c)
                self.batch(self.client.state.position)
                return
            p = self.target(c)
            self.visits[c] += 1
            if any(math.dist(p, o.position) < 1e-6 for o in self.state.channels[c].observations):
                self.optical_fallback(c)
                return
            self.batch(p, target_channel=c)
            if one_step:
                return


def build_summary(state, virtual_time_s, runtime_s=0.):
    result = build_completion_summary(state, virtual_time_s, runtime_s)
    result.update(baseline_name=getattr(state,'strategy_name','q4-mixed-batched-routing-v1'),
                  controller_class=getattr(state,'controller_class',None),
                  control_config=getattr(state,'control_config',None),
                  privileged_truth=bool(getattr(state,'privileged_truth',False)),
                  refinement_config=getattr(state,'refinement_config',None),
                  optical_probe_count=sum(a.certificate=='opportunistic_optical_probe' for a in state.clear_attempts),
                  coverage_plan_kind='mixed_direction_distance_cover',
                  total_count_basis=state.coverage_basis if state.all_cleared else 'unknown',
                  full_scan_stations=state.full_scan_stations,
                  fallback_channels=state.fallback_channels, routing_steps=state.routing_steps,
                  actions_started=state.actions_started, network_requests=state.network_requests,
                  planning_time_s=state.planning_time_s,
                  excluded_by_source_count=[c for c, d in state.channels.items()
                                            if d.status == 'excluded_by_source_count'],
                  source_count_upper=16,
                  source_count_evidence_channels=sorted(state.detected_channels)
                      if state.coverage_basis == 'source_count_upper_bound' else [],
                  negative_coverage_receipts={str(c): [i for i,p in enumerate(state.full_scan_stations)
                      if any(o.position == p and o.measure_result == 'no_signal' for o in d.observations)]
                      for c,d in state.channels.items()},
                  coverage_receipts={str(c): [i for i,p in enumerate(state.full_scan_stations)
                      if any(o.position == p for o in d.observations)] for c,d in state.channels.items()},
                  geometry_scope='positive-bearing outer relaxation; float guards, not interval certification')
    return result
