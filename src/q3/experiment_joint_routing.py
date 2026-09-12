"""实验组：集中测量与联合路线；清除仅依据 Q1 或有限后备认证。"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

from src.q3.search_clear import SearchClearState, SearchClearController, _StopRun
from src.q3.localization_control import evaluate_channel, ChannelLocalizationDecision
from src.q3.coverage_control import covers_arena, coverage_waypoints
from src.q3.world_model import open_clear_route


def compact_coverage_points():
    return [(0.0,0.0)]+[(1150*math.cos(i*math.pi/3),1150*math.sin(i*math.pi/3)) for i in range(6)]


def optimized_route(start, targets):
    """Deterministic multistart open routing, including changes of first stop."""
    keys=sorted(targets)
    if len(keys)<3:
        return open_clear_route(start,targets)
    points=[start]+[targets[k] for k in keys]
    n=len(keys)
    distances=[[math.dist(a,b) for b in points] for a in points]
    best=None
    for first in range(1,n+1):
        path=[0,first];remaining=set(range(1,n+1))-{first}
        while remaining:
            next_node=min(remaining,key=lambda k:(distances[path[-1]][k],k))
            path.append(next_node);remaining.remove(next_node)
        for _ in range(20):
            changed=False
            for i in range(1,n):
                for j in range(i+1,n+1):
                    old=distances[path[i-1]][path[i]]
                    new=distances[path[i-1]][path[j]]
                    if j<n:
                        old+=distances[path[j]][path[j+1]]
                        new+=distances[path[i]][path[j+1]]
                    if new<old-1e-8:
                        path[i:j+1]=reversed(path[i:j+1]);changed=True
            if not changed:break
        cost=sum(distances[a][b] for a,b in zip(path,path[1:]))
        item=(cost,tuple(path))
        if best is None or item<best:best=item
    return [keys[i-1] for i in best[1][1:]]


@dataclass
class EfficientState(SearchClearState):
    full_scan_stations: list[tuple[float,float]] = field(default_factory=list)
    routing_steps: int = 0
    fallback_channels: list[int] = field(default_factory=list)
    source_count_upper: int = 16

    @property
    def excluded_channels(self):
        return super().excluded_channels | {c for c,d in self.channels.items() if d.status=='excluded_by_source_count'}


class EfficientController:
    def __init__(self, client, state=None, *, virtual_limit_s=360000.0, exit_reserve_s=20.0, scan_spacing_m=1500.0, initial_baseline_m=250.0):
        self.client = client
        self.state = state if state is not None else EfficientState()
        if not isinstance(self.state, EfficientState):
            raise TypeError('EfficientState required')
        if (isinstance(self.state.source_count_upper, bool)
                or not isinstance(self.state.source_count_upper, int)
                or not 1<=self.state.source_count_upper<=20):
            raise ValueError('source count upper bound must be an integer in 1..20')
        if (not math.isfinite(scan_spacing_m) or scan_spacing_m<=0
                or not math.isfinite(initial_baseline_m) or not 0<=initial_baseline_m<=1500):
            raise ValueError('invalid sensing configuration')
        self.guard = SearchClearController(client,self.state,virtual_limit_s=virtual_limit_s,exit_reserve_s=exit_reserve_s)
        self.cache = {}
        self.scan_spacing_m = scan_spacing_m
        self.initial_baseline_m = initial_baseline_m
        self.visits = {c:0 for c in range(1,21)}
        self.anchors = compact_coverage_points()
        self.visited_anchors = set()
        self.state.planned_coverage_points = list(self.anchors)

    def evaluation(self,c):
        discovery = self.state.channels[c]
        key = (c,len(discovery.observations))
        if key not in self.cache:
            self.cache[key] = evaluate_channel(discovery)
        result = self.cache[key]
        if result.decision is ChannelLocalizationDecision.MODEL_CONFLICT:
            self.guard._conflict(c,'inconsistent Q1 evidence')
        return result

    def estimate(self,c):
        e=self.evaluation(c)
        if e.decision is ChannelLocalizationDecision.READY_TO_CLEAR:
            return e.clear_position
        return e.assessment.clear_position

    def route_position(self,c):
        return self.estimate(c)

    def plan_coverage(self,stations,targets,current):
        return coverage_waypoints(stations,targets,current)

    def auxiliary_measurement_worthwhile(self,distance,turn):
        return distance<1400 and turn>=12

    def batch(self, position, anchor=None, target_channel=None, force_scan=False):
        s = self.state
        full_scan = (force_scan or anchor is not None or not s.full_scan_stations
                     or min(math.dist(position,p) for p in s.full_scan_stations)>=self.scan_spacing_m)
        unknown = s.unknown_channels if full_scan else set()
        active=set()
        for c in s.active_channels:
            e=self.evaluation(c)
            if e.decision is ChannelLocalizationDecision.READY_TO_CLEAR:
                continue
            history=s.channels[c].observations
            if any(math.dist(position,o.position)<1e-6 for o in history):
                continue
            center=self.estimate(c)
            last=next(o for o in reversed(history) if o.measure_result=='direction')
            angle=math.degrees(math.atan2(center[1]-position[1],center[0]-position[0]))
            turn=abs((angle-last.svd_deg+180)%360-180)
            distance=math.dist(position,center)
            if c==target_channel or distance<=150 or self.auxiliary_measurement_worthwhile(distance,turn):
                active.add(c)
        # Do not leave the station in response to the first discovered source.
        order = sorted(unknown|active,reverse=self.client.state.current_channel>10)
        for c in order:
            self.guard._measure(position,c)
            if c in unknown and anchor is not None:
                s.coverage_receipts[c].add(anchor)
        if full_scan:
            s.full_scan_stations.append(position)
        if anchor is not None:
            self.visited_anchors.add(anchor)
            s.completed_coverage_points.append(position)
        if len(s.detected_channels)>s.source_count_upper:
            raise _StopRun('model_conflict','observed source count exceeds configured problem upper bound')
        if len(s.detected_channels)==s.source_count_upper:
            for c in s.unknown_channels:
                s.channels[c].status='excluded_by_source_count'
        if s.unknown_channels and covers_arena(s.full_scan_stations):
            # Every still-unknown channel was included in every full batch.
            for c in s.unknown_channels:
                observed = {o.position for o in s.channels[c].observations if o.measure_result=='no_signal'}
                if not set(s.full_scan_stations).issubset(observed):
                    raise _StopRun('incomplete_coverage','missing per-channel full-batch receipt')
                s.channels[c].status = 'excluded_after_full_cover'
            s.completed_full_cover = True

    def target(self,c):
        e = self.evaluation(c)
        if e.decision is ChannelLocalizationDecision.READY_TO_CLEAR:
            return e.clear_position
        directions = [o for o in self.state.channels[c].observations if o.measure_result=='direction']
        if len(directions)==1:
            o=directions[0]
            angle=math.radians(o.svd_deg)
            center=self.estimate(c)
            return (center[0]-100*math.sin(angle),center[1]+100*math.cos(angle))
        center = self.estimate(c)
        if math.dist(center,self.client.state.position)<=150 and not any(
                math.dist(self.client.state.position,o.position)<1e-6 for o in self.state.channels[c].observations):
            return self.client.state.position
        if math.dist(center,self.client.state.position)<50:
            direction = next(o for o in reversed(self.state.channels[c].observations) if o.measure_result=='direction')
            angle = math.radians(direction.svd_deg+90)
            step = min(250.0,max(40.0,e.assessment.r_max_m/2))
            return (center[0]+step*math.cos(angle),center[1]+step*math.sin(angle))
        return center

    def run(self):
        s = self.state
        if s.termination_reason!='not_started' or any(d.observations for d in s.channels.values()):
            raise ValueError('fresh state required')
        s.termination_reason='running'
        try:
            self.batch((0.0,0.0),0)
            if s.active_channels:
                self.batch((self.initial_baseline_m,0.0),force_scan=self.initial_baseline_m>=500)
            while not s.all_cleared:
                s.routing_steps+=1
                if s.routing_steps>256:
                    # Finite global escape: finish the fixed covering layout,
                    # then resolve each remaining observed source analytically.
                    for i,p in enumerate(self.anchors):
                        if s.unknown_channels:
                            self.batch(p,i)
                    for c in sorted(s.active_channels):
                        e=self.evaluation(c)
                        if e.decision is ChannelLocalizationDecision.READY_TO_CLEAR:
                            self.guard._clear(e.clear_position,c,'q1' if e.assessment else 'near',
                                              e.assessment.r_max_m if e.assessment else 5.0)
                        else:
                            station=next(o.position for o in reversed(s.channels[c].observations) if o.measure_result=='direction')
                            self.guard._measure(station,c)
                            s.fallback_channels.append(c)
                            self.guard._localize_and_clear(c)
                    if not s.all_cleared:
                        raise _StopRun('incomplete','finite route fallback did not complete')
                    break
                tasks = {}
                for c in sorted(s.active_channels):
                    tasks[c] = self.route_position(c)
                if s.unknown_channels:
                    stops = self.plan_coverage(s.full_scan_stations,tasks,self.client.state.position)
                    tasks.update({21+i:p for i,p in enumerate(stops)})
                if not tasks:
                    raise _StopRun('incomplete_coverage','no remaining certified cover route')
                target_id = optimized_route(self.client.state.position,tasks)[0]
                position = tasks[target_id]
                if target_id>=21:
                    self.batch(position,force_scan=True)
                    continue
                c=target_id
                position=self.target(c)
                self.visits[c]+=1
                e=self.evaluation(c)
                if e.decision is ChannelLocalizationDecision.READY_TO_CLEAR:
                    self.guard._clear(position,c,'q1' if e.assessment else 'near',
                                      e.assessment.r_max_m if e.assessment else 5.0)
                    self.batch(position)
                elif self.visits[c]>=8:
                    s.fallback_channels.append(c)
                    station=next(o.position for o in reversed(s.channels[c].observations) if o.measure_result=='direction')
                    self.guard._measure(station,c)
                    self.guard._localize_and_clear(c)
                    self.batch(self.client.state.position)
                else:
                    self.batch(position,target_channel=c)
            s.termination_reason='all_cleared'
        except _StopRun as error:
            s.termination_reason=error.reason
            s.failure_detail=str(error)
        except Exception as error:
            s.termination_reason='error'
            s.failure_detail=f'{type(error).__name__}: {error}'
            raise
        return s


def run_efficient(client, *, state=None, virtual_limit_s=360000.0,exit_reserve_s=20.0):
    return EfficientController(client,state,virtual_limit_s=virtual_limit_s,exit_reserve_s=exit_reserve_s).run()


def build_efficient_summary(state, virtual_time_s, runtime_s=0.0):
    from src.q3.search_clear import build_completion_summary
    result=build_completion_summary(state,virtual_time_s,runtime_s)
    stations=state.full_scan_stations
    receipts={str(c):[i for i,p in enumerate(stations) if any(o.position==p for o in d.observations)]
              for c,d in state.channels.items()}
    negatives={str(c):[i for i,p in enumerate(stations) if any(o.position==p and o.measure_result=='no_signal' for o in d.observations)]
               for c,d in state.channels.items()}
    count_excluded=[c for c,d in state.channels.items() if d.status=='excluded_by_source_count']
    basis=('source_count_upper_bound_and_clear_receipts' if count_excluded else
           'dynamic_reception_disk_coverage_and_clear_receipts') if state.all_cleared else 'unknown'
    result.update({'baseline_name':'q3-efficient-batched-routing-v1','total_count_basis':basis,
                   'coverage_plan_kind':'dynamic_certified_disk_union_with_fixed_layout_fallback',
                   'full_scan_stations':stations,'completed_coverage_points':stations,
                   'coverage_receipts':receipts,'negative_coverage_receipts':negatives,
                   'source_count_upper':state.source_count_upper,
                   'source_count_evidence_channels':sorted(state.detected_channels) if count_excluded else [],
                   'excluded_by_source_count':sorted(count_excluded),
                   'routing_steps':state.routing_steps,'fallback_channels':state.fallback_channels})
    return result
