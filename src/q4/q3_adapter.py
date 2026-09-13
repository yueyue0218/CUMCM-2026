"""Q3 reuse adapter frozen for the validated Q4 v2 policy.

Only Q3's routing, state fields and four localization helpers are retained.
The experimental Q3 controller continues evolving independently. Provenance:
source_snapshot.zip from runs/q4/validation/refined_fresh_70.
Parent file SHA-256: b6b29a4ffdaa2a007d7d3a900fbddf36bbdbd1d0787f74c595b49cbd40db5943
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from src.common.geometry import Point
from src.q3.search_clear import SearchClearState, SearchClearController
from src.q3.localization_control import evaluate_channel, ChannelLocalizationDecision



def open_clear_route(start: Point, positions: dict[int, Point]) -> list[int]:
    """Shortest open fixed-point path for <=10 targets, bounded 2-opt otherwise.

    No home/return leg is included. Points must already be certified by the
    caller; this function supplies an order, not a position certificate.
    """
    channels = sorted(positions)
    if len(start) != 2 or not all(math.isfinite(v) for v in start):
        raise ValueError("start must contain two finite coordinates")
    if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in positions.values()):
        raise ValueError("route positions must contain two finite coordinates")
    n = len(channels)
    if n == 0:
        return []
    points = [positions[c] for c in channels]
    distance = [[math.dist(a, b) for b in points] for a in points]
    if n <= 10:
        # Held-Karp state stores the full path for deterministic tie resolution.
        dp = {(1 << j, j): (math.dist(start, points[j]), (j,)) for j in range(n)}
        for mask in range(1, 1 << n):
            for last in range(n):
                state = dp.get((mask, last))
                if state is None:
                    continue
                cost, path = state
                for nxt in range(n):
                    if mask & (1 << nxt):
                        continue
                    key = (mask | (1 << nxt), nxt)
                    candidate = cost + distance[last][nxt], path + (nxt,)
                    if key not in dp or candidate < dp[key]:
                        dp[key] = candidate
        _, route = min(dp[((1 << n) - 1, last)] for last in range(n))
        return [channels[j] for j in route]
    remaining, route, current = set(range(n)), [], start
    while remaining:
        nxt = min(remaining, key=lambda j: (math.dist(current, points[j]), channels[j]))
        route.append(nxt)
        remaining.remove(nxt)
        current = points[nxt]
    # Fixed pass cap keeps ranking cost bounded. Reversing a suffix changes
    # only its entering edge because this is an open, symmetric path.
    for _ in range(4):
        improved = False
        for i in range(n - 1):
            before = start if i == 0 else points[route[i - 1]]
            for j in range(i + 1, n):
                old = math.dist(before, points[route[i]])
                new = math.dist(before, points[route[j]])
                if j + 1 < n:
                    old += distance[route[j]][route[j + 1]]
                    new += distance[route[i]][route[j + 1]]
                if new < old - 1e-10:
                    route[i:j + 1] = reversed(route[i:j + 1])
                    improved = True
        if not improved:
            break
    return [channels[j] for j in route]

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
        if isinstance(self.state.source_count_upper, bool) or not isinstance(self.state.source_count_upper, int) or (not 1 <= self.state.source_count_upper <= 20):
            raise ValueError('source count upper bound must be an integer in 1..20')
        if not math.isfinite(scan_spacing_m) or scan_spacing_m <= 0 or (not math.isfinite(initial_baseline_m)) or (not 0 <= initial_baseline_m <= 1500):
            raise ValueError('invalid sensing configuration')
        self.guard = SearchClearController(client, self.state, virtual_limit_s=virtual_limit_s, exit_reserve_s=exit_reserve_s)
        self.cache = {}
        self.scan_spacing_m = scan_spacing_m
        self.initial_baseline_m = initial_baseline_m
        self.visits = {c: 0 for c in range(1, 21)}
        self.anchors = compact_coverage_points()
        self.visited_anchors = set()
        self.state.planned_coverage_points = list(self.anchors)

    def evaluation(self, c):
        discovery = self.state.channels[c]
        key = (c, len(discovery.observations))
        if key not in self.cache:
            self.cache[key] = evaluate_channel(discovery)
        result = self.cache[key]
        if result.decision is ChannelLocalizationDecision.MODEL_CONFLICT:
            self.guard._conflict(c, 'inconsistent Q1 evidence')
        return result

    def estimate(self, c):
        e = self.evaluation(c)
        if e.decision is ChannelLocalizationDecision.READY_TO_CLEAR:
            return e.clear_position
        return e.assessment.clear_position

    def target(self, c):
        e = self.evaluation(c)
        if e.decision is ChannelLocalizationDecision.READY_TO_CLEAR:
            return e.clear_position
        directions = [o for o in self.state.channels[c].observations if o.measure_result == 'direction']
        if len(directions) == 1:
            o = directions[0]
            angle = math.radians(o.svd_deg)
            center = self.estimate(c)
            return (center[0] - 100 * math.sin(angle), center[1] + 100 * math.cos(angle))
        center = self.estimate(c)
        if math.dist(center, self.client.state.position) <= 150 and (not any((math.dist(self.client.state.position, o.position) < 1e-06 for o in self.state.channels[c].observations))):
            return self.client.state.position
        if math.dist(center, self.client.state.position) < 50:
            direction = next((o for o in reversed(self.state.channels[c].observations) if o.measure_result == 'direction'))
            angle = math.radians(direction.svd_deg + 90)
            step = min(250.0, max(40.0, e.assessment.r_max_m / 2))
            return (center[0] + step * math.cos(angle), center[1] + step * math.sin(angle))
        return center


def cost_breakdown(actions):
    position=(0.,0.)
    channel=1
    out={'movement_s':0.,'measurement_s':0.,'switching_s':0.,'clearance_s':0.}
    for action in actions:
        if action['path'] not in {'/measure','/clear'}:
            continue
        request=action['request'];response=action['response']
        target=(request['position']['x'],request['position']['y'])
        out['movement_s']+=math.dist(position,target)/5
        position=target
        if action['path']=='/measure':
            out['measurement_s']+=5
            out['switching_s']+=request['channel']!=channel
            channel=request['channel']
        else:
            out['clearance_s']+=5 if response['clear_result']=='success' else 3
    return out
