"""对照组：上帝视角。仅离线评估可传入真实源坐标，不能部署为未知源策略。"""
import math
from dataclasses import dataclass, field
from src.q3.experiment_joint_routing import optimized_route
from src.q3.search_clear import SearchClearState, SearchClearController, _StopRun, build_completion_summary


@dataclass
class OracleState(SearchClearState):
    known_positions: dict = field(default_factory=dict)

    @property
    def detected_channels(self):
        return set(self.known_positions)

    @property
    def excluded_channels(self):
        return {c for c, d in self.channels.items() if d.status == 'oracle_absent'}


def run_oracle(client, *, positions, state=None, virtual_limit_s=360000., exit_reserve_s=20.):
    s = state if state is not None else OracleState()
    if not isinstance(s, OracleState) or s.termination_reason != 'not_started':
        raise ValueError('fresh OracleState required')
    if any(isinstance(c, bool) or not isinstance(c, int) or not 1 <= c <= 20 or len(p) != 2
           or not all(math.isfinite(x) for x in p) or math.hypot(*p) > 1800.000001
           for c,p in positions.items()):
        raise ValueError('invalid oracle source positions')
    s.known_positions = dict(positions)
    for c,d in s.channels.items():
        d.status = 'oracle_known' if c in positions else 'oracle_absent'
    guard = SearchClearController(client, s, virtual_limit_s=virtual_limit_s, exit_reserve_s=exit_reserve_s)
    s.termination_reason = 'running'
    try:
        for c in optimized_route(client.state.position, positions):
            guard._clear(positions[c], c, 'oracle_ground_truth', 0.)
        s.termination_reason = 'all_cleared'
    except _StopRun as error:
        s.termination_reason = error.reason; s.failure_detail = str(error)
    except Exception as error:
        s.termination_reason = 'error'; s.failure_detail = f'{type(error).__name__}: {error}'
        raise
    return s


def build_oracle_summary(state, virtual_time_s, runtime_s=0.):
    result = build_completion_summary(state, virtual_time_s, runtime_s)
    result.update(baseline_name='对照组_上帝视角',
                  total_count_basis='oracle_known_source_list_and_clear_receipts',
                  total_count=len(state.known_positions),
                  clearance_ratio=len(state.cleared_channels)/len(state.known_positions) if state.known_positions else None,
                  privileged_truth=True, route_solver='multistart_nearest_neighbor_2opt_not_proven_optimal')
    return result
