"""离线特权参考：已知源坐标后直接清除；不是未知源算法或严格下界。"""
import math
import time
from dataclasses import dataclass, field
from src.q3.search_clear import _StopRun
from src.q4.controller import MixedController, MixedState
from src.q4.q3_adapter import optimized_route


@dataclass
class OracleState(MixedState):
    known_channels: set[int] = field(default_factory=set)

    @property
    def detected_channels(self):
        return set(self.known_channels)

    @property
    def excluded_channels(self):
        return {c for c, d in self.channels.items() if d.status == 'oracle_absent'}


class OracleController(MixedController):
    def __init__(self, client, *, known_positions, **kwargs):
        positions = {c: tuple(p) for c, p in known_positions.items()}
        if any(isinstance(c, bool) or not isinstance(c, int) or c not in range(1, 21)
               or len(p) != 2 or not all(math.isfinite(v) for v in p)
               or math.hypot(*p) > 1800+1e-6 for c, p in positions.items()):
            raise ValueError('invalid oracle positions')
        super().__init__(client, state=OracleState(known_channels=set(positions)), **kwargs)
        self.positions = positions
        self.state.strategy_name = 'q4-offline-privileged-oracle'
        self.state.privileged_truth = True
        self.state.coverage_basis = 'oracle_source_list_and_actual_clear_receipts'
        self.state.planned_coverage_points = []
        for c, d in self.state.channels.items():
            d.status = 'oracle_known' if c in positions else 'oracle_absent'

    def run(self):
        s = self.state
        if s.termination_reason != 'not_started':
            raise ValueError('fresh state required')
        s.termination_reason = 'running'
        try:
            for c in optimized_route(self.client.state.position, self.positions):
                self.guard._clear(self.positions[c], c, 'oracle_ground_truth', 0.)
            s.termination_reason = 'all_cleared'
        except _StopRun as error:
            s.termination_reason, s.failure_detail = error.reason, str(error)
        except Exception as error:
            s.termination_reason, s.failure_detail = 'error', f'{type(error).__name__}: {error}'
            raise
        finally:
            s.planning_time_s = time.process_time()-self.guard.cpu_started
        return s
