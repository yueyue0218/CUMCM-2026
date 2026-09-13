"""Q3-style scan controls sharing the final Q4 localization and budget guards."""
import math
import time
from src.q3.search_clear import _StopRun
from src.q4.coverage import covers_mixed
from src.q4.informed import InformedController
from src.q4.q3_adapter import optimized_route


def compact_points():
    """Public 21-station layout, including stations outside the source disk."""
    points = [(0., 0.)] + [(r*math.cos(2*math.pi*i/n), r*math.sin(2*math.pi*i/n))
        for n, r in ((8, 998.), (12, 1800/math.cos(math.pi/12)+3)) for i in range(n)]
    if not covers_mixed(points):
        raise ValueError('compact layout failed conservative mixed coverage check')
    return points


class ScanControl(InformedController):
    clear_during_scan = True
    scan_all_channels = False

    def should_scan(self, position, force_scan=False):
        # Localization detours must not introduce unplanned discovery stations.
        return bool(force_scan)

    def scan_points(self):
        raise NotImplementedError

    def resolve_detected(self):
        while self.state.active_channels:
            tasks = {c: self.estimate(c) for c in sorted(self.state.active_channels)}
            self.resolve(optimized_route(self.client.state.position, tasks)[0])

    def run(self):
        s = self.state
        if s.termination_reason != 'not_started' or any(d.observations for d in s.channels.values()):
            raise ValueError('fresh state required')
        s.termination_reason = 'running'
        try:
            points = self.scan_points()
            s.planned_coverage_points = points
            for point in points:
                if s.all_cleared:
                    break
                if self.scan_all_channels:
                    # Census deliberately measures all 20 at every station,
                    # including previously detected channels, before any clear.
                    for c in sorted(s.channels, reverse=self.client.state.current_channel > 10):
                        self.guard._measure(point, c)
                    s.full_scan_stations.append(tuple(point))
                    s.completed_coverage_points.append(tuple(point))
                    # All current-point readings exist: this adds no requests,
                    # and checks the shared receipt-based completion evidence.
                    self.batch(point)
                else:
                    self.batch(point, force_scan=True)
                if self.clear_during_scan:
                    self.resolve_detected()
            if not self.clear_during_scan:
                self.resolve_detected()
            if not s.all_cleared:
                raise _StopRun('scan_limit', 'fixed scan schedule exhausted without complete evidence')
            s.termination_reason = 'all_cleared'
        except _StopRun as error:
            s.termination_reason, s.failure_detail = error.reason, str(error)
        except Exception as error:
            s.termination_reason, s.failure_detail = 'error', f'{type(error).__name__}: {error}'
            raise
        finally:
            s.planning_time_s = time.process_time()-self.guard.cpu_started
        return s
