"""对照组：21 站全频道普查完成后，再定位清除（Q4 方向覆盖）。"""
from src.q4.control_scan_common import ScanControl, compact_points
from src.q4.q3_adapter import optimized_route


class CensusThenClearController(ScanControl):
    clear_during_scan = False
    scan_all_channels = True

    def __init__(self, client, **kwargs):
        super().__init__(client, **kwargs)
        self.state.strategy_name = 'q4-control-census-then-clear'

    def scan_points(self):
        points = compact_points()
        tasks = dict(enumerate(points[1:]))
        return [points[0]] + [tasks[k] for k in optimized_route(points[0], tasks)]
