"""对照组：螺旋扫描后补方向覆盖，每站清除已发现源。"""
from src.q3.control_spiral_scan import spiral_points as q3_spiral_points
from src.q4.control_scan_common import ScanControl, compact_points
from src.q4.coverage import covers_mixed


def spiral_points():
    # Preserve Q3's 1000 m pitch / 500 m arc spacing / 1800 m outer sweep.
    # Its omni-only layout cannot prove detection of outward directional sources.
    points = q3_spiral_points()
    if not covers_mixed(points):
        points.extend(p for p in compact_points() if p not in points)
    return points


class SpiralScanController(ScanControl):
    def __init__(self, client, **kwargs):
        super().__init__(client, **kwargs)
        self.state.strategy_name = 'q4-control-spiral-with-mixed-cover'

    def scan_points(self):
        return spiral_points()
