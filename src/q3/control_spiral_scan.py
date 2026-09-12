"""对照组：阿基米德螺旋扫描，发现后用共用 Q1 定位器清除。"""
import math
from src.q3.control_scan_common import run_scan_control


def spiral_points():
    pitch = 1000.
    b = pitch / (2 * math.pi)
    theta_end = 1800. / b
    theta = 0.
    points = [(0., 0.)]
    # Arc-length integral of r=b*theta; invert it by bounded bisection.
    def arc(t):
        return .5 * b * (t * math.sqrt(1+t*t) + math.asinh(t))
    distance = 500.
    while distance < arc(theta_end):
        lo, hi = theta, theta_end
        for _ in range(48):
            mid = (lo+hi)/2
            if arc(mid) < distance: lo = mid
            else: hi = mid
        theta = (lo+hi)/2
        points.append((b*theta*math.cos(theta), b*theta*math.sin(theta)))
        distance += 500.
    # Complete the boundary sweep, avoiding an uncovered outer-sector wedge.
    for i in range(24):
        angle = theta_end + i * 2*math.pi/23
        points.append((1800*math.cos(angle), 1800*math.sin(angle)))
    return points


def run_spiral_scan(client, **kwargs):
    return run_scan_control(client, spiral_points(), clear_during_scan=True, **kwargs)
