"""对照组：500 米步长随机游走；128 步用尽仍未获证则如实退出。"""
import math
import random
from src.q3.control_scan_common import run_scan_control


def random_walk_points(seed=0, max_steps=128):
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 0 <= max_steps <= 10000:
        raise ValueError('max_steps must be an integer in 0..10000')
    rng = random.Random(seed)
    points = [(0., 0.)]
    for _ in range(max_steps):
        # Rejection samples the feasible directions at the disk boundary.
        for _ in range(10000):
            angle = rng.uniform(0., 2*math.pi)
            p = (points[-1][0]+500*math.cos(angle), points[-1][1]+500*math.sin(angle))
            if math.hypot(*p) <= 1800.:
                points.append(p)
                break
        else:
            raise ArithmeticError('bounded random-direction sampler exhausted')
    return points


def run_random_walk(client, *, seed=0, max_steps=128, **kwargs):
    return run_scan_control(client, random_walk_points(seed, max_steps), clear_during_scan=True, **kwargs)
