"""对照组：盘内 500 米随机游走，最多 128 步，无覆盖补救。"""
import math
import random
from src.q4.control_scan_common import ScanControl


def random_walk_points(seed=410000003, steps=128):
    if isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= 128:
        raise ValueError('steps must be an integer in 0..128')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('walk seed must be an integer')
    rng = random.Random(seed)
    points = [(0., 0.)]
    for _ in range(steps):
        for _attempt in range(10000):
            angle = rng.uniform(0, 2*math.pi)
            p = (points[-1][0]+500*math.cos(angle), points[-1][1]+500*math.sin(angle))
            if math.hypot(*p) <= 1800:
                points.append(p)
                break
        else:
            raise RuntimeError('bounded random direction sampling exhausted')
    return points


class RandomWalkController(ScanControl):
    def __init__(self, client, *, walk_seed=410000003, walk_steps=128, **kwargs):
        self.points = random_walk_points(walk_seed, walk_steps)
        super().__init__(client, **kwargs)
        self.state.strategy_name = 'q4-control-random-walk'
        self.state.control_config = dict(walk_seed=walk_seed, walk_steps=walk_steps)

    def scan_points(self):
        return list(self.points)
