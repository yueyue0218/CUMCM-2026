import itertools
import math
import random
import unittest

from src.q4.q3_adapter import optimized_route
from src.q4.route_refinement import relocated_route


def distance(start, targets, route):
    path=[start]+[targets[c] for c in route]
    return sum(math.dist(a,b) for a,b in zip(path,path[1:]))


class RouteRefinementTests(unittest.TestCase):
    def test_small_open_route_matches_exhaustive_order(self):
        targets={1:(-1,4),2:(6,8),3:(2,-3),4:(10,1),5:(0,0),6:(-7,8)}
        start=(9,9)
        route=relocated_route(start,targets)
        self.assertAlmostEqual(distance(start,targets,route),
            min(distance(start,targets,p) for p in itertools.permutations(targets)))

    def test_relocation_keeps_every_stop_and_never_lengthens_route(self):
        for seed in range(8):
            rng=random.Random(seed)
            start=(rng.random()*1000,rng.random()*1000)
            targets={c:(rng.uniform(-2000,2000),rng.uniform(-2000,2000)) for c in range(1,29)}
            route=relocated_route(start,targets)
            self.assertEqual(sorted(route),sorted(targets))
            self.assertLessEqual(distance(start,targets,route),
                distance(start,targets,optimized_route(start,targets))+1e-7)

    def test_empty_and_duplicate_positions(self):
        self.assertEqual(relocated_route((0,0),{}),[])
        targets={c:(0,0) for c in range(1,15)}
        self.assertEqual(sorted(relocated_route((0,0),targets)),sorted(targets))


if __name__=='__main__':
    unittest.main()
