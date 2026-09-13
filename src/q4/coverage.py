"""Direction-aware planning and conservative continuous coverage checking.

Finite position/orientation samples rank actions only. Completion uses the
local-convex-hull theorem on boxes, with fail-closed floating-point guards,
or the analytically proved complete 600 m grid. Not an interval solver.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


def fallback_grid():
    return [(float(-2100+600*i), float(-2100+600*j))
            for j in range(8) for i in (range(8) if j % 2 == 0 else range(7, -1, -1))]


def covers_mixed(stations):
    points = tuple(sorted(set(tuple(map(float, p)) for p in stations)))
    if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in points):
        raise ValueError('finite 2D points required')
    if set(fallback_grid()).issubset(points):
        return True
    return _covers(points)


@lru_cache(maxsize=256)
def _covers(points):
    if len(points) < 3:
        return False
    stations = np.asarray(points)
    # Vectorized box witnesses avoid repeated convex-hull construction.
    centers = np.array([(x, y) for x in range(-1700, 1800, 200)
                        for y in range(-1700, 1800, 200)], dtype=float)
    h = 100.
    for _ in range(9):
        centers = centers[np.linalg.norm(np.maximum(np.abs(centers)-h, 0), axis=1) <= 1800+1e-6]
        if not len(centers):
            return True
        delta = stations[None, :, :]-centers[:, None, :]
        distance = np.linalg.norm(delta, axis=2)
        # Source enclosed by nearby stations iff their largest angular gap < pi.
        if np.any((np.linalg.norm(centers, axis=1) <= 1800) &
                  ~_enclosed(delta, distance <= 1000+1e-6, guard=-1e-9)):
            return False
        nearby = np.linalg.norm(np.abs(delta)+h, axis=2) < 1000-1e-3
        covered = np.ones(len(centers), dtype=bool)
        for dx, dy in ((-1,-1), (-1,1), (1,-1), (1,1)):
            covered &= _enclosed(delta-np.array([dx*h, dy*h]), nearby, guard=1e-8)
        centers = centers[~covered]
        if not len(centers):
            return True
        h /= 2
        centers = (centers[:, None, :]+np.array([[-h,-h],[-h,h],[h,-h],[h,h]])).reshape(-1, 2)
        if len(centers) > 12000:
            return False  # Unresolved boxes remain unknown on compute cap.
    return False


def _enclosed(delta, mask, guard):
    angles = np.where(mask, np.arctan2(delta[:, :, 1], delta[:, :, 0]), np.inf)
    angles.sort(axis=1)
    counts = mask.sum(axis=1)
    valid_pair = np.arange(angles.shape[1]-1)[None, :] < counts[:, None]-1
    with np.errstate(invalid='ignore'):
        gap = np.where(valid_pair, np.diff(angles, axis=1), 0).max(axis=1)
        last = angles[np.arange(len(angles)), np.maximum(counts-1, 0)]
        wrap = angles[:, 0]+2*math.pi-last
    return (counts >= 3) & (np.maximum(gap, wrap) < math.pi-guard)


@lru_cache(maxsize=1)
def planning_states():
    points = [(x, y) for x in range(-1800, 1801, 200) for y in range(-1800, 1801, 200)
              if x*x+y*y <= 1800**2]
    points += [(1800*math.cos(i*math.pi/48), 1800*math.sin(i*math.pi/48)) for i in range(96)]
    points = np.asarray(points)
    directions = np.array([(math.cos(i*math.pi/12), math.sin(i*math.pi/12)) for i in range(24)])
    return points, directions


def state_mask(stations):
    points, directions = planning_states()
    covered = np.zeros((len(points), len(directions)), dtype=bool)
    for p in stations:
        delta = np.asarray(p)-points
        covered |= (np.linalg.norm(delta, axis=1)[:, None] <= 975) & (delta @ directions.T >= 15)
    return covered


@lru_cache(maxsize=1)
def candidate_masks():
    candidates = [(r*math.cos(i*math.pi/12), r*math.sin(i*math.pi/12))
                  for r in (700, 1150, 1550, 2050) for i in range(24)]
    candidates += fallback_grid()
    masks = np.array([state_mask([p]).ravel() for p in candidates])
    return candidates, masks


def coverage_route(stations, targets, current, unknown_count):
    """Greedy residual position/direction cover, inserted into the clear route."""
    supplied = list(stations)+list(targets.values())
    if covers_mixed(supplied):
        return []
    # A staggered double ring leaves far less travel than repeated sample-greedy
    # replanning. Its coverage is checked continuously, never assumed from count.
    chosen = [(r*math.cos((i+phase)*2*math.pi/n), r*math.sin((i+phase)*2*math.pi/n))
              for n, r, phase in ((10, 975., 0.), (12, 1865., .5)) for i in range(n)]
    chosen = [p for p in chosen if p not in supplied]
    if not covers_mixed(supplied+chosen):
        chosen += [p for p in fallback_grid() if p not in supplied+chosen]
    for p in sorted(chosen, key=lambda p: min(math.dist(p, q) for q in supplied)):
        rest = list(chosen)
        rest.remove(p)
        if covers_mixed(supplied+rest):
            chosen = rest
    return chosen


def sampled_coverage_route(stations, targets, current, unknown_count):
    """Development comparator: greedy sampled state coverage with verification."""
    from src.q4.q3_adapter import optimized_route
    supplied = list(stations)+list(targets.values())
    if covers_mixed(supplied):
        return []
    covered = state_mask(supplied).ravel()
    candidates, masks = candidate_masks()
    route = [current]+[targets[k] for k in optimized_route(current, targets)]
    chosen = []
    for _ in range(45):
        gain = (masks & ~covered).sum(axis=1)
        if not gain.any():
            break
        scores = []
        for i, p in enumerate(candidates):
            options = [(math.dist(route[-1], p), len(route))]
            options += [(math.dist(a, p)+math.dist(p, b)-math.dist(a, b), j+1)
                        for j, (a, b) in enumerate(zip(route, route[1:]))]
            extra, index = min(options)
            scores.append((gain[i]/(extra+30*unknown_count), -i, index))
        _, ni, index = max(scores)
        i = -ni
        chosen.append(candidates[i])
        route.insert(index, candidates[i])
        covered |= masks[i]
    if not covers_mixed(supplied+chosen):
        chosen += [p for p in fallback_grid() if p not in supplied+chosen]
    # Remove superfluous stops, always checking continuous coverage.
    for p in list(reversed(chosen)):
        rest = list(chosen)
        rest.remove(p)
        if covers_mixed(supplied+rest):
            chosen = rest
    return chosen
