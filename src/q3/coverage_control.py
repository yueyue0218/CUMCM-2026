"""Conservative coverage of the arena by guaranteed reception disks."""
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np


def covers_arena(stations, *, arena_radius=1800.0, receive_radius=1000.0):
    points = tuple(sorted(set(tuple(map(float, p)) for p in stations)))
    if not math.isfinite(arena_radius) or not math.isfinite(receive_radius) or min(arena_radius, receive_radius) <= 0:
        raise ValueError('positive finite radii required')
    if any(len(p)!=2 or not all(math.isfinite(x) for x in p) for p in points):
        raise ValueError('finite 2D stations required')
    return _covers(points, arena_radius, receive_radius)


@lru_cache(maxsize=512)
def _covers(points, arena_radius, receive_radius):
    if not points:
        return False
    # Each box is either outside the disk, or covered in its entirety by
    # one reception disk using the triangle inequality. Unresolved tiny boxes
    # fail closed. This is a floating-point certificate with a 1 mm guard.
    stack = [(0.0, 0.0, arena_radius, 0)]
    while stack:
        x, y, half, depth = stack.pop()
        if math.hypot(max(abs(x)-half, 0), max(abs(y)-half, 0)) > arena_radius+1e-3:
            continue
        nearest = min(math.hypot(x-a, y-b) for a,b in points)
        if nearest + math.sqrt(2)*half <= receive_radius-1e-3:
            continue
        if math.hypot(x,y) <= arena_radius and nearest > receive_radius:
            return False
        if depth >= 12:
            return False
        h = half/2
        stack.extend((x+dx*h,y+dy*h,h,depth+1) for dx in (-1,1) for dy in (-1,1))
    return True


def coverage_waypoints(stations, targets, current):
    """Heuristic insertion of sensing stops; only covers_arena certifies exit."""
    from src.q3.world_model import open_clear_route
    grid = [(x,y) for x in range(-1800,1801,150) for y in range(-1800,1801,150) if x*x+y*y<=1800**2]
    grid += [(1800*math.cos(i*math.pi/90),1800*math.sin(i*math.pi/90)) for i in range(180)]
    samples = np.asarray(grid)
    candidates = [(r*math.cos(i*math.pi/12),r*math.sin(i*math.pi/12)) for r in (1150,1350,1550) for i in range(24)]
    candidates += [(0.,0.)]
    future = list(targets.values())
    supplied = list(stations)+future
    if supplied and covers_arena(supplied):
        return []
    covered = np.zeros(len(samples),dtype=bool)
    for p in supplied:
        covered |= np.linalg.norm(samples-p,axis=1)<=985
    masks = np.linalg.norm(samples[None,:,:]-np.asarray(candidates)[:,None,:],axis=2)<=985
    route = [current]+[targets[k] for k in open_clear_route(current,targets)]
    selected=[]
    for _ in range(12):
        gains = (masks & ~covered).sum(axis=1)
        if not gains.any():
            break
        choices=[]
        for i,p in enumerate(candidates):
            options=[(math.dist(route[-1],p),len(route))]
            options += [(math.dist(a,p)+math.dist(p,b)-math.dist(a,b),j+1) for j,(a,b) in enumerate(zip(route,route[1:]))]
            extra,index=min(options)
            choices.append((gains[i]/(extra+150),-extra,-i,index))
        _,_,negative,index=max(choices)
        i=-negative
        selected.append(candidates[i]);route.insert(index,candidates[i]);covered|=masks[i]
        if covers_arena(supplied+selected):
            break
    if not covers_arena(supplied+selected):
        # Fixed finite fallback layout handles unresolved sampled boundary gaps.
        selected += [(0.,0.)]+[(1150*math.cos(i*math.pi/3),1150*math.sin(i*math.pi/3)) for i in range(6)]
    for p in list(reversed(selected)):
        rest=list(selected);rest.remove(p)
        if covers_arena(supplied+rest):
            selected=rest
    return selected
