"""Bearing wedges and two-dimensional half-plane intersections."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from scipy.optimize import linprog

Point = tuple[float, float]


@dataclass(frozen=True)
class Observation:
    station: Point
    bearing_deg: float
    half_angle_deg: float = 1.0

    def __post_init__(self) -> None:
        values = (*self.station, self.bearing_deg, self.half_angle_deg)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("observation values must be finite")
        if not 0.0 < self.half_angle_deg < 90.0:
            raise ValueError("half_angle_deg must be in (0, 90)")


@dataclass(frozen=True)
class HalfPlane:
    a: float
    b: float
    c: float


@dataclass(frozen=True)
class Region:
    status: str
    vertices: tuple[Point, ...]
    max_violation: float


def bearing_halfplanes(observation: Observation) -> tuple[HalfPlane, HalfPlane]:
    sx, sy = observation.station
    theta = math.radians(observation.bearing_deg % 360.0)
    delta = math.radians(observation.half_angle_deg)
    lower = (math.cos(theta - delta), math.sin(theta - delta))
    upper = (math.cos(theta + delta), math.sin(theta + delta))
    return (
        HalfPlane(lower[1], -lower[0], lower[1] * sx - lower[0] * sy),
        HalfPlane(-upper[1], upper[0], -upper[1] * sx + upper[0] * sy),
    )


def point_satisfies(
    point: Point,
    halfplanes: Iterable[HalfPlane],
    tolerance: float = 1e-9,
) -> bool:
    x, y = point
    return all(plane.a * x + plane.b * y <= plane.c + tolerance
               for plane in halfplanes)


def _clip_polygon(
    vertices: list[Point], plane: HalfPlane, tolerance: float
) -> list[Point]:
    if not vertices:
        return []

    clipped: list[Point] = []
    previous = vertices[-1]
    previous_residual = plane.a * previous[0] + plane.b * previous[1] - plane.c
    previous_inside = previous_residual <= tolerance

    for current in vertices:
        current_residual = plane.a * current[0] + plane.b * current[1] - plane.c
        current_inside = current_residual <= tolerance
        if current_inside != previous_inside:
            fraction = previous_residual / (previous_residual - current_residual)
            clipped.append((
                previous[0] + fraction * (current[0] - previous[0]),
                previous[1] + fraction * (current[1] - previous[1]),
            ))
        if current_inside:
            clipped.append(current)
        previous = current
        previous_residual = current_residual
        previous_inside = current_inside
    return clipped


def _deduplicate_adjacent(vertices: list[Point], tolerance: float) -> list[Point]:
    distinct: list[Point] = []
    for vertex in vertices:
        if not distinct or math.dist(vertex, distinct[-1]) > tolerance:
            distinct.append(vertex)
    if len(distinct) > 1 and math.dist(distinct[0], distinct[-1]) <= tolerance:
        distinct.pop()
    return distinct


def _convex_hull(points: list[Point], tolerance: float) -> list[Point]:
    ordered: list[Point] = []
    for point in sorted(points):
        if not ordered or math.dist(point, ordered[-1]) > tolerance:
            ordered.append(point)
    if len(ordered) <= 1:
        return ordered

    def cross(origin: Point, first: Point, second: Point) -> float:
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower: list[Point] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= tolerance:
            lower.pop()
        lower.append(point)
    upper: list[Point] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= tolerance:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def intersect_halfplanes(
    halfplanes: Iterable[HalfPlane], tolerance: float = 1e-9
) -> Region:
    planes = tuple(halfplanes)
    matrix = [[plane.a, plane.b] for plane in planes] or None
    bounds = [plane.c for plane in planes] or None
    variable_bounds = [(None, None), (None, None)]

    feasibility = linprog(
        [0.0, 0.0], A_ub=matrix, b_ub=bounds,
        bounds=variable_bounds, method="highs",
    )
    if feasibility.status == 2:
        return Region("empty", (), 0.0)
    if feasibility.status != 0:
        raise RuntimeError(f"feasibility optimization failed: {feasibility.message}")

    objectives = ([1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0])
    extrema = [
        linprog(objective, A_ub=matrix, b_ub=bounds,
                bounds=variable_bounds, method="highs")
        for objective in objectives
    ]
    if any(result.status == 3 for result in extrema):
        return Region("unbounded", (), 0.0)
    if any(result.status != 0 for result in extrema):
        message = next(result.message for result in extrema if result.status != 0)
        raise RuntimeError(f"coordinate optimization failed: {message}")

    x_min = float(extrema[0].x[0])
    x_max = float(extrema[1].x[0])
    y_min = float(extrema[2].x[1])
    y_max = float(extrema[3].x[1])
    vertices = [(x_min, y_min), (x_max, y_min),
                (x_max, y_max), (x_min, y_max)]
    for plane in planes:
        vertices = _deduplicate_adjacent(
            _clip_polygon(vertices, plane, tolerance), tolerance
        )

    hull = _convex_hull(vertices, tolerance)
    if not hull:
        return Region("empty", (), 0.0)
    status = {1: "point", 2: "segment"}.get(len(hull), "polygon")
    max_violation = max(
        (plane.a * x + plane.b * y - plane.c
         for x, y in hull for plane in planes),
        default=0.0,
    )
    return Region(status, tuple(hull), max_violation)
