"""Deterministic minimum enclosing circles for finite planar points."""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Iterable

from src.q1.geometry import Point


@dataclass(frozen=True)
class Circle:
    center: Point
    radius: float
    max_residual: float


def _contains(circle: Circle, point: Point) -> bool:
    distance = math.dist(circle.center, point)
    tolerance = 1e-12 * max(1.0, circle.radius, distance)
    return distance <= circle.radius + tolerance


def _one_point_circle(point: Point) -> Circle:
    return Circle(point, 0.0, 0.0)


def _two_point_circle(first: Point, second: Point) -> Circle:
    center = (
        (first[0] + second[0]) / 2.0,
        (first[1] + second[1]) / 2.0,
    )
    return Circle(center, math.dist(first, second) / 2.0, 0.0)


def _three_point_circle(
    first: Point, second: Point, third: Point
) -> Circle | None:
    bx = second[0] - first[0]
    by = second[1] - first[1]
    cx = third[0] - first[0]
    cy = third[1] - first[1]
    denominator = 2.0 * (bx * cy - by * cx)
    scale = max(math.hypot(bx, by), math.hypot(cx, cy),
                math.dist(second, third), 1.0)
    if abs(denominator) <= 1e-14 * scale * scale:
        return None

    b_squared = bx * bx + by * by
    c_squared = cx * cx + cy * cy
    offset_x = (cy * b_squared - by * c_squared) / denominator
    offset_y = (bx * c_squared - cx * b_squared) / denominator
    center = (first[0] + offset_x, first[1] + offset_y)
    return Circle(center, math.dist(center, first), 0.0)


def _smallest_boundary_circle(boundary: tuple[Point, ...]) -> Circle:
    if not boundary:
        return Circle((0.0, 0.0), 0.0, 0.0)

    candidates = [_one_point_circle(point) for point in boundary]
    candidates.extend(
        _two_point_circle(first, second)
        for first, second in itertools.combinations(boundary, 2)
    )
    for points in itertools.combinations(boundary, 3):
        candidate = _three_point_circle(*points)
        if candidate is not None:
            candidates.append(candidate)

    covering = [
        circle for circle in candidates
        if all(_contains(circle, point) for point in boundary)
    ]
    if not covering:
        raise RuntimeError("could not construct a circle for boundary points")
    return min(covering, key=lambda circle: (circle.radius, circle.center))


def minimum_enclosing_circle(
    points: Iterable[Point], seed: int = 20260911
) -> Circle:
    """Return a certified minimum circle containing all ``points``."""

    original: list[Point] = []
    for point in points:
        x, y = point
        normalized = (float(x), float(y))
        if not all(math.isfinite(value) for value in normalized):
            raise ValueError("point coordinates must be finite")
        original.append(normalized)

    shuffled = sorted(set(original))
    random.Random(seed).shuffle(shuffled)

    def welzl(prefix_count: int, boundary: tuple[Point, ...]) -> Circle:
        if prefix_count == 0 or len(boundary) == 3:
            return _smallest_boundary_circle(boundary)

        point = shuffled[prefix_count - 1]
        circle = welzl(prefix_count - 1, boundary)
        if _contains(circle, point):
            return circle
        return welzl(prefix_count - 1, boundary + (point,))

    circle = welzl(len(shuffled), ())
    residuals = [math.dist(circle.center, point) - circle.radius
                 for point in original]
    max_residual = max(residuals, default=0.0)
    scale = max(1.0, circle.radius,
                *(math.dist(circle.center, point) for point in original))
    if max_residual > 1e-10 * scale:
        raise RuntimeError("minimum enclosing circle failed coverage verification")
    return Circle(circle.center, circle.radius, max_residual)
