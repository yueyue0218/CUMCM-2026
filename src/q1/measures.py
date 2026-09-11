"""Diameter and coverage measures for convex planar regions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from src.q1.geometry import Point


@dataclass(frozen=True)
class DiameterResult:
    squared_distance: float
    witness: tuple[Point, Point] | None

    @property
    def distance(self) -> float:
        return math.sqrt(self.squared_distance)


@dataclass(frozen=True)
class CoverageResult:
    covers: bool
    center: Point | None
    radius: float
    max_distance: float


@dataclass(frozen=True)
class LineSegment:
    start: Point
    end: Point

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(value)) for point in (self.start, self.end)
                   for value in point):
            raise ValueError("line segment coordinates must be finite")


@dataclass(frozen=True)
class CircularArc:
    center: Point
    radius: float
    start_angle: float
    sweep_angle: float

    def __post_init__(self) -> None:
        values = (*self.center, self.radius, self.start_angle, self.sweep_angle)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("circular arc values must be finite")
        if self.radius <= 0.0:
            raise ValueError("circular arc radius must be positive")
        if not 0.0 <= self.sweep_angle <= 2.0 * math.pi:
            raise ValueError("circular arc sweep must be in [0, 2*pi]")

    @property
    def start(self) -> Point:
        return (
            self.center[0] + self.radius * math.cos(self.start_angle),
            self.center[1] + self.radius * math.sin(self.start_angle),
        )

    @property
    def end(self) -> Point:
        angle = self.start_angle + self.sweep_angle
        return (
            self.center[0] + self.radius * math.cos(angle),
            self.center[1] + self.radius * math.sin(angle),
        )


def _squared_distance(first: Point, second: Point) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def _normalized_pair(first: Point, second: Point) -> tuple[Point, Point]:
    return (first, second) if first <= second else (second, first)


def _better_result(
    current: DiameterResult, first: Point, second: Point
) -> DiameterResult:
    squared = _squared_distance(first, second)
    pair = _normalized_pair(first, second)
    if (squared > current.squared_distance
            or (squared == current.squared_distance
                and (current.witness is None or pair < current.witness))):
        return DiameterResult(squared, pair)
    return current


def diameter_exhaustive(vertices: Sequence[Point]) -> DiameterResult:
    if not vertices:
        return DiameterResult(0.0, None)
    result = DiameterResult(0.0, (vertices[0], vertices[0]))
    for index, first in enumerate(vertices):
        for second in vertices[index + 1:]:
            result = _better_result(result, first, second)
    return result


def _twice_triangle_area(first: Point, second: Point, opposite: Point) -> float:
    return abs((second[0] - first[0]) * (opposite[1] - first[1])
               - (second[1] - first[1]) * (opposite[0] - first[0]))


def diameter_rotating_calipers(vertices: Sequence[Point]) -> DiameterResult:
    count = len(vertices)
    if count <= 2:
        return diameter_exhaustive(vertices)

    result = DiameterResult(0.0, None)
    opposite = 1
    for index in range(count):
        next_index = (index + 1) % count
        while (_twice_triangle_area(vertices[index], vertices[next_index],
                                    vertices[(opposite + 1) % count])
               > _twice_triangle_area(vertices[index], vertices[next_index],
                                      vertices[opposite])):
            opposite = (opposite + 1) % count

        result = _better_result(result, vertices[index], vertices[opposite])
        result = _better_result(result, vertices[next_index], vertices[opposite])
        if (_twice_triangle_area(vertices[index], vertices[next_index],
                                 vertices[(opposite + 1) % count])
                == _twice_triangle_area(vertices[index], vertices[next_index],
                                        vertices[opposite])):
            tied = (opposite + 1) % count
            result = _better_result(result, vertices[index], vertices[tied])
            result = _better_result(result, vertices[next_index], vertices[tied])
    return result


def diameter_circle_coverage(
    vertices: Sequence[Point], diameter: DiameterResult
) -> CoverageResult:
    if diameter.witness is None:
        return CoverageResult(True, None, 0.0, 0.0)
    first, second = diameter.witness
    center = ((first[0] + second[0]) / 2.0,
              (first[1] + second[1]) / 2.0)
    radius_squared = diameter.squared_distance / 4.0
    maximum_squared = max(
        (_squared_distance(vertex, center) for vertex in vertices), default=0.0
    )
    scale = max(1.0, radius_squared, maximum_squared)
    tolerance = 1e-12 * scale
    return CoverageResult(
        maximum_squared <= radius_squared + tolerance,
        center,
        math.sqrt(radius_squared),
        math.sqrt(maximum_squared),
    )


BoundaryElement = LineSegment | CircularArc


def _line_contributions(element: LineSegment) -> tuple[float, float, float]:
    x0, y0 = element.start
    dx = element.end[0] - x0
    dy = element.end[1] - y0
    area = 0.5 * (x0 * dy - y0 * dx)
    x_moment = 0.5 * dy * (x0 * x0 + x0 * dx + dx * dx / 3.0)
    y_moment = -0.5 * dx * (y0 * y0 + y0 * dy + dy * dy / 3.0)
    return area, x_moment, y_moment


def _arc_contributions(element: CircularArc) -> tuple[float, float, float]:
    cx, cy = element.center
    radius = element.radius
    start = element.start_angle
    end = start + element.sweep_angle
    sweep = element.sweep_angle

    sin_difference = math.sin(end) - math.sin(start)
    negative_cos_difference = math.cos(start) - math.cos(end)
    sin_double_difference = math.sin(2.0 * end) - math.sin(2.0 * start)
    integral_cos_squared = sweep / 2.0 + sin_double_difference / 4.0
    integral_sin_squared = sweep / 2.0 - sin_double_difference / 4.0
    integral_cos_cubed = (
        sin_difference - (math.sin(end) ** 3 - math.sin(start) ** 3) / 3.0
    )
    integral_sin_cubed = (
        negative_cos_difference
        + (math.cos(end) ** 3 - math.cos(start) ** 3) / 3.0
    )

    area = 0.5 * (
        radius * cx * sin_difference
        + radius * cy * negative_cos_difference
        + radius * radius * sweep
    )
    x_moment = 0.5 * radius * (
        cx * cx * sin_difference
        + 2.0 * cx * radius * integral_cos_squared
        + radius * radius * integral_cos_cubed
    )
    y_moment = 0.5 * radius * (
        cy * cy * negative_cos_difference
        + 2.0 * cy * radius * integral_sin_squared
        + radius * radius * integral_sin_cubed
    )
    return area, x_moment, y_moment


def green_area_centroid(
    elements: Sequence[BoundaryElement],
) -> tuple[float, Point]:
    if not elements:
        raise ValueError("boundary must contain at least one element")

    origin = elements[0].start

    def local_point(point: Point) -> Point:
        return point[0] - origin[0], point[1] - origin[1]

    local_elements: list[BoundaryElement] = []
    local_endpoints: list[tuple[Point, Point]] = []
    boundary_scale = 0.0
    for element in elements:
        endpoints = local_point(element.start), local_point(element.end)
        local_endpoints.append(endpoints)
        if isinstance(element, LineSegment):
            local_element: BoundaryElement = LineSegment(*endpoints)
        else:
            local_element = CircularArc(
                local_point(element.center), element.radius,
                element.start_angle, element.sweep_angle,
            )
            boundary_scale = max(boundary_scale, element.radius)
        local_elements.append(local_element)
        boundary_scale = max(
            boundary_scale,
            *(abs(value) for point in endpoints for value in point),
        )

    length_tolerance = 1e-12 * max(boundary_scale, 1e-300)
    for index, (_, end) in enumerate(local_endpoints):
        following_start = local_endpoints[(index + 1) % len(local_endpoints)][0]
        if math.dist(end, following_start) > length_tolerance:
            raise ValueError("boundary elements must form a closed boundary")

    contributions = [
        (
            _line_contributions(element) if isinstance(element, LineSegment)
            else _arc_contributions(element)
        )
        for element in local_elements
    ]
    area = math.fsum(contribution[0] for contribution in contributions)
    x_moment = math.fsum(contribution[1] for contribution in contributions)
    y_moment = math.fsum(contribution[2] for contribution in contributions)

    area_tolerance = 1e-12 * max(boundary_scale * boundary_scale, 1e-300)
    if abs(area) <= area_tolerance:
        raise ValueError("boundary encloses negligible area")
    return area, (
        origin[0] + x_moment / area,
        origin[1] + y_moment / area,
    )


def polygon_area_centroid(vertices: Sequence[Point]) -> tuple[float, Point]:
    if not vertices:
        raise ValueError("polygon must contain at least one vertex")
    elements = [
        LineSegment(vertex, vertices[(index + 1) % len(vertices)])
        for index, vertex in enumerate(vertices)
    ]
    return green_area_centroid(elements)
