"""Small dependency-free geometry helpers shared by Q1-Q4.

Angles follow the statement: 0 degrees points east and increase
counter-clockwise.  Direction measurements constrain a source to a wedge of
``bearing +/- error``; callers must not treat the measured bearing as exact.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

Point = tuple[float, float]


def normalize_angle_deg(angle: float) -> float:
    """Return an angle in [0, 360), including stable wraparound at 360."""

    normalized = angle % 360.0
    return 0.0 if math.isclose(normalized, 360.0) else normalized


def signed_angle_difference_deg(target: float, reference: float) -> float:
    """Return the signed shortest rotation from reference to target."""

    difference = (target - reference + 180.0) % 360.0 - 180.0
    return 180.0 if math.isclose(difference, -180.0) else difference


def unit_vector(angle_deg: float) -> Point:
    angle_rad = math.radians(angle_deg)
    return math.cos(angle_rad), math.sin(angle_rad)


def cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def circle_polygon(radius: float, vertex_count: int = 720) -> list[Point]:
    """Return a conservative regular-polygon approximation of a circle.

    The polygon is circumscribed so a source on the true arena boundary is not
    accidentally removed. Increase ``vertex_count`` to reduce outer error.
    """

    if radius <= 0 or vertex_count < 3:
        raise ValueError("radius must be positive and vertex_count at least 3")
    vertex_radius = radius / math.cos(math.pi / vertex_count)
    return [
        (vertex_radius * math.cos(2 * math.pi * i / vertex_count),
         vertex_radius * math.sin(2 * math.pi * i / vertex_count))
        for i in range(vertex_count)
    ]


def clip_polygon_half_plane(
    polygon: Sequence[Point],
    boundary_point: Point,
    boundary_direction: Point,
    *,
    keep_left: bool = True,
    epsilon: float = 1e-9,
) -> list[Point]:
    """Clip a polygon to one side of an oriented infinite line."""

    if not polygon:
        return []

    def side(point: Point) -> float:
        relative = (point[0] - boundary_point[0], point[1] - boundary_point[1])
        value = cross(boundary_direction, relative)
        return value if keep_left else -value

    output: list[Point] = []
    previous = polygon[-1]
    previous_side = side(previous)
    previous_inside = previous_side >= -epsilon
    for current in polygon:
        current_side = side(current)
        current_inside = current_side >= -epsilon
        if current_inside != previous_inside:
            denominator = previous_side - current_side
            if abs(denominator) > epsilon:
                t = previous_side / denominator
                output.append(
                    (
                        previous[0] + t * (current[0] - previous[0]),
                        previous[1] + t * (current[1] - previous[1]),
                    )
                )
        if current_inside:
            output.append(current)
        previous = current
        previous_side = current_side
        previous_inside = current_inside
    return output


def clip_polygon_to_bearing_wedge(
    polygon: Sequence[Point],
    station: Point,
    bearing_deg: float,
    error_deg: float = 1.0,
) -> list[Point]:
    """Intersect a polygon with the forward wedge of one measurement."""

    if not 0 <= error_deg < 90:
        raise ValueError("error_deg must be in [0, 90)")
    lower = unit_vector(bearing_deg - error_deg)
    upper = unit_vector(bearing_deg + error_deg)
    clipped = clip_polygon_half_plane(polygon, station, lower, keep_left=True)
    return clip_polygon_half_plane(clipped, station, upper, keep_left=False)


def convex_hull(points: Iterable[Point], epsilon: float = 1e-12) -> list[Point]:
    """Return vertices of the convex hull in counter-clockwise order."""

    ordered = sorted(set(points))
    if len(ordered) <= 1:
        return ordered

    def build_half(items: Sequence[Point]) -> list[Point]:
        half: list[Point] = []
        for point in items:
            while len(half) >= 2:
                ab = (half[-1][0] - half[-2][0], half[-1][1] - half[-2][1])
                bc = (point[0] - half[-1][0], point[1] - half[-1][1])
                if cross(ab, bc) > epsilon:
                    break
                half.pop()
            half.append(point)
        return half

    lower = build_half(ordered)
    upper = build_half(list(reversed(ordered)))
    return lower[:-1] + upper[:-1]


def polygon_diameter(points: Sequence[Point]) -> float:
    """Return the maximum distance between points of a convex polygon.

    The current implementation enumerates hull vertices.  It is deliberately
    simple and serves as an independent reference for a later rotating-calipers
    optimization.
    """

    hull = convex_hull(points)
    return max(
        (distance(hull[i], hull[j]) for i in range(len(hull)) for j in range(i)),
        default=0.0,
    )


def candidate_second_points(
    first_point: Point,
    first_bearing_deg: float,
    lateral_offsets_m: Iterable[float],
    forward_offsets_m: Iterable[float] = (0.0,),
) -> list[Point]:
    """Generate symmetric second-point candidates around the first bearing.

    These are candidates, not a complete Q2 policy: scoring must also account
    for reacquisition risk, localization quality, and travel time.
    """

    forward = unit_vector(first_bearing_deg)
    left = (-forward[1], forward[0])
    candidates: list[Point] = []
    for along in forward_offsets_m:
        for lateral in lateral_offsets_m:
            for sign in (-1.0, 1.0):
                candidates.append(
                    (
                        first_point[0] + along * forward[0] + sign * lateral * left[0],
                        first_point[1] + along * forward[1] + sign * lateral * left[1],
                    )
                )
    return candidates
