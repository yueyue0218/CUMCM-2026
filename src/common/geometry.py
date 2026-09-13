"""Small dependency-free geometry helpers shared by Q1-Q4.

Angles follow the statement: 0 degrees points east and increase
counter-clockwise.  Direction measurements constrain a source to a wedge of
``bearing +/- error``; callers must not treat the measured bearing as exact.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

Point = tuple[float, float]


@dataclass(frozen=True)
class Circle:
    """A circle represented by its center and non-negative radius."""

    center: Point
    radius: float


_MEC_REL_TOL = 1e-12
_MEC_ABS_TOL = 1e-12


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


def _circle_contains(circle: Circle, point: Point) -> bool:
    point_distance = distance(point, circle.center)
    tolerance = max(
        _MEC_ABS_TOL,
        _MEC_REL_TOL * max(circle.radius, point_distance),
    )
    return point_distance <= circle.radius + tolerance


def _diameter_circle(a: Point, b: Point) -> Circle:
    center = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    return Circle(center=center, radius=distance(a, b) / 2.0)


def _circumcircle(a: Point, b: Point, c: Point) -> Circle:
    bx, by = b[0] - a[0], b[1] - a[1]
    cx, cy = c[0] - a[0], c[1] - a[1]
    denominator = 2.0 * cross((bx, by), (cx, cy))
    if denominator == 0.0:
        raise ArithmeticError("cannot construct a circumcircle for collinear points")

    b_squared = bx * bx + by * by
    c_squared = cx * cx + cy * cy
    center = (
        a[0] + (cy * b_squared - by * c_squared) / denominator,
        a[1] + (bx * c_squared - cx * b_squared) / denominator,
    )
    radius = max(distance(center, point) for point in (a, b, c))
    circle = Circle(center=center, radius=radius)
    if not all(math.isfinite(value) for value in (*circle.center, circle.radius)):
        raise ArithmeticError("non-finite circumcircle produced by floating-point arithmetic")
    return circle


def _minimum_circle_for_three(a: Point, b: Point, c: Point) -> Circle:
    points = (a, b, c)
    diameter_candidates = (
        _diameter_circle(a, b),
        _diameter_circle(a, c),
        _diameter_circle(b, c),
    )
    covering_diameter_circles = [
        circle
        for circle in diameter_candidates
        if all(_circle_contains(circle, point) for point in points)
    ]
    if covering_diameter_circles:
        return min(
            covering_diameter_circles,
            key=lambda circle: (circle.radius, circle.center),
        )
    return _circumcircle(a, b, c)


def minimum_enclosing_circle(points: Sequence[Point]) -> Circle:
    """Return the minimum enclosing circle of a non-empty finite point set.

    The implementation is a deterministic incremental algorithm.  Points are
    sorted and deduplicated first, so neither random state nor input order can
    affect the processing order.  An empty input raises ``ValueError``.

    The returned circle is a finite-point geometry result only; it does not
    certify that a clear action may be issued.  Callers must independently
    validate the actual output center against the relevant localization region.
    """

    if not points:
        raise ValueError("minimum enclosing circle requires at least one point")
    if not all(math.isfinite(coordinate) for point in points for coordinate in point):
        raise ValueError("minimum enclosing circle requires finite coordinates")

    ordered = sorted(set(points))
    origin = ordered[0]
    local_points = [
        (point[0] - origin[0], point[1] - origin[1])
        for point in ordered
    ]

    circle = Circle(center=local_points[0], radius=0.0)
    for i, point in enumerate(local_points):
        if _circle_contains(circle, point):
            continue
        circle = Circle(center=point, radius=0.0)
        for j, second in enumerate(local_points[:i]):
            if _circle_contains(circle, second):
                continue
            circle = _diameter_circle(point, second)
            for third in local_points[:j]:
                if not _circle_contains(circle, third):
                    circle = _minimum_circle_for_three(point, second, third)

    if not all(_circle_contains(circle, point) for point in local_points):
        raise ArithmeticError("computed circle does not contain every input point")
    return Circle(
        center=(circle.center[0] + origin[0], circle.center[1] + origin[1]),
        radius=circle.radius,
    )


def max_distance_to_region(region: Sequence[Point], center: Point) -> float:
    """Return the maximum distance from ``center`` to a convex region.

    ``region`` is represented by its convex-polygon vertices, so the maximum
    is attained at a vertex.  An empty region returns ``math.inf``: absence of
    a region must never be interpreted as a zero-radius coverage certificate.
    """

    return max((distance(vertex, center) for vertex in region), default=math.inf)


def is_clear_point_certified(
    region: Sequence[Point],
    center: Point,
    clear_radius_m: float = 20.0,
) -> bool:
    """Return whether ``center``'s closed clear disk covers ``region``."""

    return max_distance_to_region(region, center) <= clear_radius_m


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


def clip_polygon_to_circle_outer(
    polygon: Sequence[Point],
    center: Point,
    radius: float,
    vertex_count: int = 720,
) -> list[Point]:
    """Clip ``polygon`` to a conservative outer approximation of a circle.

    ``radius`` is the incircle radius of a regular circumscribed polygon, so
    the true closed circle is contained in the clipping polygon.  ``center``
    translates that outer polygon from the origin.
    """

    outer_circle = [
        (center[0] + point[0], center[1] + point[1])
        for point in circle_polygon(radius, vertex_count)
    ]
    clipped = list(polygon)
    for index, boundary_point in enumerate(outer_circle):
        next_point = outer_circle[(index + 1) % len(outer_circle)]
        boundary_direction = (
            next_point[0] - boundary_point[0],
            next_point[1] - boundary_point[1],
        )
        clipped = clip_polygon_half_plane(
            clipped,
            boundary_point,
            boundary_direction,
            keep_left=True,
        )
        if not clipped:
            break
    return clipped


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


def polygon_diameter_exhaustive(points: Sequence[Point]) -> float:
    """Return a point set's convex-hull diameter by exhaustive enumeration.

    This ``O(h^2)`` implementation is retained as a reference oracle, where
    ``h`` is the number of convex-hull vertices.
    """

    hull = convex_hull(points)
    return max(
        (distance(hull[i], hull[j]) for i in range(len(hull)) for j in range(i)),
        default=0.0,
    )


def polygon_diameter_calipers(points: Sequence[Point]) -> float:
    """Return a point set's convex-hull diameter using rotating calipers.

    The input need not already be a convex polygon or have any particular
    ordering.  Constructing its ``h``-vertex convex hull takes ``O(n log n)``;
    the monotone antipodal-pointer scan takes ``O(h)``.  The hull is the main
    source of additional space.
    """

    hull = convex_hull(points)
    hull_size = len(hull)
    if hull_size < 2:
        return 0.0
    if hull_size == 2:
        return distance(hull[0], hull[1])

    def vertex(index: int) -> Point:
        return hull[index % hull_size]

    def doubled_area(edge_index: int, point_index: int) -> float:
        edge_start = vertex(edge_index)
        edge_end = vertex(edge_index + 1)
        edge = (
            edge_end[0] - edge_start[0],
            edge_end[1] - edge_start[1],
        )
        relative = (
            vertex(point_index)[0] - edge_start[0],
            vertex(point_index)[1] - edge_start[1],
        )
        return cross(edge, relative)

    antipodal = 1
    maximum_distance = 0.0
    for edge_index in range(hull_size):
        while (
            antipodal + 1 < edge_index + hull_size
            and doubled_area(edge_index, antipodal + 1)
            > doubled_area(edge_index, antipodal)
        ):
            antipodal += 1

        candidate_indices = (antipodal,)
        if (
            antipodal + 1 < edge_index + hull_size
            and doubled_area(edge_index, antipodal + 1)
            == doubled_area(edge_index, antipodal)
        ):
            candidate_indices = (antipodal, antipodal + 1)

        for candidate_index in candidate_indices:
            maximum_distance = max(
                maximum_distance,
                distance(vertex(edge_index), vertex(candidate_index)),
                distance(vertex(edge_index + 1), vertex(candidate_index)),
            )

    return maximum_distance


def polygon_diameter(points: Sequence[Point]) -> float:
    """Return a point set's convex-hull diameter using rotating calipers.

    The input need not already be a convex hull.  For ``n`` input points and
    ``h`` convex-hull vertices, hull construction takes ``O(n log n)`` and the
    rotating-calipers scan takes ``O(h)``.
    """

    return polygon_diameter_calipers(points)


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
