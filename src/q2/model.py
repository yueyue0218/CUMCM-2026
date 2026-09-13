"""Minimal Q2 model primitives.

This module intentionally stops at Task 1: state construction, candidate-domain
checks, and second-response validation.  It does not implement Q2 evaluators,
second-support updates, optimizers, or experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from src.common.geometry import (
    Point,
    distance,
    normalize_angle_deg,
    signed_angle_difference_deg,
)
from src.common.localization import BearingObservation, localization_region


_DISTANCE_TOLERANCE_M = 1e-9
_ANGLE_TOLERANCE_DEG = 1e-12


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


@dataclass(frozen=True)
class Q2Config:
    arena_radius_m: float = 1800.0
    reception_radius_min_m: float = 1000.0
    reception_radius_max_m: float = 1500.0
    near_radius_m: float = 5.0
    bearing_error_deg: float = 1.0
    circle_vertices: int = 720
    direction_angle_bins: int = 720

    def __post_init__(self) -> None:
        numeric_values = (
            self.arena_radius_m,
            self.reception_radius_min_m,
            self.reception_radius_max_m,
            self.near_radius_m,
            self.bearing_error_deg,
            self.circle_vertices,
            self.direction_angle_bins,
        )
        if not all(_is_finite_number(value) for value in numeric_values):
            raise ValueError("Q2Config parameters must be finite")
        if self.arena_radius_m <= 0.0:
            raise ValueError("arena_radius_m must be positive")
        if not (
            0.0
            < self.near_radius_m
            < self.reception_radius_min_m
            <= self.reception_radius_max_m
        ):
            raise ValueError(
                "radii must satisfy 0 < near < reception_min <= reception_max"
            )
        if self.bearing_error_deg < 0.0:
            raise ValueError("bearing_error_deg must be non-negative")
        if isinstance(self.circle_vertices, bool) or not isinstance(
            self.circle_vertices, int
        ):
            raise ValueError("circle_vertices must be an integer")
        if isinstance(self.direction_angle_bins, bool) or not isinstance(
            self.direction_angle_bins, int
        ):
            raise ValueError("direction_angle_bins must be an integer")
        if self.circle_vertices < 3:
            raise ValueError("circle_vertices must be at least 3")
        if self.direction_angle_bins < 3:
            raise ValueError("direction_angle_bins must be at least 3")


@dataclass(frozen=True)
class FirstDirectionObservation:
    station: Point
    bearing_deg: float

    def __post_init__(self) -> None:
        _validate_point(self.station, "station")
        if not _is_finite_number(self.bearing_deg):
            raise ValueError("bearing_deg must be finite")
        object.__setattr__(self, "bearing_deg", normalize_angle_deg(self.bearing_deg))


@dataclass(frozen=True)
class JointSample:
    position: Point
    reception_radius_m: float
    weight: float

    def __post_init__(self) -> None:
        _validate_point(self.position, "position")
        if not _is_finite_number(self.reception_radius_m):
            raise ValueError("reception_radius_m must be finite")
        if not _is_finite_number(self.weight):
            raise ValueError("weight must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be non-negative")


@dataclass(frozen=True)
class FirstState:
    observation: FirstDirectionObservation
    exact_support_label: str
    outer_region: tuple[Point, ...]
    joint_samples: tuple[JointSample, ...]


@dataclass(frozen=True)
class CandidateDomainFlags:
    in_c_poss_proxy: bool
    in_c_rec_certified: bool
    min_distance_to_outer_m: float
    max_distance_to_outer_m: float


class SecondResponseKind(str, Enum):
    NEAR = "near"
    DIRECTION = "direction"
    NO_SIGNAL = "no_signal"


@dataclass(frozen=True)
class SecondResponse:
    kind: SecondResponseKind
    bearing_deg: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SecondResponseKind):
            try:
                object.__setattr__(self, "kind", SecondResponseKind(self.kind))
            except ValueError as exc:
                raise ValueError("unknown second response kind") from exc

        if self.kind is SecondResponseKind.DIRECTION:
            if self.bearing_deg is None:
                raise ValueError("direction response requires bearing_deg")
            if not _is_finite_number(self.bearing_deg):
                raise ValueError("bearing_deg must be finite")
            object.__setattr__(
                self,
                "bearing_deg",
                normalize_angle_deg(self.bearing_deg),
            )
        elif self.bearing_deg is not None:
            raise ValueError("near and no_signal responses must not include bearing_deg")


def build_first_state(
    observation: FirstDirectionObservation,
    *,
    samples: Sequence[JointSample],
    config: Q2Config = Q2Config(),
) -> FirstState:
    """Build the first-observation Q2 state for an existing direction response."""

    if not isinstance(observation, FirstDirectionObservation):
        observation = FirstDirectionObservation(
            station=observation.station,  # type: ignore[attr-defined]
            bearing_deg=observation.bearing_deg,  # type: ignore[attr-defined]
        )

    q1_region = localization_region(
        [
            BearingObservation(
                station=observation.station,
                bearing_deg=observation.bearing_deg,
                error_deg=config.bearing_error_deg,
            )
        ],
        arena_radius_m=config.arena_radius_m,
        circle_vertices=config.circle_vertices,
        reception_radius_upper_m=config.reception_radius_max_m,
    )
    if not q1_region:
        raise ValueError("conservative first-observation outer region is empty")

    if not samples:
        raise ValueError("samples must be non-empty")
    if not any(sample.weight > 0.0 for sample in samples):
        raise ValueError("at least one input sample must have positive weight")

    filtered_samples = tuple(
        sample
        for sample in samples
        if sample.weight > 0.0
        and _is_sample_compatible_with_first_direction(sample, observation, config)
    )
    if not filtered_samples:
        raise ValueError("no samples remain after first-direction filtering")
    if not any(sample.weight > 0.0 for sample in filtered_samples):
        raise ValueError("filtered samples must include positive weight")

    return FirstState(
        observation=observation,
        exact_support_label=(
            "closure(F1)=Omega cap W(S1,theta1,delta) cap B(S1,1500); "
            "nominal direction branch additionally enforces ||G-S1|| > 5"
        ),
        outer_region=tuple(q1_region),
        joint_samples=filtered_samples,
    )


def evaluate_candidate_domains(
    q: Point,
    state: FirstState,
    *,
    config: Q2Config = Q2Config(),
) -> CandidateDomainFlags:
    """Evaluate Q2 candidate-domain proxies against the conservative outer region."""

    _validate_point(q, "q")
    if not state.outer_region:
        raise ValueError("state.outer_region must be non-empty")

    min_distance = _point_to_polygon_distance(q, state.outer_region)
    max_distance = max(distance(q, vertex) for vertex in state.outer_region)
    return CandidateDomainFlags(
        in_c_poss_proxy=(
            min_distance <= config.reception_radius_max_m + _DISTANCE_TOLERANCE_M
        ),
        in_c_rec_certified=(
            max_distance <= config.reception_radius_min_m + _DISTANCE_TOLERANCE_M
        ),
        min_distance_to_outer_m=min_distance,
        max_distance_to_outer_m=max_distance,
    )


def _is_sample_compatible_with_first_direction(
    sample: JointSample,
    observation: FirstDirectionObservation,
    config: Q2Config,
) -> bool:
    if not (
        config.reception_radius_min_m
        <= sample.reception_radius_m
        <= config.reception_radius_max_m
    ):
        return False
    if distance(sample.position, (0.0, 0.0)) > (
        config.arena_radius_m + _DISTANCE_TOLERANCE_M
    ):
        return False

    station_distance = distance(sample.position, observation.station)
    if station_distance <= config.near_radius_m + _DISTANCE_TOLERANCE_M:
        return False
    if station_distance > sample.reception_radius_m + _DISTANCE_TOLERANCE_M:
        return False

    true_bearing = _bearing_deg(observation.station, sample.position)
    return (
        abs(signed_angle_difference_deg(true_bearing, observation.bearing_deg))
        <= config.bearing_error_deg + _ANGLE_TOLERANCE_DEG
    )


def _bearing_deg(origin: Point, target: Point) -> float:
    return normalize_angle_deg(
        math.degrees(math.atan2(target[1] - origin[1], target[0] - origin[0]))
    )


def _point_to_polygon_distance(point: Point, polygon: Sequence[Point]) -> float:
    if not polygon:
        raise ValueError("polygon must be non-empty")
    if len(polygon) == 1:
        return distance(point, polygon[0])
    if len(polygon) == 2:
        return _point_segment_distance(point, polygon[0], polygon[1])
    if _point_in_convex_polygon(point, polygon):
        return 0.0
    return min(
        _point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def _point_segment_distance(point: Point, a: Point, b: Point) -> float:
    _validate_point(point, "point")
    _validate_point(a, "segment start")
    _validate_point(b, "segment end")

    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return distance(point, a)
    t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length_squared
    t = min(1.0, max(0.0, t))
    projection = (a[0] + t * dx, a[1] + t * dy)
    return distance(point, projection)


def _point_in_convex_polygon(
    point: Point,
    polygon: Sequence[Point],
    *,
    epsilon: float = 1e-9,
) -> bool:
    _validate_point(point, "point")
    if not polygon:
        raise ValueError("polygon must be non-empty")
    if len(polygon) == 1:
        return distance(point, polygon[0]) <= epsilon
    if len(polygon) == 2:
        return _point_segment_distance(point, polygon[0], polygon[1]) <= epsilon

    has_positive = False
    has_negative = False
    for index, current in enumerate(polygon):
        _validate_point(current, "polygon vertex")
        following = polygon[(index + 1) % len(polygon)]
        _validate_point(following, "polygon vertex")
        cross = _cross(
            (following[0] - current[0], following[1] - current[1]),
            (point[0] - current[0], point[1] - current[1]),
        )
        if cross > epsilon:
            has_positive = True
        elif cross < -epsilon:
            has_negative = True
        if has_positive and has_negative:
            return False
    return True


def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _validate_point(point: Point, name: str) -> None:
    if len(point) != 2 or not all(_is_finite_number(coordinate) for coordinate in point):
        raise ValueError(f"{name} must be a finite 2D point")
