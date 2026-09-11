"""Composition and conservative clearance decisions for Question 1."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from src.q1.enclosing_circle import minimum_enclosing_circle
from src.q1.geometry import Observation, bearing_halfplanes, intersect_halfplanes
from src.q1.measures import (
    diameter_circle_coverage,
    diameter_exhaustive,
    diameter_rotating_calipers,
    polygon_area_centroid,
)


def _positive_finite(payload: Mapping[str, object], key: str, default: float) -> float:
    value = float(payload.get(key, default))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{key} must be a positive finite number")
    return value


def _output_decimals(payload: Mapping[str, object]) -> int:
    value = payload.get("output_decimals", 6)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 15:
        raise ValueError("output_decimals must be an integer in [0, 15]")
    return value


def _parse_observations(payload: Mapping[str, object]) -> list[Observation]:
    raw_observations = payload.get("observations")
    if (not isinstance(raw_observations, Sequence)
            or isinstance(raw_observations, (str, bytes))):
        raise ValueError("observations must be a sequence")

    observations: list[Observation] = []
    for index, raw in enumerate(raw_observations):
        if not isinstance(raw, Mapping):
            raise ValueError(f"observations[{index}] must be a mapping")
        station = raw.get("station")
        if (not isinstance(station, Sequence) or isinstance(station, (str, bytes))
                or len(station) != 2):
            raise ValueError(f"observations[{index}].station must contain two values")
        try:
            bearing = raw["bearing_deg"]
            observation = Observation(
                (float(station[0]), float(station[1])),
                float(bearing),
                float(raw.get("half_angle_deg", 1.0)),
            )
        except KeyError as error:
            raise ValueError(
                f"observations[{index}].bearing_deg is required"
            ) from error
        observations.append(observation)
    return observations


def _point_list(point):
    return [point[0], point[1]] if point is not None else None


def _pair_list(pair):
    return [_point_list(pair[0]), _point_list(pair[1])] if pair else None


def _empty_problem_result(diameter):
    return {
        "diameter_m": diameter,
        "diameter_exhaustive_m": diameter,
        "diameter_discrepancy_m": None,
        "farthest_pair": None,
        "diameter_circle": None,
        "minimum_enclosing_circle": None,
    }


def solve_case(payload: Mapping[str, object]) -> dict[str, object]:
    """Solve the pure bearing region and report separate control semantics."""

    if not isinstance(payload, Mapping):
        raise ValueError("case payload must be a mapping")
    observations = _parse_observations(payload)
    arena_radius = _positive_finite(payload, "arena_radius_m", 1800.0)
    clear_radius = _positive_finite(payload, "clear_radius_m", 20.0)
    decimals = _output_decimals(payload)

    halfplanes = [
        plane
        for observation in observations
        for plane in bearing_halfplanes(observation)
    ]
    region = intersect_halfplanes(halfplanes)
    bounded = region.status in {"point", "segment", "polygon"}
    vertices = list(region.vertices)
    arena_tolerance = 1e-12 * max(1.0, arena_radius)
    arena_contains_region = (
        all(math.hypot(*vertex) <= arena_radius + arena_tolerance
            for vertex in vertices)
        if bounded else None
    )

    region_result: dict[str, object] = {
        "status": region.status,
        "vertices": [_point_list(vertex) for vertex in vertices],
        "max_violation": region.max_violation,
        "area_m2": None,
        "centroid": None,
        "arena_contains_region": arena_contains_region,
    }
    if region.status == "polygon":
        area, centroid = polygon_area_centroid(vertices)
        region_result["area_m2"] = area
        region_result["centroid"] = _point_list(centroid)

    control_base: dict[str, object] = {
        "arena_radius_m": arena_radius,
        "clear_radius_m": clear_radius,
        "output_decimals": decimals,
        "output_center": None,
        "rounded_center_max_distance_m": None,
    }
    if region.status == "empty":
        return {
            "region": region_result,
            "problem_1": _empty_problem_result(None),
            "control": {**control_base, "status": "NO_FEASIBLE_REGION"},
        }
    if region.status == "unbounded":
        return {
            "region": region_result,
            "problem_1": _empty_problem_result("infinity"),
            "control": {**control_base, "status": "UNBOUNDED_REGION"},
        }

    exhaustive = diameter_exhaustive(vertices)
    calipers = diameter_rotating_calipers(vertices)
    discrepancy = abs(exhaustive.distance - calipers.distance)
    diameter_tolerance = 1e-10 * max(
        1.0, exhaustive.distance, calipers.distance
    )
    if discrepancy > diameter_tolerance:
        raise AssertionError(
            "rotating-calipers diameter disagrees with exhaustive diameter"
        )

    diameter_coverage = diameter_circle_coverage(vertices, calipers)
    enclosing = minimum_enclosing_circle(vertices)
    output_center = (
        round(enclosing.center[0], decimals),
        round(enclosing.center[1], decimals),
    )
    rounded_max_distance = max(
        (math.dist(output_center, vertex) for vertex in vertices), default=0.0
    )

    problem_result: dict[str, object] = {
        "diameter_m": calipers.distance,
        "diameter_exhaustive_m": exhaustive.distance,
        "diameter_discrepancy_m": discrepancy,
        "farthest_pair": _pair_list(calipers.witness),
        "diameter_circle": {
            "center": _point_list(diameter_coverage.center),
            "radius_m": diameter_coverage.radius,
            "max_distance_m": diameter_coverage.max_distance,
            "covers": diameter_coverage.covers,
        },
        "minimum_enclosing_circle": {
            "center": _point_list(enclosing.center),
            "radius_m": enclosing.radius,
            "max_residual_m": enclosing.max_residual,
        },
    }
    control: dict[str, object] = {
        **control_base,
        "output_center": _point_list(output_center),
        "rounded_center_max_distance_m": rounded_max_distance,
    }

    if not arena_contains_region:
        control.update(
            status="COVERAGE_UNCERTAIN", reason="arena_clipping_required"
        )
    else:
        squared_tolerance = 1e-12 * max(
            1.0, clear_radius * clear_radius,
            enclosing.radius * enclosing.radius,
            rounded_max_distance * rounded_max_distance,
        )
        clear_squared = clear_radius * clear_radius
        if enclosing.radius * enclosing.radius > clear_squared + squared_tolerance:
            control["status"] = "SINGLE_DISK_IMPOSSIBLE"
        elif rounded_max_distance * rounded_max_distance <= (
            clear_squared + squared_tolerance
        ):
            control["status"] = "CLEAR_READY"
        else:
            control.update(
                status="COVERAGE_UNCERTAIN",
                reason="rounded_center_exceeds_clear_radius",
            )

    return {
        "region": region_result,
        "problem_1": problem_result,
        "control": control,
    }
