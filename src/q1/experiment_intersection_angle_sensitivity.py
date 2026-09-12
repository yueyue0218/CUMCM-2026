"""Run the Q1 pure-bearing intersection-angle sensitivity experiment.

Run from the repository root:
    python src/q1/experiment_intersection_angle_sensitivity.py

Writes results/tables/q1_intersection_angle_sensitivity.json.  The primary
experiment clips a finite box by the two production bearing-wedge operations;
it deliberately does not add the arena or reception-radius constraints.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.geometry import (  # noqa: E402
    Point,
    clip_polygon_to_bearing_wedge,
    minimum_enclosing_circle,
    polygon_diameter,
)
from src.common.localization import BearingObservation, localization_region  # noqa: E402


ALPHA_DEGREES = (5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0)
STATION_DISTANCE_M = 500.0
BEARING_ERROR_DEG = 1.0
INITIAL_BOX_HALF_EXTENT_M = 10_000.0
BOX_EXPANSION_FACTOR = 2.0
MAX_BOX_EXPANSIONS = 12
BOUNDARY_MARGIN_REL = 1e-6
SANITY_ABS_TOL_M = 1e-8
PRODUCTION_DIAMETER_ABS_TOL_M = 1e-8


def square_box(half_extent_m: float) -> list[Point]:
    """Return a counter-clockwise square centered at the origin."""

    return [
        (-half_extent_m, -half_extent_m),
        (half_extent_m, -half_extent_m),
        (half_extent_m, half_extent_m),
        (-half_extent_m, half_extent_m),
    ]


def polygon_area(points: Sequence[Point]) -> float:
    """Return polygon area using the standard shoelace formula."""

    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
    ) / 2.0


def experiment_geometry(alpha_deg: float) -> tuple[Point, Point, float, float]:
    """Return the two stations and center bearings for one alpha."""

    beta_deg = alpha_deg / 2.0
    beta_rad = math.radians(beta_deg)
    station_x = -STATION_DISTANCE_M * math.cos(beta_rad)
    station_y = STATION_DISTANCE_M * math.sin(beta_rad)
    return (station_x, station_y), (station_x, -station_y), -beta_deg, beta_deg


def pure_wedge_region(alpha_deg: float, box_half_extent_m: float) -> list[Point]:
    """Construct K_alpha from only the two production bearing wedges."""

    station_1, station_2, theta_1, theta_2 = experiment_geometry(alpha_deg)
    region = square_box(box_half_extent_m)
    region = clip_polygon_to_bearing_wedge(
        region, station_1, theta_1, BEARING_ERROR_DEG
    )
    return clip_polygon_to_bearing_wedge(
        region, station_2, theta_2, BEARING_ERROR_DEG
    )


def boundary_is_active(region: Sequence[Point], half_extent_m: float) -> bool:
    """Return whether a result vertex is within the prescribed box margin."""

    margin_m = half_extent_m * BOUNDARY_MARGIN_REL
    return any(
        half_extent_m - max(abs(point[0]), abs(point[1])) <= margin_m
        for point in region
    )


def choose_inactive_box() -> tuple[float, int]:
    """Expand the initial box until no experiment result touches its boundary."""

    half_extent_m = INITIAL_BOX_HALF_EXTENT_M
    for expansion_count in range(MAX_BOX_EXPANSIONS + 1):
        regions = [pure_wedge_region(alpha, half_extent_m) for alpha in ALPHA_DEGREES]
        if all(region for region in regions) and not any(
            boundary_is_active(region, half_extent_m) for region in regions
        ):
            return half_extent_m, expansion_count
        half_extent_m *= BOX_EXPANSION_FACTOR
    raise AssertionError("failed to find an inactive finite initial box")


def analytic_axis_intersections(alpha_deg: float) -> tuple[float, float]:
    """Return the two symmetric-boundary intersections on the x axis."""

    beta_deg = alpha_deg / 2.0
    beta_rad = math.radians(beta_deg)
    common = -STATION_DISTANCE_M * math.cos(beta_rad)
    numerator = STATION_DISTANCE_M * math.sin(beta_rad)
    left = common + numerator / math.tan(
        math.radians(beta_deg + BEARING_ERROR_DEG)
    )
    right = common + numerator / math.tan(
        math.radians(beta_deg - BEARING_ERROR_DEG)
    )
    return left, right


def make_row(alpha_deg: float, box_half_extent_m: float) -> dict[str, object]:
    """Compute primary, analytic, and auxiliary production-path results."""

    beta_deg = alpha_deg / 2.0
    station_1, station_2, theta_1, theta_2 = experiment_geometry(alpha_deg)
    region = pure_wedge_region(alpha_deg, box_half_extent_m)
    if not region:
        raise AssertionError(f"empty pure-wedge region for alpha={alpha_deg}")
    if boundary_is_active(region, box_half_extent_m):
        raise AssertionError(f"initial box is active for alpha={alpha_deg}")

    diameter_m = polygon_diameter(region)
    mec = minimum_enclosing_circle(region)
    area_m2 = polygon_area(region)
    mec_over_half_d = mec.radius / (diameter_m / 2.0)

    analytic_left_m, analytic_right_m = analytic_axis_intersections(alpha_deg)
    numerical_left = min(region, key=lambda point: point[0])
    numerical_right = max(region, key=lambda point: point[0])
    numerical_x_min_m = numerical_left[0]
    numerical_x_max_m = numerical_right[0]
    left_error_m = numerical_x_min_m - analytic_left_m
    right_error_m = numerical_x_max_m - analytic_right_m
    sanity_passed = (
        math.isclose(
            numerical_x_min_m,
            analytic_left_m,
            rel_tol=0.0,
            abs_tol=SANITY_ABS_TOL_M,
        )
        and math.isclose(
            numerical_x_max_m,
            analytic_right_m,
            rel_tol=0.0,
            abs_tol=SANITY_ABS_TOL_M,
        )
        and analytic_left_m < analytic_right_m
        and abs(numerical_left[1]) <= SANITY_ABS_TOL_M
        and abs(numerical_right[1]) <= SANITY_ABS_TOL_M
    )
    if not sanity_passed:
        raise AssertionError(
            f"analytic x-axis sanity check failed for alpha={alpha_deg}: "
            f"numeric=({numerical_x_min_m}, {numerical_x_max_m}), "
            f"analytic=({analytic_left_m}, {analytic_right_m})"
        )

    observations = [
        BearingObservation(station_1, theta_1, BEARING_ERROR_DEG),
        BearingObservation(station_2, theta_2, BEARING_ERROR_DEG),
    ]
    production_region = localization_region(observations)
    production_diameter_m = polygon_diameter(production_region)
    production_delta_m = production_diameter_m - diameter_m
    max_arena_radius_m = max(math.hypot(*point) for point in region)
    max_reception_distance_m = max(
        math.dist(point, station)
        for point in region
        for station in (station_1, station_2)
    )
    physical_constraints_contain_pure_wedge = (
        max_arena_radius_m < 1800.0 and max_reception_distance_m < 1500.0
    )
    production_diameter_matches = math.isclose(
        production_diameter_m, diameter_m, rel_tol=0.0,
        abs_tol=PRODUCTION_DIAMETER_ABS_TOL_M,
    )
    production_boundary_active = not (
        physical_constraints_contain_pure_wedge and production_diameter_matches
    )

    return {
        "alpha_deg": alpha_deg,
        "beta_deg": beta_deg,
        "station_distance_m": STATION_DISTANCE_M,
        "station_baseline_m": 2.0
        * STATION_DISTANCE_M
        * math.sin(math.radians(alpha_deg / 2.0)),
        "region_vertex_count": len(region),
        "diameter_m": diameter_m,
        "area_m2": area_m2,
        "mec_radius_m": mec.radius,
        "mec_over_half_d": mec_over_half_d,
        "mec_le_20": mec.radius <= 20.0,
        "boundary_active": False,
        "analytic_sanity_check": {
            "x_left_m": analytic_left_m,
            "x_right_m": analytic_right_m,
            "numerical_x_min_m": numerical_x_min_m,
            "numerical_x_max_m": numerical_x_max_m,
            "numerical_x_min_y_m": numerical_left[1],
            "numerical_x_max_y_m": numerical_right[1],
            "x_left_error_m": left_error_m,
            "x_right_error_m": right_error_m,
            "passed": sanity_passed,
        },
        "production_path_check": {
            "region_vertex_count": len(production_region),
            "diameter_m": production_diameter_m,
            "diameter_minus_pure_wedge_m": production_delta_m,
            "diameter_matches_within_tolerance": production_diameter_matches,
            "pure_wedge_max_distance_from_origin_m": max_arena_radius_m,
            "pure_wedge_max_distance_from_either_station_m": (
                max_reception_distance_m
            ),
            "physical_constraints_contain_pure_wedge": (
                physical_constraints_contain_pure_wedge
            ),
            "physical_boundary_active": production_boundary_active,
        },
    }


def increasing_steps(rows: Sequence[dict[str, object]], key: str) -> list[dict[str, float]]:
    """Return consecutive increases as alpha rises (potential non-monotonicity)."""

    increases: list[dict[str, float]] = []
    for previous, current in zip(rows, rows[1:]):
        previous_value = float(previous[key])
        current_value = float(current[key])
        if current_value > previous_value:
            increases.append(
                {
                    "from_alpha_deg": float(previous["alpha_deg"]),
                    "to_alpha_deg": float(current["alpha_deg"]),
                    "increase": current_value - previous_value,
                }
            )
    return increases


def main() -> None:
    box_half_extent_m, expansion_count = choose_inactive_box()
    rows = [make_row(alpha, box_half_extent_m) for alpha in ALPHA_DEGREES]

    monotonicity = {
        key: {
            "nonincreasing": not (steps := increasing_steps(rows, key)),
            "increasing_steps": steps,
        }
        for key in ("diameter_m", "area_m2", "mec_radius_m", "mec_over_half_d")
    }
    report = {
        "experiment": (
            "Pure bearing-wedge intersection sensitivity at fixed 500 m "
            "station-source distance; alpha is the sole geometry parameter and "
            "the station baseline is therefore 2 L sin(alpha/2), not constant."
        ),
        "configuration": {
            "source_m": [0.0, 0.0],
            "station_distance_m": STATION_DISTANCE_M,
            "bearing_error_deg": BEARING_ERROR_DEG,
            "alpha_degrees": list(ALPHA_DEGREES),
            "initial_box": {
                "shape": "axis-aligned square centered at the origin",
                "initial_half_extent_m": INITIAL_BOX_HALF_EXTENT_M,
                "final_half_extent_m": box_half_extent_m,
                "full_width_m": 2.0 * box_half_extent_m,
                "boundary_margin_relative": BOUNDARY_MARGIN_REL,
                "boundary_margin_m": box_half_extent_m * BOUNDARY_MARGIN_REL,
                "expansion_factor": BOX_EXPANSION_FACTOR,
                "expansion_count": expansion_count,
                "boundary_active": False,
            },
            "analytic_sanity_abs_tolerance_m": SANITY_ABS_TOL_M,
            "production_diameter_abs_tolerance_m": (
                PRODUCTION_DIAMETER_ABS_TOL_M
            ),
        },
        "rows": rows,
        "checks": {
            "all_initial_box_boundaries_inactive": all(
                not bool(row["boundary_active"]) for row in rows
            ),
            "all_analytic_sanity_checks_passed": all(
                bool(row["analytic_sanity_check"]["passed"])  # type: ignore[index]
                for row in rows
            ),
            "production_physical_boundaries_active_at_alpha_deg": [
                row["alpha_deg"]
                for row in rows
                if row["production_path_check"]["physical_boundary_active"]  # type: ignore[index]
            ],
            "monotonicity_as_alpha_increases": monotonicity,
        },
    }

    output_path = (
        ROOT / "results" / "tables" / "q1_intersection_angle_sensitivity.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
