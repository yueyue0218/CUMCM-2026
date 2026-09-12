"""Localization-region construction shared by Q1-Q4."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .geometry import (
    Point,
    circle_polygon,
    clip_polygon_to_bearing_wedge,
    clip_polygon_to_circle_outer,
    max_distance_to_region,
    minimum_enclosing_circle,
    polygon_diameter,
)


@dataclass(frozen=True)
class BearingObservation:
    station: Point
    bearing_deg: float
    error_deg: float = 1.0


class LocalizationStatus(str, Enum):
    CLEAR_READY = "CLEAR_READY"
    COVERAGE_UNCERTAIN = "COVERAGE_UNCERTAIN"
    MODEL_CONFLICT = "MODEL_CONFLICT"


@dataclass(frozen=True)
class LocalizationAssessment:
    """Clear assessment for an already constructed conservative outer region."""

    status: LocalizationStatus
    outer_region: tuple[Point, ...]
    diameter_m: float
    clear_position: Point | None
    r_max_m: float | None
    clear_ready: bool
    reason: str


def localization_region(
    observations: Sequence[BearingObservation],
    *,
    arena_radius_m: float = 1800.0,
    circle_vertices: int = 720,
    reception_radius_upper_m: float = 1500.0,
) -> list[Point]:
    """Build a conservative outer region from direction observations.

    Every valid direction observation supplies both its bearing wedge and the
    deterministic hard distance bound ``reception_radius_upper_m`` (1500 m by
    default).  The distance disk is represented by a circumscribed polygon.
    """

    region = circle_polygon(arena_radius_m, circle_vertices)
    for observation in observations:
        region = clip_polygon_to_bearing_wedge(
            region,
            observation.station,
            observation.bearing_deg,
            observation.error_deg,
        )
        if not region:
            break
        region = clip_polygon_to_circle_outer(
            region,
            observation.station,
            reception_radius_upper_m,
            circle_vertices,
        )
        if not region:
            break
    return region


def localization_quality(region: Sequence[Point]) -> dict[str, float]:
    """Return inexpensive geometry metrics suitable for candidate scoring."""

    if not region:
        return {"diameter_m": float("inf"), "area_m2": 0.0}
    twice_area = sum(
        region[i][0] * region[(i + 1) % len(region)][1]
        - region[(i + 1) % len(region)][0] * region[i][1]
        for i in range(len(region))
    )
    return {
        "diameter_m": polygon_diameter(region),
        "area_m2": abs(twice_area) / 2.0,
    }


def assess_region_for_clear(
    region: Sequence[Point],
    *,
    clear_radius_m: float = 20.0,
    clear_certification_margin_m: float = 1e-3,
) -> LocalizationAssessment:
    """Assess one candidate center for a conservative convex outer region.

    The minimum enclosing circle supplies the candidate center only.  The
    clear decision is made independently from the maximum vertex distance at
    that actual center.  A conservative numerical guard makes the float-based
    decision fail closed near the radius threshold.  This engineering margin
    is not a formal interval proof or a strict floating-point error bound.
    """

    if not math.isfinite(clear_radius_m) or clear_radius_m <= 0.0:
        raise ValueError("clear_radius_m must be finite and greater than zero")
    if (
        not math.isfinite(clear_certification_margin_m)
        or clear_certification_margin_m < 0.0
    ):
        raise ValueError(
            "clear_certification_margin_m must be finite and non-negative"
        )
    if clear_certification_margin_m >= clear_radius_m:
        raise ValueError(
            "clear_certification_margin_m must be less than clear_radius_m"
        )

    outer_region = tuple(region)
    if not outer_region:
        return LocalizationAssessment(
            status=LocalizationStatus.MODEL_CONFLICT,
            outer_region=outer_region,
            diameter_m=float("inf"),
            clear_position=None,
            r_max_m=None,
            clear_ready=False,
            reason="conservative outer region is empty",
        )

    diameter_m = polygon_diameter(outer_region)
    clear_position = minimum_enclosing_circle(outer_region).center
    r_max_m = max_distance_to_region(outer_region, clear_position)
    certification_threshold_m = clear_radius_m - clear_certification_margin_m
    clear_ready = r_max_m <= certification_threshold_m
    if clear_ready:
        status = LocalizationStatus.CLEAR_READY
        reason = (
            "candidate center covers the conservative outer region with the "
            "fail-closed engineering margin"
        )
    else:
        status = LocalizationStatus.COVERAGE_UNCERTAIN
        if r_max_m <= clear_radius_m:
            reason = (
                "candidate center is within the physical clear radius but lies in "
                "the conservative numerical guard band, so this float-based check "
                "fails closed; this is not formal interval certification"
            )
        else:
            reason = (
                "candidate center exceeds the physical clear radius for the "
                "conservative outer region; this does not prove that the true "
                "feasible set cannot be covered"
            )
    return LocalizationAssessment(
        status=status,
        outer_region=outer_region,
        diameter_m=diameter_m,
        clear_position=clear_position,
        r_max_m=r_max_m,
        clear_ready=clear_ready,
        reason=reason,
    )


def assess_observations_for_clear(
    observations: Sequence[BearingObservation],
    *,
    arena_radius_m: float = 1800.0,
    circle_vertices: int = 720,
    reception_radius_upper_m: float = 1500.0,
    clear_radius_m: float = 20.0,
    clear_certification_margin_m: float = 1e-3,
) -> LocalizationAssessment:
    """Build a conservative direction region and assess its clear candidate."""

    region = localization_region(
        observations,
        arena_radius_m=arena_radius_m,
        circle_vertices=circle_vertices,
        reception_radius_upper_m=reception_radius_upper_m,
    )
    return assess_region_for_clear(
        region,
        clear_radius_m=clear_radius_m,
        clear_certification_margin_m=clear_certification_margin_m,
    )
