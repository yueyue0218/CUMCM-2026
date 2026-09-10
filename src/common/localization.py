"""Localization-region construction shared by Q1-Q4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .geometry import (
    Point,
    circle_polygon,
    clip_polygon_to_bearing_wedge,
    polygon_diameter,
)


@dataclass(frozen=True)
class BearingObservation:
    station: Point
    bearing_deg: float
    error_deg: float = 1.0


def localization_region(
    observations: Sequence[BearingObservation],
    *,
    arena_radius_m: float = 1800.0,
    circle_vertices: int = 720,
) -> list[Point]:
    """Intersect all measurement wedges with the circular target region."""

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
