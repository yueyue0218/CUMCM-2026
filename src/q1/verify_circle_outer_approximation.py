"""Quantify circle outer-polygon error through the production Q1 path.

Run from the repository root:
    python src/q1/verify_circle_outer_approximation.py

Writes results/tables/q1_circle_outer_approximation.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.geometry import polygon_diameter  # noqa: E402
from src.common.localization import (  # noqa: E402
    BearingObservation,
    assess_region_for_clear,
    localization_quality,
    localization_region,
)


CIRCLE_VERTEX_COUNTS = (180, 360, 720, 1440)
RADII_M = (1800.0, 1500.0)
REFERENCE_VERTEX_COUNT = 1440


def radial_excess(radius_m: float, vertex_count: int) -> float:
    """Return the analytic circumradius excess of the outer polygon."""

    return radius_m * (1.0 / math.cos(math.pi / vertex_count) - 1.0)


def production_measurement(vertex_count: int) -> dict[str, object]:
    """Measure one fixed reception-circle-clipped bearing sector."""

    observations = [BearingObservation((0.0, 0.0), 0.0, 1.0)]
    region = localization_region(observations, circle_vertices=vertex_count)
    quality = localization_quality(region)
    assessment = assess_region_for_clear(region)
    return {
        "circle_vertices": vertex_count,
        "region_vertex_count": len(region),
        "diameter_m": polygon_diameter(region),
        "area_m2": quality["area_m2"],
        "clear_position_m": assessment.clear_position,
        "r_max_m": assessment.r_max_m,
        "status": assessment.status.value,
        "clear_ready": assessment.clear_ready,
    }


def main() -> None:
    analytic = {
        str(int(radius_m)): {
            str(vertex_count): radial_excess(radius_m, vertex_count)
            for vertex_count in CIRCLE_VERTEX_COUNTS
        }
        for radius_m in RADII_M
    }

    measurements = [
        production_measurement(vertex_count)
        for vertex_count in CIRCLE_VERTEX_COUNTS
    ]
    reference = next(
        item
        for item in measurements
        if item["circle_vertices"] == REFERENCE_VERTEX_COUNT
    )
    for item in measurements:
        item["delta_vs_1440"] = {
            "region_vertex_count": (
                item["region_vertex_count"] - reference["region_vertex_count"]
            ),
            "diameter_m": item["diameter_m"] - reference["diameter_m"],
            "area_m2": item["area_m2"] - reference["area_m2"],
            "r_max_m": item["r_max_m"] - reference["r_max_m"],
        }
        item["status_changed_vs_1440"] = item["status"] != reference["status"]

    report = {
        "analytic_radial_excess_m": analytic,
        "production_case": {
            "observations": [
                {"station_m": [0.0, 0.0], "bearing_deg": 0.0, "error_deg": 1.0}
            ],
            "arena_radius_m": 1800.0,
            "reception_radius_upper_m": 1500.0,
            "description": (
                "A nondegenerate bearing wedge actually clipped by the 1500 m "
                "reception-circle outer polygon; it is far from the 20 m clear scale."
            ),
            "reference_circle_vertices": REFERENCE_VERTEX_COUNT,
            "measurements": measurements,
        },
    }

    output_path = ROOT / "results" / "tables" / "q1_circle_outer_approximation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
