"""Stress-test circle discretization at the 20 m clear threshold.

Run from the repository root:
    python src/q1/verify_clear_threshold_discretization.py

Writes results/tables/q1_clear_threshold_discretization.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.geometry import circle_polygon  # noqa: E402
from src.common.localization import assess_region_for_clear  # noqa: E402


TRUE_RADII_M = (19.95, 19.98, 19.99, 19.999, 20.0, 20.001, 20.01, 20.02, 20.05)
CIRCLE_VERTEX_COUNTS = (180, 360, 720, 1440)
CLEAR_RADIUS_M = 20.0
CLEAR_CERTIFICATION_MARGIN_M = 1e-3
ANALYTIC_ABS_TOL_M = 1e-10


def analytic_outer_vertex_radius(radius_m: float, vertex_count: int) -> float:
    """Return the circumradius of the circumscribed regular polygon."""

    return radius_m / math.cos(math.pi / vertex_count)


def certification_summary(vertex_count: int) -> dict[str, object]:
    """Return the analytic true-radius boundary and its conservative split."""

    certification_threshold_m = CLEAR_RADIUS_M - CLEAR_CERTIFICATION_MARGIN_M
    cosine = math.cos(math.pi / vertex_count)
    max_true_radius_certifiable_m = certification_threshold_m * cosine
    polygon_outer_packaging_loss_m = certification_threshold_m * (1.0 - cosine)
    total_conservative_loss_m = CLEAR_RADIUS_M - max_true_radius_certifiable_m
    return {
        "circle_vertices": vertex_count,
        "certification_rule": "r_max_m <= 19.999",
        "max_true_radius_certifiable_analytic_m": max_true_radius_certifiable_m,
        "coverage_uncertain_for_true_radius_analytic": (
            f"r > {max_true_radius_certifiable_m:.15g} m"
        ),
        "loss_from_certification_margin_m": CLEAR_CERTIFICATION_MARGIN_M,
        "loss_from_polygon_outer_approximation_m": polygon_outer_packaging_loss_m,
        "total_true_radius_loss_vs_20_m": total_conservative_loss_m,
    }


def measure_case(radius_m: float, vertex_count: int) -> dict[str, object]:
    """Run one analytic disk through the production construction and assessment."""

    analytic_radius_m = analytic_outer_vertex_radius(radius_m, vertex_count)
    region = circle_polygon(radius_m, vertex_count)
    assessment = assess_region_for_clear(
        region,
        clear_radius_m=CLEAR_RADIUS_M,
        clear_certification_margin_m=CLEAR_CERTIFICATION_MARGIN_M,
    )
    if assessment.r_max_m is None:
        raise AssertionError("non-empty circle polygon unexpectedly has no r_max")

    analytic_error_m = assessment.r_max_m - analytic_radius_m
    if not math.isclose(
        assessment.r_max_m,
        analytic_radius_m,
        rel_tol=0.0,
        abs_tol=ANALYTIC_ABS_TOL_M,
    ):
        raise AssertionError(
            f"r_max mismatch for r={radius_m}, n={vertex_count}: "
            f"observed {assessment.r_max_m}, analytic {analytic_radius_m}"
        )

    true_disk_within_physical_clear_radius = radius_m <= CLEAR_RADIUS_M
    false_negative_conservative = (
        true_disk_within_physical_clear_radius and not assessment.clear_ready
    )
    erroneous_clear_ready_above_physical_radius = (
        radius_m > CLEAR_RADIUS_M and assessment.clear_ready
    )
    return {
        "true_radius_m": radius_m,
        "circle_vertices": vertex_count,
        "analytic_outer_vertex_radius": analytic_radius_m,
        "r_max_m": assessment.r_max_m,
        "status": assessment.status.value,
        "clear_ready": assessment.clear_ready,
        "conservative_increase_vs_true_radius_m": analytic_radius_m - radius_m,
        "r_max_minus_analytic_m": analytic_error_m,
        "true_disk_satisfies_r_le_20": true_disk_within_physical_clear_radius,
        "engineering_certification_satisfies_r_max_le_19_999": (
            assessment.r_max_m
            <= CLEAR_RADIUS_M - CLEAR_CERTIFICATION_MARGIN_M
        ),
        "true_clearable_but_coverage_uncertain": false_negative_conservative,
        "erroneous_clear_ready_for_true_r_gt_20": (
            erroneous_clear_ready_above_physical_radius
        ),
    }


def main() -> None:
    rows = [
        measure_case(radius_m, vertex_count)
        for radius_m in TRUE_RADII_M
        for vertex_count in CIRCLE_VERTEX_COUNTS
    ]
    if any(row["erroneous_clear_ready_for_true_r_gt_20"] for row in rows):
        raise AssertionError("a true radius above 20 m was incorrectly CLEAR_READY")

    summaries = [certification_summary(n) for n in CIRCLE_VERTEX_COUNTS]
    for summary in summaries:
        vertex_count = summary["circle_vertices"]
        cases = [row for row in rows if row["circle_vertices"] == vertex_count]
        uncertain_true_clearable = [
            row["true_radius_m"]
            for row in cases
            if row["true_clearable_but_coverage_uncertain"]
        ]
        clear_radii = [row["true_radius_m"] for row in cases if row["clear_ready"]]
        summary["largest_tested_clear_ready_true_radius_m"] = (
            max(clear_radii) if clear_radii else None
        )
        summary["first_tested_true_clearable_coverage_uncertain_radius_m"] = (
            min(uncertain_true_clearable) if uncertain_true_clearable else None
        )

    report = {
        "purpose": (
            "Numerical pressure test of conservative circle discretization near "
            "the 20 m CLEAR_READY threshold; not a formal floating-point proof."
        ),
        "configuration": {
            "true_radii_m": list(TRUE_RADII_M),
            "circle_vertex_counts": list(CIRCLE_VERTEX_COUNTS),
            "clear_radius_m": CLEAR_RADIUS_M,
            "clear_certification_margin_m": CLEAR_CERTIFICATION_MARGIN_M,
            "effective_r_max_threshold_m": (
                CLEAR_RADIUS_M - CLEAR_CERTIFICATION_MARGIN_M
            ),
            "analytic_r_max_check_abs_tolerance_m": ANALYTIC_ABS_TOL_M,
        },
        "interpretation": {
            "physical_truth": "the true disk is clearable exactly when r <= 20 m",
            "engineering_certificate": (
                "the current implementation returns CLEAR_READY only when "
                "r_max <= 19.999 m"
            ),
            "conservative_false_negative": (
                "r <= 20 m but the outer polygon or 1 mm guard makes the result "
                "COVERAGE_UNCERTAIN; this is conservative behavior, not an "
                "algorithm error"
            ),
        },
        "analytic_transition_by_vertex_count": summaries,
        "cases": rows,
        "checks": {
            "all_r_max_match_analytic_outer_radius": True,
            "maximum_absolute_r_max_error_m": max(
                abs(row["r_max_minus_analytic_m"]) for row in rows
            ),
            "any_true_r_gt_20_clear_ready": any(
                row["erroneous_clear_ready_for_true_r_gt_20"] for row in rows
            ),
            "conservative_false_negative_case_count": sum(
                bool(row["true_clearable_but_coverage_uncertain"])
                for row in rows
            ),
        },
    }

    output_path = (
        ROOT / "results" / "tables" / "q1_clear_threshold_discretization.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
