"""Common Q3/Q4 run metrics."""

from __future__ import annotations


def clearance_ratio(cleared_count: int, total_count: int) -> float:
    if cleared_count < 0 or total_count <= 0 or cleared_count > total_count:
        raise ValueError("counts must satisfy 0 <= cleared_count <= total_count")
    return cleared_count / total_count


def average_localization_clearance_time(
    localization_clearance_time_s: float, cleared_count: int
) -> float | None:
    if localization_clearance_time_s < 0 or cleared_count < 0:
        raise ValueError("time and count must be non-negative")
    return None if cleared_count == 0 else localization_clearance_time_s / cleared_count


def run_summary(
    *,
    cleared_count: int,
    total_count: int | None,
    localization_clearance_time_s: float,
    program_runtime_s: float,
    all_cleared: bool | None = None,
    failure_reason: str | None = None,
) -> dict[str, object]:
    """Build the stable summary schema used by practice and formal runs."""

    return {
        "all_cleared": all_cleared,
        "cleared_count": cleared_count,
        "total_count": total_count,
        "clearance_ratio": (
            clearance_ratio(cleared_count, total_count) if total_count else None
        ),
        "average_localization_clearance_time_s": (
            average_localization_clearance_time(
                localization_clearance_time_s, cleared_count
            )
        ),
        "program_runtime_s": program_runtime_s,
        "failure_reason": failure_reason,
    }
