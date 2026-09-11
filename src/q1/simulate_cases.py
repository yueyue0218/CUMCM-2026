"""Deterministic observation simulations and boundary-result reports for Q1."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

from src.q1.enclosing_circle import minimum_enclosing_circle
from src.q1.geometry import (
    HalfPlane,
    Observation,
    bearing_halfplanes,
    intersect_halfplanes,
    point_satisfies,
)
from src.q1.measures import (
    CircularArc,
    LineSegment,
    diameter_circle_coverage,
    diameter_rotating_calipers,
    green_area_centroid,
)
from src.q1.solver import solve_case


DEFAULT_SEED = 20260911
OBSERVATION_BOUNDARY_IDS = (
    "cross_zero",
    "error_at_positive_bound",
    "error_at_negative_bound",
    "unbounded_single_observation",
    "empty_conflicting_observations",
    "duplicate_observations",
    "near_parallel_intersection",
    "source_on_arena_boundary",
)
ANALYTIC_BOUNDARY_IDS = (
    "point_region",
    "segment_region",
    "equilateral_diameter_circle_failure",
    "square_diameter_circle_success",
    "clear_radius_exactly_20",
    "clear_radius_above_20",
    "arc_crosses_zero",
    "rounded_center_counterexample",
)


def _bearing_to(station: Sequence[float], source: Sequence[float]) -> float:
    return math.degrees(
        math.atan2(source[1] - station[1], source[0] - station[0])
    ) % 360.0


def _observation(
    station: Sequence[float], source: Sequence[float], error_deg: float
) -> dict[str, object]:
    return {
        "station": [float(station[0]), float(station[1])],
        "bearing_deg": (_bearing_to(station, source) + error_deg) % 360.0,
        "half_angle_deg": 1.0,
        "sampled_error_deg": float(error_deg),
    }


def _stations_around(
    source: Sequence[float], distance: float = 500.0
) -> list[list[float]]:
    x, y = source
    return [
        [x - distance, y],
        [x + distance, y],
        [x, y - distance],
        [x, y + distance],
    ]


def _consistent_case(
    case_id: str,
    case_kind: str,
    source: Sequence[float],
    stations: Sequence[Sequence[float]],
    errors: Sequence[float],
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "case_kind": case_kind,
        "true_source": [float(source[0]), float(source[1])],
        "expected_truth_retained": True,
        "observations": [
            _observation(station, source, error)
            for station, error in zip(stations, errors, strict=True)
        ],
    }


def _observation_boundary_cases() -> list[dict[str, object]]:
    origin = [0.0, 0.0]
    cross_zero_stations = [[-500.0, 0.0], [500.0, 0.0], [0.0, -500.0]]
    positive_source = [125.0, -240.0]
    positive_stations = _stations_around(positive_source)[:3]
    negative_source = [-360.0, 175.0]
    negative_stations = _stations_around(negative_source)[:3]
    duplicate_stations = _stations_around(origin)
    duplicate_observations = [
        _observation(station, origin, 0.0) for station in duplicate_stations
    ]
    duplicate_observations.append(dict(duplicate_observations[0]))

    near_source = [80.0, -60.0]
    near_station_angles = [0.0, math.pi, math.pi + math.radians(0.5)]
    near_distances = [1200.0, 1000.0, 700.0]
    near_stations = [
        [
            near_source[0] + distance * math.cos(angle),
            near_source[1] + distance * math.sin(angle),
        ]
        for angle, distance in zip(near_station_angles, near_distances, strict=True)
    ]

    arena_source = [1800.0, 0.0]
    return [
        _consistent_case(
            "cross_zero",
            "observation_boundary",
            origin,
            cross_zero_stations,
            [-0.5, 0.0, 0.0],
        ),
        _consistent_case(
            "error_at_positive_bound",
            "observation_boundary",
            positive_source,
            positive_stations,
            [1.0, 0.25, -0.25],
        ),
        _consistent_case(
            "error_at_negative_bound",
            "observation_boundary",
            negative_source,
            negative_stations,
            [-1.0, -0.25, 0.25],
        ),
        {
            "case_id": "unbounded_single_observation",
            "case_kind": "observation_boundary",
            "true_source": [100.0, 100.0],
            "expected_truth_retained": True,
            "observations": [
                _observation([0.0, 0.0], [100.0, 100.0], 0.0)
            ],
        },
        {
            "case_id": "empty_conflicting_observations",
            "case_kind": "observation_boundary",
            "observations": [
                {"station": [0.0, 0.0], "bearing_deg": 180.0},
                {"station": [1.0, 0.0], "bearing_deg": 0.0},
            ],
        },
        {
            "case_id": "duplicate_observations",
            "case_kind": "observation_boundary",
            "true_source": origin,
            "expected_truth_retained": True,
            "observations": duplicate_observations,
        },
        _consistent_case(
            "near_parallel_intersection",
            "observation_boundary",
            near_source,
            near_stations,
            [0.0, 0.0, 0.0],
        ),
        _consistent_case(
            "source_on_arena_boundary",
            "observation_boundary",
            arena_source,
            _stations_around(arena_source, 400.0),
            [0.0, 0.0, 0.0, 0.0],
        ),
    ]


def _random_source(generator: random.Random) -> list[float]:
    radius = 1800.0 * math.sqrt(generator.random())
    angle = generator.uniform(0.0, 2.0 * math.pi)
    return [radius * math.cos(angle), radius * math.sin(angle)]


def _random_case(
    generator: random.Random, case_kind: str, case_number: int
) -> dict[str, object]:
    source = _random_source(generator)
    base_angle = generator.uniform(0.0, 2.0 * math.pi)
    if case_kind == "regular_random":
        station_angles = [
            base_angle
            + index * 2.0 * math.pi / 3.0
            + generator.uniform(-math.pi / 12.0, math.pi / 12.0)
            for index in range(3)
        ]
    else:
        offsets = (0.0, math.pi, math.pi + math.radians(0.5))
        station_angles = [
            base_angle + offset + generator.uniform(-math.radians(0.1), math.radians(0.1))
            for offset in offsets
        ]
    distances = [generator.uniform(200.0, 1400.0) for _ in range(3)]
    stations = [
        [
            source[0] + distance * math.cos(angle),
            source[1] + distance * math.sin(angle),
        ]
        for angle, distance in zip(station_angles, distances, strict=True)
    ]
    errors = [generator.uniform(-1.0, 1.0) for _ in range(3)]
    prefix = "regular" if case_kind == "regular_random" else "near_parallel"
    return _consistent_case(
        f"{prefix}_{case_number:03d}",
        case_kind,
        source,
        stations,
        errors,
    )


def generate_cases(
    seed: int = DEFAULT_SEED,
    regular_count: int = 50,
    near_parallel_count: int = 20,
) -> list[dict[str, object]]:
    """Return boundary fixtures followed by reproducible random observations."""

    for name, count in (
        ("regular_count", regular_count),
        ("near_parallel_count", near_parallel_count),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    generator = random.Random(seed)
    cases = _observation_boundary_cases()
    cases.extend(
        _random_case(generator, "regular_random", index + 1)
        for index in range(regular_count)
    )
    cases.extend(
        _random_case(generator, "near_parallel_random", index + 1)
        for index in range(near_parallel_count)
    )
    for case in cases:
        case["simulation_seed"] = seed
    return cases


def _truth_retained(case: Mapping[str, object]) -> bool:
    source = case["true_source"]
    observations = case["observations"]
    planes = []
    for raw in observations:
        observation = Observation(
            (float(raw["station"][0]), float(raw["station"][1])),
            float(raw["bearing_deg"]),
            float(raw.get("half_angle_deg", 1.0)),
        )
        planes.extend(bearing_halfplanes(observation))
    return point_satisfies(
        (float(source[0]), float(source[1])), planes, tolerance=1e-8
    )


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {
            "minimum": None,
            "q25": None,
            "median": None,
            "q75": None,
            "maximum": None,
        }
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "minimum": ordered[0],
        "q25": percentile(0.25),
        "median": percentile(0.5),
        "q75": percentile(0.75),
        "maximum": ordered[-1],
    }


def _check(
    case_id: str,
    kind: str,
    parameters: Mapping[str, object],
    expected: Mapping[str, object],
    actual: Mapping[str, object],
    passed: bool | None = None,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "kind": kind,
        "parameters": dict(parameters),
        "expected": dict(expected),
        "actual": dict(actual),
        "pass": dict(actual) == dict(expected) if passed is None else passed,
    }


def _observation_boundary_check(
    case: Mapping[str, object], result: Mapping[str, object]
) -> dict[str, object]:
    case_id = str(case["case_id"])
    status = result["region"]["status"]
    parameters = {
        "observation_count": len(case["observations"]),
        "true_source": case.get("true_source"),
        "sampled_errors_deg": [
            observation.get("sampled_error_deg")
            for observation in case["observations"]
            if "sampled_error_deg" in observation
        ],
    }
    if case_id == "cross_zero":
        expected = {"region_status": "polygon", "truth_retained": True, "crosses_zero": True}
        bearings = [float(item["bearing_deg"]) for item in case["observations"]]
        actual = {
            "region_status": status,
            "truth_retained": _truth_retained(case),
            "crosses_zero": any(bearing > 359.0 for bearing in bearings),
        }
    elif case_id == "error_at_positive_bound":
        expected = {"region_status": "polygon", "truth_retained": True, "sampled_error_deg": 1.0}
        actual = {
            "region_status": status,
            "truth_retained": _truth_retained(case),
            "sampled_error_deg": case["observations"][0]["sampled_error_deg"],
        }
    elif case_id == "error_at_negative_bound":
        expected = {"region_status": "polygon", "truth_retained": True, "sampled_error_deg": -1.0}
        actual = {
            "region_status": status,
            "truth_retained": _truth_retained(case),
            "sampled_error_deg": case["observations"][0]["sampled_error_deg"],
        }
    elif case_id == "unbounded_single_observation":
        expected = {"region_status": "unbounded", "truth_retained": True}
        actual = {"region_status": status, "truth_retained": _truth_retained(case)}
    elif case_id == "empty_conflicting_observations":
        expected = {"region_status": "empty"}
        actual = {"region_status": status}
    elif case_id == "duplicate_observations":
        observations = case["observations"]
        keys = [
            (
                tuple(item["station"]),
                item["bearing_deg"],
                item.get("half_angle_deg", 1.0),
            )
            for item in observations
        ]
        expected = {"region_status": "polygon", "truth_retained": True, "has_duplicate": True}
        actual = {
            "region_status": status,
            "truth_retained": _truth_retained(case),
            "has_duplicate": len(keys) != len(set(keys)),
        }
    elif case_id == "near_parallel_intersection":
        expected = {"region_status": "polygon", "truth_retained": True}
        actual = {"region_status": status, "truth_retained": _truth_retained(case)}
    elif case_id == "source_on_arena_boundary":
        source = case["true_source"]
        expected = {"region_status": "polygon", "truth_retained": True, "source_radius_m": 1800.0}
        actual = {
            "region_status": status,
            "truth_retained": _truth_retained(case),
            "source_radius_m": math.hypot(float(source[0]), float(source[1])),
        }
    else:
        raise ValueError(f"unknown observation boundary case: {case_id}")
    return _check(case_id, "observation", parameters, expected, actual)


def _analytic_boundary_checks() -> list[dict[str, object]]:
    point_planes = [
        HalfPlane(1.0, 0.0, 1.0),
        HalfPlane(-1.0, 0.0, -1.0),
        HalfPlane(0.0, 1.0, 2.0),
        HalfPlane(0.0, -1.0, -2.0),
    ]
    point_region = intersect_halfplanes(point_planes)
    segment_planes = [
        HalfPlane(0.0, 1.0, 0.0),
        HalfPlane(0.0, -1.0, 0.0),
        HalfPlane(1.0, 0.0, 2.0),
        HalfPlane(-1.0, 0.0, 0.0),
    ]
    segment_region = intersect_halfplanes(segment_planes)

    equilateral = [(0.0, 0.0), (36.0, 0.0), (18.0, 18.0 * math.sqrt(3.0))]
    equilateral_diameter = diameter_rotating_calipers(equilateral)
    equilateral_coverage = diameter_circle_coverage(equilateral, equilateral_diameter)
    square = [(0.0, 0.0), (36.0, 0.0), (36.0, 36.0), (0.0, 36.0)]
    square_diameter = diameter_rotating_calipers(square)
    square_coverage = diameter_circle_coverage(square, square_diameter)

    exact_points = [(-20.0, 0.0), (20.0, 0.0)]
    exact_circle = minimum_enclosing_circle(exact_points)
    above_points = [(-20.000001, 0.0), (20.000001, 0.0)]
    above_circle = minimum_enclosing_circle(above_points)

    arc = CircularArc(
        center=(0.0, 0.0),
        radius=10.0,
        start_angle=math.radians(350.0),
        sweep_angle=math.radians(20.0),
    )
    arc_area, arc_centroid = green_area_centroid(
        [arc, LineSegment(arc.end, arc.start)]
    )

    rounded_payload = {
        "observations": [
            {"station": [0.49, 0.49], "bearing_deg": 0.0},
            {"station": [0.49, 0.49], "bearing_deg": 180.0},
        ],
        "clear_radius_m": 0.5,
        "output_decimals": 0,
    }
    rounded = solve_case(rounded_payload)

    return [
        _check(
            "point_region",
            "analytic",
            {"halfplanes": [[p.a, p.b, p.c] for p in point_planes]},
            {"region_status": "point", "vertices": [[1.0, 2.0]]},
            {"region_status": point_region.status, "vertices": [list(v) for v in point_region.vertices]},
        ),
        _check(
            "segment_region",
            "analytic",
            {"halfplanes": [[p.a, p.b, p.c] for p in segment_planes]},
            {"region_status": "segment", "vertices": [[0.0, 0.0], [2.0, 0.0]]},
            {"region_status": segment_region.status, "vertices": [list(v) for v in segment_region.vertices]},
        ),
        _check(
            "equilateral_diameter_circle_failure",
            "analytic",
            {"vertices": [list(point) for point in equilateral]},
            {"covers": False, "diameter_m": 36.0},
            {
                "covers": equilateral_coverage.covers,
                "diameter_m": equilateral_diameter.distance,
                "diameter_circle_radius_m": equilateral_coverage.radius,
                "max_distance_m": equilateral_coverage.max_distance,
            },
            not equilateral_coverage.covers
            and math.isclose(equilateral_diameter.distance, 36.0, abs_tol=1e-12),
        ),
        _check(
            "square_diameter_circle_success",
            "analytic",
            {"vertices": [list(point) for point in square]},
            {"covers": True, "diameter_m": 36.0 * math.sqrt(2.0)},
            {
                "covers": square_coverage.covers,
                "diameter_m": square_diameter.distance,
                "diameter_circle_radius_m": square_coverage.radius,
                "max_distance_m": square_coverage.max_distance,
            },
            square_coverage.covers
            and math.isclose(
                square_diameter.distance, 36.0 * math.sqrt(2.0), abs_tol=1e-12
            ),
        ),
        _check(
            "clear_radius_exactly_20",
            "analytic",
            {"points": [list(point) for point in exact_points], "clear_radius_m": 20.0},
            {"radius_m": 20.0, "status": "CLEAR_READY"},
            {"radius_m": exact_circle.radius, "status": "CLEAR_READY" if exact_circle.radius <= 20.0 else "SINGLE_DISK_IMPOSSIBLE"},
        ),
        _check(
            "clear_radius_above_20",
            "analytic",
            {"points": [list(point) for point in above_points], "clear_radius_m": 20.0},
            {"radius_above_20": True, "status": "SINGLE_DISK_IMPOSSIBLE"},
            {"radius_above_20": above_circle.radius > 20.0, "status": "SINGLE_DISK_IMPOSSIBLE" if above_circle.radius > 20.0 else "CLEAR_READY"},
        ),
        _check(
            "arc_crosses_zero",
            "analytic",
            {"center": [0.0, 0.0], "radius_m": 10.0, "start_deg": 350.0, "sweep_deg": 20.0},
            {"crosses_zero": True, "finite_result": True, "positive_area": True},
            {
                "crosses_zero": arc.start_angle < 2.0 * math.pi < arc.start_angle + arc.sweep_angle,
                "finite_result": all(math.isfinite(value) for value in (arc_area, *arc_centroid)),
                "positive_area": arc_area > 0.0,
                "area_m2": arc_area,
                "centroid": list(arc_centroid),
            },
            arc.start_angle < 2.0 * math.pi < arc.start_angle + arc.sweep_angle
            and all(math.isfinite(value) for value in (arc_area, *arc_centroid))
            and arc_area > 0.0,
        ),
        _check(
            "rounded_center_counterexample",
            "analytic",
            rounded_payload,
            {"status": "COVERAGE_UNCERTAIN", "rounded_center_exceeds_radius": True},
            {
                "status": rounded["control"]["status"],
                "rounded_center_exceeds_radius": rounded["control"]["rounded_center_max_distance_m"] > 0.5,
                "output_center": rounded["control"]["output_center"],
                "rounded_center_max_distance_m": rounded["control"]["rounded_center_max_distance_m"],
            },
            rounded["control"]["status"] == "COVERAGE_UNCERTAIN"
            and rounded["control"]["rounded_center_max_distance_m"] > 0.5,
        ),
    ]


def run_simulation(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Solve every observation case and aggregate reproducible checks."""

    solved: list[tuple[Mapping[str, object], dict[str, object]]] = [
        (case, solve_case(case)) for case in cases
    ]
    statuses = Counter(result["region"]["status"] for _, result in solved)
    status_distribution = {
        status: statuses.get(status, 0)
        for status in ("empty", "unbounded", "point", "segment", "polygon")
    }

    consistent = [
        case
        for case, _ in solved
        if case.get("case_kind") in {"regular_random", "near_parallel_random"}
        and case.get("expected_truth_retained") is True
    ]
    retained_count = sum(_truth_retained(case) for case in consistent)
    diameters: list[float] = []
    radii: list[float] = []
    discrepancies: list[float] = []
    residuals: list[float] = []
    for _, result in solved:
        diameter = result["problem_1"]["diameter_m"]
        circle = result["problem_1"]["minimum_enclosing_circle"]
        if isinstance(diameter, (int, float)):
            diameters.append(float(diameter))
            discrepancies.append(float(result["problem_1"]["diameter_discrepancy_m"]))
        if circle is not None:
            radii.append(float(circle["radius_m"]))
            residuals.append(abs(float(circle["max_residual_m"])))

    observation_checks = [
        _observation_boundary_check(case, result)
        for case, result in solved
        if case.get("case_id") in OBSERVATION_BOUNDARY_IDS
    ]
    boundary_checks = observation_checks + _analytic_boundary_checks()
    kind_counts = Counter(str(case.get("case_kind", "unspecified")) for case in cases)
    random_cases = kind_counts["regular_random"] + kind_counts["near_parallel_random"]
    seeds = {case.get("simulation_seed", DEFAULT_SEED) for case in cases}
    if len(seeds) != 1:
        raise ValueError("all cases must use the same simulation seed")
    seed = seeds.pop()
    return {
        "seed": seed,
        "counts": {
            "total_observation_cases": len(cases),
            "observation_boundary": kind_counts["observation_boundary"],
            "regular_random": kind_counts["regular_random"],
            "near_parallel_random": kind_counts["near_parallel_random"],
            "random_total": random_cases,
            "consistent_cases": len(consistent),
            "analytic_checks": len(ANALYTIC_BOUNDARY_IDS),
            "boundary_checks": len(boundary_checks),
        },
        "region_status_distribution": status_distribution,
        "consistent_truth_retention_count": retained_count,
        "consistent_truth_case_count": len(consistent),
        "consistent_truth_retention_rate": (
            retained_count / len(consistent) if consistent else 1.0
        ),
        "diameter_m_quantiles": _quantiles(diameters),
        "minimum_radius_m_quantiles": _quantiles(radii),
        "diameter_crosscheck_max_abs_error_m": max(discrepancies, default=0.0),
        "welzl_max_residual_m": max(residuals, default=0.0),
        "boundary_pass_count": sum(check["pass"] for check in boundary_checks),
        "boundary_fail_count": sum(not check["pass"] for check in boundary_checks),
        "boundary_checks": boundary_checks,
    }


def _markdown_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return "null"
    if isinstance(value, Mapping):
        return "{" + ", ".join(
            f"{key}: {_markdown_value(item)}" for key, item in value.items()
        ) + "}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "[" + ", ".join(_markdown_value(item) for item in value) + "]"
    return str(value)


def _parameters_markdown(summary: Mapping[str, object]) -> str:
    counts = summary["counts"]
    lines = [
        "# Q1 仿真参数",
        "",
        f"固定随机种子：`{summary['seed']}`。生成命令：`.venv\\Scripts\\python.exe src/q1/simulate_cases.py`。",
        "",
        "## 随机样本",
        "",
        f"- 常规样本数：{counts['regular_random']}；近平行样本数：{counts['near_parallel_random']}。",
        "- 真值源在半径 1800.000000 m 的圆域内按面积均匀抽样，即半径采用 `1800.000000 * sqrt(U)`。",
        "- 每例使用 3 个站点；站点距真值源为 [200.000000, 1400.000000] m 的均匀分布。",
        "- 字面测角误差来自 [-1.000000, 1.000000] deg 均匀分布，半角为 1.000000 deg，并保留浮点全精度。",
        "- 常规站点方位约相隔 120.000000 deg，并分别加入 [-15.000000, 15.000000] deg 扰动。",
        "- 近平行站点轴向偏移基准为 0.000000、180.000000、180.500000 deg，并加入 [-0.100000, 0.100000] deg 扰动。",
        "",
        "## 单位与数值容差",
        "",
        "- 坐标、直径和半径单位：m；面积单位：m²；角度单位：deg。",
        "- 真值半平面包含容差：1.000000e-8 m（不等式残差尺度）。",
        "- 正式求解器直径交叉校验相对容差：1.000000e-10；Welzl 最终覆盖复核相对容差：1.000000e-10。",
        "- Markdown 浮点数统一显示 6 位小数；JSON 保留 Python 浮点全精度。",
        "",
        "## 边界检查显式参数",
        "",
        "| case_id | 类别 | 参数 |",
        "|---|---|---|",
    ]
    for check in summary["boundary_checks"]:
        lines.append(
            f"| {check['case_id']} | {check['kind']} | {_markdown_value(check['parameters'])} |"
        )
    return "\n".join(lines) + "\n"


def _results_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# Q1 仿真结果",
        "",
        f"通过 {summary['boundary_pass_count']}；失败 {summary['boundary_fail_count']}；固定种子 `{summary['seed']}`。",
        "",
        "## 汇总统计",
        "",
        f"- 观测案例数：{summary['counts']['total_observation_cases']}。",
        f"- 一致案例真值保留：{summary['consistent_truth_retention_count']}/{summary['consistent_truth_case_count']}，比例 {_markdown_value(summary['consistent_truth_retention_rate'])}。",
        f"- 区域状态分布：{_markdown_value(summary['region_status_distribution'])}。",
        f"- 直径分位数（m）：{_markdown_value(summary['diameter_m_quantiles'])}。",
        f"- 最小包围圆半径分位数（m）：{_markdown_value(summary['minimum_radius_m_quantiles'])}。",
        f"- 直径交叉校验最大绝对差（m）：{_markdown_value(summary['diameter_crosscheck_max_abs_error_m'])}。",
        f"- Welzl 最大残差（m）：{_markdown_value(summary['welzl_max_residual_m'])}。",
        "",
        "## 边界检查",
        "",
        "| case_id | 类别 | 期望 | 实际 | 结论 |",
        "|---|---|---|---|---|",
    ]
    for check in summary["boundary_checks"]:
        lines.append(
            f"| {check['case_id']} | {check['kind']} | "
            f"{_markdown_value(check['expected'])} | "
            f"{_markdown_value(check['actual'])} | "
            f"{'PASS' if check['pass'] else 'FAIL'} |"
        )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_outputs(
    root: Path, cases: Sequence[Mapping[str, object]], summary: Mapping[str, object]
) -> None:
    """Write the four deterministic data and Markdown report artifacts."""

    root = Path(root)
    cases_path = root / "data" / "processed" / "q1_simulation_cases.json"
    parameters_path = root / "data" / "processed" / "q1_simulation_parameters.md"
    results_json_path = root / "results" / "tables" / "q1_simulation_results.json"
    results_md_path = root / "results" / "tables" / "q1_simulation_results.md"
    _write_json(
        cases_path,
        {"seed": summary["seed"], "counts": summary["counts"], "cases": list(cases)},
    )
    _write_json(results_json_path, dict(summary))
    parameters_path.parent.mkdir(parents=True, exist_ok=True)
    parameters_path.write_text(_parameters_markdown(summary), encoding="utf-8")
    results_md_path.parent.mkdir(parents=True, exist_ok=True)
    results_md_path.write_text(_results_markdown(summary), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    cases = generate_cases()
    summary = run_simulation(cases)
    write_outputs(root, cases, summary)
    if summary["boundary_fail_count"] or summary["consistent_truth_retention_rate"] != 1.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
