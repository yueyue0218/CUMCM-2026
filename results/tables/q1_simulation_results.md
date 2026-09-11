# Q1 仿真结果

通过 16；失败 0；固定种子 `20260911`。

## 汇总统计

- 观测案例数：78。
- 随机一致案例真值保留：70/70，比例 1.000000。
- 区域状态分布：{empty: 1, unbounded: 1, point: 0, segment: 0, polygon: 76}。
- 直径分位数（m）：{minimum: 9.100892, q25: 26.667470, median: 39.755546, q75: 563.969324, maximum: 2044.592749}。
- 最小包围圆半径分位数（m）：{minimum: 4.608208, q25: 13.574829, median: 20.051017, q75: 281.984662, maximum: 1022.296374}。
- 直径交叉校验最大绝对差（m）：0.000000。
- Welzl 最大残差（m）：0.000000。

## 边界检查

| case_id | 类别 | 期望 | 实际 | 结论 |
|---|---|---|---|---|
| cross_zero | observation | {region_status: polygon, truth_retained: true, crosses_zero: true} | {region_status: polygon, truth_retained: true, crosses_zero: true} | PASS |
| error_at_positive_bound | observation | {region_status: polygon, truth_retained: true, sampled_error_deg: 1.000000} | {region_status: polygon, truth_retained: true, sampled_error_deg: 1.000000} | PASS |
| error_at_negative_bound | observation | {region_status: polygon, truth_retained: true, sampled_error_deg: -1.000000} | {region_status: polygon, truth_retained: true, sampled_error_deg: -1.000000} | PASS |
| unbounded_single_observation | observation | {region_status: unbounded, truth_retained: true} | {region_status: unbounded, truth_retained: true} | PASS |
| empty_conflicting_observations | observation | {region_status: empty} | {region_status: empty} | PASS |
| duplicate_observations | observation | {region_status: polygon, truth_retained: true, has_duplicate: true} | {region_status: polygon, truth_retained: true, has_duplicate: true} | PASS |
| near_parallel_intersection | observation | {region_status: polygon, truth_retained: true} | {region_status: polygon, truth_retained: true} | PASS |
| source_on_arena_boundary | observation | {region_status: polygon, truth_retained: true, source_radius_m: 1800.000000} | {region_status: polygon, truth_retained: true, source_radius_m: 1800.000000} | PASS |
| point_region | analytic | {region_status: point, vertices: [[1.000000, 2.000000]]} | {region_status: point, vertices: [[1.000000, 2.000000]]} | PASS |
| segment_region | analytic | {region_status: segment, vertices: [[0.000000, 0.000000], [2.000000, 0.000000]]} | {region_status: segment, vertices: [[-0.000000, -0.000000], [2.000000, -0.000000]]} | PASS |
| equilateral_diameter_circle_failure | analytic | {covers: false, diameter_m: 36.000000} | {covers: false, diameter_m: 36.000000, diameter_circle_radius_m: 18.000000, max_distance_m: 31.176915} | PASS |
| square_diameter_circle_success | analytic | {covers: true, diameter_m: 50.911688} | {covers: true, diameter_m: 50.911688, diameter_circle_radius_m: 25.455844, max_distance_m: 25.455844} | PASS |
| clear_radius_exactly_20 | analytic | {region_status: polygon, arena_contains_region: true, control_status: CLEAR_READY, diameter_m: 40.000000, required_radius_m: 20.000000, rounded_center_max_distance_m: 20.000000} | {region_status: polygon, arena_contains_region: true, control_status: CLEAR_READY, diameter_m: 40.000000, required_radius_m: 20.000000, rounded_center_max_distance_m: 20.000000} | PASS |
| clear_radius_above_20 | analytic | {region_status: polygon, arena_contains_region: true, control_status: SINGLE_DISK_IMPOSSIBLE, diameter_m: 40.000002, required_radius_m: 20.000001, rounded_center_max_distance_m: 20.000001} | {region_status: polygon, arena_contains_region: true, control_status: SINGLE_DISK_IMPOSSIBLE, diameter_m: 40.000002, required_radius_m: 20.000001, rounded_center_max_distance_m: 20.000001} | PASS |
| arc_crosses_zero | analytic | {crosses_zero: true, finite_result: true, positive_area: true} | {crosses_zero: true, finite_result: true, positive_area: true, area_m2: 0.352285, centroid: [9.908886, 0.000000]} | PASS |
| rounded_center_counterexample | analytic | {status: COVERAGE_UNCERTAIN, rounded_center_exceeds_radius: true} | {status: COVERAGE_UNCERTAIN, rounded_center_exceeds_radius: true, output_center: [0.000000, 0.000000], rounded_center_max_distance_m: 0.692965} | PASS |
