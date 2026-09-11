# Q1 仿真参数

固定随机种子：`20260911`。生成命令：`.venv\Scripts\python.exe src/q1/simulate_cases.py`。

## 随机样本

- 常规样本数：50；近平行样本数：20。
- 真值源在半径 1800.000000 m 的圆域内按面积均匀抽样，即半径采用 `1800.000000 * sqrt(U)`。
- 每例使用 3 个站点；站点距真值源为 [200.000000, 1400.000000] m 的均匀分布。
- 字面测角误差来自 [-1.000000, 1.000000] deg 均匀分布，半角为 1.000000 deg，并保留浮点全精度。
- 常规站点方位约相隔 120.000000 deg，并分别加入 [-15.000000, 15.000000] deg 扰动。
- 近平行站点轴向偏移基准为 0.000000、180.000000、180.500000 deg，并加入 [-0.100000, 0.100000] deg 扰动。

## 单位与数值容差

- 坐标、直径和半径单位：m；面积单位：m²；角度单位：deg。
- 真值半平面包含容差：1.000000e-8 m（不等式残差尺度）。
- 正式求解器直径交叉校验相对容差：1.000000e-10；Welzl 最终覆盖复核相对容差：1.000000e-10。
- Markdown 浮点数统一显示 6 位小数；JSON 保留 Python 浮点全精度。

## 边界检查显式参数

| case_id | 类别 | 参数 |
|---|---|---|
| cross_zero | observation | {observation_count: 3, true_source: [0.000000, 0.000000], sampled_errors_deg: [-0.500000, 0.000000, 0.000000]} |
| error_at_positive_bound | observation | {observation_count: 3, true_source: [125.000000, -240.000000], sampled_errors_deg: [1.000000, 0.250000, -0.250000]} |
| error_at_negative_bound | observation | {observation_count: 3, true_source: [-360.000000, 175.000000], sampled_errors_deg: [-1.000000, -0.250000, 0.250000]} |
| unbounded_single_observation | observation | {observation_count: 1, true_source: [100.000000, 100.000000], sampled_errors_deg: [0.000000]} |
| empty_conflicting_observations | observation | {observation_count: 2, true_source: null, sampled_errors_deg: []} |
| duplicate_observations | observation | {observation_count: 5, true_source: [0.000000, 0.000000], sampled_errors_deg: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]} |
| near_parallel_intersection | observation | {observation_count: 3, true_source: [80.000000, -60.000000], sampled_errors_deg: [0.000000, 0.000000, 0.000000]} |
| source_on_arena_boundary | observation | {observation_count: 4, true_source: [1800.000000, 0.000000], sampled_errors_deg: [0.000000, 0.000000, 0.000000, 0.000000]} |
| point_region | analytic | {halfplanes: [[1.000000, 0.000000, 1.000000], [-1.000000, 0.000000, -1.000000], [0.000000, 1.000000, 2.000000], [0.000000, -1.000000, -2.000000]]} |
| segment_region | analytic | {halfplanes: [[0.000000, 1.000000, 0.000000], [0.000000, -1.000000, 0.000000], [1.000000, 0.000000, 2.000000], [-1.000000, 0.000000, 0.000000]]} |
| equilateral_diameter_circle_failure | analytic | {vertices: [[0.000000, 0.000000], [36.000000, 0.000000], [18.000000, 31.176915]]} |
| square_diameter_circle_success | analytic | {vertices: [[0.000000, 0.000000], [36.000000, 0.000000], [36.000000, 36.000000], [0.000000, 36.000000]]} |
| clear_radius_exactly_20 | analytic | {observations: [{station: [-20.000000, 0.000000], bearing_deg: 0.000000}, {station: [20.000000, 0.000000], bearing_deg: 180.000000}], arena_radius_m: 1800.000000, clear_radius_m: 20.000000, output_decimals: 6} |
| clear_radius_above_20 | analytic | {observations: [{station: [-20.000001, 0.000000], bearing_deg: 0.000000}, {station: [20.000001, 0.000000], bearing_deg: 180.000000}], arena_radius_m: 1800.000000, clear_radius_m: 20.000000, output_decimals: 6} |
| arc_crosses_zero | analytic | {center: [0.000000, 0.000000], radius_m: 10.000000, start_deg: 350.000000, sweep_deg: 20.000000} |
| rounded_center_counterexample | analytic | {observations: [{station: [0.490000, 0.490000], bearing_deg: 0.000000}, {station: [0.490000, 0.490000], bearing_deg: 180.000000}], clear_radius_m: 0.500000, output_decimals: 0} |
