# Q1 几何定位求解器

本目录实现问题一的纯测向定位区域求解、几何量计算、单圆覆盖判定和固定种子仿真。主结果始终针对测向半平面交

\[
P=\bigcap_i\{(x,y):a_i x+b_i y\le c_i\},
\]

其中每条观测把测站 `station=(s_x,s_y)`、归一到 `[0, 360)` 的方位角 `bearing_deg=theta` 和误差半角 `half_angle_deg=delta` 转为方位区间 `[theta-delta, theta+delta]` 的两个半平面。坐标和距离单位为米（m），面积单位为平方米（m²），输入角度单位为度（deg）；内部三角计算使用弧度。

## 环境与命令

在仓库根目录执行，使用项目已有 `.venv`（需要 NumPy 和 SciPy）：

```powershell
.venv\Scripts\python.exe -m unittest discover -s src/q1 -p 'test_*.py' -v
.venv\Scripts\python.exe src/q1/main.py data/processed/q1_simulation_cases.json --output results/tables/q1_manual_run.json
.venv\Scripts\python.exe src/q1/simulate_cases.py
```

第二条命令的 `q1_manual_run.json` 是人工复核用临时文件，不属于固定生成物。第三条命令会确定性重建下文列出的四份仿真数据和报告。

## JSON 输入

入口接受单个案例对象，或包含案例数组的对象：

```json
{
  "cases": [
    {
      "case_id": "example",
      "observations": [
        {
          "station": [-500.0, 0.0],
          "bearing_deg": 359.5,
          "half_angle_deg": 1.0
        }
      ],
      "arena_radius_m": 1800.0,
      "clear_radius_m": 20.0,
      "output_decimals": 6
    }
  ]
}
```

- `observations` 必须是数组。每项必须含二元有限坐标 `station` 和有限数 `bearing_deg`；`half_angle_deg` 可省略，默认 `1.0`，且必须满足 `0 < half_angle_deg < 90`。
- `arena_radius_m` 和 `clear_radius_m` 可省略，默认分别为 `1800.0` 和 `20.0`，两者必须为正有限数。
- `output_decimals` 可省略，默认 `6`，必须是 `[0, 15]` 内的整数。它只控制动作中心 `control.output_center` 的坐标舍入，不会提前舍入几何计算。
- `case_id` 可选；存在时会原样带入对应输出。仿真案例中的 `case_kind`、`true_source`、`sampled_error_deg` 等元数据不参与正式求解。

入口总是输出一个 JSON 数组；写入使用 UTF-8、两空格缩进、禁止 `NaN`，并通过同目录临时文件替换目标文件，避免留下半写结果。

## 输出解释

每个结果分为三个部分：

- `region` 描述纯测向区域 `P`。`status` 为 `empty`、`unbounded`、`point`、`segment` 或 `polygon`；非空有界状态给出逆时针凸顶点。`polygon` 另给出 `area_m2` 和 `centroid`。`max_violation` 是返回顶点对原半平面不等式的最大原始残差。
- `problem_1` 给出题目一几何量。区域直径为 `D=max_{p,q in P} ||p-q||`，`farthest_pair` 是确定性的最远点对；同时用旋转卡壳和穷举计算直径并在 `diameter_discrepancy_m` 中报告差值。`diameter_circle` 的圆心是最远点对中点、半径为 `D/2`，`covers` 表示该圆是否覆盖全部顶点。`minimum_enclosing_circle` 给出 Welzl 最小包围圆的内部圆心、半径和最大覆盖残差。
- `control` 单独报告物理约束，不会改变 `P`。`arena_contains_region` 表示有界 `P` 是否整体位于以原点为圆心、半径 `arena_radius_m` 的圆域内；`output_center` 是按 `output_decimals` 舍入后的动作中心，`rounded_center_max_distance_m` 是该舍入中心到 `P` 顶点的可靠最大距离。

`P` 可以为空或无界。空集的直径为 `null`，控制状态为 `NO_FEASIBLE_REGION`；无界区域的直径使用 JSON 兼容字符串 `"infinity"`，控制状态为 `UNBOUNDED_REGION`，二者均不输出有限覆盖圆。

控制状态含义如下：

- `CLEAR_READY`：有界 `P` 整体在目标圆域内，并且已经用舍入后的 `output_center` 复核其最大距离不超过 `clear_radius_m`。仅内部最小包围圆满足阈值并不足以产生此状态。
- `SINGLE_DISK_IMPOSSIBLE`：在无需目标圆域裁剪的已认证集合上，最小包围圆半径严格超过清除半径，单个圆盘无法覆盖。
- `COVERAGE_UNCERTAIN`：需要目标圆域裁剪，或动作中心舍入后越过清除阈值。实现不会用多边形近似圆域；当 `P` 超出目标圆域时，以 `reason="arena_clipping_required"` 保守报告，而不近似裁剪后给出结论。

## 数值策略与容差

- 线性规划直接在原始半平面系统上认证可行性及坐标有界性，不用任意大包围盒截断无界区域；仅在已认证有界后用有限坐标极值初始化多边形裁剪。
- 半平面裁剪和点包含默认绝对容差为 `1e-9`；仿真真值保留检查使用 `1e-8` 的不等式残差容差。
- 旋转卡壳直径必须与全顶点对穷举结果在 `1e-10 * max(1, D_exhaustive, D_calipers)` 内一致。
- 直径圆覆盖、目标圆域包含、清除阈值和动作中心复核使用尺度相关的 `1e-12` 容差；清除判定比较距离平方，避免开方或显示舍入改变边界结论。
- Welzl 算法默认固定种子 `20260911`，内部覆盖采用尺度相关 `1e-12` 容差，返回前再以尺度相关 `1e-10` 容差复核所有点并报告 `max_residual_m`。
- 线段和有向圆弧的面积、质心使用格林公式。圆弧显式保存起角和逆时针扫角，扫角允许 `0` 或 `2*pi`，不会从两个端点角猜测短弧。

## 固定仿真与边界案例

默认种子为 `20260911`。一次生成包含 78 个观测案例：8 个观测边界案例、50 个常规随机案例和 20 个近平行随机案例；另执行 8 个解析检查，因此边界检查总数为 16。

观测边界案例为 `cross_zero`、`error_at_positive_bound`、`error_at_negative_bound`、`unbounded_single_observation`、`empty_conflicting_observations`、`duplicate_observations`、`near_parallel_intersection`、`source_on_arena_boundary`。解析检查为 `point_region`、`segment_region`、`equilateral_diameter_circle_failure`、`square_diameter_circle_success`、`clear_radius_exactly_20`、`clear_radius_above_20`、`arc_crosses_zero`、`rounded_center_counterexample`。

随机真值在半径 1800 m 圆域内按面积均匀抽样；每例使用 3 个距真值 200--1400 m 的测站，并加入 `[-1, 1]` deg 的字面测角误差。生成顺序固定为观测边界、常规随机、近平行随机。JSON 保留 Python 浮点全精度，Markdown 数值显示 6 位小数；在相同代码、Python/NumPy/SciPy 环境下重复执行生成命令，应得到无 Git 差异的四份文件。报告中的 `PASS` 表示预先声明的边界不变量通过，不表示对所有测站布局或真实噪声的统计性能保证。

## 文件地图

- `src/q1/geometry.py`：观测校验、测向楔形、半平面交、凸包与区域分类。
- `src/q1/measures.py`：穷举/旋转卡壳直径、直径圆覆盖、直线段和圆弧格林积分。
- `src/q1/enclosing_circle.py`：固定种子的最小包围圆及最终覆盖认证。
- `src/q1/solver.py`：组合几何结果、目标圆域语义和清除状态。
- `src/q1/main.py`：单案例或多案例 JSON 命令行入口。
- `src/q1/simulate_cases.py`：固定案例生成、统计、边界检查与四份生成物写出。
- `data/processed/q1_simulation_cases.json`：78 个机器可读观测案例。
- `data/processed/q1_simulation_parameters.md`：种子、样本分布、单位、容差和全部边界参数。
- `results/tables/q1_simulation_results.json`：全精度统计与逐项边界检查结果。
- `results/tables/q1_simulation_results.md`：六位小数的人工复核摘要。

## 限制

- 这是二维欧氏平面求解器；经纬度必须先投影到一致的米制平面坐标系。
- 目标圆域只做整体包含检查。需要 `P` 与圆域精确交集时，本实现会保守返回不确定，不提供近似裁剪面积、质心或覆盖结论。
- 无界 `P` 不计算有限直径、面积或包围圆；退化为点或线段时面积和质心保持为 `null`。
- 清除结论只回答一个半径为 `clear_radius_m` 的圆盘能否覆盖已认证集合，不处理多圆盘路径规划、运动学、通信或执行误差。
- 极度近平行或近退化输入仍受浮点运算和 SciPy 线性规划器的数值条件影响；应结合 `max_violation`、直径交叉差和最小圆残差判断结果质量。
