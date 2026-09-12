# Q4 最终算法与对照组

本目录仅交付最终选定算法、对应对照算法和必要的测试评估工具，不包含中间训练集、模型权重、未采用的探索算法或过程日志。

## 算法映射

| 组别 | 命令标识 | 算法代码 | 含义 |
|---|---|---|---|
| 实验组（默认） | `informed` | `informed.py` | 联合信念主动选点，筛选预计无效的辅助补测，沿用完整覆盖、路线与清除闭环。 |
| 对照组一 | `refined` | `refined.py` | 上一版 v3 路线优化与混合源定位策略。 |
| 对照组二 | `adaptive` | `controller.py` 中的 adaptive 模式 | 早期混合源自适应扫描、定位与清除闭环。 |
| 对照组三 | `seven_grid` | `controller.py` 中的 seven_grid 模式 | 七点初筛加完整网格的保守比较策略。 |

`controller.py` 负责真实动作与预算；`coverage.py` 负责方向覆盖；`joint_model.py` 处理联合位置/方向工作信念；`route_refinement.py` 负责路线排序；`q3_adapter.py` 固定复用的 Q3 辅助接口。各组共用 `src/common/simulator_client.py`，不重复实现 HTTP 客户端。

## 运行

在仓库根目录使用现有 requirements.txt 环境，不需要训练权重或深度学习运行库。

```powershell
# 已就绪的官方演练会话；不会自行启动正式测试
python -m src.q4.main --robot-id 你的团队编号

# 三个对照分别指定 refined / adaptive / seven_grid
python -m src.q4.main --robot-id 你的团队编号 --strategy refined

# 四组在完全相同的场景、源数与误差机制上比较
python -m src.q4.compare --counts 10 11 12 13 14 15 16 --cases-per-group 10 --seed 260000000 --noise hash --output runs/q4/comparison/hash_70

# 只重现最终算法与上一版的配对实验
python -m src.q4.compare --strategies informed refined --seed 261000000 --noise spatial --output runs/q4/comparison/spatial_70

# 严格重放某组已生成的记录
python -m src.q4.replay runs/q4/comparison/hash_70/informed --output runs/q4/comparison/replay.json

python -m unittest discover -s tests -p 'test_q4*.py' -q
```

评估输出目录必须不存在，以免覆盖记录。`compare.py` 为每组保存配置、完整动作、源真值（仅供事后审计）、完成状态、费用分解与源码快照；顶层 `comparison.json` 给出同场比较。真值不会传入控制器。`benchmark.py` 可单独评估一个控制器；`evaluate.py` 保留默认算法的便捷离线入口。

## 最终策略与完整性

默认信息权重为 3、辅助几何半径 1400 米、最低预计可见概率 0.5；500 米初始基线、1100 米额外扫描间距、60 米光学尝试阈值沿用上一版。粒子近似只影响规划，不能认证无源或清除成功。当前目标和完整扫描的未知频道不会被可选补测筛选跳过。

默认方向覆盖使用原点、998 米内环 8 点和约 1866.50 米外环 12 点，无法确认覆盖时保留网格后备。未知频道必须具有完整方向覆盖阴性回执，或依据已发现 16 个不同源的题设上限排除；每个源必须获得实际成功清除回执。所有移动、检测、切频及光学失败/成功费用全部计入。

默认预算为新动作 4096、请求 12300、累计计算 900 秒、虚拟时间 360000 秒，同时检查官方现实时间余额。预算不足明确报告未完成，不伪造清除或排除；未决请求存在时不发送新动作。具体预算可通过 main 的命令参数调整。

## 最终结果与边界

冻结新 140 场，最终算法 **451.98 秒/源**，上一版 refined 为 **463.75 秒/源**，降低 **2.54%**，双方 1820 源全部清除。逐场紧凑统计及分组结果见 `results/tables/q4_final_comparison.md/.json`。该 140 场仅比较 informed/refined，另外两组的完整性能应通过上述同场命令生成，不能借用上一版数字。

总体 300–399 秒目标仍未达到。结果仅为离线仿真，没有官方成绩或全局最优性结论。浮点几何与保守工程余量尚不是完整的有向舍入区间认证器，有限测试的 100% 清除不保证所有官方场景。
