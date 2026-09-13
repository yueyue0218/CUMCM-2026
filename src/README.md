# 正式代码

- `q1/` 至 `q4/`：对应各分问题的策略、正式实现和运行入口。
- `common/geometry.py`：角度、角域裁剪、凸包、区域直径和第二点候选几何。
- `common/localization.py`：多检测点定位区域及其质量指标。
- `common/simulator_client.py`：四类 HTTP 指令、串行调用、幂等重试、协议状态和基础 JSONL 日志，不包含搜索策略。
- `common/metrics.py`：Q3/Q4 清除比例和时间指标的统一口径。
- `q3/main.py`：完整搜索定位清除入口；`--strategy scan` 为仅发现对照。定位复用既有 Q1 适配器，补测使用有限步确定性后备。运行、日志及离线验证详见 [q3/README.md](q3/README.md)。
- `q3/train_ppo.py`、`q3/evaluate_policy.py`：已接通 Q2 候选、粒子预测、开放路径和循环 PPO，可训练、续训及加载模型；使用说明见 [q3/TRAINING.md](q3/TRAINING.md)。

每个分问题形成可运行实现后以 `main.py` 为入口，路径相对于仓库根目录。官方输入只读，生成内容写入 `results/` 或对应的 `runs/` 独立目录。Notebook 中确认有效的算法必须迁移到这里。

当前Q3默认策略为 `efficient`（集中检测、联合路线、动态覆盖证明）。分层验证及运行方式见 [q3/EFFICIENCY.md](q3/EFFICIENCY.md)，原模型策略继续保留。

Q4：`q4/experiment_informed.py` 为最终算法具名入口，`q4/control_*.py` 为上一版、普查、螺旋、随机游走和已知位置参考。`q4/compare.py` 进行同场评估，`q4/comparison_tables.py` 导出总表、分组表及逐场 CSV，`q4/replay.py` 核验动作。定义与复现见 [Q4 对照说明](q4/STRATEGY_CONTROLS.md)。
