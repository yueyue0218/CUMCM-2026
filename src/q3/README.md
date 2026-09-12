# Q3 全向源运行与训练

`main.py` 当前默认运行 `efficient_v2`：同站集中测量、覆盖路线候选池、实际补测点规划、Q1可靠清除与有限后备。在全新的10/12/14/16源各20场配对验证中，等权每源均值从上一版252.23秒降至246.62秒，降低2.22%，两版均全清除80/80；详见 [第二版自主迭代](ITERATION.md)。所有真实动作复用 `src/common/simulator_client.py`，严格串行。

`--strategy efficient` 保留上一版原策略，其历史独立集均值为257.2秒，详见 [上一版效率结果](EFFICIENCY.md)。历史集和新验收集的场景不同，应使用同场配对数据评价本次增益。

现在也已接通 Q2 主动候选、粒子世界模型、跨频道清除路线和循环 PPO。新训练、续训、模型加载及独立对比命令见 [TRAINING.md](TRAINING.md)。`complete` 保留旧确定性基线，`scan` 保留仅发现对照；下文七点逐源流程描述的是complete基线。

第二版模块为 `experiment_route_pool.py`、`coverage_route_pool.py`，新旧配对比较入口为 `evaluate_route_iteration.py`。上一版模块为 `experiment_joint_routing.py`、`coverage_control.py`，历史分层比较入口为 `evaluate_efficiency.py`。

新增五类具名对照：普查后清除、螺旋扫描、随机游走、上帝视角、旧 PPO 辅助规划。六组同场比较命令、文件名和指标口径见 [STRATEGY_CONTROLS.md](STRATEGY_CONTROLS.md)，最新结果见 [策略比较表](../../results/tables/Q3_实验组与对照组_策略比较表.md)。上帝视角只在离线对照中显式取得真值。

## 文件职责

- `baseline_scan.py`：七点布局、蛇形频道顺序和原始发现记录。
- `localization_control.py`：既有 Q1 到 Q3 的状态适配，保持原接口。
- `search_clear.py`：逐频道扫描回执、清除状态、动作预算及完整控制器。
- `main.py`：命令行、进入/退出模拟器、独立运行目录与摘要。
- `offline_simulator.py`：用于训练和验证的简化合成场景；真值不传给控制器。
- `validate_offline.py`：固定种子的离线场景集、独立耗时核算和结果导出。
- `world_model.py`：已发现频道的位置—固定接收半径粒子后验、近似反馈预测、开放清除路径。
- `adaptive_control.py`：合法候选、可观测特征、规划评分、有限探索预算及连续后备。
- `ppo.py`：共享频道编码、GRU、候选策略头和价值头，完整回合 PPO 更新。
- `training_env.py`、`policy_runtime.py`：单步训练环境及训练/实际入口共用的策略选择。
- `train_ppo.py`、`evaluate_policy.py`：训练、续训、验证选模及独立同场景比较。

## 演练运行

先在官方模拟器中启动 **Q3 演练模块**，等待接口就绪，然后从仓库根目录运行：

```powershell
python src/q3/main.py --robot-id "当前登录队号"
```

也支持 `python -m src.q3.main`。脚本可以用绝对路径从其他目录启动；默认结果目录仍位于本仓库。

参数：`--strategy efficient_v2|efficient|complete|scan|planner|ppo|hybrid`、`--checkpoint`、`--policy-threads`、`--base-url`、`--run-root`、`--virtual-limit-s`、`--exit-reserve-s`。ppo/hybrid 必须提供模型，进入模拟器前检查特征版本；planner 不需模型。虚拟时限默认 360000 秒，只可降低；现实时间使用 `/enter` 的实际回执，动作前为当前请求及退出请求预留完整重试时间。虚拟动作预算包含移动、实际切频和检测/清除成本。

本入口记录 `run_type=practice`，它不能检测模拟器界面选择的模块。正式测试须先按 `docs/formal_test_checklist.md` 完成团队验收与版本冻结；本次没有执行任何官方演练或正式测试。

每次运行生成 `runs/q3/practice/<时间>_<策略>/`，包含 `config.json`、`requests.jsonl`、`responses.jsonl`、`summary.json`、`notes.md`。日志脱敏队号，失败也保存已取得的证据。若请求结果无法确认，则保留原请求与最后确认时刻、停止新请求，不再发送 `/exit`；摘要标注 `pending_action` 和 `exit_skipped_reason`。正常完成、已确认的拒绝及预算退出仍主动调用 `/exit`。返回码：0 为策略正常完成（scan 仅表示发现扫描完成）；1 为程序、通信或退出异常；2 为完整策略因预算或模型冲突未完成。

## 决策与证据

1. 未知频道在七个覆盖点按蛇形顺序检测。单次 `no_signal` 不判无源。
2. `near` 在该观测位置立即清除；`direction` 交由既有 Q1 适配层评估。
3. Q1 就绪时用其认证坐标清除；需要补测时，按论文第7节从**产生示向的实际检测点**沿示向轴移动 `U/(2*cos(1°))`，更新目标距离上界。最多6次新增检测、7次逼近移动；上界进入20米安全阈值后直接清除。
4. 只接受 `clear_result=success` 作为清除证据。已发现但未清除的频道始终保留；保证接收范围内返回无信号、Q1区域为空、认证清除失败均停止并记录冲突，不重复尝试同一证书。
5. 未知频道只有在七个指定点均有有效无信号回执后才排除。全部频道均已清除或有完整排除证据，才报告 `all_cleared=true`。若20个频道均有清除成功回执，可直接结束；没有到达的后续点不记为已扫描。

Q1 浮点几何认证和后备坐标均采用工程安全余量，没有把浮点运算宣称为严格区间证明。现有 Q1 适配器融合方向记录、优先读取 near；世界模型另外利用无信号约束固定半径后验，但这些概率不用于清除或无源认证。

自适应模式在有限自由阶段选择七点搜索、Q2 横纵向补测、认证清除及解析后备，可调整频道顺序。自由动作/虚拟时间预算耗尽后连续执行确定性后备；实际剩余时间降至120秒时也提前结束规划。120秒为工程预留，不能保证任意前缀的剩余任务都能在现实时间内完成。

世界模型是已发现频道的条件粒子模型，尚未校准量化误差与空间相关性，未知频道没有虚构的有源概率。planner 使用单步响应预测及定位尺度、路径成本启发式；hybrid 加入 PPO 对数概率排序先验。当前不是论文全部多步世界模型规划，也没有实现注意力网络或训练价值头辅助树搜索。

`summary.json` 的平均定位清除时间为最终累计虚拟时间除以清除数量，包含搜索、转场、切频等全部成本。未完成时总源数未知，`total_count`、`clearance_ratio` 保持 null；完成时总源数由覆盖与清除证据推得，并标注来源，不使用离线隐藏真值提前停止。

## 离线验证

```powershell
python -m unittest discover -s tests -v
python src/q3/validate_offline.py --seed 20260912 --random-scenes 40
```

验证包含空场景健全性检查、5米 near 边界、1800米圆周、±1°固定误差、1500米接收边界、同位置多源，以及40组10—20源随机场景。随机误差为按位置确定的有界场，同点不会重采样误差。输入场景、程序文件 SHA-256、参数和每场景指标保存在 `results/tables/q3_offline_validation.json`；扁平结果表为同名 CSV；20源圆周示例交互日志单独放在 `runs/q3/offline/`。

这些结果用于回归与数值核验，不是官方模拟器演练成绩，也不替代三次正式测试及官方加密日志。
