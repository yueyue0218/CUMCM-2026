# Q3/Q4 运行记录

`runs/` 保存团队自行生成的演练和正式运行记录；模拟器导出的官方加密日志另存于 `support/`，且不得改名。

每次运行使用独立目录：

```text
runs/q3/practice/2026-09-10_run01/
  config.json
  requests.jsonl
  responses.jsonl
  summary.json
  notes.md
```

`config.json` 至少记录问题、practice/formal、算法名称与版本、Git commit、参数、随机种子、robot_id 的非敏感标识和开始时间。不要提交密码、令牌或完整个人信息。

`requests.jsonl` 与 `responses.jsonl` 按顺序记录接口交互；网络重试应显示相同请求内容和 `request_id`。`summary.json` 使用 `src/common/metrics.py` 的口径，至少记录是否全部清除、清除数量、总数（演练结束后可得）、清除比例、平均定位清除时间、程序运行时间和失败原因。

`JsonlRunLogger` 位于 `src/common/simulator_client.py`。创建运行目录时要求目标目录尚不存在，避免覆盖既有演练证据。

`runs/q3/offline/` 单独保存合成场景验证的示例交互记录，`run_type=offline_synthetic`。它们不属于官方演练或正式测试，不得作为官方成绩或加密日志提交。由 `python src/q3/validate_offline.py` 生成，汇总指标写入 `results/tables/q3_offline_validation.json` 与 CSV。

`runs/q3/training/<时间>_ppo/` 保存本地训练产物：`config.json`、`training.jsonl`、`initial.pt`、`best.pt`、`last.pt`、`training_state.pt`、`trajectory_sample.npz` 及对应源代码快照 ZIP。`last.pt` 与 `training_state.pt` 成对用于续训；`best.pt` 按固定验证集全清除且平均总时间最小选取，实验状态不代表优于基线。初始模型是示范预热之后、PPO之前的权重。

源代码变化默认拒绝续训；显式 `--allow-code-change` 保存新代码快照并重新评估 incumbent。原训练来源与当前选模复评来源分别记录。旧日志/快照保留为历史证据，不能以当前文件哈希替代它们。独立同场景四策略对比由 `evaluate_policy.py` 输出至 `results/tables/`；默认12场合成场景，不是官方成绩。具体命令见 [训练说明](../src/q3/TRAINING.md)。

`runs/q3/efficiency/<时间>_stratified/` 保存10/12/14/16源等权评估，每组场景数相同，各策略共用输入。包含配置、源码ZIP、逐案例输入/动作/证据及归档报告；汇总CSV/JSON写入 `results/tables/q3_efficiency_stratified.*`。失败场景保留，任一场景失败时不报告综合速度分数。

`runs/q3/strategy_controls/<时间>_实验组与五类对照组/` 保存六策略同场实验。每个轨迹文件名均以 `实验组_当前策略_集中测量与联合路线` 或相应 `对照组_<策略名>` 开头，再标源数、种子与逐场轨迹。配置、源码快照、模型哈希和具名总表同时归档；当前最新结果在 `results/tables/Q3_实验组与对照组_策略比较表.*`。初轮普查实现存在空批次缺陷的目录附“不是最终对照结果”说明，修复后完整重跑。
