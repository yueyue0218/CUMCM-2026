# Q3 可训练策略实现计划

目标：把已有确定性闭环扩展为可训练、可保存、可加载和独立比较的候选动作策略，支持无网络规划、独立PPO、PPO辅助规划；对比原complete基线，保留完整耗时目标和硬约束。

依据：`paper/drafts/B-q3：PPO训练方案.md`、主稿第6—9节，以及用户要求接入训练以寻找较优方案。原有未提交的Q2文件保留，Q2函数可只读复用，不改写。工作在已有 `codex/q3-complete-runtime` 分支，产物按协作指南归档。

## 固定接口和任务划分

1. 世界模型（独立模块 `src/q3/world_model.py`）
   - 根据逐频道真实历史与Q1外包构造位置—半径粒子信念；复用Q2采样/似然工具，说明独立误差近似，不伪称校准完成。
   - `build_belief(discovery, particle_count=64, seed=0)` 返回 `ChannelBelief` 或不可用；`predict_measurement(discovery, position, particle_count=64, seed=0)` 返回响应概率、预测直径及可用性。不得读真实world。
   - 有限反馈场景预测经Q1评估新的定位状态；仅作排名，不作清除认证。保留概率失效标志。
   - `open_clear_route(start, positions: dict[int, tuple[float,float]]) -> list[int]`：小规模精确开放路径，大规模有界计算启发式。
   - 新测试文件 `tests/test_q3_world_model.py`。
2. 循环PPO（独立模块 `src/q3/ppo.py`）
   - `RecurrentCandidatePolicy(global_dim, channel_dim, candidate_dim, hidden_size=64)`。
   - `forward(features, hidden=None)`，features字典含 global:[G]、channels:[20,C]、candidates:[A,F]、candidate_channels:[A]（0起）、mask:[A]；返回 logits:[A]、value:标量、new_hidden:[H]。候选数可变。
   - 轨迹保存完整原始输入快照、实际采样行号、旧概率/价值、奖励和终止；GRU每轮更新从每回合零隐状态完整重算，不能用旧隐状态冒充当前参数状态。
   - 完整回合gamma=1、GAE、PPO ratio clip、梯度范数/KL保护。强制单一动作无策略梯度，后备全部耗时进入奖励及价值目标；不截断尾段，不混用规划器轨迹更新PPO。
   - 保存/安全加载网络参数与结构版本，新增 `tests/test_q3_ppo.py`。
3. 主流程（本任务）
   - `src/q3/adaptive_control.py`：显式候选列表、Q1认证缓存、有限全局探索预算、逐频道覆盖证据、独立后备；自由阶段可选搜索、Q2横纵向补测、认证清除和解析后备，允许跨频道调度。
   - 候选输入含动作成本、已观察频道状态、定位尺度、后备证明及路线提示，不含场景种子或隐藏真值。概率特征仅由真实历史构造；预测分支不改变真状态。
   - `src/q3/training_env.py`：分布明确的10—16源训练场景、固定误差场、按最终±1°硬界的两位小数量化、单动作step、真实证据终止。低源数课程可用于调试，验收必须独立完整场景。
   - `src/q3/train_ppo.py`：种子、超参数、完整回合采样、训练日志、checkpoint、续训、独立验证选模。
   - `src/q3/evaluate_policy.py`：同场景比较complete/无网络规划/PPO/组合；输出逐案例耗时、清除率和配对改善，训练与测试种子独立，不能保证训练必然优于基线。
   - 实际main入口支持加载checkpoint，推理使用同一候选/特征版本。

## 验证与交付

- [x] 世界模型后验/分支隔离/路线测试。
- [x] PPO掩码、旧输入快照、强制尾段成本、GRU重算、参数实际更新、保存加载测试。
- [x] 候选合法性、证据终止、全局预算、同点固定误差和隐藏信息隔离测试。
- [x] 完成CPU小规模训练并生成模型；继续训练命令可直接执行。
- [x] 独立场景比较结果与日志落到 `results/tables/`、`runs/q3/training/`；模型在独立训练目录。
- [x] 全套测试与独立代码审查；更新README、TODO、实验记录。

工程选择：使用已安装的CPU PyTorch，64维网络为起点，不新增大框架。训练先完成短程验证，再由日志与独立评估决定增加轮数；不会把短程训练当作收敛或最优性证明。

交付证据：159项回归通过；独立审查修复测试集与示范种子交叉、续训验证集漂移、失败场景伪改善、现实时间后备预留、源码溯源和随机流恢复。主训练16回合示范+70回合PPO，固定验证4场；新测试12场全部完成，hybrid平均虚拟耗时比基线降低2.96%，纯PPO尚未优于基线。操作命令、实际算法范围及限制见 `src/q3/TRAINING.md`。代码已完成本地实现与检查，团队看板保持REVIEW供队友验收。
