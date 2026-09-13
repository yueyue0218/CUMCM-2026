# CUMCM 2026 B 题

本仓库用于三人协作完成 2026 年全国大学生数学建模竞赛 B 题“无线电干扰源的快速自动定位与清除”。比赛期间保持仓库为 Private，不在论文、代码、路径、日志或元数据中写入姓名、学校、赛区等身份信息。

## 当前状态

- 已确认题面和两个官方附件，关键协议规则已整理。
- 已建立公共几何、定位、指标和模拟器客户端的最小框架。
- Q3 已补全确定性搜索、定位、清除闭环，复用 Q1 认证接口；离线验证结果已生成，待队友复核及官方模拟器联调。
- 实时任务见 [TODO.md](TODO.md)，建模结论见 [notes/modeling.md](notes/modeling.md)。

## 四个问题与代码

- `src/q1/`：带 ±1° 示向误差的定位区域、直径与覆盖判据。
- `src/q2/`：第二检测点候选区域和稳健选点评价。
- `src/q3/`：全向干扰源搜索、定位、清除和停止策略。
- `src/q4/`：在 Q3 基础上处理定向源盲区和多方向观测。
- `src/common/`：两个以上问题共享的几何、定位、指标和模拟器接口；不放搜索策略。

各问题入口形成后统一从仓库根目录运行，例如：

```powershell
python src\q1\main.py
python src\q2\main.py
python src\q3\main.py
python -m src.q4.main --robot-id "当前登录队号"
```

Q3 运行方式为 `python src/q3/main.py --robot-id "当前登录队号"`，具体参数和离线验证见 [Q3 说明](src/q3/README.md)。其他问题入口按各自实现进度验收，不应把 Notebook 当作最终唯一实现。

## 环境与测试

建议统一使用 Python 3.11：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`tests/` 只覆盖本题高风险逻辑：角度跨界、±1° 角域、近平行交会、定位区域、第二点几何关系，以及请求幂等、状态更新和三类检测响应。

## Q3/Q4 模拟器

先在模拟器中登录并启动正确的演练模块，等待界面显示接口就绪。统一使用 `src/common/simulator_client.py` 连接默认地址 `http://127.0.0.1:2026`，不要在 Q3/Q4 重复实现 HTTP 调用。

客户端保证：

- `/enter`、`/measure`、`/clear`、`/exit` 严格串行；
- 新动作使用新 `request_id`，网络结果不确定时以完全相同请求重试；
- 重试耗尽或响应无法核实时保留待核实动作，停止发送新动作（包括 `/exit`），摘要记录最后确认时刻；
- 同时检查 HTTP 状态和 `accepted`；
- 只有 `direction` 读取 `svd_deg`；`near` 不含示向度，应优先考虑 `clear`；
- `no_signal` 只作为观测，不能直接判定频道不存在；
- `/clear` 的半径为 20 m，且不改变当前检测频道；
- 保存 `/enter` 返回的 `remaining_real_duration_s`，策略不得写死 1200 秒。

演练应遵循“搜索 → 发现频道 → 获取示向度 → 选择下一检测点 → 更新定位区域 → 判断是否可清除 → clear → 继续搜索”的闭环。具体策略放在 Q3/Q4，不与客户端耦合。

## 运行和正式日志

- `runs/q3/practice/`、`runs/q4/practice/`：每次演练的配置、自建请求/响应日志、摘要和笔记。
- `runs/q3/formal/`、`runs/q4/formal/`：正式运行的版本与结果记录。
- `support/q3_official_logs/`、`support/q4_official_logs/`：模拟器导出的官方加密日志，必须保持原文件名。

目录格式和字段见 [runs/README.md](runs/README.md)。Q3、Q4 正式测试各只有 3 次机会；开始前逐项完成 [正式测试检查清单](docs/formal_test_checklist.md)，并用 `q3-formal-vN` 或 `q4-formal-vN` 标签冻结代码版本。

## 目录

```text
problems/B/                 官方题面和附件，只读
src/q1/ ... src/q4/         四问正式代码
src/common/                 跨问题公共实现
tests/                      关键回归测试
runs/q3/、runs/q4/          自建演练与正式运行记录
support/                    官方加密日志
results/                    程序生成的图、表和 Excel
paper/drafts/latex/         可选 LaTeX 论文骨架
notes/                      建模、假设、符号和实验结论
docs/                       团队协作和正式测试规范
data/                       非核心；仅保存确有需要的原始/处理数据
scratch/                    个人临时文件，不承担团队规范职责
```

官方文件和 `data/raw/` 不得被程序修改。生成结果写入 `results/`；论文采用的图可复制到 `paper/figures/`。

## 三人协作

Q3 已提供本地训练与模型评估入口：见 [Q3 训练说明](src/q3/TRAINING.md)，包含新训练、续训、四策略比较和模拟器加载命令。实验模型及日志位于 `runs/q3/training/`，独立比较表位于 `results/tables/q3_policy_comparison.*`。

大模块保持 owner 连续性，每个工作包使用“执行人 + 复核人 + 第三人检查上下游”。核心模型代码、模拟器客户端、正式测试代码和论文核心结果走短分支与 PR；拼写、小型 notes 和图标题等小修可简化流程。完整规则见 [docs/collaboration.md](docs/collaboration.md)。

论文框架位于 `paper/drafts/latex/`，Q1–Q4 章节已拆分以减少并行冲突。最终提交要求见 [paper/final/README.md](paper/final/README.md)，并以竞赛官网、赛区和学校的最新通知为准。

Q3最新默认策略为 `efficient`，10/12/14/16源各20场等权验证的每源平均时间257.2秒，全部清除80/80。新旧策略对比与命令见 [效率改进与分层结果](src/q3/EFFICIENCY.md)；这是本地合成测试，尚未作为官方成绩。

Q4 最终算法为 `informed`，附 `refined`、`adaptive`、`seven_grid` 三个具名对照。正式代码和同场评估命令见 [Q4 说明](src/q4/README.md)，最终对比见 [Q4 结果](results/tables/q4_final_comparison.md)。本工作包不包含中间训练集、训练模型或探索日志。
