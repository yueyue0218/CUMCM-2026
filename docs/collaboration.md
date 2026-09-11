# B 题三人协作指南

本指南约定 Q1–Q4、模拟器演练、正式测试和论文整合的协作方式。稳定规范放在这里；实时任务、负责人和阻塞项放在根目录 `TODO.md`；个人临时材料才放入被 Git 忽略的 `scratch/`。

## 1. 责任连续性

大模块保持 owner 连续性，小任务按需要轮换。每个工作包明确：

- 执行人：持续维护实现和实验记录；
- 复核人：独立检查公式、代码、结果和边界情况；
- 第三人：检查上游输入、下游接口和论文口径。

Q3/Q4 主程序、`src/common/simulator_client.py` 等核心模块不要频繁换人或整体重写。其他队员仍可提出算法、运行演练、review 代码、核对结果和补写论文。

## 2. 文件所有权与冲突规避

- Q1–Q4 正式代码分别进入 `src/q1/` 至 `src/q4/`。
- 两个以上问题真正复用的逻辑才进入 `src/common/`。
- Q3/Q4 每次演练写入各自独立的 `runs/.../<date>_runNN/`，不得共用同一日志文件。
- 官方加密日志保持原文件名，分别放入 `support/q3_official_logs/` 或 `support/q4_official_logs/`。
- 多人并行写论文时分别维护 Q1–Q4 章节；`main.tex`、摘要和结论由论文整合 owner 统一修改。
- 不要同时编辑同一个二进制 DOCX、XLSX 或官方附件。

## 3. 任务看板

状态统一使用 `TODO`、`DOING`、`REVIEW`、`DONE`、`BLOCKED`。任务至少写明执行人、复核人、分支、产物和验收条件，例如：

```markdown
- [ ] `DOING` Q3：实现固定覆盖扫描基线
  - 执行：A；复核：B；上下游检查：C
  - 分支：q3/coverage-baseline
  - 产物：src/q3/、runs/q3/practice/2026-09-10_run01/
  - 验收：关键测试通过，日志可复盘，无并发动作
```

## 4. 开发与复核流程

1. 从最新 `main` 建短分支，如 `q1/geometry`、`q3/search-baseline`、`paper/q3-results`。
2. 先实现最小可运行闭环，再优化；Notebook 不能是唯一正式实现。
3. 固定并记录参数、随机种子、Git commit 和运行命令。
4. 执行 `python -m unittest discover -s tests -v`。
5. 更新 `TODO.md` 和对应笔记，将任务交给复核人。
6. 核心内容经 PR 合并；拼写、小型笔记、图标题等小修可直接提交。

必须走 PR 的内容包括 Q1–Q4 模型代码、模拟器客户端、正式测试相关代码，以及论文核心模型和结果。

## 5. Q3/Q4 演练交接

每次演练至少能追踪：算法版本、Git commit、参数、随机种子、请求序列、响应、清除数、是否全部清除、平均定位清除时间、程序运行时间和失败原因。建议目录包含：

```text
config.json
requests.jsonl
responses.jsonl
summary.json
notes.md
```

失败演练也要保留摘要和原因。比较策略时使用同一指标口径，并同时关注清除率、平均表现、波动和最差案例。

## 6. 模拟器安全规则

- 动作严格串行，收到完整响应后才能发送下一个新动作。
- 新动作使用新 `request_id`；只有不确定是否成功的网络重试才复用完全相同的请求。
- 同时检查 HTTP 状态和 `accepted`。
- `near` 优先考虑清除；`no_signal` 不能直接证明频道无源。
- `/clear` 不切换当前检测频道。
- 策略读取 `/enter` 返回的 `remaining_real_duration_s`，预留安全退出时间并主动 `/exit`。

正式测试前必须逐项完成 [formal_test_checklist.md](formal_test_checklist.md)，并以 `q3-formal-vN` 或 `q4-formal-vN` 标签冻结版本。
