# CUMCM 2026 — B 题

本仓库用于三人协作完成 2026 年全国大学生数学建模竞赛 B 题。比赛期间请保持仓库为 **Private**，不要在论文、代码、文件名或元数据中写入姓名、学校、赛区等身份信息。

## 当前状态

- 已选题：B 题
- 题面与附件：保存于 `problems/B/`
- 当前阶段：读题、拆分 q1–q4、核对附件内容
- 任务分工与进度：见 [TODO.md](TODO.md)
- 建模思路与结论：见 [notes/modeling.md](notes/modeling.md)
- 实验记录：见 [notes/experiments.md](notes/experiments.md)

每次开始或完成一个工作块时，优先更新 `TODO.md` 和对应笔记，让队友及后续进入仓库的 Codex/ChatGPT 能快速恢复上下文。

## 目录

```text
problems/       A/B/C 题面和官方附件，原文件只读
data/raw/       建模使用的原始数据，程序不得覆盖
data/processed/ 清洗、转换后的建模数据
src/q1..q4/     各分问题的正式 Python 代码
src/common/     数据读取、绘图、公共算法等共享代码
notebooks/      探索性分析，不作为最终唯一实现
results/        程序生成的图片、表格和 Excel
paper/          论文草稿、参考资料、选用图片和最终稿
notes/          思路、假设、符号和实验结论
```

## 环境

建议三人统一使用 Python 3.11：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

仅在代码确实使用新库时更新 `requirements.txt`，避免队友安装不必要依赖。

## 数据与结果约定

- `problems/` 和 `data/raw/` 中的官方原始文件永不由程序修改或覆盖。
- 清洗数据写入 `data/processed/`。
- 所有程序生成内容写入 `results/figures/`、`results/tables/` 或 `results/excel/`。
- 最终采用的图可以复制到 `paper/figures/`；不要手工修改程序生成的原始结果。
- 固定随机种子，记录关键参数、输入文件和输出文件。
- 每个写入 Excel 的脚本都要明确工作表名称、字段含义和单位。

## 三人协作

- 成员 A：题意拆解、模型设计、q1/q2 主责。
- 成员 B：数据清洗、实验验证、q3 主责。
- 成员 C：q4、可视化、论文整合与格式检查。
- 以上只是初始分工，可根据题目调整；关键结论至少由一名非主责队员复核。
- 开工前在 `TODO.md` 认领任务，避免同时编辑同一文件。
- 建议使用短分支：`q1/...`、`q2/...`、`data/...`、`paper/...`。
- 每 2–3 小时同步一次：已确认结论、当前阻塞、下一步接口。
- 提交信息保持简短，例如：`q1: 完成数据预处理`、`paper: 更新模型假设`。

## 推荐运行方式

每个分问题保留清晰入口，例如：

```powershell
python src\q1\main.py
python src\q2\main.py
```

正式代码不要只存在于 Notebook。公共读取、指标和绘图函数放在 `src/common/`，避免四个问题间复制粘贴。

## 最终提交检查

- 电子论文第一页为摘要页，不包含承诺书和编号页。
- 摘要原则上不超过一页；正文不设目录且不超过 30 页。
- 论文 PDF 与支撑材料分别不超过 20 MB。
- 支撑材料包含可运行源码、自主查阅数据和必要中间结果。
- 论文附录中的支撑材料清单与压缩包一致。
- 所有文件无队员、学校、赛区等身份信息。
- 在另一台电脑按 `requirements.txt` 完成一次复现。
- 最终以竞赛官网、赛区和学校发布的最新通知为准。

详细检查项见 [paper/final/README.md](paper/final/README.md)。
