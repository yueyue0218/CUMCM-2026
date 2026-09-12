# Q4 最终代码交接

按用户最终确认范围，本 PR 仅包含最终算法 informed、三个对照 refined/adaptive/seven_grid、必要的测试和评估代码，以及紧凑的最终统计。不提交中间训练集、训练模型、未采用的研究代码、过程计划或大量原始日志，也不制作中间资料归档包。

- 正式算法与对照：`src/q4/`，逐组文件映射见 `src/q4/README.md`。
- 高风险测试：`tests/test_q4*.py`。
- 最终结果：`results/tables/q4_final_comparison.md/.json`，保留固定 140 场的逐场配对指标，不包含训练样本。
- 运行时新生成的比较记录：`runs/q4/comparison/`，本地生成，不预先纳入 PR。
- 过程结论和协作状态：`notes/experiments.md`、`TODO.md`，状态 REVIEW，复核人和上下游检查人待队友认领。

分支为 `codex/q4-complete-runtime`，以最新 `origin/codex/q3-complete-runtime` 的 `caf0dda` 为基点。Q4 复用的公共客户端和 Q3 辅助尚在 [PR #9](https://github.com/yueyue0218/CUMCM-2026/pull/9)，未全部进入 main，因此 PR 暂向该运行分支提出；先合入 #9，再调整本 PR 的目标为 main 并核对测试。Q4 建模稿继续由 [PR #8](https://github.com/yueyue0218/CUMCM-2026/pull/8) 独立维护，本 PR 不覆盖该稿。

最终算法的行为没有因收缩提交范围而修改，Python 文件只做换行与尾部空行规范化；评估与同场对比工具脱离中间研究模块独立运行。原工作区中 Q2/Q3 的未提交改动不在本 PR 中。官方模拟器尚未运行，总体 300–399 秒目标未达到。
