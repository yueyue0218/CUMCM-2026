# Q4 最终代码交接

按用户最终确认范围，本 PR 包含最终算法 informed、参照 Q3 设立的具名对照、必要测试评估代码和紧凑结果。主表六组为 informed、refined、census_then_clear、spiral_scan、random_walk、oracle；adaptive/seven_grid 保留为可选补充。不提交中间训练集、训练模型、未采用的研究代码、过程计划或大量原始日志，也不制作中间资料归档包。

- 正式算法与对照：`src/q4/`，逐组文件映射见 `src/q4/README.md`。
- 高风险测试：`tests/test_q4*.py`。
- 最终结果：`results/tables/q4_final_comparison.md/.json`，保留固定 140 场的逐场配对指标，不包含训练样本。
- Q3 对照迁移与代码映射：`src/q4/STRATEGY_CONTROLS.md`；六组新 80 场主表为 `results/tables/Q4_实验组与对照组_策略比较表.md/.csv/.json`，逐组 CSV 文件名含算法标识。
- 运行时新生成的比较记录：`runs/q4/comparison/`，本地生成，不预先纳入 PR。
- 过程结论和协作状态：`notes/experiments.md`、`TODO.md`，状态 REVIEW，复核人和上下游检查人待队友认领。

原 Q3 运行 [PR #9](https://github.com/yueyue0218/CUMCM-2026/pull/9) 和 Q4 最终运行 [PR #10](https://github.com/yueyue0218/CUMCM-2026/pull/10) 均已合并。本次新增对照分支为 `codex/q4-strategy-controls`，从最新 main（`c3a8a55`）建立，后续 PR 直接面向 main。Q4 建模稿继续由 [PR #8](https://github.com/yueyue0218/CUMCM-2026/pull/8) 独立维护，本 PR 不覆盖该稿。

最终算法的行为没有因收缩提交范围而修改，Python 文件只做换行与尾部空行规范化；评估与同场对比工具脱离中间研究模块独立运行。原工作区中 Q2/Q3 的未提交改动不在本 PR 中。官方模拟器尚未运行，总体 300–399 秒目标未达到。
