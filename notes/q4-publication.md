# Q4 最终代码交接

按用户最终确认范围，本 PR 包含最终算法 informed、参照 Q3 设立的具名对照、必要测试评估代码和紧凑结果。主表六组为 informed、refined、census_then_clear、spiral_scan、random_walk、oracle；adaptive/seven_grid 保留为可选补充。不提交中间训练集、训练模型、未采用的研究代码、过程计划或大量原始日志，也不制作中间资料归档包。
截至2026-09-13的合入状态：Q3/公共模块PR #9已合入main，随后Q4代码PR #10已合入main；Q4论文PR #8已同步上述实现，仍以main为目标并待审核。代码交付没有等待论文PR合入的前置条件。

本记录的代码交付范围指已合入的[PR #10](https://github.com/yueyue0218/CUMCM-2026/pull/10)：最终算法 informed、三个对照 refined/adaptive/seven_grid、必要的测试和评估代码，以及紧凑的最终统计。不包含中间训练集、训练模型、未采用的研究代码、过程计划或大量原始日志，也未制作中间资料归档包。当前[PR #8](https://github.com/yueyue0218/CUMCM-2026/pull/8)只修订论文及协作说明。

- 正式算法与对照：`src/q4/`，逐组文件映射见 `src/q4/README.md`。
- 高风险测试：`tests/test_q4*.py`。
- 最终结果：`results/tables/q4_final_comparison.md/.json`，保留固定 140 场的逐场配对指标，不包含训练样本。
- Q3 对照迁移与代码映射：`src/q4/STRATEGY_CONTROLS.md`；六组新 80 场主表为 `results/tables/Q4_实验组与对照组_策略比较表.md/.csv/.json`，逐组 CSV 文件名含算法标识。
- 运行时新生成的比较记录：`runs/q4/comparison/`，本地生成，不预先纳入 PR。
- 过程结论和协作状态：`notes/experiments.md`、`TODO.md`；代码交付已合入，跨环境复现及Q4官方验证仍按各自任务跟踪。

代码分支`codex/q4-complete-runtime`最初从Q3运行分支的`caf0dda`开发，并曾临时以运行分支为PR目标；这属于合入前的历史。已核验的合入记录如下：

| 交付 | 当前状态 | main中的合并提交 |
|---|---|---|
| [PR #9：Q3与公共模块](https://github.com/yueyue0218/CUMCM-2026/pull/9) | 已合入main | `c2cf3f29da0ffd3c85860a347dcd9b1c6b7109b5` |
| [PR #10：Q4最终代码](https://github.com/yueyue0218/CUMCM-2026/pull/10) | 已合入main | `db2816a559b0fb1bdf9b605642ec8e5bc5144ae6` |
| [PR #8：Q4论文与建模](https://github.com/yueyue0218/CUMCM-2026/pull/8) | 已同步main，待审核 | 尚未合入 |

文稿当前核验的main基点为`c3a8a55ae74a8d88d05aedaad304219e92a25330`，已包含前两项合并；它与历史开发基点的用途不同。源码指纹与文稿映射见[对齐记录](q4-code-model-alignment.md)。

原 Q3 运行 [PR #9](https://github.com/yueyue0218/CUMCM-2026/pull/9) 和 Q4 最终运行 [PR #10](https://github.com/yueyue0218/CUMCM-2026/pull/10) 均已合并。本次新增对照分支为 `codex/q4-strategy-controls`，从最新 main（`c3a8a55`）建立，后续 PR 直接面向 main。Q4 建模稿继续由 [PR #8](https://github.com/yueyue0218/CUMCM-2026/pull/8) 独立维护，本 PR 不覆盖该稿。
最终算法的行为没有因收缩提交范围而修改，Python 文件只做换行与尾部空行规范化；评估与同场对比工具脱离中间研究模块独立运行。原工作区中 Q2/Q3 的未提交改动未纳入代码PR #10。官方模拟器尚未运行，总体 300–399 秒目标未达到。

“140场、1820源全清”仅指固定140场离线合成配对测试：informed与refined各自清除同一组1820个源，仓库保存其紧凑统计，不能将两组累加为3640个不同源。该结果不是Q4官方模拟器成绩；Q4官方闭环测试与官方加密日志仍待取得，不由已经存在的Q3官方记录替代。
