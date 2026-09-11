# 正式代码

- `q1/` 至 `q4/`：对应各分问题的策略、正式实现和运行入口。
- `common/geometry.py`：角度、角域裁剪、凸包、区域直径和第二点候选几何。
- `common/localization.py`：多检测点定位区域及其质量指标。
- `common/simulator_client.py`：四类 HTTP 指令、串行调用、幂等重试、协议状态和基础 JSONL 日志，不包含搜索策略。
- `common/metrics.py`：Q3/Q4 清除比例和时间指标的统一口径。

每个分问题形成可运行实现后以 `main.py` 为入口，路径相对于仓库根目录。官方输入只读，生成内容写入 `results/` 或对应的 `runs/` 独立目录。Notebook 中确认有效的算法必须迁移到这里。
