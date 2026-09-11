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
