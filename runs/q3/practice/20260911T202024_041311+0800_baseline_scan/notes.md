# Q3-v0 Practice Run 02

- 模拟器真实全向源数：10
- detected_count：10
- measure_count：91
- direction_count：10
- no_signal_count：81
- near_count：0
- final_virtual_time_s：2339
- completed_full_cover：true
- failure_reason：null
- exit_failure：null

## 独立时间核验

- 移动距离：9000 m
- 移动时间：1800 s
- 测量时间：91 × 5 = 455 s
- 频道切换：84 次 = 84 s
- 总虚拟时间：1800 + 455 + 84 = 2339 s

## 可复现性

本局记录的 Git commit：
5dec938c6e0950c3d56239b98b5a81b336e2cb01

运行时 git_dirty=true，因此该局用于真实功能验证，
但尚不作为“仅凭 commit 即可完全复现”的 clean run。