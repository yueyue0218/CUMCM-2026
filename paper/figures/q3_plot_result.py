import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

# 设置中文字体支持
plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 根据文本提取的数据构建数据框
data = [
    # (源数, 策略, 每源均值)
    (10, '正式组_第二版路线候选池', 294.26), (10, '对照组_上一版集中测量与联合路线', 297.91),
    (10, '对照组_普查后清除', 384.49), (10, '对照组_螺旋扫描', 608.01), (10, '对照组_随机游走', 2055.21),
    (10, '对照组_上帝视角', 158.76), (10, '对照组_旧PPO辅助规划', 663.96),
    (11, '正式组_第二版路线候选池', 271.30), (11, '对照组_上一版集中测量与联合路线', 279.10),
    (11, '对照组_普查后清除', 362.64), (11, '对照组_螺旋扫描', 572.28), (11, '对照组_随机游走', 1987.07),
    (11, '对照组_上帝视角', 148.85), (11, '对照组_旧PPO辅助规划', 652.62),
    (12, '正式组_第二版路线候选池', 256.84), (12, '对照组_上一版集中测量与联合路线', 267.66),
    (12, '对照组_普查后清除', 344.05), (12, '对照组_螺旋扫描', 535.18), (12, '对照组_随机游走', 1555.06),
    (12, '对照组_上帝视角', 150.33), (12, '对照组_旧PPO辅助规划', 618.64),
    (13, '正式组_第二版路线候选池', 237.87), (13, '对照组_上一版集中测量与联合路线', 242.15),
    (13, '对照组_普查后清除', 320.59), (13, '对照组_螺旋扫描', 504.28), (13, '对照组_随机游走', 1616.79),
    (13, '对照组_上帝视角', 139.72), (13, '对照组_旧PPO辅助规划', 605.13),
    (14, '正式组_第二版路线候选池', 224.91), (14, '对照组_上一版集中测量与联合路线', 233.17),
    (14, '对照组_普查后清除', 306.58), (14, '对照组_螺旋扫描', 471.24), (14, '对照组_随机游走', 1395.14),
    (14, '对照组_上帝视角', 134.29), (14, '对照组_旧PPO辅助规划', 576.09),
    (15, '正式组_第二版路线候选池', 214.66), (15, '对照组_上一版集中测量与联合路线', 219.80),
    (15, '对照组_普查后清除', 290.49), (15, '对照组_螺旋扫描', 438.08), (15, '对照组_随机游走', 1305.38),
    (15, '对照组_上帝视角', 134.17), (15, '对照组_旧PPO辅助规划', 582.74),
    (16, '正式组_第二版路线候选池', 199.41), (16, '对照组_上一版集中测量与联合路线', 204.10),
    (16, '对照组_普查后清除', 275.39), (16, '对照组_螺旋扫描', 336.87), (16, '对照组_随机游走', 603.50),
    (16, '对照组_上帝视角', 127.07), (16, '对照组_旧PPO辅助规划', 558.44),
]

df = pd.DataFrame(data, columns=['source_count', 'method', 'mean_per_source_s'])

methods = [
    '对照组_上帝视角',
    '正式组_第二版路线候选池',
    '对照组_上一版集中测量与联合路线',
    '对照组_普查后清除',
    '对照组_螺旋扫描',
    '对照组_旧PPO辅助规划',
    '对照组_随机游走'
]

source_counts = list(range(10, 17))
z_pivot = df.pivot(index='method', columns='source_count', values='mean_per_source_s').reindex(methods)

colors = ['#9bbbe1', '#55a868', '#82c8c4', '#e2b978', '#e49cb5', '#be9cd3', '#e38784']

fig = plt.figure(figsize=(14, 8.5))
ax = fig.add_subplot(111, projection='3d')

x = np.array(source_counts)

for y_idx, (method, color) in enumerate(zip(methods, colors)):
    z = z_pivot.loc[method].values

    verts_x = np.concatenate([[x[0]], x, [x[-1]]])
    verts_y = np.full_like(verts_x, y_idx)
    verts_z = np.concatenate([[0], z, [0]])

    verts = [list(zip(verts_x, verts_y, verts_z))]

    poly = Poly3DCollection(verts, facecolors=color, alpha=0.55, edgecolors=color, linewidths=1.2)
    ax.add_collection3d(poly, zs=y_idx, zdir='y')

    ax.plot(x, np.full_like(x, y_idx), z, color=color, linewidth=2, marker='o', markersize=4)

ax.set_xlabel('声源个数 (x)', fontsize=11, labelpad=10)
ax.set_zlabel('每源均值 (秒/源) (z)', fontsize=11, labelpad=10)

ax.set_xticks(source_counts)
ax.set_yticks([])  # 隐藏 Y 轴文字

ax.set_xlim(min(source_counts) - 0.5, max(source_counts) + 0.5)
ax.set_ylim(-0.5, len(methods) - 0.5)
ax.set_zlim(0, z_pivot.values.max() * 1.05)

ax.view_init(elev=25, azim=-55)

legend_labels = [
    '上帝视角 (特权参照)',
    '正式组_第二版路线候选池',
    '上一版集中测量与联合路线',
    '普查后清除',
    '螺旋扫描',
    '旧PPO辅助规划',
    '随机游走'
]

legend_patches = [
    mpatches.Patch(color=color, alpha=0.6, label=label)
    for color, label in zip(colors, legend_labels)
]

ax.legend(
    handles=legend_patches,
    loc='upper left',
    bbox_to_anchor=(0.02, 0.98),
    fontsize=9.5,
    frameon=True,
    facecolor='white',
    edgecolor='#cccccc',
    framealpha=0.9
)

plt.title('10—16源正式组与六类对照组每源均值 (秒/源) 比较', fontsize=14, pad=20)
plt.savefig('new_3d_filled_surface.png', dpi=300, bbox_inches='tight')
plt.show()