import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

# 1. 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 2. 读取 JSON 文件数据
json_path = 'Q3_实验组与对照组_策略比较表.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

records = []
for strat_key, strat_info in data['aggregate'].items():
    strat_name = strat_info.get('stem', strat_key)
    for source_count, metrics in strat_info['groups'].items():
        records.append({
            'method': strat_name,
            'source_count': int(source_count),
            'mean_per_source_s': metrics['mean_per_source_s']
        })

df = pd.DataFrame(records)

# 策略列表与对应的柔和配色（莫兰迪浅色系）
methods = [
    '对照组_上帝视角',
    '实验组_当前策略_集中测量与联合路线',
    '对照组_普查后清除',
    '对照组_螺旋扫描',
    '对照组_旧PPO辅助规划',
    '对照组_随机游走'
]

source_counts = [10, 12, 14, 16]
z_pivot = df.pivot(index='method', columns='source_count', values='mean_per_source_s').reindex(methods)

pastel_colors = [
    '#9bbbe1',  # 柔和浅蓝
    '#82c8c4',  # 柔和青绿
    '#e2b978',  # 柔和黄/橙
    '#e49cb5',  # 柔和粉红
    '#be9cd3',  # 柔和淡紫
    '#e38784'  # 柔和浅红
]

# 3. 创建三维画布
fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(111, projection='3d')

x = np.array(source_counts)

# 4. 逐层绘制三维填充面
for y_idx, (method, color) in enumerate(zip(methods, pastel_colors)):
    z = z_pivot.loc[method].values

    verts_x = np.concatenate([[x[0]], x, [x[-1]]])
    verts_y = np.full_like(verts_x, y_idx)
    verts_z = np.concatenate([[0], z, [0]])

    verts = [list(zip(verts_x, verts_y, verts_z))]

    poly = Poly3DCollection(verts, facecolors=color, alpha=0.55, edgecolors=color, linewidths=1.2)
    ax.add_collection3d(poly, zs=y_idx, zdir='y')

    ax.plot(x, np.full_like(x, y_idx), z, color=color, linewidth=2, marker='o', markersize=4)

# 5. 坐标轴设置
ax.set_xlabel('声源个数 (x)', fontsize=11, labelpad=10)
ax.set_zlabel('单声源平均用时/s (z)', fontsize=11, labelpad=10)

ax.set_xticks(source_counts)
ax.set_yticks([])  # 隐藏 Y 轴上的刻度与文字

ax.set_xlim(min(source_counts) - 0.5, max(source_counts) + 0.5)
ax.set_ylim(-0.5, len(methods) - 0.5)
ax.set_zlim(0, z_pivot.values.max() * 1.05)

ax.view_init(elev=25, azim=-55)

# 6. 在左上角创建独立图例（参照示例图）
y_labels = [m.replace('对照组_', '').replace('实验组_当前策略_', '') for m in methods]

legend_patches = [
    mpatches.Patch(color=color, alpha=0.6, label=label)
    for color, label in zip(pastel_colors, y_labels)
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

plt.title('不同策略下声源个数与单声源平均用时的三维填充面图', fontsize=14, pad=20)
plt.savefig('3d_filled_surface_top_left_legend.png', dpi=300, bbox_inches='tight')
plt.show()