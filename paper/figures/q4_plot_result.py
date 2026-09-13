import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

# 设置字体支持
plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 完整数据硬编码
data = [
    ('已知位置参考', 10, 162.312326), ('已知位置参考', 11, 158.160692), ('已知位置参考', 12, 143.739941),
    ('已知位置参考', 13, 148.109112), ('已知位置参考', 14, 135.499726), ('已知位置参考', 15, 131.730720),
    ('已知位置参考', 16, 130.667217),
    ('正式组 informed', 10, 574.373403), ('正式组 informed', 11, 541.013639), ('正式组 informed', 12, 486.166646),
    ('正式组 informed', 13, 445.002211), ('正式组 informed', 14, 416.355720), ('正式组 informed', 15, 386.117803),
    ('正式组 informed', 16, 315.382214),
    ('对照 refined', 10, 590.056386), ('对照 refined', 11, 555.387460), ('对照 refined', 12, 497.764407),
    ('对照 refined', 13, 456.565865), ('对照 refined', 14, 428.123374), ('对照 refined', 15, 396.162462),
    ('对照 refined', 16, 330.535358),
    ('初版自适应', 10, 634.418464), ('初版自适应', 11, 589.429471), ('初版自适应', 12, 520.705876),
    ('初版自适应', 13, 486.093857), ('初版自适应', 14, 443.714878), ('初版自适应', 15, 419.197813),
    ('初版自适应', 16, 333.499148),
    ('普查后清除', 10, 776.919950), ('普查后清除', 11, 709.727231), ('普查后清除', 12, 654.977112),
    ('普查后清除', 13, 617.800480), ('普查后清除', 14, 575.520064), ('普查后清除', 15, 538.309984),
    ('普查后清除', 16, 512.016220),
    ('螺旋与方向覆盖', 10, 1476.107328), ('螺旋与方向覆盖', 11, 1350.943500), ('螺旋与方向覆盖', 12, 1226.778149),
    ('螺旋与方向覆盖', 13, 1112.387744), ('螺旋与方向覆盖', 14, 1014.797027), ('螺旋与方向覆盖', 15, 942.938594),
    ('螺旋与方向覆盖', 16, 437.218786),
    ('七站与网格后备', 10, 1772.658943), ('七站与网格后备', 11, 1617.374998), ('七站与网格后备', 12, 1464.721838),
    ('七站与网格后备', 13, 1370.498245), ('七站与网格后备', 14, 1265.414178), ('七站与网格后备', 15, 1170.879411),
    ('七站与网格后备', 16, 714.519198),
    ('随机游走', 10, 2416.271162), ('随机游走', 11, 2180.357629), ('随机游走', 12, 1973.714238),
    ('随机游走', 13, 1786.533317), ('随机游走', 14, 1595.912916), ('随机游走', 15, 1490.304750),
    ('随机游走', 16, 1049.130683)
]

df = pd.DataFrame(data, columns=['算法', '源数', '每源均值（秒/源）'])

methods = [
    '已知位置参考', '正式组 informed', '对照 refined', '初版自适应',
    '普查后清除', '螺旋与方向覆盖', '七站与网格后备', '随机游走'
]

method_labels = [
    '已知位置参考 (上帝视角)', '正式组 informed', '对照 refined', '初版自适应',
    '普查后清除', '螺旋与方向覆盖', '七站与网格后备', '随机游走'
]

source_counts = list(range(10, 17))
z_pivot = df.pivot(index='算法', columns='源数', values='每源均值（秒/源）').reindex(methods)

# 1:1 还原示例图配色方案
colors = [
    '#A0BCE2',  # 浅蓝
    '#58A868',  # 绿
    '#82C8C3',  # 青绿
    '#E7C58A',  # 浅黄/沙色
    '#EEA3B8',  # 浅粉
    '#BFA5D7',  # 浅紫
    '#CFC27A',  # 橄榄黄
    '#E88288'  # 珊瑚红
]

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')

x = np.array(source_counts)

for y_idx, (method, color) in enumerate(zip(methods, colors)):
    z = z_pivot.loc[method].values

    verts_x = np.concatenate([[x[0]], x, [x[-1]]])
    verts_y = np.full_like(verts_x, y_idx)
    verts_z = np.concatenate([[0], z, [0]])

    verts = [list(zip(verts_x, verts_y, verts_z))]

    # 填充面 alpha 设置为 0.42，与示例图完全对齐
    poly = Poly3DCollection(verts, facecolors=color, alpha=0.42, edgecolors=color, linewidths=1.2)
    ax.add_collection3d(poly, zs=y_idx, zdir='y')

    # 顶部折线及数据点
    ax.plot(x, np.full_like(x, y_idx), z, color=color, linewidth=2.2, marker='o', markersize=4.5, alpha=0.95)

ax.set_xlabel('声源个数 (x)', fontsize=11, labelpad=10)
ax.set_zlabel('每源均值 (秒/源) (z)', fontsize=11, labelpad=10)

ax.set_xticks(source_counts)
ax.set_yticks([])

ax.set_xlim(min(source_counts) - 0.5, max(source_counts) + 0.5)
ax.set_ylim(-0.5, len(methods) - 0.5)
ax.set_zlim(0, 2500)

ax.view_init(elev=22, azim=-55)

legend_patches = [
    mpatches.Patch(color=color, alpha=0.42, label=label)
    for color, label in zip(colors, method_labels)
]

ax.legend(
    handles=legend_patches,
    loc='upper left',
    bbox_to_anchor=(0.01, 0.98),
    fontsize=9.5,
    frameon=True,
    facecolor='white',
    edgecolor='#cccccc',
    framealpha=0.9
)

plt.title('10—16源全部八组策略每源均值 (秒/源) 3D填充面图比较', fontsize=14, pad=18)
plt.tight_layout()
plt.savefig('matched_sample_3d_filled_surface.png', dpi=300, bbox_inches='tight')
plt.show()