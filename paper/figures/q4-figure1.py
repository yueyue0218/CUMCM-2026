import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置全局绘图样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']  # 保证中文字符显示
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(15, 8), dpi=300)

# 定义模块及配色 (RGBA)
boxes = [
    # 阶段 1
    {"id": "S1", "text": "高维初始化\n(Np=2000)", "x": 1, "y": 6, "w": 2.2, "h": 1.2, "color": "#E3F2FD",
     "edge": "#1E88E5"},
    {"id": "S1_2", "text": "移动至骨干点\n(大圆域/扫描点)", "x": 4, "y": 6, "w": 2.2, "h": 1.2, "color": "#E3F2FD",
     "edge": "#1E88E5"},

    # 阶段 2
    {"id": "S2_1", "text": "全频段信号扫描\n(观测向量 z_i)", "x": 1, "y": 3.5, "w": 2.2, "h": 1.2, "color": "#E8F5E9",
     "edge": "#43A047"},
    {"id": "S2_2", "text": "似然概率计算\np(z_i|Θ_j)", "x": 4, "y": 3.5, "w": 2.2, "h": 1.2, "color": "#E8F5E9",
     "edge": "#43A047"},
    {"id": "S2_3", "text": "粒子权重更新\n与归一化", "x": 7, "y": 3.5, "w": 2.2, "h": 1.2, "color": "#E8F5E9",
     "edge": "#43A047"},
    {"id": "S2_4", "text": "协方差矩阵 Σ\n与 N_eff 校验", "x": 10, "y": 3.5, "w": 2.2, "h": 1.2, "color": "#E8F5E9",
     "edge": "#43A047"},

    # 阶段 3
    {"id": "S3_1", "text": "动态双环候选点集\nX_cand (内外环)", "x": 1, "y": 1, "w": 2.2, "h": 1.2, "color": "#FFF3E0",
     "edge": "#FB8C00"},
    {"id": "S3_2", "text": "综合效用函数 U(x)\n(熵减/成本/接近度)", "x": 4, "y": 1, "w": 2.2, "h": 1.2,
     "color": "#FFF3E0", "edge": "#FB8C00"},
    {"id": "S3_3", "text": "开放路径 DP/TSP\n路径规划", "x": 7, "y": 1, "w": 2.2, "h": 1.2, "color": "#FFF3E0",
     "edge": "#FB8C00"},

    # 阶段 4 / 出口
    {"id": "S4_1", "text": "两级光学检测\n& 清除状态标定", "x": 10, "y": 1, "w": 2.2, "h": 1.2, "color": "#F3E5F5",
     "edge": "#8E24AA"},
    {"id": "S4_2", "text": "解析网格覆盖\n后备分支 (Fallback)", "x": 13, "y": 2.25, "w": 2.4, "h": 1.2,
     "color": "#FFEBEE", "edge": "#E53935"},
    {"id": "END", "text": "全频道清除\n完备退出", "x": 13, "y": 4.75, "w": 2.2, "h": 1.2, "color": "#ECEFF1",
     "edge": "#546E7A"}
]

# 绘制文本框
box_dict = {}
for b in boxes:
    rect = patches.FancyBboxPatch(
        (b["x"], b["y"]), b["w"], b["h"],
        boxstyle="round,pad=0.1,rounding_size=0.15",
        linewidth=1.5, edgecolor=b["edge"], facecolor=b["color"]
    )
    ax.add_patch(rect)
    ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] / 2, b["text"],
            ha='center', va='center', fontsize=9, fontweight='bold', color='#212121')
    box_dict[b["id"]] = b

# 定义并绘制连接线条
arrows = [
    ("S1", "S1_2"), ("S1_2", "S2_1"),
    ("S2_1", "S2_2"), ("S2_2", "S2_3"), ("S2_3", "S2_4"),
    ("S2_4", "S3_1"), ("S3_1", "S3_2"), ("S3_2", "S3_3"),
    ("S3_3", "S4_1"), ("S4_1", "END"), ("S4_1", "S4_2"),
    ("S4_2", "S2_1")
]

for src, dst in arrows:
    b1, b2 = box_dict[src], box_dict[dst]

    # 计算起点与终点连接坐标
    if b1["x"] < b2["x"] and b1["y"] == b2["y"]:
        x1, y1 = b1["x"] + b1["w"], b1["y"] + b1["h"] / 2
        x2, y2 = b2["x"], b2["y"] + b2["h"] / 2
    elif b1["y"] > b2["y"]:
        x1, y1 = b1["x"] + b1["w"] / 2, b1["y"]
        x2, y2 = b2["x"] + b2["w"] / 2, b2["y"] + b2["h"]
    elif b1["y"] < b2["y"]:
        x1, y1 = b1["x"] + b1["w"] / 2, b1["y"] + b1["h"]
        x2, y2 = b2["x"] + b2["w"] / 2, b2["y"]
    else:
        x1, y1 = b1["x"], b1["y"] + b1["h"] / 2
        x2, y2 = b2["x"] + b2["w"], b2["y"] + b2["h"] / 2

    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#424242", lw=1.5, mutation_scale=12))

# 调整视图界限与样式
ax.set_xlim(0, 16)
ax.set_ylim(0.5, 7.5)
ax.axis('off')
plt.title("5.4.1 NS-PPO 决策与求解架构流程图", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()

# 保存高质量矢量图与 PNG
plt.savefig("section_5_4_1_architecture.svg", format="svg", bbox_inches='tight')
plt.savefig("section_5_4_1_architecture.png", dpi=300, bbox_inches='tight')
plt.show()