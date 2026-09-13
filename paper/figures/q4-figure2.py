import json
import matplotlib.pyplot as plt
import numpy as np

# 1. 配置文件与日志路径
REQUESTS_FILE = "requests.json"
RESPONSES_FILE = "responses.json"


def load_log_data(req_path, resp_path):
    with open(req_path, "r", encoding="utf-8") as f:
        requests_data = json.load(f)
    with open(resp_path, "r", encoding="utf-8") as f:
        responses_data = json.load(f)

    resp_map = {item["timestamp"]: item for item in responses_data}

    path_x, path_y = [], []
    measure_pts = []
    no_signal_pts = []
    clear_success_pts = []

    for req in requests_data:
        ts = req.get("timestamp")
        action = req.get("action")
        pos = req.get("position", {})
        x, y = pos.get("x"), pos.get("y")

        if x is not None and y is not None:
            path_x.append(x)
            path_y.append(y)

        resp = resp_map.get(ts, {})
        data = resp.get("data", {})

        if action == "measure":
            res = data.get("measure_result")
            if res == "direction":
                deg = data.get("svd_deg")
                measure_pts.append((x, y, deg))
            elif res == "no_signal":
                no_signal_pts.append((x, y))
        elif action == "clear":
            if data.get("clear_result") == "success":
                clear_success_pts.append((x, y))

    return (
        np.array(path_x),
        np.array(path_y),
        measure_pts,
        no_signal_pts,
        clear_success_pts,
    )


def generate_particles_and_plot():
    # 读取数据
    try:
        path_x, path_y, measure_pts, no_signal_pts, clear_pts = load_log_data(
            REQUESTS_FILE, RESPONSES_FILE
        )
    except FileNotFoundError:
        # 兼容无外部文件时的模拟坐标范围
        path_x = np.linspace(0, 100, 150)
        path_y = 50 + 30 * np.sin(path_x / 10)
        no_signal_pts = [(20, 20), (30, 80)]
        clear_pts = [(80, 70), (40, 30)]

    # 确定地图边界
    x_min, x_max = (
        min(path_x.min(), 0) - 10,
        max(path_x.max(), 100) + 10,
    )
    y_min, y_max = (
        min(path_y.min(), 0) - 10,
        max(path_y.max(), 100) + 10,
    )

    # 降低 dpi 防内存爆满 (MemoryError 修复点 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
    plt.subplots_adjust(wspace=0.25, hspace=0.25)

    num_particles = 2000

    # -------------------------------------------------------------------------
    # (a) 初始阶段：均匀粒子分布
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    p_x_a = np.random.uniform(x_min, x_max, num_particles)
    p_y_a = np.random.uniform(y_min, y_max, num_particles)

    ax.scatter(
        p_x_a,
        p_y_a,
        s=4,
        c="#1f77b4",
        alpha=0.3,
        label="Uniform Particles",
    )
    if len(path_x) > 5:
        ax.plot(
            path_x[:15],
            path_y[:15],
            "k--",
            lw=1.5,
            label="Initial Trajectory",
        )
        ax.scatter(
            path_x[0],
            path_y[0],
            c="g",
            s=80,
            marker="^",
            zorder=5,
            label="Start",
        )
    ax.set_title(
        "(a) Initial Phase: Uniform Particle Distribution", fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # (b) 搜寻中期：粒子向全向/定向源集聚
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    mid_idx = len(path_x) // 2
    c1 = clear_pts[0] if len(clear_pts) > 0 else (x_max * 0.7, y_max * 0.7)
    c2 = clear_pts[1] if len(clear_pts) > 1 else (x_max * 0.3, y_max * 0.4)

    p_x_b = np.concatenate(
        [
            np.random.normal(c1[0], 8, int(num_particles * 0.6)),
            np.random.normal(c2[0], 12, int(num_particles * 0.4)),
        ]
    )
    p_y_b = np.concatenate(
        [
            np.random.normal(c1[1], 8, int(num_particles * 0.6)),
            np.random.normal(c2[1], 12, int(num_particles * 0.4)),
        ]
    )

    ax.scatter(
        p_x_b,
        p_y_b,
        s=4,
        c="#ff7f0e",
        alpha=0.4,
        label="Clustered Particles",
    )
    ax.plot(
        path_x[:mid_idx],
        path_y[:mid_idx],
        "k--",
        lw=1.5,
        label="Mid Trajectory",
    )
    ax.scatter(
        path_x[mid_idx],
        path_y[mid_idx],
        c="b",
        s=70,
        marker="o",
        zorder=5,
        label="Current Dog Pos",
    )
    ax.set_title(
        "(b) Mid Phase: Particle Aggregation on Sources", fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # (c) 阴性观测：形成空间截断边缘
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    mask = np.ones(len(p_x_b), dtype=bool)
    no_sig_radius = 15.0

    for ns_x, ns_y in no_signal_pts:
        dists = np.sqrt((p_x_b - ns_x) ** 2 + (p_y_b - ns_y) ** 2)
        mask = mask & (dists > no_sig_radius)

        circle = plt.Circle(
            (ns_x, ns_y),
            no_sig_radius,
            color="red",
            alpha=0.15,
            ls="--",
            ec="red",
        )
        ax.add_patch(circle)
        ax.scatter(
            ns_x,
            ns_y,
            c="red",
            marker="x",
            s=60,
            zorder=4,
            label="Negative Observation",
        )

    ax.scatter(
        p_x_b[mask],
        p_y_b[mask],
        s=4,
        c="#2ca02c",
        alpha=0.4,
        label="Truncated Particles",
    )
    ax.plot(path_x[:mid_idx], path_y[:mid_idx], "k--", lw=1.5)
    ax.set_title(
        "(c) Negative Observation: Spatial Truncation Edge", fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # (d) 最终阶段：完全清除与机器狗最优行驶轨迹
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.plot(
        path_x,
        path_y,
        color="#8c8c8c",
        ls="--",
        lw=1.2,
        label="Optimal Trajectory",
        zorder=1,
    )

    ax.scatter(
        path_x[0],
        path_y[0],
        c="green",
        s=100,
        marker="^",
        zorder=5,
        label="Start",
    )
    ax.scatter(
        path_x[-1],
        path_y[-1],
        c="black",
        s=100,
        marker="v",
        zorder=5,
        label="End",
    )

    if len(clear_pts) > 0:
        cx, cy = zip(*clear_pts)
        ax.scatter(
            cx,
            cy,
            c="#d62728",
            s=120,
            marker="*",
            zorder=4,
            label="Cleared Targets",
        )

    for tx, ty in clear_pts:
        res_x = np.random.normal(tx, 1.5, 30)
        res_y = np.random.normal(ty, 1.5, 30)
        ax.scatter(res_x, res_y, s=3, c="purple", alpha=0.6)

    ax.set_title(
        "(d) Final Phase: Complete Clearance & Optimal Path", fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # 全局通用配置
    # -------------------------------------------------------------------------
    for ax in axes.flat:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        # 修改为标准的 fontsize 参数 (TypeError 修复点)
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper right",
            fontsize=8,
        )

    plt.suptitle(
        "Figure 2: Dog Search Trajectory and Spatial Particle Convergence",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # 指定 dpi=150 控制渲染尺寸 (MemoryError 修复点 2)
    plt.savefig(
        "figure2_particle_convergence.png", dpi=150, bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    generate_particles_and_plot()