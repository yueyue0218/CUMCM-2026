"""Generate the publication-ready Q2 method flowchart."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path("results/q2")


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9.5,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def add_box(ax, x: float, y: float, w: float, h: float, *, number: str,
            title: str, body: str, facecolor: str, edgecolor: str) -> None:
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.016",
        linewidth=1.15, edgecolor=edgecolor, facecolor=facecolor,
        transform=ax.transAxes, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + 0.026, y + h - 0.035, number, transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color="white", weight="bold",
            bbox={"boxstyle": "circle,pad=0.27", "facecolor": edgecolor,
                  "edgecolor": edgecolor, "linewidth": 0.8}, zorder=3)
    ax.text(x + 0.055, y + h - 0.035, title, transform=ax.transAxes,
            ha="left", va="center", fontsize=10.2, weight="bold", color="#1F2937", zorder=3)
    ax.text(x + w / 2, y + h * 0.42, body, transform=ax.transAxes,
            ha="center", va="center", fontsize=8.6, color="#374151",
            linespacing=1.45, zorder=3)


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float],
              *, connectionstyle: str = "arc3") -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, transform=ax.transAxes,
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.15, color="#6B7280",
        connectionstyle=connectionstyle, shrinkA=3, shrinkB=3, zorder=4,
    ))


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.8, 5.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue_fill, blue_edge = "#EAF3F8", "#0072B2"
    green_fill, green_edge = "#ECF7F1", "#27865F"
    orange_fill, orange_edge = "#FDF1E8", "#D55E00"

    add_box(
        ax, 0.035, 0.60, 0.265, 0.27,
        number="1", title="首次观测与联合状态", facecolor=blue_fill, edgecolor=blue_edge,
        body=(r"输入：$H_1=(S_1,\mathrm{direction},\theta_1)$" "\n"
              r"硬支持集 $\mathcal{F}_1$　联合后验 $(G,R,w)$" "\n"
              r"保守外包 $K_{1,\mathrm{out}}$"),
    )
    add_box(
        ax, 0.365, 0.60, 0.265, 0.27,
        number="2", title="候选生成", facecolor=green_fill, edgecolor=green_edge,
        body=(r"构造 $\mathcal{C}_{\mathrm{poss}}$ 与 $\mathcal{C}_{\mathrm{rec}}$" "\n"
              r"粗网格 $h_0=300\,\mathrm{m}$" "\n"
              "剔除无再次接收可能的点"),
    )
    add_box(
        ax, 0.695, 0.60, 0.265, 0.27,
        number="3", title="粗阶段评价与选种", facecolor=orange_fill, edgecolor=orange_edge,
        body=(r"预测 near / direction / no_signal" "\n"
              r"计算 $\Psi_D$ 与 $u_{\mathrm{proxy}}$" "\n"
              "提取 Bayesian、Minimax、Hybrid 种子"),
    )
    add_box(
        ax, 0.695, 0.13, 0.265, 0.27,
        number="4", title="多尺度局部精化", facecolor=blue_fill, edgecolor=blue_edge,
        body=(r"步长 $h\in\{120,40\}\,\mathrm{m}$" "\n"
              r"沿 $0^\circ,45^\circ,\ldots,315^\circ$ 扩展" "\n"
              "合并粗网格与精化点并去重"),
    )
    add_box(
        ax, 0.365, 0.13, 0.265, 0.27,
        number="5", title="最终候选统一评分", facecolor=green_fill, edgecolor=green_edge,
        body=(r"同一后验、同一候选集、同一评价口径" "\n"
              r"$\Psi_D,\ u_{\mathrm{proxy}},\ \bar U,$ movement" "\n"
              "输出可复核的 CandidateScore 集合"),
    )
    add_box(
        ax, 0.035, 0.13, 0.265, 0.27,
        number="6", title="Robust-envelope 决策", facecolor=orange_fill, edgecolor=orange_edge,
        body=(r"Bayesian：名义期望近最优" "\n"
              r"Minimax：最小化 $u_{\mathrm{proxy}}$" "\n"
              r"Hybrid：$\mathcal{A}_\rho$ 内优化 $\Psi_D$ → $q^*$"),
    )

    add_arrow(ax, (0.300, 0.735), (0.365, 0.735))
    add_arrow(ax, (0.630, 0.735), (0.695, 0.735))
    add_arrow(ax, (0.8275, 0.60), (0.8275, 0.40))
    add_arrow(ax, (0.695, 0.265), (0.630, 0.265))
    add_arrow(ax, (0.365, 0.265), (0.300, 0.265))

    ax.text(0.50, 0.94, "硬约束提供安全边界，名义概率模型仅用于平均性能评价",
            transform=ax.transAxes, ha="center", va="center", fontsize=9.2, color="#4B5563")
    ax.text(0.50, 0.055, r"对 $\rho$ 扫描形成名义性能—鲁棒性能 trade-off，并与等移动距离垂直布点比较",
            transform=ax.transAxes, ha="center", va="center", fontsize=9.2, color="#4B5563")

    fig.tight_layout(pad=0.25)
    fig.savefig(OUTPUT_DIR / "q2_method_flow.pdf")
    fig.savefig(OUTPUT_DIR / "q2_method_flow.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
