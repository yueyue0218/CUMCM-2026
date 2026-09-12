"""Draw the analytical same-diameter Q1 coverage comparison.

Run from the repository root:
    python src/q1/plot_coverage_comparison.py

The two geometries and their analytical radii are fixed; this script only
renders them in the common Q1 paper-figure style.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PNG_PATH = ROOT / "results" / "figures" / "q1_same_diameter_coverage.png"
PDF_PATH = ROOT / "results" / "figures" / "q1_same_diameter_coverage.pdf"

DIAMETER_M = 36.0
CLEAR_RADIUS_M = 20.0
SQUARE_MEC_RADIUS_M = 18.0
TRIANGLE_MEC_RADIUS_M = 12.0 * math.sqrt(3.0)

REGION_FACE = "#E6F2F7"
REGION_EDGE = "#617990"
MEC_COLOR = "#176B87"
CLEAR_COLOR = "#929AA3"
POINT_COLOR = "#20242A"


def configure_style() -> None:
    """Use the common white, print-friendly Q1 figure style."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#30343B",
            "axes.labelcolor": "#20242A",
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "lines.markersize": 5.0,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def geometry_vertices() -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return the unchanged D=36 square and equilateral-triangle vertices."""

    square = [
        (
            x * SQUARE_MEC_RADIUS_M / math.sqrt(2.0),
            y * SQUARE_MEC_RADIUS_M / math.sqrt(2.0),
        )
        for x, y in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    ]
    triangle = [
        (
            TRIANGLE_MEC_RADIUS_M * math.cos(math.radians(angle)),
            TRIANGLE_MEC_RADIUS_M * math.sin(math.radians(angle)),
        )
        for angle in (90.0, 210.0, 330.0)
    ]
    return square, triangle


def draw_panel(
    axis: Axes,
    vertices: list[tuple[float, float]],
    mec_radius_m: float,
    panel_index: int,
) -> None:
    """Draw one region with its MEC and the 20 m comparison circle."""

    axis.add_patch(
        Polygon(
            vertices,
            closed=True,
            facecolor=REGION_FACE,
            edgecolor=REGION_EDGE,
            linewidth=1.15,
            zorder=1,
        )
    )
    axis.add_patch(
        Circle(
            (0.0, 0.0),
            CLEAR_RADIUS_M,
            fill=False,
            edgecolor=CLEAR_COLOR,
            linewidth=1.25,
            linestyle=(0, (4, 3)),
            zorder=2,
        )
    )
    axis.add_patch(
        Circle(
            (0.0, 0.0),
            mec_radius_m,
            fill=False,
            edgecolor=MEC_COLOR,
            linewidth=1.4,
            zorder=3,
        )
    )
    axis.plot(
        [point[0] for point in vertices],
        [point[1] for point in vertices],
        linestyle="none",
        marker="o",
        markersize=5.0,
        markerfacecolor=POINT_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.5,
        zorder=4,
    )
    axis.plot(0.0, 0.0, marker="o", markersize=4.5, color=MEC_COLOR, zorder=5)

    axis.text(
        0.0,
        -22.0,
        rf"$D={DIAMETER_M:.0f}\,\mathrm{{m}}$",
        ha="center",
        va="center",
    )
    radius_text = (
        rf"$r^*={mec_radius_m:.0f}\,\mathrm{{m}}$"
        if math.isclose(mec_radius_m, round(mec_radius_m))
        else rf"$r^*={mec_radius_m:.4f}\,\mathrm{{m}}$"
    )
    axis.text(0.0, -25.2, radius_text, ha="center", va="center")
    axis.text(
        0.0,
        1.025,
        f"({chr(ord('a') + panel_index)})",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

    axis.set_xlim(-27.0, 27.0)
    axis.set_ylim(-27.0, 24.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def make_figure() -> plt.Figure:
    """Build the two-panel comparison without title or explanatory prose."""

    configure_style()
    square, triangle = geometry_vertices()
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
    figure.subplots_adjust(left=0.035, right=0.985, top=0.93, bottom=0.15, wspace=0.08)
    draw_panel(axes[0], square, SQUARE_MEC_RADIUS_M, 0)
    draw_panel(axes[1], triangle, TRIANGLE_MEC_RADIUS_M, 1)

    legend_handles = (
        Line2D([], [], color=MEC_COLOR, linewidth=1.4, label=r"$r^*$ (MEC)"),
        Line2D(
            [],
            [],
            color=CLEAR_COLOR,
            linewidth=1.25,
            linestyle=(0, (4, 3)),
            label=r"$20\,\mathrm{m}$",
        ),
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
        handlelength=2.6,
        columnspacing=2.0,
        fontsize=9,
    )
    return figure


def verify_outputs() -> None:
    """Confirm that both saved formats exist and can be opened."""

    if not PNG_PATH.is_file() or PNG_PATH.stat().st_size == 0:
        raise OSError(f"PNG output was not created: {PNG_PATH}")
    png = mpimg.imread(PNG_PATH)
    if png.size == 0:
        raise OSError(f"PNG output could not be decoded: {PNG_PATH}")
    if not PDF_PATH.is_file() or PDF_PATH.stat().st_size == 0:
        raise OSError(f"PDF output was not created: {PDF_PATH}")
    if not PDF_PATH.read_bytes().startswith(b"%PDF-"):
        raise OSError(f"PDF output has an invalid header: {PDF_PATH}")


def main() -> None:
    figure = make_figure()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PNG_PATH, dpi=350)
    figure.savefig(PDF_PATH)
    plt.close(figure)
    verify_outputs()
    print(f"D={DIAMETER_M:.0f} m")
    print(f"r_square={SQUARE_MEC_RADIUS_M:.6f} m")
    print(f"r_triangle={TRIANGLE_MEC_RADIUS_M:.6f} m")
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
