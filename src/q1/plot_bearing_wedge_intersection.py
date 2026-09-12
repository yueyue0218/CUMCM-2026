"""Plot the Q1 bearing-wedge constraints and their convex intersection.

Run from the repository root:
    python src/q1/plot_bearing_wedge_intersection.py

The multi-observation panel uses the production bearing-wedge clipping helper;
the checks in this script reject an empty, inconsistent, or domain-truncated
intersection before either output file is written.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

import matplotlib


matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import Arc, Polygon  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.geometry import (  # noqa: E402
    clip_polygon_to_bearing_wedge,
    cross,
    signed_angle_difference_deg,
    unit_vector,
)


Point = tuple[float, float]

PNG_PATH = ROOT / "results" / "figures" / "q1_bearing_wedge_intersection.png"
PDF_PATH = ROOT / "results" / "figures" / "q1_bearing_wedge_intersection.pdf"

ERROR_DEG = 1.0
TRUE_POINT: Point = (3.0, 4.0)
STATIONS: tuple[Point, ...] = ((-55.0, -35.0), (55.0, -35.0), (0.0, 65.0))
INITIAL_LIMIT = 500.0
INITIAL_DOMAIN: tuple[Point, ...] = (
    (-INITIAL_LIMIT, -INITIAL_LIMIT),
    (INITIAL_LIMIT, -INITIAL_LIMIT),
    (INITIAL_LIMIT, INITIAL_LIMIT),
    (-INITIAL_LIMIT, INITIAL_LIMIT),
)

WEDGE_COLORS = ("#4C78A8", "#E08B3E", "#59A14F")
BOUNDARY_COLOR = "#4B5563"
OMEGA_COLOR = "#8B5E9F"


def bearing_to(source: Point, target: Point) -> float:
    """Return the counter-clockwise bearing from source to target in degrees."""

    return math.degrees(math.atan2(target[1] - source[1], target[0] - source[0])) % 360.0


BEARINGS: tuple[float, ...] = tuple(
    bearing_to(station, TRUE_POINT) for station in STATIONS
)


def polygon_area(polygon: Sequence[Point]) -> float:
    """Return the unsigned shoelace area of a polygon."""

    return abs(
        sum(cross(polygon[index], polygon[(index + 1) % len(polygon)]) for index in range(len(polygon)))
    ) / 2.0


def point_in_convex_polygon(point: Point, polygon: Sequence[Point], tolerance: float = 1e-9) -> bool:
    """Return whether a point lies in or on a counter-clockwise convex polygon."""

    return all(
        cross(
            (polygon[(index + 1) % len(polygon)][0] - polygon[index][0],
             polygon[(index + 1) % len(polygon)][1] - polygon[index][1]),
            (point[0] - polygon[index][0], point[1] - polygon[index][1]),
        )
        >= -tolerance
        for index in range(len(polygon))
    )


def compute_intersection() -> list[Point]:
    """Apply all three production wedge clips to the large initial domain."""

    region = list(INITIAL_DOMAIN)
    for station, bearing in zip(STATIONS, BEARINGS):
        region = clip_polygon_to_bearing_wedge(region, station, bearing, ERROR_DEG)
    return region


def validate_geometry(region: Sequence[Point]) -> None:
    """Reject empty, infeasible, initial-domain-bound, or truth-excluding geometry."""

    if len(region) < 3 or polygon_area(region) <= 1e-8:
        raise ValueError("the three bearing wedges do not form a non-empty area")

    tolerance_deg = 1e-8
    for vertex in region:
        for station, bearing in zip(STATIONS, BEARINGS):
            vertex_bearing = bearing_to(station, vertex)
            error = abs(signed_angle_difference_deg(vertex_bearing, bearing))
            if error > ERROR_DEG + tolerance_deg:
                raise ValueError(
                    f"intersection vertex {vertex} violates a bearing wedge: {error} deg"
                )

    domain_margin = 1e-7
    if any(
        abs(abs(coordinate) - INITIAL_LIMIT) <= domain_margin
        for vertex in region
        for coordinate in vertex
    ):
        raise ValueError("the artificial initial domain is an active intersection boundary")

    for station, bearing in zip(STATIONS, BEARINGS):
        truth_error = abs(
            signed_angle_difference_deg(bearing_to(station, TRUE_POINT), bearing)
        )
        if truth_error > ERROR_DEG + tolerance_deg:
            raise ValueError("the displayed reference truth violates a bearing wedge")
    if not point_in_convex_polygon(TRUE_POINT, region):
        raise ValueError("the displayed reference truth is outside the final intersection")


def configure_style() -> None:
    """Use a restrained, white, print-friendly figure style."""

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
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def add_ray(axis: Axes, origin: Point, angle_deg: float, length: float, **kwargs: object) -> None:
    """Draw a finite visible segment of a mathematical ray."""

    direction = unit_vector(angle_deg)
    axis.plot(
        [origin[0], origin[0] + length * direction[0]],
        [origin[1], origin[1] + length * direction[1]],
        **kwargs,
    )


def draw_single_observation(axis: Axes) -> None:
    """Draw a schematic single bearing and its two half-plane boundaries."""

    station = (-4.2, -1.2)
    bearing = 27.0
    schematic_error = 10.0
    view_domain = [(-6.0, -3.6), (6.0, -3.6), (6.0, 5.3), (-6.0, 5.3)]
    wedge = clip_polygon_to_bearing_wedge(
        view_domain, station, bearing, schematic_error
    )
    axis.add_patch(
        Polygon(wedge, closed=True, facecolor="#5B8DB8", edgecolor="none", alpha=0.16)
    )

    ray_length = 12.0
    for angle in (bearing - schematic_error, bearing + schematic_error):
        add_ray(axis, station, angle, ray_length, color=BOUNDARY_COLOR, linewidth=1.25)
    add_ray(
        axis,
        station,
        bearing,
        ray_length,
        color="#235D86",
        linewidth=1.3,
        linestyle=(0, (4, 3)),
    )

    arc_radius = 2.25
    axis.add_patch(
        Arc(
            station,
            2 * arc_radius,
            2 * arc_radius,
            theta1=bearing,
            theta2=bearing + schematic_error,
            color="#235D86",
            linewidth=1.0,
        )
    )
    mid_angle = math.radians(bearing + schematic_error / 2.0)
    axis.text(
        station[0] + 2.65 * math.cos(mid_angle),
        station[1] + 2.65 * math.sin(mid_angle),
        r"$\delta$",
        color="#235D86",
        ha="center",
        va="center",
    )

    axis.scatter(*station, s=35, color="#20242A", zorder=5)
    axis.text(station[0] - 0.22, station[1] - 0.42, r"$S_i$", ha="right", va="top")

    label_specs = (
        (bearing - schematic_error, r"$\theta_i-\delta$", -0.12),
        (bearing, r"$\theta_i$", 0.00),
        (bearing + schematic_error, r"$\theta_i+\delta$", 0.13),
    )
    for angle, label, vertical_shift in label_specs:
        direction = unit_vector(angle)
        label_radius = 9.7
        axis.text(
            station[0] + label_radius * direction[0],
            station[1] + label_radius * direction[1] + vertical_shift,
            label,
            ha="left",
            va="center",
            fontsize=9.5,
        )

    axis.set_xlim(-5.8, 6.0)
    axis.set_ylim(-3.3, 5.1)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def clipped_display_wedge(station: Point, bearing: float) -> list[Point]:
    """Clip one wedge to the panel's rectangular display domain."""

    display_domain = [(-70.0, -48.0), (70.0, -48.0), (70.0, 78.0), (-70.0, 78.0)]
    return clip_polygon_to_bearing_wedge(display_domain, station, bearing, ERROR_DEG)


def draw_multi_observation(axis: Axes, region: Sequence[Point]) -> None:
    """Draw three real wedges, their common intersection, and a local inset."""

    for index, (station, bearing, color) in enumerate(
        zip(STATIONS, BEARINGS, WEDGE_COLORS), start=1
    ):
        display_wedge = clipped_display_wedge(station, bearing)
        axis.add_patch(
            Polygon(display_wedge, closed=True, facecolor=color, edgecolor="none", alpha=0.105)
        )
        for angle in (bearing - ERROR_DEG, bearing + ERROR_DEG):
            add_ray(axis, station, angle, 155.0, color=color, linewidth=0.9, alpha=0.9)
        add_ray(
            axis,
            station,
            bearing,
            155.0,
            color=color,
            linewidth=1.05,
            linestyle=(0, (4, 3)),
        )
        axis.scatter(*station, s=31, color=color, edgecolor="white", linewidth=0.6, zorder=5)
        offsets = ((-3.0, -4.2), (3.0, -4.2), (0.0, 4.0))
        dx, dy = offsets[index - 1]
        axis.text(station[0] + dx, station[1] + dy, rf"$S_{index}$", ha="center", va="center")

    axis.add_patch(
        Polygon(region, closed=True, facecolor=OMEGA_COLOR, edgecolor="#5F3C70", linewidth=1.1, alpha=0.62, zorder=6)
    )
    axis.scatter(*TRUE_POINT, marker="*", s=76, color="#20242A", edgecolor="white", linewidth=0.5, zorder=8)
    axis.text(TRUE_POINT[0] + 2.3, TRUE_POINT[1] + 1.8, r"$G$", ha="left", va="bottom", zorder=9)

    x_values = [point[0] for point in region]
    y_values = [point[1] for point in region]
    omega_center = (sum(x_values) / len(x_values), sum(y_values) / len(y_values))
    axis.text(omega_center[0] - 3.0, omega_center[1] - 4.2, r"$\Omega$", color="#5F3C70", fontweight="bold", zorder=9)

    inset = axis.inset_axes([0.62, 0.61, 0.35, 0.35])
    for station, bearing, color in zip(STATIONS, BEARINGS, WEDGE_COLORS):
        inset_wedge = clip_polygon_to_bearing_wedge(
            [(-20.0, -20.0), (25.0, -20.0), (25.0, 25.0), (-20.0, 25.0)],
            station,
            bearing,
            ERROR_DEG,
        )
        inset.add_patch(
            Polygon(inset_wedge, closed=True, facecolor=color, edgecolor=color, linewidth=0.65, alpha=0.13)
        )
    inset.add_patch(
        Polygon(region, closed=True, facecolor=OMEGA_COLOR, edgecolor="#5F3C70", linewidth=1.25, alpha=0.67, zorder=5)
    )
    inset.scatter(*TRUE_POINT, marker="*", s=53, color="#20242A", edgecolor="white", linewidth=0.4, zorder=7)
    inset.text(TRUE_POINT[0] + 0.35, TRUE_POINT[1] + 0.25, r"$G$", fontsize=8.5, zorder=8)
    inset.text(0.08, 0.10, r"$\Omega$", transform=inset.transAxes, color="#5F3C70", fontsize=9, fontweight="bold")

    width = max(x_values) - min(x_values)
    height = max(y_values) - min(y_values)
    padding = max(0.75, 0.35 * max(width, height))
    inset.set_xlim(min(x_values) - padding, max(x_values) + padding)
    inset.set_ylim(min(y_values) - padding, max(y_values) + padding)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("#737A82")
        spine.set_linewidth(0.7)
    axis.indicate_inset_zoom(inset, edgecolor="#737A82", linewidth=0.7, alpha=0.8)

    axis.set_xlim(-67.0, 68.0)
    axis.set_ylim(-46.0, 73.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$x$ (m)")
    axis.set_ylabel(r"$y$ (m)")
    axis.tick_params(direction="out", length=3.0, width=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def make_figure(region: Sequence[Point]) -> plt.Figure:
    """Build the horizontal two-panel paper figure."""

    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), constrained_layout=True)
    draw_single_observation(axes[0])
    draw_multi_observation(axes[1], region)
    for index, axis in enumerate(axes):
        axis.text(
            0.0,
            1.025,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    return figure


def verify_outputs() -> None:
    """Confirm that the PNG decodes and the PDF has a valid header."""

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
    region = compute_intersection()
    validate_geometry(region)
    figure = make_figure(region)
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PNG_PATH, dpi=350)
    figure.savefig(PDF_PATH)
    plt.close(figure)
    verify_outputs()
    print(f"stations={STATIONS}")
    print(f"G={TRUE_POINT}, delta={ERROR_DEG} deg")
    print(f"bearings={tuple(round(value, 6) for value in BEARINGS)}")
    print(f"Omega: vertices={len(region)}, area={polygon_area(region):.6f} m^2")
    print("geometry checks: PASS")
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
