"""Plot the validated Q1 intersection-angle sensitivity experiment.

Run from the repository root:
    python src/q1/plot_intersection_angle_sensitivity.py

Reads results/tables/q1_intersection_angle_sensitivity.json and writes the
paper-ready PNG and PDF figures under results/figures/.  This script does not
rerun or alter the experiment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "results" / "tables" / "q1_intersection_angle_sensitivity.json"
PNG_PATH = ROOT / "results" / "figures" / "q1_intersection_angle_sensitivity.png"
PDF_PATH = ROOT / "results" / "figures" / "q1_intersection_angle_sensitivity.pdf"
EXPECTED_ALPHA_DEG = (5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0)
RATIO_ABS_TOL = 1e-12


def is_nonincreasing(values: list[float]) -> bool:
    """Return whether values never increase, matching the experiment report."""

    return all(current <= previous for previous, current in zip(values, values[1:]))


def load_and_validate() -> dict[str, list[float]]:
    """Read the experiment JSON and reject any inconsistency before plotting."""

    report: dict[str, Any] = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError("experiment JSON must contain exactly nine rows")

    keys = (
        "alpha_deg",
        "diameter_m",
        "area_m2",
        "mec_radius_m",
        "mec_over_half_d",
    )
    data = {key: [float(row[key]) for row in rows] for key in keys}
    alpha = data["alpha_deg"]
    if tuple(alpha) != EXPECTED_ALPHA_DEG:
        raise ValueError(f"unexpected alpha grid: {alpha}")
    if not all(right > left for left, right in zip(alpha, alpha[1:])):
        raise ValueError("alpha values must be strictly increasing")

    for key in ("diameter_m", "area_m2", "mec_radius_m"):
        if not all(math.isfinite(value) and value > 0.0 for value in data[key]):
            raise ValueError(f"{key} must contain only positive finite values")

    ratios = data["mec_over_half_d"]
    if not all(
        math.isfinite(value)
        and math.isclose(value, 1.0, rel_tol=0.0, abs_tol=RATIO_ABS_TOL)
        for value in ratios
    ):
        raise ValueError("mec_over_half_d is not uniformly one within tolerance")
    if not all(
        math.isclose(radius, diameter / 2.0, rel_tol=1e-12, abs_tol=1e-12)
        for radius, diameter in zip(data["mec_radius_m"], data["diameter_m"])
    ):
        raise ValueError("mec_radius_m is inconsistent with D/2")

    stored_monotonicity = report["checks"]["monotonicity_as_alpha_increases"]
    for key in ("diameter_m", "area_m2", "mec_radius_m", "mec_over_half_d"):
        computed = is_nonincreasing(data[key])
        stored = bool(stored_monotonicity[key]["nonincreasing"])
        if computed != stored:
            raise ValueError(
                f"stored monotonicity for {key} does not match the plotted data"
            )

    return data


def configure_style() -> None:
    """Set a restrained print-friendly style using bundled matplotlib fonts."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#30343B",
            "axes.labelcolor": "#20242A",
            "axes.titlecolor": "#20242A",
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


def make_figure(data: dict[str, list[float]]) -> plt.Figure:
    """Create the horizontal two-panel sensitivity figure."""

    configure_style()
    alpha = data["alpha_deg"]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.7),
        constrained_layout=True,
        sharex=True,
    )

    panels = (
        (axes[0], data["diameter_m"], "#176B87", r"Diameter $D$ (m)"),
        (axes[1], data["area_m2"], "#B55233", r"Area $A$ (m$^2$)"),
    )
    for panel_index, (axis, values, color, ylabel) in enumerate(panels):
        axis.plot(
            alpha,
            values,
            color=color,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.15,
            zorder=3,
        )
        axis.set_xticks(alpha)
        axis.set_xlim(2.0, 93.0)
        axis.set_ylim(bottom=0.0)
        axis.set_xlabel(r"Intersection angle $\alpha$ ($^\circ$)")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D8DCE1", linewidth=0.65, alpha=0.8)
        axis.tick_params(direction="out", length=3.5, width=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
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
    data = load_and_validate()
    figure = make_figure(data)
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PNG_PATH, dpi=350)
    figure.savefig(PDF_PATH)
    plt.close(figure)
    verify_outputs()
    print(f"Read {len(data['alpha_deg'])} validated rows from {DATA_PATH}")
    print("D and A trends match the JSON monotonicity checks")
    print("All rows satisfy r*=D/2 for this symmetric configuration")
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
