"""Task 7B-1: posterior-sample convergence and strategy stability.

This is a small experiment runner around the frozen Q2 ``run_experiment``
pipeline.  It changes only ``prior_draws`` and writes machine-readable and
human-readable summaries; it does not alter any model or optimizer semantics.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.q2.main import ExperimentSummary, run_experiment


DEFAULT_SAMPLE_SIZES = (10_000, 30_000, 60_000)
STRATEGIES = ("pure_bayesian", "pure_minimax", "hybrid")
CSV_FIELDS = (
    "prior_draws", "runtime_s", "posterior_retained_draws", "posterior_ess", "acceptance_rate", "strategy", "q_x", "q_y", "psi_d_m",
    "u_proxy_m", "u_bar_m", "movement_m", "in_c_rec_certified",
    "q_displacement_m", "psi_abs_change_m", "psi_relative_change",
    "u_proxy_abs_change_m", "u_proxy_relative_change",
    "u_bar_abs_change_m", "u_bar_relative_change",
)


@dataclass(frozen=True)
class ConvergenceRow:
    prior_draws: int
    runtime_s: float
    posterior_retained_draws: int
    posterior_ess: float
    acceptance_rate: float
    strategy: str
    q_x: float
    q_y: float
    psi_d_m: float
    u_proxy_m: float
    u_bar_m: float
    movement_m: float
    in_c_rec_certified: bool
    q_displacement_m: float | None = None
    psi_abs_change_m: float | None = None
    psi_relative_change: float | None = None
    u_proxy_abs_change_m: float | None = None
    u_proxy_relative_change: float | None = None
    u_bar_abs_change_m: float | None = None
    u_bar_relative_change: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in CSV_FIELDS}


def _relative_change(current: float, previous: float) -> float | None:
    scale = max(abs(previous), 1e-12)
    return (current - previous) / scale


def symmetry_aware_displacement(q_old: tuple[float, float], q_new: tuple[float, float], *,
                                mode: str | None = None) -> float:
    """Return ordinary displacement, or explicit x-reflection-aware distance.

    ``mode='x_reflection'`` is valid only for the symmetric benchmark used by
    this task.  It is never inferred automatically for arbitrary Q2 scenarios.
    """
    ordinary = math.hypot(q_old[0] - q_new[0], q_old[1] - q_new[1])
    if mode is None:
        return ordinary
    if mode != "x_reflection":
        raise ValueError("unknown symmetry mode")
    reflected = math.hypot(q_old[0] - q_new[0], q_old[1] + q_new[1])
    return min(ordinary, reflected)


def _validate_sizes(values: Sequence[int]) -> tuple[int, ...]:
    if not values:
        raise ValueError("prior_draws_values must be non-empty")
    prepared = tuple(values)
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in prepared):
        raise ValueError("prior draw sizes must be positive integers")
    if len(set(prepared)) != len(prepared):
        raise ValueError("prior draw sizes must be unique")
    return prepared


def _scores_from_summary(summary: ExperimentSummary):
    return {
        "pure_bayesian": summary.pure_bayesian,
        "pure_minimax": summary.pure_minimax,
        "hybrid": summary.hybrid,
    }


def run_convergence(
    prior_draws_values: Sequence[int] = DEFAULT_SAMPLE_SIZES,
    *,
    output_dir: Path | str = Path("results/q2"),
    seed: int = 7,
    append_existing: bool = False,
) -> tuple[ConvergenceRow, ...]:
    """Run identical Q2 pipelines for each requested posterior sample size."""
    sizes = _validate_sizes(prior_draws_values)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    directory = Path(output_dir)
    existing: list[ConvergenceRow] = _read_existing_rows(directory / "posterior_convergence.csv") if append_existing else []
    if existing and any(size <= max(row.prior_draws for row in existing) for size in sizes):
        raise ValueError("requested prior draw size already exists in output")
    summaries: list[tuple[int, float, ExperimentSummary]] = []
    for draws in sizes:
        started = time.perf_counter()
        summary = run_experiment(
            seed=seed,
            prior_draws=draws,
            station=(-900.0, 0.0),
            first_bearing_deg=0.0,
            coarse_spacing_m=300.0,
            refinement_steps_m=(120.0, 40.0),
            direction_bins=180,
            circle_vertices=72,
            rho=0.10,
            tau_m=0.5,
        )
        summaries.append((draws, time.perf_counter() - started, summary))

    previous: dict[str, tuple[float, float, float, float, float]] = {}
    rows: list[ConvergenceRow] = list(existing)
    for old in existing:
        previous[old.strategy] = (old.q_x, old.q_y, old.psi_d_m, old.u_proxy_m, old.u_bar_m)
    for draws, runtime, summary in summaries:
        for strategy, score in _scores_from_summary(summary).items():
            current = (score.q[0], score.q[1], score.bayes.psi_d_m,
                       score.robust.u_proxy_m, score.robust.u_bar_m)
            old = previous.get(strategy)
            rows.append(ConvergenceRow(
                prior_draws=draws, runtime_s=runtime,
                posterior_retained_draws=summary.posterior.retained_draws,
                posterior_ess=summary.posterior.effective_sample_size,
                acceptance_rate=summary.posterior.acceptance_rate,
                strategy=strategy,
                q_x=current[0], q_y=current[1], psi_d_m=current[2],
                u_proxy_m=current[3], u_bar_m=current[4],
                movement_m=score.robust.movement_m,
                in_c_rec_certified=score.robust.in_c_rec_certified,
                q_displacement_m=(math.hypot(current[0] - old[0], current[1] - old[1]) if old else None),
                psi_abs_change_m=(abs(current[2] - old[2]) if old else None),
                psi_relative_change=(_relative_change(current[2], old[2]) if old else None),
                u_proxy_abs_change_m=(abs(current[3] - old[3]) if old else None),
                u_proxy_relative_change=(_relative_change(current[3], old[3]) if old else None),
                u_bar_abs_change_m=(abs(current[4] - old[4]) if old else None),
                u_bar_relative_change=(_relative_change(current[4], old[4]) if old else None),
            ))
            previous[strategy] = current

    rows.sort(key=lambda row: (row.prior_draws, STRATEGIES.index(row.strategy)))
    write_outputs(rows, output_dir)
    return tuple(rows)


def _read_existing_rows(path: Path) -> list[ConvergenceRow]:
    if not path.exists():
        return []
    rows: list[ConvergenceRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            def optional(name: str) -> float | None:
                value = raw.get(name, "")
                return None if value in (None, "") else float(value)
            rows.append(ConvergenceRow(
                prior_draws=int(raw["prior_draws"]), runtime_s=float(raw["runtime_s"]),
                posterior_retained_draws=int(raw["posterior_retained_draws"]),
                posterior_ess=float(raw["posterior_ess"]), acceptance_rate=float(raw["acceptance_rate"]),
                strategy=raw["strategy"], q_x=float(raw["q_x"]), q_y=float(raw["q_y"]),
                psi_d_m=float(raw["psi_d_m"]), u_proxy_m=float(raw["u_proxy_m"]),
                u_bar_m=float(raw["u_bar_m"]), movement_m=float(raw["movement_m"]),
                in_c_rec_certified=raw["in_c_rec_certified"].lower() == "true",
                q_displacement_m=optional("q_displacement_m"), psi_abs_change_m=optional("psi_abs_change_m"),
                psi_relative_change=optional("psi_relative_change"), u_proxy_abs_change_m=optional("u_proxy_abs_change_m"),
                u_proxy_relative_change=optional("u_proxy_relative_change"), u_bar_abs_change_m=optional("u_bar_abs_change_m"),
                u_bar_relative_change=optional("u_bar_relative_change"),
            ))
    return rows


def stability_gate(rows: Sequence[ConvergenceRow], *, previous_size: int, current_size: int,
                   q_limit_m: float = 80.0, relative_limit: float = 0.10,
                   symmetry_mode: str | None = None) -> dict[str, object]:
    """Engineering gate for adjacent posterior sizes; not a mathematical theorem."""
    selected = [row for row in rows if row.prior_draws == current_size]
    previous = {row.strategy: row for row in rows if row.prior_draws == previous_size}
    if not selected or not any(row.prior_draws == previous_size for row in rows):
        return {"passed": False, "reason": "missing adjacent-size comparisons", "checks": {}}
    checks = {
        row.strategy: {
            "q_displacement_m": symmetry_aware_displacement(
                (previous[row.strategy].q_x, previous[row.strategy].q_y),
                (row.q_x, row.q_y), mode=symmetry_mode,
            ) if row.strategy in previous else None,
            "psi_relative_change": row.psi_relative_change,
            "u_proxy_relative_change": row.u_proxy_relative_change,
            "pass": (row.strategy in previous
                      and symmetry_aware_displacement(
                          (previous[row.strategy].q_x, previous[row.strategy].q_y),
                          (row.q_x, row.q_y), mode=symmetry_mode,
                      ) <= q_limit_m
                      and abs(row.psi_relative_change or 0.0) <= relative_limit
                      and abs(row.u_proxy_relative_change or 0.0) <= relative_limit),
        }
        for row in selected
    }
    return {"passed": len(checks) == len(STRATEGIES) and all(item["pass"] for item in checks.values()),
            "reason": f"{previous_size}->{current_size} engineering thresholds", "checks": checks}


def run_gated(*, existing_output_dir: Path | str = Path("results/q2"),
              added_sizes: Sequence[int] = (300_000, 600_000, 1_000_000),
              seed: int = 7) -> dict[str, object]:
    """Run Part A and enter Part B/C only when the explicit gate passes."""
    rows = run_convergence(added_sizes, output_dir=existing_output_dir, seed=seed, append_existing=True)
    sizes = sorted({row.prior_draws for row in rows})
    gate = stability_gate(rows, previous_size=sizes[-2], current_size=sizes[-1], symmetry_mode="x_reflection")
    result: dict[str, object] = {"posterior_rows": rows, "gate": gate, "numerical_rows": None}
    if gate["passed"]:
        # Use the smaller member of the passing adjacent pair as the lower-cost stable configuration.
        result["numerical_rows"] = run_numerical_convergence(
            prior_draws=sizes[-2], output_dir=existing_output_dir, seed=seed,
        )
    return result


def run_numerical_convergence(*, prior_draws: int, output_dir: Path | str = Path("results/q2"), seed: int = 7,
                              direction_bins: Sequence[int] = (180, 360),
                              coarse_spacings: Sequence[float] = (300.0, 200.0)) -> list[dict[str, object]]:
    """Run Part B/C one-factor checks at a fixed posterior sample size."""
    records: list[dict[str, object]] = []
    cached_posterior = None
    for bins in direction_bins:
        summary = run_experiment(seed=seed, prior_draws=prior_draws, station=(-900.0, 0.0), first_bearing_deg=0.0,
                                 coarse_spacing_m=300.0, refinement_steps_m=(120.0, 40.0), direction_bins=bins,
                                 circle_vertices=72, rho=0.10, tau_m=0.5,
                                 posterior_override=cached_posterior)
        cached_posterior = summary.posterior
        for strategy, score in _scores_from_summary(summary).items():
            records.append({"part": "B", "parameter": "direction_bins", "value": bins, "runtime_s": None,
                            "prior_draws": prior_draws, "strategy": strategy, "q_x": score.q[0], "q_y": score.q[1],
                            "psi_d_m": score.bayes.psi_d_m, "u_proxy_m": score.robust.u_proxy_m,
                            "u_bar_m": score.robust.u_bar_m, "movement_m": score.robust.movement_m,
                            "in_c_rec_certified": score.robust.in_c_rec_certified})
    for spacing in coarse_spacings:
        summary = run_experiment(seed=seed, prior_draws=prior_draws, station=(-900.0, 0.0), first_bearing_deg=0.0,
                                 coarse_spacing_m=spacing, refinement_steps_m=(120.0, 40.0), direction_bins=360,
                                 circle_vertices=72, rho=0.10, tau_m=0.5,
                                 posterior_override=cached_posterior)
        cached_posterior = summary.posterior
        for strategy, score in _scores_from_summary(summary).items():
            records.append({"part": "C", "parameter": "coarse_spacing_m", "value": spacing, "runtime_s": None,
                            "prior_draws": prior_draws, "strategy": strategy, "q_x": score.q[0], "q_y": score.q[1],
                            "psi_d_m": score.bayes.psi_d_m, "u_proxy_m": score.robust.u_proxy_m,
                            "u_bar_m": score.robust.u_bar_m, "movement_m": score.robust.movement_m,
                            "in_c_rec_certified": score.robust.in_c_rec_certified})
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    fields = tuple(records[0]) if records else ()
    with (directory / "numerical_convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    lines = ["# Q2 Task 7B-2 numerical convergence", "", f"Fixed prior_draws={prior_draws}; refinement=(120,40), circle_vertices=72, rho=0.10, tau_m=0.5.", "", "| part | parameter | value | strategy | q | psi_d_m | u_proxy_m | u_bar_m | movement_m | C_rec |", "|---|---|---:|---|---|---:|---:|---:|---:|---|"]
    for row in records:
        lines.append(f"| {row['part']} | {row['parameter']} | {row['value']} | {row['strategy']} | ({row['q_x']:.2f},{row['q_y']:.2f}) | {row['psi_d_m']:.3f} | {row['u_proxy_m']:.3f} | {row['u_bar_m']:.3f} | {row['movement_m']:.3f} | {row['in_c_rec_certified']} |")
    (directory / "numerical_convergence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


def write_outputs(rows: Sequence[ConvergenceRow], output_dir: Path | str) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "posterior_convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)

    md_path = directory / "posterior_convergence.md"
    sizes = sorted({row.prior_draws for row in rows})
    lines = [
        "# Q2 Task 7B-1: posterior convergence and strategy stability", "",
        "Fixed configuration: station `(-900,0)`, first bearing `0°`, seed `7`, "
        "coarse spacing `300 m`, refinement `(120,40) m`, direction grid `180`, "
        "circle vertices `72`, `rho=0.10`, `tau_m=0.5`. The nominal error model "
        "is the explicit experiment model reported by `src/q2/main.py`.", "",
        "The main robust-envelope hybrid uses `u_proxy_m`; `u_bar_m` is reported "
        "only as a conservative outer diagnostic.", "",
        "## Posterior diagnostics", "",
        "| prior_draws | runtime (s) | retained | ESS | acceptance |",
        "|---:|---:|---:|---:|---:|",
    ]
    # Diagnostics are repeated per strategy row; select the first row per size.
    for size in sizes:
        row = next(row for row in rows if row.prior_draws == size)
        lines.append(f"| {size} | {row.runtime_s:.3f} | {row.posterior_retained_draws} | {row.posterior_ess:.3f} | {row.acceptance_rate:.6f} |")
    lines += ["", "## Strategy stability", "",
              "| prior_draws | strategy | q | psi_d_m | u_proxy_m | u_bar_m | movement_m | C_rec | Δq from previous |",
              "|---:|---|---|---:|---:|---:|---:|---|---:|"]
    for row in rows:
        delta = "—" if row.q_displacement_m is None else f"{row.q_displacement_m:.3f}"
        lines.append(f"| {row.prior_draws} | {row.strategy} | ({row.q_x:.2f},{row.q_y:.2f}) | {row.psi_d_m:.3f} | {row.u_proxy_m:.3f} | {row.u_bar_m:.3f} | {row.movement_m:.3f} | {row.in_c_rec_certified} | {delta} |")
    lines += ["", "## Adjacent-size changes", "",
              "Relative changes use `max(|previous|, 1e-12)` as denominator; absolute changes are retained in CSV.", ""]
    for row in rows:
        if row.q_displacement_m is None:
            continue
        lines.append(f"- `{row.strategy}` at {row.prior_draws}: Δq={row.q_displacement_m:.3f} m; "
                     f"Δpsi={row.psi_abs_change_m:.3f} m ({row.psi_relative_change:.6g}); "
                     f"Δu_proxy={row.u_proxy_abs_change_m:.3f} m ({row.u_proxy_relative_change:.6g}); "
                     f"Δu_bar={row.u_bar_abs_change_m:.3f} m ({row.u_bar_relative_change:.6g})")
    lines += ["", "## Interpretation and risks", ""]
    if len(sizes) >= 2:
        previous_size, current_size = sizes[-2], sizes[-1]
        latest = {row.strategy: row for row in rows if row.prior_draws == current_size}
        lines.append(
            f"Observed {previous_size:,}→{current_size:,}: "
            + "; ".join(
                f"{strategy} Δq={latest[strategy].q_displacement_m:.3f} m, "
                f"Δpsi={latest[strategy].psi_relative_change:.3g}, "
                f"Δu_proxy={latest[strategy].u_proxy_relative_change:.3g}"
                for strategy in STRATEGIES
            )
            + "."
        )
        lines.append(
            "The selections therefore show partial stability but not blanket convergence; "
            "judge each strategy using ESS and the reported adjacent changes."
        )
    lines += [
              "",
              "These finite-sample results are a stability diagnostic, not a proof of posterior convergence. "
              "ESS, retained draws, direction-grid approximation, and the fixed nominal error model remain limitations. "
              "The latest adjacent-size comparison above should be used to decide whether more samples are needed.", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-draws", nargs="+", type=int, default=list(DEFAULT_SAMPLE_SIZES))
    parser.add_argument("--output-dir", type=Path, default=Path("results/q2"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--append-existing", action="store_true",
                        help="append new sample sizes to an existing posterior CSV")
    args = parser.parse_args(argv)
    run_convergence(args.prior_draws, output_dir=args.output_dir, seed=args.seed,
                    append_existing=args.append_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
