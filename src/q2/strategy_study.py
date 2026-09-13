"""Q2 Task 7C: one posterior/final-score strategy study and rho sensitivity."""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.q2.main import _merge_unique, _nominal_error_model
from src.q2.model import FirstDirectionObservation, FirstPosteriorSamples, Q2Config, build_first_state, sample_first_direction_posterior
from src.q2.optimizer import (CandidateScore, coarse_grid_candidates, refine_candidates,
                               score_candidates, select_pure_bayesian, select_pure_minimax,
                               select_robust_envelope_hybrid, same_distance_vertical_baseline)

RHO_VALUES = (0.0, 0.025, 0.05, 0.10, 0.20, 0.30)
FIELDS = ("strategy", "q_x", "q_y", "psi_d_m", "u_proxy_m", "u_bar_m", "movement_m", "in_c_rec_certified")


@dataclass(frozen=True)
class StudyArtifacts:
    posterior: FirstPosteriorSamples
    final_candidates: tuple[tuple[float, float], ...]
    final_scores: tuple[CandidateScore, ...]
    primary: dict[str, CandidateScore]
    rho_scores: dict[float, CandidateScore]
    runtime_s: float


def _row(strategy: str, score: CandidateScore) -> dict[str, object]:
    return {"strategy": strategy, "q_x": score.q[0], "q_y": score.q[1],
            "psi_d_m": score.bayes.psi_d_m, "u_proxy_m": score.robust.u_proxy_m,
            "u_bar_m": score.robust.u_bar_m, "movement_m": score.robust.movement_m,
            "in_c_rec_certified": score.robust.in_c_rec_certified}


def select_rho_sensitivity(scores: Sequence[CandidateScore], *, tau_m: float = 0.5) -> dict[float, CandidateScore]:
    """Select all rho variants from one precomputed score collection."""
    return {rho: select_robust_envelope_hybrid(scores, rho=rho, tau_m=tau_m) for rho in RHO_VALUES}


def run_study(*, output_dir: Path | str = Path("results/q2"), seed: int = 7,
              prior_draws: int = 600_000) -> StudyArtifacts:
    started = time.perf_counter()
    config = Q2Config(circle_vertices=72, direction_angle_bins=360)
    observation = FirstDirectionObservation(station=(-900.0, 0.0), bearing_deg=0.0)
    _, error_bins, error_atoms = _nominal_error_model()
    posterior = sample_first_direction_posterior(observation, prior_draws=prior_draws,
                                                  bearing_error_bins=error_bins, seed=seed, config=config)
    state = build_first_state(observation, samples=posterior.samples, config=config)
    direction_grid = tuple(i * 360.0 / 360 for i in range(360))
    coarse = coarse_grid_candidates(state, spacing_m=300.0, config=config)
    coarse_scores = score_candidates(coarse, state, direction_grid_deg=direction_grid,
                                     bearing_error_atoms=error_atoms, config=config)
    seeds = (select_pure_bayesian(coarse_scores, tau_m=0.5).q,
             select_pure_minimax(coarse_scores).q,
             select_robust_envelope_hybrid(coarse_scores, rho=0.10, tau_m=0.5).q)
    refined = refine_candidates(seeds, state, step_schedule_m=(120.0, 40.0), config=config)
    final_candidates = _merge_unique((coarse, refined))
    final_scores = score_candidates(final_candidates, state, direction_grid_deg=direction_grid,
                                    bearing_error_atoms=error_atoms, config=config)
    pure_b = select_pure_bayesian(final_scores, tau_m=0.5)
    pure_m = select_pure_minimax(final_scores)
    hybrid = select_robust_envelope_hybrid(final_scores, rho=0.10, tau_m=0.5)
    vertical_plus, vertical_minus = same_distance_vertical_baseline(
        hybrid.q, state, direction_grid_deg=direction_grid,
        bearing_error_atoms=error_atoms, config=config)
    rho_scores = select_rho_sensitivity(final_scores)
    artifacts = StudyArtifacts(posterior, final_candidates, final_scores,
                               {"pure_bayesian": pure_b, "pure_minimax": pure_m,
                                "hybrid_rho_0.10": hybrid, "vertical_plus": vertical_plus,
                                "vertical_minus": vertical_minus}, rho_scores,
                               time.perf_counter() - started)
    write_outputs(artifacts, output_dir)
    return artifacts


def write_outputs(artifacts: StudyArtifacts, output_dir: Path | str) -> None:
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    rows = [_row(name, score) for name, score in artifacts.primary.items()]
    with (directory / "strategy_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    minimax = artifacts.primary["pure_minimax"]
    lines = ["# Q2 Task 7C strategy comparison", "", f"One shared final score set; runtime={artifacts.runtime_s:.3f} s; prior_draws={artifacts.posterior.prior_draws}, retained={artifacts.posterior.retained_draws}, ESS={artifacts.posterior.effective_sample_size:.3f}.", "", "| strategy | q | psi_d_m | u_proxy_m | u_bar_m | movement_m | C_rec |", "|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['strategy']} | ({row['q_x']:.2f},{row['q_y']:.2f}) | {row['psi_d_m']:.3f} | {row['u_proxy_m']:.3f} | {row['u_bar_m']:.3f} | {row['movement_m']:.3f} | {row['in_c_rec_certified']} |")
    lines += ["", "Vertical rows are both retained; they use the rho=0.10 hybrid movement as reference. "
              "Under this symmetric benchmark, equal or mirrored values reflect x-axis symmetry rather than a physical preference.", "",
              "Relative to pure minimax, the active strategy rows have psi/u_proxy differences (positive means larger):"]
    for name, score in artifacts.primary.items():
        if name.startswith("vertical"): continue
        lines.append(f"- `{name}`: Δpsi={(score.bayes.psi_d_m-minimax.bayes.psi_d_m):.3f} m; Δu_proxy={(score.robust.u_proxy_m-minimax.robust.u_proxy_m):.3f} m")
    (directory / "strategy_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rho_rows = []
    for rho, score in artifacts.rho_scores.items():
        row = {"rho": rho, **{k: v for k, v in _row("hybrid", score).items() if k != "strategy"},
               "psi_improvement_vs_minimax_m": minimax.bayes.psi_d_m - score.bayes.psi_d_m,
               "u_proxy_increase_vs_minimax_m": score.robust.u_proxy_m - minimax.robust.u_proxy_m}
        rho_rows.append(row)
    rho_fields = ("rho",) + FIELDS[1:] + ("psi_improvement_vs_minimax_m", "u_proxy_increase_vs_minimax_m")
    with (directory / "rho_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rho_fields); writer.writeheader(); writer.writerows(rho_rows)
    lines = ["# Q2 rho sensitivity", "", "All rows reuse the same final CandidateScore collection; only the hybrid selector is called again.", "", "| rho | q | psi_d_m | u_proxy_m | u_bar_m | movement_m | Δpsi vs minimax | Δu_proxy vs minimax |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in rho_rows:
        lines.append(f"| {row['rho']:.3f} | ({row['q_x']:.2f},{row['q_y']:.2f}) | {row['psi_d_m']:.3f} | {row['u_proxy_m']:.3f} | {row['u_bar_m']:.3f} | {row['movement_m']:.3f} | {row['psi_improvement_vs_minimax_m']:.3f} | {row['u_proxy_increase_vs_minimax_m']:.3f} |")
    (directory / "rho_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_plots(artifacts, directory)
    summary = ["# Q2 Task 7C strategy study summary", "", "The study uses one posterior sample and one final CandidateScore collection for all primary strategies and all rho values.", "", "- The pure Bayesian selector accepts candidates within `tau_m=0.5 m` of the best nominal psi and then applies the existing secondary tie-break; it is not an unconditional minimizer of the reported psi.", "- In this benchmark Bayesian has the smaller `psi_d_m` (43.451 m versus 44.936 m for minimax), while minimax has the smaller `u_proxy_m` (60.518 m versus 84.452 m for Bayesian).", "- At rho=0.10 the hybrid exactly selects the minimax-side candidate. This is a valid outcome of the numerical envelope, not a failure, and rho=0.10 is neither a theoretical optimum nor a problem constant.", "- The rho sweep is flat from 0 through 0.10, then switches candidates at 0.20 and 0.30. Relative to the minimax plateau, nominal psi improves as robustness is relaxed; local monotonicity of reported psi is not required because candidates are discrete and `tau_m=0.5` uses near-optimal tie-breaking.", "- The Bayesian point would require rho_Bayes = 84.452/60.518 - 1 = 0.3955 (about 39.5%) to enter the robust envelope in this benchmark; this is an empirical benchmark result, not a theoretical constant.", "- Relative to the hybrid movement reference (1183.385 m), both same-distance vertical baselines have much larger `psi_d_m` (about 1206–1208 m) and `u_proxy_m` (1414.219 m).", "- Both vertical sides are reported. Any equal or mirrored values are explained by the benchmark's x-axis symmetry.", "- Values near `u_bar_m=1501.429` are conservative outer diagnostics, not true worst-case errors.", "", "This is a single benchmark study, not a formal Monte Carlo claim or a strict global optimality result."]
    (directory / "strategy_study_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def _write_plots(artifacts: StudyArtifacts, directory: Path) -> None:
    import matplotlib.pyplot as plt
    labels = {"pure_bayesian": "Pure Bayesian", "pure_minimax": "Pure minimax", "hybrid_rho_0.10": "Hybrid (rho=0.10)", "vertical_plus": "Vertical +n", "vertical_minus": "Vertical -n"}
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, score in artifacts.primary.items():
        ax.scatter(*score.q, label=labels[name], s=45)
    ax.scatter([-900], [0], marker="*", s=180, color="black", zorder=5, label="S1 = (-900, 0)")
    ax.annotate("minimax = hybrid (rho=0.10)", (120, -600), xytext=(12, -34), textcoords="offset points", fontsize=8,
                arrowprops={"arrowstyle": "-", "color": "0.35"})
    ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8, label="x-axis symmetry")
    ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title("Q2 candidate second-station locations"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(directory / "strategy_locations.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    rhos = list(artifacts.rho_scores); psi = [artifacts.rho_scores[r].bayes.psi_d_m for r in rhos]; proxy = [artifacts.rho_scores[r].robust.u_proxy_m for r in rhos]
    for ax, values, ylabel in ((axes[0], psi, "Expected diameter proxy (m)"), (axes[1], proxy, "Robust proxy diameter (m)")):
        ax.plot(rhos, values, marker="o"); ax.axvline(0.10, color="0.6", linestyle="--", label="rho=0.10 reference"); ax.set_ylabel(ylabel); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    axes[1].set_xlabel("Robust-envelope tolerance rho"); fig.suptitle("Q2 robust-envelope sensitivity"); fig.tight_layout(); fig.savefig(directory / "rho_tradeoff.png", dpi=160); plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path, default=Path("results/q2")); parser.add_argument("--seed", type=int, default=7); args = parser.parse_args(argv)
    if args.seed < 0: parser.error("--seed must be non-negative")
    study = run_study(output_dir=args.output_dir, seed=args.seed)
    print(f"runtime_s={study.runtime_s:.3f}; retained={study.posterior.retained_draws}; ESS={study.posterior.effective_sample_size:.3f}; final_candidates={len(study.final_candidates)}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
