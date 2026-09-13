"""Small, reproducible end-to-end Q2 smoke experiment.

This module only orchestrates the frozen Q2 model/optimizer APIs; it is not a
Monte-Carlo experiment or a network client.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Sequence

from src.q2.model import (
    BearingErrorAtom,
    BearingErrorBin,
    FirstDirectionObservation,
    FirstPosteriorSamples,
    Q2Config,
    sample_first_direction_posterior,
    build_first_state,
)
from src.q2.optimizer import (
    CandidateScore,
    coarse_grid_candidates,
    refine_candidates,
    score_candidates,
    select_pure_bayesian,
    select_pure_minimax,
    select_robust_envelope_hybrid,
    same_distance_vertical_baseline,
)


@dataclass(frozen=True)
class ExperimentSummary:
    observation: FirstDirectionObservation
    nominal_error_model_label: str
    posterior: FirstPosteriorSamples
    coarse_count: int
    final_count: int
    rho: float
    tau_m: float
    pure_bayesian: CandidateScore
    pure_minimax: CandidateScore
    hybrid: CandidateScore
    vertical_plus: CandidateScore
    vertical_minus: CandidateScore


def _nominal_error_model() -> tuple[str, tuple[BearingErrorBin, ...], tuple[BearingErrorAtom, ...]]:
    """Return one explicit, reportable nominal model used only by this smoke run.

    The hard +/-1 degree bound is official; the three-bin/three-atom masses are
    an experiment modeling assumption, not a claim about the simulator law.
    """
    label = (
        "experiment modeling assumption (not the true simulator law): "
        "three equal-width first-error bins with masses (0.2,0.6,0.2); "
        "second-error quadrature atoms at (-2/3,0,+2/3)"
    )
    probabilities = (0.2, 0.6, 0.2)
    bins = (
        BearingErrorBin(-1.0, -1.0 / 3.0, probabilities[0]),
        BearingErrorBin(-1.0 / 3.0, 1.0 / 3.0, probabilities[1]),
        BearingErrorBin(1.0 / 3.0, 1.0, probabilities[2]),
    )
    atoms = (
        BearingErrorAtom(-2.0 / 3.0, probabilities[0]),
        BearingErrorAtom(0.0, probabilities[1]),
        BearingErrorAtom(2.0 / 3.0, probabilities[2]),
    )
    return label, bins, atoms


def _merge_unique(groups: Sequence[Sequence[tuple[float, float]]], tolerance_m: float = 1e-6) -> tuple[tuple[float, float], ...]:
    result: list[tuple[float, float]] = []
    for group in groups:
        for point in group:
            q = (float(point[0]), float(point[1]))
            if not any(math.hypot(q[0] - p[0], q[1] - p[1]) <= tolerance_m for p in result):
                result.append(q)
    return tuple(result)


def run_experiment(
    *,
    station: tuple[float, float] = (-900.0, 0.0),
    first_bearing_deg: float = 0.0,
    seed: int = 7,
    prior_draws: int = 3000,
    coarse_spacing_m: float = 300.0,
    refinement_steps_m: Sequence[float] = (120.0, 40.0),
    direction_bins: int = 360,
    circle_vertices: int = 72,
    rho: float = 0.10,
    tau_m: float = 0.5,
) -> ExperimentSummary:
    if prior_draws <= 0 or seed < 0 or direction_bins < 3 or circle_vertices < 3:
        raise ValueError("prior_draws/seed must be non-negative (draws positive), and grid sizes >= 3")
    if not math.isfinite(rho) or rho < 0 or not math.isfinite(tau_m) or tau_m < 0:
        raise ValueError("rho and tau_m must be finite and non-negative")
    if not all(math.isfinite(float(v)) for v in station + (first_bearing_deg,)):
        raise ValueError("station and bearing must be finite")
    config = Q2Config(circle_vertices=circle_vertices, direction_angle_bins=direction_bins)
    observation = FirstDirectionObservation(station=station, bearing_deg=first_bearing_deg)
    label, error_bins, error_atoms = _nominal_error_model()
    posterior = sample_first_direction_posterior(
        observation, prior_draws=prior_draws, bearing_error_bins=error_bins,
        seed=seed, config=config,
    )
    state = build_first_state(observation, samples=posterior.samples, config=config)
    direction_grid = tuple(i * 360.0 / direction_bins for i in range(direction_bins))
    coarse = coarse_grid_candidates(state, spacing_m=coarse_spacing_m, config=config)
    coarse_scores = score_candidates(
        coarse, state, direction_grid_deg=direction_grid,
        bearing_error_atoms=error_atoms, config=config,
    )
    coarse_b = select_pure_bayesian(coarse_scores, tau_m=tau_m)
    coarse_m = select_pure_minimax(coarse_scores)
    coarse_h = select_robust_envelope_hybrid(
        coarse_scores, rho=rho, tau_m=tau_m,
    )
    refined = refine_candidates(
        (coarse_b.q, coarse_m.q, coarse_h.q), state,
        step_schedule_m=refinement_steps_m, config=config,
    )
    final_candidates = _merge_unique((coarse, refined))
    final_scores = score_candidates(
        final_candidates, state, direction_grid_deg=direction_grid,
        bearing_error_atoms=error_atoms, config=config,
    )
    pure_b = select_pure_bayesian(final_scores, tau_m=tau_m)
    pure_m = select_pure_minimax(final_scores)
    hybrid = select_robust_envelope_hybrid(
        final_scores, rho=rho, tau_m=tau_m,
    )
    vertical_plus, vertical_minus = same_distance_vertical_baseline(
        hybrid.q, state, direction_grid_deg=direction_grid,
        bearing_error_atoms=error_atoms, config=config,
    )
    return ExperimentSummary(
        observation, label, posterior, len(coarse), len(final_scores), rho,
        tau_m, pure_b, pure_m, hybrid, vertical_plus, vertical_minus,
    )


def _format_score(name: str, score: CandidateScore) -> str:
    r = score.robust
    return (f"{name}: q=({score.q[0]:.2f}, {score.q[1]:.2f}), "
            f"psi_d_m={score.bayes.psi_d_m:.3f}, u_proxy_m={r.u_proxy_m:.3f}, "
            f"u_bar_m={r.u_bar_m:.3f}, movement_m={r.movement_m:.3f}, "
            f"in_c_rec_certified={r.in_c_rec_certified}")


def print_summary(summary: ExperimentSummary) -> None:
    p = summary.posterior
    print("Q2 Task 7A quick end-to-end summary")
    print(f"observation station={summary.observation.station}, first bearing={summary.observation.bearing_deg:.2f} deg")
    print(f"nominal error model={summary.nominal_error_model_label}")
    print(f"prior_draws={p.prior_draws}, posterior retained_draws={p.retained_draws}, "
          f"posterior ESS={p.effective_sample_size:.3f}, acceptance_rate={p.acceptance_rate:.6f}")
    print(f"coarse candidate count={summary.coarse_count}, refined/final candidate count={summary.final_count}")
    print(f"rho={summary.rho}, tau_m={summary.tau_m}")
    for name, score in (("pure Bayesian", summary.pure_bayesian), ("pure minimax", summary.pure_minimax),
                        ("robust-envelope hybrid", summary.hybrid), ("vertical +n", summary.vertical_plus),
                        ("vertical -n", summary.vertical_minus)):
        print(_format_score(name, score))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run the reproducible smoke configuration")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if not args.quick:
        parser.error("only quick mode is available; pass --quick")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    print_summary(run_experiment(seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
