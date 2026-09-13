"""Minimal Q2 candidate scoring and selector utilities.

Implemented through Task 5B:
- coarse candidate generation over the C_poss outer proxy;
- pure minimax selection using RobustEvaluation.u_proxy_m;
- one shared Bayesian + robust scoring path;
- pure Bayesian selection with a numerical near-optimality tolerance; and
- robust-envelope hybrid selection.

It does not implement local refinement, vertical baselines, plotting, or experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from src.common.geometry import Point
from src.q2.model import (
    BayesianEvaluation,
    BearingErrorAtom,
    FirstState,
    Q2Config,
    RobustEvaluation,
    evaluate_bayesian,
    evaluate_candidate_domains,
    evaluate_robust,
)


@dataclass(frozen=True)
class CandidateScore:
    """Unified nominal/robust score for one admissible candidate point."""

    q: Point
    bayes: BayesianEvaluation
    robust: RobustEvaluation

    def __post_init__(self) -> None:
        if self.bayes.q != self.q or self.robust.q != self.q:
            raise ValueError("candidate score components must refer to the same q")




def coarse_grid_candidates(
    state: FirstState,
    *,
    spacing_m: float,
    config: Q2Config = Q2Config(),
) -> tuple[Point, ...]:
    """Return a deterministic coarse grid over the C_poss outer proxy.

    The search box is the bounding box of ``state.outer_region`` expanded by the
    maximum reception radius.  The grid is aligned to integer multiples of
    ``spacing_m`` so the result does not depend on the polygon's first vertex.
    Points are retained only when
    ``evaluate_candidate_domains(...).in_c_poss_proxy`` is true.

    Q2 detector locations are *not* restricted to the 1800 m source arena.
    """

    if not isinstance(state, FirstState):
        raise TypeError("state must be a FirstState")
    if not state.outer_region:
        raise ValueError("state.outer_region must be non-empty")
    if isinstance(spacing_m, bool) or not isinstance(spacing_m, (int, float)):
        raise ValueError("spacing_m must be a finite positive number")
    spacing = float(spacing_m)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing_m must be a finite positive number")

    xs = [point[0] for point in state.outer_region]
    ys = [point[1] for point in state.outer_region]
    margin = config.reception_radius_max_m

    min_x = min(xs) - margin
    max_x = max(xs) + margin
    min_y = min(ys) - margin
    max_y = max(ys) + margin

    ix_min = math.floor(min_x / spacing)
    ix_max = math.ceil(max_x / spacing)
    iy_min = math.floor(min_y / spacing)
    iy_max = math.ceil(max_y / spacing)

    candidates: list[Point] = []
    for ix in range(ix_min, ix_max + 1):
        x = ix * spacing
        for iy in range(iy_min, iy_max + 1):
            y = iy * spacing
            q = (x, y)
            if evaluate_candidate_domains(q, state, config=config).in_c_poss_proxy:
                candidates.append(q)

    return tuple(candidates)


def minimax_baseline(
    candidates: Sequence[Point],
    state: FirstState,
    *,
    direction_grid_deg: Sequence[float],
    config: Q2Config = Q2Config(),
) -> RobustEvaluation:
    """Return the best finite-candidate, finite-direction minimax baseline.

    The primary criterion is ``u_proxy_m``.  Movement distance is the first
    deterministic tie-break, followed by coordinates.  ``u_bar_m`` is carried
    with the selected evaluation as a conservative diagnostic; it is not the
    primary ranking criterion in this baseline.

    This is therefore a numerical minimax baseline over the supplied candidate
    set, not a continuous-space global optimum.
    """

    if not isinstance(state, FirstState):
        raise TypeError("state must be a FirstState")
    if not candidates:
        raise ValueError("candidates must be non-empty")

    admissible: list[RobustEvaluation] = []
    for q in candidates:
        result = evaluate_robust(
            q,
            state,
            direction_grid_deg=direction_grid_deg,
            config=config,
        )
        if result.in_c_poss_proxy:
            admissible.append(result)

    if not admissible:
        raise ValueError("no candidate lies in the C_poss outer proxy")

    return min(
        admissible,
        key=lambda result: (
            result.u_proxy_m,
            result.movement_m,
            result.q[0],
            result.q[1],
        ),
    )



def score_candidates(
    candidates: Sequence[Point],
    state: FirstState,
    *,
    direction_grid_deg: Sequence[float],
    bearing_error_atoms: Sequence[BearingErrorAtom],
    config: Q2Config = Q2Config(),
) -> tuple[CandidateScore, ...]:
    """Score admissible candidates through one shared Bayesian/robust path.

    The same candidate set and direction grid feed both evaluators so later
    strategy comparisons differ only in the selection rule.  The nominal
    bearing-error PMF/quadrature is explicit and is never inferred from the hard
    ±1 degree bound.

    Candidates outside the ``C_poss`` outer proxy are skipped before either
    expensive evaluator is called.  A fully inadmissible input fails explicitly.
    """

    if not isinstance(state, FirstState):
        raise TypeError("state must be a FirstState")
    if not candidates:
        raise ValueError("candidates must be non-empty")

    scores: list[CandidateScore] = []
    for q in candidates:
        domain = evaluate_candidate_domains(q, state, config=config)
        if not domain.in_c_poss_proxy:
            continue

        bayes = evaluate_bayesian(
            q,
            state,
            direction_grid_deg=direction_grid_deg,
            bearing_error_atoms=bearing_error_atoms,
            config=config,
        )
        robust = evaluate_robust(
            q,
            state,
            direction_grid_deg=direction_grid_deg,
            config=config,
        )
        if not robust.in_c_poss_proxy:
            raise ValueError(
                "candidate-domain inconsistency: precheck admitted q but "
                "robust evaluator rejected it"
            )

        scores.append(CandidateScore(q=q, bayes=bayes, robust=robust))

    if not scores:
        raise ValueError("no candidate lies in the C_poss outer proxy")
    return tuple(scores)


def select_pure_bayesian(
    scores: Sequence[CandidateScore],
    *,
    tau_m: float = 0.0,
) -> CandidateScore:
    """Select the nominal Bayesian candidate from precomputed shared scores.

    Let ``psi*`` be the smallest sample/grid Bayesian proxy.  Every point with
    ``psi_d_m <= psi* + tau_m`` is treated as numerically near-optimal, and the
    shortest movement is selected among that set.  Coordinates provide the final
    deterministic tie-break.

    ``tau_m`` is a numerical tolerance, not a preference weight.
    """

    if not scores:
        raise ValueError("scores must be non-empty")
    if isinstance(tau_m, bool) or not isinstance(tau_m, (int, float)):
        raise ValueError("tau_m must be a finite non-negative number")
    tolerance = float(tau_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tau_m must be a finite non-negative number")

    best_psi = min(score.bayes.psi_d_m for score in scores)
    near_optimal = [
        score
        for score in scores
        if score.bayes.psi_d_m <= best_psi + tolerance
    ]
    return min(
        near_optimal,
        key=lambda score: (
            score.bayes.movement_m,
            score.q[0],
            score.q[1],
        ),
    )



def select_pure_minimax(scores: Sequence[CandidateScore]) -> CandidateScore:
    """Select the pure finite-grid minimax candidate from shared scores.

    The primary criterion is ``robust.u_proxy_m``.  Movement distance and then
    coordinates provide deterministic tie-breaks.  ``u_bar_m`` is reported with
    the chosen score but is not used as the ranking metric here.
    """

    if not scores:
        raise ValueError("scores must be non-empty")
    return min(
        scores,
        key=lambda score: (
            score.robust.u_proxy_m,
            score.robust.movement_m,
            score.q[0],
            score.q[1],
        ),
    )


def select_robust_envelope_hybrid(
    scores: Sequence[CandidateScore],
    *,
    rho: float,
    tau_m: float = 0.0,
    use_outer_envelope: bool = False,
) -> CandidateScore:
    """Select the Bayesian optimum inside a robust near-minimax envelope.

    For the default numerical strategy, let

        U* = min_q u_proxy(q)

    and retain

        A_rho = {q : u_proxy(q) <= (1 + rho) U*}.

    If ``use_outer_envelope`` is true, the same construction instead uses
    ``u_bar_m``.  That variant should be described as a conservative
    outer-envelope implementation because ``u_bar_m`` is an engineering upper
    bound, not exact ``U``.

    Within the envelope, minimize ``psi_d_m``.  Points with
    ``psi_d_m <= psi_best + tau_m`` are treated as numerically near-optimal, and
    movement distance then coordinates break ties.

    ``rho`` is a modeling/sensitivity parameter expressing allowed relative
    degradation from the minimax robust score.  It is not a problem constant.
    ``tau_m`` is only a numerical tolerance.
    """

    if not scores:
        raise ValueError("scores must be non-empty")

    for name, value in (("rho", rho), ("tau_m", tau_m)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite non-negative number")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")

    rho_value = float(rho)
    tolerance = float(tau_m)

    def robust_metric(score: CandidateScore) -> float:
        value = (
            score.robust.u_bar_m
            if use_outer_envelope
            else score.robust.u_proxy_m
        )
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("robust envelope metric must be finite and non-negative")
        return value

    robust_values = tuple(robust_metric(score) for score in scores)
    robust_best = min(robust_values)
    threshold = (1.0 + rho_value) * robust_best

    envelope = [
        score
        for score, value in zip(scores, robust_values)
        if value <= threshold
    ]
    if not envelope:
        # Mathematically impossible for valid finite non-negative metrics because
        # the minimax point itself satisfies the threshold. Keep fail-closed in
        # case future numeric representations violate that invariant.
        raise ValueError("robust envelope is unexpectedly empty")

    for score in envelope:
        if not math.isfinite(score.bayes.psi_d_m) or score.bayes.psi_d_m < 0.0:
            raise ValueError("Bayesian score must be finite and non-negative")
        if (
            not math.isfinite(score.bayes.movement_m)
            or score.bayes.movement_m < 0.0
        ):
            raise ValueError("movement distance must be finite and non-negative")

    best_psi = min(score.bayes.psi_d_m for score in envelope)
    bayes_near_optimal = [
        score
        for score in envelope
        if score.bayes.psi_d_m <= best_psi + tolerance
    ]
    return min(
        bayes_near_optimal,
        key=lambda score: (
            score.bayes.movement_m,
            score.q[0],
            score.q[1],
        ),
    )
