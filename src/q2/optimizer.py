"""Minimal Q2 candidate scoring and selector utilities.

Implemented through Task 6B:
- coarse candidate generation over the C_poss outer proxy;
- deterministic derivative-free local candidate refinement;
- pure minimax selection using RobustEvaluation.u_proxy_m;
- one shared Bayesian + robust scoring path;
- pure Bayesian selection with a numerical near-optimality tolerance;
- robust-envelope hybrid selection; and
- the same-distance two-sided vertical baseline.

It does not implement plotting or large experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from src.common.geometry import Point, distance, unit_vector
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


def refine_candidates(
    seeds: Sequence[Point],
    state: FirstState,
    *,
    step_schedule_m: Sequence[float],
    dedup_tolerance_m: float = 1e-6,
    config: Q2Config = Q2Config(),
) -> tuple[Point, ...]:
    """Generate a deterministic multi-scale local cloud around seed points.

    Refinement is deliberately objective-free: it does not evaluate Bayesian,
    minimax, or hybrid scores.  The caller supplies already interesting seeds,
    this function adds eight equally spaced neighbors at each requested radius,
    and the combined set is rescored later through ``score_candidates(...)``.

    Only points inside the engineering ``C_poss`` outer proxy are retained.
    Detector positions are not clipped to the 1800 m source arena.

    Points within ``dedup_tolerance_m`` Euclidean distance of an already retained
    point are treated as duplicates.  The default tolerance is 1 micrometre,
    negligible relative to metre-scale search steps but sufficient to remove
    floating-point duplicates produced by repeated seeds or overlapping stars.

    Output order is deterministic: admissible seeds first (input order), then
    neighbors by step-schedule order, seed order, and angles
    0, 45, ..., 315 degrees.
    """

    if not isinstance(state, FirstState):
        raise TypeError("state must be a FirstState")
    if not seeds:
        raise ValueError("seeds must be non-empty")
    if not step_schedule_m:
        raise ValueError("step_schedule_m must be non-empty")

    if (
        isinstance(dedup_tolerance_m, bool)
        or not isinstance(dedup_tolerance_m, (int, float))
        or not math.isfinite(float(dedup_tolerance_m))
        or float(dedup_tolerance_m) < 0.0
    ):
        raise ValueError("dedup_tolerance_m must be a finite non-negative number")
    tolerance = float(dedup_tolerance_m)

    normalized_seeds: list[Point] = []
    for seed in seeds:
        try:
            coordinates = tuple(seed)
        except TypeError as exc:
            raise ValueError("each seed must be a finite 2D point") from exc
        if len(coordinates) != 2 or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in coordinates
        ):
            raise ValueError("each seed must be a finite 2D point")
        normalized_seeds.append((float(coordinates[0]), float(coordinates[1])))

    steps: list[float] = []
    for step in step_schedule_m:
        if (
            isinstance(step, bool)
            or not isinstance(step, (int, float))
            or not math.isfinite(float(step))
            or float(step) <= 0.0
        ):
            raise ValueError("every refinement step must be a finite positive number")
        steps.append(float(step))

    retained: list[Point] = []

    def is_duplicate(q: Point) -> bool:
        return any(distance(q, existing) <= tolerance for existing in retained)

    def retain_if_admissible(q: Point) -> None:
        if is_duplicate(q):
            return
        domain = evaluate_candidate_domains(q, state, config=config)
        if domain.in_c_poss_proxy:
            retained.append(q)

    for seed in normalized_seeds:
        retain_if_admissible(seed)

    angles_deg = tuple(float(angle) for angle in range(0, 360, 45))
    unit_offsets = tuple(unit_vector(angle) for angle in angles_deg)
    for step in steps:
        for seed in normalized_seeds:
            for ux, uy in unit_offsets:
                q = (seed[0] + step * ux, seed[1] + step * uy)
                retain_if_admissible(q)

    if not retained:
        raise ValueError("no refined candidate lies in the C_poss outer proxy")
    return tuple(retained)


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



def same_distance_vertical_baseline(
    reference_q: Point,
    state: FirstState,
    *,
    direction_grid_deg: Sequence[float],
    bearing_error_atoms: Sequence[BearingErrorAtom],
    config: Q2Config = Q2Config(),
) -> tuple[CandidateScore, CandidateScore]:
    """Score both perpendicular points at the reference movement distance.

    Let ``L = ||reference_q - S1||`` and let ``u`` be the first measured bearing
    axis.  With the left normal ``n = (-u_y, u_x)``, the two baseline points are

        q_plus  = S1 + L n
        q_minus = S1 - L n.

    Both sides are always evaluated and returned in ``(+n, -n)`` order.  The
    function does not silently pick the better side.  It also does not discard a
    side merely because it falls outside the ``C_poss`` outer proxy: that flag is
    preserved in the returned robust diagnostic so the comparison can report a
    weak baseline honestly.

    The nominal bearing-error quadrature/PMF is explicit, matching the shared
    Bayesian evaluator; no probability law is inferred from the hard error bound.
    """

    if not isinstance(state, FirstState):
        raise TypeError("state must be a FirstState")
    try:
        coordinates = tuple(reference_q)
    except TypeError as exc:
        raise ValueError("reference_q must be a finite 2D point") from exc
    if len(coordinates) != 2 or not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in coordinates
    ):
        raise ValueError("reference_q must be a finite 2D point")

    reference = (float(coordinates[0]), float(coordinates[1]))
    station = state.observation.station
    movement = distance(reference, station)

    forward = unit_vector(state.observation.bearing_deg)
    normal = (-forward[1], forward[0])
    q_plus = (
        station[0] + movement * normal[0],
        station[1] + movement * normal[1],
    )
    q_minus = (
        station[0] - movement * normal[0],
        station[1] - movement * normal[1],
    )

    def evaluate(q: Point) -> CandidateScore:
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
        return CandidateScore(q=q, bayes=bayes, robust=robust)

    return evaluate(q_plus), evaluate(q_minus)
