"""Minimal Q2 model primitives.

This module covers state construction, candidate-domain checks, second-response
support updates, the nominal Bayesian evaluator, and the single-candidate robust
evaluator.  It intentionally does not implement the optimizer or large experiments.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from src.common.geometry import (
    Point,
    clip_polygon_to_bearing_wedge,
    clip_polygon_to_circle_outer,
    distance,
    minimum_enclosing_circle,
    normalize_angle_deg,
    polygon_diameter,
    signed_angle_difference_deg,
)
from src.common.localization import BearingObservation, localization_region


_DISTANCE_TOLERANCE_M = 1e-9
_ANGLE_TOLERANCE_DEG = 1e-12


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


@dataclass(frozen=True)
class Q2Config:
    arena_radius_m: float = 1800.0
    reception_radius_min_m: float = 1000.0
    reception_radius_max_m: float = 1500.0
    near_radius_m: float = 5.0
    bearing_error_deg: float = 1.0
    circle_vertices: int = 720
    direction_angle_bins: int = 720

    def __post_init__(self) -> None:
        numeric_values = (
            self.arena_radius_m,
            self.reception_radius_min_m,
            self.reception_radius_max_m,
            self.near_radius_m,
            self.bearing_error_deg,
            self.circle_vertices,
            self.direction_angle_bins,
        )
        if not all(_is_finite_number(value) for value in numeric_values):
            raise ValueError("Q2Config parameters must be finite")
        if self.arena_radius_m <= 0.0:
            raise ValueError("arena_radius_m must be positive")
        if not (
            0.0
            < self.near_radius_m
            < self.reception_radius_min_m
            <= self.reception_radius_max_m
        ):
            raise ValueError(
                "radii must satisfy 0 < near < reception_min <= reception_max"
            )
        if self.bearing_error_deg < 0.0:
            raise ValueError("bearing_error_deg must be non-negative")
        if isinstance(self.circle_vertices, bool) or not isinstance(
            self.circle_vertices, int
        ):
            raise ValueError("circle_vertices must be an integer")
        if isinstance(self.direction_angle_bins, bool) or not isinstance(
            self.direction_angle_bins, int
        ):
            raise ValueError("direction_angle_bins must be an integer")
        if self.circle_vertices < 3:
            raise ValueError("circle_vertices must be at least 3")
        if self.direction_angle_bins < 3:
            raise ValueError("direction_angle_bins must be at least 3")


@dataclass(frozen=True)
class FirstDirectionObservation:
    station: Point
    bearing_deg: float

    def __post_init__(self) -> None:
        _validate_point(self.station, "station")
        if not _is_finite_number(self.bearing_deg):
            raise ValueError("bearing_deg must be finite")
        object.__setattr__(self, "bearing_deg", normalize_angle_deg(self.bearing_deg))


@dataclass(frozen=True)
class JointSample:
    position: Point
    reception_radius_m: float
    weight: float

    def __post_init__(self) -> None:
        _validate_point(self.position, "position")
        if not _is_finite_number(self.reception_radius_m):
            raise ValueError("reception_radius_m must be finite")
        if not _is_finite_number(self.weight):
            raise ValueError("weight must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be non-negative")


@dataclass(frozen=True)
class FirstState:
    observation: FirstDirectionObservation
    exact_support_label: str
    outer_region: tuple[Point, ...]
    joint_samples: tuple[JointSample, ...]


@dataclass(frozen=True)
class CandidateDomainFlags:
    in_c_poss_proxy: bool
    in_c_rec_certified: bool
    min_distance_to_outer_m: float
    max_distance_to_outer_m: float


class SecondResponseKind(str, Enum):
    NEAR = "near"
    DIRECTION = "direction"
    NO_SIGNAL = "no_signal"


@dataclass(frozen=True)
class SecondResponse:
    kind: SecondResponseKind
    bearing_deg: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SecondResponseKind):
            try:
                object.__setattr__(self, "kind", SecondResponseKind(self.kind))
            except ValueError as exc:
                raise ValueError("unknown second response kind") from exc

        if self.kind is SecondResponseKind.DIRECTION:
            if self.bearing_deg is None:
                raise ValueError("direction response requires bearing_deg")
            if not _is_finite_number(self.bearing_deg):
                raise ValueError("bearing_deg must be finite")
            object.__setattr__(
                self,
                "bearing_deg",
                normalize_angle_deg(self.bearing_deg),
            )
        elif self.bearing_deg is not None:
            raise ValueError("near and no_signal responses must not include bearing_deg")


@dataclass(frozen=True)
class SecondSupport:
    """Nominal sample support plus a conservative positional outer region."""

    response: SecondResponse
    true_support_label: str
    sample_support: tuple[JointSample, ...]
    conservative_outer_region: tuple[Point, ...]


@dataclass(frozen=True)
class BearingErrorBin:
    """One explicit bin of the nominal first-bearing error density.

    The hard ±delta bound is official; probabilities inside that bound are a
    modeling choice.  Within each bin the nominal density is piecewise constant.
    """

    lower_deg: float
    upper_deg: float
    probability: float

    def __post_init__(self) -> None:
        if not all(
            _is_finite_number(value)
            for value in (self.lower_deg, self.upper_deg, self.probability)
        ):
            raise ValueError("bearing error bin values must be finite")
        if self.upper_deg <= self.lower_deg:
            raise ValueError("bearing error bin must have positive width")
        if self.probability < 0.0:
            raise ValueError("bearing error bin probability must be non-negative")


@dataclass(frozen=True)
class FirstPosteriorSamples:
    """Diagnostics plus normalized nominal samples for p(G,R | first direction)."""

    samples: tuple[JointSample, ...]
    prior_draws: int
    retained_draws: int
    effective_sample_size: float
    acceptance_rate: float
    seed: int


@dataclass(frozen=True)
class BearingErrorAtom:
    """One explicit quadrature atom for the nominal bearing-error model.

    The official problem supplies only a hard error bound, not a probability
    density.  Task 3A therefore requires callers to provide the nominal error
    law explicitly instead of silently assuming a uniform distribution.
    """

    offset_deg: float
    probability: float

    def __post_init__(self) -> None:
        if not _is_finite_number(self.offset_deg):
            raise ValueError("bearing error offset must be finite")
        if not _is_finite_number(self.probability):
            raise ValueError("bearing error probability must be finite")
        if self.probability < 0.0:
            raise ValueError("bearing error probability must be non-negative")


@dataclass(frozen=True)
class ResponseMetric:
    response: SecondResponse
    probability: float
    diameter_true_proxy_m: float
    diameter_outer_m: float
    clear_radius_outer_m: float | None


@dataclass(frozen=True)
class BayesianEvaluation:
    q: Point
    psi_d_m: float
    response_probabilities: dict[SecondResponseKind, float]
    metrics: tuple[ResponseMetric, ...]
    movement_m: float


@dataclass(frozen=True)
class RobustEvaluation:
    """Finite-grid robust proxy plus continuous-response conservative outer bound."""

    q: Point
    u_proxy_m: float
    u_bar_m: float
    worst_response_proxy: SecondResponse | None
    worst_response_outer: SecondResponse | None
    in_c_poss_proxy: bool
    in_c_rec_certified: bool
    movement_m: float


def build_first_state(
    observation: FirstDirectionObservation,
    *,
    samples: Sequence[JointSample],
    config: Q2Config = Q2Config(),
) -> FirstState:
    """Build the first-observation Q2 state for an existing direction response."""

    if not isinstance(observation, FirstDirectionObservation):
        observation = FirstDirectionObservation(
            station=observation.station,  # type: ignore[attr-defined]
            bearing_deg=observation.bearing_deg,  # type: ignore[attr-defined]
        )

    q1_region = localization_region(
        [
            BearingObservation(
                station=observation.station,
                bearing_deg=observation.bearing_deg,
                error_deg=config.bearing_error_deg,
            )
        ],
        arena_radius_m=config.arena_radius_m,
        circle_vertices=config.circle_vertices,
        reception_radius_upper_m=config.reception_radius_max_m,
    )
    if not q1_region:
        raise ValueError("conservative first-observation outer region is empty")

    if not samples:
        raise ValueError("samples must be non-empty")
    if not any(sample.weight > 0.0 for sample in samples):
        raise ValueError("at least one input sample must have positive weight")

    filtered_samples = tuple(
        sample
        for sample in samples
        if sample.weight > 0.0
        and _is_sample_compatible_with_first_direction(sample, observation, config)
    )
    if not filtered_samples:
        raise ValueError("no samples remain after first-direction filtering")
    if not any(sample.weight > 0.0 for sample in filtered_samples):
        raise ValueError("filtered samples must include positive weight")

    return FirstState(
        observation=observation,
        exact_support_label=(
            "closure(F1)=Omega cap W(S1,theta1,delta) cap B(S1,1500); "
            "nominal direction branch additionally enforces ||G-S1|| > 5"
        ),
        outer_region=tuple(q1_region),
        joint_samples=filtered_samples,
    )


def sample_first_direction_posterior(
    observation: FirstDirectionObservation,
    *,
    prior_draws: int,
    bearing_error_bins: Sequence[BearingErrorBin],
    seed: int = 0,
    min_effective_sample_size: float | None = None,
    config: Q2Config = Q2Config(),
) -> FirstPosteriorSamples:
    """Sample the nominal joint posterior after the first ``direction`` response.

    Priors follow the adopted nominal assumptions:
    ``G ~ Uniform(area on the arena disk)`` and
    ``R ~ Uniform[reception_radius_min_m, reception_radius_max_m]``.

    The first-bearing likelihood is *not* inferred from the official hard bound.
    Callers must provide ``bearing_error_bins`` explicitly.  Each bin specifies a
    probability mass over an error interval and induces a piecewise-constant
    density within that interval.  Returned ``JointSample.weight`` values are
    normalized posterior importance weights.

    This is a rejection/importance-sampling proxy, not an exact posterior
    integral.  The effective sample size and acceptance rate are returned so
    experiments can check convergence instead of silently trusting a thin cloud.
    """

    if not isinstance(observation, FirstDirectionObservation):
        observation = FirstDirectionObservation(
            station=observation.station,  # type: ignore[attr-defined]
            bearing_deg=observation.bearing_deg,  # type: ignore[attr-defined]
        )
    if isinstance(prior_draws, bool) or not isinstance(prior_draws, int):
        raise ValueError("prior_draws must be an integer")
    if prior_draws <= 0:
        raise ValueError("prior_draws must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if min_effective_sample_size is not None:
        if not _is_finite_number(min_effective_sample_size):
            raise ValueError("min_effective_sample_size must be finite")
        if min_effective_sample_size <= 0.0:
            raise ValueError("min_effective_sample_size must be positive")

    bins = _validate_bearing_error_bins(bearing_error_bins, config)
    rng = random.Random(seed)
    weighted_draws: list[tuple[Point, float, float]] = []

    for _ in range(prior_draws):
        # Uniform by area on the disk: radius = R_arena * sqrt(U).
        radial = config.arena_radius_m * math.sqrt(rng.random())
        azimuth = math.tau * rng.random()
        position = (
            radial * math.cos(azimuth),
            radial * math.sin(azimuth),
        )
        reception_radius = rng.uniform(
            config.reception_radius_min_m,
            config.reception_radius_max_m,
        )

        station_distance = distance(position, observation.station)
        if station_distance <= config.near_radius_m + _DISTANCE_TOLERANCE_M:
            continue
        if station_distance > reception_radius + _DISTANCE_TOLERANCE_M:
            continue

        true_bearing = _bearing_deg(observation.station, position)
        observed_error = signed_angle_difference_deg(
            observation.bearing_deg,
            true_bearing,
        )
        if (
            abs(observed_error)
            > config.bearing_error_deg + _ANGLE_TOLERANCE_DEG
        ):
            continue

        likelihood_density = _bearing_error_density(
            observed_error,
            bins,
            config,
        )
        if likelihood_density <= 0.0:
            continue
        weighted_draws.append(
            (position, reception_radius, likelihood_density)
        )

    if not weighted_draws:
        raise ValueError(
            "no posterior samples retained; increase prior_draws or revise the "
            "explicit nominal bearing-error model"
        )

    total_weight = sum(weight for _, _, weight in weighted_draws)
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("posterior importance weights have invalid total mass")

    samples = tuple(
        JointSample(position, reception_radius, weight / total_weight)
        for position, reception_radius, weight in weighted_draws
    )
    squared_weight_sum = sum(sample.weight * sample.weight for sample in samples)
    effective_sample_size = 1.0 / squared_weight_sum
    if (
        min_effective_sample_size is not None
        and effective_sample_size + 1e-12 < min_effective_sample_size
    ):
        raise ValueError(
            "effective sample size below requested threshold: "
            f"{effective_sample_size:.6g} < {min_effective_sample_size:.6g}"
        )

    return FirstPosteriorSamples(
        samples=samples,
        prior_draws=prior_draws,
        retained_draws=len(samples),
        effective_sample_size=effective_sample_size,
        acceptance_rate=len(samples) / prior_draws,
        seed=seed,
    )


def evaluate_candidate_domains(
    q: Point,
    state: FirstState,
    *,
    config: Q2Config = Q2Config(),
) -> CandidateDomainFlags:
    """Evaluate Q2 candidate-domain proxies against the conservative outer region."""

    _validate_point(q, "q")
    if not state.outer_region:
        raise ValueError("state.outer_region must be non-empty")

    min_distance = _point_to_polygon_distance(q, state.outer_region)
    max_distance = max(distance(q, vertex) for vertex in state.outer_region)
    return CandidateDomainFlags(
        in_c_poss_proxy=(
            min_distance <= config.reception_radius_max_m + _DISTANCE_TOLERANCE_M
        ),
        in_c_rec_certified=(
            max_distance <= config.reception_radius_min_m + _DISTANCE_TOLERANCE_M
        ),
        min_distance_to_outer_m=min_distance,
        max_distance_to_outer_m=max_distance,
    )


def second_support(
    q: Point,
    response: SecondResponse,
    state: FirstState,
    *,
    config: Q2Config = Q2Config(),
) -> SecondSupport:
    """Update the nominal samples and conservative outer region for one response.

    ``sample_support`` is a weighted-sample proxy for the theoretical joint
    ``(G, R)`` support.  ``conservative_outer_region`` is a positional outer
    region used for engineering bounds; it must not be interpreted as the exact
    support.
    """

    _validate_point(q, "q")
    if not isinstance(response, SecondResponse):
        raise TypeError("response must be a SecondResponse")
    if not isinstance(state, FirstState):
        raise TypeError("state must be a FirstState")
    if not state.outer_region:
        raise ValueError("state.outer_region must be non-empty")

    sample_support = tuple(
        sample
        for sample in state.joint_samples
        if sample.weight > 0.0
        and _is_sample_compatible_with_second_response(sample, q, response, config)
    )

    if response.kind is SecondResponseKind.NEAR:
        outer_region = clip_polygon_to_circle_outer(
            state.outer_region,
            q,
            config.near_radius_m,
            config.circle_vertices,
        )
        true_support_label = "F1 cap B(q,5)"

    elif response.kind is SecondResponseKind.DIRECTION:
        assert response.bearing_deg is not None
        outer_region = clip_polygon_to_bearing_wedge(
            state.outer_region,
            q,
            response.bearing_deg,
            config.bearing_error_deg,
        )
        outer_region = clip_polygon_to_circle_outer(
            outer_region,
            q,
            config.reception_radius_max_m,
            config.circle_vertices,
        )
        true_support_label = (
            "joint (G,R) support compatible with 5 < ||G-q|| <= R and "
            "the second bearing hard bound"
        )

    else:
        # ``no_signal`` implies ||G-q|| > R >= 1000 in the theoretical joint
        # support.  Until a reliable non-convex outer construction is added,
        # retaining K1_out is deliberately loose but conservative.
        outer_region = list(state.outer_region)
        true_support_label = "joint (G,R) support compatible with ||G-q|| > R"

    return SecondSupport(
        response=response,
        true_support_label=true_support_label,
        sample_support=sample_support,
        conservative_outer_region=tuple(outer_region),
    )



def evaluate_bayesian(
    q: Point,
    state: FirstState,
    *,
    direction_grid_deg: Sequence[float],
    bearing_error_atoms: Sequence[BearingErrorAtom],
    config: Q2Config = Q2Config(),
) -> BayesianEvaluation:
    """Evaluate the nominal expected post-measurement support diameter.

    ``state.joint_samples`` is interpreted as a weighted nominal approximation
    to the joint posterior ``p(G, R | H1)``.  The weights need not be normalized.

    The statement gives only the hard bearing-error bound.  A probability law is
    therefore *not* invented here: callers must provide an explicit discrete
    quadrature/PMF through ``bearing_error_atoms``.  The returned ``psi_d_m`` is
    a sample-and-grid proxy, not an exact Bayesian integral.

    For a second measurement at exactly the first station, the fixed same-place
    error rule is respected: the observed direction is reused exactly and no
    independent error draw is applied.
    """

    _validate_point(q, "q")
    if not isinstance(state, FirstState):
        raise TypeError("state must be a FirstState")
    if not state.outer_region:
        raise ValueError("state.outer_region must be non-empty")

    centers, bin_width_deg = _prepare_direction_grid(direction_grid_deg)
    atoms = _validate_bearing_error_atoms(bearing_error_atoms, config)

    weighted_samples = tuple(sample for sample in state.joint_samples if sample.weight > 0.0)
    total_weight = sum(sample.weight for sample in weighted_samples)
    if not weighted_samples or total_weight <= 0.0 or not math.isfinite(total_weight):
        raise ValueError("state.joint_samples must contain positive finite total weight")

    # Same-location repeats reuse the fixed first error and cannot create new
    # independent bearing information.
    if distance(q, state.observation.station) <= _DISTANCE_TOLERANCE_M:
        positions = tuple(sample.position for sample in weighted_samples)
        true_proxy = polygon_diameter(positions)
        outer = tuple(state.outer_region)
        outer_diameter = polygon_diameter(outer)
        clear_radius = minimum_enclosing_circle(outer).radius
        response = SecondResponse(
            SecondResponseKind.DIRECTION,
            state.observation.bearing_deg,
        )
        metric = ResponseMetric(
            response=response,
            probability=1.0,
            diameter_true_proxy_m=true_proxy,
            diameter_outer_m=outer_diameter,
            clear_radius_outer_m=clear_radius,
        )
        return BayesianEvaluation(
            q=q,
            psi_d_m=true_proxy,
            response_probabilities={
                SecondResponseKind.NEAR: 0.0,
                SecondResponseKind.DIRECTION: 1.0,
                SecondResponseKind.NO_SIGNAL: 0.0,
            },
            metrics=(metric,),
            movement_m=0.0,
        )

    # Each branch stores total probability mass and the posterior-support
    # positions that receive non-zero mass under the explicit nominal model.
    branch_mass: dict[tuple[SecondResponseKind, float | None], float] = {}
    branch_positions: dict[
        tuple[SecondResponseKind, float | None],
        set[Point],
    ] = {}

    def add_mass(
        key: tuple[SecondResponseKind, float | None],
        sample: JointSample,
        mass: float,
    ) -> None:
        if mass <= 0.0:
            return
        branch_mass[key] = branch_mass.get(key, 0.0) + mass
        branch_positions.setdefault(key, set()).add(sample.position)

    for sample in weighted_samples:
        source_distance = distance(sample.position, q)
        if source_distance <= config.near_radius_m + _DISTANCE_TOLERANCE_M:
            add_mass((SecondResponseKind.NEAR, None), sample, sample.weight)
            continue
        if source_distance > sample.reception_radius_m + _DISTANCE_TOLERANCE_M:
            add_mass((SecondResponseKind.NO_SIGNAL, None), sample, sample.weight)
            continue

        true_bearing = _bearing_deg(q, sample.position)
        for atom in atoms:
            if atom.probability <= 0.0:
                continue
            measured_bearing = normalize_angle_deg(true_bearing + atom.offset_deg)
            center = _nearest_direction_center(measured_bearing, centers)
            add_mass(
                (SecondResponseKind.DIRECTION, center),
                sample,
                sample.weight * atom.probability,
            )

    total_branch_mass = sum(branch_mass.values())
    if total_branch_mass <= 0.0 or not math.isfinite(total_branch_mass):
        raise ValueError("nominal response model produced zero total probability")

    probability_tolerance = 1e-10
    expected_total = total_weight
    if not math.isclose(
        total_branch_mass,
        expected_total,
        rel_tol=probability_tolerance,
        abs_tol=probability_tolerance,
    ):
        raise ValueError("nominal response masses do not sum to the sample weight")

    def sort_key(
        item: tuple[SecondResponseKind, float | None],
    ) -> tuple[int, float]:
        kind, bearing = item
        if kind is SecondResponseKind.NEAR:
            return (0, 0.0)
        if kind is SecondResponseKind.DIRECTION:
            assert bearing is not None
            return (1, bearing)
        return (2, 0.0)

    metrics: list[ResponseMetric] = []
    kind_probabilities = {
        SecondResponseKind.NEAR: 0.0,
        SecondResponseKind.DIRECTION: 0.0,
        SecondResponseKind.NO_SIGNAL: 0.0,
    }

    for key in sorted(branch_mass, key=sort_key):
        kind, bearing = key
        probability = branch_mass[key] / total_weight
        positions = tuple(sorted(branch_positions[key]))
        true_proxy = polygon_diameter(positions)

        if kind is SecondResponseKind.NEAR:
            response = SecondResponse(SecondResponseKind.NEAR)
            outer = second_support(q, response, state, config=config).conservative_outer_region
        elif kind is SecondResponseKind.NO_SIGNAL:
            response = SecondResponse(SecondResponseKind.NO_SIGNAL)
            outer = second_support(q, response, state, config=config).conservative_outer_region
        else:
            assert bearing is not None
            response = SecondResponse(SecondResponseKind.DIRECTION, bearing)
            outer = _direction_bin_outer_region(
                q,
                bearing,
                bin_width_deg,
                state,
                config,
            )

        if not outer:
            raise ValueError(
                "positive-probability response has an empty conservative outer region"
            )

        outer_diameter = polygon_diameter(outer)
        clear_radius = minimum_enclosing_circle(outer).radius
        metric = ResponseMetric(
            response=response,
            probability=probability,
            diameter_true_proxy_m=true_proxy,
            diameter_outer_m=outer_diameter,
            clear_radius_outer_m=clear_radius,
        )
        metrics.append(metric)
        kind_probabilities[kind] += probability

    probability_sum = sum(metric.probability for metric in metrics)
    if not math.isclose(probability_sum, 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("response probabilities do not sum to one")

    psi_d = sum(
        metric.probability * metric.diameter_true_proxy_m
        for metric in metrics
    )
    return BayesianEvaluation(
        q=q,
        psi_d_m=psi_d,
        response_probabilities=kind_probabilities,
        metrics=tuple(metrics),
        movement_m=distance(q, state.observation.station),
    )



def evaluate_robust(
    q: Point,
    state: FirstState,
    *,
    direction_grid_deg: Sequence[float],
    config: Q2Config = Q2Config(),
) -> RobustEvaluation:
    """Evaluate one candidate under hard-bound worst-case geometry.

    ``u_proxy_m`` is a finite center-grid, sample-supported proxy for
    ``U(q) = sup_z D(S2(z,q))``.  It is useful for ranking candidates but is not
    a continuous-response certificate.

    ``u_bar_m`` is a conservative outer upper bound over continuous direction
    responses.  Each direction grid center represents a full angular bin; its
    outer region uses the widened half-angle
    ``bearing_error_deg + bin_width_deg/2``.  ``near`` uses the certified 5 m
    outer disk.  ``no_signal`` uses ``K1_out`` unless it is ruled out by the
    certified reception condition ``C_rec``.

    The two quantities intentionally serve different roles and must not be
    reported as equal or as exact worst-case values.
    """

    _validate_point(q, "q")
    if not isinstance(state, FirstState):
        raise TypeError("state must be a FirstState")
    if not state.outer_region:
        raise ValueError("state.outer_region must be non-empty")

    centers, bin_width_deg = _prepare_direction_grid(direction_grid_deg)
    domain = evaluate_candidate_domains(q, state, config=config)
    movement_m = distance(q, state.observation.station)

    weighted_samples = tuple(
        sample for sample in state.joint_samples if sample.weight > 0.0
    )
    if not weighted_samples:
        raise ValueError("state.joint_samples must contain positive-weight samples")

    # At the exact first station the same-place error is fixed, so a repeated
    # measurement cannot split the support or provide an independent bearing.
    if movement_m <= _DISTANCE_TOLERANCE_M:
        positions = tuple(sample.position for sample in weighted_samples)
        response = SecondResponse(
            SecondResponseKind.DIRECTION,
            state.observation.bearing_deg,
        )
        return RobustEvaluation(
            q=q,
            u_proxy_m=polygon_diameter(positions),
            u_bar_m=polygon_diameter(state.outer_region),
            worst_response_proxy=response,
            worst_response_outer=response,
            in_c_poss_proxy=domain.in_c_poss_proxy,
            in_c_rec_certified=domain.in_c_rec_certified,
            movement_m=0.0,
        )

    proxy_candidates: list[tuple[float, SecondResponse]] = []

    near_response = SecondResponse(SecondResponseKind.NEAR)
    near_support = second_support(q, near_response, state, config=config)
    if near_support.sample_support:
        proxy_candidates.append(
            (
                polygon_diameter(
                    tuple(sample.position for sample in near_support.sample_support)
                ),
                near_response,
            )
        )

    no_signal_response = SecondResponse(SecondResponseKind.NO_SIGNAL)
    no_signal_support = second_support(q, no_signal_response, state, config=config)
    if no_signal_support.sample_support:
        proxy_candidates.append(
            (
                polygon_diameter(
                    tuple(sample.position for sample in no_signal_support.sample_support)
                ),
                no_signal_response,
            )
        )

    for center in centers:
        response = SecondResponse(SecondResponseKind.DIRECTION, center)
        support = second_support(q, response, state, config=config)
        if not support.sample_support:
            continue
        proxy_candidates.append(
            (
                polygon_diameter(
                    tuple(sample.position for sample in support.sample_support)
                ),
                response,
            )
        )

    if not proxy_candidates:
        raise ValueError(
            "finite direction grid has no sample-supported response; "
            "refine direction_grid_deg"
        )

    # max() is stable for ties because candidates are appended in deterministic
    # near / no_signal / ascending-direction order.
    u_proxy_m, worst_response_proxy = max(
        proxy_candidates,
        key=lambda item: item[0],
    )

    outer_candidates: list[tuple[float, SecondResponse]] = []

    # A non-empty near outer is a conservative witness that near may be feasible.
    if near_support.conservative_outer_region:
        outer_candidates.append(
            (
                polygon_diameter(near_support.conservative_outer_region),
                near_response,
            )
        )

    # If C_rec is certified, every true source position is within 1000 m and
    # R >= 1000 m, so no_signal is impossible.  Otherwise the first safe outer
    # remains K1_out; it is deliberately loose but cannot under-cover.
    if not domain.in_c_rec_certified:
        outer_candidates.append(
            (
                polygon_diameter(state.outer_region),
                no_signal_response,
            )
        )

    for center in centers:
        outer = _direction_bin_outer_region(
            q,
            center,
            bin_width_deg,
            state,
            config,
        )
        if not outer:
            continue
        outer_candidates.append(
            (
                polygon_diameter(outer),
                SecondResponse(SecondResponseKind.DIRECTION, center),
            )
        )

    if not outer_candidates:
        raise ValueError("no conservative second-response outer region is non-empty")

    u_bar_m, worst_response_outer = max(
        outer_candidates,
        key=lambda item: item[0],
    )

    return RobustEvaluation(
        q=q,
        u_proxy_m=u_proxy_m,
        u_bar_m=u_bar_m,
        worst_response_proxy=worst_response_proxy,
        worst_response_outer=worst_response_outer,
        in_c_poss_proxy=domain.in_c_poss_proxy,
        in_c_rec_certified=domain.in_c_rec_certified,
        movement_m=movement_m,
    )


def _validate_bearing_error_bins(
    bins: Sequence[BearingErrorBin],
    config: Q2Config,
) -> tuple[BearingErrorBin, ...]:
    if not bins:
        raise ValueError(
            "bearing_error_bins is required because the problem gives no error density"
        )

    prepared = tuple(
        item if isinstance(item, BearingErrorBin) else BearingErrorBin(*item)  # type: ignore[arg-type]
        for item in bins
    )
    ordered = tuple(sorted(prepared, key=lambda item: (item.lower_deg, item.upper_deg)))

    previous_upper: float | None = None
    for item in ordered:
        if item.lower_deg < -config.bearing_error_deg - _ANGLE_TOLERANCE_DEG:
            raise ValueError("bearing error bin lies outside the hard error bound")
        if item.upper_deg > config.bearing_error_deg + _ANGLE_TOLERANCE_DEG:
            raise ValueError("bearing error bin lies outside the hard error bound")
        if (
            previous_upper is not None
            and item.lower_deg < previous_upper - _ANGLE_TOLERANCE_DEG
        ):
            raise ValueError("bearing error bins must not overlap")
        previous_upper = item.upper_deg

    probability_sum = sum(item.probability for item in ordered)
    if not math.isclose(probability_sum, 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("bearing error bin probabilities must sum to one")
    if not any(item.probability > 0.0 for item in ordered):
        raise ValueError("bearing error model must contain positive probability")
    return ordered


def _bearing_error_density(
    error_deg: float,
    bins: Sequence[BearingErrorBin],
    config: Q2Config,
) -> float:
    if abs(error_deg) > config.bearing_error_deg + _ANGLE_TOLERANCE_DEG:
        return 0.0

    for index, item in enumerate(bins):
        is_last = index == len(bins) - 1
        inside = (
            item.lower_deg - _ANGLE_TOLERANCE_DEG
            <= error_deg
            < item.upper_deg - _ANGLE_TOLERANCE_DEG
        )
        if is_last and math.isclose(
            error_deg,
            item.upper_deg,
            rel_tol=0.0,
            abs_tol=_ANGLE_TOLERANCE_DEG,
        ):
            inside = True
        if inside:
            return item.probability / (item.upper_deg - item.lower_deg)
    return 0.0

def _validate_bearing_error_atoms(
    atoms: Sequence[BearingErrorAtom],
    config: Q2Config,
) -> tuple[BearingErrorAtom, ...]:
    if not atoms:
        raise ValueError(
            "bearing_error_atoms is required because the problem gives no error density"
        )

    prepared = tuple(
        atom if isinstance(atom, BearingErrorAtom) else BearingErrorAtom(*atom)  # type: ignore[arg-type]
        for atom in atoms
    )
    for atom in prepared:
        if abs(atom.offset_deg) > config.bearing_error_deg + _ANGLE_TOLERANCE_DEG:
            raise ValueError("bearing error atom lies outside the hard error bound")

    probability_sum = sum(atom.probability for atom in prepared)
    if not math.isclose(probability_sum, 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("bearing error atom probabilities must sum to one")
    if not any(atom.probability > 0.0 for atom in prepared):
        raise ValueError("bearing error model must contain positive probability")
    return prepared


def _prepare_direction_grid(
    direction_grid_deg: Sequence[float],
) -> tuple[tuple[float, ...], float]:
    if len(direction_grid_deg) < 3:
        raise ValueError("direction_grid_deg must contain at least three centers")
    if not all(_is_finite_number(value) for value in direction_grid_deg):
        raise ValueError("direction_grid_deg centers must be finite")

    centers = tuple(sorted(normalize_angle_deg(float(value)) for value in direction_grid_deg))
    if len(set(centers)) != len(centers):
        raise ValueError("direction_grid_deg centers must be unique modulo 360 degrees")

    expected_gap = 360.0 / len(centers)
    gaps = [
        (centers[(index + 1) % len(centers)] - centers[index]) % 360.0
        for index in range(len(centers))
    ]
    if not all(
        math.isclose(gap, expected_gap, rel_tol=1e-10, abs_tol=1e-10)
        for gap in gaps
    ):
        raise ValueError(
            "direction_grid_deg must form an equally spaced circular partition"
        )
    return centers, expected_gap


def _nearest_direction_center(
    angle_deg: float,
    centers: Sequence[float],
) -> float:
    return min(
        centers,
        key=lambda center: (
            abs(signed_angle_difference_deg(angle_deg, center)),
            center,
        ),
    )


def _direction_bin_outer_region(
    q: Point,
    center_bearing_deg: float,
    bin_width_deg: float,
    state: FirstState,
    config: Q2Config,
) -> tuple[Point, ...]:
    widened_error = config.bearing_error_deg + bin_width_deg / 2.0
    if widened_error >= 90.0:
        raise ValueError(
            "direction grid is too coarse for a bearing-wedge outer certificate"
        )

    region = clip_polygon_to_bearing_wedge(
        state.outer_region,
        q,
        center_bearing_deg,
        widened_error,
    )
    region = clip_polygon_to_circle_outer(
        region,
        q,
        config.reception_radius_max_m,
        config.circle_vertices,
    )
    return tuple(region)


def _is_sample_compatible_with_second_response(
    sample: JointSample,
    q: Point,
    response: SecondResponse,
    config: Q2Config,
) -> bool:
    source_distance = distance(sample.position, q)

    if response.kind is SecondResponseKind.NEAR:
        return source_distance <= config.near_radius_m + _DISTANCE_TOLERANCE_M

    if response.kind is SecondResponseKind.NO_SIGNAL:
        return source_distance > sample.reception_radius_m + _DISTANCE_TOLERANCE_M

    assert response.bearing_deg is not None
    if source_distance <= config.near_radius_m + _DISTANCE_TOLERANCE_M:
        return False
    if source_distance > sample.reception_radius_m + _DISTANCE_TOLERANCE_M:
        return False

    true_bearing = _bearing_deg(q, sample.position)
    return (
        abs(signed_angle_difference_deg(true_bearing, response.bearing_deg))
        <= config.bearing_error_deg + _ANGLE_TOLERANCE_DEG
    )


def _is_sample_compatible_with_first_direction(
    sample: JointSample,
    observation: FirstDirectionObservation,
    config: Q2Config,
) -> bool:
    if not (
        config.reception_radius_min_m
        <= sample.reception_radius_m
        <= config.reception_radius_max_m
    ):
        return False
    if distance(sample.position, (0.0, 0.0)) > (
        config.arena_radius_m + _DISTANCE_TOLERANCE_M
    ):
        return False

    station_distance = distance(sample.position, observation.station)
    if station_distance <= config.near_radius_m + _DISTANCE_TOLERANCE_M:
        return False
    if station_distance > sample.reception_radius_m + _DISTANCE_TOLERANCE_M:
        return False

    true_bearing = _bearing_deg(observation.station, sample.position)
    return (
        abs(signed_angle_difference_deg(true_bearing, observation.bearing_deg))
        <= config.bearing_error_deg + _ANGLE_TOLERANCE_DEG
    )


def _bearing_deg(origin: Point, target: Point) -> float:
    return normalize_angle_deg(
        math.degrees(math.atan2(target[1] - origin[1], target[0] - origin[0]))
    )


def _point_to_polygon_distance(point: Point, polygon: Sequence[Point]) -> float:
    if not polygon:
        raise ValueError("polygon must be non-empty")
    if len(polygon) == 1:
        return distance(point, polygon[0])
    if len(polygon) == 2:
        return _point_segment_distance(point, polygon[0], polygon[1])
    if _point_in_convex_polygon(point, polygon):
        return 0.0
    return min(
        _point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def _point_segment_distance(point: Point, a: Point, b: Point) -> float:
    _validate_point(point, "point")
    _validate_point(a, "segment start")
    _validate_point(b, "segment end")

    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return distance(point, a)
    t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length_squared
    t = min(1.0, max(0.0, t))
    projection = (a[0] + t * dx, a[1] + t * dy)
    return distance(point, projection)


def _point_in_convex_polygon(
    point: Point,
    polygon: Sequence[Point],
    *,
    epsilon: float = 1e-9,
) -> bool:
    _validate_point(point, "point")
    if not polygon:
        raise ValueError("polygon must be non-empty")
    if len(polygon) == 1:
        return distance(point, polygon[0]) <= epsilon
    if len(polygon) == 2:
        return _point_segment_distance(point, polygon[0], polygon[1]) <= epsilon

    has_positive = False
    has_negative = False
    for index, current in enumerate(polygon):
        _validate_point(current, "polygon vertex")
        following = polygon[(index + 1) % len(polygon)]
        _validate_point(following, "polygon vertex")
        cross = _cross(
            (following[0] - current[0], following[1] - current[1]),
            (point[0] - current[0], point[1] - current[1]),
        )
        if cross > epsilon:
            has_positive = True
        elif cross < -epsilon:
            has_negative = True
        if has_positive and has_negative:
            return False
    return True


def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _validate_point(point: Point, name: str) -> None:
    if len(point) != 2 or not all(_is_finite_number(coordinate) for coordinate in point):
        raise ValueError(f"{name} must be a finite 2D point")
