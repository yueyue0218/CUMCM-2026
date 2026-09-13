"""History-only nominal particle predictions and open clearance routes.

The conditional prior is uniform source area and uniform fixed radius in
[1000, 1500] m, *conditional on an already observed source*. There is no
independent occupancy prior for unknown channels. Final bearing errors are
treated as uniform within one degree, independently at distinct stations;
this continuous-density approximation is not calibrated for quantized readings
or spatially correlated errors. Repeated stations replay their fixed reading.
All outputs rank actions only: neither particles nor hypothetical observations
can certify a real clear or absence decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.common.geometry import Point, circle_polygon, clip_polygon_to_circle_outer
from src.q2 import bayesian_design as q2
from src.q3.baseline_scan import ChannelDiscovery, MeasureObservation
from src.q3.localization_control import assess_channel_localization, evaluate_channel


@dataclass(frozen=True)
class ChannelBelief:
    points: tuple[Point, ...]
    radii_m: tuple[float, ...]
    weights: tuple[float, ...]
    outer_region: tuple[Point, ...]
    diameter_m: float
    calibrated: bool = False


@dataclass(frozen=True)
class MeasurementPrediction:
    available: bool
    p_no_signal: float | None = None
    p_near: float | None = None
    p_direction: float | None = None
    expected_diameter_m: float | None = None
    calibrated: bool = False
    certified: bool = False
    reason: str = "nominal continuous independent-error approximation; ranking only"


def _history(discovery: ChannelDiscovery) -> tuple[MeasureObservation, ...]:
    # Copy coordinates into immutable keys; no mutable discovery object is cached.
    return tuple(MeasureObservation(tuple(o.position), o.channel, o.measure_result,
                                    o.svd_deg, o.virtual_time_s)
                 for o in discovery.observations)


@lru_cache(maxsize=512)
def _outer(history: tuple[MeasureObservation, ...]) -> tuple[tuple[Point, ...], float]:
    if any(o.measure_result == "direction" for o in history):
        assessment = assess_channel_localization(ChannelDiscovery("detected", list(history)))
        region = list(assessment.outer_region)
    else:
        region = circle_polygon(1800.0, 720)
    for o in history:
        if o.measure_result == "near":
            region = clip_polygon_to_circle_outer(region, o.position, 5.0, 64)
    return tuple(region), q2.polygon_diameter(np.asarray(region)) if region else math.inf


@lru_cache(maxsize=512)
def _build_cached(history: tuple[MeasureObservation, ...], particle_count: int,
                  seed: int) -> ChannelBelief | None:
    if not any(o.measure_result in {"direction", "near"} for o in history):
        return None
    try:
        region, diameter = _outer(history)
    except ArithmeticError:
        return None
    if len(region) < 3 or not math.isfinite(diameter) or q2.polygon_area(np.asarray(region)) <= 0:
        return None
    rng = np.random.default_rng(seed)
    # Bounded proposal effort; particle starvation is explicitly unavailable,
    # never an absence certificate. Sampling the full Q1 intersection avoids
    # the catastrophic rejection rate of sampling the entire arena.
    points = q2.sample_uniform_convex_polygon(np.asarray(region), max(2048, 32 * particle_count), rng)
    valid = np.linalg.norm(points, axis=1) <= 1800.0
    lower = np.full(len(points), 1000.0)
    upper = np.full(len(points), 1500.0)
    for o in history:
        offsets = points - np.asarray(o.position)
        distance = np.linalg.norm(offsets, axis=1)
        if o.measure_result == "direction":
            if o.svd_deg is None or not math.isfinite(o.svd_deg):
                raise ValueError("direction observation requires a finite bearing")
            angle_error = (np.arctan2(offsets[:, 1], offsets[:, 0]) - math.radians(o.svd_deg) + math.pi) % (2 * math.pi) - math.pi
            valid &= (distance > 5.0) & (np.abs(angle_error) <= math.radians(1.0) + 1e-12)
            lower = np.maximum(lower, distance)
        elif o.measure_result == "near":
            valid &= distance <= 5.0
        elif o.measure_result == "no_signal":
            upper = np.minimum(upper, distance)
        else:
            raise ValueError(f"unsupported measurement result: {o.measure_result}")
    # A SINGLE fixed radius is shared by every observation. Its integrated
    # likelihood is the prior mass of [max received distance, min failed
    # distance), not a product of independent per-observation receive chances.
    mass = np.maximum(0.0, q2.receive_probability(lower) - q2.receive_probability(upper))
    mass[~valid] = 0.0
    total = float(mass.sum())
    if total <= 0 or not math.isfinite(total):
        return None
    selected = rng.choice(len(points), particle_count, p=mass / total)
    radii = lower[selected] + rng.random(particle_count) * (upper[selected] - lower[selected])
    return ChannelBelief(tuple(map(tuple, points[selected].tolist())),
                         tuple(radii.tolist()), (1.0 / particle_count,) * particle_count,
                         region, diameter)


def build_belief(discovery: ChannelDiscovery, particle_count: int = 64,
                 seed: int = 0) -> ChannelBelief | None:
    """Sample a joint location/radius posterior from all saved real evidence.

    Negative evidence is retained even though the conservative Q1 polygon
    ignores its nonconvex exclusion. ``None`` means no model or no surviving
    particles, not evidence that a source does not exist.
    """
    if isinstance(particle_count, bool) or not isinstance(particle_count, int) or particle_count <= 0:
        raise ValueError("particle_count must be a positive integer")
    if discovery.status in {"cleared", "excluded_after_full_cover", "model_conflict"}:
        return None
    return _build_cached(_history(discovery), particle_count, seed)


@lru_cache(maxsize=2048)
def _predict_cached(history: tuple[MeasureObservation, ...], position: Point,
                    particle_count: int, seed: int) -> MeasurementPrediction:
    belief = _build_cached(history, particle_count, seed)
    if belief is None:
        return MeasurementPrediction(False, reason="no conditional model or particle support")
    previous = next((o for o in reversed(history) if o.position == position), None)
    if previous is not None:
        # The actual sensor has fixed error at a fixed source/station pair.
        return MeasurementPrediction(True, float(previous.measure_result == "no_signal"),
                                     float(previous.measure_result == "near"),
                                     float(previous.measure_result == "direction"), belief.diameter_m)
    points = np.asarray(belief.points)
    offsets = points - np.asarray(position)
    distances = np.linalg.norm(offsets, axis=1)
    no_signal = distances > np.asarray(belief.radii_m)
    near = (distances <= 5.0) & ~no_signal
    direction = ~no_signal & ~near
    weights = np.asarray(belief.weights)
    p_no, p_near, p_dir = (float(weights[mask].sum()) for mask in (no_signal, near, direction))
    # Q1's current direction-only outer region does not shrink on no_signal.
    # A near reply gives the existing adapter's 5 m disk certificate, used here
    # only as a hypothetical diameter bound, never returned as real clearance.
    expected = p_no * belief.diameter_m + p_near * min(10.0, belief.diameter_m)
    if p_dir > 0:
        bearings = np.arctan2(offsets[direction, 1], offsets[direction, 0])
        center = math.atan2(float(np.sin(bearings).mean()), float(np.cos(bearings).mean()))
        unwrapped = center + (bearings - center + math.pi) % (2 * math.pi) - math.pi
        # Three point quadrature scenarios approximate the predictive integral;
        # these are NOT bins whose full probability is falsely conditioned on a
        # midpoint. The approximation is deliberately exposed as uncalibrated.
        representative = q2.weighted_quantile(unwrapped, weights[direction], [1 / 6, 1 / 2, 5 / 6])
        branch_diameters = []
        for angle, error in zip(representative, (-2 / 3, 0.0, 2 / 3)):
            reading = math.degrees(angle) + error
            branch = ChannelDiscovery("detected", list(history))
            branch.observations.append(MeasureObservation(position, history[-1].channel,
                                                          "direction", reading % 360.0,
                                                          history[-1].virtual_time_s))
            try:
                evaluation = evaluate_channel(branch)
                assessment = evaluation.assessment
                diameter = min(10.0, belief.diameter_m) if assessment is None else assessment.diameter_m
            except ArithmeticError:
                diameter = belief.diameter_m
            # A numerical/conflicting hypothetical branch cannot invent gain.
            branch_diameters.append(min(diameter, belief.diameter_m) if math.isfinite(diameter) else belief.diameter_m)
        expected += p_dir * float(np.mean(branch_diameters))
    return MeasurementPrediction(True, p_no, p_near, p_dir, expected)


def predict_measurement(discovery: ChannelDiscovery, position: Point,
                        particle_count: int = 64, seed: int = 0) -> MeasurementPrediction:
    """Return nominal response probabilities and three-scenario Q1 size estimate."""
    position = tuple(float(v) for v in position)
    if len(position) != 2 or not all(math.isfinite(v) for v in position):
        raise ValueError("position must contain two finite coordinates")
    if isinstance(particle_count, bool) or not isinstance(particle_count, int) or particle_count <= 0:
        raise ValueError("particle_count must be a positive integer")
    if discovery.status in {"cleared", "excluded_after_full_cover", "model_conflict"}:
        return MeasurementPrediction(False, reason="channel no longer eligible for prediction")
    return _predict_cached(_history(discovery), position, particle_count, seed)


def open_clear_route(start: Point, positions: dict[int, Point]) -> list[int]:
    """Shortest open fixed-point path for <=10 targets, bounded 2-opt otherwise.

    No home/return leg is included. Points must already be certified by the
    caller; this function supplies an order, not a position certificate.
    """
    channels = sorted(positions)
    if len(start) != 2 or not all(math.isfinite(v) for v in start):
        raise ValueError("start must contain two finite coordinates")
    if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in positions.values()):
        raise ValueError("route positions must contain two finite coordinates")
    n = len(channels)
    if n == 0:
        return []
    points = [positions[c] for c in channels]
    distance = [[math.dist(a, b) for b in points] for a in points]
    if n <= 10:
        # Held-Karp state stores the full path for deterministic tie resolution.
        dp = {(1 << j, j): (math.dist(start, points[j]), (j,)) for j in range(n)}
        for mask in range(1, 1 << n):
            for last in range(n):
                state = dp.get((mask, last))
                if state is None:
                    continue
                cost, path = state
                for nxt in range(n):
                    if mask & (1 << nxt):
                        continue
                    key = (mask | (1 << nxt), nxt)
                    candidate = cost + distance[last][nxt], path + (nxt,)
                    if key not in dp or candidate < dp[key]:
                        dp[key] = candidate
        _, route = min(dp[((1 << n) - 1, last)] for last in range(n))
        return [channels[j] for j in route]
    remaining, route, current = set(range(n)), [], start
    while remaining:
        nxt = min(remaining, key=lambda j: (math.dist(current, points[j]), channels[j]))
        route.append(nxt)
        remaining.remove(nxt)
        current = points[nxt]
    # Fixed pass cap keeps ranking cost bounded. Reversing a suffix changes
    # only its entering edge because this is an open, symmetric path.
    for _ in range(4):
        improved = False
        for i in range(n - 1):
            before = start if i == 0 else points[route[i - 1]]
            for j in range(i + 1, n):
                old = math.dist(before, points[route[i]])
                new = math.dist(before, points[route[j]])
                if j + 1 < n:
                    old += distance[route[j]][route[j + 1]]
                    new += distance[route[i]][route[j + 1]]
                if new < old - 1e-10:
                    route[i:j + 1] = reversed(route[i:j + 1])
                    improved = True
        if not improved:
            break
    return [channels[j] for j in route]
