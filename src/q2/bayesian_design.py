"""Bayesian second-station design for CUMCM 2026 problem B."""

from __future__ import annotations

import math

import numpy as np


def polygon_area(vertices: np.ndarray) -> float:
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_diameter(vertices: np.ndarray) -> float:
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) < 2:
        return 0.0
    differences = vertices[:, None, :] - vertices[None, :, :]
    squared = np.einsum("ijk,ijk->ij", differences, differences)
    return float(math.sqrt(float(np.max(squared))))


def max_vertex_distance(point: np.ndarray, vertices: np.ndarray) -> float:
    point = np.asarray(point, dtype=float)
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) == 0:
        return 0.0
    return float(np.max(np.linalg.norm(vertices - point, axis=1)))


def is_guaranteed_reception(
    point: np.ndarray,
    vertices: np.ndarray,
    minimum_radius: float = 1000.0,
) -> bool:
    return max_vertex_distance(point, vertices) <= minimum_radius + 1e-9


def sector_polygon(
    origin: np.ndarray,
    bearing: float,
    delta: float,
    radius: float,
    arc_points: int = 65,
) -> np.ndarray:
    if arc_points < 2:
        raise ValueError("arc_points must be at least 2")
    origin = np.asarray(origin, dtype=float)
    angles = np.linspace(bearing - delta, bearing + delta, arc_points)
    arc = origin + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    return np.vstack((origin, arc))


def _clip_halfplane(
    polygon: np.ndarray,
    normal: np.ndarray,
    bound: float,
    tolerance: float = 1e-10,
) -> np.ndarray:
    polygon = np.asarray(polygon, dtype=float)
    if len(polygon) == 0:
        return polygon.reshape(0, 2)
    normal = np.asarray(normal, dtype=float)
    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = float(np.dot(normal, previous) - bound)
    previous_inside = previous_value <= tolerance
    for current in polygon:
        current_value = float(np.dot(normal, current) - bound)
        current_inside = current_value <= tolerance
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > 1e-15:
                fraction = previous_value / denominator
                output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    if not output:
        return np.empty((0, 2), dtype=float)
    cleaned = [output[0]]
    for point in output[1:]:
        if np.linalg.norm(point - cleaned[-1]) > 1e-9:
            cleaned.append(point)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= 1e-9:
        cleaned.pop()
    return np.asarray(cleaned, dtype=float)


def clip_bearing_wedge(
    polygon: np.ndarray,
    station: np.ndarray,
    observed_bearing: float,
    delta: float,
) -> np.ndarray:
    station = np.asarray(station, dtype=float)
    lower = np.array(
        [math.cos(observed_bearing - delta), math.sin(observed_bearing - delta)]
    )
    upper = np.array(
        [math.cos(observed_bearing + delta), math.sin(observed_bearing + delta)]
    )
    normal_lower = np.array([lower[1], -lower[0]])
    normal_upper = np.array([-upper[1], upper[0]])
    clipped = _clip_halfplane(polygon, normal_lower, float(np.dot(normal_lower, station)))
    return _clip_halfplane(
        clipped,
        normal_upper,
        float(np.dot(normal_upper, station)),
    )


def receive_probability(
    distances: np.ndarray,
    radius_min: float = 1000.0,
    radius_max: float = 1500.0,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=float)
    if radius_max <= radius_min:
        raise ValueError("radius_max must exceed radius_min")
    return np.clip((radius_max - distances) / (radius_max - radius_min), 0.0, 1.0)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def direction_likelihood(
    points: np.ndarray,
    station: np.ndarray,
    observed_bearing: float,
    delta: float,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    station = np.asarray(station, dtype=float)
    offsets = points - station
    distances = np.linalg.norm(offsets, axis=1)
    bearings = np.arctan2(offsets[:, 1], offsets[:, 0])
    angular_match = np.abs(_wrap_angle(observed_bearing - bearings)) <= delta + 1e-12
    likelihood = receive_probability(distances) * angular_match.astype(float)
    likelihood[distances <= 5.0] = 0.0
    return likelihood / (2.0 * delta)


def sample_uniform_convex_polygon(polygon, sample_count, rng):
    polygon = np.asarray(polygon, dtype=float)
    if len(polygon) < 3 or polygon_area(polygon) <= 0.0:
        raise ValueError("polygon must have positive area")
    anchor = polygon[0]
    left = polygon[1:-1]
    right = polygon[2:]
    cross = (
        (left[:, 0] - anchor[0]) * (right[:, 1] - anchor[1])
        - (left[:, 1] - anchor[1]) * (right[:, 0] - anchor[0])
    )
    triangle_areas = np.abs(cross) / 2.0
    probabilities = triangle_areas / triangle_areas.sum()
    triangle_indices = rng.choice(len(left), size=sample_count, p=probabilities)
    root_u = np.sqrt(rng.random(sample_count))
    v = rng.random(sample_count)
    selected_left = left[triangle_indices]
    selected_right = right[triangle_indices]
    return (
        (1.0 - root_u)[:, None] * anchor
        + (root_u * (1.0 - v))[:, None] * selected_left
        + (root_u * v)[:, None] * selected_right
    )


def weighted_quantile(values, weights, probabilities):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(values) == 0 or len(values) != len(weights):
        raise ValueError("values and weights must be nonempty and have equal length")
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("weights must be nonnegative with positive sum")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    positions = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / sorted_weights.sum()
    return np.interp(
        probabilities,
        positions,
        sorted_values,
        left=sorted_values[0],
        right=sorted_values[-1],
    )


def outside_disk_support_diameter(polygon, station, radius):
    polygon = np.asarray(polygon, dtype=float)
    station = np.asarray(station, dtype=float)
    points: list[np.ndarray] = []
    distances = np.linalg.norm(polygon - station, axis=1)
    for vertex, distance in zip(polygon, distances):
        if distance >= radius - 1e-10:
            points.append(vertex)
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        direction = end - start
        offset = start - station
        coefficients = (
            float(np.dot(direction, direction)),
            float(2.0 * np.dot(offset, direction)),
            float(np.dot(offset, offset) - radius * radius),
        )
        discriminant = coefficients[1] ** 2 - 4.0 * coefficients[0] * coefficients[2]
        if coefficients[0] <= 0.0 or discriminant < -1e-9:
            continue
        root = math.sqrt(max(0.0, discriminant))
        for numerator in (-coefficients[1] - root, -coefficients[1] + root):
            fraction = numerator / (2.0 * coefficients[0])
            if -1e-12 <= fraction <= 1.0 + 1e-12:
                points.append(start + np.clip(fraction, 0.0, 1.0) * direction)
    if not points:
        return 0.0
    return polygon_diameter(np.asarray(points))


def expected_posterior_diameter(prior, station, delta, sources, errors):
    prior = np.asarray(prior, dtype=float)
    station = np.asarray(station, dtype=float)
    sources = np.asarray(sources, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if len(sources) != len(errors) or len(sources) == 0:
        raise ValueError("sources and errors must have the same positive length")
    diameters = []
    prior_diameter = polygon_diameter(prior)
    for source, error in zip(sources, errors):
        offset = source - station
        distance = float(np.linalg.norm(offset))
        if distance <= 5.0:
            diameters.append(min(10.0, prior_diameter))
            continue
        observed = math.atan2(offset[1], offset[0]) + float(error)
        posterior = clip_bearing_wedge(prior, station, observed, delta)
        diameters.append(polygon_diameter(posterior))
    return float(np.mean(diameters))
