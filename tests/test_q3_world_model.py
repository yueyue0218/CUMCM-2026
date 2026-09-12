import copy
import itertools
import math
import unittest
from unittest.mock import patch

import numpy as np

from src.q2 import bayesian_design as q2
from src.q3.baseline_scan import ChannelDiscovery, MeasureObservation
from src.q3 import world_model


def observation(result, position=(0.0, 0.0), bearing=None):
    return MeasureObservation(position, 3, result, bearing, 5.0)


def discovery(*observations):
    return ChannelDiscovery("detected", list(observations))


class BeliefTests(unittest.TestCase):
    def test_unknown_channel_has_no_fabricated_occupancy_probability(self):
        history = discovery(observation("no_signal"))
        self.assertIsNone(world_model.build_belief(history))
        result = world_model.predict_measurement(history, (1000.0, 0.0))
        self.assertFalse(result.available)
        self.assertIsNone(result.p_direction)

    def test_particles_jointly_satisfy_all_received_and_negative_evidence(self):
        history = discovery(observation("no_signal", (-1000.0, 0.0)),
                            observation("direction", bearing=0.0))
        belief = world_model.build_belief(history, 256, seed=14)
        self.assertIsNotNone(belief)
        for point, radius in zip(belief.points, belief.radii_m):
            self.assertGreater(math.dist(point, (-1000.0, 0.0)), radius)
            self.assertLessEqual(math.dist(point, (0.0, 0.0)), radius)
            self.assertGreater(math.dist(point, (0.0, 0.0)), 5.0)
            self.assertLessEqual(abs(math.degrees(math.atan2(point[1], point[0]))), 1.0)
            self.assertTrue(1000.0 <= radius <= 1500.0)
        self.assertAlmostEqual(sum(belief.weights), 1.0)

    def test_new_negative_evidence_changes_cached_posterior(self):
        history = discovery(observation("direction", bearing=0.0))
        before = world_model.build_belief(history, 256, seed=4)
        history.observations.append(observation("no_signal", (1800.0, 0.0)))
        after = world_model.build_belief(history, 256, seed=4)
        self.assertIsNotNone(after)
        self.assertLess(np.mean(np.asarray(after.points)[:, 0]),
                        np.mean(np.asarray(before.points)[:, 0]) - 100.0)
        self.assertTrue(all(math.dist(p, (1800.0, 0.0)) > r
                            for p, r in zip(after.points, after.radii_m)))

    def test_actual_q2_helpers_supply_sampling_and_radius_likelihood(self):
        history = discovery(observation("direction", bearing=24.0))
        with patch.object(q2, "sample_uniform_convex_polygon", wraps=q2.sample_uniform_convex_polygon) as sample:
            with patch.object(q2, "receive_probability", wraps=q2.receive_probability) as likelihood:
                belief = world_model.build_belief(history, 91, seed=75)
        self.assertIsNotNone(belief)
        self.assertGreater(sample.call_count, 0)
        self.assertGreater(likelihood.call_count, 0)

    def test_prediction_probabilities_and_branch_history_are_isolated(self):
        history = discovery(observation("no_signal", (-1000.0, 0.0)),
                            observation("direction", bearing=0.0))
        saved = copy.deepcopy(history)
        with patch.object(world_model, "evaluate_channel", wraps=world_model.evaluate_channel) as assess:
            prediction = world_model.predict_measurement(history, (600.0, 100.0), seed=12)
        self.assertGreater(assess.call_count, 0)
        self.assertLessEqual(assess.call_count, 3)
        for call in assess.call_args_list:
            branch = call.args[0]
            self.assertIsNot(branch.observations, history.observations)
            self.assertEqual(branch.observations[:-1], saved.observations)
        self.assertTrue(prediction.available)
        self.assertAlmostEqual(prediction.p_no_signal + prediction.p_near + prediction.p_direction, 1.0)
        self.assertTrue(math.isfinite(prediction.expected_diameter_m))
        self.assertFalse(prediction.calibrated)
        self.assertFalse(prediction.certified)
        self.assertEqual(saved, history)
        self.assertEqual(prediction, world_model.predict_measurement(history, (600.0, 100.0), seed=12))

    def test_repeat_measurement_replays_fixed_reading_without_information_gain(self):
        history = discovery(observation("direction", bearing=0.0))
        belief = world_model.build_belief(history)
        prediction = world_model.predict_measurement(history, (0.0, 0.0))
        self.assertEqual(prediction.p_direction, 1.0)
        self.assertAlmostEqual(prediction.expected_diameter_m, belief.diameter_m)

    def test_inconsistent_evidence_is_unavailable_and_near_particles_stay_near(self):
        inconsistent = discovery(observation("direction", bearing=0.0), observation("no_signal"))
        self.assertIsNone(world_model.build_belief(inconsistent))
        near = world_model.build_belief(discovery(observation("near", (100.0, 100.0))))
        self.assertIsNotNone(near)
        self.assertTrue(all(math.dist(point, (100.0, 100.0)) <= 5.0 for point in near.points))


def route_length(start, route, positions):
    points = [start] + [positions[channel] for channel in route]
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


class OpenRouteTests(unittest.TestCase):
    def test_exact_route_matches_brute_force_without_home_leg(self):
        start = (1.0, 2.0)
        positions = {7: (20.0, 0.0), 2: (3.0, 8.0), 9: (-4.0, 7.0), 1: (5.0, -4.0), 3: (9.0, 6.0)}
        route = world_model.open_clear_route(start, positions)
        expected = min(route_length(start, p, positions) for p in itertools.permutations(positions))
        self.assertAlmostEqual(route_length(start, route, positions), expected)
        self.assertEqual(set(route), set(positions))
        self.assertEqual(world_model.open_clear_route((0, 0), {1: (1, 0), 2: (10, 0)}), [1, 2])

    def test_large_route_is_deterministic_complete_and_no_worse_than_nearest(self):
        positions = {i: (math.sin(i) * 100, math.cos(i * 1.7) * 100) for i in range(1, 21)}
        route = world_model.open_clear_route((0.0, 0.0), positions)
        remaining, nearest, position = set(positions), [], (0.0, 0.0)
        while remaining:
            channel = min(remaining, key=lambda j: (math.dist(position, positions[j]), j))
            nearest.append(channel)
            remaining.remove(channel)
            position = positions[channel]
        self.assertEqual(sorted(route), sorted(positions))
        self.assertEqual(route, world_model.open_clear_route((0.0, 0.0), dict(reversed(list(positions.items())))))
        self.assertLessEqual(route_length((0.0, 0.0), route, positions), route_length((0.0, 0.0), nearest, positions) + 1e-8)
        self.assertEqual(world_model.open_clear_route((0.0, 0.0), {}), [])


if __name__ == "__main__":
    unittest.main()
