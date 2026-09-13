import unittest
import numpy as np

from src.common.simulator_client import SimulatorClient
from src.q3.adaptive_control import AdaptiveController
from src.q3.offline_simulator import OfflineSimulator, Source


class AdaptiveTests(unittest.TestCase):
    def controller(self, sources, **kwargs):
        world = OfflineSimulator(sources)
        client = SimulatorClient('offline', transport=world.transport)
        client.enter()
        return world, AdaptiveController(client, **kwargs)

    def test_zero_exploration_still_finishes_entire_task(self):
        world, controller = self.controller([Source(9, (1500, 0), 1500)], max_free_actions=0)
        total_reward = 0
        for _ in range(400):
            candidates = controller.candidates()
            if not candidates:
                break
            self.assertEqual(len(candidates), 1)
            total_reward += controller.step(candidates[0])
        self.assertTrue(controller.state.all_cleared)
        self.assertEqual(world.cleared, {9})
        self.assertAlmostEqual(total_reward, -world.virtual_time_s / controller.reward_scale)

    def test_features_do_not_depend_on_hidden_scene(self):
        _, a = self.controller([Source(9, (1500, 0), 1500)])
        _, b = self.controller([Source(1, (-1500, 0), 1000), Source(3, (1000, 1000), 1000)])
        fa, fb = a.features(a.candidates()), b.features(b.candidates())
        for key in fa:
            np.testing.assert_array_equal(fa[key], fb[key])

    def test_discovery_adds_active_measurement_and_other_channel_actions(self):
        _, controller = self.controller([Source(1, (900, 0), 1000)])
        action = next(a for a in controller.candidates() if a.position == (0, 0) and a.channel == 1)
        controller.step(action)
        candidates = controller.candidates()
        self.assertTrue(any(a.channel == 1 and a.source == 'q2' for a in candidates))
        self.assertTrue(any(a.channel != 1 and a.source == 'coverage' for a in candidates))

    def test_global_quota_locks_fallback_without_ending_episode(self):
        _, controller = self.controller([Source(1, (900, 0), 1000)], max_free_actions=1)
        controller.step(controller.candidates()[0])
        self.assertTrue(controller.fallback_mode)
        self.assertFalse(controller.done)
        self.assertEqual(len(controller.candidates()), 1)

    def test_short_real_time_switches_to_fallback_before_planning(self):
        import time
        _, controller = self.controller([])
        controller.client.state.real_deadline_monotonic = time.monotonic()+90
        self.assertEqual(len(controller.candidates()), 1)
        self.assertTrue(controller.fallback_mode)

    def test_candidate_snapshot_cannot_execute_unsupplied_action(self):
        from src.q3.adaptive_control import Candidate
        _, controller = self.controller([])
        controller.candidates()
        with self.assertRaises(ValueError):
            controller.step(Candidate('clear', 1, (0, 0), 'invented'))


if __name__ == '__main__':
    unittest.main()
