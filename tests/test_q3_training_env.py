import unittest
import math
from src.q3.training_env import TrainingEnvironment, sample_sources


class TrainingEnvironmentTests(unittest.TestCase):
    def test_full_episode_reward_includes_forced_tail(self):
        env = TrainingEnvironment(seed=7, source_count=2, max_free_actions=1, prediction_limit=0)
        features = env.reset()
        reward = 0
        for _ in range(400):
            features, r, done, info = env.step(0)
            reward += r
            if done:
                break
        self.assertTrue(info['all_cleared'])
        self.assertGreater(info['forced_steps'], 0)
        self.assertAlmostEqual(reward, -info['final_virtual_time_s']/1000)

    def test_same_scene_seed_reproducible_and_default_count_in_training_range(self):
        self.assertEqual(sample_sources(12), sample_sources(12))
        for seed in range(10):
            self.assertTrue(10 <= len(sample_sources(seed)) <= 16)

    def test_quantized_bearings_remain_in_final_error_bound(self):
        from src.q3.offline_simulator import OfflineSimulator, Source
        from src.common.simulator_client import SimulatorClient
        for angle in (0.001, 359.999, 90.0049, 180.0099):
            source = Source(1, (500*math.cos(math.radians(angle)),500*math.sin(math.radians(angle))),1000)
            world = OfflineSimulator([source], bearing_error_deg=1, quantize_bearings=True)
            client = SimulatorClient('offline', transport=world.transport)
            client.enter()
            a = client.measure((0,0),1).svd_deg
            b = client.measure((0,0),1).svd_deg
            self.assertEqual(a,b)
            self.assertAlmostEqual(a*100,round(a*100))
            self.assertLessEqual(abs((a-angle+180)%360-180),1.0+1e-12)


if __name__ == '__main__':
    unittest.main()
