import math
import unittest

from src.common.simulator_client import SimulatorClient


class MixedTests(unittest.TestCase):
    def test_directional_front_back_boundary_and_optical(self):
        from src.q4.offline_simulator import MixedSimulator, MixedSource
        world = MixedSimulator([MixedSource(1, (0, 0), 1000, 0)])
        client = SimulatorClient('test', transport=world.transport)
        client.enter()
        self.assertEqual(client.measure((-1, 0), 1).result, 'no_signal')
        self.assertEqual(client.measure((0, 1000), 1).result, 'direction')
        self.assertEqual(client.measure((1, 0), 1).result, 'near')
        self.assertEqual(client.clear((-19, 0), 1).result, 'success')

    def test_seven_points_not_directional_cover_but_grid_is(self):
        from src.q4.coverage import covers_mixed, fallback_grid
        from src.q3.baseline_scan import coverage_points
        self.assertFalse(covers_mixed(coverage_points()))
        self.assertEqual(len(fallback_grid()), 64)
        self.assertTrue(covers_mixed(fallback_grid()))
        self.assertFalse(covers_mixed([(0, 0)]))

    def test_optical_rectangle_covers_extreme_sector(self):
        from src.q4.controller import optical_points
        points = optical_points((0, 0), 0)
        self.assertEqual(len(points), 108)
        for radius in range(0, 1501, 15):
            for degree in (-1, -.5, 0, .5, 1):
                p = (radius*math.cos(math.radians(degree)), radius*math.sin(math.radians(degree)))
                self.assertLessEqual(min(math.dist(p, q) for q in points), 20)

    def test_back_facing_edge_source_is_found_and_cleared(self):
        from src.q4.offline_simulator import MixedSimulator, MixedSource
        from src.q4.controller import MixedController
        world = MixedSimulator([MixedSource(1, (1800, 0), 1000, 0),
                                MixedSource(2, (500, 0), 1000, None)],
                               bearing_error_deg=1, error_field='spatial')
        client = SimulatorClient('test', transport=world.transport)
        client.enter()
        state = MixedController(client).run()
        self.assertTrue(state.all_cleared, state.failure_detail)
        self.assertEqual(world.cleared, {1, 2})
        self.assertTrue(state.completed_full_cover)

    def test_budget_is_incomplete_not_absence(self):
        from src.q4.offline_simulator import MixedSimulator, MixedSource
        from src.q4.controller import MixedController
        world = MixedSimulator([MixedSource(1, (1800, 0), 1000, 0)])
        client = SimulatorClient('test', transport=world.transport)
        client.enter()
        state = MixedController(client, max_actions=3).run()
        self.assertEqual(state.termination_reason, 'action_budget')
        self.assertFalse(state.all_cleared)
        self.assertEqual(state.excluded_channels, set())

    def test_cli_writes_logs_and_exits(self):
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path
        from unittest.mock import patch
        from src.q4.main import build_parser, run
        from src.q4.offline_simulator import MixedSimulator, MixedSource
        world = MixedSimulator([MixedSource(c, (500, 0), 1000, None) for c in range(1, 17)])
        def factory(robot_id, **kwargs):
            return SimulatorClient(robot_id, transport=world.transport, **kwargs)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            args = build_parser().parse_args(['--robot-id', 'test', '--run-root', folder])
            with patch('src.q4.main.SimulatorClient', side_effect=factory):
                code = run(args)
            summary = json.loads((next(Path(folder).iterdir())/'summary.json').read_text('utf-8'))
        self.assertEqual(code, 0)
        self.assertTrue(world.exited)
        self.assertEqual(summary['cleared_count'], 16)
        self.assertEqual(summary['network_requests'], len(world.actions))

    def test_missing_receipt_cannot_complete_batch(self):
        from src.q4.offline_simulator import MixedSimulator, MixedSource
        from src.q4.controller import MixedController
        world = MixedSimulator([MixedSource(1, (1800, 0), 1000, 0)])
        def transport(url, body, timeout):
            if len(world.actions) >= 4:
                raise TimeoutError('missing receipt')
            return world.transport(url, body, timeout)
        client = SimulatorClient('test', transport=transport, max_network_retries=0)
        client.enter()
        controller = MixedController(client)
        with self.assertRaises(Exception):
            controller.run()
        self.assertFalse(controller.state.all_cleared)
        self.assertFalse(controller.state.full_scan_stations)
        self.assertIsNotNone(client.pending_action)

    def test_joint_model_keeps_back_side_and_strict_radius(self):
        from src.q3.baseline_scan import MeasureObservation
        from src.q4.joint_model import hypotheses
        obs = [MeasureObservation((500,0), 1, 'direction', 180, 0),
               MeasureObservation((-500,0), 1, 'no_signal', None, 0)]
        weight, direction, lower = hypotheses([(0,0)], obs)
        self.assertGreater(weight[0], 0)
        self.assertTrue(direction[0,0])
        # A no_signal exactly at the minimum radius cannot fit an omni source.
        obs = [MeasureObservation((1000,0), 1, 'no_signal', None, 0)]
        weight, direction, lower = hypotheses([(0,0)], obs)
        self.assertFalse(direction[0,0])
        self.assertTrue(direction[0,36])
        self.assertLess(weight[0], .5)

    def test_request_budget_leaves_exit_possible(self):
        from src.q4.offline_simulator import MixedSimulator, MixedSource
        from src.q4.controller import MixedController
        world = MixedSimulator([MixedSource(1, (1800,0), 1000, 0)])
        client = SimulatorClient('test', transport=world.transport)
        client.enter()
        state = MixedController(client, max_requests=3).run()
        self.assertEqual(state.termination_reason, 'request_budget')
        self.assertIsNone(client.pending_action)
        client.exit()
        self.assertEqual(state.network_requests, 3)

    def test_optical_fallback_after_zero_active_steps(self):
        from src.q4.offline_simulator import MixedSimulator, MixedSource
        from src.q4.controller import MixedController
        world = MixedSimulator([MixedSource(c, (700,0), 1000, 180 if c%2 else None)
                                for c in range(1,17)])
        client = SimulatorClient('test', transport=world.transport)
        client.enter()
        state = MixedController(client, active_steps=0).run()
        self.assertTrue(state.all_cleared, state.failure_detail)
        self.assertTrue(state.fallback_channels)
        self.assertEqual(world.cleared, set(range(1,17)))


if __name__ == '__main__':
    unittest.main()
