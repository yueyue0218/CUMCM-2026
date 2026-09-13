import math
import unittest

from src.common.simulator_client import SimulatorClient
from src.q3.offline_simulator import OfflineSimulator, Source


class StrategyControlTests(unittest.TestCase):
    def test_comparison_names_macro_weights_and_failed_case_retention(self):
        import io, json, tempfile
        from contextlib import redirect_stdout
        from pathlib import Path
        from unittest.mock import patch
        from src.q3.evaluate_strategy_controls import STRATEGIES, evaluate, aggregate_controls
        self.assertEqual(len(STRATEGIES), 6)
        self.assertTrue(all(s['stem'].startswith(('实验组_', '对照组_')) for s in STRATEGIES.values()))
        rows = [{'mode': 'oracle', 'source_count': n, 'all_cleared': True, 'cleared_count': n,
                 'clearance_ratio': 1., 'per_source_s': t, 'total_time_s': n*t, 'program_runtime_s': 1.}
                for n,t in ((10, 300), (12, 250), (14, 220), (16, 200))]
        result = aggregate_controls(rows, ['oracle'])['oracle']
        self.assertEqual(result['equal_weight_mean_per_source_s'], 242.5)
        self.assertEqual(result['equal_weight_mean_total_time_s'], 3070.)
        rows[0].update(all_cleared=False, cleared_count=5, clearance_ratio=.5)
        result = aggregate_controls(rows, ['oracle'])['oracle']
        self.assertEqual(result['equal_weight_clearance_ratio'], .875)
        self.assertIsNone(result['equal_weight_mean_per_source_s'])
        self.assertEqual(result['equal_weight_observed_per_source_s'], 242.5)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            with patch('src.q3.evaluate_strategy_controls.run_strategy', side_effect=ArithmeticError('injected')):
                report = evaluate(cases_per_group=1, modes=['oracle'], output_dir=Path(tmp)/'tables', run_root=Path(tmp)/'runs')
            self.assertEqual(len(report['cases']), 4)
            self.assertIsNone(report['aggregate']['oracle']['equal_weight_mean_per_source_s'])
            traces = list(Path(report['run_directory']).glob('对照组_上帝视角/*_逐场轨迹.json'))
            self.assertEqual(len(traces), 4)
            self.assertTrue(all('injected' in json.loads(p.read_text('utf-8'))['failure'] for p in traces))
            self.assertTrue((Path(tmp)/'tables/Q3_实验组与对照组_策略比较表.md').is_file())

    def test_census_finishes_seven_stations_before_first_clear(self):
        from src.q3.control_census_then_clear import run_census_then_clear
        from src.q3.experiment_joint_routing import compact_coverage_points
        world = OfflineSimulator([Source(1, (600, 100), 1100)])
        client = SimulatorClient('test', transport=world.transport); client.enter()
        state = run_census_then_clear(client)
        first_clear = next(i for i, a in enumerate(world.actions) if a['path'] == '/clear')
        measured = {tuple(a['request']['position'].values()) for a in world.actions[:first_clear] if a['path'] == '/measure'}
        self.assertTrue(set(compact_coverage_points()).issubset(measured))
        self.assertTrue(state.all_cleared)
        self.assertEqual(world.cleared, {1})

    def test_census_does_not_skip_stations_after_all_sixteen_sources_are_located(self):
        from src.q3.control_census_then_clear import run_census_then_clear
        world = OfflineSimulator([Source(c, (600, 100), 1100) for c in range(1,17)])
        client = SimulatorClient('test', transport=world.transport); client.enter()
        state = run_census_then_clear(client)
        first_clear = next(i for i,a in enumerate(world.actions) if a['path'] == '/clear')
        measurements = [a for a in world.actions[:first_clear] if a['path'] == '/measure']
        self.assertEqual(len(measurements), 140)
        measured_stations = {tuple(a['request']['position'].values()) for a in measurements}
        self.assertEqual(measured_stations, set(state.full_scan_stations))
        self.assertEqual(len(measured_stations), 7)
        self.assertTrue(state.all_cleared)

    def test_spiral_and_random_routes_are_bounded_and_reproducible(self):
        from src.q3.control_spiral_scan import spiral_points
        from src.q3.control_random_walk import random_walk_points
        from src.q3.coverage_control import covers_arena
        points = spiral_points()
        self.assertEqual(points[0], (0., 0.))
        self.assertTrue(all(math.hypot(*p) <= 1800.000001 for p in points))
        self.assertTrue(covers_arena(points))
        walk = random_walk_points(12)
        self.assertEqual(walk, random_walk_points(12))
        self.assertNotEqual(walk, random_walk_points(13))
        self.assertEqual(len(walk), 129)
        self.assertTrue(all(math.hypot(*p) <= 1800.000001 for p in walk))
        self.assertTrue(all(abs(math.dist(a, b)-500) < 1e-6 for a,b in zip(walk, walk[1:])))

    def test_search_controls_clear_boundary_sources_with_measurement_error(self):
        from src.q3.control_spiral_scan import run_spiral_scan
        from src.q3.control_random_walk import run_random_walk
        for run in (run_spiral_scan, run_random_walk):
            world = OfflineSimulator([Source(1, (1800, 0), 1000), Source(20, (-1800, 0), 1000)],
                                     bearing_error_deg=1, error_field='spatial', quantize_bearings=True)
            client = SimulatorClient('test', transport=world.transport); client.enter()
            state = run(client)
            self.assertTrue(state.all_cleared, state.failure_detail)
            self.assertEqual(world.cleared, {1, 20})

    def test_random_scan_limit_is_not_reported_as_full_coverage(self):
        from src.q3.control_random_walk import run_random_walk
        world = OfflineSimulator([Source(1, (1800, 0), 1000)])
        client = SimulatorClient('test', transport=world.transport); client.enter()
        state = run_random_walk(client, max_steps=0)
        self.assertFalse(state.all_cleared)
        self.assertEqual(state.termination_reason, 'scan_limit')

    def test_oracle_uses_no_measurements_and_labels_its_privileged_evidence(self):
        from src.q3.control_oracle import run_oracle, build_oracle_summary
        positions = {1: (100, 0), 3: (300, 0)}
        world = OfflineSimulator([Source(c, p, 1000) for c,p in positions.items()])
        client = SimulatorClient('test', transport=world.transport); client.enter()
        state = run_oracle(client, positions=positions)
        self.assertTrue(state.all_cleared)
        self.assertFalse(any(a['path'] == '/measure' for a in world.actions))
        self.assertAlmostEqual(client.state.virtual_time_s, 70.)
        summary = build_oracle_summary(state, client.state.virtual_time_s)
        self.assertEqual(summary['total_count_basis'], 'oracle_known_source_list_and_clear_receipts')
        self.assertEqual(summary['total_count'], 2)

    def test_budget_stops_search_and_oracle_without_false_success(self):
        from src.q3.control_census_then_clear import run_census_then_clear
        from src.q3.control_oracle import run_oracle
        for run, kwargs in ((run_census_then_clear, {}), (run_oracle, {'positions': {1: (1800, 0)}})):
            world = OfflineSimulator([Source(1, (1800, 0), 1000)])
            client = SimulatorClient('test', transport=world.transport); client.enter()
            state = run(client, virtual_limit_s=50, **kwargs)
            self.assertFalse(state.all_cleared)
            self.assertEqual(state.termination_reason, 'virtual_time_budget')


if __name__ == '__main__':
    unittest.main()
