import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from src.common.simulator_client import SimulatorClient
from src.q4.benchmark import evaluate_benchmark, make_controller
from src.q4.control_census_then_clear import CensusThenClearController
from src.q4.control_oracle import OracleController
from src.q4.control_random_walk import RandomWalkController, random_walk_points
from src.q4.control_spiral_scan import spiral_points
from src.q4.coverage import covers_mixed
from src.q4.offline_simulator import MixedSimulator, MixedSource
from src.q4.replay import replay


class ScanControlTests(unittest.TestCase):
    def run_control(self, cls, sources, **kwargs):
        world = MixedSimulator(sources)
        client = SimulatorClient('test-q4-controls', transport=world.transport)
        control = cls(client, **kwargs)
        client.enter()
        control.run()
        client.exit()
        return control.state, world

    def test_census_scans_all_21_times_20_before_first_clear(self):
        sources = [MixedSource(1, (1800., 0.), 1000., 0.),
                   MixedSource(2, (0., 0.), 1000., None)]
        state, world = self.run_control(CensusThenClearController, sources)
        self.assertTrue(state.all_cleared)
        self.assertEqual(world.cleared, {1, 2})
        first = next(i for i, a in enumerate(world.actions) if a['path'] == '/clear')
        measures = [a for a in world.actions[:first] if a['path'] == '/measure']
        self.assertGreaterEqual(len(measures), 420)
        for offset in range(0, 420, 20):
            station = measures[offset:offset+20]
            self.assertEqual({a['request']['channel'] for a in station}, set(range(1, 21)))
            self.assertTrue(all(a['request']['position'] == station[0]['request']['position'] for a in station))
        self.assertEqual(len(state.full_scan_stations), 21)
        self.assertTrue(covers_mixed(state.full_scan_stations))

    def test_spiral_has_mixed_direction_coverage(self):
        self.assertTrue(covers_mixed(spiral_points()))

    def test_random_schedule_is_reproducible_bounded_and_has_no_coverage_claim(self):
        points = random_walk_points()
        self.assertEqual(points, random_walk_points())
        self.assertNotEqual(points, random_walk_points(123))
        self.assertEqual(len(points), 129)
        self.assertTrue(all(math.hypot(*p) <= 1800 for p in points))
        self.assertTrue(all(abs(math.dist(a, b)-500) < 1e-8 for a, b in zip(points, points[1:])))
        state, world = self.run_control(RandomWalkController,
            [MixedSource(1, (0., 0.), 1000., None)], walk_steps=0)
        self.assertEqual(world.cleared, {1})
        self.assertFalse(state.all_cleared)
        self.assertEqual(state.termination_reason, 'scan_limit')
        self.assertEqual(state.full_scan_stations, [(0., 0.)])

    def test_oracle_uses_actual_receipts_zero_measures_and_respects_budget(self):
        sources = [MixedSource(1, (100., 0.), 1000., None),
                   MixedSource(2, (200., 0.), 1000., 180.)]
        positions = {s.channel: s.position for s in sources}
        state, world = self.run_control(OracleController, sources, known_positions=positions)
        self.assertTrue(state.all_cleared)
        self.assertTrue(state.privileged_truth)
        self.assertFalse(any(a['path'] == '/measure' for a in world.actions))
        self.assertEqual(len(state.clear_attempts), 2)
        self.assertAlmostEqual(world.virtual_time_s, 200/5+2*5)
        state, world = self.run_control(OracleController, sources, known_positions=positions, max_actions=1)
        self.assertFalse(state.all_cleared)
        self.assertEqual(state.termination_reason, 'action_budget')
        self.assertEqual(len(world.cleared), 1)

    def test_truth_is_not_given_to_an_arbitrary_controller(self):
        class Ordinary:
            def __init__(self, client, **kwargs):
                self.options = kwargs
        control = make_controller(Ordinary, object(), object(), {'public_option': 4})
        self.assertEqual(control.options, {'public_option': 4})

    def test_incomplete_random_and_oracle_replay(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            for name, spec, options in [
                ('random', 'src.q4.control_random_walk:RandomWalkController', {'walk_steps': 0}),
                ('oracle', 'src.q4.control_oracle:OracleController', {})]:
                report = evaluate_benchmark(root/name, controller=spec, options=options,
                    counts=(10,), cases_per_group=2, seed=280010000)
                self.assertEqual(report['all_cases_cleared'], name == 'oracle')
                if name == 'random':
                    self.assertIsNone(report['equal_weight_mean_per_source_s'])
                self.assertTrue(replay(root/name, root/f'{name}-replay.json')['all_identical'])

    def test_oracle_not_available_in_official_entry(self):
        from src.q4.main import build_parser
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(['--robot-id', 'test', '--strategy', 'oracle'])

    def test_tables_preserve_incomplete_results_and_exclude_truth_reference(self):
        from src.q4.comparison_tables import summarize
        case = dict(source_count=10, cleared_count=10, all_cleared=False,
            actual_all_cleared=True, audit_passed=True, clearance_ratio=1.,
            raw_time_per_source_s=100., total_time_s=1000., program_runtime_s=1.)
        group = dict(label='random', controller='src.q4.control_random_walk:RandomWalkController',
            privileged_truth=False, cases=[case])
        oracle = dict(label='oracle', controller='src.q4.control_oracle:OracleController',
            privileged_truth=True, cases=[dict(case, all_cleared=True)])
        rows = summarize(dict(strategies=dict(random_walk=group, oracle=oracle)))
        self.assertEqual(rows[0]['clearance_percent'], 100.)
        self.assertIsNone(rows[0]['valid_mean_per_source_s'])
        self.assertEqual(rows[0]['raw_mean_per_source_s'], 100.)
        self.assertFalse(rows[0]['ranking_eligible'])
        self.assertEqual(rows[1]['valid_mean_per_source_s'], 100.)
        self.assertFalse(rows[1]['ranking_eligible'])
