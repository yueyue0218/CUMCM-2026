import math
import unittest

from src.q3.coverage_control import covers_arena
from src.q3.experiment_joint_routing import run_efficient, build_efficient_summary, EfficientState, EfficientController, optimized_route
from src.q3.offline_simulator import OfflineSimulator, Source
from src.common.simulator_client import SimulatorClient


class CoverageTests(unittest.TestCase):
    def test_empty_and_central_only_do_not_cover(self):
        self.assertFalse(covers_arena([]))
        self.assertFalse(covers_arena([(0, 0)]))

    def test_compact_hexagon_covers_including_boundary(self):
        points = [(0, 0)] + [(1150*math.cos(i*math.pi/3), 1150*math.sin(i*math.pi/3)) for i in range(6)]
        self.assertTrue(covers_arena(points))
        self.assertFalse(covers_arena(points[:-1]))

    def test_sparse_boundary_samples_cannot_hide_hole(self):
        self.assertFalse(covers_arena([(1800*math.cos(i*math.pi/3), 1800*math.sin(i*math.pi/3)) for i in range(6)]))


class EfficientTests(unittest.TestCase):
    def test_evaluation_retains_failed_cases_and_produces_an_invalid_speed_score(self):
        import io,json,tempfile
        from pathlib import Path
        from contextlib import redirect_stdout
        from unittest.mock import patch
        from src.q3.evaluate_efficiency import evaluate
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            with patch('src.q3.evaluate_efficiency.run_efficient',side_effect=ArithmeticError('injected')):
                report=evaluate(cases_per_group=1,modes=('efficient',),output_dir=Path(directory)/'tables',run_root=Path(directory)/'runs')
            self.assertEqual(len(report['cases']),4)
            self.assertIsNone(report['aggregate']['efficient']['equal_weight_mean_per_source_s'])
            traces=list(Path(report['run_directory']).glob('n*_efficient.json'))
            self.assertEqual(len(traces),4)
            self.assertTrue(all('injected' in json.loads(p.read_text('utf-8'))['failure'] for p in traces))

    def test_stratified_aggregation_uses_equal_group_weights_and_rejects_failures(self):
        from src.q3.evaluate_efficiency import aggregate
        rows=[{'mode':'efficient','source_count':n,'all_cleared':True,'per_source_s':v,
               'total_time_s':n*v,'program_runtime_s':1} for n,v in ((10,300),(12,250),(14,220),(16,200))]
        summary=aggregate(rows,['efficient'])['efficient']
        self.assertEqual(summary['equal_weight_mean_per_source_s'],242.5)
        rows[0]['all_cleared']=False
        summary=aggregate(rows,['efficient'])['efficient']
        self.assertFalse(summary['all_cases_cleared'])
        self.assertIsNone(summary['equal_weight_mean_per_source_s'])

    def test_cli_efficient_entry_logs_dynamic_evidence_and_exits(self):
        import io,json,tempfile
        from pathlib import Path
        from contextlib import redirect_stdout
        from unittest.mock import patch
        from src.q3.main import build_parser,run
        world=OfflineSimulator([Source(c,(500,0),1000) for c in range(1,17)])
        def factory(robot_id,**kwargs):
            return SimulatorClient(robot_id,transport=world.transport,**kwargs)
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            args=build_parser().parse_args(['--robot-id','test','--strategy','efficient','--run-root',directory])
            with patch('src.q3.main.SimulatorClient',side_effect=factory):code=run(args)
            summary=json.loads((next(Path(directory).iterdir())/'summary.json').read_text('utf-8'))
        self.assertEqual(code,0)
        self.assertTrue(world.exited)
        self.assertEqual(summary['strategy'],'efficient')
        self.assertEqual(summary['total_count_basis'],'source_count_upper_bound_and_clear_receipts')

    def test_batches_channels_before_departing_and_clears_all(self):
        world = OfflineSimulator([Source(1, (700, 0), 1100), Source(2, (0, 650), 1200)],
                                 bearing_error_deg=1, error_field='spatial', quantize_bearings=True)
        client = SimulatorClient('test', transport=world.transport)
        client.enter()
        state = run_efficient(client)
        self.assertTrue(state.all_cleared)
        self.assertEqual(world.cleared, {1, 2})
        first = [a for a in world.actions if a['path']=='/measure'][:20]
        self.assertEqual({tuple(a['request']['position'].values()) for a in first}, {(0, 0)})
        self.assertTrue(state.completed_full_cover)
        summary=build_efficient_summary(state,client.state.virtual_time_s)
        self.assertGreater(len(summary['full_scan_stations']),1)
        for c in state.excluded_channels:
            self.assertEqual(summary['negative_coverage_receipts'][str(c)],list(range(len(state.full_scan_stations))))
        self.assertEqual(summary['total_count_basis'],'dynamic_reception_disk_coverage_and_clear_receipts')

    def test_upper_bound_proof_is_reported_separately(self):
        world=OfflineSimulator([Source(c,(700,0),1000) for c in range(1,17)])
        client=SimulatorClient('test',transport=world.transport);client.enter()
        state=run_efficient(client)
        summary=build_efficient_summary(state,client.state.virtual_time_s)
        self.assertTrue(state.all_cleared)
        self.assertFalse(state.completed_full_cover)
        self.assertEqual(summary['total_count_basis'],'source_count_upper_bound_and_clear_receipts')
        self.assertEqual(summary['source_count_evidence_channels'],list(range(1,17)))
        self.assertEqual(world.cleared,set(range(1,17)))

    def test_incomplete_batch_does_not_certify_a_station(self):
        world=OfflineSimulator([Source(1,(1600,0),1000)])
        def transport(url,body,timeout):
            if url.endswith('/measure') and len(world.actions)>=5:
                raise TimeoutError('lost receipt')
            return world.transport(url,body,timeout)
        client=SimulatorClient('test',transport=transport,max_network_retries=0)
        client.enter();state=EfficientState()
        with self.assertRaises(Exception):run_efficient(client,state=state)
        self.assertFalse(state.all_cleared)
        self.assertEqual(state.full_scan_stations,[])
        self.assertIsNotNone(client.pending_action)

    def test_real_time_exhaustion_keeps_unresolved_channels(self):
        world=OfflineSimulator([Source(1,(1000,0),1000)],remaining_real_duration_s=1)
        client=SimulatorClient('test',transport=world.transport);client.enter()
        state=run_efficient(client)
        self.assertFalse(state.all_cleared)
        self.assertEqual(state.termination_reason,'real_time_budget')

    def test_global_route_limit_finishes_with_fixed_cover_and_certified_fallback(self):
        world=OfflineSimulator([Source(1,(1700,0),1000),Source(2,(-1600,300),1000)],
                               bearing_error_deg=1,error_field='spatial',quantize_bearings=True)
        client=SimulatorClient('test',transport=world.transport);client.enter()
        state=EfficientState(routing_steps=256)
        run_efficient(client,state=state)
        self.assertTrue(state.all_cleared)
        self.assertEqual(world.cleared,{1,2})
        self.assertTrue(state.completed_full_cover)

    def test_boundary_sources_with_fixed_extreme_bearing_errors(self):
        for error in (-1,1):
            sources=[Source(c,(1800*math.cos(c*math.pi/8),1800*math.sin(c*math.pi/8)),1000) for c in range(1,17)]
            world=OfflineSimulator(sources,bearing_error_deg=error,quantize_bearings=True)
            client=SimulatorClient('test',transport=world.transport);client.enter()
            state=run_efficient(client)
            self.assertTrue(state.all_cleared)
            self.assertEqual(world.cleared,set(range(1,17)))

    def test_multistart_route_preserves_targets_and_matches_small_exhaustive_case(self):
        import itertools
        start=(0,0);targets={1:(3,0),2:(3,4),3:(0,4),4:(-2,3)}
        def length(order):
            points=[start]+[targets[c] for c in order]
            return sum(math.dist(a,b) for a,b in zip(points,points[1:]))
        route=optimized_route(start,targets)
        self.assertEqual(set(route),set(targets))
        self.assertAlmostEqual(length(route),min(map(length,itertools.permutations(targets))))

    def test_budget_failure_does_not_claim_completion(self):
        world = OfflineSimulator([Source(1, (1800, 0), 1000)])
        client = SimulatorClient('test', transport=world.transport)
        client.enter()
        state = run_efficient(client, virtual_limit_s=50)
        self.assertFalse(state.all_cleared)
        self.assertEqual(state.termination_reason, 'virtual_time_budget')


if __name__ == '__main__':
    unittest.main()
