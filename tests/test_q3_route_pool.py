import math
import unittest
from src.q3.coverage_control import covers_arena, coverage_waypoints
from src.q3.experiment_joint_routing import optimized_route


class RoutePoolTests(unittest.TestCase):
    def test_cli_v2_runs_and_records_its_version_and_completion_evidence(self):
        import io,json,tempfile
        from pathlib import Path
        from contextlib import redirect_stdout,redirect_stderr
        from unittest.mock import patch
        from src.q3.main import build_parser,run
        from src.common.simulator_client import SimulatorClient
        from src.q3.offline_simulator import OfflineSimulator,Source
        world=OfflineSimulator([Source(c,(500,0),1000) for c in range(1,17)])
        def factory(robot_id,**kwargs):
            return SimulatorClient(robot_id,transport=world.transport,**kwargs)
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            args=build_parser().parse_args(['--robot-id','test','--strategy','efficient_v2','--run-root',directory])
            with patch('src.q3.main.SimulatorClient',side_effect=factory):code=run(args)
            summary=json.loads((next(Path(directory).iterdir())/'summary.json').read_text('utf-8'))
        self.assertEqual(code,0)
        self.assertTrue(world.exited)
        self.assertEqual(summary['strategy'],'efficient_v2')
        self.assertEqual(summary['algorithm_version'],'route-pool-v2')
        self.assertEqual(summary['total_count_basis'],'source_count_upper_bound_and_clear_receipts')

    def test_pool_is_certified_and_does_not_worsen_its_static_route_objective(self):
        from src.q3.coverage_route_pool import pooled_coverage_waypoints, coverage_route_cost
        for stations,targets,current in [([(0,0)],{1:(600,700),2:(-900,-200)},(250,0)),
                                          ([(0,0),(1200,0)],{3:(400,-1400)},(1200,0)),
                                          ([(0,0)],{},(0,0))]:
            old=coverage_waypoints(stations,targets,current)
            new=pooled_coverage_waypoints(stations,targets,current)
            self.assertTrue(covers_arena(stations+list(targets.values())+new))
            self.assertLessEqual(coverage_route_cost(current,targets,new),coverage_route_cost(current,targets,old)+1e-6)
            self.assertTrue(all(len(p)==2 and all(math.isfinite(v) for v in p) for p in new))

    def test_hypothetical_complete_cover_adds_no_search_points(self):
        from src.q3.coverage_route_pool import pooled_coverage_waypoints
        from src.q3.experiment_joint_routing import compact_coverage_points
        self.assertEqual(pooled_coverage_waypoints(compact_coverage_points(),{},(0,0)),[])

    def test_default_candidate_clears_with_certificates_and_handles_budget(self):
        from src.q3.experiment_route_pool import run_route_pool
        from src.common.simulator_client import SimulatorClient
        from src.q3.offline_simulator import OfflineSimulator,Source
        for budget in (360000,50):
            world=OfflineSimulator([Source(1,(1700,0),1000),Source(2,(-1600,300),1000)],
                                   bearing_error_deg=1,error_field='spatial',quantize_bearings=True)
            client=SimulatorClient('test',transport=world.transport);client.enter()
            state=run_route_pool(client,virtual_limit_s=budget)
            if budget==50:
                self.assertEqual(state.termination_reason,'virtual_time_budget')
                self.assertFalse(state.all_cleared)
            else:
                self.assertTrue(state.all_cleared)
                self.assertEqual(world.cleared,{1,2})
                self.assertTrue(all(a.certificate in {'q1','near','analytic_fallback'} for a in state.clear_attempts))


if __name__=='__main__':unittest.main()
