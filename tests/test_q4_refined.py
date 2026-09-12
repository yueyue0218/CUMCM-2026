import unittest
from src.common.simulator_client import SimulatorClient
from src.q4.offline_simulator import MixedSimulator, MixedSource


class RefinedTests(unittest.TestCase):
    def test_failed_probe_retains_source_and_later_clear_succeeds(self):
        from src.q4.refined import RefinedController
        world = MixedSimulator([MixedSource(1, (500, 0), 1000, 180)])
        client = SimulatorClient('test', transport=world.transport)
        control = RefinedController(client)
        client.enter()
        control.batch((0, 0), force_scan=True)
        self.assertFalse(control.optical_probe(1, (550, 0)))
        self.assertIn(1, control.state.active_channels)
        self.assertEqual(control.state.clear_attempts[-1].result, 'no_target_in_range')
        self.assertTrue(control.optical_probe(1, (501, 0)))
        self.assertEqual(control.state.cleared_channels, {1})

    def test_outward_source_and_empty_channels_still_require_cover(self):
        from src.q4.refined import RefinedController
        world = MixedSimulator([MixedSource(1, (1800, 0), 1000, 0),
                                MixedSource(2, (600, 500), 1000, None)],
                               bearing_error_deg=1, error_field='spatial')
        client = SimulatorClient('test', transport=world.transport)
        control = RefinedController(client)
        client.enter()
        state = control.run()
        self.assertTrue(state.all_cleared, state.failure_detail)
        self.assertEqual(world.cleared, {1, 2})
        self.assertTrue(state.completed_full_cover)

    def test_compact_layout_continuous_check_and_orientation_samples(self):
        import math
        from src.q4.coverage import covers_mixed
        from src.q4.refined import RefinedController
        world = MixedSimulator([])
        client = SimulatorClient('test', transport=world.transport)
        control = RefinedController(client)
        client.enter()
        control.batch((0,0), force_scan=True)
        planned = control.plan_coverage({})
        self.assertEqual(len(planned),20)
        points = [(0.,0.)]+planned
        self.assertTrue(covers_mixed(points))
        for angle in range(0,360,3):
            a=math.radians(angle)
            source=(1800*math.cos(a),1800*math.sin(a))
            for orientation in range(0,360,15):
                b=math.radians(orientation)
                self.assertTrue(any(math.dist(source,p)<=1000 and
                    (p[0]-source[0])*math.cos(b)+(p[1]-source[1])*math.sin(b)>=0 for p in points))

    def test_outward_short_radius_constant_extreme_error(self):
        import math
        from src.q4.refined import RefinedController
        for count,error in ((10,-1.),(16,1.)):
            sources=[MixedSource(1,(0,0),1000,None)]
            for c in range(2,count+1):
                angle=(c-2)*360/(count-1)+17
                a=math.radians(angle)
                sources.append(MixedSource(c,(1800*math.cos(a),1800*math.sin(a)),1000,angle))
            world=MixedSimulator(sources,bearing_error_deg=error,error_field='constant')
            client=SimulatorClient('test',transport=world.transport)
            control=RefinedController(client)
            client.enter()
            state=control.run()
            self.assertTrue(state.all_cleared,state.failure_detail)
            self.assertEqual(world.cleared,set(range(1,count+1)))

    def test_refined_request_budget_keeps_confirmed_receipts_only(self):
        from src.q4.refined import RefinedController
        world=MixedSimulator([MixedSource(1,(1800,0),1000,0)])
        client=SimulatorClient('test',transport=world.transport)
        control=RefinedController(client,max_requests=3)
        client.enter()
        state=control.run()
        self.assertFalse(state.all_cleared)
        self.assertFalse(state.full_scan_stations)
        self.assertIsNone(client.pending_action)
        client.exit()
        self.assertEqual(state.network_requests,3)

    def test_radius_integration_respects_available_radius_mass(self):
        from src.q3.baseline_scan import MeasureObservation
        from src.q4.joint_model import hypotheses
        observations=[MeasureObservation((0,0),1,'direction',0,0)]
        weights,_,_=hypotheses([(800,0),(1400,0)],observations,integrate_radius=True)
        self.assertGreater(weights[0],weights[1]*4)


if __name__ == '__main__':
    unittest.main()
