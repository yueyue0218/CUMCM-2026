import math
import unittest
from src.common.simulator_client import SimulatorClient
from src.q4.informed import InformedController
from src.q4.offline_simulator import sample_sources
from src.q4.benchmark import make_world


class InformedTests(unittest.TestCase):
    def test_invalid_information_parameters_fail_before_transport(self):
        for value in (True,float('inf'),float('nan'),-1,101):
            with self.assertRaises(ValueError):
                InformedController(SimulatorClient('test'),information_penalty=value)
        for key,values in (('minimum_visibility',(True,-.1,1.1,float('nan'))),
                           ('auxiliary_radius_m',(False,0,149,1501,float('inf')))):
            for value in values:
                with self.assertRaises(ValueError):
                    InformedController(SimulatorClient('test'),**{key:value})

    def test_budget_exit_does_not_claim_unobserved_channels_absent(self):
        world=make_world(sample_sources(210000000,10),210000000,'hash')
        client=SimulatorClient('test',transport=world.transport)
        controller=InformedController(client,max_requests=3)
        client.enter();state=controller.run();client.exit()
        self.assertFalse(state.all_cleared)
        self.assertFalse(state.completed_full_cover)
        self.assertFalse(state.full_scan_stations)
        self.assertEqual(state.network_requests,3)
        self.assertTrue(state.unknown_channels)


if __name__=='__main__':
    unittest.main()
