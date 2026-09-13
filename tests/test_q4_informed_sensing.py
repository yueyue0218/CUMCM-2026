"""Safety of optional radio-sensing suppression, using public transport calls."""
import math
import unittest
from unittest.mock import patch

import numpy as np

from src.common.simulator_client import SimulatorClient
from src.q4.informed import InformedController
from src.q4.offline_simulator import MixedSimulator, MixedSource


class InformedSensingTests(unittest.TestCase):
    @staticmethod
    def west_facing_particles():
        directional=np.zeros((1,72),dtype=bool)
        directional[0,36]=True
        return (np.asarray([[700.,0.]]),np.asarray([1.]),directional,
                np.asarray([1000.]),np.asarray([0.]))

    def test_rejected_optional_revisit_preserves_target_and_full_scan_unknowns(self):
        world=MixedSimulator([MixedSource(1,(700.,0.),1000.,180.),
                              MixedSource(2,(700.,0.),1000.,180.)])
        client=SimulatorClient('test',transport=world.transport)
        control=InformedController(client)
        client.enter()
        control.batch((0.,0.),force_scan=True)
        unknown_before=set(control.state.unknown_channels)
        self.assertEqual(unknown_before,set(range(3,21)))
        target=(1000.,200.)  # Behind the west-facing sources, >150 m away.
        begin=len(world.actions)
        with patch.object(control,'particles',return_value=self.west_facing_particles()):
            self.assertFalse(control.should_measure_auxiliary(2,target,math.sqrt(130000),45.))
            control.batch(target,target_channel=1,force_scan=True)
        measured=[a['request']['channel'] for a in world.actions[begin:] if a['path']=='/measure']
        self.assertEqual(set(measured),{1}|unknown_before)
        self.assertEqual(len(measured),19)
        self.assertIn(1,measured)  # The mandatory target bypasses visibility.
        self.assertNotIn(2,measured)  # Only the optional active channel is skipped.
        self.assertEqual(control.state.channels[1].observations[-1].measure_result,'no_signal')
        self.assertEqual(len(control.state.channels[2].observations),1)
        self.assertIn(2,control.state.active_channels)
        self.assertEqual(control.state.excluded_channels,set())
        for channel in unknown_before:
            last=control.state.channels[channel].observations[-1]
            self.assertEqual(last.position,target)
            self.assertEqual(last.measure_result,'no_signal')
        self.assertIn(target,control.state.full_scan_stations)

    def test_nearby_optional_revisit_is_kept_without_particle_inference(self):
        control=InformedController(SimulatorClient('test'))
        with patch.object(control,'particles',side_effect=AssertionError('near case must bypass particles')):
            self.assertTrue(control.should_measure_auxiliary(1,(100.,0.),149.999,0.))

    def test_failed_particle_approximation_keeps_geometrically_useful_revisit(self):
        control=InformedController(SimulatorClient('test'))
        with patch.object(control,'particles',return_value=None) as particles:
            self.assertTrue(control.should_measure_auxiliary(1,(300.,300.),500.,30.))
            particles.assert_called_once()
        self.assertEqual(control.state.excluded_channels,set())

    def test_visibility_respects_front_and_back_half_planes(self):
        control=InformedController(SimulatorClient('test'))
        with patch.object(control,'particles',return_value=self.west_facing_particles()):
            self.assertTrue(control.should_measure_auxiliary(1,(300.,200.),500.,30.))
            self.assertFalse(control.should_measure_auxiliary(1,(1000.,200.),500.,30.))

    def test_radius_prior_keeps_exact_half_visibility_threshold(self):
        control=InformedController(SimulatorClient('test'))
        # Omni source at the origin; uniform radius in [1000,1500] gives
        # exactly 1/2 reception probability at distance 1250 metres.
        particles=(np.asarray([[0.,0.]]),np.asarray([1.]),np.zeros((1,72),dtype=bool),
                   np.asarray([1000.]),np.asarray([1.]))
        with patch.object(control,'particles',return_value=particles):
            self.assertTrue(control.should_measure_auxiliary(1,(1250.,0.),1250.,30.))
            self.assertFalse(control.should_measure_auxiliary(1,(1250.01,0.),1250.01,30.))


if __name__=='__main__':
    unittest.main()
