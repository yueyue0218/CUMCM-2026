import math
import unittest
import numpy as np
from src.q3.baseline_scan import MeasureObservation
from src.q4.joint_model import continuous_weights,hypotheses


class ContinuousOrientationTests(unittest.TestCase):
    def test_coincident_near_does_not_restrict_orientation(self):
        near=MeasureObservation((0,0),1,'near',None,0)
        self.assertEqual(continuous_weights([(0,0)],[near])[0],1.)
        negative=MeasureObservation((100,0),1,'no_signal',None,0)
        self.assertAlmostEqual(continuous_weights([(0,0)],[near,negative])[0],.25)
        contradictory=MeasureObservation((0,0),1,'no_signal',None,0)
        self.assertEqual(continuous_weights([(0,0)],[near,contradictory])[0],0.)

    def test_quarter_circle_and_rotation_invariance(self):
        for a in range(-180,181,15):
            rad=math.radians(a)
            positive=(100*math.cos(rad),100*math.sin(rad))
            negative=(-100*math.sin(rad),100*math.cos(rad))
            observations=[MeasureObservation(positive,1,'direction',(a+180)%360,0),
                          MeasureObservation(negative,1,'no_signal',None,0)]
            self.assertAlmostEqual(continuous_weights([(0,0)],observations)[0],.125)

    def test_identical_positive_and_negative_have_zero_mass(self):
        observations=[MeasureObservation((100,0),1,'direction',180,0),
                      MeasureObservation((100,0),1,'no_signal',None,0)]
        self.assertEqual(continuous_weights([(0,0)],observations)[0],0)

    def test_continuous_measure_matches_dense_orientation_quadrature(self):
        points=np.array([(0,0),(1,0),(0,1),(-.5,.5)])
        observations=[MeasureObservation((100,20),1,'direction',math.degrees(math.atan2(-20,-100)),0),
                      MeasureObservation((30,110),1,'no_signal',None,0),
                      MeasureObservation((-1400,0),1,'no_signal',None,0)]
        discrete,_,_=hypotheses(points,observations,orientation_count=72000)
        np.testing.assert_allclose(continuous_weights(points,observations),discrete,atol=2e-5)


if __name__=='__main__':
    unittest.main()
