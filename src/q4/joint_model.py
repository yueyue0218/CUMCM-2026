"""Finite mixed-type/radius/orientation hypotheses used ONLY for action ranking.

Radius is eliminated at L(g)=max(1000, positive distances). A negative inside
L must be strictly behind a directional source. These sampled hypotheses may
not certify absence, clearance, or shrink the positive-bearing outer bound.
"""
from __future__ import annotations

import math
import numpy as np


def hypotheses(points, observations, orientation_count=72, integrate_radius=False):
    points = np.asarray(points, dtype=float)
    angles = np.arange(orientation_count)*2*math.pi/orientation_count
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    lower = np.full(len(points), 1000.)
    distance = []
    for o in observations:
        delta = np.asarray(o.position)-points
        d = np.linalg.norm(delta, axis=1)
        distance.append((o, delta, d))
        if o.measure_result in {'near','direction'}:
            lower = np.maximum(lower, d)
    omni = lower <= 1500
    directional = np.broadcast_to(omni[:, None], (len(points), orientation_count)).copy()
    upper_o = np.full(len(points),1500.)
    upper_d = np.full((len(points),orientation_count),1500.)
    for o, delta, d in distance:
        if o.measure_result == 'no_signal':
            omni &= d > lower
            directional &= (d[:, None] > lower[:, None]) | (delta @ normals.T < 0)
            upper_o = np.minimum(upper_o,d)
            upper_d = np.minimum(upper_d,np.where(delta @ normals.T >= 0,d[:,None],1500.))
        else:
            if o.measure_result == 'near':
                valid = d <= 5
            else:
                angle = np.degrees(np.arctan2(-delta[:,1], -delta[:,0]))
                valid = (d > 5) & (np.abs((angle-o.svd_deg+180) % 360-180) <= 1+1e-8)
            omni &= valid
            directional &= valid[:, None] & (delta @ normals.T >= 0)
    # Equal O/D prior; flat discrete orientation working approximation.
    weights = .5*omni + .5*directional.mean(axis=1)
    if integrate_radius:
        weights = (.5*omni*np.maximum(upper_o-lower,0)/500
                   +.5*(directional*np.maximum(upper_d-lower[:,None],0)/500).mean(axis=1))
    return weights, directional, lower


def continuous_weights(points, observations):
    """Exact angular measure of feasible normals at each sampled position.

    Radius is existentially eliminated as in hypotheses(), not integrated.
    A first positive half-circle anchors an unwrapped interval of width pi.
    Strict versus closed endpoints have equal angular measure. This remains
    a planning likelihood, never a certificate about unsampled positions.
    """
    points=np.asarray(points,dtype=float)
    positive=[o for o in observations if o.measure_result in {'direction','near'}]
    if not positive:
        raise ValueError('at least one positive observation required')
    reference=np.zeros(len(points))
    anchored=np.zeros(len(points),dtype=bool)
    lower=np.full(len(points),1000.)
    for o in positive:
        lower=np.maximum(lower,np.linalg.norm(np.asarray(o.position)-points,axis=1))
    omni=lower<=1500
    valid=omni.copy()
    lo=np.full(len(points),-math.pi/2)
    hi=np.full(len(points), math.pi/2)
    constraints=[]
    for o in observations:
        delta=np.asarray(o.position)-points
        d=np.linalg.norm(delta,axis=1)
        angle=np.arctan2(delta[:,1],delta[:,0])
        if o.measure_result=='no_signal':
            constrained=d<=lower
            omni &= ~constrained
            valid &= d>0  # A coincident station is never strictly behind.
            angle+=math.pi
        else:
            constrained=d>0  # At the source the closed half-plane is unrestricted.
            if o.measure_result=='near':
                legal=d<=5
            else:
                bearing=np.degrees(np.arctan2(-delta[:,1],-delta[:,0]))
                legal=(d>5)&(np.abs((bearing-o.svd_deg+180)%360-180)<=1+1e-8)
            omni &= legal
            valid &= legal
        new=constrained & ~anchored
        reference=np.where(new,angle,reference)
        anchored |= constrained
        constraints.append((angle,constrained))
    for angle,constrained in constraints:
        relative=(angle-reference+math.pi)%(2*math.pi)-math.pi
        lo=np.maximum(lo,np.where(constrained,relative-math.pi/2,-math.pi/2))
        hi=np.minimum(hi,np.where(constrained,relative+math.pi/2, math.pi/2))
    angular_mass=np.where(anchored,np.maximum(hi-lo,0)/(2*math.pi),1.)
    return .5*omni+.5*valid*angular_mass


def halton(count, base):
    result=np.zeros(count)
    digits=np.arange(1,count+1)
    factor=1.
    while digits.any():
        factor/=base
        result+=factor*(digits%base)
        digits//=base
    return result


def sampled_center(region, observations, integrate_radius=False, *, sample_count=256,
                   sampler='random', continuous_orientation=False):
    if sampler not in {'random','halton'} or not 16<=sample_count<=4096:
        raise ValueError('invalid joint sampling configuration')
    if continuous_orientation and integrate_radius:
        raise ValueError('continuous angular measure does not integrate radius')
    polygon = np.asarray(region)
    if len(polygon) < 3:
        return None
    center = polygon.mean(axis=0)
    nxt = np.roll(polygon, -1, axis=0)
    first, second = polygon-center, nxt-center
    areas = np.abs(first[:,0]*second[:,1]-first[:,1]*second[:,0])
    if areas.sum() <= 1e-12:
        return None
    if sampler=='random':
        rng = np.random.default_rng(20260912)
        idx = rng.choice(len(polygon), sample_count, p=areas/areas.sum())
        ab = rng.random((sample_count, 2))
    else:
        idx=np.searchsorted(np.cumsum(areas/areas.sum()),halton(sample_count,2))
        idx=np.minimum(idx,len(polygon)-1)
        ab=np.column_stack((halton(sample_count,3),halton(sample_count,5)))
    ab[ab.sum(axis=1) > 1] = 1-ab[ab.sum(axis=1) > 1]
    samples = center+ab[:, :1]*first[idx]+ab[:, 1:]*second[idx]
    if continuous_orientation:
        weight=continuous_weights(samples,observations)
    else:
        weight, _, _ = hypotheses(samples, observations,integrate_radius=integrate_radius)
    if weight.sum() <= 1e-12:
        return None
    return tuple(np.average(samples, axis=0, weights=weight))
