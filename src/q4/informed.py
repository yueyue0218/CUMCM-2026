"""Information-aware Q4 sensing with the existing complete-search safeguards.

The policy samples a joint position/type/orientation belief to choose the next
measurement. Predicted signal/absence outcomes and bounded-angle likelihoods
score movement, onward travel and residual localization uncertainty. Samples
never certify absence or clearance. All physical actions and fallback budgets
remain owned by RefinedController and MixedController.

Extracted from the independently tested information3 research candidate.
"""
from __future__ import annotations
import math
import numpy as np
from src.q3.localization_control import ChannelLocalizationDecision
from src.q4.joint_model import hypotheses,halton
from src.q4.refined import RefinedController

def polygon_particles(region, count=256):
    polygon = np.asarray(region, dtype=float)
    center = polygon.mean(axis=0)
    first, second = polygon-center, np.roll(polygon, -1, axis=0)-center
    areas = np.abs(first[:,0]*second[:,1]-first[:,1]*second[:,0])
    if areas.sum() <= 1e-12:
        return np.repeat(center[None,:], count, axis=0)
    idx = np.minimum(np.searchsorted(np.cumsum(areas/areas.sum()), halton(count,2)),len(polygon)-1)
    ab = np.column_stack((halton(count,3),halton(count,5)))
    ab[ab.sum(axis=1)>1] = 1-ab[ab.sum(axis=1)>1]
    return center+ab[:,:1]*first[idx]+ab[:,1:]*second[idx]

class InformedController(RefinedController):
    def __init__(self,client,state=None,*,information_penalty=3.,
                 auxiliary_radius_m=1400.,minimum_visibility=.5,**kwargs):
        if (isinstance(information_penalty,bool) or not math.isfinite(information_penalty)
                or not 0<=information_penalty<=100):
            raise ValueError('information penalty must be finite in [0,100]')
        for name,value,low,high in (
                ('auxiliary radius',auxiliary_radius_m,150,1500),
                ('minimum visibility',minimum_visibility,0,1)):
            if isinstance(value,bool) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError(f'{name} must be finite in [{low},{high}]')
        super().__init__(client,state,**kwargs)
        self.information_penalty=information_penalty
        self.auxiliary_radius_m=auxiliary_radius_m
        self.minimum_visibility=minimum_visibility
        self.research_mode='information'
        self.particle_cache={}
        self.state.strategy_name='q4-informed-batched-routing-v4'
        self.state.refinement_config.update(information_penalty=information_penalty,
            auxiliary_radius_m=auxiliary_radius_m,minimum_visibility=minimum_visibility)
        self.state.controller_class=f'{type(self).__module__}:{type(self).__name__}'

    def auxiliary_measurement_worthwhile(self,distance,turn):
        return distance<150 or (distance<self.auxiliary_radius_m and turn>=12)

    def should_measure_auxiliary(self,c,position,distance,turn):
        """Skip only optional revisits; target and unknown-channel scans bypass this.

        Visibility is a working probability, never absence evidence. A channel
        rejected here stays active and is still resolved by the ordinary loop.
        """
        if not super().should_measure_auxiliary(c,position,distance,turn):
            return False
        if not self.minimum_visibility or distance<150:
            return True
        data=self.particles(c,count=96)
        if data is None:
            return True
        points,weight,directional,lower,omni=data
        delta=np.asarray(position)-points
        radius=np.linalg.norm(delta,axis=1)
        angles=np.arange(72)*math.tau/72
        normals=np.column_stack((np.cos(angles),np.sin(angles)))
        radial=np.clip((1500-np.maximum(lower,radius))/np.maximum(1500-lower,1e-6),0,1)
        raw=.5*omni+.5*directional.mean(axis=1)
        visible=radial*(.5*omni+.5*(directional&(delta@normals.T>=0)).mean(axis=1))
        probability=np.divide(visible,raw,out=np.zeros_like(raw),where=raw>0)
        return weight@probability>=self.minimum_visibility

    def particles(self,c,count=256):
        history=self.state.channels[c].observations
        failures=[a.position for a in self.state.clear_attempts
                  if a.channel==c and a.result=='no_target_in_range']
        key=c,len(history),len(failures),count
        if key not in self.particle_cache:
            points=polygon_particles(self.evaluation(c).assessment.outer_region,count)
            weight,directional,lower=hypotheses(points,history)
            omni=np.clip(2*weight-directional.mean(axis=1),0,1)
            for position in failures:
                weight[np.linalg.norm(points-position,axis=1)<=20.]=0
            if weight.sum()<=1e-12:
                self.particle_cache[key]=None
            else:
                self.particle_cache[key]=(points,weight/weight.sum(),directional,lower,omni)
        return self.particle_cache[key]

    def target(self,c):
        base=super().target(c)
        if self.research_mode!='information':
            return base
        e=self.evaluation(c)
        if e.decision is ChannelLocalizationDecision.READY_TO_CLEAR:
            return base
        data=self.particles(c,count=96)
        if data is None:
            return base
        points,weight,directional,lower,omni=data
        center=weight@points
        current=np.asarray(self.client.state.position)
        history=self.state.channels[c].observations
        last=next(o for o in reversed(history) if o.measure_result=='direction')
        angle=math.radians(last.svd_deg)
        offsets=[0.,30.,60.,120.,240.]
        candidates=[np.asarray(base),center,current]
        for radius in offsets[1:]:
            for turn in (0,45,90,135,180,225,270,315):
                a=angle+math.radians(turn)
                candidates.append(center+radius*np.asarray([math.cos(a),math.sin(a)]))
        normals=np.column_stack((np.cos(np.arange(72)*math.tau/72),
                                 np.sin(np.arange(72)*math.tau/72)))
        raw=.5*omni+.5*directional.mean(axis=1)
        best=None
        for candidate in candidates:
            if any(math.dist(candidate,o.position)<1e-6 for o in history):
                continue
            delta=candidate-points
            distance=np.linalg.norm(delta,axis=1)
            radial=np.clip((1500-np.maximum(lower,distance))/np.maximum(1500-lower,1e-6),0,1)
            visible=radial*(.5*omni+.5*(directional&(delta@normals.T>=0)).mean(axis=1))
            prob=np.divide(visible,raw,out=np.zeros_like(raw),where=raw>0)
            positive_mass=weight*prob
            negative_mass=weight*(1-prob)
            # Expected posterior spread for bounded angle observations. The
            # triangular angular kernel corresponds to independent uniform
            # working errors; it is never used to infer the simulator phase.
            bearing=np.arctan2(-delta[:,1],-delta[:,0])
            diff=np.abs((bearing[:,None]-bearing[None,:]+math.pi)%math.tau-math.pi)
            kernel=np.maximum(1-diff/math.radians(2.),0)
            posterior=kernel*positive_mass[None,:]
            sums=posterior.sum(axis=1)
            means=np.divide(posterior@points,sums[:,None],out=np.zeros_like(points),where=sums[:,None]>0)
            squared=(points*points).sum(axis=1)
            variance=np.divide(posterior@squared,sums,out=np.zeros_like(sums),where=sums>0)-(means*means).sum(axis=1)
            spread=float(positive_mass@np.sqrt(np.maximum(variance,0)))
            if negative_mass.sum()>1e-9:
                mean=negative_mass@points/negative_mass.sum()
                spread+=float(negative_mass@np.linalg.norm(points-mean,axis=1))
            travel=np.linalg.norm(candidate-current)/5
            onward=float(weight@distance)/5
            score=travel+onward+self.information_penalty*spread/5+6
            item=(score,tuple(candidate))
            if best is None or item<best:
                best=item
        return best[1] if best else base
