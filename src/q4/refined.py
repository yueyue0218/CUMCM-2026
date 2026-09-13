"""Measured Q4 efficiency variants; completion evidence stays in MixedController."""
from __future__ import annotations

import math

from src.q3.search_clear import ClearAttempt
from src.q3.localization_control import ChannelLocalizationDecision
from src.q4.controller import MixedController
from src.q4.coverage import coverage_route
from src.q4.coverage import covers_mixed, fallback_grid
from src.q4.q3_adapter import optimized_route


class RefinedController(MixedController):
    def __init__(self, client, state=None, *, finish_local=False, probe_radius_m=60.,
                 scan_spacing_m=1100., initial_baseline_m=500., future_scans=False,
                 task_order='route', source_priority_m=500.,
                 layout_mode='compact',
                 estimate_mode='hybrid',
                 scan_mode='spacing',
                 optical_cover_radius_m=0.,
                 baseline_full_scan=False,
                 baseline_mode='fixed',
                 route_mode='relocated',
                 **kwargs):
        if (not math.isfinite(probe_radius_m) or not 0 <= probe_radius_m <= 200
                or not math.isfinite(initial_baseline_m) or not 0 <= initial_baseline_m <= 1500):
            raise ValueError('invalid refinement parameter')
        super().__init__(client, state, scan_spacing_m=scan_spacing_m, **kwargs)
        self.finish_local = finish_local
        self.probe_radius_m = probe_radius_m
        self.initial_baseline_m = initial_baseline_m
        self.future_scans = future_scans
        if task_order not in {'route','nearest','source_near'}:
            raise ValueError('invalid task order')
        self.task_order = task_order
        self.source_priority_m = source_priority_m
        if layout_mode not in {'legacy','compact','rotated'}:
            raise ValueError('invalid layout mode')
        self.layout_mode = layout_mode
        self.layout = None
        if estimate_mode not in {'joint','q1','hybrid','weighted'}:
            raise ValueError('invalid estimate mode')
        self.estimate_mode = estimate_mode
        if scan_mode not in {'spacing','replacement'}:
            raise ValueError('invalid scan mode')
        self.scan_mode = scan_mode
        if not math.isfinite(optical_cover_radius_m) or not 0 <= optical_cover_radius_m <= 400:
            raise ValueError('invalid optical cover radius')
        self.optical_cover_radius_m=optical_cover_radius_m
        self.baseline_full_scan=baseline_full_scan
        if baseline_mode not in {'fixed','mean'} or route_mode not in {'original','relocated'}:
            raise ValueError('invalid baseline or route mode')
        self.baseline_mode=baseline_mode
        self.route_mode=route_mode
        self.probed = set()
        self.state.strategy_name=('q4-refined-batched-routing-v3' if route_mode=='relocated'
                                  else 'q4-refined-batched-routing-v2')
        self.state.refinement_config=dict(finish_local=finish_local,probe_radius_m=probe_radius_m,
            scan_spacing_m=scan_spacing_m,initial_baseline_m=initial_baseline_m,future_scans=future_scans,
            task_order=task_order,source_priority_m=source_priority_m,layout_mode=layout_mode,
            estimate_mode=estimate_mode,scan_mode=scan_mode,optical_cover_radius_m=optical_cover_radius_m,
            baseline_full_scan=baseline_full_scan,baseline_mode=baseline_mode,route_mode=route_mode)

    def initial_baseline_position(self):
        if self.baseline_mode=='fixed':
            return super().initial_baseline_position()
        positions=[self.estimate(c) for c in sorted(self.state.active_channels)]
        x=sum(p[0] for p in positions)
        y=sum(p[1] for p in positions)
        norm=math.hypot(x,y)
        if norm<=1e-6:
            return super().initial_baseline_position()
        return (self.initial_baseline_m*x/norm,self.initial_baseline_m*y/norm)

    def should_scan(self,position,force_scan=False):
        if not self.state.unknown_channels:
            return False
        if self.scan_mode == 'spacing' or force_scan or not self.state.full_scan_stations:
            return super().should_scan(position,force_scan)
        layout=self.layout
        if layout is None:
            layout=[(r*math.cos((i+phase)*2*math.pi/n),r*math.sin((i+phase)*2*math.pi/n))
                    for n,r,phase in ((10,975.,0),(12,1865.,.5)) for i in range(n)]
        stations=self.state.full_scan_stations
        if any(math.dist(position,p)<1e-6 for p in stations):
            return False
        remaining=[p for p in layout if p not in stations]
        for p in remaining:
            if math.dist(p,position)<850 and covers_mixed(stations+[tuple(position)]+[q for q in remaining if q!=p]):
                return True
        return False

    def estimate(self,c):
        if self.estimate_mode == 'joint':
            return super().estimate(c)
        e=self.evaluation(c)
        if e.decision is ChannelLocalizationDecision.READY_TO_CLEAR:
            return e.clear_position
        history=self.state.channels[c].observations
        if self.estimate_mode == 'q1' or (self.estimate_mode == 'hybrid'
                and not any(o.measure_result=='no_signal' for o in history)):
            return e.assessment.clear_position
        if self.estimate_mode == 'hybrid':
            return super().estimate(c)
        from src.q4.joint_model import sampled_center
        key=c,len(history),'weighted'
        if key not in self.joint_cache:
            self.joint_cache[key]=sampled_center(e.assessment.outer_region,history,integrate_radius=True)
        return self.joint_cache[key] or e.assessment.clear_position

    def optical_probe(self, c, position):
        """An explicit optical trial, never a claimed single-point certificate."""
        self.guard.check_budget(position, c, 'clear')
        result = self.client.clear(position, c)
        self.state.clear_attempts.append(ClearAttempt(c, position, result.result,
            self.client.state.virtual_time_s, 'opportunistic_optical_probe', None))
        self.probed.add((c, tuple(position)))
        if result.result == 'success':
            self.state.channels[c].status = 'cleared'
            return True
        return False

    def resolve(self, c, one_step=False):
        e = self.evaluation(c)
        if (e.decision is not ChannelLocalizationDecision.READY_TO_CLEAR
                and e.assessment.r_max_m <= max(self.probe_radius_m,self.optical_cover_radius_m)):
            p = self.estimate(c)
            if (c, tuple(p)) not in self.probed and self.optical_probe(c, p):
                self.batch(self.client.state.position)
                return
            if e.assessment.r_max_m <= self.optical_cover_radius_m:
                self.optical_fallback(c)
                self.batch(self.client.state.position)
                return
        super().resolve(c, one_step=one_step and not self.finish_local)

    def plan_coverage(self, tasks):
        if self.layout_mode != 'legacy':
            supplied = self.state.full_scan_stations+(list(tasks.values()) if self.future_scans else [])
            if covers_mixed(supplied):
                return []
            if self.layout is None:
                options=[]
                for rotation in (range(0,90,10) if self.layout_mode == 'rotated' else [0]):
                    points=[(r*math.cos((i+phase)*2*math.pi/n+math.radians(rotation)),
                             r*math.sin((i+phase)*2*math.pi/n+math.radians(rotation)))
                            for n,r,phase in ((8,998.,0),(12,1800/math.cos(math.pi/12)+3,0))
                            for i in range(n)]
                    if not covers_mixed([(0.,0.)]+points):
                        continue
                    planned={**tasks, **{21+i:p for i,p in enumerate(points)}}
                    order=optimized_route(self.client.state.position,planned)
                    path=[self.client.state.position]+[planned[c] for c in order]
                    options.append((sum(math.dist(a,b) for a,b in zip(path,path[1:])),rotation,points))
                self.layout=min(options)[2] if options else fallback_grid()
            chosen=[p for p in self.layout if p not in supplied]
            if not covers_mixed(supplied+chosen):
                chosen += [p for p in fallback_grid() if p not in supplied+chosen]
            for p in sorted(chosen,key=lambda p:min(math.dist(p,q) for q in supplied)):
                rest=list(chosen);rest.remove(p)
                if covers_mixed(supplied+rest):
                    chosen=rest
            return chosen
        return coverage_route(self.state.full_scan_stations, tasks if self.future_scans else {},
                              self.client.state.position, len(self.state.unknown_channels))

    def choose_task(self, tasks):
        position = self.client.state.position
        if self.task_order == 'nearest':
            return min(tasks, key=lambda c: (math.dist(position, tasks[c]), c))
        if self.task_order == 'source_near':
            near = [c for c in tasks if c <= 20 and math.dist(position, tasks[c]) <= self.source_priority_m]
            if near:
                return min(near, key=lambda c: (math.dist(position, tasks[c]), c))
        if self.route_mode=='relocated':
            from src.q4.route_refinement import relocated_route
            return relocated_route(position,tasks)[0]
        return optimized_route(position, tasks)[0]
