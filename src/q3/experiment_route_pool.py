"""实验组第二版：完整覆盖路线候选池、实际补测点和近程辅助测量。"""
from src.q3.experiment_joint_routing import EfficientController,build_efficient_summary
from src.q3.coverage_route_pool import pooled_coverage_waypoints


class RoutePoolController(EfficientController):
    def route_position(self,c):
        # Plan the next sensing stop actually executed, including its lateral
        # offset, instead of treating an uncertain source estimate as a clear.
        return self.target(c)

    def plan_coverage(self,stations,targets,current):
        return pooled_coverage_waypoints(stations,targets,current)

    def auxiliary_measurement_worthwhile(self,distance,turn):
        # Targeted localization and very close measurements remain mandatory.
        # This 1000 m estimate filter is only an efficiency heuristic; it does
        # not certify reception, absence, localization, or clearance.
        return distance<1000 and turn>=12


def run_route_pool(client, *, state=None, virtual_limit_s=360000.,exit_reserve_s=20.):
    return RoutePoolController(client,state,virtual_limit_s=virtual_limit_s,exit_reserve_s=exit_reserve_s).run()


def build_route_pool_summary(state,virtual_time_s,runtime_s=0.):
    result=build_efficient_summary(state,virtual_time_s,runtime_s)
    result.update(baseline_name='实验组_第二版路线候选池',algorithm_version='route-pool-v2',
                  coverage_plan_kind='certified_route_pool_with_legacy_candidate',
                  route_pool_layouts=18,auxiliary_measurement_range_m=1000,
                  route_targets='actual_next_measurement_or_certified_clear_position')
    return result
