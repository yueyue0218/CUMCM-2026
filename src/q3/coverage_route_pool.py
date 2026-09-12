"""Compare complete Q3 coverage routes instead of one greedy insertion layout."""
import math
from src.q3.coverage_control import covers_arena, coverage_waypoints
from src.q3.experiment_joint_routing import optimized_route


def coverage_route_cost(current, targets, stops):
    """Distance plus a fixed 150 m planning penalty per additional scan stop.

    This is a ranking score, not recorded virtual time or a clearance proof.
    """
    tasks={**targets,**{21+i:p for i,p in enumerate(stops)}}
    route=[current]+[tasks[k] for k in optimized_route(current,tasks)]
    return sum(math.dist(a,b) for a,b in zip(route,route[1:]))+150*len(stops)


def pooled_coverage_waypoints(stations, targets, current):
    """Choose among the legacy plan and 18 rotated/radially shifted layouts.

    Future targets participate only in hypothetical planning. Completion still
    requires actual station receipts in the controller's existing cover check.
    """
    supplied=list(stations)+list(targets.values())
    if covers_arena(supplied):return []
    plans=[coverage_waypoints(stations,targets,current)]
    route=[current]+[targets[k] for k in optimized_route(current,targets)]
    def insertion(p):
        return min([math.dist(route[-1],p)]
                   +[math.dist(a,p)+math.dist(p,b)-math.dist(a,b) for a,b in zip(route,route[1:])])
    for degrees in range(0,60,10):
        for radius in (1150,1350,1550):
            stops=[(0.,0.)]+[(radius*math.cos(math.radians(degrees+i*60)),
                             radius*math.sin(math.radians(degrees+i*60))) for i in range(6)]
            if not covers_arena(supplied+stops):continue
            for p in sorted(stops,key=insertion,reverse=True):
                rest=[x for x in stops if x!=p]
                if covers_arena(supplied+rest):stops=rest
            plans.append(stops)
    return min(plans,key=lambda stops:coverage_route_cost(current,targets,stops))
