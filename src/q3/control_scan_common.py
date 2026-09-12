"""Shared Q1 localization and receipt checks for the three scanning controls.

Only the scan schedule and clearance timing differ. No simulator truth is read.
"""
from src.q3.experiment_joint_routing import EfficientController, EfficientState, optimized_route
from src.q3.localization_control import ChannelLocalizationDecision
from src.q3.search_clear import _StopRun


def resolve_detected(controller):
    """Clear currently detected sources, without adding exploration waypoints."""
    s = controller.state
    while s.active_channels:
        targets = {c: controller.estimate(c) for c in sorted(s.active_channels)}
        c = optimized_route(controller.client.state.position, targets)[0]
        for _ in range(8):
            result = controller.evaluation(c)
            if result.decision is ChannelLocalizationDecision.READY_TO_CLEAR:
                controller.guard._clear(result.clear_position, c, 'q1' if result.assessment else 'near',
                                        result.assessment.r_max_m if result.assessment else 5.)
                break
            controller.guard._measure(controller.target(c), c)
        else:
            # Restore a real receiving station before starting the analytic bound.
            latest = next(o for o in reversed(s.channels[c].observations)
                          if o.measure_result in {'near', 'direction'})
            controller.guard._measure(latest.position, c)
            s.fallback_channels.append(c)
            controller.guard._localize_and_clear(c)


def run_scan_control(client, points, *, clear_during_scan, state=None,
                     virtual_limit_s=360000., exit_reserve_s=20.):
    s = state if state is not None else EfficientState()
    controller = EfficientController(client, s, virtual_limit_s=virtual_limit_s,
                                     exit_reserve_s=exit_reserve_s)
    if s.termination_reason != 'not_started' or any(d.observations for d in s.channels.values()):
        raise ValueError('fresh state required')
    s.planned_coverage_points = list(points)
    s.termination_reason = 'running'
    try:
        for point in s.planned_coverage_points:
            controller.batch(point, force_scan=True)
            if clear_during_scan:
                resolve_detected(controller)
                if s.all_cleared:
                    break
        if not clear_during_scan:
            resolve_detected(controller)
        if not s.all_cleared:
            raise _StopRun('scan_limit', 'scan schedule exhausted before absence/clearance proof completed')
        s.termination_reason = 'all_cleared'
    except _StopRun as error:
        s.termination_reason = error.reason
        s.failure_detail = str(error)
    except Exception as error:
        s.termination_reason = 'error'
        s.failure_detail = f'{type(error).__name__}: {error}'
        raise
    return s
