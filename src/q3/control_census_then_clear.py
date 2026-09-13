"""对照组：七站各扫描全20频道，再按估计位置选择清除路线。"""
from src.q3.experiment_joint_routing import compact_coverage_points, EfficientController, EfficientState
from src.q3.control_scan_common import resolve_detected
from src.q3.coverage_control import covers_arena
from src.q3.search_clear import _StopRun


def run_census_then_clear(client, *, state=None, virtual_limit_s=360000., exit_reserve_s=20.):
    s = state if state is not None else EfficientState()
    controller = EfficientController(client, s, virtual_limit_s=virtual_limit_s, exit_reserve_s=exit_reserve_s)
    if s.termination_reason != 'not_started' or any(d.observations for d in s.channels.values()):
        raise ValueError('fresh state required')
    s.planned_coverage_points = compact_coverage_points()
    s.termination_reason = 'running'
    try:
        for index, point in enumerate(s.planned_coverage_points):
            for c in sorted(s.channels, reverse=client.state.current_channel > 10):
                controller.guard._measure(point, c)
                s.coverage_receipts[c].add(index)
            s.full_scan_stations.append(point)
            s.completed_coverage_points.append(point)
        if len(s.detected_channels) > s.source_count_upper:
            raise _StopRun('model_conflict', 'observed source count exceeds problem upper bound')
        if not covers_arena(s.full_scan_stations):
            raise _StopRun('incomplete_coverage', 'census stations did not certify arena coverage')
        for c in s.unknown_channels:
            negatives = {o.position for o in s.channels[c].observations if o.measure_result == 'no_signal'}
            if not set(s.full_scan_stations).issubset(negatives):
                raise _StopRun('incomplete_coverage', 'missing census negative receipts')
            s.channels[c].status = 'excluded_after_full_cover'
        s.completed_full_cover = True
        resolve_detected(controller)
        s.termination_reason = 'all_cleared' if s.all_cleared else 'incomplete'
    except _StopRun as error:
        s.termination_reason = error.reason; s.failure_detail = str(error)
    except Exception as error:
        s.termination_reason = 'error'; s.failure_detail = f'{type(error).__name__}: {error}'
        raise
    return s
