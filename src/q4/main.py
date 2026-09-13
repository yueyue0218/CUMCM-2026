"""Run Q4 against an already ready official simulator session."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from src.common.simulator_client import JsonlRunLogger, SimulatorClient
from src.q4.controller import MixedController, build_summary

ROOT = Path(__file__).resolve().parents[2]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-id', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:2026')
    parser.add_argument('--run-root', type=Path, default=ROOT/'runs/q4/practice')
    parser.add_argument('--strategy', choices=['informed','refined','adaptive','seven_grid',
        'census_then_clear','spiral_scan','random_walk'], default='informed')
    parser.add_argument('--virtual-limit-s', type=float, default=360000.)
    parser.add_argument('--max-actions', type=int, default=4096)
    parser.add_argument('--max-requests', type=int, default=12300)
    parser.add_argument('--max-compute-s', type=float, default=900.)
    parser.add_argument('--route-mode',choices=['original','relocated'],default='relocated',
                        help='informed/refined route optimizer; refined + original reproduces v2')
    return parser


def run(args):
    directory = args.run_root/datetime.now().strftime('%Y%m%dT%H%M%S_%f')
    client = SimulatorClient(args.robot_id, base_url=args.base_url)
    if args.strategy == 'informed':
        from src.q4.experiment_informed import ExperimentInformedController
        controller_class = ExperimentInformedController
        base_strategy = 'adaptive'
    elif args.strategy == 'refined':
        from src.q4.control_refined import ControlRefinedController
        controller_class = ControlRefinedController
        base_strategy = 'adaptive'
    elif args.strategy in {'census_then_clear','spiral_scan','random_walk'}:
        from src.q4.benchmark import controller_class as load_controller
        from src.q4.compare import STRATEGIES
        controller_class = load_controller(STRATEGIES[args.strategy][0])
        base_strategy = 'adaptive'
    else:
        controller_class = MixedController
        base_strategy = args.strategy
    controller = controller_class(client, strategy=base_strategy, virtual_limit_s=args.virtual_limit_s,
                                 max_actions=args.max_actions, max_requests=args.max_requests,
                                 max_compute_s=args.max_compute_s,
                                 **({'route_mode':getattr(args,'route_mode','relocated')}
                                    if args.strategy in {'informed','refined'} else {}))
    client.logger = JsonlRunLogger(directory, {k:str(v) if isinstance(v, Path) else v
                                              for k,v in vars(args).items()})
    from src.q4.evaluate import snapshot_sources
    hashes = snapshot_sources(directory)
    (directory/'source_sha256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    started = time.monotonic()
    failure = exit_failure = None
    try:
        client.enter()
        controller.run()
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
    finally:
        if client.pending_action is None and client.state.entered and not client.state.exited:
            try:
                client.exit()
            except Exception as error:
                exit_failure = f'{type(error).__name__}: {error}'
        summary = build_summary(controller.state, client.state.virtual_time_s, time.monotonic()-started)
        summary.update(failure=failure, exit_failure=exit_failure,
                       pending_action=JsonlRunLogger._redact(client.pending_action),
                       strategy=args.strategy, evidence_scope='HTTP session; use official exported log for official results')
        (directory/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Q4 logs: {directory}')
        print(json.dumps({k:summary.get(k) for k in ('cleared_count','average_localization_clearance_time_s',
                                                    'termination_reason','program_runtime_s')}, ensure_ascii=False))
    return 0 if controller.state.all_cleared and failure is None and exit_failure is None and client.state.exited else 1


if __name__ == '__main__':
    raise SystemExit(run(build_parser().parse_args()))
