"""Reproducible, truth-isolated Q4 experiments (not official simulator results)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.common.simulator_client import SimulatorClient
from src.q4.q3_adapter import cost_breakdown
from src.q4.controller import MixedController, build_summary
from src.q4.offline_simulator import MixedSimulator, sample_sources

ROOT = Path(__file__).resolve().parents[2]


def snapshot_sources(directory):
    files = sorted((ROOT/'src/q4').glob('*.py'))+sorted((ROOT/'src/common').glob('*.py'))
    files += [ROOT/'src/q3'/name for name in ('baseline_scan.py','search_clear.py',
              'localization_control.py','offline_simulator.py')]
    files += [ROOT/'src/__init__.py',ROOT/'requirements.txt']
    with zipfile.ZipFile(directory/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, str(p.relative_to(ROOT)))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def evaluate(cases_per_group=5, seed=94260912, counts=(10,12,14,16), fraction=.5,
             strategy='informed', output_dir=None, controller_options=None):
    if (not 1 <= cases_per_group < 1000 or not counts or len(set(counts)) != len(counts)
            or any(n not in range(10,17) for n in counts) or not 0 <= fraction <= 1
            or strategy not in {'adaptive','seven_grid','refined','informed'}):
        raise ValueError('invalid evaluation configuration')
    directory = Path(output_dir) if output_dir else ROOT/'runs/q4/offline'/datetime.now().strftime('%Y%m%dT%H%M%S_%f')
    directory.mkdir(parents=True, exist_ok=False)
    config = dict(cases_per_group=cases_per_group, seed=seed, counts=counts,
                  directional_fraction=fraction, strategy=strategy,
                  controller_options=controller_options or {},
                  evidence_scope='synthetic offline; not official simulator',
                  seed_stride=1000, error='spatial amplitude 1 degree; final 0.01 degree quantization',
                  source_sha256=snapshot_sources(directory))
    (directory/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    for group, count in enumerate(counts):
        for case in range(cases_per_group):
            scene_seed = seed+1000*group+case
            sources = sample_sources(scene_seed, count, fraction)
            world = MixedSimulator(sources, bearing_error_deg=1, error_field='spatial',
                                   quantize_bearings=True, error_phase=scene_seed*.61803398875)
            client = SimulatorClient('offline-q4', transport=world.transport)
            if strategy == 'informed':
                from src.q4.informed import InformedController
                controller = InformedController(client, **(controller_options or {}))
            elif strategy == 'refined':
                from src.q4.refined import RefinedController
                controller = RefinedController(client, **(controller_options or {}))
            else:
                controller = MixedController(client, strategy=strategy, **(controller_options or {}))
            started = time.monotonic()
            failure = None
            try:
                client.enter()
                controller.run()
            except Exception as error:
                failure = f'{type(error).__name__}: {error}'
            finally:
                if client.pending_action is None and client.state.entered and not client.state.exited:
                    client.exit()
            runtime = time.monotonic()-started
            state = controller.state
            cost = cost_breakdown(world.actions)
            audit = (world.cleared == state.cleared_channels
                     and math.isclose(sum(cost.values()), world.virtual_time_s, abs_tol=1e-7)
                     and (not state.all_cleared or world.cleared == set(world.sources)))
            row = dict(seed=scene_seed, source_count=count, directional_count=sum(s.direction_deg is not None for s in sources),
                       cleared_count=len(world.cleared), clearance_ratio=len(world.cleared)/count,
                       all_cleared=state.all_cleared and audit and failure is None,
                       total_time_s=world.virtual_time_s,
                       per_source_s=world.virtual_time_s/len(world.cleared) if world.cleared else None,
                       program_runtime_s=runtime, fallback_count=len(state.fallback_channels),
                       full_scan_count=len(state.full_scan_stations),
                       termination_reason=state.termination_reason, failure=failure or state.failure_detail,
                       audit_passed=audit, **cost)
            payload = dict(case=row, sources=[asdict(s) for s in sources],
                           summary=build_summary(state, world.virtual_time_s, runtime), actions=world.actions)
            (directory/f'n{count}_seed{scene_seed}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            rows.append(row)
            print(f'n={count} seed={scene_seed} cleared={len(world.cleared)}/{count} '
                  f'time/source={row["per_source_s"]} runtime={runtime:.2f}s scans={row["full_scan_count"]}', flush=True)
    groups = {}
    for count in counts:
        selected = [r for r in rows if r['source_count'] == count]
        completed = all(r['all_cleared'] for r in selected)
        groups[str(count)] = dict(cases=len(selected), successes=sum(r['all_cleared'] for r in selected),
                                  mean_per_source_s=statistics.mean(r['per_source_s'] for r in selected) if completed else None,
                                  mean_runtime_s=statistics.mean(r['program_runtime_s'] for r in selected),
                                  clearance_ratio=statistics.mean(r['clearance_ratio'] for r in selected),
                                  worst_per_source_s=max(r['per_source_s'] for r in selected) if completed else None)
    complete = all(r['all_cleared'] for r in rows)
    score = statistics.mean(g['mean_per_source_s'] for g in groups.values()) if complete else None
    report = dict(config=config, groups=groups, all_cases_cleared=complete,
                  equal_weight_mean_per_source_s=score, target_below_400_met=score is not None and score < 400,
                  cases=rows, run_directory=str(directory))
    (directory/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (directory/'cases.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k:v for k,v in report.items() if k not in {'config','cases'}}, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-per-group', type=int, default=5)
    parser.add_argument('--seed', type=int, default=94260912)
    parser.add_argument('--counts', type=int, nargs='+', default=[10,12,14,16])
    parser.add_argument('--fraction', type=float, default=.5)
    parser.add_argument('--strategy', choices=['adaptive','seven_grid','refined','informed'], default='informed')
    parser.add_argument('--controller-options', type=json.loads)
    parser.add_argument('--output-dir', type=Path)
    evaluate(**vars(parser.parse_args()))
