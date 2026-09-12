"""Paired held-out comparison of complete, planner, PPO and hybrid policies."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ''}:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from src.q3.policy_runtime import policy_configuration, rollout
from src.q3.ppo import load_policy


def assert_held_out(metadata, seed, cases):
    ranges = [
        (metadata['seed'], metadata.get('selection_exposure_episodes', metadata['episodes_completed'])),
        (metadata['warmup_seed_start'], metadata['warmup_episodes']),
        (metadata['validation_seed_start'], metadata.get('validation_exposure_cases', metadata['validation']['case_count'])),
    ]
    if any(max(seed, start) < min(seed+cases, start+count) for start, count in ranges):
        raise ValueError('evaluation seeds overlap PPO, imitation, or model-selection scenes')


def summarize_rows(rows, mode):
    selected = [r for r in rows if r['mode'] == mode]
    times = [r['virtual_time_s'] for r in selected if r['all_cleared']]
    improvements = [r['improvement_vs_complete_pct'] for r in selected
                    if r['improvement_vs_complete_pct'] is not None]
    return {'success_count': len(times), 'case_count': len(selected),
            'mean_virtual_time_s': float(np.mean(times)) if times else None,
            'worst_virtual_time_s': max(times) if times else None,
            'successful_pair_count': len(improvements),
            'mean_paired_improvement_pct': float(np.mean(improvements)) if improvements else None}


def evaluate(checkpoint, *, seed=40260912, cases=12, output_dir=ROOT/'results/tables'):
    if cases <= 0:
        raise ValueError('cases must be positive')
    torch.set_num_threads(1)
    policy = load_policy(checkpoint)
    config = policy_configuration(policy)
    metadata = policy.checkpoint_metadata
    # Legacy local checkpoints can recover split history from their run manifest.
    config_path = Path(checkpoint).parent/'config.json'
    if config_path.exists():
        recorded = json.loads(config_path.read_text('utf-8'))
        metadata.setdefault('warmup_seed_start', metadata['seed']+3000000000)
        metadata.setdefault('warmup_episodes', recorded['arguments']['warmup_episodes'])
    log_path = Path(checkpoint).parent/'training.jsonl'
    if log_path.exists():
        records = [json.loads(line) for line in log_path.read_text('utf-8').splitlines() if line.strip()]
        metadata['selection_exposure_episodes'] = max(
            [metadata.get('selection_exposure_episodes', metadata['episodes_completed'])]
            + [r.get('episodes_completed', 0) for r in records])
        metadata['validation_exposure_cases'] = max(
            [metadata.get('validation_exposure_cases', metadata['validation']['case_count'])]
            + [r['validation']['case_count'] for r in records if 'validation' in r])
    if 'warmup_episodes' not in metadata or 'warmup_seed_start' not in metadata:
        raise ValueError('checkpoint lacks complete training split provenance')
    assert_held_out(metadata, seed, cases)
    rows = []
    for case in range(cases):
        baseline_time = None
        baseline_success = False
        for mode in ('complete', 'planner', 'ppo', 'hybrid'):
            summary, _ = rollout(seed+case, mode=mode, policy=policy if mode in {'ppo','hybrid'} else None,
                                 environment_config=config)
            seconds = summary['final_virtual_time_s']
            if mode == 'complete':
                baseline_time = seconds
                baseline_success = summary['all_cleared']
            row = {'seed': seed+case, 'mode': mode, 'all_cleared': summary['all_cleared'],
                   'cleared_count': summary['cleared_count'], 'total_count': summary['total_count'],
                   'virtual_time_s': seconds, 'program_runtime_s': summary['program_runtime_s'],
                   'improvement_vs_complete_pct': ((baseline_time-seconds)/baseline_time*100
                       if baseline_success and summary['all_cleared'] else None),
                   'termination_reason': summary['termination_reason']}
            rows.append(row)
        print(json.dumps({'case': case+1, 'cases': cases, 'seed': seed+case}), flush=True)
    aggregate = {}
    for mode in ('complete','planner','ppo','hybrid'):
        aggregate[mode] = summarize_rows(rows, mode)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {'checkpoint': str(Path(checkpoint).resolve()), 'checkpoint_metadata': metadata,
              'test_seed_start': seed, 'evidence_scope': 'independent synthetic cases; not official results',
              'aggregate': aggregate, 'cases': rows}
    (output_dir/'q3_policy_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (output_dir/'q3_policy_comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(aggregate, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=40260912)
    parser.add_argument('--cases', type=int, default=12)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/tables')
    args = parser.parse_args()
    evaluate(args.checkpoint, seed=args.seed, cases=args.cases, output_dir=args.output_dir)
