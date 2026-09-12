"""Q3 实验组与五个对照组：10/12/14/16 源同场比较及完整轨迹审计。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ''}:
    sys.path.insert(0, str(ROOT))

from src.common.simulator_client import SimulatorClient
from src.q3.offline_simulator import OfflineSimulator
from src.q3.training_env import sample_sources
from src.q3.experiment_joint_routing import EfficientState, run_efficient, build_efficient_summary
from src.q3.control_census_then_clear import run_census_then_clear
from src.q3.control_spiral_scan import run_spiral_scan
from src.q3.control_random_walk import run_random_walk
from src.q3.control_oracle import OracleState, run_oracle, build_oracle_summary
from src.q3.control_legacy_ppo_planner import run_legacy_ppo_planner
from src.q3.search_clear import SearchClearState, build_completion_summary
from src.q3.evaluate_efficiency import aggregate, cost_breakdown
from src.q3.validate_offline import source_hashes

STRATEGIES = {
    'joint_routing': {'stem': '实验组_当前策略_集中测量与联合路线', 'module': 'experiment_joint_routing.py'},
    'census_then_clear': {'stem': '对照组_普查后清除', 'module': 'control_census_then_clear.py'},
    'spiral_scan': {'stem': '对照组_螺旋扫描', 'module': 'control_spiral_scan.py'},
    'random_walk': {'stem': '对照组_随机游走', 'module': 'control_random_walk.py'},
    'oracle': {'stem': '对照组_上帝视角', 'module': 'control_oracle.py'},
    'legacy_ppo_planner': {'stem': '对照组_旧PPO辅助规划', 'module': 'control_legacy_ppo_planner.py'},
}
REPORT_STEM = 'Q3_实验组与对照组_策略比较表'


def run_strategy(mode, client, state, *, scene_seed, sources, policy, environment_config):
    # Ground truth crosses into a controller only in the explicitly privileged oracle branch.
    if mode == 'oracle':
        return run_oracle(client, state=state, positions={s.channel: s.position for s in sources})
    if mode == 'legacy_ppo_planner':
        return run_legacy_ppo_planner(client, state=state, policy=policy, environment_config=environment_config)
    if mode == 'random_walk':
        return run_random_walk(client, state=state, seed=scene_seed ^ 0x5EED5EED)
    runners = {'joint_routing': run_efficient, 'census_then_clear': run_census_then_clear,
               'spiral_scan': run_spiral_scan}
    return runners[mode](client, state=state)


def aggregate_controls(rows, modes):
    result = aggregate(rows, modes)
    for mode in modes:
        selected = [r for r in rows if r['mode'] == mode]
        info = result[mode]
        for count, group in info['groups'].items():
            cases = [r for r in selected if r['source_count'] == int(count)]
            group['clearance_ratio'] = statistics.mean(r['clearance_ratio'] for r in cases) if cases else None
            group['mean_observed_total_time_s'] = statistics.mean(r['total_time_s'] for r in cases) if cases else None
            group['mean_observed_per_source_s'] = statistics.mean(r['total_time_s']/r['source_count'] for r in cases) if cases else None
        info.update(STRATEGIES[mode])
        complete_groups = all(g['case_count'] for g in info['groups'].values())
        info['equal_weight_clearance_ratio'] = statistics.mean(g['clearance_ratio'] for g in info['groups'].values()) if complete_groups else None
        info['equal_weight_mean_total_time_s'] = statistics.mean(g['mean_observed_total_time_s'] for g in info['groups'].values()) if complete_groups else None
        info['equal_weight_observed_per_source_s'] = statistics.mean(g['mean_observed_per_source_s'] for g in info['groups'].values()) if complete_groups else None
        info['cumulative_total_time_s'] = sum(r['total_time_s'] for r in selected)
        info['success_count'] = sum(r['all_cleared'] for r in selected)
        info['case_count'] = len(selected)
    return result


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_reports(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/f'{REPORT_STEM}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    summary_rows = []
    group_rows = []
    for mode, info in report['aggregate'].items():
        summary_rows.append({'策略及组别': info['stem'], '清除比例': info['equal_weight_clearance_ratio'],
                             '平均每源虚拟时间_s': info['equal_weight_observed_per_source_s'],
                             '获证全清除的有效每源速度分数_s': info['equal_weight_mean_per_source_s'],
                             '平均每场总虚拟时间_s': info['equal_weight_mean_total_time_s'],
                             '全部场次累计虚拟时间_s': info['cumulative_total_time_s'],
                             '获证全清除场次': info['success_count'], '场次': info['case_count']})
        cases = [r for r in report['cases'] if r['mode'] == mode]
        write_csv(output_dir/f"Q3_{info['stem']}_逐场结果.csv", cases)
        group_report = {'strategy': info, 'config': {k: v for k,v in report.items() if k not in {'cases','aggregate'}}, 'cases': cases}
        (output_dir/f"Q3_{info['stem']}_结果.json").write_text(json.dumps(group_report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        for count, group in info['groups'].items():
            group_rows.append({'策略及组别': info['stem'], '源数': count, **group})
    write_csv(output_dir/f'{REPORT_STEM}.csv', summary_rows)
    write_csv(output_dir/'Q3_实验组与对照组_按源数分组结果.csv', group_rows)
    def seconds(value):
        return f'{value:.2f}' if value is not None else '未获证全部完成，不计速度排名'
    lines = ['# Q3 实验组与对照组比较', '',
             f"每种策略 {report['cases_per_group']*4} 场；10、12、14、16 源各 {report['cases_per_group']} 场，使用相同场景。",
             '清除比例先按每场真实成功清除数/真实源数计算，再对四种源数等权平均。平均时间是每源虚拟时间 T/N 的四组等权均值；总时间是每场总虚拟时间的四组等权均值。',
             '移动、检测、切频、清除均计入。表中实际耗时包含未完成场景，星号表示未获证全部完成；该行不计入全清除速度排名。有效速度成绩只在所有场景获证全清除且通过审计时生成，不删除失败场景。', '',
             '| 策略及组别 | 清除比例 | 平均时间（秒/源） | 总时间（秒/场） | 获证全清除 |',
             '|---|---:|---:|---:|---:|']
    for mode, info in report['aggregate'].items():
        flag = '' if info['all_cases_cleared'] else '*'
        lines.append(f"| {info['stem']} | {info['equal_weight_clearance_ratio']:.2%} | {seconds(info['equal_weight_observed_per_source_s'])}{flag} | {seconds(info['equal_weight_mean_total_time_s'])} | {info['success_count']}/{info['case_count']} |")
    lines += ['', '## 分源数的平均每源时间（秒/源）', '', '| 策略及组别 | 10源 | 12源 | 14源 | 16源 |', '|---|---:|---:|---:|---:|']
    for info in report['aggregate'].values():
        values = [seconds(g['mean_observed_per_source_s'])+('' if g['success_count'] == g['case_count'] else '*')
                  for g in info['groups'].values()]
        lines.append('| '+info['stem']+' | '+' | '.join(values)+' |')
    lines += ['', '上帝视角拥有真实坐标，使用多起点最近邻和 2-opt 开放路线，无检测成本；不是严格最优下界，也不是可部署的未知源策略。',
              '随机游走为 500 米步长、最多 128 步的有界版本，越界重抽方向，没有追加普查后备。螺旋扫描径向间距 1000 米、沿弧约 500 米测量，必要时扫完外圈。',
              '旧 PPO 使用原有 best.pt 及原 hybrid 配置，不重新训练。所有普通策略仅接收测量回执；扫描对照共用 Q1 定位认证和有限定位后备。',
              '这些结果来自本地合成场景，不是官方模拟器成绩。扫描参数事先固定，未针对这 80 场调优。', '',
              f"完整场景、动作回执、源码快照与模型哈希：`{report['run_directory']}`", '']
    (output_dir/f'{REPORT_STEM}.md').write_text('\n'.join(lines), encoding='utf-8')


def evaluate(*, cases_per_group=20, seed=70260912, checkpoint=None, modes=None,
             output_dir=ROOT/'results/tables', run_root=ROOT/'runs/q3/strategy_controls'):
    modes = list(STRATEGIES) if modes is None else list(modes)
    if not 1 <= cases_per_group < 1000:
        raise ValueError('cases per group must be in 1..999')
    if not modes or any(m not in STRATEGIES for m in modes) or len(set(modes)) != len(modes):
        raise ValueError('unsupported, empty or duplicated comparison modes')
    policy = None; config = {}
    checkpoint_hash = None
    if 'legacy_ppo_planner' in modes:
        if checkpoint is None:
            raise ValueError('old PPO control requires its saved checkpoint')
        import torch
        from src.q3.ppo import load_policy
        from src.q3.policy_runtime import policy_configuration
        from src.q3.evaluate_policy import assert_held_out
        torch.set_num_threads(1)
        policy = load_policy(checkpoint); config = policy_configuration(policy)
        for group in range(4):
            assert_held_out(policy.checkpoint_metadata, seed+group*1000, cases_per_group)
        checkpoint_hash = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    directory = Path(run_root)/datetime.now().astimezone().strftime('%Y%m%dT%H%M%S_%f%z_实验组与五类对照组')
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {'source_counts': [10,12,14,16], 'cases_per_group': cases_per_group, 'seed_start': seed,
                'seed_group_stride': 1000, 'modes': modes, 'strategies': {m: STRATEGIES[m] for m in modes},
                'checkpoint': str(checkpoint) if checkpoint else None, 'checkpoint_sha256': checkpoint_hash,
                'source_sha256': source_hashes(), 'legacy_ppo_environment_config': config,
                'environment': 'synthetic fixed spatial error amplitude 1 degree; final 0.01 degree quantization',
                'random_walk_seed': 'scene_seed XOR 0x5EED5EED', 'random_walk_max_steps': 128,
                'virtual_limit_s': 360000, 'evidence_scope': 'local synthetic comparison; not official results'}
    (directory/'Q3_实验组与对照组_运行配置.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with zipfile.ZipFile(directory/'Q3_实验组与对照组_源码快照.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in metadata['source_sha256']:
            archive.write(ROOT/path, path)
    for mode in modes:
        (directory/STRATEGIES[mode]['stem']).mkdir()
    rows = []
    for group, count in enumerate((10,12,14,16)):
        for case in range(cases_per_group):
            scene_seed = seed + group*1000 + case
            sources = sample_sources(scene_seed, count)
            for mode in modes:
                world = OfflineSimulator(sources, bearing_error_deg=1, error_field='spatial',
                                         quantize_bearings=True, error_phase=scene_seed*.61803398875)
                client = SimulatorClient('offline-strategy-controls', transport=world.transport)
                state = OracleState() if mode == 'oracle' else SearchClearState() if mode == 'legacy_ppo_planner' else EfficientState()
                failure = exit_failure = None
                started = time.monotonic()
                try:
                    client.enter()
                    run_strategy(mode, client, state, scene_seed=scene_seed, sources=sources,
                                 policy=policy, environment_config=config)
                except Exception as error:
                    failure = f'{type(error).__name__}: {error}'
                    state.termination_reason = 'error'; state.failure_detail = failure
                finally:
                    if client.pending_action is None and client.state.entered and not client.state.exited:
                        try: client.exit()
                        except Exception as error: exit_failure = f'{type(error).__name__}: {error}'
                runtime = time.monotonic()-started
                builder = build_oracle_summary if mode == 'oracle' else build_completion_summary if mode == 'legacy_ppo_planner' else build_efficient_summary
                summary = builder(state, client.state.virtual_time_s, runtime)
                summary.update(baseline_name=STRATEGIES[mode]['stem'], privileged_truth=mode == 'oracle')
                if mode in {'census_then_clear','spiral_scan','random_walk'}:
                    summary['coverage_plan_kind'] = f'{mode}_scan_stations_with_receipt_certification'
                if state.cleared_channels != world.cleared or (state.all_cleared and world.cleared != set(world.sources)):
                    failure = 'independent truth audit rejected clearance claims'
                costs = cost_breakdown(world.actions)
                if not math.isclose(sum(costs.values()), client.state.virtual_time_s, rel_tol=1e-10, abs_tol=1e-6):
                    failure = 'independent costs disagree with virtual clock'
                successful = state.all_cleared and not failure and not exit_failure and client.pending_action is None
                row = {'source_count': count, 'seed': scene_seed, 'mode': mode, 'strategy': STRATEGIES[mode]['stem'],
                       'all_cleared': successful, 'cleared_count': len(world.cleared), 'clearance_ratio': len(world.cleared)/count,
                       'total_time_s': client.state.virtual_time_s,
                       'per_source_s': client.state.virtual_time_s/count if successful else None,
                       'program_runtime_s': runtime, 'termination_reason': state.termination_reason,
                       'failure': failure or state.failure_detail, 'exit_failure': exit_failure, **costs}
                rows.append(row)
                trace = {'strategy': STRATEGIES[mode], 'source_count': count, 'seed': scene_seed,
                         'sources': [asdict(s) for s in sources], 'summary': summary, 'audit': row,
                         'failure': failure, 'exit_failure': exit_failure,
                         'pending_action': client.pending_action, 'actions': world.actions}
                stem = STRATEGIES[mode]['stem']
                (directory/stem/f'{stem}_{count}源_种子{scene_seed}_逐场轨迹.json').write_text(json.dumps(trace, ensure_ascii=False)+'\n', encoding='utf-8')
            print(json.dumps({'source_count': count, 'case': case+1, 'cases_per_group': cases_per_group,
                              'results': {r['mode']: round(r['total_time_s'], 2) for r in rows[-len(modes):]}}), flush=True)
    report = {**metadata, 'run_directory': str(directory), 'aggregate': aggregate_controls(rows, modes), 'cases': rows}
    write_reports(report, Path(output_dir))
    write_reports(report, directory)
    print(json.dumps(report['aggregate'], ensure_ascii=False, indent=2), flush=True)
    print(f'Run directory: {directory}', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-per-group', type=int, default=20)
    parser.add_argument('--seed', type=int, default=70260912)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--modes', nargs='+', choices=list(STRATEGIES))
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/tables')
    parser.add_argument('--run-root', type=Path, default=ROOT/'runs/q3/strategy_controls')
    evaluate(**vars(parser.parse_args()))
