"""Export compact Q4 comparison tables, retaining failures and truth references."""
import argparse
import csv
import json
from pathlib import Path
import statistics

STEM = 'Q4_实验组与对照组_策略比较表'


def equal_group_mean(cases, key):
    counts = sorted({c['source_count'] for c in cases})
    return statistics.mean(statistics.mean(c[key] for c in cases if c['source_count'] == n) for n in counts)


def summarize(report):
    rows = []
    for name, group in report['strategies'].items():
        cases = group['cases']
        complete = all(c['all_cleared'] for c in cases)
        privileged = group['privileged_truth']
        rows.append(dict(strategy=name, label=group['label'], controller=group['controller'],
            scene_count=len(cases), cleared_count=sum(c['cleared_count'] for c in cases),
            source_count=sum(c['source_count'] for c in cases),
            clearance_percent=100*equal_group_mean(cases, 'clearance_ratio'),
            complete_scenes=sum(c['all_cleared'] for c in cases),
            actual_complete_scenes=sum(c['actual_all_cleared'] for c in cases),
            all_audits_passed=all(c['audit_passed'] for c in cases),
            raw_mean_per_source_s=equal_group_mean(cases, 'raw_time_per_source_s'),
            valid_mean_per_source_s=equal_group_mean(cases, 'raw_time_per_source_s') if complete else None,
            mean_total_time_s=equal_group_mean(cases, 'total_time_s'),
            sum_total_time_s=sum(c['total_time_s'] for c in cases),
            mean_program_runtime_s=equal_group_mean(cases, 'program_runtime_s'),
            privileged_truth=privileged, ranking_eligible=complete and not privileged,
            by_source_count={str(n): dict(scene_count=len(selected),
                complete_scenes=sum(c['all_cleared'] for c in selected),
                raw_mean_per_source_s=statistics.mean(c['raw_time_per_source_s'] for c in selected))
                for n in sorted({c['source_count'] for c in cases})
                for selected in [[c for c in cases if c['source_count'] == n]]}))
    return rows


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_tables(comparison, output):
    report = json.loads(Path(comparison).read_text(encoding='utf-8'))
    if not report.get('same_scenes_verified'):
        raise ValueError('same-scene verification is required')
    rows = summarize(report)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    compact = dict(scenario=report['scenario'], evidence_scope=report['evidence_scope'],
        same_scenes_verified=True, rows=rows,
        cases={name: group['cases'] for name, group in report['strategies'].items()})
    (directory/f'{STEM}.json').write_text(json.dumps(compact, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    write_csv(directory/f'{STEM}.csv', [{k: v for k, v in row.items() if k != 'by_source_count'} for row in rows])
    lines = ['# Q4 实验组与对照组同场比较', '',
        '本表为本地合成离线实验。所有组使用完全相同的场景、定向源比例和测量误差场；场景源码与种子可复现。', '',
        f'配置：`{json.dumps(report["scenario"], ensure_ascii=False, sort_keys=True)}`。', '',
        '| 策略 | 实际清除率 | 平均时间（秒/源） | 总时间（秒/场） | 确认完成场景 |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        mark = '' if r['valid_mean_per_source_s'] is not None else '*'
        lines.append(f'| {r["label"]} | {r["clearance_percent"]:.2f}% | '
            f'{r["raw_mean_per_source_s"]:.2f}{mark} | {r["mean_total_time_s"]:.2f} | '
            f'{r["complete_scenes"]}/{r["scene_count"]} |')
    lines += ['', '平均时间按各源数组等权计算每场 T/N；T 包括移动、测量、切频、成功及失败清除的全部虚拟成本。'
        '实际清除率同样按源数组等权。* 含未完成场景，仅报告原始耗时，不参与全清除速度排名。'
        '实际清除率 100% 不自动代表已取得未知频道不存在其他源的证据。已知位置组使用特权真值，仅作参考，既不参与排名，也不是严格最优下界。', '']
    counts = sorted({int(n) for r in rows for n in r['by_source_count']})
    lines += ['| 策略 | '+' | '.join(f'{n} 源（秒/源）' for n in counts)+' |',
              '|---|'+'---:|'*len(counts)]
    for r in rows:
        cells = []
        for n in counts:
            g = r['by_source_count'][str(n)]
            mark = '*' if g['complete_scenes'] < g['scene_count'] else ''
            cells.append(f'{g["raw_mean_per_source_s"]:.2f}{mark}')
        lines.append('| '+r['label']+' | '+' | '.join(cells)+' |')
    lines += ['', '| 组别 | 独立算法入口 |', '|---|---|']
    for r in rows:
        source = r['controller'].split(':')[0].replace('.', '/')+'.py'
        lines.append(f'| {r["label"]} | [{Path(source).name}](../../{source}) |')
    lines += ['', 'Q3 对照的迁移定义、方向覆盖调整及复现命令见 [Q4 对照说明](../../src/q4/STRATEGY_CONTROLS.md)。'
        'JSON 保留逐场数值和场景指纹，CSV 可直接用 Excel 打开。未提交训练集、权重、搜索过程和原始动作归档。', '']
    (directory/f'{STEM}.md').write_text('\n'.join(lines), encoding='utf-8')
    for name, group in report['strategies'].items():
        write_csv(directory/f'Q4_{name}_逐场结果.csv', group['cases'])
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/tables'))
    rows = export_tables(**vars(parser.parse_args()))
    print(json.dumps(rows, ensure_ascii=False, indent=2))
