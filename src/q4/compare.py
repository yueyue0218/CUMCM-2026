"""Run the final Q4 algorithm and named controls on identical synthetic scenes."""
import argparse
import json
from pathlib import Path
from src.q4.benchmark import evaluate_benchmark

STRATEGIES = {
    'informed': ('src.q4.experiment_informed:ExperimentInformedController', {}),
    'refined': ('src.q4.control_refined:ControlRefinedController', {}),
    'census_then_clear': ('src.q4.control_census_then_clear:CensusThenClearController', {}),
    'spiral_scan': ('src.q4.control_spiral_scan:SpiralScanController', {}),
    'random_walk': ('src.q4.control_random_walk:RandomWalkController', {}),
    'oracle': ('src.q4.control_oracle:OracleController', {}),
    'adaptive': ('src.q4.controller:MixedController', {'strategy': 'adaptive'}),
    'seven_grid': ('src.q4.controller:MixedController', {'strategy': 'seven_grid'}),
}
DEFAULT_STRATEGIES = ('informed', 'refined', 'census_then_clear', 'spiral_scan', 'random_walk', 'oracle')
LABELS = dict(informed='实验组：最终信息感知算法', refined='对照组：上一版 Q4 算法',
    census_then_clear='对照组：普查后清除', spiral_scan='对照组：螺旋扫描及方向覆盖补齐',
    random_walk='对照组：随机游走', oracle='特权参考：已知真实位置',
    adaptive='补充对照：初版自适应算法', seven_grid='补充对照：七站及网格后备')


def compare(output, strategies=DEFAULT_STRATEGIES, **scenario):
    if not strategies or len(set(strategies)) != len(strategies) or any(s not in STRATEGIES for s in strategies):
        raise ValueError('choose distinct known strategies')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name in strategies:
        cls, options = STRATEGIES[name]
        reports[name] = evaluate_benchmark(output/name, controller=cls, options=options, **scenario)
    report = dict(scenario=scenario, evidence_scope='synthetic offline; not official',
        strategies={name: dict(label=LABELS[name], controller=STRATEGIES[name][0],
            privileged_truth=name == 'oracle', ranking_eligible=r['all_cases_cleared'] and name != 'oracle',
            all_cleared=r['all_cases_cleared'],
            mean_per_source_s=r['equal_weight_mean_per_source_s'], cases=r['cases'])
            for name, r in reports.items()})
    inventories = [[(c['seed'], c['source_count'], c['source_sha256']) for c in r['cases']]
                   for r in reports.values()]
    if not all(rows == inventories[0] for rows in inventories):
        raise ValueError('comparison scene inventories differ')
    report['same_scenes_verified'] = True
    (output/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({name: {k: v for k, v in r.items() if k != 'cases'}
                      for name, r in report['strategies'].items()}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=list(DEFAULT_STRATEGIES))
    parser.add_argument('--seed', type=int, default=260000000)
    parser.add_argument('--cases-per-group', type=int, default=10)
    parser.add_argument('--counts', type=int, nargs='+', default=list(range(10, 17)))
    parser.add_argument('--noise', choices=['hash', 'spatial', 'positive', 'negative'], default='hash')
    parser.add_argument('--profile', choices=['uniform', 'outward', 'clusters'], default='uniform')
    parser.add_argument('--fraction', type=float, default=.5)
    result = compare(**vars(parser.parse_args()))
    raise SystemExit(0 if all(r['all_cleared'] for r in result['strategies'].values()) else 1)
