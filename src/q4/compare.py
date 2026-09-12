"""Run the final Q4 algorithm and named controls on identical synthetic scenes."""
import argparse
import json
from pathlib import Path
from src.q4.benchmark import evaluate_benchmark

STRATEGIES = {
    'informed': ('src.q4.informed:InformedController', {}),
    'refined': ('src.q4.refined:RefinedController', {}),
    'adaptive': ('src.q4.controller:MixedController', {'strategy': 'adaptive'}),
    'seven_grid': ('src.q4.controller:MixedController', {'strategy': 'seven_grid'}),
}


def compare(output, strategies=tuple(STRATEGIES), **scenario):
    if not strategies or len(set(strategies)) != len(strategies) or any(s not in STRATEGIES for s in strategies):
        raise ValueError('choose distinct known strategies')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name in strategies:
        cls, options = STRATEGIES[name]
        reports[name] = evaluate_benchmark(output/name, controller=cls, options=options, **scenario)
    report = dict(scenario=scenario, evidence_scope='synthetic offline; not official',
        strategies={name: dict(all_cleared=r['all_cases_cleared'],
            mean_per_source_s=r['equal_weight_mean_per_source_s'], cases=r['cases'])
            for name, r in reports.items()})
    (output/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({name: {k: v for k, v in r.items() if k != 'cases'}
                      for name, r in report['strategies'].items()}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument('--seed', type=int, default=260000000)
    parser.add_argument('--cases-per-group', type=int, default=10)
    parser.add_argument('--counts', type=int, nargs='+', default=list(range(10, 17)))
    parser.add_argument('--noise', choices=['hash', 'spatial', 'positive', 'negative'], default='hash')
    parser.add_argument('--profile', choices=['uniform', 'outward', 'clusters'], default='uniform')
    parser.add_argument('--fraction', type=float, default=.5)
    result = compare(**vars(parser.parse_args()))
    raise SystemExit(0 if all(r['all_cleared'] for r in result['strategies'].values()) else 1)
