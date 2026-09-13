"""Strictly replay a saved final-algorithm or control evaluation directory."""
import argparse
import inspect
import json
import math
from pathlib import Path
from src.common.simulator_client import SimulatorClient
from src.q4.benchmark import controller_class, make_world, make_controller
from src.q4.offline_simulator import MixedSource
from src.q4.q3_adapter import cost_breakdown


def trace(actions):
    return [(a['path'], a['request'].get('channel'), a['request'].get('position'), a['response'])
            for a in actions]


def replay(directory, output):
    directory, output = Path(directory), Path(output)
    config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
    paths = sorted(directory.glob('n*_seed*.json'))
    expected = {f'n{n}_seed{config["seed"]+g*1000+i}.json'
                for g, n in enumerate(config['counts']) for i in range(config['cases_per_group'])}
    if not paths or {p.name for p in paths} != expected:
        raise ValueError('missing or unexpected saved scenarios')
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding='utf-8'))
        cls = controller_class(config.get('controller') or data['summary']['controller_class'])
        allowed = {name for base in cls.__mro__ if base is not object
                   for name in inspect.signature(base.__init__).parameters}
        options = {k: v for k, v in (data['summary'].get('refinement_config') or {}).items() if k in allowed}
        options.update(config.get('options') or config.get('controller_options') or {})
        sources = [MixedSource(s['channel'], tuple(s['position']), s['receive_radius_m'], s['direction_deg'])
                   for s in data['sources']]
        world = make_world(sources, data['case']['seed'], config.get('noise', 'spatial'))
        client = SimulatorClient('offline-replay', transport=world.transport)
        case_index = data['case']['seed']-config['seed']
        control = make_controller(cls, client, sources, options, case_index)
        client.enter()
        state = control.run()
        client.exit()
        complete = state.all_cleared and world.cleared == set(world.sources)
        valid = (bool(complete) == bool(data['case']['all_cleared'])
                 and len(world.cleared) == data['case']['cleared_count']
                 and state.termination_reason == data['case']['termination_reason']
                 and trace(world.actions) == trace(data['actions'])
                 and world.virtual_time_s == data['case']['total_time_s']
                 and math.isclose(sum(cost_breakdown(world.actions).values()), world.virtual_time_s, abs_tol=1e-7))
        rows.append(dict(file=path.name, valid=bool(valid), all_cleared=bool(complete)))
    report = dict(scene_count=len(rows), all_identical=all(r['valid'] for r in rows), cases=rows)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    result = replay(**vars(parser.parse_args()))
    raise SystemExit(0 if result['all_identical'] else 1)
