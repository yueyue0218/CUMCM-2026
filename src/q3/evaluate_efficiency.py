"""Equal-weight 10/12/14/16-source comparison on matched independent scenes."""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if __package__ in {None,''}:
    sys.path.insert(0,str(ROOT))

from src.common.simulator_client import SimulatorClient
from src.q3.offline_simulator import OfflineSimulator
from src.q3.training_env import sample_sources
from src.q3.experiment_joint_routing import EfficientState, run_efficient, build_efficient_summary
from src.q3.search_clear import SearchClearState, run_search_and_clear, build_completion_summary
from src.q3.validate_offline import source_hashes


def cost_breakdown(actions):
    position=(0.,0.)
    channel=1
    out={'movement_s':0.,'measurement_s':0.,'switching_s':0.,'clearance_s':0.}
    for action in actions:
        if action['path'] not in {'/measure','/clear'}:
            continue
        request=action['request'];response=action['response']
        target=(request['position']['x'],request['position']['y'])
        out['movement_s']+=math.dist(position,target)/5
        position=target
        if action['path']=='/measure':
            out['measurement_s']+=5
            out['switching_s']+=request['channel']!=channel
            channel=request['channel']
        else:
            out['clearance_s']+=5 if response['clear_result']=='success' else 3
    return out


def aggregate(rows, modes):
    result={}
    for mode in modes:
        groups={}
        for count in (10,12,14,16):
            selected=[r for r in rows if r['mode']==mode and r['source_count']==count]
            success=[r for r in selected if r['all_cleared']]
            averages=[r['per_source_s'] for r in success]
            ordered=sorted(averages)
            groups[str(count)]={
                'case_count':len(selected),'success_count':len(success),
                'mean_total_time_s':statistics.mean(r['total_time_s'] for r in success) if success else None,
                'mean_per_source_s':statistics.mean(averages) if averages else None,
                'std_per_source_s':statistics.stdev(averages) if len(averages)>1 else None,
                'p90_per_source_s':ordered[max(0,math.ceil(.9*len(ordered))-1)] if ordered else None,
                'worst_per_source_s':max(averages) if averages else None,
                'mean_program_runtime_s':statistics.mean(r['program_runtime_s'] for r in selected) if selected else None}
        valid=all(g['case_count']>0 and g['success_count']==g['case_count'] for g in groups.values())
        result[mode]={'groups':groups,'all_cases_cleared':valid,
                      'equal_weight_mean_per_source_s':statistics.mean(g['mean_per_source_s'] for g in groups.values()) if valid else None,
                      'metric':'mean of four source-count-group means of total virtual time / source count; valid only if every case completes'}
    return result


def evaluate(*,cases_per_group=20,seed=70260912,checkpoint=None,
             modes=('complete','hybrid','efficient'),output_dir=ROOT/'results/tables',run_root=ROOT/'runs/q3/efficiency'):
    if not 1<=cases_per_group<1000:
        raise ValueError('cases per group must be in 1..999 for disjoint seed blocks')
    if any(m not in {'complete','hybrid','efficient'} for m in modes) or len(set(modes))!=len(modes):
        raise ValueError('unsupported or duplicated comparison mode')
    policy=None;config={}
    if 'hybrid' in modes:
        if checkpoint is None:raise ValueError('hybrid comparison requires checkpoint')
        import torch
        from src.q3.ppo import load_policy
        from src.q3.policy_runtime import policy_configuration
        from src.q3.evaluate_policy import assert_held_out
        torch.set_num_threads(1)
        policy=load_policy(checkpoint);config=policy_configuration(policy)
        for group in range(4):assert_held_out(policy.checkpoint_metadata,seed+group*1000,cases_per_group)
    directory=Path(run_root)/datetime.now().astimezone().strftime('%Y%m%dT%H%M%S_%f%z_stratified')
    directory.mkdir(parents=True,exist_ok=False)
    metadata={'source_counts':[10,12,14,16],'cases_per_group':cases_per_group,'seed_start':seed,
              'seed_group_stride':1000,'modes':list(modes),'checkpoint':str(checkpoint) if checkpoint else None,
              'source_sha256':source_hashes(),'environment':'synthetic fixed spatial error, amplitude 1 degree; final 0.01-degree quantization',
              'hybrid_environment_config':config,'evidence_scope':'independent local synthetic evaluation; not official results'}
    (directory/'config.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(directory/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in metadata['source_sha256']:archive.write(ROOT/path,path)
    rows=[]
    for group,count in enumerate((10,12,14,16)):
        for case in range(cases_per_group):
            scene_seed=seed+group*1000+case
            sources=sample_sources(scene_seed,count)
            for mode in modes:
                world=OfflineSimulator(sources,bearing_error_deg=1,error_field='spatial',quantize_bearings=True,error_phase=scene_seed*.61803398875)
                client=SimulatorClient('offline-stratified',transport=world.transport)
                state=EfficientState() if mode=='efficient' else SearchClearState()
                failure=None;exit_failure=None
                started=time.monotonic()
                try:
                    client.enter()
                    if mode=='efficient':
                        run_efficient(client,state=state)
                    elif mode=='complete':
                        run_search_and_clear(client,state=state)
                    else:
                        from src.q3.policy_runtime import run_adaptive
                        run_adaptive(client,mode='hybrid',policy=policy,state=state,environment_config=config)
                except Exception as error:
                    failure=f'{type(error).__name__}: {error}'
                    state.termination_reason='error';state.failure_detail=failure
                finally:
                    if client.pending_action is None and client.state.entered and not client.state.exited:
                        try:client.exit()
                        except Exception as error:exit_failure=f'{type(error).__name__}: {error}'
                runtime=time.monotonic()-started
                summary=(build_efficient_summary if mode=='efficient' else build_completion_summary)(state,client.state.virtual_time_s,runtime)
                if state.all_cleared and world.cleared!=set(world.sources):
                    failure='false full-clearance claim detected by independent audit'
                costs=cost_breakdown(world.actions)
                if not math.isclose(sum(costs.values()),client.state.virtual_time_s,rel_tol=1e-10,abs_tol=1e-6):
                    failure='independent action costs disagree with virtual clock'
                successful=state.all_cleared and not failure and not exit_failure and client.pending_action is None
                row={'source_count':count,'seed':scene_seed,'mode':mode,'all_cleared':successful,
                     'cleared_count':len(state.cleared_channels),'total_time_s':client.state.virtual_time_s,
                     'per_source_s':client.state.virtual_time_s/count if successful else None,
                     'program_runtime_s':runtime,'termination_reason':state.termination_reason,
                     'failure':failure or state.failure_detail,'exit_failure':exit_failure,**costs}
                rows.append(row)
                trace={'source_count':count,'seed':scene_seed,'sources':[asdict(s) for s in sources],
                       'summary':summary,'failure':failure,'exit_failure':exit_failure,
                       'pending_action':client.pending_action,'actions':world.actions}
                (directory/f'n{count}_seed{scene_seed}_{mode}.json').write_text(json.dumps(trace,ensure_ascii=False)+'\n',encoding='utf-8')
            print(json.dumps({'source_count':count,'case':case+1,'cases_per_group':cases_per_group}),flush=True)
    report={**metadata,'run_directory':str(directory),'aggregate':aggregate(rows,modes),'cases':rows}
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    (directory/'report.json').write_text(encoded,encoding='utf-8')
    (output_dir/'q3_efficiency_stratified.json').write_text(encoded,encoding='utf-8')
    with (output_dir/'q3_efficiency_stratified.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(report['aggregate'],ensure_ascii=False,indent=2),flush=True)
    print(f'Run directory: {directory}',flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-per-group',type=int,default=20)
    parser.add_argument('--seed',type=int,default=70260912)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--modes',nargs='+',default=['complete','hybrid','efficient'])
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/tables')
    parser.add_argument('--run-root',type=Path,default=ROOT/'runs/q3/efficiency')
    evaluate(**vars(parser.parse_args()))
