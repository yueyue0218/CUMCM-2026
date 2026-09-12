"""Q3 第二版与上一版在独立源数分层场景中的配对验证。"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if __package__ in {None,''}:sys.path.insert(0,str(ROOT))
from src.q3.experiment_joint_routing import EfficientState,run_efficient,build_efficient_summary
from src.q3.experiment_route_pool import run_route_pool,build_route_pool_summary
from src.q3.evaluate_efficiency import aggregate,cost_breakdown
from src.q3.offline_simulator import OfflineSimulator
from src.q3.training_env import sample_sources
from src.q3.validate_offline import source_hashes
from src.common.simulator_client import SimulatorClient

LABELS={'previous':'对照组_上一版联合路线','route_pool':'实验组_第二版路线候选池'}
STEM='Q3_自主迭代_实验组第二版与对照组上一版'


def compare_pairs(rows):
    groups=aggregate(rows,LABELS)
    valid=all(v['all_cases_cleared'] for v in groups.values())
    result={'valid':valid,'aggregate':groups,'mean_saved_per_source_s':None,
            'relative_reduction':None,'paired_bootstrap_95ci_saved_s':None,
            'improved_cases':0,'worse_cases':0,'tied_cases':0}
    if not valid:return result
    index={(r['source_count'],r['seed'],r['mode']):r for r in rows}
    if len(index)!=len(rows):raise ValueError('duplicate paired evaluation case')
    rng=np.random.default_rng(193017)
    boot=np.zeros(5000);means=[]
    for n in (10,12,14,16):
        old={r['seed']:r for r in rows if r['source_count']==n and r['mode']=='previous'}
        new={r['seed']:r for r in rows if r['source_count']==n and r['mode']=='route_pool'}
        if old.keys()!=new.keys():raise ValueError('paired seeds differ')
        delta=np.asarray([old[s]['per_source_s']-new[s]['per_source_s'] for s in sorted(old)])
        means.append(float(delta.mean()))
        boot+=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(axis=1)/4
        result['improved_cases']+=int((delta>1e-8).sum())
        result['worse_cases']+=int((delta<-1e-8).sum())
        result['tied_cases']+=int((np.abs(delta)<=1e-8).sum())
    result['mean_saved_per_source_s']=sum(means)/4
    result['relative_reduction']=result['mean_saved_per_source_s']/groups['previous']['equal_weight_mean_per_source_s']
    result['paired_bootstrap_95ci_saved_s']=[float(x) for x in np.quantile(boot,[.025,.975])]
    return result


def write_tables(report,directory):
    directory.mkdir(parents=True,exist_ok=True)
    (directory/f'{STEM}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (directory/f'{STEM}_逐场.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(report['cases'][0]));writer.writeheader();writer.writerows(report['cases'])
    summary=report['comparison'];groups=summary['aggregate']
    lines=['# Q3 自主迭代：第二版与上一版', '',
           f"10、12、14、16 源各 {report['cases_per_group']} 场，逐场相同源位置、接收半径与误差。数据为本地合成实验，非官方成绩。",'',
           '| 源数 | 上一版平均秒/源 | 第二版平均秒/源 | 上一版总秒/场 | 第二版总秒/场 | 上一版/第二版获证完成 |',
           '|---|---:|---:|---:|---:|---:|']
    def fmt(x):return '未全部完成' if x is None else f'{x:.2f}'
    for n in ('10','12','14','16'):
        a=groups['previous']['groups'][n];b=groups['route_pool']['groups'][n]
        lines.append(f"| {n} | {fmt(a['mean_per_source_s'])} | {fmt(b['mean_per_source_s'])} | {fmt(a['mean_total_time_s'])} | {fmt(b['mean_total_time_s'])} | {a['success_count']}/{a['case_count']}；{b['success_count']}/{b['case_count']} |")
    lines+=['',f"四组等权每源均值：上一版 {fmt(groups['previous']['equal_weight_mean_per_source_s'])} 秒，第二版 {fmt(groups['route_pool']['equal_weight_mean_per_source_s'])} 秒。"]
    if summary['valid']:
        ci=summary['paired_bootstrap_95ci_saved_s']
        lines += [f"平均节省 {summary['mean_saved_per_source_s']:.2f} 秒/源，降低 {summary['relative_reduction']:.2%}；逐场改善/变慢/持平 = {summary['improved_cases']}/{summary['worse_cases']}/{summary['tied_cases']}。",
                  f"分层配对重采样 5000 次，平均节省的 95% 区间为 [{ci[0]:.2f}, {ci[1]:.2f}] 秒/源。该区间仅描述当前合成分布的抽样波动。"]
    lines += ['', '平均时间先按每场总虚拟时间除实际源数，再对四种源数组等权平均。总时间包含移动、检测、切频和清除；实际程序运行时间另存。失败场景保留，任一场未获证完成则综合速度分数无效。',
              '完整 JSON 同时保存每组标准差、90分位、最坏值及配对统计。训练/开发集为80260912起的独立种子块；本次默认验收集为90260912起的新种子块，冻结后不以验收结果回调参数。',
              f"运行归档：`{report['run_directory']}`",'']
    (directory/f'{STEM}.md').write_text('\n'.join(lines),encoding='utf-8')


def evaluate(*,cases_per_group=20,seed=90260912,output_dir=ROOT/'results/tables',run_root=ROOT/'runs/q3/iteration'):
    if isinstance(cases_per_group,bool) or not isinstance(cases_per_group,int) or not 1<=cases_per_group<1000:
        raise ValueError('cases per group must be an integer in 1..999')
    seeds={seed+group*1000+case for group in range(4) for case in range(cases_per_group)}
    exposed={80260912+group*1000+case for group in range(4) for case in range(10)}
    if seeds&exposed:raise ValueError('evaluation seeds overlap development exposure')
    directory=Path(run_root)/datetime.now().astimezone().strftime('%Y%m%dT%H%M%S_%f%z_第二版与上一版独立验收')
    directory.mkdir(parents=True,exist_ok=False)
    metadata={'source_counts':[10,12,14,16],'cases_per_group':cases_per_group,'seed_start':seed,'seed_group_stride':1000,
              'development_seed_start':80260912,'development_cases_per_group':10,
              'source_sha256':source_hashes(),'strategies':LABELS,
              'environment':'synthetic fixed spatial error amplitude 1 degree, final 0.01 degree quantization',
              'evidence_scope':'independent local synthetic evaluation; not official results'}
    (directory/'运行配置.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(directory/'实际运行源码快照.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in metadata['source_sha256']:archive.write(ROOT/path,path)
    for label in LABELS.values():(directory/label).mkdir()
    rows=[]
    for group,n in enumerate((10,12,14,16)):
        for case in range(cases_per_group):
            scene_seed=seed+group*1000+case;sources=sample_sources(scene_seed,n)
            for mode,label in LABELS.items():
                world=OfflineSimulator(sources,bearing_error_deg=1,error_field='spatial',quantize_bearings=True,error_phase=scene_seed*.61803398875)
                client=SimulatorClient('iteration-independent',transport=world.transport)
                state=EfficientState();failure=exit_failure=None;start=time.monotonic()
                try:
                    client.enter()
                    (run_efficient if mode=='previous' else run_route_pool)(client,state=state)
                except Exception as error:
                    failure=f'{type(error).__name__}: {error}';state.termination_reason='error';state.failure_detail=failure
                finally:
                    if client.pending_action is None and client.state.entered and not client.state.exited:
                        try:client.exit()
                        except Exception as error:exit_failure=f'{type(error).__name__}: {error}'
                runtime=time.monotonic()-start
                costs=cost_breakdown(world.actions)
                if state.cleared_channels!=world.cleared or (state.all_cleared and world.cleared!=set(world.sources)):
                    failure='independent clearance audit failed'
                if not math.isclose(sum(costs.values()),client.state.virtual_time_s,rel_tol=1e-10,abs_tol=1e-6):
                    failure='independent virtual clock audit failed'
                successful=state.all_cleared and not failure and not exit_failure and client.pending_action is None
                row={'mode':mode,'strategy':label,'source_count':n,'seed':scene_seed,'all_cleared':successful,
                     'cleared_count':len(world.cleared),'clearance_ratio':len(world.cleared)/n,
                     'per_source_s':client.state.virtual_time_s/n if successful else None,
                     'total_time_s':client.state.virtual_time_s,'program_runtime_s':runtime,
                     'termination_reason':state.termination_reason,'failure':failure or state.failure_detail,
                     'exit_failure':exit_failure,**costs}
                rows.append(row)
                summary=(build_efficient_summary if mode=='previous' else build_route_pool_summary)(state,client.state.virtual_time_s,runtime)
                trace={'sources':[asdict(s) for s in sources],'audit':row,'summary':summary,
                       'pending_action':client.pending_action,'actions':world.actions}
                (directory/label/f'{label}_{n}源_种子{scene_seed}_逐场轨迹.json').write_text(json.dumps(trace,ensure_ascii=False)+'\n',encoding='utf-8')
            print(json.dumps({'source_count':n,'case':case+1,'old':rows[-2]['per_source_s'],'new':rows[-1]['per_source_s']}),flush=True)
    report={**metadata,'run_directory':str(directory),'cases':rows,'comparison':compare_pairs(rows)}
    write_tables(report,Path(output_dir));write_tables(report,directory)
    print(json.dumps(report['comparison'],ensure_ascii=False,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-per-group',type=int,default=20)
    parser.add_argument('--seed',type=int,default=90260912)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/tables')
    parser.add_argument('--run-root',type=Path,default=ROOT/'runs/q3/iteration')
    evaluate(**vars(parser.parse_args()))
