"""Build a seven-strategy table from audited historical and extension records."""
import hashlib,json,math,statistics,zipfile
from pathlib import Path
root=next(p for p in Path(__file__).resolve().parents if (p/'src/q3').is_dir())
old_path=root/'results/tables/Q3_实验组与对照组_策略比较表.json'
old=json.loads(old_path.read_text('utf-8'))
extension_root=root/'runs/q3/iteration/historical_extension_70260912'
new_path=next(extension_root.glob('*/Q3_自主迭代_实验组第二版与对照组上一版.json'))
new=json.loads(new_path.read_text('utf-8'))
assert new['seed_start']==old['seed_start']==70260912
assert new['cases_per_group']==old['cases_per_group']==20
assert new['comparison']['valid']
oldrows={(r['mode'],r['source_count'],r['seed']):r for r in old['cases']}
for row in new['cases']:
    if row['mode']=='previous':
        earlier=oldrows['joint_routing',row['source_count'],row['seed']]
        for key in ('total_time_s','cleared_count','clearance_ratio','per_source_s','movement_s','measurement_s','switching_s','clearance_s'):
            assert math.isclose(row[key],earlier[key],abs_tol=1e-8),(key,row)
scenes={};trace_count=0
for report in (old,new):
    for path in Path(report['run_directory']).rglob('*.json'):
        trace=json.loads(path.read_text('utf-8'))
        if 'sources' not in trace or 'actions' not in trace:continue
        row=trace['audit'];key=(row['source_count'],row['seed'])
        if key in scenes:assert scenes[key]==trace['sources']
        scenes[key]=trace['sources'];trace_count+=1
        # Independently recompute every new run's action ledger.
        if report is not new:continue
        position=(0,0);channel=1;time=0;cleared=set()
        for action in trace['actions']:
            if action['path'] not in ('/measure','/clear'):continue
            req=action['request'];res=action['response'];q=tuple(req['position'][k] for k in ('x','y'))
            time+=math.dist(position,q)/5;position=q
            if action['path']=='/measure':time+=5+(req['channel']!=channel);channel=req['channel']
            else:
                assert res['clear_result']=='success';time+=5;cleared.add(req['channel'])
            assert math.isclose(time,res['virtual_time_s'],abs_tol=1e-6)
        assert row['all_cleared'] and cleared=={s['channel'] for s in trace['sources']}
        assert not trace['pending_action'] and trace['actions'][-1]['path']=='/exit'
        assert math.isclose(time,row['total_time_s'],abs_tol=1e-6)
assert trace_count==640 and len(scenes)==80
with zipfile.ZipFile(Path(new['run_directory'])/'实际运行源码快照.zip') as archive:
    for name,digest in new['source_sha256'].items():
        saved=archive.read(name)
        assert hashlib.sha256(saved).hexdigest()==digest
        assert saved.replace(b'\r\n',b'\n')==(root/name).read_bytes().replace(b'\r\n',b'\n')
names={'route_pool':'实验组：第二版路线候选池','joint_routing':'对照组：上一版集中测量与联合路线',
       'census_then_clear':'对照组：普查后清除','spiral_scan':'对照组：螺旋扫描','random_walk':'对照组：随机游走',
       'oracle':'对照组：上帝视角','legacy_ppo_planner':'对照组：旧PPO辅助规划'}
rows=[]
for mode,name in names.items():
    selected=[r for r in (new['cases'] if mode=='route_pool' else old['cases']) if r['mode']==mode]
    def equal_mean(fn):return statistics.mean(statistics.mean(fn(r) for r in selected if r['source_count']==n) for n in (10,12,14,16))
    rows.append({'mode':mode,'strategy':name,'clearance_ratio':equal_mean(lambda r:r['clearance_ratio']),
                 'per_source_s':equal_mean(lambda r:r['total_time_s']/r['source_count']),
                 'total_time_s':equal_mean(lambda r:r['total_time_s']),
                 'success_count':sum(bool(r['all_cleared']) for r in selected),'case_count':len(selected),
                 'groups':{str(n):statistics.mean(r['total_time_s']/n for r in selected if r['source_count']==n) for n in (10,12,14,16)}})
lines=['| 策略及组别 | 清除比例 | 平均时间（秒/源） | 总时间（秒/场） | 获证全清除 |','|---|---:|---:|---:|---:|']
for r in rows:
    star='*' if r['success_count']<r['case_count'] else ''
    lines.append(f"| {r['strategy']} | {r['clearance_ratio']:.2%} | {r['per_source_s']:.2f}{star} | {r['total_time_s']:.2f} | {r['success_count']}/{r['case_count']} |")
table='\n'.join(lines)
groups=['| 策略及组别 | 10源秒/源 | 12源秒/源 | 14源秒/源 | 16源秒/源 |','|---|---:|---:|---:|---:|']
for r in rows:
    groups.append('| '+r['strategy']+' | '+' | '.join(f"{r['groups'][str(n)]:.2f}"+('*' if r['mode']=='random_walk' and n<16 else '') for n in (10,12,14,16))+' |')
record={'scope':'historical matched-scene extension; not a new independent holdout',
        'source_counts':[10,12,14,16],'cases_per_group':20,'seed_start':70260912,
        'historical_report':old_path.relative_to(root).as_posix(),'extension_report':new_path.relative_to(root).as_posix(),
        'source_scene_consistency':'all 640 historical/replay/extension traces verified, 80 unique scenes',
        'previous_replay':'80/80 costs and outcomes equal historical records',
        'extension_action_audit':'160/160 virtual clocks and clearance sets verified',
        'source_snapshot':'SHA-256 verified, implementation unchanged',
        'rows':rows}
stem=root/'results/tables/Q3_七策略同场比较_第二版实验组与六类对照组'
stem.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stem.with_suffix('.md').write_text('# Q3 第二版实验组与六类对照组同场比较\n\n70260912起的历史80场；第二版冻结后补跑，上一版重放80场与历史记录一致。此表用于同场策略比较，不替代90260912独立验收。\n\n'+table+'\n\n总时间为平均每场总虚拟时间，平均时间按10/12/14/16源四组等权。随机游走星号含未完成场景，只报告原始耗时，不计入获证全清除速度排名；上帝视角使用真值，不是严格最优下界。\n\n'+'\n'.join(groups)+'\n\n数据来自本地合成场景；来源和逐场核对见同名JSON。\n',encoding='utf-8')
(extension_root/'历史扩展范围与复核.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(extension_root/'README.md').write_text('# 历史同场扩展\n\n本目录使用已公开的70260912历史对照场景，不是新的独立验收集。子目录及自动报告中的“独立验收”字样来自复用评估入口的固定名称，应按本说明解释。算法已经冻结，未按本轮结果调参。六类旧策略保留历史记录，本轮重新执行第二版80场及上一版80场；后者用于确认历史结果可复现。七策略总表保存于results/tables/Q3_七策略同场比较_第二版实验组与六类对照组.md。\n',encoding='utf-8')
print(json.dumps({'rows':rows,'trace_count':trace_count},ensure_ascii=False))
