"""Truth-isolated benchmark for the final Q4 algorithm and controls.

Ordinary controllers get only SimulatorClient. Only the explicitly whitelisted
offline OracleController receives true positions. All groups share scene specs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from src.common.simulator_client import SimulatorClient
from src.q4.controller import build_summary
from src.q4.evaluate import snapshot_sources
from src.q4.offline_simulator import MixedSimulator,MixedSource,sample_sources
from src.q4.q3_adapter import cost_breakdown


class IndependentErrorSimulator(MixedSimulator):
    """A reproducible, position-fixed error field unrelated to the old sine."""
    def __init__(self,sources,*,noise_seed,**kwargs):
        super().__init__(sources,error_field='constant',quantize_bearings=True,**kwargs)
        self.noise_seed=noise_seed

    def transport(self,url,body,timeout_s):
        request=json.loads(body)
        if url.endswith('/measure'):
            p=request['position']
            # Normalize signed zero, so geometrically identical positions have
            # identical errors even if represented as -0.0 in a request.
            coords=[0. if float(p[k])==0 else float(p[k]) for k in ('x','y')]
            key=f'{self.noise_seed}|{request["channel"]}|{coords[0].hex()}|{coords[1].hex()}'
            value=int.from_bytes(hashlib.blake2b(key.encode(),digest_size=8).digest(),'big')
            self.bearing_error_deg=2*value/(2**64-1)-1
        return super().transport(url,body,timeout_s)


def sample_scene(seed,count,profile='uniform',fraction=.5):
    if profile not in {'uniform','outward','clusters'}:
        raise ValueError('unknown scene profile')
    sources=sample_sources(seed,count,fraction)
    if profile=='uniform':return sources
    rng=random.Random(seed+7919)
    transformed=[]
    cluster_angle=rng.uniform(0,2*math.pi)
    for index,s in enumerate(sources):
        if profile=='outward':
            angle=rng.uniform(0,2*math.pi)
            radius=rng.uniform(1650,1800)
            p=radius*math.cos(angle),radius*math.sin(angle)
            direction=math.degrees(angle)+rng.uniform(-15,15) if s.direction_deg is not None else None
            reception=rng.uniform(1000,1100)
        else:
            angle=cluster_angle+(math.pi if index%2 else 0)
            center=1100*math.cos(angle),1100*math.sin(angle)
            p=center[0]+rng.gauss(0,220),center[1]+rng.gauss(0,220)
            norm=math.hypot(*p)
            if norm>1799:p=p[0]*1799/norm,p[1]*1799/norm
            direction=s.direction_deg
            reception=s.receive_radius_m
        transformed.append(MixedSource(s.channel,p,reception,direction))
    return transformed


def make_world(sources,seed,noise):
    if noise=='hash':return IndependentErrorSimulator(sources,noise_seed=seed)
    if noise not in {'spatial','positive','negative'}:raise ValueError('invalid noise')
    return MixedSimulator(sources,bearing_error_deg=-1 if noise=='negative' else 1,
        error_field='spatial' if noise=='spatial' else 'constant',
        quantize_bearings=True,error_phase=seed*.61803398875)


def controller_class(spec):
    module,name=spec.split(':',1)
    if not module.startswith('src.q4.'):
        raise ValueError('benchmark expects a local src.q4 controller')
    return getattr(importlib.import_module(module),name)


def make_controller(cls, client, sources, options, case_index=0):
    """Keep privileged data in one explicit, offline-only construction branch."""
    from src.q4.control_oracle import OracleController
    from src.q4.control_random_walk import RandomWalkController
    kwargs = dict(options)
    if cls is OracleController:
        return cls(client, known_positions={s.channel: s.position for s in sources}, **kwargs)
    if cls is RandomWalkController:
        # This index and independent public seed are unrelated to source truth.
        kwargs['walk_seed'] = kwargs.get('walk_seed', 410000003) + case_index
    return cls(client, **kwargs)


def evaluate_benchmark(output,*,controller='src.q4.refined:RefinedController',options=None,
                      seed=230000000,cases_per_group=10,counts=(10,11,12,13,14,15,16),
                      profile='uniform',noise='hash',fraction=.5):
    if (not 1<=cases_per_group<1000 or not counts or len(set(counts))!=len(counts)
            or any(n not in range(10,17) for n in counts) or not 0<=fraction<=1):
        raise ValueError('invalid benchmark configuration')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    cls=controller_class(controller)
    configuration=dict(controller=controller,options=options or {},seed=seed,seed_stride=1000,
        cases_per_group=cases_per_group,counts=counts,profile=profile,noise=noise,fraction=fraction,
        source_sha256=snapshot_sources(output),evidence_scope='synthetic offline, not official')
    execution_options=dict(options or {})
    configuration['execution_options']=execution_options
    (output/'config.json').write_text(json.dumps(configuration,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for group,n in enumerate(counts):
        for case in range(cases_per_group):
            scene_seed=seed+1000*group+case
            sources=sample_scene(scene_seed,n,profile,fraction)
            world=make_world(sources,scene_seed,noise)
            client=SimulatorClient('research-q4',transport=world.transport)
            control=make_controller(cls,client,sources,execution_options,group*1000+case)
            error=None
            started=time.monotonic()
            try:
                client.enter()
                control.run()
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
            finally:
                if client.state.entered and not client.state.exited and client.pending_action is None:
                    try:client.exit()
                    except Exception as exc:error=error or f'exit {type(exc).__name__}: {exc}'
            runtime=time.monotonic()-started
            state=control.state
            cost=cost_breakdown(world.actions)
            audit=(world.cleared==state.cleared_channels and math.isclose(sum(cost.values()),world.virtual_time_s,abs_tol=1e-7)
                   and (not state.all_cleared or world.cleared==set(world.sources)))
            complete=error is None and audit and state.all_cleared and world.cleared==set(world.sources)
            row=dict(seed=scene_seed,source_count=n,cleared_count=len(world.cleared),
                source_sha256=hashlib.sha256(json.dumps([asdict(s) for s in sources],sort_keys=True).encode()).hexdigest(),
                actual_all_cleared=world.cleared==set(world.sources),
                clearance_ratio=len(world.cleared)/n, raw_time_per_source_s=world.virtual_time_s/n,
                privileged_truth=bool(getattr(state,'privileged_truth',False)),
                all_cleared=complete,audit_passed=audit,total_time_s=world.virtual_time_s,
                per_source_s=world.virtual_time_s/len(world.cleared) if world.cleared else None,
                program_runtime_s=runtime,full_scan_count=len(state.full_scan_stations),
                failure=error or state.failure_detail,termination_reason=state.termination_reason,**cost)
            payload=dict(case=row,sources=[asdict(s) for s in sources],
                summary=build_summary(state,world.virtual_time_s,runtime),actions=world.actions)
            (output/f'n{n}_seed{scene_seed}.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
            rows.append(row)
        group_rows=rows[-cases_per_group:]
        completed_group=all(r['all_cleared'] for r in group_rows)
        group_mean=(f'{statistics.mean(r["per_source_s"] for r in group_rows):.3f}'
                    if completed_group else 'NA (incomplete)')
        print(f'{controller} n={n} complete={sum(r["all_cleared"] for r in group_rows)}/{len(group_rows)} '
              f'mean={group_mean}',flush=True)
    complete=all(r['all_cleared'] for r in rows)
    report=dict(config=configuration,cases=rows,all_cases_cleared=complete,
        equal_weight_mean_per_source_s=statistics.mean(r['per_source_s'] for r in rows) if complete else None)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--controller',default='src.q4.refined:RefinedController')
    parser.add_argument('--options',type=json.loads)
    parser.add_argument('--seed',type=int,default=230000000)
    parser.add_argument('--cases-per-group',type=int,default=10)
    parser.add_argument('--counts',nargs='+',type=int,default=list(range(10,17)))
    parser.add_argument('--profile',choices=['uniform','outward','clusters'],default='uniform')
    parser.add_argument('--noise',choices=['hash','spatial','positive','negative'],default='hash')
    parser.add_argument('--fraction',type=float,default=.5)
    report=evaluate_benchmark(**vars(parser.parse_args()))
    print(json.dumps({k:v for k,v in report.items() if k not in {'config','cases'}},indent=2))
    raise SystemExit(0 if report['all_cases_cleared'] else 1)
