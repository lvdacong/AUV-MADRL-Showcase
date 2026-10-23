"""Refs #103: one fixed, paired four-decision collector audit; no optimizer."""
import copy
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from contextlib import ExitStack
from collections.abc import Mapping
from unittest.mock import patch

import psutil
import torch
from tensordict import TensorDictBase


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'Artifacts/GoalReach/viewer-review-terminal.json'


def digest(value):
    h = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().contiguous()
            h.update(str((x.dtype, tuple(x.shape))).encode()); h.update(x.numpy().tobytes())
        elif isinstance(x, (Mapping, TensorDictBase)):
            for key in sorted(x.keys()): h.update(str(key).encode()); add(x[key])
        elif isinstance(x, (list, tuple)):
            for item in x: add(item)
        else: h.update(repr(x).encode())
    add(value)
    return h.hexdigest()


def distribution(values):
    values = sorted(values)
    return dict(n=len(values), minimum=min(values), median=statistics.median(values),
                mean=statistics.mean(values), p95=values[int((len(values)-1)*.95)], maximum=max(values)) if values else dict(n=0)


def trace_events(path):
    """Parse the top-level object and stream its event array, including EOF/CRC."""
    decoder=json.JSONDecoder();buffer='';position=0;eof=False
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        def more():
            nonlocal buffer,position,eof
            chunk=stream.read(262144);buffer=buffer[position:]+chunk;position=0
            eof=not chunk
            return bool(chunk)
        def peek():
            nonlocal position
            while True:
                while position<len(buffer) and buffer[position].isspace():position+=1
                if position<len(buffer):return buffer[position]
                if eof or not more():raise ValueError('Unexpected trace EOF')
        def token(wanted):
            nonlocal position
            if peek()!=wanted:raise ValueError('Expected trace token '+wanted)
            position+=1
        def value():
            nonlocal position
            peek()
            while True:
                try:obj,end=decoder.raw_decode(buffer,position)
                except json.JSONDecodeError as error:
                    if eof or not more():raise ValueError('Invalid/truncated trace JSON') from error
                else:position=end;return obj
        token('{');seen=set()
        while peek()!='}':
            key=value()
            if not isinstance(key,str) or key in seen:raise ValueError('Invalid/duplicate trace field')
            seen.add(key);token(':')
            if key=='traceEvents':
                token('[')
                if peek()!=']':
                    while True:
                        event=value()
                        if not isinstance(event,dict):raise ValueError('Trace event must be object')
                        yield event
                        if peek()==']':break
                        token(',')
                token(']')
            else:value()
            if peek()=='}':break
            token(',')
            if peek()=='}':raise ValueError('Trailing trace comma')
        token('}')
        if 'traceEvents' not in seen:raise ValueError('Missing traceEvents')
        # Drain gzip to verify its trailer even when the JSON ended earlier.
        if buffer[position:].strip() or stream.read().strip():raise ValueError('Trailing trace content')


def summarize_trace(path, prefix='issue103::'):
    """Stream the original trace rather than loading hundreds of MB into RAM."""
    import math
    totals={};markers=[];synchronizations=[];kernels=[]
    for event in trace_events(path):
            if event.get('ph')!='X':continue
            if any(k not in event for k in ('cat','name','dur','ts','pid','tid')):raise ValueError('Missing complete-event fields')
            category=event['cat'];name=event['name'];duration=event['dur']
            if not isinstance(name,str) or not isinstance(category,str) or any(type(event[k]) not in (int,float) or not math.isfinite(event[k]) for k in ('ts','dur')) or duration<0:raise ValueError('Invalid complete-event fields')
            if category in ('kernel','gpu_memcpy','cuda_runtime','gpu_memset'):
                key=(category,name);row=totals.setdefault(key,dict(category=category,name=name,calls=0,microseconds=0))
                row['calls']+=1;row['microseconds']+=duration
            if name.startswith(prefix):markers.append(dict(name=name,ts=event['ts'],dur=duration,cat=category,pid=event['pid'],tid=event['tid']))
            if name in ('cudaStreamSynchronize','cudaDeviceSynchronize','cudaMemcpyAsync','aten::nonzero','aten::_local_scalar_dense'):
                synchronizations.append(dict(name=name,ts=event['ts'],dur=duration,cat=category,pid=event['pid'],tid=event['tid'],event_index=event.get('args',{}).get('Ev Idx')))
            if category=='kernel':kernels.append((event['ts'],event['ts']+duration))
    attributed={}
    for event in synchronizations:
        containing=[m for m in markers if m['cat']=='user_annotation' and event['cat'] in ('cpu_op','cuda_runtime') and m['pid']==event['pid'] and m['tid']==event['tid'] and m['ts']<=event['ts'] and event['ts']+event['dur']<=m['ts']+m['dur']+.01]
        parent=min(containing,key=lambda m:m['dur'])['name'] if containing else 'collector_outside_named_stage'
        key=(parent,event['cat'],event['pid'],event['tid'],event['name']);row=attributed.setdefault(key,dict(stage=parent,cat=event['cat'],pid=event['pid'],tid=event['tid'],name=event['name'],calls=0,microseconds=0,example=event))
        row['calls']+=1;row['microseconds']+=event['dur']
    union=0.;end=0.
    for start,stop in sorted(kernels):union+=max(0.,stop-max(start,end));end=max(end,stop)
    return dict(events=sorted(totals.values(),key=lambda x:x['microseconds'],reverse=True),
        innermost_stage_synchronizations=sorted(attributed.values(),key=lambda x:x['microseconds'],reverse=True),
        named_stage_intervals=markers,kernel_union_seconds=union/1e6,
        scope='Original trace categories; kernel union is this process across streams, not whole GPU utilization. CPU operator inclusive times overlap runtime calls.')


def active_run_identities():
    """Read current manifests, never require an old attempt to remain alive."""
    from auv_madrl import actor_training as at
    rows=[]
    for path in sorted((ROOT/'Artifacts/GoalReach').glob('*/run.json')):
        m=json.loads(path.read_text('utf-8'))
        if m.get('archived'):continue
        pid=m.get('pid');created=None;matches=False
        try:
            proc=psutil.Process(pid) if pid else None
            if proc:created=proc.create_time();matches=at.process_start_token(pid)==m.get('process_start_token')
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        rows.append(dict(output=str(path.parent),pid=pid,created=created,identity_matches=matches,
            status=m.get('status'),group=m.get('run_group_id'),attempt=m.get('launch_attempt',{}).get('id')))
    return rows


def logs_and_identity(recent=None):
    from auv_madrl import actor_training as at
    result = []
    for source in active_run_identities():
        output = Path(source['output']); manifest = json.loads((output/'run.json').read_text('utf-8'))
        if not (output/'metrics.jsonl').exists():continue
        raw = (output/'metrics.jsonl').read_bytes(); rows=[]; malformed=[]
        lines=list(enumerate(raw.splitlines(), 1))
        for n,line in reversed(lines) if recent is not None else lines:
            try:
                row=json.loads(line)
                if recent is not None and row.get('runtime',{}).get('launch_attempt_id')!=manifest['launch_attempt']['id']:continue
                rows.append(row)
                if recent is not None and len(rows)>=recent:break
            except ValueError: malformed.append(n)
        if recent is not None:rows.reverse()
        by_attempt={}
        for row in rows:
            runtime=row.get('runtime',{}); key=runtime.get('launch_attempt_id')
            by_attempt.setdefault(str(key),[]).append(row)
        attempts=[]
        for key,items in by_attempt.items():
            intervals=[]; gaps=[]
            for a,b in zip(items,items[1:]):
                ar,br=a.get('runtime',{}),b.get('runtime',{})
                if (key!='None' and ar.get('pid')==br.get('pid') and b['train/joint_global_update']==a['train/joint_global_update']+1
                    and 'update_started_monotonic' in ar and 'update_started_monotonic' in br):
                    elapsed=br['update_started_monotonic']-ar['update_started_monotonic']
                    if elapsed>0: intervals.append(dict(update=a['train/joint_global_update'],seconds=elapsed,
                        after_record_seconds=elapsed-a['time/update_seconds'],periodic_save_due=a['train/joint_global_update']%50==0))
                else:gaps.append([a['train/joint_global_update'],b['train/joint_global_update']])
            numeric={k:distribution([float(r[k]) for r in items if isinstance(r.get(k),(int,float))]) for k in items[-1]
                     if k.startswith(('time/','collect/'))}
            pg={k:distribution([r['parameter_groups'][k]['optimization_seconds'] for r in items]) for k in items[-1]['parameter_groups']}
            attempts.append(dict(attempt=key,pid=items[-1].get('runtime',{}).get('pid'),updates=[r['train/joint_global_update'] for r in items],
                start_to_start=distribution([r['seconds'] for r in intervals]),intervals=intervals,gaps=gaps,metrics=numeric,optimization_by_pg=pg,
                latest_roles=items[-1]['roles'],latest_parameter_groups=items[-1]['parameter_groups'],latest_runtime=items[-1].get('runtime'),
                environment_plus_reset_fraction=distribution([(r['time/collect_environment_step_seconds']+r['time/collect_reset_seconds'])/r['time/update_seconds'] for r in items]),
                update_minus_collect_optimization=distribution([r['time/update_seconds']-r['time/collect_seconds']-r['time/optimization_seconds'] for r in items])))
        result.append(dict(output=str(output),pid=source['pid'],created=source['created'],identity_matches=source['identity_matches'],status=source['status'],group=manifest['run_group_id'],attempt=manifest['launch_attempt']['id'],
            runtime=manifest['runtime'],launch_settings=manifest['launch_settings'],metrics_sha256=hashlib.sha256(raw).hexdigest(),malformed_lines=malformed,attempts=attempts))
    return result


def main():
    from auv_madrl import goal_reach as gr, training as tr, actor_training as at
    from auv_madrl.multi_goal_reach_env import MultiGoalReachEnv
    from auv_madrl.dynamics import BatchedAuvDynamics
    data=json.loads(EVIDENCE.read_text('utf-8'))
    assert not data.get('issue103',{}).get('batch_decisions',0), 'Consumed attempt cannot be repeated'
    e=dict(status='preparing',source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
           driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),batch_decisions=0,optimizer_steps=0,runs=[],logs=logs_and_identity())
    def save():
        d=json.loads(EVIDENCE.read_text('utf-8'));d['issue103']=e;at.write_json_atomic(EVIDENCE,d,compact=True)
    e['system']=dict(available_ram=psutil.virtual_memory().available,physical_ram=psutil.virtual_memory().total,
        power=psutil.sensors_battery()._asdict() if psutil.sensors_battery() else None,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv'],text=True),
        profiler_activities=[str(a) for a in torch.profiler.supported_activities()],torch=torch.__version__)
    settings=copy.deepcopy(e['logs'][1]['launch_settings']);e['settings']=settings
    assert settings['training_settings']['parallel_environments']==1024
    assert settings['training_settings']['rollout_decisions']==32
    assert settings['learner_count_range']==[2,2] and settings['defender_count_range']==[2,2] and settings['target_count_range']==[1,1]
    save()
    # Match this production contract's process-local threading; never touch workers.
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    tr.seed_everything(42)
    t=time.perf_counter();env,groups,runtimes,bindings,frozen=gr._build_training_group(settings,seed=42,device='cuda')
    torch.cuda.synchronize();e['initialization_seconds']=time.perf_counter()-t
    assert env.batch_size==1024 and env.config.shared['clock']['physics_step_s']==.02
    assert env.config.goal_reach['clock']['physics_ticks_per_decision']==25
    initial=copy.deepcopy(env.state_dict())
    fields=('observation','is_init','episode_returns','match_event_counts','agent_frames','update_index')
    runtime_initial={role:{key:copy.deepcopy(getattr(rt,key)) for key in fields} for role,rt in runtimes.items()}
    # Use the actual standard generators; saved only outside collection.
    import random, numpy as np
    def rng_state():return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state())
    def restore_rng(r):
        torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state(r['cuda']);random.setstate(r['python']);np.random.set_state(r['numpy'])
    rng=rng_state()
    state_path=Path(tempfile.gettempdir())/'auv103-collection-initial.pt'
    torch.save(dict(environment=initial,runtimes=runtime_initial,rng=rng,settings=settings),state_path)
    e['initial']=dict(path=str(state_path),file_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),environment=digest(initial),rng=digest(rng),
        parameters={key:digest(pg.model.state_dict()) for key,pg in groups.items()},bindings=bindings,
        observation_keys={key:dict(actor=list(pg.model.variant.actor_observation_keys),critic=list(pg.model.variant.critic_observation_keys)) for key,pg in groups.items()})
    activities=[torch.profiler.ProfilerActivity.CPU]
    if torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities():activities.append(torch.profiler.ProfilerActivity.CUDA)
    start_event,end_event=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start_event.record();end_event.record();torch.cuda.synchronize();e['zero_work_cuda_event_ms']=start_event.elapsed_time(end_event)
    # Verify restoration without consuming physics before either segment.
    env.load_state_dict(copy.deepcopy(initial));assert digest(env.state_dict())==e['initial']['environment']
    save()
    for mode in ('baseline','profile'):
        env.load_state_dict(copy.deepcopy(initial))
        for role,rt in runtimes.items():
            for key,value in runtime_initial[role].items():setattr(rt,key,copy.deepcopy(value))
        restore_rng(rng)
        row=dict(mode=mode,initial_environment=digest(env.state_dict()),initial_rng=digest(rng_state()),calls={},audit=[],reset=[],geometry=[],batch_decisions=0)
        e['runs'].append(row);save()
        audit=[];resets=[];geometry=[];lane_ticks=[];audit_events=[]
        def audit_copy(value):
            if isinstance(value,torch.Tensor):return value.detach().clone()
            if isinstance(value,dict):return {k:audit_copy(v) for k,v in value.items()}
            if isinstance(value,(tuple,list)):return [audit_copy(v) for v in value]
            return value
        def wrapped(owner,name):
            original=getattr(owner,name);descriptor=inspect.getattr_static(owner,name)
            def call(*args,**kwargs):
                row['calls'][name]=row['calls'].get(name,0)+1
                context=torch.profiler.record_function('issue103::'+name) if mode=='profile' else ExitStack()
                with context:
                    if name=='training_step_joint':
                        assert e['batch_decisions']<8
                        e['batch_decisions']+=1;row['batch_decisions']+=1
                        before=env.physics_ticks.clone()
                    result=original(*args,**kwargs)
                    if name in ('training_step_joint','reset','_static_contact_times','_local_obstacle_observation'):
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
                        if name=='training_step_joint':
                            lane_ticks.append(env.physics_ticks-before)
                            audit.append(dict(actions=audit_copy(args[1:]),target=audit_copy(kwargs.get('target_action')),
                                transitions={role:dict(reward=audit_copy(getattr(result,role).reward),terminated=audit_copy(getattr(result,role).terminated),
                                  info=audit_copy(getattr(result,role).info)) for role in runtimes}))
                        elif name=='reset':resets.append(audit_copy(kwargs.get('env_indices')))
                        elif name=='_static_contact_times':
                            safe=kwargs.get('safe');geometry.append(dict(kind='contacts',lanes=args[1].shape[0],K=args[3].shape[1],
                                safe_count=safe.sum().detach() if safe is not None else None,total=safe.numel() if safe is not None else None))
                        else:geometry.append(dict(kind='observation',K=env.obstacle_centers.shape[1],M=result[1].shape[-1]))
                        b.record();audit_events.append((a,b))
                    return result
            return patch.object(owner,name,staticmethod(call) if isinstance(descriptor,staticmethod) else call)
        profiler=None
        try:
            with ExitStack() as patches:
                for owner,names in ((MultiGoalReachEnv,('training_step_joint','_step_joint_match','_resolve_contacts','_obstacle_neighbor_safe','_static_contact_times','_local_obstacle_observation','reset')),
                                    (BatchedAuvDynamics,('step',)),(tr,('_sample_parameter_groups','_attach_gae'))):
                    for name in names:patches.enter_context(wrapped(owner,name))
                if mode=='profile':
                    profiler=patches.enter_context(torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False))
                torch.cuda.synchronize();started=time.perf_counter()
                rollouts,collected=tr.collect_joint_goal_reach(runtimes,rollout_length=4)
                torch.cuda.synchronize();row['outer_wall']=time.perf_counter()-started
            row['collected']=collected
            row['audit_device_elapsed_seconds']=sum(a.elapsed_time(b) for a,b in audit_events)/1000
            started=time.perf_counter()
            row['audit_digest']=digest(audit);row['reset_digest']=digest(resets);row['end_environment']=digest(env.state_dict());row['end_rng']=digest(rng_state())
            row['rollout_identity']={role:{key:digest(rollout[key]) for key in ('action','reward','terminated','done','is_init','actor_alive','credit_member')} for role,rollout in rollouts.items()}
            row['active_lane_ticks']=int(torch.stack(lane_ticks).sum().cpu())
            row['geometry']=[{k:(int(v.cpu()) if isinstance(v,torch.Tensor) else v) for k,v in item.items()} for item in geometry]
            row['reset_lanes']=[v.cpu().tolist() if isinstance(v,torch.Tensor) else v for v in resets]
            row['audit_transfer_and_hash_seconds']=time.perf_counter()-started
            row['memory']=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
            if profiler:
                row['profile']=[dict(name=x.key,calls=x.count,self_cpu_us=x.self_cpu_time_total,inclusive_cpu_us=x.cpu_time_total,
                    self_device_us=x.self_device_time_total,inclusive_device_us=x.device_time_total) for x in profiler.key_averages()]
                trace=Path(tempfile.gettempdir())/'auv103-collection-trace.json';profiler.export_chrome_trace(str(trace))
                target=ROOT/'Artifacts/GoalReach/issue103-collection-trace.json.gz'
                with trace.open('rb') as src,gzip.open(target,'wb') as dst:
                    import shutil;shutil.copyfileobj(src,dst)
                row['trace']=dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size)
            row['status']='complete';save();del rollouts
        except BaseException as error:
            row['error']=repr(error);row['status']='failed';e['status']='partial';save();raise
    a,b=e['runs'];e['equivalence']={key:a[key]==b[key] for key in ('initial_environment','initial_rng','audit_digest','reset_digest','end_environment','end_rng','rollout_identity')}
    e['optimizer_steps']=sum(pg.optimizer_steps for pg in groups.values());assert e['optimizer_steps']==0
    e['batch_physics_tick_slots']=e['batch_decisions']*25;e['lane_physics_tick_slots']=e['batch_physics_tick_slots']*1024
    e['workers_after']=active_run_identities()
    e['status']='complete' if all(e['equivalence'].values()) else 'partial: paired identities differ'
    save();print(json.dumps(dict(status=e['status'],batch_decisions=e['batch_decisions'],equivalence=e['equivalence'])),flush=True)


def profile_terrain_fields():
    """Refs #123: fixed existing block sets, 120 ordinary calls plus two trace calls, no physics."""
    import os, sys, types
    from collections import Counter
    from auv_madrl import terrain as candidate, goal_reach as gr, actor_training as at
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    data=json.loads(EVIDENCE.read_text('utf8'))
    e=data.setdefault('issue123',dict(start_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        native_thread=os.environ.get('CODEX_THREAD_ID'),calls=0,trace_calls=0,rows=[],traces=[],status='preflight'))
    def save():at.update_json_atomic(EVIDENCE,{'issue123':e},compact=True)
    assert not e['rows'],'Do not rerun a completed benchmark without resolving its identity and budget'
    source=subprocess.check_output(['git','show',e['start_sha']+':Python/src/auv_madrl/terrain.py']).decode('utf8')
    baseline=types.ModuleType('auv_madrl._terrain_before');sys.modules[baseline.__name__]=baseline
    exec(compile(source,'baseline/terrain.py','exec'),baseline.__dict__)
    module={'before':baseline,'after':candidate}
    e['workers_before']=active_run_identities()
    root=ROOT/'Artifacts/GoalReach/20260911-220651-592666600-unified-match-seed42'
    release=max(root.glob('checkpoint-[0-9]*.pt'))
    begin=time.perf_counter();_,_,payload,_=gr._read_group_release(release);state=payload['environment']
    e.update(release=str(release),release_sha256=hashlib.sha256(release.read_bytes()).hexdigest(),
        run_group_id=payload['run_group_id'],update=payload['global_update'],
        launch_attempt=json.loads((root/'run.json').read_text('utf8'))['launch_attempt'],
        torch=str(torch.__version__),device=torch.cuda.get_device_name(),threads=1,interop_threads=1,
        source_sha256={'before':hashlib.sha256(source.encode()).hexdigest(),'after':hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest()},
        initialization_wall=time.perf_counter()-begin,physics_ticks=0,optimizer=0)
    frozen={k:state[k].clone() for k in ('terrain_world_seed','terrain_episode_density','terrain_phase','terrain_block_lanes','terrain_block_coordinates')}
    e['inputs']={k:v.tolist() for k,v in frozen.items()};e['inputs_digest']=digest(frozen)
    cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state();save()
    try:
        for device in ('cpu','cuda'):
            for lanes in (1,min(32,state['batch_size']),state['batch_size']):
                mask=frozen['terrain_block_lanes']<lanes
                args=tuple(x.to(device) for x in (frozen['terrain_world_seed'][:lanes],frozen['terrain_episode_density'][:lanes],
                    frozen['terrain_block_lanes'][mask],frozen['terrain_block_coordinates'][mask],frozen['terrain_phase'][:lanes]))
                kwargs=dict(block_size_m=25.,obstacle_radius_range_m=(.5,3.))
                e['status']='measuring';save();expected=None
                for index in range(10):
                    for variant in (('before','after') if index%2==0 else ('after','before')):
                        assert e['calls']<168;e['calls']+=1
                        row=dict(device=device,lanes=lanes,variant=variant,index=index,warmup=index<2,blocks=args[2].numel(),status='running')
                        e['rows'].append(row)
                        mod=module[variant];rss=psutil.Process().memory_info()
                        if device=='cuda':
                            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                            start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
                            start_event.record()
                        begin=time.perf_counter();result=mod.generate_terrain(*args,**kwargs)
                        if device=='cuda':end_event.record();end_event.synchronize()
                        row['wall']=time.perf_counter()-begin
                        row['cuda_event_ms']=start_event.elapsed_time(end_event) if device=='cuda' else None
                        row['peak_allocated']=torch.cuda.max_memory_allocated() if device=='cuda' else None
                        row['process_rss_before']=rss.rss;row['process_peak_wset']=getattr(psutil.Process().memory_info(),'peak_wset',None)
                        row['obstacles']=result.obstacle_lanes.numel()
                        row['field_axis_bytes']=result.obstacle_lanes.numel()*6*8 if variant=='after' else None
                        row['digest']=digest(vars(result));row['status']='complete'
                        if expected is None:expected=row['digest']
                        assert row['digest']==expected
                        # Phase is a separate, non-nested measurement with the same boundary discipline.
                        if device=='cuda':torch.cuda.synchronize();start_event.record()
                        begin=time.perf_counter();phase=mod.terrain_phase(args[0],25.)
                        if device=='cuda':end_event.record();end_event.synchronize()
                        row['phase_wall']=time.perf_counter()-begin
                        row['phase_cuda_event_ms']=start_event.elapsed_time(end_event) if device=='cuda' else None
                        assert torch.equal(phase,args[4])
                        del result,phase
                    save()
                if device=='cuda' and lanes==state['batch_size']:
                    for variant in ('before','after'):
                        assert e['trace_calls']<2;e['trace_calls']+=1;save()
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],profile_memory=True) as prof:
                            result=module[variant].generate_terrain(*args,**kwargs);torch.cuda.synchronize()
                        with tempfile.TemporaryDirectory(prefix='auv-terrain-trace-') as tmp:
                            target=Path(tmp)/'trace.json';prof.export_chrome_trace(str(target));trace=json.loads(target.read_text('utf8'))
                            counts=Counter((x.get('cat',''),x.get('name','')) for x in trace['traceEvents'] if x.get('ph')=='X')
                            e['traces'].append(dict(variant=variant,counts=[dict(category=k[0],name=k[1],count=v) for k,v in counts.items()],digest=digest(vars(result))))
                        del result;save()
        assert digest(frozen)==e['inputs_digest']
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        e['summary']=[]
        for device in ('cpu','cuda'):
            for lanes in (1,min(32,state['batch_size']),state['batch_size']):
                values={variant:[r for r in e['rows'] if r['device']==device and r['lanes']==lanes and r['variant']==variant and not r['warmup']] for variant in module}
                medians={v:statistics.median(r['wall'] for r in rows) for v,rows in values.items()}
                e['summary'].append(dict(device=device,lanes=lanes,median_wall=medians,speedup=medians['before']/medians['after'],
                    median_phase_wall={v:statistics.median(r['phase_wall'] for r in rows) for v,rows in values.items()}))
        e['status']='complete';e['rng_unchanged']=True
    except BaseException as error:e.update(status='failed',error=repr(error));raise
    finally:
        sys.modules.pop(baseline.__name__,None);e['workers_after']=active_run_identities();e['cleanup_complete']=True;save()
        print(json.dumps({k:e.get(k) for k in ('status','calls','trace_calls','summary','error')},ensure_ascii=False),flush=True)


def collection_candidate_sample(variant, *, trace=False, issue=122):
    """Refs #122: one isolated fixed-source H32 or 2-warm/4-trace collector, no updates."""
    import gc, os, random, sys, types
    import numpy as np
    from auv_madrl import goal_reach as gr, goal_reach_env as ge, training as tr, actor_training as at
    from auv_madrl.dynamics import BatchedAuvDynamics
    assert issue in (122,124) and variant in (('B','F') if issue==124 else ('B0','T','F'))
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    data=json.loads(EVIDENCE.read_text('utf8'));e=data[f'issue{issue}'];pool=data['issue122']
    def save():at.update_json_atomic(EVIDENCE,{f'issue{issue}':e,'issue122':pool},compact=True)
    modules=[]
    def frozen_module(name, sha, file):
        source=subprocess.check_output(['git','show',sha+':Python/src/auv_madrl/'+file]).decode('utf8')
        module=types.ModuleType('auv_madrl.'+name);sys.modules[module.__name__]=module;modules.append(module)
        exec(compile(source,'baseline/'+file,'exec'),module.__dict__)
        return module,hashlib.sha256(source.encode()).hexdigest()
    cls=gr.MultiGoalReachEnv
    source_identity={'terrain':hashlib.sha256((ROOT/'Python/src/auv_madrl/terrain.py').read_bytes()).hexdigest()}
    if variant!='F':
        module,source_identity['multi_goal_reach_env']=frozen_module('_collection_'+variant,
            '62689986c9eae4cb1a05c16838b0c2869f823ebc' if variant=='B0' else e['start_sha'],'multi_goal_reach_env.py')
        cls=module.MultiGoalReachEnv
    else:source_identity['multi_goal_reach_env']=hashlib.sha256((ROOT/'Python/src/auv_madrl/multi_goal_reach_env.py').read_bytes()).hexdigest()
    if 'release' not in e:
        root=ROOT/'Artifacts/GoalReach/20260911-220651-592666600-unified-match-seed42'
        release=max(root.glob('checkpoint-[0-9]*.pt'))
        e['release']=str(release);e['release_sha256']=hashlib.sha256(release.read_bytes()).hexdigest();save()
    release=Path(e['release']);assert hashlib.sha256(release.read_bytes()).hexdigest()==e['release_sha256']
    _,_,payload,_=gr._read_group_release(release);settings=copy.deepcopy(payload['launch_settings'])
    assert settings['training_settings']['parallel_environments']==1024 and settings['training_settings']['rollout_decisions']==32
    row=dict(variant=variant,trace=trace,status='preparing',pid=os.getpid(),process_start_token=at.process_start_token(os.getpid()),
        source=source_identity,workers_before=active_run_identities(),lane_tick_slots=0,decision_calls=0,physics_calls=0,
        reserved_lane_ticks=(6 if trace else 12 if issue==124 else 32)*25*1024, issue=issue, decision_audit=trace or issue==122)
    assert pool['lane_tick_slots']+row['reserved_lane_ticks']+sum(r.get('reserved_lane_ticks',0) for r in e['samples'])<=pool['limit_lane_tick_slots']
    e['samples'].append(row);save();env=None;groups=None;runtimes=None
    def clone(value):
        if isinstance(value,torch.Tensor):return value.detach().clone()
        if isinstance(value,dict):return {k:clone(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [clone(v) for v in value]
        return value
    def rng():return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state())
    def device_state():
        derived = {'_collision_neighbor_counts', '_contact_inside', '_contact_pair_squared', '_contact_pair_limit'}
        return dict(tensors={k:v.detach().clone() for k,v in vars(env).items() if isinstance(v,torch.Tensor) and k not in derived},
                    roles={r:clone(vars(env.role_state(r))) for r in env.physical_roles})
    try:
        with ExitStack() as patches:
            patches.enter_context(patch.object(gr,'MultiGoalReachEnv',cls))
            if variant!='F' and issue==122:
                original_load=cls.load_state_dict
                def exact_load(self,state,**kwargs):
                    # Baseline restore repair only: grow before loading persisted contacts.
                    self.ensure_circle_storage(state['goal_active'].shape[1],state['obstacle_active'].shape[1])
                    return original_load(self,state,**kwargs)
                patches.enter_context(patch.object(cls,'load_state_dict',exact_load))
                row['baseline_restore_adapter']='grow circle storage before loading contacts; no physics changes'
            if variant=='B0':
                terrain,source_identity['terrain']=frozen_module('_collection_terrain_before','62689986c9eae4cb1a05c16838b0c2869f823ebc','terrain.py')
                for name in ('generate_terrain','terrain_phase','episode_density'):patches.enter_context(patch.object(ge,name,getattr(terrain,name)))
            begin=time.perf_counter();env,groups,runtimes,bindings,frozen=gr._build_training_group(settings,seed=payload['seed'],device='cuda',resume=payload)
            torch.cuda.synchronize();row['initialization_wall']=time.perf_counter()-begin
            row['initial']=dict(environment=digest(env.state_dict()),rng=digest(rng()),observations=digest({r:rt.observation for r,rt in runtimes.items()}),
                parameters={k:digest(pg.model.state_dict()) for k,pg in groups.items()},settings=settings,release_update=payload['global_update'])
            assert row['initial']['environment']==digest(payload['environment'])
            row['optimizer_initial']={k:pg.optimizer_steps for k,pg in groups.items()}
            audit=[];audit_events=[];geometry=[];stage_calls={};stages=[]
            def wrap(owner,name):
                original=getattr(owner,name)
                def call(*args,**kwargs):
                    stage_calls[name]=stage_calls.get(name,0)+1
                    if name=='training_step_joint':
                        charge=env.batch_size*int(env.config.goal_reach['clock']['physics_ticks_per_decision'])
                        assert row['lane_tick_slots']+charge<=row['reserved_lane_ticks']
                        row['lane_tick_slots']+=charge;row['decision_calls']+=1
                    if name=='_resolve_contacts':row['physics_calls']+=1
                    context=torch.profiler.record_function(f'issue{issue}::'+name) if tracing[0] else ExitStack()
                    with context:
                        result=original(*args,**kwargs)
                        if name=='training_step_joint' and not warming[0] and row['decision_audit']:
                            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
                            audit.append(dict(state=device_state(),step={r:clone(vars(getattr(result,r))) for r in runtimes}))
                            b.record();audit_events.append((a,b))
                        if tracing[0]:
                            if name in ('_static_contact_candidates','_joint_contact_times'):
                                minimum,narrow,index,mask,dense_rows,dense=result[:6]
                                geometry.append(dict(kind=name,R=minimum.shape[0],N=minimum.shape[1],W=index.shape[-1],D=dense_rows.numel(),
                                    K=env.obstacle_centers.shape[1],narrow_bytes=narrow.numel()*narrow.element_size(),dense_bytes=dense.numel()*dense.element_size(),safe_device=mask.any(-1).any(-1).sum().detach()))
                            elif name=='_static_contact_times':
                                geometry.append(dict(kind=name,R=result.shape[0],N=result.shape[1],K=result.shape[2],full_time_bytes=result.numel()*result.element_size()))
                            elif name in ('_assign_terrain','_repack_terrain_lanes','reset'):
                                indices=kwargs.get('env_indices',args[1] if len(args)>1 else None)
                                geometry.append(dict(kind=name,indices=clone(indices),K=env.obstacle_centers.shape[1]))
                    return result
                return patch.object(owner,name,staticmethod(call) if isinstance(inspect.getattr_static(owner,name),staticmethod) else call)
            tracing=[False];warming=[False]
            names=['training_step_joint','_resolve_contacts']
            if trace:names+=['_step_joint_match','_obstacle_neighbor_safe','_static_contact_times','_local_obstacle_observation','_refresh_terrain','_assign_terrain','_repack_terrain_lanes','reset']+(['_static_contact_candidates'] if hasattr(cls,'_static_contact_candidates') else [])
            if trace and issue==124:names += [n for n in ('_joint_contact_times','_swept_circle_time','_swept_circle_time_fields') if hasattr(cls,n)]
            for name in names:patches.enter_context(wrap(cls,name))
            if trace:
                for owner,name in ((BatchedAuvDynamics,'step'),(tr,'_sample_parameter_groups'),(tr,'_attach_gae'),
                    (ge,'required_blocks'),(ge,'generate_terrain'),(ge,'pack_obstacles'),(ge,'sample_actor_spawns')):
                    patches.enter_context(wrap(owner,name))
            # Check real serializers and CUDA timing before spending the first physical decision.
            json.dumps(row,allow_nan=False);zero_a=torch.cuda.Event(enable_timing=True);zero_b=torch.cuda.Event(enable_timing=True)
            zero_a.record();zero_b.record();zero_b.synchronize();row['zero_event_ms']=zero_a.elapsed_time(zero_b);save()
            if trace:
                warming[0]=True;warm=tr.collect_joint_goal_reach(runtimes,2);del warm;warming[0]=False
            profiler=None
            if trace:
                profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
                profiler.__enter__();tracing[0]=True
            try:
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
                rollouts,collected=tr.collect_joint_goal_reach(runtimes,4 if trace else 12 if issue==124 else 32)
                torch.cuda.synchronize();row['wall']=time.perf_counter()-begin
            finally:
                if profiler:profiler.__exit__(None,None,None);tracing[0]=False
            row.update(collected=collected,audit_event_seconds=sum(a.elapsed_time(b) for a,b in audit_events)/1000,
                decision_audits=[digest(item) for item in audit],end_environment=digest(env.state_dict()),end_rng=digest(rng()),
                final_observations=digest({r:rt.observation for r,rt in runtimes.items()}),
                rollout={r:{k:digest(v) for k,v in rollout.items()} for r,rollout in rollouts.items()},
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),calls=stage_calls)
            # Device diagnostics are transferred only after the measured segment.
            row['geometry']=[{k:(v.cpu().tolist() if isinstance(v,torch.Tensor) else v) for k,v in item.items()} for item in geometry]
            assert row['optimizer_initial']=={k:pg.optimizer_steps for k,pg in groups.items()}
            if profiler:
                row['profile']=[dict(name=x.key,calls=x.count,self_cpu_us=x.self_cpu_time_total,inclusive_cpu_us=x.cpu_time_total,
                    self_device_us=x.self_device_time_total,inclusive_device_us=x.device_time_total) for x in profiler.key_averages()]
                target=ROOT/f'Artifacts/GoalReach/issue{issue}-{variant}-collection-trace.json.gz'
                with tempfile.TemporaryDirectory(prefix='auv-collection-trace-') as tmp:
                    source=Path(tmp)/'trace.json';profiler.export_chrome_trace(str(source))
                    with source.open('rb') as src,gzip.open(target,'wb') as dst:
                        import shutil;shutil.copyfileobj(src,dst)
                row['trace_file']=dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size)
            row['status']='complete';del rollouts,audit
    except BaseException as error:row.update(status='failed',error=repr(error));raise
    finally:
        charged=row['lane_tick_slots']
        pool['lane_tick_slots']+=charged
        if issue!=122:e['lane_tick_slots']+=charged
        row['reserved_lane_ticks']=0
        for module in modules:sys.modules.pop(module.__name__,None)
        env=groups=runtimes=None;gc.collect();torch.cuda.empty_cache()
        row['workers_after']=active_run_identities();row['cleanup_complete']=True;save()
        print(json.dumps({k:row.get(k) for k in ('variant','trace','status','wall','decision_calls','lane_tick_slots','peak_allocated','error')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()


def periodic_collection_sample(label, *, issue=128, screen=None):
    """P1024/H32 fixed cold-input collector; no updates, profiler or stale reset observations."""
    import gc
    from auv_madrl import goal_reach as gr, training as tr, actor_training as at
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(42)
    settings_path=ROOT/'.git/periodic-collection-settings.json'
    if not settings_path.exists():
        settings=gr._validated_viewer_settings(json.loads((ROOT/'Artifacts/GoalReach/viewer-settings.json').read_text('utf8')))
        settings['world_map_mode']='auto'
        settings_path.write_text(json.dumps(settings),encoding='utf8')
    settings=json.loads(settings_path.read_text('utf8'))
    row=dict(label=label,settings=settings,workers_before=active_run_identities(),status='preparing',
             source={name:hashlib.sha256((ROOT/'Python/src/auv_madrl'/name).read_bytes()).hexdigest()
                     for name in ('terrain.py','goal_reach_env.py','multi_goal_reach_env.py','training.py')})
    def save():
        data=json.loads(EVIDENCE.read_text('utf8'));entry=data.get(f'issue{issue}',{})
        samples=entry.setdefault('collection_samples',[])
        if row not in samples:samples.append(row)
        at.update_json_atomic(EVIDENCE,{f'issue{issue}':entry},compact=True)
    try:
        start=time.perf_counter();env,groups,runtimes,bindings,frozen=gr._build_training_group(settings,device='cuda')
        torch.cuda.synchronize();row['initialization_s']=time.perf_counter()-start
        if screen is True:
            raise ValueError('Screening prototype rejected; reproduce using Git prototype 0eda8ef8')
        if screen is not None:
            env._decision_screen_enabled=lambda:screen
            env._decision_probe=[]
            row['screen_enabled']=screen
        # No reset between building runtime observations and the measured collector.
        row['initial']=dict(environment=digest(env.state_dict()),models={k:digest(v.model.state_dict()) for k,v in groups.items()},
                            rng=digest([torch.get_rng_state(),torch.cuda.get_rng_state()]),
                            observations=digest({k:v.observation for k,v in runtimes.items()}))
        row['bank_hash']=env.periodic_bank.asset_hash;row['P']=env.batch_size;row['H']=32
        row['bank_shape']=list(env.periodic_bank.centers.shape)
        start=time.perf_counter()
        for _ in range(3):
            for role in env.policy_roles: env.training_observation_for(role)
        torch.cuda.synchronize();row['observation_all_roles_s']=(time.perf_counter()-start)/3
        assert row['initial']['rng']==digest([torch.get_rng_state(),torch.cuda.get_rng_state()])
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        rollouts,collected=tr.collect_joint_goal_reach(runtimes,32)
        torch.cuda.synchronize();row['collection_h32_s']=time.perf_counter()-start
        row['collected']=collected;row['physical_width']=env.obstacle_capacity
        if screen is not None:row['decision_screen']=env._decision_probe
        row['peak_allocated']=torch.cuda.max_memory_allocated();row['peak_reserved']=torch.cuda.max_memory_reserved()
        row['final_environment']=digest(env.state_dict());row['rollout_hash']=digest({k:vars(v) for k,v in rollouts.items()})
        row['status']='complete'
        del rollouts,env,groups,runtimes,bindings,frozen;gc.collect();torch.cuda.empty_cache()
    except Exception as error:
        row.update(status='failed',error=repr(error));raise
    finally:
        row['workers_after']=active_run_identities();save()
        print(json.dumps({k:row.get(k) for k in ('status','label','initialization_s','observation_all_roles_s','collection_h32_s','physical_width','peak_allocated','peak_reserved','error')}),flush=True)
