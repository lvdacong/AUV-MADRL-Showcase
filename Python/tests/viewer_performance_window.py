"""One explicitly budgeted #77 window on the production Viewer scheduler.

Run with --condition baseline/default/context_off/far. The isolated HTTP page
exposes issue77.start()/stats(); the real Viewer and training are untouched.
"""
import argparse


class Finished(BaseException):
    pass


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--condition',choices=['baseline','default','context_off','far','ten','profile'],required=True)
    parser.add_argument('--issue',type=int,choices=[77,84,88,89,93,94,101,104,106],default=77)
    args=parser.parse_args()
    if args.issue == 106:
        assert args.condition == 'profile'
        return compare_owned_replay()
    if args.issue in (93,101,104):
        assert args.condition == 'profile'
        return profile_cpu_contacts(args.issue)
    import copy
    import gc
    import hashlib
    import json
    import os
    from pathlib import Path
    import socket
    import subprocess
    import tempfile
    import threading
    import time
    from unittest.mock import patch
    from http import HTTPStatus
    import psutil
    import torch
    from auv_madrl import goal_reach as gr

    root=Path(__file__).resolve().parents[2]
    path=root/'Artifacts/GoalReach/viewer-review-terminal.json'
    section=f'issue{args.issue}'
    limit_ticks=10125 if args.issue==94 else 14000 if args.issue in (84,88) else 34000
    maximum_windows=1 if args.issue==94 else 3 if args.issue in (84,88) else 6
    data=json.loads(path.read_text('utf-8'));e=data.setdefault(section,{})
    windows=e.setdefault('profiles' if args.condition=='profile' else 'windows',[])
    inherited=data['issue84'] if args.issue==88 else data['issue77'] if args.issue==89 else {}
    inherited_ticks=sum(w.get('actual_ticks',0) for k in ('windows','profiles') for w in inherited.get(k,[]))
    inherited_slots=len(inherited.get('windows',[]))
    if args.issue==88:
        assert args.condition=='default' and inherited_ticks==7048 and inherited_slots==2
    if args.issue==89:
        assert args.condition=='far' and inherited_ticks==6314 and inherited_slots==5
        assert data['issue88']['windows'][-1]['status']=='completed'
        e.setdefault('inputs',copy.deepcopy(data['issue88']['inputs']))
    if args.issue==94:
        assert args.condition=='far'
        if 'inputs' not in e:
            e['inputs']=copy.deepcopy(data['issue88']['inputs'])
            r=Path(data['issue92']['new_groups']['B']['output'])/gr.INITIAL_CHECKPOINT_NAME
            e['inputs'].update(release=str(r),release_sha256=hashlib.sha256(r.read_bytes()).hexdigest(),run_group_id=data['issue92']['new_groups']['B']['run_group_id'])
            for d in e['inputs']['deployment'].values():d['checkpoint']=str(r)
    profile_mode=args.condition=='profile'
    warmup_target=0 if profile_mode else 125
    if profile_mode:
        assert args.issue==84 and not windows
        assert any(w['condition']=='default' and w.get('actual_time_scale',0)<4.75 for w in e.get('windows',[]))
    import cProfile,pstats
    profiler=cProfile.Profile() if profile_mode else None
    assert inherited_slots+len(windows)<maximum_windows,'All window slots have been consumed'
    if args.issue==84:
        assert args.condition in {'baseline','default','ten','profile'}
        if args.condition=='ten':assert any(w.get('actual_time_scale',0)>=4.75 and w['condition']=='default' for w in windows)
    assert not any(w['condition']==args.condition and w.get('status')=='completed' for w in windows)
    prior=inherited_ticks+sum(w.get('actual_ticks',0) for key in ('windows','profiles') for w in e.get(key,[]))
    assert prior<limit_ticks
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if 'inputs' not in e:
        settings=gr._validated_viewer_settings(json.loads((root/'Artifacts/GoalReach/viewer-settings.json').read_text('utf-8')))
        b=Path(data['issue86']['new_groups']['B']['output']) if args.issue in (88,89,94) else root/('Artifacts/GoalReach/20260911-035857-284325100-unified-match-seed42' if args.issue==84 else 'Artifacts/GoalReach/20260911-011358-598329800-unified-match-seed42')
        manifest=json.loads((b/'run.json').read_text('utf-8'))
        update=manifest['checkpoint_update'];release=b/(f'checkpoint-{update:06d}.pt' if update else 'checkpoint-initial.pt')
        assert release.is_file()
        settings.update(policy_sampling='stochastic',time_scale=5.,snapshot_rate_hz=60.,main_view_zoom_step=0,main_view_frame='union',deployment_model_ids={r:'' for r in ('learner','defender','target')})
        e['inputs']=dict(settings=settings,release=str(release),release_sha256=hashlib.sha256(release.read_bytes()).hexdigest(),run_group_id=manifest['run_group_id'],update=update,policy_session_seed=42,initial_state=str(Path(tempfile.gettempdir())/f'auv{args.issue}-initial-state.pt'))
    inputs=e['inputs'];release=Path(inputs['release'])
    assert hashlib.sha256(release.read_bytes()).hexdigest()==inputs['release_sha256']
    settings=copy.deepcopy(inputs['settings'])
    if args.condition=='ten':settings['time_scale']=10.
    if args.condition=='far':settings.update(main_view_zoom_step=-16,main_view_frame='union_follow')
    row=dict(condition=args.condition,status='preparing',source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip(),pid=os.getpid(),actual_ticks=0,warmup_ticks=0,measurement_ticks=0)
    row['source_files']={name:hashlib.sha256((root/name).read_text('utf-8').encode('utf-8')).hexdigest() for name in ['Python/src/auv_madrl/goal_reach.py','Python/src/auv_madrl/live.py','Python/src/auv_madrl/multi_goal_reach_env.py','Python/src/auv_madrl/actor_training.py','Python/src/auv_madrl/viewer_replay.py','Python/src/auv_madrl/viewer_terrain.py','Python/viewer/goal_reach.html']}
    row['source_hash_normalization']='UTF8 universal newlines'
    row['source_dirty']=bool(subprocess.check_output(['git','diff','--','Python/src','Python/viewer/goal_reach.html'],cwd=root))
    windows.append(row)
    def save():
        from auv_madrl.run_evidence import write_json_atomic
        current=json.loads(path.read_text('utf-8'));current[section]=e
        write_json_atomic(path,current,compact=True)
    save()
    started=time.perf_counter();v=None
    owned_before=set(threading.enumerate());process=psutil.Process()
    row['training_before']=[]
    for folder in ([Path(r['output']) for r in data['issue92' if args.issue==94 else 'issue86']['new_groups'].values()] if args.issue in (88,89,94) else [root/'Artifacts/GoalReach/20260911-035848-972975800-unified-match-seed42',root/'Artifacts/GoalReach/20260911-035857-284325100-unified-match-seed42'] if args.issue==84 else [root/'Artifacts/GoalReach/20260911-011350-208558400-unified-match-seed42',root/'Artifacts/GoalReach/20260911-011358-598329800-unified-match-seed42']):
        m=json.loads((folder/'run.json').read_text('utf-8'));row['training_before'].append(dict(pid=m['pid'],created=psutil.Process(m['pid']).create_time(),update=m['global_update'],run_group_id=m['run_group_id'],launch_attempt=m['launch_attempt']))
    battery=psutil.sensors_battery();row['power']=None if battery is None else dict(plugged=battery.power_plugged,percent=battery.percent)
    row['temperature']='Unavailable through this Windows psutil runtime'
    begin=threading.Event();finish=threading.Event();end_requested=threading.Event()
    counter=dict(total=0,measured=0,started=None,ended=None,advance_wall=0.,pending_wall=0.,snapshots=0)
    hot_calls={}
    instrumentation=[]
    if args.issue==84 and args.condition!='baseline':
        from auv_madrl.multi_goal_reach_env import MultiGoalReachEnv
        from auv_madrl.viewer_replay import ViewerEpisodeRecord
        def watch(owner,name):
            original=getattr(owner,name)
            def measured(self,*a,**kw):
                active=counter['started'] is not None and counter['ended'] is None
                started_at=time.perf_counter()
                before=len(self.intervals) if name=='append' else 0
                try:return original(self,*a,**kw)
                finally:
                    if active:
                        entry=hot_calls.setdefault(name,dict(calls=0,wall=0.,new_intervals=0,maximum_record_bytes=0))
                        entry['calls']+=1;entry['wall']+=time.perf_counter()-started_at
                        if name=='append':
                            entry['new_intervals']+=len(self.intervals)-before
                            entry['maximum_record_bytes']=max(entry['maximum_record_bytes'],self.bytes)
            instrument=patch.object(owner,name,measured);instrument.start();instrumentation.append(instrument)
        for name in ('_resolve_contacts','_static_contact_times','_obstacle_neighbor_safe','_remember_obstacle_neighbors'):
            watch(MultiGoalReachEnv,name)
        watch(ViewerEpisodeRecord,'append')
    try:
        with tempfile.TemporaryDirectory(prefix='auv77-viewer-') as tmp:
            temp=Path(tmp)
            with patch.object(gr,'_load_viewer_settings',lambda config:(temp/'settings.json',copy.deepcopy(settings))):
                v=gr.GoalReachViewerSession()
            sources={};model_id='saved_target:'+release.relative_to(root).as_posix()
            deployments=[v._prepare_viewer_deployment(v._deployment_generations[role],role,model_id,v.active_actor_variant_id,'checkpoint',None,release,'manual',None,False,False,prepared_sources=sources) for role in v.env.policy_roles]
            v._install_viewer_deployments(deployments)
            v._auto_deploy_training_run_group_id=None;v._auto_deploy_training_sides.clear()
            assert v.device.type=='cpu' and not v._deployment_futures
            initial_path=Path(inputs['initial_state'])
            if not initial_path.exists():torch.save(copy.deepcopy(v.env.state_dict()),initial_path)
            initial=torch.load(initial_path,map_location='cpu',weights_only=False)
            if args.issue in (88,89,94):
                import random,numpy as np
                rng_path=initial_path.with_name(initial_path.stem+'-rng.pt')
                if not rng_path.exists():torch.save(dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state()),rng_path)
                rng=torch.load(rng_path,map_location='cpu',weights_only=False)
                torch.set_rng_state(rng['torch']);random.setstate(rng['python']);np.random.set_state(rng['numpy'])
                inputs['initial_state_sha256']=hashlib.sha256(initial_path.read_bytes()).hexdigest()
                inputs['initial_rng']=str(rng_path);inputs['initial_rng_sha256']=hashlib.sha256(rng_path.read_bytes()).hexdigest()
            v._viewer_policy_session_seed=inputs['policy_session_seed']
            v._reset(episode_index=int(initial['episode_counters'][0]),initial_state=copy.deepcopy(initial))
            v._environment_update_future.result(timeout=60);v._poll_environment_update()
            assert v.reset_progress['phase']=='ready'
            row['initial_digest']=gr.physical_digest(v.env).hex()
            row['policy_seed']=str(v._policy_noise.seed)
            row['deployment']={r:{k:d.get(k) for k in ['checkpoint','update','actor_hash','following']} for r,d in ((r,v._role_deployment_descriptor(r)) for r in v.role_controls)}
            if 'initial_digest' not in inputs:inputs['initial_digest']=row['initial_digest'];inputs['policy_seed']=row['policy_seed'];inputs['deployment']=row['deployment']
            assert row['initial_digest']==inputs['initial_digest'] and row['policy_seed']==inputs['policy_seed'] and row['deployment']==inputs['deployment']
            original_advance=v._advance_viewer_physics;original_drain=v._drain_commands;original_publish=v._publish_snapshot;original_submit=v.submit
            if args.issue in (88,89,94):
                original_store=v.snapshot.store
                flag_state={'last':None,'samples':0,'intervals':[]}
                def store_snapshot(payload,**kwargs):
                    viewer=payload['goal_reach_viewer']
                    assert type(viewer['time_scale_limited']) is bool
                    assert all(k in viewer for k in ('command_error','checkpoint_refresh_error','reset'))
                    reset=viewer['reset']
                    flags=(bool(v._viewer_fault),viewer['command_error'],viewer['checkpoint_refresh_error'],reset.get('error') or (reset.get('phase') if reset.get('phase')=='failed' else None),viewer['time_scale_limited'])
                    row['flag_fields_verified']=True;flag_state['last']=flags
                    original_store(payload,**kwargs)
                    if counter['started'] is not None and counter['ended'] is None:
                        now=time.perf_counter()-counter['started'];counter['snapshots']+=1;flag_state['samples']+=1
                        intervals=flag_state['intervals']
                        if not intervals or tuple(intervals[-1]['flags'])!=flags:intervals.append(dict(start=now,end=now,flags=flags,samples=1))
                        else:intervals[-1]['end']=now;intervals[-1]['samples']+=1
                v.snapshot.store=store_snapshot
            last_drain=time.perf_counter();pending=False
            def advance(ticks,deadline=None):
                if counter['ended'] is not None:v.running=False;return None
                if counter['started'] is not None:
                    stop_at=counter['started']+20
                    if time.perf_counter()>=stop_at:v.running=False;return None
                    deadline=min(stop_at,deadline) if deadline is not None else stop_at
                available=limit_ticks-prior-counter['total']
                if counter['started'] is None:available=min(available,warmup_target-counter['total'])
                if profile_mode:available=min(available,500-counter['total'])
                if available<=0:
                    v.running=False
                    if prior+counter['total']>=limit_ticks:raise Finished('physics budget exhausted')
                    return None
                before=int(v.env.physics_ticks[0]);at=time.perf_counter()
                result=original_advance(min(ticks,available),deadline=deadline)
                actual=int(v.env.physics_ticks[0])-before
                counter['total']+=actual
                if counter['started'] is not None:counter['measured']+=actual;counter['advance_wall']+=time.perf_counter()-at
                if counter['started'] is None and counter['total']>=warmup_target:v.running=False
                return result
            def submit(command):
                if command.get('type')=='issue77_start':begin.set();return
                if command.get('type')=='issue77_finish':end_requested.set();return
                return original_submit(command)
            def publish(*a,**kw):
                original_publish(*a,**kw)
                if args.issue not in (88,89,94) and counter['started'] is not None and counter['ended'] is None:counter['snapshots']+=1
            def drain():
                nonlocal last_drain,pending
                now=time.perf_counter()
                if counter['started'] is not None and counter['ended'] is None and pending:counter['pending_wall']+=now-last_drain
                last_drain=now;pending=bool(v._terminal_environment_update_pending or v._terminal_snapshot_until>now or v._environment_update_future is not None)
                if end_requested.is_set():raise Finished()
                if begin.is_set() and counter['started'] is None:
                    assert counter['total']==warmup_target,'Warmup must finish before measurement'
                    if args.issue in (88,89,94):assert row.get('flag_fields_verified') and flag_state['last'] is not None
                    row.update(status='measuring',warmup_ticks=counter['total']);save()
                    if profiler:profiler.enable()
                    now=time.perf_counter();counter['started']=now;last_drain=now;v._reset_performance_diagnostics();v.running=True;v._next_physics_step_at=0.
                if counter['started'] is not None and counter['ended'] is None and (counter['total']>=500 if profile_mode else now-counter['started']>=20):
                    counter['ended']=now;v.running=False;finish.set()
                    if profiler:profiler.disable()
                    original_publish(0.)
                    row.update(status='measured',actual_ticks=counter['total'],measurement_ticks=counter['measured'],measurement_wall=now-counter['started'],actual_time_scale=counter['measured']*.02/(now-counter['started']),active_physics_time_scale=counter['measured']*.02/max(.0001,now-counter['started']-counter['pending_wall']),advance_wall=counter['advance_wall'],pending_wall=counter['pending_wall'],snapshots=counter['snapshots'],snapshot=json.loads(v.snapshot.read())['goal_reach_viewer']['performance'])
                    if args.issue in (88,89,94):
                        row['fault_flags']=dict(fields=['runtime_fault','command_error','checkpoint_refresh_error','reset_error','time_scale_limited'],samples=flag_state['samples'],intervals=flag_state['intervals'])
                        row['snapshot_actual_hz']=counter['snapshots']/(now-counter['started'])
                    save()
                return original_drain()
            v._advance_viewer_physics=advance;v._drain_commands=drain;v._publish_snapshot=publish;v.submit=submit
            # This is only an isolated measurement switch, never a production control.
            html=(root/'Python/viewer/goal_reach.html').read_text('utf-8')
            if args.issue==94:
                html=html.replace('const text=await response.text(),value=JSON.parse(text);','const text=await response.text(),parseStart94=performance.now(),value=JSON.parse(text),parseEnd94=performance.now();')
                html=html.replace('dirtyTerrainTiles(changed);','dirtyTerrainTiles(changed);(window.terrainResponses94??=[]).push({world:key,blocks:value.blocks.length,bytes:text.length,parse:parseEnd94-parseStart94,install:performance.now()-parseEnd94});')
                import auv_madrl.viewer_terrain as terrain_module
                original_generate=terrain_module.generate_terrain;generation_wall=[0.]
                def timed_generate(*a,**kw):
                    at=time.perf_counter()
                    try:return original_generate(*a,**kw)
                    finally:generation_wall[0]+=time.perf_counter()-at
                instrument=patch.object(terrain_module,'generate_terrain',timed_generate);instrument.start();instrumentation.append(instrument)
                original_terrain_read=v._viewer_terrain.read;row['terrain_reads']=[]
                def terrain_read(query):
                    at=time.perf_counter();generation=generation_wall[0];payload=None
                    try:payload=original_terrain_read(query);return payload
                    finally:row['terrain_reads'].append(dict(at=at-started,wall=time.perf_counter()-at,generation_wall=generation_wall[0]-generation,bytes=len(payload) if payload else 0,query=query))
                v._viewer_terrain.read=terrain_read
            hook="""
      const issue77Loading=[];let issue77LastLoading=null;const terrainResolveMeasured=resolveTerrain;resolveTerrain=(...args)=>{const result=terrainResolveMeasured(...args),now=performance.now();if(issue77LastLoading!==terrainView.loading){issue77Loading.push({at:now,loading:terrainView.loading,world:terrainView.key,wanted:terrainView.wanted.length});issue77LastLoading=terrainView.loading;}return result;};
      const issue77Tasks=[];new PerformanceObserver(list=>{issue77Tasks.push(...list.getEntries().map(e=>({start:e.startTime,duration:e.duration})));}).observe({type:'longtask',buffered:true});
      window.issue77={start(){if(terrainView.loading||terrainView.pending||terrainView.job)throw Error('Terrain is not complete');for(const w of Object.values(browserPerformanceWindows)){w.values.fill(null);w.count=0;w.next=0;}issue77Tasks.length=0;window.issue77MeasurementStartedAt=performance.now();lastSnapshotReceivedAt=null;lastRafAt=null;return send({type:'issue77_start',client_request_id:'issue77-measure-start'});},finish(){return send({type:'issue77_finish',client_request_id:'issue77-measure-finish'});},stats(){return structuredClone({capturedAt:performance.now(),measurementStartedAt:window.issue77MeasurementStartedAt??null,windows:Object.fromEntries(Object.entries(browserPerformanceWindows).map(([k,w])=>[k,w.values.filter(Number.isFinite)])),longtasks:issue77Tasks,terrain:{loading:terrainView.loading,wanted:terrainView.wanted.length,blocks:terrainView.blocks.size,requests:terrainView.requests,bytes:terrainView.bytes,peakBytes:terrainView.peakBytes,receivedBytes:terrainView.receivedBytes,builds:terrainView.builds,contourBuilds:terrainView.contourBuilds,workMs:terrainView.workMs,wantedBuilds:terrainView.wantedBuilds,coverageBuilds:terrainView.coverageBuilds,missing:terrainView.missing?.size,error:terrainView.error,loadingWindows:issue77Loading},running:snapshot?.goal_reach_viewer?.running,episode:snapshot?.arenas?.[0]?.episode,world:snapshot?.world_descriptor});}};
"""
            if args.condition=='context_off':hook+="""
      resolvedGhosts=()=>[];resolveTerrain=()=>({tiles:[],complete:true});
      drawTerrainContext=(render,box)=>{const camera=render.camera;ctx.save();ctx.beginPath();for(const circle of render.sensing_circles){const p=cameraProject(camera,circle.x,circle.z);ctx.moveTo(p[0]+circle.r*camera.scale,p[1]);ctx.arc(...p,circle.r*camera.scale,0,2*Math.PI);}ctx.clip();drawObstacleUnion(render.arena.obstacles,(x,z)=>cameraProject(camera,x,z),camera.scale,false,null);ctx.restore();};
"""
            if args.condition=='far':hook+="const issue77Transform=mainViewTransform;mainViewTransform=p=>({...issue77Transform(p),rotation:Math.PI/4});\n"
            if args.issue==94:
                hook=hook.replace('let issue77LastLoading=null','let issue77LastLoading=null,issue77LastWorld=null').replace('issue77LastLoading!==terrainView.loading','issue77LastLoading!==terrainView.loading||issue77LastWorld!==terrainView.key').replace('issue77LastLoading=terrainView.loading;','issue77LastLoading=terrainView.loading;issue77LastWorld=terrainView.key;')
                hook=hook.replace('longtasks:issue77Tasks','terrainResponses:window.terrainResponses94||[],longtasks:issue77Tasks')
            marker="      applyViewerTheme(document.documentElement.dataset.theme, false);"
            assert html.count(marker)==1;html=html.replace(marker,hook+marker)
            html_path=temp/'goal_reach.html';html_path.write_text(html,encoding='utf-8')
            factory=gr._handler_factory
            def handler(*a,**kw):
                base=factory(*a,**kw)
                class Handler(base):
                    def do_GET(self):
                        if self.path=='/api/issue77-status':
                            payload=json.dumps(dict(condition=args.condition,warmed=counter['total']>=warmup_target,measuring=counter['started'] is not None and counter['ended'] is None,finished=finish.is_set(),ticks=counter['total'])).encode()
                            self.send_response(HTTPStatus.OK);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
                        else:super().do_GET()
                return Handler
            sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen();port=sock.getsockname()[1]
            row.update(port=port,prepare_wall=time.perf_counter()-started,settings=settings,status='warming',owned_children_before=[dict(pid=p.pid,created=p.create_time()) for p in process.children(recursive=True)])
            save();print(json.dumps(dict(url=f'http://127.0.0.1:{port}/',pid=os.getpid(),condition=args.condition)),flush=True)
            v.running=True
            with patch.object(gr,'_handler_factory',handler):
                try:v.run(html_path,port,reserved_socket=sock)
                except Finished:pass
    except BaseException as error:
        row['error']=repr(error);raise
    finally:
        if v is not None:
            if v._gc_callback in gc.callbacks:gc.callbacks.remove(v._gc_callback)
            for name,executor in vars(v).items():
                if name.endswith('_executor') and executor is not None:executor.shutdown(wait=True,cancel_futures=True)
        for instrument in instrumentation:instrument.stop()
        row['hot_calls']=hot_calls
        if profiler:
            profiler.disable()
            rows=[dict(function=f'{k[0]}:{k[1]}:{k[2]}',calls=v[1],self_seconds=v[2],cumulative_seconds=v[3]) for k,v in pstats.Stats(profiler).stats.items()]
            row['profile']={key:sorted(rows,key=lambda r:r[key],reverse=True)[:30] for key in ('self_seconds','cumulative_seconds')}
        for thread in set(threading.enumerate())-owned_before:thread.join(timeout=5)
        row['remaining_threads']=[t.name for t in set(threading.enumerate())-owned_before if t.is_alive()]
        row['owned_children_after']=[dict(pid=p.pid,created=p.create_time()) for p in process.children(recursive=True)]
        row['actual_ticks']=counter['total'];row['warmup_ticks']=counter['total']-counter['measured'];row['measurement_ticks']=counter['measured']
        row['status']='completed' if finish.is_set() and not row.get('error') else 'failed'
        for worker in row['training_before']:assert psutil.Process(worker['pid']).create_time()==worker['created']
        e['budget']=dict(actual_ticks=inherited_ticks+sum(w.get('actual_ticks',0) for key in ('windows','profiles') for w in e.get(key,[])),limit_ticks=limit_ticks,window_slots_used=inherited_slots+len(windows),maximum_windows=maximum_windows)
        save();print(json.dumps({k:row.get(k) for k in ['condition','status','actual_ticks','measurement_ticks','measurement_wall','actual_time_scale','remaining_threads','error']}),flush=True)


def counted_advance(counter, run, get_tick, advance):
    """Account real work even when advance or the following observer fails."""
    before=get_tick()
    try:return advance()
    finally:
        delta=get_tick()-before
        counter['actual_ticks']+=delta;run['actual_ticks']+=delta
        if delta<0:raise ValueError('Physics counter moved backwards')


def reusable_profile(attempts, identity, mode):
    return next((r for r in reversed(attempts) if r['mode']==mode and r.get('identity')==identity
                 and r.get('status')=='complete' and r.get('cleanup_complete') is True),None)


def cpu_profile_rows(profiler, ticks, decisions, timer_kind):
    import pstats
    def identity(key):return dict(file=key[0],line=key[1],name=key[2])
    rows=[]
    for key,(primitive,calls,exclusive,inclusive,callers) in pstats.Stats(profiler).stats.items():
        rows.append(dict(function=identity(key),calls=calls,primitive_calls=primitive,
            **{'inclusive_'+timer_kind:inclusive,'exclusive_'+timer_kind:exclusive},
            inclusive_per_tick=inclusive/ticks if ticks else None,exclusive_per_tick=exclusive/ticks if ticks else None,
            inclusive_per_decision=inclusive/decisions if decisions else None,exclusive_per_decision=exclusive/decisions if decisions else None,
            denominator_reason=None if ticks and decisions else 'No effective ticks or decisions',
            callers=[dict(function=identity(caller),primitive_calls=v[0],calls=v[1],
                **{'exclusive_'+timer_kind:v[2],'inclusive_'+timer_kind:v[3]}) for caller,v in callers.items()]))
    return sorted(rows,key=lambda r:r['exclusive_'+timer_kind],reverse=True)


def finish_profile_attempt(counter, run, error, remaining_threads, save):
    if error is not None:
        run.update(status='failed',error=repr(error));counter['status']='recoverable_failure'
    run['remaining_threads']=remaining_threads;run['cleanup_complete']=not remaining_threads
    save()


def restore_cpu_profile_session(session, inputs, release, initial, rng):
    """Restore the recorded CPU source, shared by fixed-tick and Viewer-loop probes."""
    import copy,random
    import numpy as np
    import torch
    from auv_madrl import goal_reach as gr
    session._auto_deploy_training_run_group_id=None;session._auto_deploy_training_sides.clear()
    torch.set_rng_state(rng['torch']);random.setstate(rng['python']);np.random.set_state(rng['numpy'])
    session._viewer_policy_session_seed=inputs['policy_session_seed']
    session._reset(episode_index=int(initial['episode_counters'][0]),initial_state=copy.deepcopy(initial))
    session._environment_update_future.result(timeout=60);session._poll_environment_update()
    assert gr.physical_digest(session.env).hex()==inputs['initial_digest'] and str(session._policy_noise.seed)==inputs['policy_seed'], (gr.physical_digest(session.env).hex(), inputs['initial_digest'], str(session._policy_noise.seed), inputs['policy_seed'],session.reset_progress)
    _,_,payload,_=gr._read_group_release(release);models={}
    for role,control in session.role_controls.items():
        actor=gr._group_actor_payload(payload,role,release);key=actor['source_metadata']['parameter_group']
        if key not in models:
            _,models[key]=session._prepare_checkpoint_model(actor)
            models[key]._viewer_actor_hash=gr.actor_state_hash(models[key].state_dict())
        control.neural_policy=gr._ViewerPolicy(models[key])
    assert all(session._role_deployment_descriptor(r)['actor_hash']==inputs['deployment'][r]['actor_hash'] for r in session.role_controls)
    torch.set_rng_state(rng['torch']);random.setstate(rng['python']);np.random.set_state(rng['numpy'])
    return session


def profile_cpu_contacts(issue=104, *, output=None, variant=None, parent_attempt_id=None, expected_identity=None):
    """Refs #104: recoverable same-input CPU pair, isolated from user services."""
    import copy,cProfile,gc,hashlib,json,os,random,subprocess,tempfile,threading,time
    from pathlib import Path
    from unittest.mock import patch
    import numpy as np
    import psutil
    import torch
    from auv_madrl import goal_reach as gr,actor_training as at
    from profile_training_collection import digest,active_run_identities
    if issue not in (104,106,114):raise ValueError('Unsupported execution round')
    if issue in (106,114) and (output is None or variant not in ('before','after')):
        raise ValueError('Issue106 requires an isolated output and explicit source variant')
    root=Path(__file__).resolve().parents[2];path=root/'Artifacts/GoalReach/viewer-review-terminal.json'
    data=json.loads(path.read_text('utf-8'));section=f'issue{issue}'
    if output is not None:path=Path(output)
    e=data.get(section,dict(actual_ticks=0,attempts=[],status='preparing')) if output is None else dict(actual_ticks=0,attempts=[],status='preparing')
    e.update(native_thread_id=os.environ.get('CODEX_THREAD_ID'),tick_limit=10000,historical_ticks=501 if issue==104 else 0,
        formal_windows_used=0,gpu_ticks=0,optimizer_steps=0)
    def save():
        current=json.loads(path.read_text('utf-8')) if output is None else {};current[section]=e;at.write_json_atomic(path,current,compact=output is None)
    e['workers_before']=active_run_identities();save()
    inputs=copy.deepcopy(data['issue114' if issue==114 else 'issue101']['inputs'])
    # Known source and archive mapping only; never substitute a different weight.
    candidates=[Path(inputs['release']),Path(data['issue101']['resolved_release']),root/'Artifacts/GoalReach/archive'/Path(inputs['release']).parent.name/Path(inputs['release']).name]
    release=next((p for p in candidates if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==inputs['release_sha256']),None)
    if release is None:raise FileNotFoundError('Recorded archived release unavailable; isolated random-input fallback needs preparation')
    for key in ('initial_state','initial_rng'):
        source=Path(inputs[key])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=inputs[key+'_sha256']:raise ValueError('Frozen input hash mismatch: '+key)
    initial=torch.load(inputs['initial_state'],map_location='cpu',weights_only=False)
    rng=torch.load(inputs['initial_rng'],map_location='cpu',weights_only=False)
    clocks={'wall':time.perf_counter_ns,'thread_cpu':time.thread_time_ns,'process_cpu':time.process_time_ns}
    e['clocks']={}
    for label,clock in clocks.items():
        values=[];until=time.perf_counter()+.04
        while time.perf_counter()<until:values.append(clock())
        deltas=[b-a for a,b in zip(values,values[1:]) if b>a]
        e['clocks'][label]={'minimum_positive_ns':min(deltas) if deltas else None,'positive_samples':len(deltas)}
    resolution=e['clocks']['thread_cpu']['minimum_positive_ns'];timer_kind='thread_cpu' if resolution is not None and resolution<=1000 else 'wall'
    production={(str(p.relative_to(root)) if issue==104 else p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(gr.__file__).parent.glob('*.py'))}
    identity=dict(inputs={k:v for k,v in inputs.items() if k not in ('release','initial_state','initial_rng')},production=production,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch=torch.__version__,threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),timer=timer_kind,mode_definition='same frozen initial/RNG/actors; <=250 ticks; natural terminal; production-only profile')
    if issue in (106,114):
        identity=replay_comparison_identity(root,data,Path(gr.__file__).parent,
            threads=(torch.get_num_threads(),torch.get_num_interop_threads()),input_section='issue114' if issue==114 else 'issue101')
        if identity!=expected_identity:
            raise ValueError('Replay child input/source/measurement identity changed before advance')
        e['parent_attempt_id']=parent_attempt_id
    e.update(inputs=inputs,resolved_release=str(release),source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        profile_timer=timer_kind,identity=identity,pid=os.getpid(),status='profiling');save()
    def stamp():return {k:f() for k,f in clocks.items()}
    def rng_digest(v):return digest(dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),noise=v._policy_noise.tables,seed=v._policy_noise.seed))
    def observe(v):
        record=v._live_record or v._last_complete_record
        if record is None:raise ValueError('Missing production Replay record')
        row=dict(tick=int(v.env.physics_ticks[0]),K=int(v.env.obstacle_centers.shape[1]),digest=gr.physical_digest(v.env).hex(),
            role_active={r:getattr(v.env,r+'_active').tolist() for r in ('learner','defender','target')},
            actions={r:digest(getattr(v,'last_'+r+'_action')) for r in sorted(v.role_controls)},done=bool(v.env.done[0]),
            replay_checks={str(k):x.hex() for k,x in record.checks.items()},
            events=[{k:f[k] for k in ('physics_ticks','event_type','actor_kind','actor_id')} for f in v.event_freeze_frames])
        json.dumps(row,allow_nan=False);return row
    def close(v):
        if v is None:return
        if v._gc_callback in gc.callbacks:gc.callbacks.remove(v._gc_callback)
        for name,executor in vars(v).items():
            if name.endswith('_executor') and executor is not None:executor.shutdown(wait=True,cancel_futures=True)
    for mode in ((variant,) if issue in (106,114) else ('baseline','profile')):
        if reusable_profile(e['attempts'],identity,mode):continue
        v=None;threads=set(threading.enumerate());profiler=None;caught=None
        run=dict(mode=mode,identity=identity,status='preparing',actual_ticks=0,ticks=[],decisions=0,stages={},audit_seconds=0.,initialization_seconds=0.)
        e['attempts'].append(run);save()
        started_init=time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix='auv-cpu-profile-') as tmp:
                def restore():
                    with patch.object(gr,'_load_viewer_settings',lambda config:(Path(tmp)/'settings.json',copy.deepcopy(inputs['settings']))):session=gr.GoalReachViewerSession()
                    nonlocal_v[0]=session
                    return restore_cpu_profile_session(session,inputs,release,initial,rng)
                nonlocal_v=[None]
                try:v=restore()
                finally:v=nonlocal_v[0]
                # Actual observer/production refresh/event/serialization calls, zero advance.
                pre=int(v.env.physics_ticks[0]);probe=cProfile.Profile(timer=clocks[timer_kind],timeunit=1e-9)
                probe.enable();v._refresh_actor_frame();probe.disable()
                v._record_event_freeze_frames();v._record_trajectory_positions()
                preflight=observe(v);preflight['rng']=rng_digest(v)
                preflight['profile']=cpu_profile_rows(probe,0,0,timer_kind)
                json.dumps(preflight,allow_nan=False);assert int(v.env.physics_ticks[0])==pre
                run['preflight']={'tick':pre,'K':preflight['K'],'observer_serialized':True,'zero_denominator_profile_serialized':True}
                close(v);v=None
                try:v=restore()
                finally:v=nonlocal_v[0]
                run['initial']=observe(v);run['initial_rng']=rng_digest(v)
                run['initialization_seconds']=time.perf_counter()-started_init
                profiler=cProfile.Profile(timer=clocks[timer_kind],timeunit=1e-9) if mode=='profile' else None
                def production(label,fn,*args):
                    begin=stamp()
                    if profiler:profiler.enable()
                    try:return fn(*args)
                    finally:
                        if profiler:profiler.disable()
                        elapsed={k:(f()-begin[k])*1e-9 for k,f in clocks.items()}
                        slot=run['stages'].setdefault(label,dict(calls=0,**{k:0. for k in clocks}));slot['calls']+=1
                        for k,x in elapsed.items():slot[k]+=x
                baseline=reusable_profile(e['attempts'],identity,'baseline')
                limit=len(baseline['ticks']) if baseline else 250
                cadence=int(v.config.shared['clock']['physics_ticks_per_decision']);outer=stamp()
                with torch.inference_mode():
                    for index in range(limit):
                        if bool(v.env.done[0]):break
                        if e['actual_ticks']>=e['tick_limit']:raise RuntimeError('Authorized physics budget exhausted')
                        before=int(v.env.physics_ticks[0])
                        if before%cadence==0:
                            production('refresh',v._refresh_actor_frame);run['decisions']+=1
                        step=counted_advance(e,run,lambda:int(v.env.physics_ticks[0]),lambda:production('advance',v._advance_viewer_physics,1))
                        assert int(v.env.physics_ticks[0])-before==1
                        production('outcomes',v._update_current_learner_outcomes,step);production('facts',v._update_current_match_facts,step)
                        production('events',v._record_event_freeze_frames);production('trajectory',v._record_trajectory_positions)
                        audit_start=time.perf_counter();run['ticks'].append(observe(v));run['audit_seconds']+=time.perf_counter()-audit_start
                run['outer']={k:(f()-outer[k])*1e-9 for k,f in clocks.items()}
                run['end']=observe(v);run['end_rng']=rng_digest(v)
                if issue in (106,114):
                    record=v._live_record or v._last_complete_record
                    run['record']=dict(intervals=[{**x,'actions':{r:a.tolist() for r,a in x['actions'].items()}} for x in record.intervals],checks={str(k):x.hex() for k,x in record.checks.items()})
                    run['record_bytes']=record.bytes
                    run['microbench']=replay_append_measurement(record,variant) if issue==106 else {}
                if profiler:run['profile']=cpu_profile_rows(profiler,len(run['ticks']),run['decisions'],timer_kind)
                run['production_wall']=sum(x['wall'] for x in run['stages'].values())
                run['outside_production_and_audit_wall']=run['outer']['wall']-run['production_wall']-run['audit_seconds']
                run['status']='complete'
        except BaseException as error:
            caught=error;raise
        finally:
            if profiler:profiler.disable()
            close(v)
            remaining=[t.name for t in set(threading.enumerate())-threads if t.is_alive()]
            e['workers_after']=active_run_identities();finish_profile_attempt(e,run,caught,remaining,save)
        if not run['cleanup_complete']:raise RuntimeError('Temporary threads remain; do not resume yet')
    if issue in (106,114):
        e['status']='complete';save();return
    a=reusable_profile(e['attempts'],identity,'baseline');b=reusable_profile(e['attempts'],identity,'profile')
    e['equivalence']={k:a[k]==b[k] for k in ('initial','initial_rng','ticks','end','end_rng')}
    e['status']='profile_complete' if all(e['equivalence'].values()) else 'production_mismatch'
    e['profile_scope']='Only production entry calls under cProfile; test observations/digests/JSON and initialization excluded. Inclusive nested times must not be summed. Wall is elapsed, not thread CPU.'
    save();print(json.dumps(dict(status=e['status'],actual_ticks=e['actual_ticks'],equivalence=e['equivalence'])),flush=True)



def replay_append_measurement(record, variant):
    """Finite real-frame microbench, separate from production and generation."""
    import cProfile, pstats, time, tracemalloc
    from auv_madrl.viewer_replay import ViewerEpisodeRecord
    def append_sequence():
        target=ViewerEpisodeRecord({}, {}, record.contract)
        for key,item in enumerate(record.intervals):
            frame=item['frame']
            target.append(item['start'],item['end'],item['actions'],key,
                          (lambda:frame) if variant=='before' else frame,None)
        assert not target.error
        return target
    samples=[]
    for _ in range(10):
        start=time.perf_counter();target=append_sequence();samples.append(time.perf_counter()-start)
    profiler=cProfile.Profile();profiler.enable();append_sequence();profiler.disable()
    direct_copies=0
    for key,stats in pstats.Stats(profiler).stats.items():
        if key[2]=='deepcopy':
            direct_copies+=sum(v[1] for caller,v in stats[4].items() if caller[2]=='append')
    tracemalloc.start();append_sequence();current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    assert direct_copies == (len(record.intervals) if variant=='before' else 0)
    assert all(x['actions'][r].data_ptr()!=y['actions'][r].data_ptr()
               for x,y in zip(target.intervals,record.intervals) for r in x['actions'])
    return dict(samples_wall=samples,frame_count=len(record.intervals),interval_count=len(target.intervals),
                accounted_bytes=target.bytes,peak_python_bytes=peak,direct_frame_deepcopies=direct_copies,
                action_clones_independent=True,initialization='Empty record creation included equally; production frame generation excluded')


def replay_comparison_identity(root, data, source, *, threads=(8,8), input_section='issue101'):
    """Zero-advance content identity; relocation of identical inputs is harmless."""
    import copy,hashlib
    from pathlib import Path
    import torch
    inputs=copy.deepcopy(data[input_section]['inputs'])
    for key in ('release','initial_state','initial_rng'):
        candidates=[Path(inputs[key])]
        if key=='release':
            candidates += [Path(data['issue101']['resolved_release']),
                root/'Artifacts/GoalReach/archive'/Path(inputs[key]).parent.name/Path(inputs[key]).name]
        if not any(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==inputs[key+'_sha256'] for p in candidates):
            raise ValueError('Frozen input unavailable or hash changed: '+key)
        del inputs[key]
    configs={str(p.relative_to(root/'Python/configs')):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((root/'Python/configs').rglob('*.yaml'))}
    if not configs:raise ValueError('Missing effective configuration files')
    return dict(inputs=inputs,config_files=configs,
        production={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('*.py'))},
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch=str(torch.__version__),
        threads=threads[0],interop_threads=threads[1],
        timer='perf_counter_ns/thread_time_ns/process_time_ns; production-call-only; append-wall-10',
        mode_definition='same frozen initial/RNG/actors; <=250 ticks; natural terminal; production-only profile')


def import_replay_test_ticks(section, path):
    """Import one monotone cumulative counter, never an additive reread."""
    import json
    if type(section['actual_ticks']) is not int or section['actual_ticks']<0:
        raise ValueError('Invalid confirmed Replay tick total')
    previous=section.get('test_ticks')
    if not path.is_file():
        if previous is not None:raise ValueError('Imported Replay test counter disappeared')
        return
    current=json.loads(path.read_text('utf-8'))
    if not isinstance(current,dict) or not isinstance(current.get('tests'),dict):
        raise ValueError('Malformed Replay test counter')
    if any(not isinstance(row,dict) for row in current['tests'].values()):
        raise ValueError('Malformed Replay per-test counter')
    def count(value):
        if type(value) is not int or value<0:raise ValueError('Invalid Replay tick counter')
        return value
    total=count(current['actual_ticks']);old=count((previous or {}).get('actual_ticks',0))
    if total<old:raise ValueError('Replay test counter moved backwards')
    if previous and previous.get('accounting_id')!=current.get('accounting_id'):
        raise ValueError('Replay test counter identity changed')
    for name,row in (previous or {}).get('tests',{}).items():
        if name not in current.get('tests',{}) or count(current['tests'][name]['actual_ticks'])<count(row['actual_ticks']):
            raise ValueError('Replay per-test counter moved backwards')
    if sum(count(row['actual_ticks']) for row in current.get('tests',{}).values())!=total:
        raise ValueError('Replay test counter total mismatch')
    section['actual_ticks']=count(section['actual_ticks'])+total-old
    section['test_ticks']=current


def replay_child_sample(row, identity):
    """Only a terminal, cleaned, correctly identified complete child is reusable."""
    import math
    child=row.get('child',{})
    if not isinstance(child,dict):return None
    attempts=child.get('attempts',[])
    if not isinstance(attempts,list):return None
    if (row.get('status')!='complete' or row.get('exit_code')!=0 or child.get('status')!='complete'
            or child.get('identity')!=identity or len(attempts)!=1):return None
    if row.get('attempt_id') and (child.get('parent_attempt_id')!=row['attempt_id']
            or row.get('process_ended') is not True or row.get('settled') is not True):return None
    sample=attempts[0]
    if not isinstance(sample,dict):return None
    if (sample.get('identity')!=identity or sample.get('mode')!=row.get('variant')
            or sample.get('status')!='complete' or sample.get('cleanup_complete') is not True):return None
    count=child.get('actual_ticks')
    if (type(count) is not int or not 0<=count<=250 or type(sample.get('actual_ticks')) is not int
            or sample['actual_ticks']!=count or row.get('actual_ticks')!=count):return None
    if not all(key in sample for key in ('initial','initial_rng','ticks','end','end_rng','record',
                                        'production_wall','microbench','record_bytes')):return None
    if (not isinstance(sample['ticks'],list) or len(sample['ticks'])!=count
            or any(not isinstance(t,dict) for t in sample['ticks'])
            or any(not isinstance(sample[k],dict) for k in ('initial','end','record','microbench'))
            or any(not isinstance(sample[k],str) or not sample[k] for k in ('initial_rng','end_rng'))
            or sample['initial'].get('digest')!=identity['inputs']['initial_digest']
            or type(sample['record_bytes']) is not int or sample['record_bytes']<0
            or type(sample['production_wall']) not in (int,float)
            or not math.isfinite(sample['production_wall']) or sample['production_wall']<0):return None
    return sample


def settle_replay_child(section, row, child, ended):
    """Idempotent actual-count settlement; an unproven ledger keeps its reservation."""
    if not isinstance(child,dict):raise ValueError('Malformed child ledger')
    count=child.get('actual_ticks');runs=child.get('attempts',[]);identity=row['identity']
    if not isinstance(runs,list) or any(not isinstance(run,dict) for run in runs):
        raise ValueError('Malformed child attempts')
    if (not ended or child.get('parent_attempt_id')!=row['attempt_id'] or child.get('identity')!=identity
            or type(count) is not int or not 0<=count<=250 or len(runs)!=1
            or runs[0].get('identity')!=identity or runs[0].get('mode')!=row['variant']
            or runs[0].get('cleanup_complete') is not True or type(runs[0].get('actual_ticks')) is not int
            or runs[0]['actual_ticks']!=count or runs[0].get('status') not in ('complete','failed')
            or child.get('status') not in ('complete','recoverable_failure')):
        raise ValueError('Child terminal count/identity/cleanup not proven')
    if row.get('settled'):
        if row['actual_ticks']!=count:raise ValueError('Settled Replay count changed')
        return
    section['actual_ticks']+=count
    row.update(child=child,actual_ticks=count,reserved_ticks=0,settled=True,process_ended=True)


def compare_owned_replay(issue=106):
    """Refs #106: serial full-source subprocess pairs; durable failed-work accounting."""
    import json, os, shutil, subprocess, sys, tempfile, zipfile, uuid
    from pathlib import Path
    from auv_madrl import actor_training as at
    root=Path(__file__).resolve().parents[2]
    path=root/'Artifacts/GoalReach/viewer-review-terminal.json'
    data=json.loads(path.read_text('utf-8'))
    if issue not in (106,114):raise ValueError('Unsupported comparison round')
    section=f'issue{issue}'
    e=data.setdefault(section,dict(actual_ticks=0,attempts=[],tick_limit=10000,gpu_ticks=0,optimizer_steps=0))
    def save():at.write_json_atomic(path,data,compact=True)
    baseline='c929429347dd7e89fd310dd5de85cd1712182f8c' if issue==114 else '6fd52198b369adbc1263f1873134c3b198e7c7b5'
    e['before_sha']=baseline
    test_path=Path(os.environ.get('AUV_REPLAY_TICK_OUTPUT',str(Path(os.environ['TEMP'])/'auv-issue106-test-ticks.json')))
    try:
        if issue==106:import_replay_test_ticks(e,test_path)
    except (OSError,ValueError,KeyError,TypeError) as error:
        e['resume_error']=repr(error);save();raise
    save()
    tmp=Path(tempfile.mkdtemp(prefix='auv-replay-pair-'));keep_workspace=False
    try:
        archive=tmp/'source.zip'
        subprocess.run(['git','archive','--format=zip','--output',str(archive),baseline,'Python/src'],cwd=root,check=True)
        with zipfile.ZipFile(archive) as z:z.extractall(tmp/'before')
        shutil.copytree(root/'Python/src',tmp/'after/Python/src',ignore=shutil.ignore_patterns('__pycache__'))
        try:
            identities={v:replay_comparison_identity(root,data,tmp/v/'Python/src/auv_madrl',**(dict(threads=(1,1),input_section='issue114') if issue==114 else {})) for v in ('before','after')}
            attempt_ids=[row['attempt_id'] for row in e['attempts'] if row.get('attempt_id')]
            if len(set(attempt_ids))!=len(attempt_ids):raise ValueError('Duplicate Replay attempt identity')
            for row in e['attempts']:
                # A killed parent may not have persisted a child handle. Never infer idle.
                if row.get('reserved_ticks') or (row.get('status')!='complete' and not row.get('settled')):
                    row.setdefault('reserved_ticks',250)
                    save()
                    process=row.get('process',{})
                    try:state,token=at._query_process_identity(process['pid']) if process.get('pid') else ('unknown',None)
                    except OSError:state,token='unknown',None
                    row['recovery_process_status']=state
                    ended=bool(process.get('process_start_token')) and (state=='dead' or
                        (state=='alive' and token is not None and token!=process['process_start_token']))
                    try:
                        if not ended:raise ValueError('Prior child termination is unconfirmed')
                        child=json.loads(Path(row['child_output']).read_text('utf-8'))[section]
                        settle_replay_child(e,row,child,ended)
                        row.update(status='failed',error='Interrupted parent recovered final child charge; sample not reused')
                        save()
                    except (OSError,ValueError,KeyError,TypeError) as recovery_error:
                        row['accounting_error']=repr(recovery_error)
                        raise RuntimeError('Unsettled Replay reservation; child termination/accounting not proven') from recovery_error
                    retained=Path(row['child_output']).parent.resolve()
                    if (str(retained)==row.get('retained_workspace') and retained.parent==Path(tempfile.gettempdir()).resolve()
                            and retained.name.startswith('auv-replay-pair-')):
                        shutil.rmtree(retained)
                expected=identities.get(row.get('variant'))
                known=row.get('identity')
                if known is None and isinstance(row.get('child'),dict):known=row['child'].get('identity')
                if known!=expected or (row.get('status')=='complete' and replay_child_sample(row,expected) is None):
                    raise ValueError('Replay input/source/measurement identity or terminal evidence mismatch')
            if e.get('comparison_identity') not in (None,identities):
                raise ValueError('Replay comparison identity changed')
            e['comparison_identity']=identities
            confirmed=sum(row['actual_ticks'] for row in e['attempts'] if type(row.get('actual_ticks')) is int)
            if confirmed>e['actual_ticks']:raise ValueError('Confirmed Replay total is below its attempt records')
        except (ValueError,RuntimeError) as error:
            e['resume_error']=str(error);save();raise
        for pair in range(3):
            for variant in (('after','before') if issue==114 and pair%2 else ('before','after')):
                identity=identities[variant]
                if any(x.get('pair')==pair and x.get('variant')==variant and replay_child_sample(x,identity) for x in e['attempts']):continue
                if e['actual_ticks']+sum(x.get('reserved_ticks',0) for x in e['attempts'])+250>e['tick_limit']:
                    raise RuntimeError('Insufficient Issue106 budget')
                output=tmp/f'{pair}-{variant}.json'
                row=dict(pair=pair,variant=variant,status='preparing',actual_ticks=None,reserved_ticks=250,
                         attempt_id=uuid.uuid4().hex,identity=identity,settled=False,child_output=str(output))
                e['status']='comparison_running'
                e['attempts'].append(row);save()
                threads=1 if issue==114 else 8
                code=f"import sys,json,torch;sys.path[:0]=sys.argv[1:3];torch.set_num_threads({threads});torch.set_num_interop_threads({threads});from viewer_performance_window import profile_cpu_contacts;profile_cpu_contacts({issue},output=sys.argv[3],variant=sys.argv[4],parent_attempt_id=sys.argv[5],expected_identity=json.loads(sys.argv[6]))"
                process=None;error=None;not_spawned=False
                try:
                    try:
                        process=subprocess.Popen([sys.executable,'-c',code,str(tmp/variant/'Python/src'),str(root/'Python/tests'),str(output),variant,row['attempt_id'],json.dumps(identity)],
                            cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
                    except OSError:
                        not_spawned=True;raise
                    row['process']={'pid':process.pid,'process_start_token':at.process_start_token(process.pid)};save()
                    row['stdout'],row['stderr']=process.communicate()
                except BaseException as caught:
                    error=caught
                finally:
                    ended=False
                    try:
                        if process is not None:
                            if process.poll() is None:
                                process.terminate()
                                try:row['stdout'],row['stderr']=process.communicate(timeout=10)
                                except subprocess.TimeoutExpired:
                                    process.kill();row['stdout'],row['stderr']=process.communicate(timeout=10)
                            ended=process.poll() is not None
                            row['exit_code']=process.returncode
                    except BaseException as cleanup_error:
                        row['cleanup_error']=repr(cleanup_error)
                    row['process_ended']=ended
                    if not ended and not not_spawned:
                        keep_workspace=True
                        row['retained_workspace']=str(tmp)
                    if error is not None:row['error']=repr(error)
                    row['status']='failed'
                    try:
                        if not_spawned:
                            row.update(actual_ticks=0,reserved_ticks=0,settled=True,spawn_failed=True)
                        else:
                            child=json.loads(output.read_text('utf-8'))[section];row['child']=child
                            settle_replay_child(e,row,child,ended)
                            if error is None and row.get('exit_code')==0 and child['status']=='complete':
                                row['status']='complete'
                                if replay_child_sample(row,identity) is None:raise ValueError('Incomplete Replay sample')
                    except (OSError,ValueError,KeyError,TypeError) as ledger_error:
                        row.update(status='failed',accounting_error=repr(ledger_error))
                    if row['status']!='complete':e['status']='recoverable_failure'
                    # This save precedes TemporaryDirectory.__exit__, including KeyboardInterrupt.
                    save()
                if error is not None:raise error
                print(json.dumps(dict(pair=pair,variant=variant,status=row['status'],actual_ticks=e['actual_ticks'])),flush=True)
                if row['status']!='complete':raise RuntimeError(row.get('accounting_error') or row.get('stderr') or 'Replay child failed')
        pairs=[]
        for pair in range(3):
            a,b=[next(replay_child_sample(x,identities[v]) for x in e['attempts']
                      if x['pair']==pair and x['variant']==v and replay_child_sample(x,identities[v])) for v in ('before','after')]
            equivalent={k:a[k]==b[k] for k in ('initial','initial_rng','ticks','end','end_rng','record')}
            pairs.append(dict(pair=pair,equivalence=equivalent,before_wall=a['production_wall'],after_wall=b['production_wall'],
                              before_micro=a['microbench'],after_micro=b['microbench'],before_record_bytes=a['record_bytes'],after_record_bytes=b['record_bytes']))
        e['pairs']=pairs;e['status']='comparison_complete' if all(all(p['equivalence'].values()) for p in pairs) else 'mismatch'
        save();print(json.dumps(dict(status=e['status'],actual_ticks=e['actual_ticks'])),flush=True)
    finally:
        if not keep_workspace:shutil.rmtree(tmp)


def cpu_preparation_window(variant, scale):
    """One authorized 15s real Viewer loop window; isolated server, no formal controls."""
    import copy,gc,hashlib,json,os,socket,tempfile,threading,time,uuid
    from pathlib import Path
    from unittest.mock import patch
    import torch
    from auv_madrl import goal_reach as gr,actor_training as at
    from profile_training_collection import active_run_identities
    if variant not in ('before','after') or scale not in ('full','paced'):raise ValueError('Invalid window')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    root=Path(__file__).resolve().parents[2];path=root/'Artifacts/GoalReach/viewer-review-terminal.json'
    data=json.loads(path.read_text('utf-8'));e=data['issue114'];windows=e.setdefault('windows',[])
    if len(windows)>=6:raise RuntimeError('Six authorized window slots exhausted')
    inputs=e['inputs'];identity=replay_comparison_identity(root,data,Path(gr.__file__).parent,threads=(1,1),input_section='issue114')
    row=dict(attempt_id=uuid.uuid4().hex,variant=variant,scale=scale,status='preparing',identity=identity,
        pid=os.getpid(),process_start_token=at.process_start_token(os.getpid()),window_seconds=15,
        actual_ticks=None,window_slot_reserved=True,workers_before=active_run_identities())
    windows.append(row)
    def save():
        current=json.loads(path.read_text('utf-8'));current['issue114']=e;at.write_json_atomic(path,current,compact=True)
    save();v=None;owned=set(threading.enumerate());clock={'start':None,'end':None,'ticks':0,'advance_wall':0.,'pending_wall':0.}
    timestamps=[];begin=threading.Event();caught=None
    class Finished(BaseException):pass
    try:
        with tempfile.TemporaryDirectory(prefix='auv-cpu-window-') as tmp:
            settings=copy.deepcopy(inputs['settings']);settings['time_scale']=gr.FULL_SPEED if scale=='full' else 5.
            with patch.object(gr,'_load_viewer_settings',lambda config:(Path(tmp)/'settings.json',settings)):v=gr.GoalReachViewerSession()
            # Settle constructor deployments before installing the frozen actors.
            for _, future in tuple(v._deployment_futures.values()):future.result(timeout=60)
            v._poll_viewer_deployments()
            initial=torch.load(inputs['initial_state'],map_location='cpu',weights_only=False);rng=torch.load(inputs['initial_rng'],map_location='cpu',weights_only=False)
            restore_cpu_profile_session(v,inputs,Path(inputs['release']),initial,rng)
            v._refresh_actor_frame();v._publish_snapshot(0.)
            assert int(v.env.physics_ticks[0])==0 and not v._deployment_futures
            json.dumps(json.loads(v.snapshot.read()),allow_nan=False)
            v.running=False
            advance=v._advance_viewer_physics;drain=v._drain_commands;store=v.snapshot.store;submit=v.submit
            last=[None,False];command={}
            def measured_advance(ticks,deadline=None):
                if clock['start'] is None:return None
                until=clock['start']+15
                if time.perf_counter()>=until:v.running=False;return None
                before=int(v.env.physics_ticks[0]);started=time.perf_counter()
                try:return advance(ticks,deadline=min(deadline,until) if deadline is not None else until)
                finally:clock['ticks']+=int(v.env.physics_ticks[0])-before;clock['advance_wall']+=time.perf_counter()-started
            def measured_store(payload,**kwargs):
                if clock['start'] is not None and clock['end'] is None:timestamps.append(time.perf_counter())
                return store(payload,**kwargs)
            def measured_submit(value):
                if value.get('type')=='issue77_start':begin.set();return
                return submit(value)
            def measured_drain():
                now=time.perf_counter()
                if begin.is_set() and clock['start'] is None:
                    row['status']='measuring';save()
                    v._reset_performance_diagnostics();clock['start']=time.perf_counter();last[0]=clock['start'];v.running=True;v._next_physics_step_at=0.
                if clock['start'] is not None:
                    if last[1]:clock['pending_wall']+=now-last[0]
                    last[:]=[now,bool(v._terminal_environment_update_pending or v._terminal_snapshot_until>now or v._environment_update_future is not None)]
                    if now-clock['start']>=15:
                        clock['end']=now;v.running=False
                        row.update(status='completed',actual_ticks=clock['ticks'],wall=now-clock['start'],actual_time_scale=clock['ticks']*.02/(now-clock['start']),advance_wall=clock['advance_wall'],pending_wall=clock['pending_wall'],performance=v._performance_snapshot(now),snapshots=len(timestamps),snapshot_hz=len(timestamps)/(now-clock['start']),snapshot_gaps=[b-a for a,b in zip(timestamps,timestamps[1:])],command=command)
                        raise Finished()
                    if now-clock['start']>=5 and not command:
                        command['submitted_at']=time.perf_counter();submit(dict(type='set_rate',name='time_scale',value=v.time_scale))
                result=drain()
                if command and 'applied_latency' not in command:command['applied_latency']=time.perf_counter()-command['submitted_at'];command['value']=v.time_scale;command['error']=v.command_error
                return result
            v._advance_viewer_physics=measured_advance;v._drain_commands=measured_drain;v.snapshot.store=measured_store;v.submit=measured_submit
            sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen();port=sock.getsockname()[1]
            row.update(port=port,status='ready',initial_digest=gr.physical_digest(v.env).hex(),preflight_zero_ticks=True);save()
            print(json.dumps(dict(port=port,variant=variant,scale=scale,status='ready')),flush=True)
            try:v.run(root/'Python/viewer/goal_reach.html',port,reserved_socket=sock)
            except Finished:pass
    except BaseException as error:
        caught=error;row.update(status='failed',error=repr(error));raise
    finally:
        if v is not None:
            if v._gc_callback in gc.callbacks:gc.callbacks.remove(v._gc_callback)
            for name,executor in vars(v).items():
                if name.endswith('_executor') and executor is not None:executor.shutdown(wait=True,cancel_futures=True)
        remaining=[t.name for t in set(threading.enumerate())-owned if t.is_alive()]
        row.update(actual_ticks=clock['ticks'],window_slot_reserved=False,cleanup_complete=not remaining,remaining_threads=remaining,workers_after=active_run_identities())
        save();print(json.dumps({k:row.get(k) for k in ['variant','scale','status','actual_ticks','wall','actual_time_scale','snapshot_hz','cleanup_complete','error']}),flush=True)

def cpu_event_sample(kind, variant, *, issue=121):
    """Refs #121: fixed production pair or one real HTTP/scheduler window, same frozen actors."""
    import copy, gc, hashlib, json, os, random, socket, subprocess, sys, tempfile, threading, time, types
    from pathlib import Path
    from unittest.mock import patch
    import numpy as np
    import torch
    from auv_madrl import goal_reach as gr, actor_training as at
    from profile_training_collection import digest, active_run_identities
    assert kind in ('fixed', 'full', 'paced') and variant in ('before','after')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    root=Path(__file__).resolve().parents[2];path=root/'Artifacts/GoalReach/viewer-review-terminal.json'
    data=json.loads(path.read_text('utf8'));e=data[f'issue{issue}'];inputs=e['inputs'];pool=data['issue121']
    initial=torch.load(inputs['initial_state'],map_location='cpu',weights_only=False)
    rng=torch.load(inputs['initial_rng'],map_location='cpu',weights_only=False)
    for name in ('release','initial_state','initial_rng'):
        assert hashlib.sha256(Path(inputs[name]).read_bytes()).hexdigest()==inputs[name+'_sha256']
    original_class=gr.MultiGoalReachEnv
    module=None
    if variant=='before':
        source=subprocess.check_output(['git','show',e['start_sha']+':Python/src/auv_madrl/multi_goal_reach_env.py']).decode('utf8')
        module=types.ModuleType('auv_madrl._event_baseline');sys.modules[module.__name__]=module
        exec(compile(source,'baseline/multi_goal_reach_env.py','exec'),module.__dict__)
        cls=module.MultiGoalReachEnv
    else:cls=original_class
    row=dict(kind=kind,variant=variant,status='preparing',actual_ticks=0,workers_before=active_run_identities(),
             pid=os.getpid(),process_start_token=at.process_start_token(os.getpid()),
             source_sha256=hashlib.sha256((source if variant=='before' else Path(gr.__file__).with_name('multi_goal_reach_env.py').read_text('utf8')).encode()).hexdigest(),
             threads=1,interop_threads=1,torch=str(torch.__version__))
    if kind=='fixed':
        assert pool['actual_ticks']+250+sum(r.get('reserved_ticks',0) for r in e['attempts'])<=10000
        row['reserved_ticks']=250;e['attempts'].append(row)
    else:
        assert len(e['windows'])+(len(data['issue122']['viewer_windows']) if issue==124 else 0)<(8 if issue==124 else 6);e['windows'].append(row)
    def save():gr.update_json_atomic(path,{f'issue{issue}':e,'issue121':pool},compact=True)
    save();v=None;owned=set(threading.enumerate());init_start=time.perf_counter()
    def observe():
        record=v._live_record or v._last_complete_record
        result=dict(physical=gr.physical_digest(v.env).hex(),state=digest(v.env.state_dict()),
                    rng=digest((torch.get_rng_state(),random.getstate(),np.random.get_state(),v._policy_noise.tables)),
                    record=digest((record.initial,record.intervals,record.checks)),events=digest(v.event_freeze_frames),
                    actor_hashes={r:v._role_deployment_descriptor(r)['actor_hash'] for r in v.role_controls})
        json.dumps(result,allow_nan=False);return result
    try:
        with tempfile.TemporaryDirectory(prefix='auv-event-') as tmp, patch.object(gr,'MultiGoalReachEnv',cls):
            settings=copy.deepcopy(inputs['settings'])
            settings['time_scale']=gr.FULL_SPEED if kind=='full' else 5.
            with patch.object(gr,'_load_viewer_settings',lambda config:(Path(tmp)/'settings.json',settings)):
                v=gr.GoalReachViewerSession.__new__(gr.GoalReachViewerSession);v.__init__()
            for _,future in tuple(v._deployment_futures.values()):future.result(timeout=60)
            v._poll_viewer_deployments()
            restore_cpu_profile_session(v,inputs,Path(inputs['release']),initial,rng)
            v._refresh_actor_frame();v._publish_snapshot(0.)
            assert int(v.env.physics_ticks[0])==0
            json.loads(v.snapshot.read());row['initial']=observe();row['initialization_wall']=time.perf_counter()-init_start
            assert row['initial']['actor_hashes']=={r:x['actor_hash'] for r,x in inputs['deployment'].items()}
            save()
            if kind=='fixed':
                # Reserve above, count in memory, serialize after the measured production segment.
                wall=0.;steps=[]
                for index in range(250):
                    if bool(v.env.done[0]):break
                    before=int(v.env.physics_ticks[0]);start=time.perf_counter()
                    try:
                        if before%25==0:v._refresh_actor_frame()
                        step=v._advance_viewer_physics(1)
                    finally:
                        wall+=time.perf_counter()-start;row['actual_ticks']+=int(v.env.physics_ticks[0])-before
                    steps.append(digest((v.env.state_dict(),step.__dict__ if step else None)))
                row.update(wall=wall,steps=steps,end=observe(),status='complete')
            else:
                clock={'start':None,'ticks':0,'advance_wall':0.,'pending_wall':0.};timestamps=[];last=[None,False]
                advance=v._advance_viewer_physics;drain=v._drain_commands;store=v.snapshot.store
                class EndWindow(BaseException):pass
                def measured_advance(ticks,deadline=None):
                    until=clock['start']+15
                    if time.perf_counter()>=until:v.running=False;return None
                    before=int(v.env.physics_ticks[0]);start=time.perf_counter()
                    try:return advance(ticks,deadline=min(deadline,until) if deadline else until)
                    finally:clock['ticks']+=int(v.env.physics_ticks[0])-before;clock['advance_wall']+=time.perf_counter()-start
                def measured_store(payload,**kw):
                    if clock['start'] is not None:timestamps.append(time.perf_counter())
                    return store(payload,**kw)
                def measured_drain():
                    now=time.perf_counter()
                    if clock['start'] is None:
                        v._reset_performance_diagnostics();clock['start']=now;last[0]=now;v.running=True;v._next_physics_step_at=0.
                    if last[1]:clock['pending_wall']+=now-last[0]
                    last[:]=[now,bool(v._terminal_environment_update_pending or v._terminal_snapshot_until>now or v._environment_update_future is not None)]
                    if now-clock['start']>=15:
                        wall=now-clock['start'];gaps=np.diff(timestamps)
                        row.update(status='complete',actual_ticks=clock['ticks'],wall=wall,actual_time_scale=clock['ticks']*.02/wall,
                            advance_wall=clock['advance_wall'],pending_wall=clock['pending_wall'],
                            pure_advance_scale=clock['ticks']*.02/clock['advance_wall'] if clock['advance_wall'] else None,
                            snapshot_hz=len(timestamps)/wall,snapshot_gap_p95=float(np.quantile(gaps,.95)),snapshot_gap_p99=float(np.quantile(gaps,.99)),
                            snapshot_times=timestamps,performance=v._performance_snapshot(now))
                        raise EndWindow()
                    # An accepted selection measures command processing without resetting rate diagnostics.
                    if now-clock['start']>=5 and 'command_latency' not in row:
                        t=time.perf_counter();v.submit(dict(type='select_arena',value=0));result=drain()
                        row['command_latency']=time.perf_counter()-t;row['command_error']=v.command_error
                        return result
                    return drain()
                v._advance_viewer_physics=measured_advance;v._drain_commands=measured_drain;v.snapshot.store=measured_store
                sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen();row['port']=sock.getsockname()[1]
                try:v.run(root/'Python/viewer/goal_reach.html',row['port'],reserved_socket=sock)
                except EndWindow:pass
                finally:row['actual_ticks']=clock['ticks']
                row['end']=observe()
    except BaseException as error:
        row.update(status='failed',error=repr(error));raise
    finally:
        if v is not None:
            callback=getattr(v,'_gc_callback',None)
            if callback in gc.callbacks:gc.callbacks.remove(callback)
            for name,executor in vars(v).items():
                if name.endswith('_executor') and executor is not None:executor.shutdown(wait=True,cancel_futures=True)
        if module:sys.modules.pop(module.__name__,None)
        if kind=='fixed':
            pool['actual_ticks']+=row['actual_ticks']
            if issue!=121:e['actual_ticks']+=row['actual_ticks']
            row['reserved_ticks']=0
        for thread in set(threading.enumerate())-owned:
            if thread.is_alive():thread.join(timeout=5)
        row['remaining_threads']=[t.name for t in set(threading.enumerate())-owned if t.is_alive()]
        row['cleanup_complete']=not row['remaining_threads'];row['workers_after']=active_run_identities();save()
        print(json.dumps({k:row.get(k) for k in ('kind','variant','status','actual_ticks','wall','actual_time_scale','snapshot_hz','cleanup_complete','error')}),flush=True)


if __name__=='__main__':
    main()
