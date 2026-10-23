from contextlib import nullcontext, contextmanager, ExitStack
from types import SimpleNamespace
from unittest.mock import Mock
import copy
import gc
import time
import json
from pathlib import Path
import re
import threading
import urllib.request

import pytest
import torch

from auv_madrl import goal_reach as gr
from auv_madrl.controllers import ActorControlMux
from auv_madrl.models import TanhNormal, ClippedNormal, TruncatedNormal
from auv_madrl.live import _expected_desired_velocity
from auv_madrl.viewer_replay import ViewerPolicyNoise, ViewerEpisodeRecord, physical_digest


def test_noise_is_slot_keyed_and_does_not_consume_global_rng():
    before = torch.get_rng_state().clone()
    noise = ViewerPolicyNoise(42, 150)
    expected = noise.at('learner', [0, 9], 4)
    assert torch.equal(before, torch.get_rng_state())
    torch.rand(17)
    assert torch.equal(expected.flip(0), noise.at('learner', [9, 0], 4))
    assert torch.equal(expected, ViewerPolicyNoise(42, 150).at('learner', [0, 9], 4))
    assert not torch.equal(expected, noise.at('defender', [0, 9], 4))
    assert len(noise.tables) == 4


def test_explicit_noise_matches_distribution_transform():
    distribution = TanhNormal(torch.tensor([[.1,.2],[.3,.4]]), torch.ones(2,2))
    policy = Mock(actor_distribution=Mock(return_value=distribution))
    control = ActorControlMux(2,'cpu',policy,default_mode='nn',stochastic_neural=True)
    noise = ViewerPolicyNoise(2,5).at('target',[0,1],0)
    before = torch.get_rng_state().clone()
    active, _ = control.resolve_active(torch.arange(2),torch.zeros(2,2),distribution,noise)
    full, _ = control.resolve({'vector':torch.zeros(2,1)},torch.ones(2,dtype=torch.bool),torch.zeros(2,2),noise)
    assert torch.equal(active, (distribution.loc+distribution.scale*noise).tanh())
    assert torch.equal(full,active) and torch.equal(before,torch.get_rng_state())
    control.set_mode(0,'manual')
    active, _ = control.resolve_active(torch.arange(2),torch.zeros(2,2),distribution,noise)
    assert torch.equal(active[1],full[1])
    control.set_neural_sampling(False)
    active, _ = control.resolve_active(torch.arange(2),torch.zeros(2,2),distribution,noise)
    assert torch.equal(active[1],distribution.deterministic_sample[1])


def test_default_sampling_path_is_unchanged():
    distribution=ClippedNormal(torch.zeros(2,2),torch.ones(2,2))
    control=ActorControlMux(2,'cpu',Mock(actor_distribution=Mock(return_value=distribution)),default_mode='nn',stochastic_neural=True)
    state=torch.get_rng_state()
    expected=distribution.rsample()
    expected_rng=torch.get_rng_state()
    torch.set_rng_state(state)
    actual,_=control.resolve({'vector':torch.zeros(2,1)},torch.ones(2,dtype=torch.bool),torch.zeros(2,2))
    assert torch.equal(actual,expected) and torch.equal(torch.get_rng_state(),expected_rng)


@contextmanager
def _cpu_test_state():
    import random
    import numpy as np
    threads=torch.get_num_threads()
    state=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    try:
        torch.set_num_threads(1)
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        yield
    finally:
        random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
        torch.set_num_threads(threads)


@pytest.fixture
def cpu_test_state():
    with _cpu_test_state():yield


@pytest.fixture
def session(cpu_test_state):
    settings = gr._viewer_settings_defaults()
    settings['training_settings'].update(parallel_environments=1,maximum_episode_decisions=2)
    settings['obstacle_density_mean_per_1000m2'] = settings['obstacle_density_std_per_1000m2'] = 0
    settings['learner_count_range'] = settings['defender_count_range'] = [1,1]
    env, groups, _, _, _ = gr._build_training_group(settings,device='cpu')
    value = gr.GoalReachViewerSession.__new__(gr.GoalReachViewerSession)
    value.env, value.config, value.device, value.viewer_settings = env,env.config,torch.device('cpu'),settings
    value.active_actor_kind = 'learner'
    value.role_controls = {role:ActorControlMux(1,'cpu',None,default_mode='manual') for role in env.policy_roles}
    for role in value.role_controls:
        setattr(value,'last_'+role+'_action',torch.tensor([[-1.,0.]]))
    value._replay_record = value._replay_frame = value._replay_error = None
    value._replay_index = 0
    value._actor_inputs_revision = 0
    value._applied_frame_revision = 0
    value.actor_frame_physics_ticks = 0
    value._invalidate_actor_inputs = Mock()
    value._record_applied_frame = lambda:dict(inputs={},tick=0,modes={r:list(c.modes) for r,c in value.role_controls.items()},sources={})
    value._live_record = ViewerEpisodeRecord(env.state_dict(),dict(episode=1),value._replay_contract())
    value._last_complete_record = None
    value.running = True
    return value


def test_record_and_replay_production_physics_with_different_chunks(session):
    states = {}
    while not bool(session.env.done[0]):
        session._advance_viewer_physics(8)
        states[int(session.env.physics_ticks[0])] = copy.deepcopy(session.env.state_dict())
    record = session._last_complete_record
    assert record.complete and not record.error
    expected = physical_digest(session.env)
    session.env.load_state_dict(copy.deepcopy(record.initial))
    session._replay_record = record
    session._live_record = None
    while not bool(session.env.done[0]) and not session._replay_error:
        session._advance_viewer_physics(1)
    def differences(a, b, path=''):
        if isinstance(a, torch.Tensor):
            return [] if torch.equal(a,b) else [f'{path}: {a} != {b}']
        if isinstance(a, dict):
            return [d for k in a for d in differences(a[k],b[k],path+'.'+str(k))]
        return [] if a == b else [path]
    assert session._replay_error is None, differences(states[int(session.env.physics_ticks[0])], session.env.state_dict())
    assert physical_digest(session.env) == expected
    assert session._last_complete_record is record


def test_record_is_bounded_and_does_not_record_zero_ticks():
    record = ViewerEpisodeRecord({}, {}, 'contract')
    record.append(0,0,{},0,{},b'')
    assert not record.intervals
    record.BYTE_LIMIT = 100
    record.append(0,1,{'learner':torch.zeros(1,2)},0,{},b'check')
    assert record.error and not record.complete and not record.intervals


def test_deadline_records_only_actual_ticks(session):
    session._advance_viewer_physics(8,time.perf_counter()-1)
    tick = int(session.env.physics_ticks[0])
    assert 0 < tick < 8
    assert session._live_record.intervals[-1]['end'] == tick


@pytest.fixture(params=[1,2])
def full_session(tmp_path, monkeypatch,request,cpu_test_state):
    settings = gr._viewer_settings_defaults()
    settings['training_settings'].update(parallel_environments=1, maximum_episode_decisions=2)
    settings.update(obstacle_density_mean_per_1000m2=0, obstacle_density_std_per_1000m2=0, learner_count_range=[1,1], defender_count_range=[1,1],
                    target_count_range=[request.param,request.param],policy_sampling='stochastic')
    monkeypatch.setattr(gr, '_load_viewer_settings', lambda config:(tmp_path/'settings.json',settings))
    with _owned_cpu_session() as value:
        yield value


def test_async_reset_coalesces_pause_and_recovers_first_fault(full_session, monkeypatch):
    from concurrent.futures import Future
    value = full_session
    old = value.env
    pending = Future()
    original = value._environment_update_executor.submit
    monkeypatch.setattr(value._environment_update_executor, 'submit', lambda *a, **k: pending)
    value.running = True
    value._reset()
    generation = value._environment_update_generation
    value._reset()
    assert value._environment_update_generation == generation and value.env is old
    value.submit({'type':'run','value':False})
    value._drain_commands()
    pending.set_exception(ValueError('candidate reset fixture failure'))
    assert value._poll_environment_update()
    assert value.env is old and not value.running
    assert value.reset_progress['phase'] == 'failed'
    assert 'candidate reset fixture failure' in value.reset_progress['traceback']
    monkeypatch.setattr(value._environment_update_executor, 'submit', original)
    value._reset()
    value._environment_update_future.result(timeout=20)
    value._poll_environment_update()
    assert value.env is not old and not value.running
    assert value.reset_progress['phase'] == 'ready'


@pytest.mark.parametrize('stage',['submit','prepare','controls','initial','frame'])
def test_reset_failure_restores_replay_and_complete_runtime(full_session,monkeypatch,stage):
    value=full_session
    while not bool(value.env.done[0]):
        value._refresh_actor_frame();value._advance_viewer_physics(8)
    value._start_replay()
    old=value.env;record=value._replay_record;controls=value.role_controls
    old_inputs=copy.deepcopy(value._actor_inputs());old_trajectories=copy.deepcopy(value.trajectories)
    old_revision=value._viewer_episode_revision
    old_models={r:c.neural_policy.model for r,c in controls.items()}
    generation=value._environment_update_generation
    with monkeypatch.context() as patch:
        if stage=='submit':patch.setattr(value._environment_update_executor,'submit',Mock(side_effect=ValueError('injected submit')))
        elif stage=='prepare':patch.setattr(value,'_build_environment_update',Mock(side_effect=ValueError('injected prepare')))
        else:
            if stage=='controls':
                patch.setattr(value,'_environment_capacity_signature',lambda env:id(env))
                name='_initialize_actor_runtime_state'
            else:name='_remember_episode_initial_state' if stage=='initial' else '_refresh_actor_frame'
            original=getattr(value,name)
            def fail(*a,**k):
                original(*a,**k)
                raise ValueError('injected '+stage)
            patch.setattr(value,name,fail)
        value._reset()
        future=value._environment_update_future
        if future is not None:
            try:future.result(timeout=20)
            except ValueError:pass
            value._poll_environment_update()
    assert value.env is old and value._replay_record is record
    assert value.role_controls is controls and value._actor_inputs()==old_inputs
    assert value.trajectories==old_trajectories and value._viewer_episode_revision==old_revision
    assert all(value.role_controls[r].neural_policy.model is model for r,model in old_models.items())
    assert value.reset_progress['phase']=='failed' and 'injected '+stage in value.reset_progress['error']
    assert value.reset_progress['generation']>generation and not value.running
    assert value._environment_reset_backup is None and value._environment_update_future is None
    value._publish_snapshot(0)
    failed=_http_snapshot(value)
    assert failed['goal_reach_viewer']['reset']['phase']=='failed'
    assert not failed['goal_reach_viewer']['environment_update_pending']
    value._reset();value._environment_update_future.result(timeout=20);value._poll_environment_update()
    assert value.reset_progress['phase']=='ready' and value._replay_record is None and not value.running
    value._publish_snapshot(0)
    assert _http_snapshot(value)['goal_reach_viewer']['reset']['phase']=='ready'


def _http_snapshot(value):
    handler=gr._handler_factory(Path(gr.__file__).parents[2]/'viewer'/'goal_reach.html',
        value.snapshot.read,value.submit,value._read_training_metrics,value._read_training_lineage)
    server=gr.ThreadingHTTPServer(('127.0.0.1',0),handler)
    worker=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True)
    worker.start()
    try:
        base='http://127.0.0.1:'+str(server.server_port)
        with urllib.request.urlopen(base,timeout=3) as response:html=response.read().decode()
        token=re.search(r"const viewerInstanceToken = '([^']+)'",html).group(1)
        with urllib.request.urlopen(base+'/api/snapshot?viewer_instance='+token,timeout=3) as response:
            assert response.status==200
            return json.load(response)
    finally:
        server.shutdown();server.server_close();worker.join()


def test_single_role_deployment_hash_tracks_loaded_model_and_failed_candidate(full_session):
    from concurrent.futures import Future
    value=full_session
    before={r:gr.actor_state_hash(c.neural_policy.model.state_dict()) for r,c in value.role_controls.items()}
    value._queue_viewer_deployment(actor_kind='learner',model_id='random_current_architecture',
        architecture_id=value.active_actor_variant_id,source_kind='random',selection_mode='random',persist=False)
    candidate=value._deployment_futures['learner'][1].result(timeout=10)
    with torch.no_grad():
        candidate.model.actor_parameters()[0].add_(.125)
    value._poll_viewer_deployments()
    current=value._role_deployment_descriptor('learner')
    loaded=gr.actor_state_hash(value.role_controls['learner'].neural_policy.model.state_dict())
    assert current['actor_hash']==loaded and not current['pending'] and current['error'] is None
    assert loaded!=before['learner']
    for role in ('defender','target'):
        assert gr.actor_state_hash(value.role_controls[role].neural_policy.model.state_dict())==before[role]
    failed=Future();failed.set_exception(ValueError('invalid candidate schema'))
    value._deployment_generations['learner']+=1
    value._deployment_futures['learner']=(value._deployment_generations['learner'],failed)
    value._poll_viewer_deployments()
    rejected=value._role_deployment_descriptor('learner')
    assert rejected['actor_hash']==loaded and not rejected['pending']
    assert rejected['error']=='invalid candidate schema'


def test_settings_command_capacity_and_first_actions(full_session):
    value=full_session
    for target,mode,learner,defender in [(3,'dynamic',2,2),(1,'dynamic',2,2),
                                       (1,'static',2,2),(3,'dynamic',2,2),(1,'dynamic',1,1)]:
        models={r:c.neural_policy.model for r,c in value.role_controls.items()}
        value.selected_actor_kind='target';value.selected_actor_index=value.env.target_capacity-1
        value.submit(dict(type='set_viewer_settings',client_request_id=f'capacity-{target}-{mode}-{learner}',
            target_count_range=[target,target],target_mode=mode,
            learner_count_range=[learner,learner],defender_count_range=[defender,defender]))
        value._drain_commands()
        if value._training_scene_load_future is not None:
            value._training_scene_load_future.result(timeout=20);value._poll_training_scene_load()
        assert value._viewer_settings_result['phase']=='applied',value._viewer_settings_result
        for role in value.env.policy_roles:
            capacity=value.env.role_state(role).position.shape[1]
            assert value.role_controls[role].capacity==capacity
            assert getattr(value,'last_'+role+'_action').shape==(capacity,2)
            assert len(value.trajectories[role])==capacity
            if role in models:assert value.role_controls[role].neural_policy.model is models[role]
        value._refresh_actor_frame();value._advance_viewer_physics(1)
        assert 0<=value.selected_actor_index<value.env.role_state(value.selected_actor_kind).position.shape[1]
    value.env.learner_active.zero_();value._refresh_actor_frame()
    assert value.actor_frames['learner'].indices.numel()==0 and value.control.capacity==1


def test_reset_poll_rebuilds_target_only_capacity(full_session):
    value=full_session
    for capacity in (3,1):
        model=value.target_control.neural_policy.model
        value.viewer_settings['target_count_range']=[capacity,capacity]
        value._reset();value._environment_update_future.result(timeout=20);value._poll_environment_update()
        assert value.reset_progress['phase']=='ready'
        assert value.env.target_capacity==value.target_control.capacity==capacity
        assert value.last_target_action.shape==(capacity,2) and len(value.trajectories['target'])==capacity
        assert value.target_control.neural_policy.model is model
        value._refresh_actor_frame();value._advance_viewer_physics(1)


def test_superseded_reset_future_cannot_restore_old_generation(full_session,monkeypatch):
    from concurrent.futures import Future
    value=full_session;old=value.env;late=Future()
    submit=value._environment_update_executor.submit
    monkeypatch.setattr(value._environment_update_executor,'submit',lambda *a,**k:late)
    value.running=True;value._reset()
    first=value._environment_update_generation
    value._reset(episode_index=int(old.episode_counters[0])+2)
    second=value._environment_update_generation
    value.submit({'type':'run','value':False});value._drain_commands()
    monkeypatch.setattr(value._environment_update_executor,'submit',submit)
    late.set_exception(ValueError('obsolete worker'))
    assert not value._poll_environment_update() and value.env is old
    assert second>first and value._environment_reset_request['generation']==second
    assert value._environment_reset_backup is None
    value._environment_update_future.result(timeout=20);value._poll_environment_update()
    assert value.reset_progress==dict(phase='ready',generation=second)
    assert not value.running and value._environment_reset_backup is None


def test_event_freeze_limit_matches_enabled_producer(full_session,monkeypatch):
    value=full_session
    value.viewer_settings['target_count_range']=[3,3]
    value._reset();value._environment_update_future.result(timeout=20);value._poll_environment_update()
    capacity=value.env._viewer_event_tick.shape[1]
    frames=[dict(physics_ticks=1,event_type='contact',actor_kind='target',actor_id=i)
            for i in range(capacity)]
    monkeypatch.setattr(value.env,'drain_viewer_event_freeze_frames',lambda _:frames)
    value._record_event_freeze_frames();value._record_event_freeze_frames()
    assert len(value.event_freeze_frames)==capacity


def test_settings_scene_commit_failure_restores_sources_and_recovers(full_session,monkeypatch):
    value=full_session;old=value.env;controls=value.role_controls
    sources=copy.deepcopy(value._deployed_sources);trajectories=copy.deepcopy(value.trajectories)
    value.submit(dict(type='set_viewer_settings',target_count_range=[3,3],client_request_id='reject-frame'))
    value._drain_commands();value._training_scene_load_future.result(timeout=20)
    original=value._refresh_actor_frame
    with monkeypatch.context() as patch:
        def fail():
            original();raise ValueError('settings first frame rejected')
        patch.setattr(value,'_refresh_actor_frame',fail)
        value._poll_training_scene_load()
    assert value.env is old and value.role_controls is controls
    assert value._deployed_sources==sources and value.trajectories==trajectories
    assert value._viewer_settings_result['phase']=='failed' and not value.running
    value._publish_snapshot(0)
    failed=_http_snapshot(value)
    assert not failed['settings_pending'] and failed['requested_viewer_settings'] is None
    assert failed['goal_reach_viewer']['settings_command_result']['phase']=='failed'
    value.submit(dict(type='set_viewer_settings',target_count_range=[3,3],client_request_id='recover-frame'))
    value._drain_commands();value._training_scene_load_future.result(timeout=20);value._poll_training_scene_load()
    assert value._viewer_settings_result['phase']=='applied' and value.reset_progress['phase']=='ready'
    assert value._viewer_fault is None and value.target_control.capacity==3


def test_full_stochastic_record_metadata_and_replay_without_inference(full_session, monkeypatch):
    value = full_session
    while not bool(value.env.done[0]):
        if int(value.env.physics_ticks[0]) % 25 == 0:
            value._refresh_actor_frame()
        value._advance_viewer_physics(8)
    record = value._last_complete_record
    expected = physical_digest(value.env)
    assert record.complete
    assert record.intervals[0]['frame']['inputs']['learners']
    for control in value.role_controls.values():
        monkeypatch.setattr(control.neural_policy.model, 'actor_distribution_from_observation',
                            Mock(side_effect=AssertionError('Replay sampled a model')))
    value._start_replay()
    assert not value.running and int(value.env.physics_ticks[0]) == 0
    assert value._actor_inputs() == record.intervals[0]['frame']['inputs']
    while not bool(value.env.done[0]) and not value._replay_error:
        value._refresh_actor_frame()
        value._advance_viewer_physics(1)
    assert not value._replay_error
    assert physical_digest(value.env) == expected
    value._start_replay()
    assert value._replay_record is record and not value.running


def test_decision_context_is_atomic_with_encoded_revision(full_session, monkeypatch):
    value = full_session
    monkeypatch.setattr(torch.optim.Adam, 'step', Mock(side_effect=AssertionError('No optimizer')))
    value._cache_actor_inputs(1.0, force=True)
    old = copy.deepcopy(value.encoded_actor_inputs)
    revision = value._encoded_actor_inputs_revision
    frame = value.actor_frames['learner']
    original_context = copy.deepcopy(frame.display_context)
    original_action = frame.action.clone()
    # Pending B does not mean runtime A changed; the descriptor exposes both identities.
    value._deployment_requested_models = {'learner': 'B'}
    value._pending_deployments['learner'] = Mock()
    descriptor = value._role_deployment_descriptor('learner')
    assert descriptor['pending'] and descriptor['selected_model_id'] == 'B'
    assert descriptor['model_id'] == original_context['deployment']['model_id']
    value._invalidate_actor_inputs()
    value._cache_actor_inputs(1.01)
    assert value.encoded_actor_inputs == old and value._encoded_actor_inputs_revision == revision
    value._cache_actor_inputs(1.01, force=True)
    row = value.encoded_actor_inputs['learners'][0]
    assert row['display_context'] == original_context
    assert torch.equal(frame.action, original_action)
    assert row['physics_ticks'] == old['learners'][0]['physics_ticks']
    assert 'pending' not in row['display_context']['deployment']
    value._pending_deployments.clear()
    # A source/sampling change at the same tick must produce a new decision context.
    value.control.set_neural_sampling(False)
    value._resolve_actor_frame_actions()
    value._cache_actor_inputs(1.02)
    assert value.encoded_actor_inputs['learners'][0]['display_context'] == original_context
    value._cache_actor_inputs(1.02, force=True)
    current = value.encoded_actor_inputs['learners'][0]
    assert current['display_context']['deployment']['sampling'] == 'deterministic'
    assert current['display_context']['source_key'] != original_context['source_key']
    assert current['physics_ticks'] == row['physics_ticks']
    assert old['learners'][0]['display_context'] == original_context
    # Publish must not reconstruct frame provenance from a later runtime descriptor.
    monkeypatch.setattr(value, '_role_deployment_descriptor', Mock(side_effect=AssertionError('late descriptor')))
    saved = value._record_applied_frame()
    value.actor_frames['learner'].display_context['deployment']['model_id'] = 'changed-after-record'
    assert saved['sources']['learner']['model_id'] != 'changed-after-record'
    # The producer owns the immutable frame; the record shares it.
    record = ViewerEpisodeRecord({}, {}, 'metadata')
    record.append(0, 1, {'learner': torch.zeros(1, 2)}, 1, saved, b'check')
    recorded = copy.deepcopy(record.intervals[0]['frame']['inputs'])
    assert record.intervals[0]['frame'] is saved
    value.actor_frames['learner'].display_context['deployment']['model_id'] = 'later'
    value._replay_record, value._replay_frame = record, record.intervals[0]['frame']
    assert value._actor_inputs() == recorded
    assert value._actor_inputs()['learners'][0]['display_context']['deployment']['model_id'] != 'later'


def test_browser_render_trace_has_exact_live_replay_terminal(full_session, tmp_path):
    """Small real CPU Viewer record for offline production-HTML/export verification."""
    value=full_session
    frames=[]
    value.running=True
    value._publish_snapshot(0);frames.append(json.loads(value.snapshot.read()))
    while not bool(value.env.done[0]):
        if int(value.env.physics_ticks[0])%25==0:value._refresh_actor_frame()
        value._advance_viewer_physics(min(5,25-int(value.env.physics_ticks[0])%25))
        value._record_trajectory_positions()
        value._publish_snapshot(0);frames.append(json.loads(value.snapshot.read()))
    terminal=physical_digest(value.env)
    record=value._last_complete_record
    assert record.complete and frames[-1]['arenas'][0]['physics_ticks']==int(value.env.physics_ticks[0])
    value._start_replay()
    while not bool(value.env.done[0]) and not value._replay_error:
        value._refresh_actor_frame();value._advance_viewer_physics(1);value._record_trajectory_positions()
    value._publish_snapshot(0)
    replay=json.loads(value.snapshot.read())
    assert not value._replay_error and physical_digest(value.env)==terminal
    assert replay['goal_reach_viewer']['replay']['active']
    assert replay['arenas'][0]['physics_ticks']==frames[-1]['arenas'][0]['physics_ticks']
    path=tmp_path/'browser-render-trace.json'
    path.write_text(json.dumps({'provenance':'isolated CPU fixture; P1, 2 decisions, random weights; not formal training or research sampling',
                                'frames':frames,'replay':replay}),encoding='utf-8')
    print(path)


def test_multi_target_edited_scene_and_evaluation_preserve_identity(full_session):
    value=full_session
    saved=value._capture_edited_scene()
    expected=value.env.targets.position.clone()
    value._edited_scene=True
    value._save_evaluation_scene()
    preset=value.viewer_settings['evaluation_scene_preset']
    assert preset['edited_scene']['target_count']==value.env.target_capacity
    value._apply_preserved_edited_scene(preset['edited_scene'])
    assert torch.equal(value.env.targets.position,expected)
    assert value._capture_edited_scene()['target_positions']==saved['target_positions']


def test_fixed_scene_obstacle_edit_commit_and_delete_publish_physical_geometry(full_session):
    value = full_session
    # Saving/restoring an edited scene is the production entry into fixed geometry.
    saved=value._capture_edited_scene()
    x,z=value._viewer_sensing_centers()[0]
    saved['obstacle_centers']=[(x+3.,z)]
    saved['obstacle_radii']=[.3]
    value._apply_preserved_edited_scene(saved)
    obstacles = value._obstacle_snapshot()
    source = next(o for o in obstacles if value._viewer_point_is_sensed((o['x'], o['z']), radius=o['r']))
    key = source['id']
    x, z, radius = source['x'], source['z'], source['r']
    value._commit_edit(dict(edit_kind='obstacle', index=key, mode='replace',
                           x=x, z=z, radius=radius / 2))
    changed = next(o for o in value._obstacle_snapshot() if o['id'] == key)
    assert changed['r'] == pytest.approx(radius / 2)
    count = int(value.env.obstacle_active.sum())
    value._commit_edit(dict(edit_kind='obstacle', index=key, mode='replace', duplicate=True,
                           x=x, z=z, radius=radius / 3))
    assert int(value.env.obstacle_active.sum()) == count + 1
    value._delete_circle(dict(edit_kind='obstacle', index=key))
    assert int(value.env.obstacle_active.sum()) == count
    assert all(o['id'] != key for o in value._obstacle_snapshot())
    value._publish_snapshot(0)
    assert json.loads(value.snapshot.read())['world_descriptor']['manual_fixed']


@pytest.mark.parametrize('family',[TanhNormal, ClippedNormal, TruncatedNormal])
def test_distribution_explicit_noise_uses_own_transform(family):
    distribution = family(torch.tensor([[.2, -.3]]),torch.tensor([[.7,.8]]))
    noise = torch.tensor([[.4,-.6]])
    before = torch.get_rng_state().clone()
    actual = distribution.from_standard_noise(noise)
    assert torch.equal(before,torch.get_rng_state())
    if family is TruncatedNormal:
        expected = distribution._from_uniform(torch.special.ndtr(noise.double()))
    else:
        raw = distribution.loc+distribution.scale*noise
        expected = raw.tanh() if family is TanhNormal else raw.clamp(-1,1)
    assert torch.equal(actual,expected)


def test_clipped_expected_and_snapshot_read_only(full_session):
    assert _expected_desired_velocity(torch.zeros(2),torch.ones(2),True,4,.4,'clipped_normal') == pytest.approx([2,0])
    assert _expected_desired_velocity(torch.tensor([2.,-2.]),torch.ones(2),False,4,.4,'clipped_normal') == pytest.approx([4,-.4])
    value=full_session
    before={r:getattr(value,'last_'+r+'_action').clone() for r in value.role_controls}
    seed=value._policy_noise.seed
    rng=torch.get_rng_state().clone()
    for _ in range(3):
        value._publish_snapshot(0)
        value._actor_inputs()
    assert not value._live_record.intervals
    assert seed==value._policy_noise.seed and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(a,getattr(value,'last_'+r+'_action')) for r,a in before.items())


def test_early_terminal_replay(session):
    session.env.learners.position[0,0] = session.env.targets.position[0,0]
    session.env._bind_primary_learner()
    session._live_record = ViewerEpisodeRecord(session.env.state_dict(),{},session._replay_contract())
    session._advance_viewer_physics(8)
    assert bool(session.env.done[0]) and int(session.env.physics_ticks[0]) < 25
    record = session._last_complete_record
    expected = physical_digest(session.env)
    session.env.load_state_dict(copy.deepcopy(record.initial))
    session._replay_record,session._live_record = record,None
    session._advance_viewer_physics(8)
    assert not session._replay_error and physical_digest(session.env) == expected


def test_zero_ticks_tamper_and_manual_middecision(session, monkeypatch):
    original = session.env._step_training_learner
    monkeypatch.setattr(session.env,'_step_training_learner',lambda *a,**kw:None)
    session._advance_viewer_physics(8)
    assert not session._live_record.intervals
    monkeypatch.setattr(session.env,'_step_training_learner',original)
    session._advance_viewer_physics(3)
    session.last_learner_action[0] = torch.tensor([.5,.3])
    # The production manual-control path commits a new applied frame, not
    # merely an encoded-input/display revision.
    session._applied_frame_revision += 1
    while not bool(session.env.done[0]):session._advance_viewer_physics(8)
    record = session._last_complete_record
    assert record.intervals[1]['start'] == 3
    assert not torch.equal(record.intervals[0]['actions']['learner'],record.intervals[1]['actions']['learner'])
    tick = min(record.checks)
    record.checks[tick] = b'tampered'
    session.env.load_state_dict(copy.deepcopy(record.initial))
    session._replay_record,session._live_record = record,None
    while not session._replay_error:session._advance_viewer_physics(1)
    assert int(session.env.physics_ticks[0]) == tick and not session.running


def test_replay_lifecycle_lock_and_async_isolation(full_session):
    value=full_session
    initial=physical_digest(value.env)
    seed=value._policy_noise.seed
    value.commands.append(dict(type='set_scene_lock',value=True));value._drain_commands()
    value._advance_viewer_physics(3)
    value._reset()
    value._environment_update_future.result(timeout=10)
    value._poll_environment_update()
    assert physical_digest(value.env)==initial and value._policy_noise.seed!=seed
    while not bool(value.env.done[0]):value._advance_viewer_physics(8)
    record=value._last_complete_record
    value._reset()
    value._environment_update_future.result(timeout=10)
    value._poll_environment_update()
    cursor=int(value.env.episode_counters[0]);seed=value._policy_noise.seed
    value._start_replay()
    assert value._policy_noise.seed==seed
    value._deployment_futures['learner']=(0,Mock(done=Mock(side_effect=AssertionError('late deployment read'))))
    assert not value._poll_viewer_deployments()
    value._follow_training_checkpoints({})
    value._deployment_futures.clear()
    value.commands.append(dict(type='manual_action',actor='learner',index=0,surge=1,yaw=0))
    with pytest.raises(ValueError,match='LIVE'):value._drain_commands()
    value._reset()
    value._environment_update_future.result(timeout=10)
    value._poll_environment_update()
    assert value._replay_record is None and int(value.env.episode_counters[0])==cursor
    assert value._last_complete_record is record and value._policy_noise.seed!=seed
    value.viewer_settings['sense_radius_m']+=1
    assert not value._replay_status()['available']


def test_failed_target_configuration_preserves_scene_lock_and_record(full_session):
    value=full_session
    value.commands.append(dict(type='set_scene_lock',value=True));value._drain_commands()
    initial=physical_digest(value.env)
    record=value._live_record
    with pytest.raises(ValueError):value._set_viewer_settings({'target_count_range':[1,2.5]})
    assert value.scene_locked and physical_digest(value.env)==initial
    assert value._live_record is record


def test_required_trajectory_points_exceed_soft_budget_without_dropping_events(full_session):
    indices=full_session._uniform_trajectory_indices(200,set(range(100)))
    assert len(indices)>gr.VIEWER_TRAJECTORY_MAX_POINTS
    assert set(range(100)).issubset(indices)
    snapshot=full_session._trajectory_snapshot()
    assert snapshot['point_limit']>=gr.VIEWER_TRAJECTORY_MAX_POINTS
    assert len(snapshot['sample_physics_ticks'])<=snapshot['point_limit']


def test_terminal_replay_accepts_capture_pause_but_not_restart(full_session):
    value = full_session
    value._replay_record = object()
    value.env.done[0] = True
    value.commands.append(dict(type='run', value=False, client_request_id='capture-end'))
    value._drain_commands()
    assert not value.running
    assert value._run_request_id == 'capture-end'
    assert value.command_error is None
    value.commands.append(dict(type='run', value=True))
    with pytest.raises(ValueError, match='Replay is finished'):
        value._drain_commands()


def test_stale_reset_failure_cannot_replace_new_generation(full_session,monkeypatch):
    from concurrent.futures import Future
    value=full_session; old=value.env; pending=Future()
    original=value._environment_update_executor.submit
    monkeypatch.setattr(value._environment_update_executor,'submit',lambda *a,**k:pending)
    value._reset();value._reset(episode_index=7)
    pending.set_exception(AssertionError('stale candidate failure'))
    monkeypatch.setattr(value._environment_update_executor,'submit',original)
    assert not value._poll_environment_update()
    assert value.env is old and value.command_error is None
    value._environment_update_future.result(timeout=20);value._poll_environment_update()
    assert int(value.env.episode_counters[0])==7 and value.reset_progress['phase']=='ready'


def test_edited_terminal_respawn_uses_independent_candidate(full_session):
    value=full_session; initial=copy.deepcopy(value._current_episode_initial_state['environment'])
    value._edited_scene=True;old=value.env
    value._respawn_actor_in_edited_scene()
    assert value.env is old and value.reset_progress['phase']=='preparing'
    value._environment_update_future.result(timeout=20);value._poll_environment_update()
    assert value.env is not old and value._edited_scene
    for key in ('physics_ticks','episode_counters'):
        assert torch.equal(getattr(value.env,key),initial[key])


@pytest.mark.parametrize('outcome', ['prepare', 'frame', 'success'])
def test_pending_reset_keeps_later_controls_and_http(full_session, monkeypatch, outcome):
    from concurrent.futures import Future
    value = full_session
    old = value.env
    value.submit(dict(type='set_policy_sampling', mode='deterministic')); value._drain_commands()
    value._publish_snapshot(0)
    inputs = copy.deepcopy(value._actor_inputs())
    trajectories = copy.deepcopy(value.trajectories)
    initial_revision = value.rate_revision
    submit = value._environment_update_executor.submit
    pending = Future()
    monkeypatch.setattr(value._environment_update_executor, 'submit', lambda *a, **k: pending)
    value._reset()
    generation = value._environment_update_generation
    assert value._environment_reset_backup is None
    for command in [
        dict(type='set_rate', name='time_scale', value=10),
        dict(type='set_main_view_frame', value='local'),
        dict(type='set_main_view_zoom_step', value=1),
        dict(type='set_policy_sampling', mode='stochastic'),
        dict(type='set_mode', actor='learner', mode='manual'),
        dict(type='manual_action', actor='learner', index=0, surge=.3, yaw=.2),
        dict(type='run', value=True, client_request_id='pending-run'),
    ]:
        value.submit(command)
        value._drain_commands()
    assert not value.running and value._resume_after_environment_update
    assert value.env is old and value._actor_inputs() == inputs
    value.submit(dict(type='run', value=False, client_request_id='latest-pause'))
    value._drain_commands()
    latest_revision = value.rate_revision
    assert latest_revision > initial_revision
    persisted = json.loads(value.viewer_settings_path.read_text())
    assert persisted['time_scale'] == 10 and persisted['main_view_frame'] == 'local'
    # A distinct training receipt arriving during preparation belongs to that request.
    value._training_command_results['fixture'] = dict(phase='applied', client_request_id='independent')
    with monkeypatch.context() as patch:
        if outcome == 'prepare':
            pending.set_exception(ValueError('interleaved worker failure'))
        else:
            config, cls, extra = value._environment_specification(value.active_actor_variant_id, settings=copy.deepcopy(value.viewer_settings))
            candidate = value._build_environment_update(generation, value.active_actor_variant_id,
                int(old.episode_counters[0])+1, config, cls, extra, False, None)
            pending.set_result(candidate)
            if outcome == 'frame':
                patch.setattr(value, '_refresh_actor_frame', Mock(side_effect=ValueError('interleaved frame failure')))
        value._poll_environment_update()
    assert value.rate_revision == latest_revision
    assert value.viewer_settings['time_scale'] == 10
    assert value.viewer_settings['main_view_frame'] == 'local'
    assert value.control.source_names()[0] == 'manual' and value.control.stochastic_neural
    assert torch.allclose(value.control.manual_action[0], torch.tensor([.3,.2]))
    assert json.loads(value.viewer_settings_path.read_text()) == persisted
    assert not value.running
    assert value._training_command_results['fixture']['client_request_id'] == 'independent'
    if outcome != 'success':
        assert value.env is old and value.trajectories == trajectories and value._actor_inputs() == inputs
        assert value.reset_progress['phase'] == 'failed'
        with monkeypatch.context() as patch:
            patch.setattr(value, '_refresh_actor_frame', Mock(side_effect=AssertionError('fault inference')))
            value.submit(dict(type='set_main_view_zoom_step', value=2)); value._drain_commands()
            value._publish_snapshot(0)
        snapshot = _http_snapshot(value)
        assert snapshot['viewer_settings']['main_view_zoom_step'] == 2
        assert snapshot['goal_reach_viewer']['reset']['phase'] == 'failed'
        assert not snapshot['goal_reach_viewer']['environment_update_pending']
        assert snapshot['goal_reach_viewer']['training_command_results']['fixture']['client_request_id'] == 'independent'
        monkeypatch.setattr(value._environment_update_executor, 'submit', submit)
        value._reset(); value._environment_update_future.result(timeout=20); value._poll_environment_update()
    assert value.reset_progress['phase'] == 'ready' and value.env is not old
    assert torch.allclose(value.actor_frames['learner'].action[0], torch.tensor([.3,.2]))
    value._publish_snapshot(0)
    snapshot = _http_snapshot(value)
    assert snapshot['viewer_settings']['time_scale'] == 10
    assert snapshot['goal_reach_viewer']['rate_revision'] == latest_revision
    assert snapshot['goal_reach_viewer']['run_request_id'] == 'latest-pause'


def test_replay_pending_reset_retains_restrictions_and_latest_preferences(full_session, monkeypatch):
    from concurrent.futures import Future
    value=full_session
    while not bool(value.env.done[0]):
        value._refresh_actor_frame(); value._advance_viewer_physics(8)
    value._start_replay()
    record=value._replay_record; frame=value._replay_frame
    source=value._replay_status()['source']
    submit=value._environment_update_executor.submit; pending=Future()
    monkeypatch.setattr(value._environment_update_executor,'submit',lambda *a,**k:pending)
    value._reset()
    assert value._replay_record is record and value._replay_frame is frame
    value.submit(dict(type='set_rate',name='time_scale',value=10));value._drain_commands()
    value.submit(dict(type='set_main_view_frame',value='local'));value._drain_commands()
    modes=list(value.control.modes)
    value.submit(dict(type='set_mode',actor='learner',mode='manual'))
    with pytest.raises(ValueError,match='LIVE'):value._drain_commands()
    assert value.control.modes==modes
    pending.set_exception(ValueError('replay candidate failure'));value._poll_environment_update()
    assert value._replay_record is record and value._replay_frame is frame
    assert value._replay_status()['source']==source
    assert value.time_scale==10 and value.viewer_settings['main_view_frame']=='local'
    monkeypatch.setattr(value._environment_update_executor,'submit',submit)
    value._reset();value._environment_update_future.result(timeout=20);value._poll_environment_update()
    assert value.reset_progress['phase']=='ready' and value._replay_record is None


@pytest.mark.parametrize('fail', [False, True])
def test_deployment_during_reset_survives_without_rewriting_applied_frame(full_session, monkeypatch, fail):
    from concurrent.futures import Future
    value=full_session; pending=Future()
    value._publish_snapshot(0)
    frozen=copy.deepcopy(value._actor_inputs())
    monkeypatch.setattr(value._environment_update_executor,'submit',lambda *a,**k:pending)
    value._reset(); generation=value._environment_update_generation
    value._queue_viewer_deployment(actor_kind='learner',model_id='random_current_architecture',
        architecture_id=value.active_actor_variant_id,source_kind='random',selection_mode='random',persist=False)
    candidate=value._deployment_futures['learner'][1].result(timeout=10)
    with torch.no_grad():candidate.model.actor_parameters()[0].add_(.125)
    value._poll_viewer_deployments()
    loaded=value.control.neural_policy.model
    assert loaded is candidate.model and value._actor_inputs()==frozen
    if fail:
        pending.set_exception(ValueError('reset after deploy'))
    else:
        config,cls,extra=value._environment_specification(value.active_actor_variant_id,settings=copy.deepcopy(value.viewer_settings))
        pending.set_result(value._build_environment_update(generation,value.active_actor_variant_id,
            int(value.env.episode_counters[0])+1,config,cls,extra,False,None))
    value._poll_environment_update()
    assert value.control.neural_policy.model is loaded
    value._publish_snapshot(0)
    snapshot=_http_snapshot(value)
    assert snapshot['role_deployments']['learner']['actor_hash']==gr.actor_state_hash(loaded.state_dict())
    if fail:assert value._actor_inputs()==frozen
    else:assert value.actor_frames['learner'].display_context['deployment']['actor_hash']==gr.actor_state_hash(loaded.state_dict())


@pytest.mark.parametrize('late_failure', [False, True])
def test_apply_supersedes_pending_reset_without_late_rollback(full_session, monkeypatch, late_failure):
    from concurrent.futures import Future
    value=full_session; late=Future(); late.set_running_or_notify_cancel()
    monkeypatch.setattr(value._environment_update_executor,'submit',lambda *a,**k:late)
    value._reset(); generation=value._environment_update_generation
    if not late_failure:
        config,cls,extra=value._environment_specification(value.active_actor_variant_id,settings=copy.deepcopy(value.viewer_settings))
        candidate=value._build_environment_update(generation,value.active_actor_variant_id,
            int(value.env.episode_counters[0])+1,config,cls,extra,False,None)
    value.submit(dict(type='set_viewer_settings',target_count_range=[3,3],client_request_id='superseding-apply'))
    value._drain_commands(); value._training_scene_load_future.result(timeout=20); value._poll_training_scene_load()
    assert value._viewer_settings_result['phase']=='applied'
    current=value.env; current_generation=value._environment_update_generation
    if late_failure:late.set_exception(ValueError('obsolete R'))
    else:late.set_result(candidate)
    value._poll_environment_update()
    assert value.env is current and value.env.target_capacity==3
    assert value._environment_update_generation==current_generation>generation
    assert value._environment_reset_request is None
    value._publish_snapshot(0)
    assert _http_snapshot(value)['goal_reach_viewer']['settings_command_result']['client_request_id']=='superseding-apply'


def test_terrain_descriptor_tracks_committed_scene_not_ticks_or_pending_reset(full_session, monkeypatch):
    from concurrent.futures import Future
    value = full_session
    def descriptor():
        value._publish_snapshot(0)
        return json.loads(value.snapshot.read())['goal_reach_contract']['terrain_descriptor']
    before = descriptor()
    value._advance_viewer_physics(1)
    assert descriptor() == before
    submit = value._environment_update_executor.submit
    pending = Future()
    monkeypatch.setattr(value._environment_update_executor, 'submit', lambda *a, **k: pending)
    value._reset()
    assert descriptor() == before
    pending.set_exception(ValueError('fixture preparation cancelled'))
    value._poll_environment_update()
    monkeypatch.setattr(value._environment_update_executor, 'submit', submit)
    value._reset(episode_index=7)
    value._environment_update_future.result(timeout=20)
    value._poll_environment_update()
    after = descriptor()
    assert after['scene_epoch'] != before['scene_epoch']
    assert after['world_epoch'] == 7
    assert after['density_per_1000m2'] == value.env.world_parameters['obstacle_density_mean_per_1000m2']
    assert before != after


@torch.inference_mode()
def test_owned_production_frame_has_no_runtime_aliases(full_session, monkeypatch):
    value = full_session
    value._refresh_actor_frame()
    saved = value._record_applied_frame()
    actions = {r:getattr(value, 'last_'+r+'_action') for r in value.role_controls}
    record = ViewerEpisodeRecord({}, {}, 'ownership')
    record.append(0, 1, actions, value._applied_frame_revision, saved, None)
    assert record.intervals[0]['frame'] is saved is value._applied_frame_cache
    canonical = json.dumps(saved, sort_keys=True, allow_nan=False)
    for role, action in actions.items():
        stored = record.intervals[0]['actions'][role]
        assert stored.data_ptr() != action.data_ptr()
        action.add_(.25)
        assert not torch.equal(stored, action)
        frame = value.actor_frames[role]
        frame.action.add_(.5)
        for tensor in frame.observation.values():
            if isinstance(tensor, torch.Tensor):tensor.zero_()
        value.role_controls[role].set_mode(0, 'manual')
        frame.display_context['deployment']['model_id'] = 'mutated'
    with monkeypatch.context() as scoped:
        scoped.setitem(gr.ROLE_DESCRIPTORS['learner'], 'label', 'mutation')
        assert json.dumps(saved, sort_keys=True, allow_nan=False) == canonical
    record.append(1, 2, actions, value._applied_frame_revision, None, None)
    assert len(record.intervals) == 1 and record.intervals[0]['end'] == 2
    value.env.learner_active.zero_()
    assert value._filter_actor_inputs(saved['inputs'])['learners'] == []
    value.selected_actor_kind = 'defender'
    value._cache_actor_inputs(0, force=True)
    value._publish_snapshot(0)
    json.loads(value.snapshot.read())
    assert json.dumps(saved, sort_keys=True, allow_nan=False) == canonical
    value._refresh_actor_frame()
    assert value._record_applied_frame() is not saved
    assert json.dumps(saved, sort_keys=True, allow_nan=False) == canonical


def test_owned_record_rejects_missing_frame_and_gap():
    for start, frame, message in ((0, None, 'decision frame'), (1, {}, 'actions')):
        record = ViewerEpisodeRecord({}, {}, 'missing')
        record.append(start, start+1, {}, 1, frame, None)
        record.finish(start+1)
        assert message in record.error and not record.complete and not record.intervals
    record = ViewerEpisodeRecord({}, {}, 'new revision')
    record.append(0, 1, {}, 1, {}, None)
    record.append(1, 2, {}, 2, None, None)
    assert record.error and len(record.intervals) == 1


@pytest.fixture(autouse=True)
def account_authorized_replay_ticks(monkeypatch, request):
    """Optional Issue driver accounting, including failed production advances."""
    import os,uuid
    if ('recorded_cpu_preparation_session' in request.fixturenames
            and os.environ.get('AUV_RECORDED_VIEWER_TESTS')!='1'):
        pytest.skip('Historical release/profile test requires AUV_RECORDED_VIEWER_TESTS=1')
    output = os.environ.get('AUV_REPLAY_TICK_OUTPUT')
    if not output:return
    path = Path(output)
    from viewer_performance_window import import_replay_test_ticks
    from auv_madrl.actor_training import write_json_atomic
    parent_path=Path(__file__).resolve().parents[2]/'Artifacts/GoalReach/viewer-review-terminal.json'
    section=os.environ.get('AUV_REPLAY_ISSUE','issue106')
    parent=json.loads(parent_path.read_text('utf-8')).get(section,{'actual_ticks':0}) if parent_path.is_file() else {'actual_ticks':0}
    import_replay_test_ticks(parent,path)  # Reject lost/rebuilt counters before any test advance.
    state = json.loads(path.read_text('utf-8')) if path.exists() else dict(actual_ticks=0, tests={},accounting_id=uuid.uuid4().hex)
    inherited=parent['actual_ticks']-state['actual_ticks']+sum(r.get('reserved_ticks',0) for r in parent.get('attempts',[]))
    row = state['tests'].setdefault(request.node.nodeid, dict(actual_ticks=0))
    original = gr.GoalReachViewerSession._advance_viewer_physics
    def advance(value, ticks, *args, **kwargs):
        from viewer_performance_window import counted_advance
        if inherited+state['actual_ticks']+ticks > 10000:raise RuntimeError(section+' tick budget exhausted')
        try:return counted_advance(state,row,lambda:int(value.env.physics_ticks[0]),lambda:original(value,ticks,*args,**kwargs))
        finally:write_json_atomic(path,state)
    monkeypatch.setattr(gr.GoalReachViewerSession, '_advance_viewer_physics', advance)


@pytest.mark.parametrize('chunk', [1, 8])
@pytest.mark.parametrize('objective', ['outcome', 'outcome_time', 'protection_time'])
@pytest.mark.parametrize('case,first,mode,count', [
    ('open', False, 'static', 1), ('open', False, 'dynamic', 3),
    ('dead_member', False, 'dynamic', 2), ('done', False, 'dynamic', 2),
    ('horizon', False, 'dynamic', 2), ('capture_target', False, 'dynamic', 2),
    ('target_obstacle', False, 'dynamic', 2), ('all_attackers', False, 'dynamic', 2),
    ('capture_attacker', False, 'dynamic', 2), ('capture_attacker', True, 'dynamic', 2),
    ('same_tick', True, 'dynamic', 2), ('defender_obstacle', False, 'dynamic', 2),
])
def test_cpu_viewer_event_path_matches_full_joint(cpu_test_state, monkeypatch, chunk, objective, case, first, mode, count):
    """Compare the optimized Viewer entry with the unchanged full joint event path."""
    from test_first_casualty_rule import _environment, _set_open_scene
    from profile_training_collection import digest
    env = _environment(terminate_on_any_casualty=first, target_mode=mode,
                       reward_objective=objective, learner_count_range=(count,count),
                       defender_count_range=(count,count))
    positions = torch.tensor([[[10., 6.*i] for i in range(count)]])
    _set_open_scene(env, learner_positions=positions, defender_positions=-positions)
    for role, pos in [('learner', positions), ('defender', -positions), ('target', torch.zeros(1,1,2))]:
        state = env.role_state(role)
        env.learner_dynamics.reset(state, pos, torch.zeros_like(state.heading), getattr(env,role+'_active'))
    if case == 'dead_member':env.learner_active[0,0]=False;env.learner_status[0,0]=2
    if case == 'done':env.done.fill_(True)
    if case == 'horizon':env.physics_ticks.fill_(env.config.goal_reach['clock']['maximum_physics_ticks']-2)
    if case in ('capture_target','same_tick'):env.learners.position[0,0]=env.targets.position[0,0]
    if case in ('target_obstacle','defender_obstacle','same_tick'):
        env.obstacle_active[0,0]=True;env.obstacle_radii[0,0]=1.
        env.obstacle_centers[0,0]=env.targets.position[0,0] if case=='target_obstacle' else env.defenders.position[0,0]
    if case=='capture_attacker':env.defenders.position[0,0]=env.learners.position[0,0]
    if case=='all_attackers':env.defenders.position.copy_(env.learners.position)
    env._bind_primary_learner()
    initial=copy.deepcopy(env.state_dict());rng=torch.get_rng_state();runs=[];calls=[];states=[]
    actions={role+'_action':torch.tensor([-1.,0.]).expand(*getattr(env,role+'_active').shape,2).clone() for role in env.roles}
    for full in (True,False):
        env.load_state_dict(copy.deepcopy(initial));torch.set_rng_state(rng)
        original=env.learner_dynamics.step;observed=[]
        def observe(*args,**kwargs):
            observed.append(1)
            return original(*args,**kwargs)
        with monkeypatch.context() as scoped:
            scoped.setattr(env.learner_dynamics,'step',observe)
            steps=[]
            for _ in range(2):
                step=env._step_joint_match(**actions,physics_ticks=chunk,build_observation=False,return_both=full)
                steps.append(digest((getattr(step,env.training_target) if full else step).__dict__))
            runs.append((steps,digest(env.state_dict()),digest(torch.get_rng_state())))
            states.append(copy.deepcopy(env.state_dict()))
            calls.append(len(observed))
    def differences(a,b,path='state'):
        if isinstance(a,dict):return [v for k in a for v in differences(a[k],b[k],path+'.'+k)]
        return [] if digest(a)==digest(b) else [(path,str(a),str(b))]
    assert runs[0]==runs[1], (case,objective,chunk,differences(states[0],states[1]))
    if case=='done':assert calls==[2*chunk,0]
    elif env.done[0] and chunk>1:assert calls[1]<calls[0]


def test_cpu_event_fast_path_is_taken(session):
    import inspect, sys
    method=inspect.unwrap(type(session.env)._advance_joint_ticks)
    code=method.__code__
    lines,start=inspect.getsourcelines(method)
    line=start+next(i for i,text in enumerate(lines) if text.strip()=='continue')
    hits=[];old=sys.gettrace()
    def trace(frame,event,arg):
        if frame.f_code is code and event=='line' and frame.f_lineno==line:hits.append(1)
        return trace
    try:
        sys.settrace(trace);session._advance_viewer_physics(8)
    finally:sys.settrace(old)
    assert len(hits)==8


@pytest.fixture
def local_cpu_inputs(monkeypatch):
    import subprocess
    read=Path.read_text
    def local_read(path,*args,**kwargs):
        if path.name=='viewer-review-terminal.json' or 'archive' in path.parts:
            raise AssertionError('Default correctness must not read recorded inputs')
        return read(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',local_read)
    monkeypatch.setattr(torch,'load',Mock(side_effect=AssertionError('No historical model in default fixture')))
    monkeypatch.setattr(subprocess,'check_output',Mock(side_effect=AssertionError('No git history in default fixture')))


@pytest.fixture
def cpu_preparation_session(local_cpu_inputs, session):
    """Self-contained P1 production environment; no historical inputs."""
    return session,None,None,None


def _recorded_cpu_inputs(evidence=None, *, follow=False):
    import os,hashlib,subprocess
    if os.environ.get('AUV_RECORDED_VIEWER_TESTS')!='1':
        pytest.skip('Historical release/profile test requires AUV_RECORDED_VIEWER_TESTS=1')
    output=os.environ.get('AUV_REPLAY_TICK_OUTPUT')
    if not output or not Path(output).is_file():
        raise ValueError('Recorded Viewer tests require an existing authorized tick counter')
    counter=json.loads(Path(output).read_text('utf-8'))
    if type(counter.get('actual_ticks')) is not int or not 0<=counter['actual_ticks']<10000 or not isinstance(counter.get('tests'),dict) or not counter.get('accounting_id'):
        raise ValueError('Invalid authorized tick counter')
    evidence=Path(evidence) if evidence is not None else Path(__file__).resolve().parents[2]/'Artifacts/GoalReach/viewer-review-terminal.json'
    if Path(output).resolve()==evidence.resolve():raise ValueError('Tick output must be isolated from review evidence')
    inputs=json.loads(evidence.read_text('utf-8'))['issue114']['inputs']
    for key in ('release','initial_state','initial_rng'):
        path=Path(inputs[key])
        if not path.is_file():raise FileNotFoundError('Recorded Viewer input missing: '+str(path))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=inputs[key+'_sha256']:
            raise ValueError('Recorded Viewer input hash mismatch: '+key)
    subprocess.check_output(['git','show','c929429347dd7e89fd310dd5de85cd1712182f8c:Python/src/auv_madrl/multi_goal_reach_env.py'],cwd=evidence.parents[2])
    if follow:
        for name in ('run.json','checkpoint-latest.pt'):
            path=Path(inputs['release']).parent/name
            if not path.is_file():raise FileNotFoundError('Recorded Follow input missing: '+str(path))
        manifest=json.loads((Path(inputs['release']).parent/'run.json').read_text('utf-8'))
        if manifest.get('run_group_id')!=inputs['run_group_id']:raise ValueError('Recorded Follow group mismatch')
    return inputs


@contextmanager
def _owned_cpu_session():
    # Own the object before __init__, including partial construction failures.
    value=gr.GoalReachViewerSession.__new__(gr.GoalReachViewerSession)
    try:
        value.__init__()
        yield value
    finally:
        value.close()


@pytest.mark.parametrize('locked', [False, True])
def test_horizon_control_applies_on_reset_and_binds_replay(tmp_path, monkeypatch, locked):
    settings = gr._viewer_settings_defaults()
    settings.update(target_mode='dynamic', target_count_range=[1, 1],
                    obstacle_density_mean_per_1000m2=0, obstacle_density_std_per_1000m2=0)
    settings['training_settings']['parallel_environments'] = 1
    settings = gr._validated_viewer_settings(settings)
    monkeypatch.setattr(gr, '_load_viewer_settings', lambda config: (tmp_path/'settings.json', copy.deepcopy(settings)))
    with _cpu_test_state(), _owned_cpu_session() as value:
        value.running = False
        value.scene_locked = locked
        if locked:
            value._locked_scene_state = copy.deepcopy(value._current_episode_initial_state['environment'])
        previous = value.env
        before = copy.deepcopy(value.env.state_dict())
        old_contract = value._replay_contract()
        old_models = {role: control.neural_policy.model for role, control in value.role_controls.items()}
        value._set_viewer_settings(dict(training_settings=dict(maximum_episode_decisions=1000,
                                                               time_limit_boundary='truncation')))
        if value._training_scene_load_future is not None:
            value._training_scene_load_future.result(timeout=20)
            value._poll_training_scene_load()
        assert value.env is previous and value.env.physics_ticks.equal(before['physics_ticks'])
        assert not value.env.bootstrap_on_time_limit
        assert value._horizon_pending()['maximum_episode_decisions'] == 150
        assert value._horizon_pending()['time_limit_boundary'] == 'termination'
        assert value._replay_contract() == old_contract
        assert value.scene_locked == locked
        assert value.viewer_settings['training_settings']['maximum_episode_decisions'] == 150
        assert value._requested_viewer_settings['training_settings']['maximum_episode_decisions'] == 1000
        value._publish_snapshot(0.)
        snapshot = json.loads(value.snapshot.read())
        assert snapshot['horizon_pending'] == value._horizon_pending()
        assert snapshot['goal_reach_contract']['defender_episode_end'].endswith('_termination')
        value._reset()
        if value._training_scene_load_future is not None:
            value._training_scene_load_future.result(timeout=20)
            value._poll_training_scene_load()
        assert value.reset_progress['phase'] == 'ready', value.reset_progress
        assert value.env is not previous and value.env.bootstrap_on_time_limit
        assert value.config.shared['clock']['maximum_decisions'] == 1000
        assert value._policy_noise.maximum_decisions == 1000
        assert value._horizon_pending() is None and value._replay_contract() != old_contract
        assert all(value.role_controls[role].neural_policy.model is model for role, model in old_models.items())
        value._publish_snapshot(0.)
        snapshot = json.loads(value.snapshot.read())
        assert snapshot['goal_reach_contract']['maximum_episode_decisions'] == 1000
        assert snapshot['goal_reach_contract']['defender_episode_end'].endswith('_truncation')
        assert snapshot['horizon_pending'] is None
        # Switching only boundary at the same horizon also waits for reset and changes replay identity.
        truncation_contract = value._replay_contract()
        value._set_viewer_settings(dict(training_settings=dict(time_limit_boundary='termination')))
        assert value.env.bootstrap_on_time_limit and value._horizon_pending() is not None
        value._reset()
        if value._training_scene_load_future is not None:
            value._training_scene_load_future.result(timeout=20)
            value._poll_training_scene_load()
        assert not value.env.bootstrap_on_time_limit and value._horizon_pending() is None, (value.command_error, value._viewer_settings_result, value.reset_progress)
        assert value._replay_contract() != truncation_contract


@pytest.fixture
def recorded_cpu_preparation_session(tmp_path, monkeypatch, request):
    inputs=_recorded_cpu_inputs(follow=bool(request and 'follow_revision' in request.node.name))
    from viewer_performance_window import restore_cpu_profile_session
    with _cpu_test_state(),ExitStack() as owned:
        monkeypatch.setattr(gr,'_load_viewer_settings',lambda config:(tmp_path/'settings.json',copy.deepcopy(inputs['settings'])))
        factory=lambda:owned.enter_context(_owned_cpu_session())
        value=factory();value._test_session_factory=factory
        initial=torch.load(inputs['initial_state'],map_location='cpu',weights_only=False)
        rng=torch.load(inputs['initial_rng'],map_location='cpu',weights_only=False)
        restore_cpu_profile_session(value,inputs,Path(inputs['release']),initial,rng)
        yield value,inputs,initial,rng


def test_cpu_preparation_cache_value_and_binding_invalidation(cpu_preparation_session):
    value,inputs,initial,rng=cpu_preparation_session
    env=value.env
    value._advance_viewer_physics(1)
    combined=env._cpu_state_cache[0]
    assert env._cpu_joint_state() is combined
    before=combined.desired_body_velocity.clone()
    env.learners.desired_body_velocity[...,0].add_(.125)
    assert not torch.equal(combined.desired_body_velocity,before)
    for state in (env.defenders,env.learners,env.targets):
        for field in tuple(state.__dict__):
            old=getattr(state,field);setattr(state,field,old.clone())
            assert env._cpu_joint_state() is not combined
            setattr(state,field,old)
            assert env._cpu_joint_state() is combined
    old=env.learners.position
    replacement=old.clone();old.set_(replacement)
    assert env._cpu_joint_state() is not combined
    value._advance_viewer_physics(1);combined=env._cpu_state_cache[0]
    old=env.learners.position
    backing=torch.empty((*old.shape[:-1],old.shape[-1]*2));view=backing[...,::2];view.copy_(old)
    env.learners.position=view
    assert env._cpu_joint_state() is not combined
    value._advance_viewer_physics(1)
    assert env._cpu_joint_state() is env._cpu_state_cache[0]
    caps=env._cpu_joint_limits();assert env._cpu_joint_limits()==caps
    for role in ('learner','defender','target'):
        ratio=getattr(env,role+'_speed_ratio');ratio.add_(.1)
        previous=env._cpu_limit_cache;env._cpu_joint_limits();assert env._cpu_limit_cache is not previous
        setattr(env,role+'_speed_ratio',ratio.clone()+.1)
        previous=env._cpu_limit_cache;env._cpu_joint_limits();assert env._cpu_limit_cache is not previous
    for field in ('maximum_surge','maximum_yaw','horizontal_thruster_limit','linear_drag','quadratic_drag'):
        dynamics=env.learner_dynamics;previous=env._cpu_limit_cache
        if field.endswith('drag'):getattr(dynamics,field)[0].add_(.01)
        else:setattr(dynamics,field,getattr(dynamics,field)+.01)
        env._cpu_joint_limits();assert env._cpu_limit_cache is not previous
    with torch.inference_mode():
        env.learner_speed_ratio.add_(.1);env._cpu_joint_limits()
    assert all(not x.is_inference() for limits in env._cpu_limit_cache[2] for x in limits)
    assert all(not x.is_inference() for x in env._cpu_limit_cache[3])
    for role,limits in zip(('defender','learner','target'),env._cpu_limit_cache[2]):
        assert all(torch.equal(a,b) for a,b in zip(limits,env.patrol_limits(getattr(env,role+'_speed_ratio'))))
    assert not any(key.startswith('_cpu_') for key in env.state_dict())


@pytest.mark.parametrize('chunk',[1,8])
def test_cpu_preparation_full_source_equivalence(recorded_cpu_preparation_session, monkeypatch, chunk):
    import random,subprocess,sys,types
    import numpy as np
    from profile_training_collection import digest
    from viewer_performance_window import restore_cpu_profile_session
    value,inputs,initial,rng=recorded_cpu_preparation_session
    source=subprocess.check_output(['git','show','c929429347dd7e89fd310dd5de85cd1712182f8c:Python/src/auv_madrl/multi_goal_reach_env.py']).decode('utf-8')
    module=types.ModuleType('auv_madrl._cpu_preparation_before');sys.modules[module.__name__]=module
    exec(compile(source,'before/multi_goal_reach_env.py','exec'),module.__dict__)
    after_class=type(value.env);runs=[];records=[];extra=None
    try:
        for index,cls in enumerate((module.MultiGoalReachEnv,after_class)):
            if index:
                value=extra=recorded_cpu_preparation_session[0]._test_session_factory()
                restore_cpu_profile_session(value,inputs,Path(inputs['release']),initial,rng)
            value.env.__class__=cls
            ticks=[];steps=[]
            original=value.env.learner_dynamics.step
            def observed(state,*a,**kw):
                result=original(state,*a,**kw)
                ticks.append(digest({k:v.clone() for k,v in state.__dict__.items()}))
                return result
            with monkeypatch.context() as patch:
                patch.setattr(gr,'MultiGoalReachEnv',cls)
                patch.setattr(value.env.learner_dynamics,'step',observed)
                for index in range(16):
                    value._refresh_actor_frame()
                    if index==2:
                        value.role_controls['learner'].set_mode(0,'manual')
                        value.last_learner_action[0]=torch.tensor([-.2,.5])
                    if index==3:value.last_learner_action[0]=torch.tensor([-1.,0.])
                    if index==4:
                        for control in value.role_controls.values():control.set_neural_sampling(False)
                    if index==6:value.env.learner_speed_ratio.add_(.2)
                    if index==7:value.env.defender_speed_ratio=value.env.defender_speed_ratio.clone()+.1
                    if index==8:value.env.target_speed_ratio.add_(.1)
                    if index==9:value.env.learners.heading=value.env.learners.heading.clone()
                    step=value._advance_viewer_physics(chunk,deadline=time.perf_counter()-1 if index==10 else None)
                    if step is not None:steps.append(digest(step.__dict__))
            record=value._live_record or value._last_complete_record
            records.append(copy.deepcopy(record.intervals))
            runs.append(dict(ticks=ticks,steps=steps,state=digest(value.env.state_dict()),rng=digest((torch.get_rng_state(),random.getstate(),np.random.get_state())),record=digest((record.intervals,record.checks))))
        def compare(a,b,path='frame'):
            if isinstance(a,torch.Tensor):assert torch.equal(a,b),path
            elif isinstance(a,dict):
                assert a.keys()==b.keys(),path
                for key in a:compare(a[key],b[key],path+'.'+str(key))
            elif isinstance(a,(list,tuple)):
                assert len(a)==len(b),path
                for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'.'+str(i))
            else:assert a==b,(path,a,b)
        compare(records[0],records[1])
        assert runs[0]==runs[1]
    finally:
        value.env.__class__=after_class
        sys.modules.pop(module.__name__,None)


def test_cpu_preparation_production_profile(recorded_cpu_preparation_session, monkeypatch, tmp_path):
    """Paired real preparation calls; profiler costs are not ordinary wall evidence."""
    import cProfile,subprocess,sys,types
    from profile_training_collection import digest
    from viewer_performance_window import restore_cpu_profile_session,cpu_profile_rows
    from auv_madrl.actor_training import write_json_atomic
    value,inputs,initial,rng=recorded_cpu_preparation_session
    source=subprocess.check_output(['git','show','c929429347dd7e89fd310dd5de85cd1712182f8c:Python/src/auv_madrl/multi_goal_reach_env.py']).decode('utf-8')
    module=types.ModuleType('auv_madrl._cpu_preparation_profile_before');sys.modules[module.__name__]=module
    exec(compile(source,'before/multi_goal_reach_env.py','exec'),module.__dict__)
    after_class=type(value.env);rows=[];extra=None
    try:
        for index,cls in enumerate((module.MultiGoalReachEnv,after_class)):
            if index:
                value=extra=recorded_cpu_preparation_session[0]._test_session_factory()
                restore_cpu_profile_session(value,inputs,Path(inputs['release']),initial,rng)
            value.env.__class__=cls
            profile=cProfile.Profile();hits=0
            with monkeypatch.context() as scoped:
                scoped.setattr(gr,'MultiGoalReachEnv',cls)
                for tick in range(250):
                    if index and tick:hits+=value.env._cpu_joint_state() is value.env._cpu_state_cache[0]
                    profile.enable()
                    try:
                        if tick%25==0:value._refresh_actor_frame()
                        value._advance_viewer_physics(1)
                    finally:profile.disable()
                record=value._live_record or value._last_complete_record
                rows.append(dict(variant='after' if index else 'before',ticks=250,cache_hits=hits,
                    state=digest(value.env.state_dict()),record=digest((record.intervals,record.checks)),
                    profile=cpu_profile_rows(profile,250,10,'wall')))
        assert rows[0]['state']==rows[1]['state'] and rows[0]['record']==rows[1]['record']
        assert rows[1]['cache_hits']==249
        write_json_atomic(tmp_path/'preparation-profile.json',{'preparation_profile':rows})
    finally:
        sys.modules.pop(module.__name__,None)


def test_cpu_preparation_follow_revision_and_dead_pose(recorded_cpu_preparation_session):
    from concurrent.futures import Future
    value,inputs,initial,rng=recorded_cpu_preparation_session
    for _,future in tuple(value._deployment_futures.values()):future.result(timeout=30)
    value._poll_viewer_deployments()
    value._refresh_actor_frame();value._advance_viewer_physics(1)
    combined=value.env._cpu_state_cache[0]
    source=Path(inputs['release'])
    manifest=json.loads((source.parent/'run.json').read_text('utf-8'))
    revisions=[]
    for checkpoint in (source,source.parent/'checkpoint-latest.pt'):
        generations={}
        for role in value.env.roles:
            value._deployment_generations[role]+=1
            generations[role]=value._deployment_generations[role]
        bundle=value._prepare_follow_bundle(checkpoint,manifest,generations)
        value._auto_deploy_training_run_group_id=value._follow_bundle_group=manifest['run_group_id']
        value._auto_deploy_training_sides=set(value.env.roles)
        future=Future();future.set_result(bundle);value._follow_bundle_future=future
        value.running=False;value._follow_training_checkpoints({})
        assert value.checkpoint_refresh_error is None
        hashes={r:value._role_deployment_descriptor(r)['actor_hash'] for r in value.env.roles}
        assert all(hashes[item.actor_kind]==gr.actor_state_hash(item.model.state_dict()) for item in bundle)
        revisions.append((value._applied_frame_revision,hashes))
        assert value.env._cpu_joint_state() is combined
        value._advance_viewer_physics(1);combined=value.env._cpu_state_cache[0]
    assert revisions[0][0]<revisions[1][0] and revisions[0][1]!=revisions[1][1]
    # A dead member's pose remains frozen across cache hits, while others advance.
    env=value.env
    env.learner_active[0,0]=False;env.learner_status[0,0]=2
    frozen={k:getattr(env.learners,k)[0,0].clone() for k in ('position','heading')}
    for _ in range(2):
        value._advance_viewer_physics(1)
        assert all(torch.equal(getattr(env.learners,k)[0,0],v) for k,v in frozen.items())

@pytest.mark.parametrize('chunk',[1,4])
def test_cpu_preparation_local_reference_and_replay(local_cpu_inputs, session, monkeypatch, chunk):
    """Uncached production seam vs cached path, fixed local actions and exact states."""
    value=session;env=value.env;initial=copy.deepcopy(env.state_dict());rng=torch.get_rng_state();runs=[]
    def uncached_state():
        return type(env.learners)(**{name:torch.cat([getattr(side,name) for side in (env.defenders,env.learners,env.targets)],dim=1) for name in env.learners.__dict__})
    def uncached_limits():
        limits=[env.patrol_limits(getattr(env,role+'_speed_ratio')) for role in ('defender','learner','target')]
        return limits,tuple(torch.cat(values,dim=1) for values in zip(*limits))
    for reference in (True,False):
        env.load_state_dict(copy.deepcopy(initial));torch.set_rng_state(rng)
        value._live_record=ViewerEpisodeRecord(env.state_dict(),dict(episode=1),value._replay_contract())
        value._last_complete_record=None;value._replay_record=None;value._replay_error=None
        ticks=[]
        with monkeypatch.context() as scoped:
            if reference:
                scoped.setattr(env,'_cpu_joint_state',uncached_state);scoped.setattr(env,'_cpu_joint_limits',uncached_limits)
            for index in range(8):
                value.last_learner_action[0]=torch.tensor([-.8+index*.1,.25 if index%2 else -.25])
                if index==3:env.learner_speed_ratio.add_(.1)
                if index==4:env.defender_speed_ratio=env.defender_speed_ratio.clone()+.1
                if index==5:env.targets.heading=env.targets.heading.clone()
                value._advance_viewer_physics(chunk)
                ticks.append(physical_digest(env).hex())
        record=value._live_record or value._last_complete_record
        runs.append((ticks,copy.deepcopy(record.intervals),copy.deepcopy(record.checks)))
    def same(a,b):
        if isinstance(a,torch.Tensor):assert torch.equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for key in a:same(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):same(x,y)
        else:assert a==b
    same(runs[0],runs[1])


def test_cpu_preparation_local_dead_pose(local_cpu_inputs, session):
    value=session;env=value.env
    value._advance_viewer_physics(1)
    env.learner_active[0,0]=False;env.learner_status[0,0]=2
    frozen={k:getattr(env.learners,k).clone() for k in ('position','heading')}
    for _ in range(2):
        value._advance_viewer_physics(1)
        assert all(torch.equal(getattr(env.learners,k),v) for k,v in frozen.items())


@pytest.mark.parametrize('failure',['construct','first_load','second_load','restore','second_session','yield'])
def test_recorded_cpu_fixture_failure_cleanup(tmp_path,monkeypatch,failure):
    """Exercise the real fixture control flow with resource-owning zero-physics doubles."""
    import random
    import numpy as np
    import viewer_performance_window as driver
    inputs=dict(settings={},initial_state='initial',initial_rng='rng',release='release')
    monkeypatch.setattr(__import__(__name__,fromlist=['_recorded_cpu_inputs']),'_recorded_cpu_inputs',lambda **kwargs:inputs)
    created=[];loads=[];external=lambda *args:None;gc.callbacks.append(external)
    class FakeSession:
        close = gr.GoalReachViewerSession.close
        def __init__(self):
            self._close_lock = threading.Lock()
            self._closed = self._close_requested = False
            self._gc_callback=lambda *args:None;gc.callbacks.append(self._gc_callback)
            self._training_inventory_executor=Mock();created.append(self)
            random.random();np.random.rand();torch.rand(1)
            if failure=='construct' or (failure=='second_session' and len(created)==2):raise RuntimeError('injected '+failure)
    monkeypatch.setattr(gr,'GoalReachViewerSession',FakeSession)
    def load(path,**kwargs):
        loads.append(path)
        if failure==('first_load' if len(loads)==1 else 'second_load'):raise RuntimeError('injected '+failure)
        return {}
    monkeypatch.setattr(torch,'load',load)
    def restore(*args):
        if failure=='restore':raise RuntimeError('injected restore')
    monkeypatch.setattr(driver,'restore_cpu_profile_session',restore)
    threads=torch.get_num_threads();rng=torch.get_rng_state().clone();py=random.getstate();npstate=np.random.get_state()
    fixture=recorded_cpu_preparation_session.__wrapped__(tmp_path,monkeypatch,None)
    try:
        with pytest.raises(RuntimeError,match='injected'):
            value,*_=next(fixture)
            if failure=='second_session':value._test_session_factory()
            else:fixture.throw(RuntimeError('injected yield'))
    finally:
        fixture.close()
        assert external in gc.callbacks;gc.callbacks.remove(external)
    assert torch.get_num_threads()==threads and torch.equal(torch.get_rng_state(),rng)
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],npstate[1])
    assert created and all(v._gc_callback not in gc.callbacks for v in created)
    for v in created:v._training_inventory_executor.shutdown.assert_called_once_with(wait=True,cancel_futures=True)


def test_recorded_cpu_disabled_before_side_effects(tmp_path,monkeypatch):
    monkeypatch.delenv('AUV_RECORDED_VIEWER_TESTS',raising=False)
    read=Mock(side_effect=AssertionError('unexpected evidence read'));load=Mock();construct=Mock()
    monkeypatch.setattr(Path,'read_text',read);monkeypatch.setattr(torch,'load',load)
    monkeypatch.setattr(gr,'GoalReachViewerSession',construct)
    fixture=recorded_cpu_preparation_session.__wrapped__(tmp_path,monkeypatch,None)
    with pytest.raises(pytest.skip.Exception):next(fixture)
    read.assert_not_called();load.assert_not_called();construct.assert_not_called()

@pytest.mark.parametrize('failure',['budget','missing','hash','baseline'])
def test_recorded_cpu_preflight_rejects_before_construction(tmp_path,monkeypatch,failure):
    import hashlib,subprocess
    monkeypatch.setenv('AUV_RECORDED_VIEWER_TESTS','1')
    counter=tmp_path/'counter.json';counter.write_text(json.dumps(dict(actual_ticks=0,tests={},accounting_id='fixture')))
    monkeypatch.setenv('AUV_REPLAY_TICK_OUTPUT',str(counter) if failure!='budget' else str(tmp_path/'missing-counter'))
    inputs={}
    for key in ('release','initial_state','initial_rng'):
        p=tmp_path/key;p.write_bytes(b'fixture');inputs[key]=str(p);inputs[key+'_sha256']=hashlib.sha256(b'fixture').hexdigest()
    if failure=='missing':inputs['initial_state']=str(tmp_path/'not-present')
    if failure=='hash':inputs['initial_rng_sha256']='0'*64
    evidence=tmp_path/'evidence.json';evidence.write_text(json.dumps({'issue114':{'inputs':inputs}}))
    monkeypatch.setattr(subprocess,'check_output',Mock(side_effect=subprocess.CalledProcessError(128,'git show')))
    construct=Mock();load=Mock();monkeypatch.setattr(gr,'GoalReachViewerSession',construct);monkeypatch.setattr(torch,'load',load)
    with pytest.raises((ValueError,FileNotFoundError,subprocess.CalledProcessError)):_recorded_cpu_inputs(evidence)
    construct.assert_not_called();load.assert_not_called()

def test_recorded_profile_output_is_temporary(tmp_path):
    import ast
    from auv_madrl.run_evidence import write_json_atomic
    tree=ast.parse(Path(__file__).read_text('utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_cpu_preparation_production_profile')
    calls=[n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='write_json_atomic']
    assert len(calls)==1
    rows=[{'variant':'fixture','ticks':0}]
    program=ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=calls[0])],type_ignores=[]))
    exec(compile(program,'<actual profile writer>','exec'),dict(write_json_atomic=write_json_atomic,tmp_path=tmp_path,rows=rows))
    assert json.loads((tmp_path/'preparation-profile.json').read_text('utf-8'))=={'preparation_profile':rows}
