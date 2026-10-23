"""Independent finite-tiling references for the periodic production map contract."""
import copy
import itertools
import json

import numpy as np
import pytest
import torch

from auv_madrl import terrain
from auv_madrl import goal_reach as gr
from auv_madrl.viewer_terrain import ViewerTerrain, TerrainReadError


def make_env(device='cpu', density=5., batch=3):
    torch.set_num_threads(1)
    settings = gr._viewer_settings_defaults()
    settings.update(world_map_seed=41001, obstacle_density_mean_per_1000m2=density, obstacle_density_std_per_1000m2=0,
                    learner_speed_ratio_range=[2.8,2.8], defender_count_range=[3,3])
    settings['training_settings']['parallel_environments'] = batch
    return settings, gr._build_training_group(settings,device=device)[0]


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_complete_query_against_independent_finite_tiles(device, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA')
    bank = terrain.load_periodic_maps(device)
    points = [[-1000.,-1000.],[-25.,0.],[0.,25.],[1000.,1000.],
              [3025.,-4975.],[-9999.,9001.],[999.,1.],[23.,23.]]
    lower = torch.tensor(points,device=device)-7
    upper = lower+14
    maps = torch.zeros(len(points),device=device,dtype=torch.long)
    density = torch.tensor([0.,10.,5.,.5,10.,5.,5.,10.],device=device)
    originals = [x.clone() for x in (maps,lower,upper,density)]
    def forbidden(*args, **kwargs): raise AssertionError('hot query sorted/generated terrain')
    with monkeypatch.context() as m:
        for name in ('sort','argsort','topk'): m.setattr(torch,name,forbidden)
        m.setattr(terrain,'generate_terrain',forbidden)
        centers,radii,keys,active = terrain.periodic_query(bank,maps,lower,upper)
    reference = terrain.load_periodic_maps('cpu')
    for row in range(len(points)):
        lo,hi = lower[row].cpu().numpy(),upper[row].cpu().numpy()
        family,rho = int(maps[row]),float(density[row])
        c = reference.centers[family,:reference.counts[family]].numpy()
        r = reference.radii[family,:len(c)].numpy()
        expected = {}
        # Full finite translated base maps, no grid helper/filter in the reference.
        for tx,tz in itertools.product(range(int(np.floor(lo[0]/1000))-1,int(np.floor(hi[0]/1000))+2),
                                       range(int(np.floor(lo[1]/1000))-1,int(np.floor(hi[1]/1000))+2)):
            translated = c+np.array([tx*1000,tz*1000],dtype=np.float32)
            nearest = np.clip(translated,lo,hi)
            visible = ((translated-nearest)**2).sum(-1) <= r*r
            for i in np.flatnonzero(visible):
                expected[(family,int(i),tx,tz)] = translated[i]
        actual_keys = [tuple(k) for k in keys[row,active[row]].cpu().tolist()]
        assert len(actual_keys) == len(set(actual_keys))
        assert set(actual_keys) == set(expected)
        for k,c_actual in zip(actual_keys,centers[row,active[row]].cpu().numpy()):
            np.testing.assert_array_equal(c_actual,expected[k])
    assert all(torch.equal(x,y) for x,y in zip(originals,(maps,lower,upper,density)))
    other = terrain.periodic_query(bank,maps,lower,upper)
    assert all(torch.equal(x,y) for x,y in zip((centers,radii,keys,active),other))
    other[0].zero_()
    assert centers.abs().sum() > 0


def test_generated_map_reproducible_hash_and_parameters():
    first=terrain.load_periodic_maps()
    terrain._PERIODIC_BANKS.clear()
    second=terrain.load_periodic_maps()
    assert first.asset_hash==second.asset_hash
    assert torch.equal(first.centers,second.centers)
    assert torch.equal(first.radii,second.radii)
    with pytest.raises(ValueError):terrain.load_periodic_maps(parameters={'obstacle_radius_mean_m':float('nan')})


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('count',[0,9,10,11,20])
def test_first10_surface_threshold_dead_and_complete_physics(device,count,monkeypatch):
    settings,env = make_env(device,0.,1)
    env.ensure_circle_storage(0,max(count,1))
    env.terrain_manual_override.fill_(True)
    env.obstacle_active.zero_()
    env.obstacle_active[0,:count] = True
    # Reverse distance order, final/nearest is discarded when count > 10.
    env.obstacle_centers[0,:count,0] = torch.linspace(15.5,.1,count,device=device)
    env.obstacle_centers[0,:count,1] = 0
    env.obstacle_radii[0,:count] = .5
    position = torch.zeros((1,2),device=device); heading = position[:,0]
    lanes = torch.zeros(1,dtype=torch.long,device=device);alive = lanes==0
    def forbidden(*args,**kwargs):raise AssertionError('observation sorting')
    with monkeypatch.context() as m:
        for name in ('sort','argsort','topk'):m.setattr(torch,name,forbidden)
        entities,mask = env._local_obstacle_observation(position,heading,alive,lanes,record=('learner',lanes))
    assert entities.shape == (1,10,3) and int(mask.sum()) == min(count,10)
    assert torch.equal(entities[0,:min(count,10),2],torch.full((min(count,10),),.5/env.position_scale_m,device=device))
    if count:
        assert torch.allclose(entities[0,0,:2].norm()*env.position_scale_m,torch.tensor(15.5,device=device))
    dead,deadmask = env._local_obstacle_observation(position,heading,~alive,lanes)
    assert not deadmask.any() and not dead.any()
    if count>10:
        # The exact swept solver still sees the excluded, initially overlapping circle.
        contacts = env._multi_collection_contact_matrix(position[:,None],position[:,None],
            env.obstacle_centers,env.obstacle_radii,env.obstacle_active)
        assert contacts[0,0,count-1]
        if device=='cuda':
            cache=env._obstacle_neighbors;slot=env.defender_count
            assert cache['mask'][0,slot].sum()==count


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_origin_spawn_reset_restore_and_shared_bank(device):
    settings,env = make_env(device)
    other = make_env(device)[1]
    assert env.periodic_bank is other.periodic_bank
    assert env.map_id.tolist()==[0,0,0]
    distance=(env.obstacle_centers-env.region_reference_position[:,None]).norm(dim=-1)
    assert not (env.obstacle_active & (distance<=env.obstacle_radii+env.vessel_radius_m)).any()
    initial=copy.deepcopy(env.state_dict())
    other.load_state_dict(initial,source_launch_settings=settings)
    assert torch.equal(other.map_origin_xy,env.map_origin_xy)
    for role in env.policy_roles:
        a,b=env.training_observation_for(role),other.training_observation_for(role)
        assert all(torch.equal(a[k],b[k]) for k in a)
    old_origin=env.map_origin_xy.clone()
    env.reset(mask=torch.tensor([False,True,False],device=device),env_indices=torch.tensor([1],device=device),build_observation=False)
    assert torch.equal(env.map_origin_xy[[0,2]],old_origin[[0,2]])
    assert not torch.equal(env.map_origin_xy[1],old_origin[1])
    bad=copy.deepcopy(initial);bad['scene_contract']['observations']['obstacles']['schema']='shared_local_all_visible_v1'
    with pytest.raises(ValueError,match='incompatible'):other.load_state_dict(bad,source_launch_settings=settings)


def test_preview_geometry_identity_and_stale():
    _,env=make_env(batch=1)
    d=env.terrain_descriptor(0,scene_epoch=1);d['world_geometry_revision']=0
    reader=ViewerTerrain();reader.publish(d,{},[])
    q=dict(preview='1',world_epoch=str(d['world_epoch']),world_geometry_revision='0',request_id='4')
    result=json.loads(reader.read(q))
    assert result['id']=='41001' and result['bank_hash']==env.periodic_bank.asset_hash
    bank=env.periodic_bank;count=int(bank.counts[0])
    assert len(result['obstacles'])==count
    q['world_epoch']='-1'
    with pytest.raises(TerrainReadError) as error:reader.read(q)
    assert error.value.status==409


def test_contact_identity_survives_reorder_departure_edit_and_restore():
    settings,env=make_env(batch=1)
    slot=int(env.obstacle_active[0].nonzero()[0]);key=env.obstacle_keys[0,slot].clone()
    env.contact_static[0,0,slot]=True
    env.episode_contact_counts[0,0,0]=1
    for role in env.physical_roles:env.role_state(role).position.add_(500.)
    env._refresh_terrain(torch.tensor([0]))
    found=(env.obstacle_keys[0]==key).all(-1).nonzero().flatten()
    assert len(found)==1 and env.contact_static[0,0,found[0]]
    assert not env.obstacle_active[0,found[0]]  # historical slot is not collision geometry
    env.edit_terrain_obstacle(0,env.terrain_key_text(key),delete=True)
    found=(env.obstacle_keys[0]==key).all(-1).nonzero().flatten()
    assert len(found)==1 and env.contact_static[0,0,found[0]] and not env.obstacle_active[0,found[0]]
    state=copy.deepcopy(env.state_dict());other=make_env(batch=1)[1]
    other.load_state_dict(state,source_launch_settings=settings)
    assert torch.equal(other.contact_static,env.contact_static)
    assert torch.equal(other.obstacle_keys,env.obstacle_keys)
    assert int(env.episode_contact_counts[0,0,0])==1


@pytest.mark.parametrize('fault',['missing','shape','nonfinite','family'])
def test_bad_map_restore_rejected_before_mutation(fault):
    settings,env=make_env(batch=1);before=copy.deepcopy(env.state_dict());bad=copy.deepcopy(before)
    if fault=='missing':bad.pop('map_origin_xy')
    elif fault=='shape':bad['map_origin_xy']=torch.zeros(1)
    elif fault=='nonfinite':bad['map_origin_xy'][0,0]=torch.nan
    else:bad['map_id'][0]=1
    with pytest.raises(ValueError,match='map identity'):env.load_state_dict(bad,source_launch_settings=settings)
    assert torch.equal(before['map_origin_xy'],env.map_origin_xy)
    assert torch.equal(before['learner']['position'],env.learner.position)
