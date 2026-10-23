import copy
import itertools
import json
import math
import time

import numpy as np
import pytest
import torch

from auv_madrl import terrain, goal_reach as gr
from auv_madrl.viewer_terrain import ViewerTerrain


def make_env(device='cpu', batch=3, **parameters):
    torch.set_num_threads(1)
    s=gr._viewer_settings_defaults();s.update(parameters)
    s['training_settings']['parallel_environments']=batch
    return gr._build_training_group(s,device=device)[0]


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('parameters',[{}, {'obstacle_density_mean_per_1000m2':0},
    {'obstacle_density_std_per_1000m2':0,'obstacle_radius_std_m':0},
    {'obstacle_density_std_per_1000m2':5,'obstacle_radius_std_m':3}])
def test_complete_query_independent_tiling(device,parameters):
    bank=terrain.load_periodic_maps(device,parameters=parameters)
    lower=torch.tensor([[-1010.,-1010.],[-7.,-7.],[993.,993.],[24.,24.],[3025.,-4975.]],device=device)
    upper=lower+35
    c,r,k,active=terrain.periodic_query(bank,torch.zeros(len(lower),device=device,dtype=torch.long),lower,upper)
    xy=bank.centers[0].cpu().numpy();rr=bank.radii[0].cpu().numpy()
    for row,(lo,hi) in enumerate(zip(lower.cpu().numpy(),upper.cpu().numpy())):
        expected={}
        for tx,tz in itertools.product(range(math.floor((lo[0]-bank.max_radius)/1000),math.floor((hi[0]+bank.max_radius)/1000)+1),range(math.floor((lo[1]-bank.max_radius)/1000),math.floor((hi[1]+bank.max_radius)/1000)+1)):
            moved=xy+np.array([tx*1000,tz*1000],np.float32)
            hit=((moved-np.clip(moved,lo,hi))**2).sum(1)<=rr**2
            for index in np.flatnonzero(hit):expected[(0,int(index),tx,tz)]=moved[index]
        actual={tuple(key):pos for key,pos in zip(k[row][active[row]].cpu().tolist(),c[row][active[row]].cpu().numpy())}
        assert actual.keys()==expected.keys()
        for key in expected:np.testing.assert_allclose(actual[key],expected[key],rtol=0,atol=0)


def test_zero_deviation_cache_and_identity():
    bank=terrain.load_periodic_maps();assert int(bank.counts[0])==2489
    assert terrain.load_periodic_maps() is bank
    assert terrain.load_periodic_maps(parameters={'world_map_seed':41002}).asset_hash!=bank.asset_hash
    uniform=terrain.load_periodic_maps(parameters={'obstacle_density_std_per_1000m2':0,'obstacle_radius_std_m':0})
    assert torch.unique(uniform.radii).numel()==1
    cells=torch.floor(uniform.centers[0]/100).long();ids=cells[:,0]*10+cells[:,1]
    assert torch.equal(torch.bincount(ids,minlength=100),torch.full((100,),25))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_map_frozen_reset_restore(device):
    e=make_env(device);bank=e.periodic_bank;origin=e.map_origin_xy.clone()
    for _ in range(3):e.reset(build_observation=False)
    assert e.periodic_bank is bank and not e.map_id.any()
    assert not torch.equal(origin,e.map_origin_xy)
    state=copy.deepcopy(e.state_dict());e.load_state_dict(state)
    state['map_id'][0]=1
    with pytest.raises(ValueError,match='identity'):e.load_state_dict(state)
    other=make_env(device,world_map_seed=41002)
    with pytest.raises(ValueError,match='scene_contract'):other.load_state_dict(e.state_dict())


@pytest.mark.parametrize('key',terrain.WORLD_BOUNDS)
@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,True])
def test_validation(key,value):
    with pytest.raises(ValueError):gr._validated_viewer_settings({key:value})
