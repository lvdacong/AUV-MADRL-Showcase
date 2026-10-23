from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActorSpawnBatch:
    learner_counts: torch.Tensor
    learner_positions: torch.Tensor
    learner_headings: torch.Tensor
    learner_active: torch.Tensor
    defender_counts: torch.Tensor
    defender_positions: torch.Tensor
    defender_headings: torch.Tensor
    defender_active: torch.Tensor
    target_counts: torch.Tensor
    target_positions: torch.Tensor
    target_headings: torch.Tensor
    target_active: torch.Tensor


@dataclass(frozen=True)
class SceneBatch:
    obstacle_counts: torch.Tensor
    obstacle_centers: torch.Tensor
    obstacle_radii: torch.Tensor
    learner_counts: torch.Tensor
    learner_positions: torch.Tensor
    defender_counts: torch.Tensor
    defender_positions: torch.Tensor
    target_positions: torch.Tensor
    target_counts: torch.Tensor | None = None


def compose_episode_seeds(
    worker_seed: int,
    area_indices: torch.Tensor,
    episode_indices: torch.Tensor,
) -> torch.Tensor:
    """Vectorized form of the shared 32-bit episode-seed composition."""

    area_indices = area_indices.to(dtype=torch.int64)
    episode_indices = episode_indices.to(device=area_indices.device, dtype=torch.int64)
    mask = 0xFFFFFFFF
    value = torch.full_like(area_indices, int(worker_seed) & mask)
    mixed = (0x9E3779B9 + area_indices + ((value << 6) & mask) + (value >> 2)) & mask
    value = torch.bitwise_xor(value, mixed) & mask
    mixed = (0x85EBCA6B + episode_indices + ((value << 6) & mask) + (value >> 2)) & mask
    return torch.bitwise_xor(value, mixed) & 0x7FFFFFFF


def deterministic_uniform(
    episode_seeds: torch.Tensor,
    shape: tuple[int, ...],
    *,
    counter_offset: int = 0,
) -> torch.Tensor:
    """Return deterministic independent-looking U(0,1) values for every episode."""

    count = math.prod(shape)
    counter = torch.arange(
        int(counter_offset),
        int(counter_offset) + count,
        dtype=torch.int64,
        device=episode_seeds.device,
    )
    value = (episode_seeds[:, None] + counter[None, :] * 1_103_515_245 + 12_345) & 0x7FFFFFFF
    value = ((value ^ (value >> 16)) * 73_212_421) & 0x7FFFFFFF
    value = ((value ^ (value >> 15)) * 1_291_822_179) & 0x7FFFFFFF
    value ^= value >> 16
    random = (value.to(torch.float32) + 0.5) / 2_147_483_648.0
    random.clamp_max_(1.0 - torch.finfo(torch.float32).eps / 2.0)
    return random.reshape(episode_seeds.shape[0], *shape)


_uniform = deterministic_uniform


def _sample_counts(random: torch.Tensor, bounds: tuple[int, int]) -> torch.Tensor:
    minimum, maximum = (int(value) for value in bounds)
    if minimum == maximum:
        return torch.full(random.shape, minimum, dtype=torch.int64, device=random.device)
    return torch.floor(random * (maximum - minimum + 1)).to(torch.int64) + minimum


def _active_mask(counts: torch.Tensor, capacity: int) -> torch.Tensor:
    slots = torch.arange(int(capacity), dtype=torch.int64, device=counts.device)
    return slots[None, :] < counts[:, None]


def episode_actor_randomization(
    episode_seeds: torch.Tensor,
    *,
    learner_counts: torch.Tensor,
    learner_capacity: int,
    learner_speed_ratio_range: tuple[float, float],
    defender_counts: torch.Tensor,
    defender_capacity: int,
    defender_speed_ratio_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate deterministic per-episode headings and speed ratios on device."""

    learner_active = _active_mask(learner_counts, learner_capacity)
    defender_active = _active_mask(defender_counts, defender_capacity)
    learner_heading = 2.0 * torch.pi * deterministic_uniform(
        episode_seeds ^ 0x4C484541,
        (learner_capacity,),
    )
    defender_heading = 2.0 * torch.pi * deterministic_uniform(
        episode_seeds ^ 0x44484541,
        (defender_capacity,),
    )
    learner_minimum, learner_maximum = learner_speed_ratio_range
    learner_speed_ratio = float(learner_minimum) + (
        float(learner_maximum) - float(learner_minimum)
    ) * deterministic_uniform(
        episode_seeds ^ 0x4C535044,
        (learner_capacity,),
    )
    defender_minimum, defender_maximum = defender_speed_ratio_range
    defender_speed_ratio = float(defender_minimum) + (
        float(defender_maximum) - float(defender_minimum)
    ) * deterministic_uniform(
        episode_seeds ^ 0x44535044,
        (defender_capacity,),
    )
    return (
        torch.where(learner_active, learner_heading, 0.0),
        torch.where(defender_active, defender_heading, 0.0),
        torch.where(learner_active, learner_speed_ratio, 1.0),
        torch.where(defender_active, defender_speed_ratio, 1.0),
    )


def area_uniform_positions(random, inner, outer):
    radius = (inner*inner + random[..., 0]*(outer*outer-inner*inner)).sqrt()
    theta = 2*torch.pi*random[..., 1]
    return torch.stack((radius*theta.sin(), radius*theta.cos()), -1)


def _packed_plan(counts, obstacle_counts, capacities):
    """Only fixed-size integer metadata is padded; candidates and distances are packed."""
    device, m = counts.device, counts.shape[0]
    total_capacity = sum(capacities)
    slots = torch.arange(total_capacity, device=device)
    kinds = torch.repeat_interleave(torch.arange(3, device=device), torch.tensor(capacities, device=device), output_size=total_capacity)
    starts = torch.tensor((0, capacities[0], capacities[0]+capacities[1]), device=device)
    local_slots = slots-starts[kinds]
    live = local_slots[None] < counts[:, kinds]
    lanes, global_slots = live.nonzero(as_tuple=True)
    obstacle_capacity = int(obstacle_counts.shape[1]) if obstacle_counts.ndim == 2 else 0
    obstacle_live = obstacle_counts if obstacle_counts.ndim == 2 else counts.new_zeros((m, 0), dtype=torch.bool)
    ol, os = obstacle_live.nonzero(as_tuple=True)
    lookup = torch.full((m,total_capacity),-1,device=device,dtype=torch.long)
    lookup[lanes,global_slots] = torch.arange(lanes.numel(),device=device)
    olookup = torch.full((m,obstacle_capacity),-1,device=device,dtype=torch.long)
    olookup[ol,os] = torch.arange(ol.numel(),device=device)
    lane, left, right = (live[:,:,None] & live[:,None,:] & (slots[:,None]<slots[None,:])).nonzero(as_tuple=True)
    pairs = (lookup[lane,left],lookup[lane,right])
    lane, vessel, obstacle = (live[:,:,None]&obstacle_live[:,None,:]).nonzero(as_tuple=True)
    hazards = (lookup[lane,vessel],olookup[lane,obstacle])
    return lanes, global_slots, kinds[global_slots], local_slots[global_slots], ol, os, pairs, hazards


def _candidate_validity(points, origin, plan, obstacles, radii, spec, m):
    lanes, _, kinds, _, _, _, pairs, hazards = plan
    k = points.shape[1]
    fail = torch.zeros(points.shape[:2], dtype=torch.bool, device=points.device)
    distance = (points-origin[lanes,None]).norm(dim=-1)
    fail |= (kinds[:,None]==0) & ((distance < spec.attacker_spawn_inner_radius_m-1e-5) | (distance>spec.attacker_spawn_outer_radius_m+1e-5))
    fail |= (kinds[:,None]==1) & (distance > min(spec.defender_spawn_radius_m,spec.engagement_radius_m-spec.vessel_radius_m)-1e-5)
    failures = torch.zeros((m,k),device=points.device,dtype=torch.int32)
    failures.index_add_(0, lanes, fail.int())
    left,right = pairs
    conflict = (points[left]-points[right]).norm(dim=-1) <= 2*spec.vessel_radius_m+1e-5
    failures.index_add_(0,lanes[left],conflict.int())
    vessel, obstacle = hazards
    conflict = (points[vessel]-obstacles[obstacle,None]).norm(dim=-1) <= radii[obstacle,None]+spec.vessel_radius_m+1e-5
    failures.index_add_(0,lanes[vessel],conflict.int())
    return failures==0


def _proposal(seeds, origin, plan, spec, attempt, k=32):
    lanes, _, kinds, slots, *_ = plan
    domain = seeds[lanes] ^ (kinds*100003 + slots*200003 + 0x43524146)
    random = deterministic_uniform(domain, (k,2), counter_offset=attempt*64)
    world = spec.sense_radius_m-spec.vessel_radius_m-1e-5
    inner = torch.where(kinds==0, spec.attacker_spawn_inner_radius_m, 0.).float()
    outer = torch.where(kinds==0, spec.attacker_spawn_outer_radius_m,
                        min(spec.defender_spawn_radius_m,spec.engagement_radius_m-spec.vessel_radius_m)-1e-5).float()
    use_world = (kinds==2) | (world*world < outer.square()-inner.square())
    lower = torch.where(use_world,0.,inner)[:,None]
    upper = torch.where(use_world,world,outer)[:,None]
    offset = torch.where(use_world[:,None],0.,origin[lanes])
    return area_uniform_positions(random,lower,upper)+offset[:,None]


def sample_scenes(episode_seeds, spec, *, fixed=None, statistics=None):
    """32 whole-team proposals, eight batches, eight layouts; counts/radii never retried."""
    device, m = episode_seeds.device, episode_seeds.numel()
    caps = (spec.learner_count_range[1],spec.defender_count_range[1],spec.target_count_range[1])
    target_start = caps[0]+caps[1]
    ocap = fixed.obstacle_centers.shape[1] if fixed is not None else 0
    if m == 0:
        return SceneBatch(torch.empty(0,dtype=torch.long,device=device),torch.empty((0,ocap,2),device=device),
            torch.empty((0,ocap),device=device),torch.empty(0,dtype=torch.long,device=device),torch.empty((0,caps[0],2),device=device),
            torch.empty(0,dtype=torch.long,device=device),torch.empty((0,caps[1],2),device=device),torch.empty((0,caps[2],2),device=device),torch.empty(0,dtype=torch.long,device=device))
    counts = torch.stack((_sample_counts(deterministic_uniform(episode_seeds ^ 0x4341544B,()),spec.learner_count_range),
                          _sample_counts(deterministic_uniform(episode_seeds ^ 0x43444546,()),spec.defender_count_range),
                          _sample_counts(deterministic_uniform(episode_seeds ^ 0x43544754,()),spec.target_count_range)),1)
    obstacle_counts = torch.zeros(m, dtype=torch.long, device=device)
    obstacles = torch.zeros((m,ocap,2),device=device)
    radii = torch.zeros((m,ocap),device=device)
    positions = torch.zeros((m,sum(caps),2),device=device)
    if fixed is not None:
        counts = torch.stack((fixed.learner_counts,fixed.defender_counts,fixed.target_counts if fixed.target_counts is not None else torch.ones_like(fixed.learner_counts)),1)
        obstacle_counts = fixed.obstacle_counts.clone()
        obstacles = fixed.obstacle_centers.clone()
        radii = fixed.obstacle_radii.clone()
        positions[:,target_start:] = fixed.target_positions
    pending = torch.arange(m,device=device)
    stats = dict(candidate_trials=0,candidate_accepts=0,layout_resamples=0)
    for layout in range(1 if fixed is not None else 8):
        if not pending.numel():break
        n = pending.numel()
        seeds = episode_seeds[pending]
        obs_live = _active_mask(obstacle_counts[pending],ocap)
        ol,os = obs_live.nonzero(as_tuple=True)
        if fixed is None:
            centers = area_uniform_positions(deterministic_uniform(seeds[ol] ^ (os*200003+0x4F425343),(2,),counter_offset=layout*2),0.,spec.sense_radius_m)
            obstacles[pending[ol],os] = centers
        # TGT #0 is the fixed per-episode task anchor. Dynamic motion starts only
        # after reset; other configurable targets are sampled independently below.
        target = positions[pending,target_start].clone()
        target_ok = torch.ones(n,device=device,dtype=torch.bool)
        positions[pending,target_start]=target
        successful = torch.zeros(n,device=device,dtype=torch.bool)
        todo = target_ok.nonzero(as_tuple=True)[0]
        for batch in range(8):
            if not todo.numel():break
            plan = _packed_plan(counts[pending[todo]],obs_live[todo],caps)
            lanes,gslots,kinds,local_slots,sl,ss,_,_ = plan
            pts = _proposal(seeds[todo],target[todo],plan,spec,layout*16+8+batch)
            target_slots = (kinds==2) & ((local_slots==0) | (fixed is not None))
            pts[target_slots] = positions[pending[todo[lanes[target_slots]]],gslots[target_slots],None]
            valid = _candidate_validity(pts,target[todo],plan,obstacles[pending[todo[sl]],ss],radii[pending[todo[sl]],ss],spec,todo.numel())
            ok = valid.any(1);first=valid.int().argmax(1)
            chosen = ok[lanes]
            positions[pending[todo[lanes[chosen]]],gslots[chosen]] = pts[chosen,first[lanes[chosen]]]
            successful[todo[ok]]=True
            stats['candidate_trials']+=todo.numel()*32
            stats['candidate_accepts']+=valid.sum()
            todo=todo[~ok]
        pending = pending[~successful]
        stats['layout_resamples'] += pending.numel() if layout<7 and fixed is None else 0
    if pending.numel():
        raise ValueError(f"Scene cannot be generated: {pending.numel()} lanes exhausted the bounded candidates; fixed geometry preserved")
    if statistics is not None:statistics.update(stats)
    return SceneBatch(obstacle_counts,obstacles,radii,counts[:,0],positions[:,:caps[0]],
                      counts[:,1],positions[:,caps[0]:target_start],positions[:,target_start:],counts[:,2])


def scene_validity(scene, spec):
    counts = torch.stack((scene.learner_counts,scene.defender_counts,scene.target_counts if scene.target_counts is not None else torch.ones_like(scene.learner_counts)),1)
    caps = (scene.learner_positions.shape[1],scene.defender_positions.shape[1],scene.target_positions.shape[1])
    plan = _packed_plan(counts,_active_mask(scene.obstacle_counts,scene.obstacle_centers.shape[1]),caps)
    lanes,slots,_,_,ol,os,_,_ = plan
    points = torch.cat((scene.learner_positions,scene.defender_positions,scene.target_positions),1)[lanes,slots,None]
    valid = _candidate_validity(points,scene.target_positions[:,0],plan,scene.obstacle_centers[ol,os],scene.obstacle_radii[ol,os],spec,counts.shape[0])[:,0]
    obstacle_ok = scene.obstacle_radii[ol,os] > 0
    fails=torch.zeros(counts.shape[0],device=counts.device,dtype=torch.int32)
    fails.index_add_(0,ol,(~obstacle_ok).int())
    return valid & (fails==0)


def sample_actor_spawns(episode_seeds, **arguments):
    from .scene_bank import SceneSpec
    v=arguments; radius=v['world_radius']; safety=v['safety_zone_specification']
    nc=v['learner_capacity'];nd=v['defender_capacity']
    spec=SceneSpec(radius,v['vessel_radius'],
        (0,nc),(0,nd),float(safety['defender_spawn_radius_m']),float(safety['attacker_spawn_inner_radius_m']),
        radius-v['vessel_radius']-1e-5,float(safety['engagement_radius_m']),(1,v['target_positions'].shape[1]))
    fixed=SceneBatch(v['obstacle_active'].sum(1),v['obstacle_centers'],v['obstacle_radii'],
        v['learner_counts'],torch.zeros((episode_seeds.numel(),nc,2),device=episode_seeds.device),
        v['defender_counts'],torch.zeros((episode_seeds.numel(),nd,2),device=episode_seeds.device),v['target_positions'],v.get('target_counts'))
    scene=sample_scenes(episode_seeds,spec,fixed=fixed)
    headings=episode_actor_randomization(episode_seeds,learner_counts=scene.learner_counts,learner_capacity=nc,
        learner_speed_ratio_range=(1.,1.),defender_counts=scene.defender_counts,defender_capacity=nd,defender_speed_ratio_range=(1.,1.))
    return ActorSpawnBatch(scene.learner_counts,scene.learner_positions,headings[0],_active_mask(scene.learner_counts,nc),
        scene.defender_counts,scene.defender_positions,headings[1],_active_mask(scene.defender_counts,nd),
        scene.target_counts,scene.target_positions,2*torch.pi*deterministic_uniform(episode_seeds ^ 0x54484541,(scene.target_positions.shape[1],)),
        _active_mask(scene.target_counts,scene.target_positions.shape[1]))
