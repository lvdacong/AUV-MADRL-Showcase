from __future__ import annotations

import math
import hashlib
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch


TERRAIN_BLOCK_SIZE_M = 25.0
PERIODIC_MAP_SCHEMA = "periodic_lognormal_regions_v1"
PERIODIC_MAP_LENGTH_M = 1000.0
WORLD_DEFAULTS = dict(obstacle_radius_mean_m=1.8, obstacle_radius_std_m=.7,
                      obstacle_density_mean_per_1000m2=2.5, obstacle_density_std_per_1000m2=1.5,
                      world_map_seed=41001)
WORLD_BOUNDS = dict(obstacle_radius_mean_m=(.2,5.), obstacle_radius_std_m=(0.,3.),
                    obstacle_density_mean_per_1000m2=(0.,10.), obstacle_density_std_per_1000m2=(0.,5.))
_PERIODIC_BANKS = {}


def world_parameters(value):
    result = {k: value.get(k,v) for k,v in WORLD_DEFAULTS.items()}
    for k,(lo,hi) in WORLD_BOUNDS.items():
        v=result[k]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f'{k} must be finite inside [{lo}, {hi}]')
        result[k]=float(v)
    if type(result['world_map_seed']) is not int or not 0 <= result['world_map_seed'] <= 2147483647:
        raise ValueError('world_map_seed must be an integer inside [0, 2147483647]')
    return result


def _world_uniform(seed,a,b,c):
    # Exact uint32 arithmetic shared with the approved HTML example.
    x=((a+1)*374761393 ^ (b+1)*668265263 ^ (c+1)*1442695041 ^ seed)&0xffffffff
    x=((x^(x>>13))*1274126177)&0xffffffff
    return ((x^(x>>16))&0xffffffff)/4294967296


def _world_normal(seed,a,b,c):
    return math.sqrt(-2*math.log(max(1e-10,_world_uniform(seed,a,b,c))))*math.cos(2*math.pi*_world_uniform(seed,a,b,c+1))


def _world_sample(mean,std,z):
    if mean == 0: return 0.
    if std == 0: return mean
    v=math.log1p((std/mean)**2)
    return mean*math.exp(math.sqrt(v)*z-v/2)


@dataclass(frozen=True)
class PeriodicMapBank:
    centers: torch.Tensor
    radii: torch.Tensor
    counts: torch.Tensor
    cell_ids: torch.Tensor
    cell_mask: torch.Tensor
    asset_hash: str
    max_radius: float


def load_periodic_maps(device='cpu', *, parameters=None):
    import json
    parameters=world_parameters(parameters or {})
    identity=json.dumps(parameters,sort_keys=True,separators=(',',':'))
    device=str(torch.device(device));key=(identity,device)
    if key in _PERIODIC_BANKS: return _PERIODIC_BANKS[key]
    if device != 'cpu':
        cpu=load_periodic_maps(parameters=parameters)
        bank=PeriodicMapBank(*(v.to(device) for v in (cpu.centers,cpu.radii,cpu.counts,cpu.cell_ids,cpu.cell_mask)),cpu.asset_hash,cpu.max_radius)
    else:
        seed=parameters['world_map_seed'];centers=[];radii=[]
        for cell in range(100):
            density=_world_sample(parameters['obstacle_density_mean_per_1000m2'],parameters['obstacle_density_std_per_1000m2'],_world_normal(seed,cell,0,10))
            for j in range(math.floor(density*10+.5)):
                centers.append(((cell%10)*100+100*_world_uniform(seed,cell,j,20),(cell//10)*100+100*_world_uniform(seed,cell,j,21)))
                radii.append(_world_sample(parameters['obstacle_radius_mean_m'],parameters['obstacle_radius_std_m'],_world_normal(seed,cell,j,30)))
        xy=np.asarray(centers,dtype=np.float32).reshape(-1,2);r=np.asarray(radii,dtype=np.float32)
        cell=np.floor(xy/25).astype(np.int64);cell=cell[:,0]*40+cell[:,1]
        occupancy=np.bincount(cell,minlength=1600);width=int(occupancy.max(initial=0))
        ids=np.zeros((1,1600,width),dtype=np.int64);mask=np.arange(width)[None,None]<occupancy[None,:,None]
        order=np.argsort(cell,kind='stable');offset=np.r_[0,np.cumsum(occupancy)]
        for c in range(1600): ids[0,c,:occupancy[c]]=order[offset[c]:offset[c+1]]
        digest=hashlib.sha256(PERIODIC_MAP_SCHEMA.encode()+identity.encode()+xy.tobytes()+r.tobytes()).hexdigest()
        bank=PeriodicMapBank(torch.from_numpy(xy[None]),torch.from_numpy(r[None]),torch.tensor([len(r)]),torch.from_numpy(ids),torch.from_numpy(mask),digest,float(r.max(initial=0)))
    # A small LRU retains active map objects via their owning environments, without
    # retaining every slider position forever.
    if len(_PERIODIC_BANKS)>=8: _PERIODIC_BANKS.pop(next(iter(_PERIODIC_BANKS)))
    _PERIODIC_BANKS[key]=bank
    return bank


def periodic_map_origins(seeds, attempts):
    zero = torch.zeros_like(seeds)
    return _keyed_uniform_fields(seeds, zero, zero, attempts,
                                 (0x4F524758, 0x4F52475A)).float()*1000


def periodic_query(bank, map_ids, lower, upper):
    """Complete circle/AABB query in continuous map coordinates, stable x/z/ID order.

    Returns independently owned centers, radii, (map,base,tile_x,tile_z), mask.
    Only relevant cells are gathered. A single batch extent sync is permitted for
    full physical candidates; no per-lane generation or all-map distance matrix.
    """
    hi_r = bank.max_radius
    torch._assert_async((torch.isfinite(lower).all() & torch.isfinite(upper).all()
                         & (upper >= lower).all()
                         & (map_ids >= 0).all() & (map_ids == 0).all()), 'Invalid periodic query')
    first = torch.floor((lower-hi_r)/25).long()
    last = torch.floor((upper+hi_r)/25).long()
    extent = (last-first+1).amax(0).tolist() if len(lower) else [0, 0]
    x, z = torch.meshgrid(torch.arange(extent[0], device=lower.device),
                          torch.arange(extent[1], device=lower.device), indexing='ij')
    cells = first[:, None] + torch.stack((x,z), -1).reshape(1,-1,2)
    valid_cell = (cells <= last[:, None]).all(-1)
    canonical = cells.remainder(40)
    tile = torch.div(cells, 40, rounding_mode='floor')
    c = canonical[...,0]*40+canonical[...,1]
    ids = bank.cell_ids[map_ids[:,None], c]
    active = bank.cell_mask[map_ids[:,None], c] & valid_cell[...,None]
    maps = map_ids[:,None,None].expand_as(ids)
    centers = bank.centers[maps, ids] + tile[:,:,None].to(lower.dtype)*1000
    radii = bank.radii[maps, ids]
    nearest = centers.maximum(lower[:,None,None]).minimum(upper[:,None,None])
    active &= (centers-nearest).square().sum(-1) <= radii.square()
    keys = torch.cat((maps[...,None],ids[...,None],tile[:,:,None].expand(*ids.shape,2)), -1)
    return (centers.flatten(1,2).clone(), radii.flatten(1,2).clone(),
            keys.flatten(1,2).clone(), active.flatten(1,2).clone())


def compact_first_indices(mask, capacity=None):
    """Stable compaction; fixed policy capacity never synchronizes or sorts."""
    count = mask.sum(-1)
    width = int(count.max().item()) if capacity is None and count.numel() else (capacity or 0)
    index = torch.zeros((len(mask),width+1),dtype=torch.long,device=mask.device)
    rank = mask.long().cumsum(-1)-1
    destination = torch.where(mask & (rank < width), rank, width)
    columns = torch.arange(mask.shape[1],device=mask.device)[None].expand_as(mask)
    index.scatter_(1,destination,columns)
    return index[:,:width].clone(), torch.arange(width,device=mask.device)[None] < count[:,None]


_POISSON_CDF_CACHE: dict[tuple[str, torch.dtype, float], torch.Tensor] = {}
_POISSON_CDF_TABLE_CACHE: dict[tuple[str, torch.dtype, float], torch.Tensor] = {}


@dataclass(frozen=True)
class TerrainBatch:
    """Packed deterministic terrain for a bounded set of world blocks."""

    block_lanes: torch.Tensor
    block_coordinates: torch.Tensor
    block_counts: torch.Tensor
    obstacle_lanes: torch.Tensor
    obstacle_centers: torch.Tensor
    obstacle_radii: torch.Tensor
    obstacle_keys: torch.Tensor


def _zigzag(value: torch.Tensor) -> torch.Tensor:
    value = value.to(torch.int64)
    return torch.where(value >= 0, value * 2, -value * 2 - 1)


def _mix31(value: torch.Tensor) -> torch.Tensor:
    """Stable integer mixing supported identically on CPU and CUDA."""

    mask = 0x7FFFFFFF
    value = value.to(torch.int64) & mask
    value = ((value ^ (value >> 16)) * 73_212_421) & mask
    value = ((value ^ (value >> 15)) * 1_291_822_179) & mask
    return (value ^ (value >> 16)) & mask


def _keyed_uniform_fields(
    episode_seed: torch.Tensor,
    coordinate_x: torch.Tensor,
    coordinate_z: torch.Tensor,
    slot: torch.Tensor,
    fields: tuple[int, ...],
) -> torch.Tensor:
    """Share address mixing and batch the fixed field/high-low streams."""
    seed, coordinate_x, coordinate_z, slot = torch.broadcast_tensors(
        episode_seed, coordinate_x, coordinate_z, slot)
    value = torch.stack([seed ^ key for field in fields
                         for key in (int(field), int(field) ^ 0x6D2B79F5)], dim=-1)
    value = _mix31(value)
    for component, salt in (
        (_zigzag(coordinate_x), 0x13579B),
        (_zigzag(coordinate_z), 0x2468AD),
        (slot.to(torch.int64), 0x10203),
    ):
        low = component & 0x7FFFFFFF
        high = component >> 31
        component_hash = _mix31(low + salt) ^ _mix31(high + 3 * salt)
        value = _mix31(value ^ component_hash.unsqueeze(-1))
    value = (value[..., 0::2] << 1) | (value[..., 1::2] & 1)
    return (value.to(torch.float64) + 0.5) / 4_294_967_296.0


def keyed_uniform(
    episode_seed: torch.Tensor,
    coordinate_x: torch.Tensor,
    coordinate_z: torch.Tensor,
    slot: torch.Tensor,
    field: int,
) -> torch.Tensor:
    """Address one random value by full signed block coordinate and field."""

    return _keyed_uniform_fields(episode_seed, coordinate_x, coordinate_z, slot, (field,))[..., 0]


def poisson_cdf(mean: float) -> tuple[float, ...]:
    """Build a complete inverse-CDF table for every representable keyed U."""

    if not math.isfinite(mean) or mean < 0:
        raise ValueError("Poisson mean must be finite and non-negative")
    probability = math.exp(-mean)
    cumulative = probability
    result = [cumulative]
    maximum_u = 1.0 - 2.0**-33
    count = 0
    while cumulative < maximum_u:
        count += 1
        probability *= mean / count
        cumulative += probability
        if count > 4096 or probability == 0.0:
            raise ValueError("Poisson CDF did not cover the keyed random domain")
        result.append(min(1.0, cumulative))
    return tuple(result)


def poisson_counts(uniform: torch.Tensor, mean: float) -> torch.Tensor:
    key = (str(uniform.device), torch.float64, float(mean))
    cdf = _POISSON_CDF_CACHE.get(key)
    if cdf is None:
        cdf = torch.tensor(poisson_cdf(mean), dtype=torch.float64, device=uniform.device)
        _POISSON_CDF_CACHE[key] = cdf
    return torch.searchsorted(cdf, uniform.to(torch.float64), right=False).to(torch.int64)


def poisson_count_table(device: torch.device, block_size_m: float) -> torch.Tensor:
    """Return padded inverse-CDF rows for every supported terrain density."""

    size = float(block_size_m)
    key = (str(device), torch.float64, size)
    cached = _POISSON_CDF_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    rows = [poisson_cdf(0.5 * density * size**2 / 1000.0) for density in range(21)]
    # Keep an explicit padded one at the tail of even the longest row.
    width = max(map(len, rows)) + 1
    table = torch.ones((21, width), dtype=torch.float64, device=device)
    for density, row in enumerate(rows):
        table[density, : len(row)] = torch.tensor(
            row, dtype=torch.float64, device=device
        )
    _POISSON_CDF_TABLE_CACHE[key] = table
    return table


def terrain_phase(episode_seeds: torch.Tensor, block_size_m: float) -> torch.Tensor:
    zeros = torch.zeros_like(episode_seeds)
    values = _keyed_uniform_fields(episode_seeds, zeros, zeros, zeros, (0x50484158, 0x5048415A))
    return values.to(torch.float32) * float(block_size_m)


def validate_density_bounds(bounds):
    if (not isinstance(bounds, (tuple, list)) or len(bounds) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or x * 2 != round(x * 2) for x in bounds)
            or not 0 <= bounds[0] <= bounds[1] <= 10):
        raise ValueError("Terrain density requires [0,10] bounds in steps of 0.5")
    return tuple(float(x) for x in bounds)


def episode_density(
    episode_seeds: torch.Tensor,
    bounds: tuple[float, float],
) -> torch.Tensor:
    """Sample one physical density (centers / 1000 m²) per episode."""
    minimum, maximum = validate_density_bounds(bounds)
    minimum, maximum = int(minimum * 2), int(maximum * 2)
    if episode_seeds.ndim != 1 or episode_seeds.dtype != torch.int64:
        raise ValueError("Episode seeds must be an int64 lane vector")
    zeros = torch.zeros_like(episode_seeds)
    uniform = keyed_uniform(
        episode_seeds, zeros, zeros, zeros, 0x45504445
    )
    return (torch.floor(uniform * (maximum - minimum + 1)) + minimum).to(torch.float32) * 0.5


def required_blocks(
    positions: torch.Tensor,
    active: torch.Tensor,
    phase: torch.Tensor,
    *,
    block_size_m: float,
    coverage_radius_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unique ``(lane,cx,cz)`` rows intersecting active coverage discs."""

    if positions.ndim != 3 or positions.shape[-1] != 2 or active.shape != positions.shape[:2]:
        raise ValueError("Terrain coverage requires [lanes,craft,2] positions and matching mask")
    block_size = float(block_size_m)
    radius = float(coverage_radius_m)
    if block_size <= 0 or radius <= 0:
        raise ValueError("Terrain block size and coverage radius must be positive")
    extent = math.ceil(radius / block_size) + 1
    offsets_1d = torch.arange(-extent, extent + 1, device=positions.device, dtype=torch.int64)
    ox, oz = torch.meshgrid(offsets_1d, offsets_1d, indexing="ij")
    offsets = torch.stack((ox, oz), dim=-1).reshape(-1, 2)
    base = torch.floor((positions - phase[:, None]) / block_size).to(torch.int64)
    candidates = base[:, :, None] + offsets[None, None]
    block_min = phase[:, None, None] + candidates.to(torch.float32) * block_size
    block_max = block_min + block_size
    nearest = torch.maximum(block_min, torch.minimum(positions[:, :, None], block_max))
    intersects = (nearest - positions[:, :, None]).square().sum(-1) <= radius * radius
    intersects &= active[:, :, None]
    lanes, craft, candidate = intersects.nonzero(as_tuple=True)
    if lanes.numel() == 0:
        return lanes, torch.empty((0, 2), dtype=torch.int64, device=positions.device)
    rows = torch.cat((lanes[:, None], candidates[lanes, craft, candidate]), dim=1)
    rows = torch.unique(rows, dim=0, sorted=True)
    return rows[:, 0], rows[:, 1:]


def generate_terrain(
    episode_seeds: torch.Tensor,
    episode_densities: torch.Tensor,
    block_lanes: torch.Tensor,
    block_coordinates: torch.Tensor,
    phase: torch.Tensor,
    *,
    block_size_m: float,
    obstacle_radius_range_m: tuple[float, float],
) -> TerrainBatch:
    """Generate packed block contents without depending on access order."""

    if (episode_densities.shape != episode_seeds.shape or episode_seeds.ndim != 1
            or episode_densities.dtype != torch.float32
            or episode_densities.device != episode_seeds.device):
        raise ValueError("Terrain density must be a float32 physical-density lane vector on the seed device")
    # CUDA assertion stays on device: never synchronize density to the host on refresh.
    valid_density = ((episode_densities >= 0) & (episode_densities <= 10) & (episode_densities * 2 == (episode_densities * 2).round())).all()
    if episode_densities.is_cuda:
        torch._assert_async(valid_density, "Terrain density must stay inside [0,10] in steps of 0.5")
    elif not valid_density:
        raise ValueError("Terrain density must stay inside [0,10] in steps of 0.5")
    if block_coordinates.shape != (block_lanes.numel(), 2):
        raise ValueError("Packed terrain block coordinates do not align")
    if block_lanes.numel() == 0:
        empty = torch.empty(0, dtype=torch.int64, device=episode_seeds.device)
        return TerrainBatch(block_lanes, block_coordinates, empty, empty,
                            torch.empty((0, 2), device=episode_seeds.device),
                            torch.empty(0, device=episode_seeds.device),
                            torch.empty((0, 3), dtype=torch.int64, device=episode_seeds.device))
    cx, cz = block_coordinates[:, 0], block_coordinates[:, 1]
    block_seed = episode_seeds[block_lanes]
    zero = torch.zeros_like(cx)
    density = episode_densities[block_lanes]
    count_u = keyed_uniform(block_seed, cx, cz, zero, 0x434F554E)
    # Build all 21 complete rows once, then use one hot-path gather/comparison.
    # ``>`` is equivalent to searchsorted(..., right=False), including exact
    # CDF boundaries.  Padding with one cannot change a representable keyed U.
    count_table = poisson_count_table(episode_seeds.device, block_size_m)
    selected_cdf = count_table[(density * 2).to(torch.int64)]
    counts = (count_u[:, None].to(torch.float64) > selected_cdf).sum(-1).to(torch.int64)
    slots = torch.arange(
        count_table.shape[1], dtype=torch.int64, device=episode_seeds.device
    )
    live = slots[None] < counts[:, None]
    block_index, slot = live.nonzero(as_tuple=True)
    obstacle_lanes = block_lanes[block_index]
    obstacle_cx = cx[block_index]
    obstacle_cz = cz[block_index]
    obstacle_seed = block_seed[block_index]
    values = _keyed_uniform_fields(obstacle_seed, obstacle_cx, obstacle_cz, slot,
                                   (0x43454E58, 0x43454E5A, 0x52414449))
    lower, upper = (float(value) for value in obstacle_radius_range_m)
    centers = values[..., :2].to(torch.float32) * float(block_size_m)
    centers += phase[obstacle_lanes] + torch.stack((obstacle_cx, obstacle_cz), dim=-1).to(torch.float32) * float(block_size_m)
    radii = (lower + (upper - lower) * values[..., 2]).to(torch.float32)
    keys = torch.stack((obstacle_cx, obstacle_cz, slot), dim=-1)
    return TerrainBatch(block_lanes, block_coordinates, counts, obstacle_lanes, centers, radii, keys)


def pack_obstacles(
    batch_size: int, terrain: TerrainBatch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack terrain obstacles into dense lane storage with a truthful mask."""

    device = terrain.obstacle_lanes.device
    counts = torch.bincount(terrain.obstacle_lanes, minlength=int(batch_size))
    capacity = int(counts.max().item()) if counts.numel() else 0
    centers = torch.zeros((batch_size, capacity, 2), dtype=torch.float32, device=device)
    radii = torch.zeros((batch_size, capacity), dtype=torch.float32, device=device)
    keys = torch.zeros((batch_size, capacity, terrain.obstacle_keys.shape[-1]), dtype=torch.int64, device=device)
    active = torch.arange(capacity, device=device)[None] < counts[:, None]
    if terrain.obstacle_lanes.numel():
        order = torch.arange(terrain.obstacle_lanes.numel(), device=device)
        for field in (*reversed(terrain.obstacle_keys.unbind(-1)), terrain.obstacle_lanes):
            order = order[torch.argsort(field[order], stable=True)]
        ordered_keys = terrain.obstacle_keys[order]
        ordered_lanes = terrain.obstacle_lanes[order]
        duplicate = (ordered_keys[1:] == ordered_keys[:-1]).all(-1) & (ordered_lanes[1:] == ordered_lanes[:-1])
        torch._assert_async(~duplicate.any(), 'Duplicate active obstacle world key')
        lanes = terrain.obstacle_lanes[order]
        starts = torch.cumsum(counts, 0) - counts
        slots = torch.arange(lanes.numel(), device=device) - starts[lanes]
        centers[lanes, slots] = terrain.obstacle_centers[order]
        radii[lanes, slots] = terrain.obstacle_radii[order]
        keys[lanes, slots] = terrain.obstacle_keys[order]
    return centers, radii, active, keys


def validate_episode_density_state(state: dict, bounds: tuple[float, float], batch_size: int) -> None:
    """Validate checkpoint density before any consumer mutates live state."""
    density = state.get("terrain_episode_density")
    lo, hi = bounds
    if (not isinstance(density, torch.Tensor) or density.dtype != torch.float32
            or density.shape != (batch_size,)
            or not bool(((density >= lo) & (density <= hi)).all())):
        raise ValueError("Checkpoint terrain_episode_density is missing or out of bounds")
    if not torch.equal(density, episode_density(
            state["episode_seeds"].to(density.device), bounds)):
        raise ValueError("Checkpoint density does not match original episode seed")
    revision = state.get("terrain_override_revision")
    if (not isinstance(revision, torch.Tensor) or revision.dtype != torch.int64
            or revision.shape != (batch_size,) or bool((revision < 0).any())):
        raise ValueError("Checkpoint terrain override revision is missing or invalid")
