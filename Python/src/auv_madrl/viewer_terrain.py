"""Read-only Viewer tiles from frozen world descriptions, outside physics locks."""
from __future__ import annotations

from collections import OrderedDict
import copy
import json
import threading

import torch

from .terrain import load_periodic_maps, periodic_query


class TerrainReadError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class ViewerTerrain:
    def __init__(self):
        self._lock = threading.Lock()
        self._world = None
        self._cache = OrderedDict()
        self.hits = self.generated_blocks = 0

    def publish(self, descriptor, overrides, fixed):
        # The caller is the owning Viewer thread. Never retain environment tensors.
        world = (copy.deepcopy(descriptor), copy.deepcopy(overrides), copy.deepcopy(fixed))
        with self._lock:
            previous = self._world
            if previous is not None and previous[0] == world[0]:
                return
            self._world = world
            self._cache.clear()

    @staticmethod
    def _integer(query, name):
        raw = query.get(name)
        if isinstance(raw, list):
            if len(raw) != 1:
                raise TerrainReadError(400, 'Duplicate terrain field')
            raw = raw[0]
        try:
            value = int(raw)
        except (ValueError, TypeError, OverflowError):
            raise TerrainReadError(400, 'Invalid terrain integer: ' + name) from None
        if str(value) != str(raw) or abs(value) > 2**53 - 1:
            raise TerrainReadError(400, 'Unsafe terrain integer: ' + name)
        return value

    def read(self, query):
        if query.get('preview') in ('1', ['1']):
            return self.preview(query)
        fields = {k: self._integer(query, k) for k in
                  ('world_epoch', 'world_geometry_revision', 'request_id', 'cx0', 'cz0', 'nx', 'nz')}
        if fields['nx'] <= 0 or fields['nz'] <= 0 or fields['nx'] * fields['nz'] > 1024:
            raise TerrainReadError(400, 'Terrain request must contain 1..1024 blocks')
        if any(abs(fields[k] + fields[n] - 1) > 2**53 - 1 for k, n in (('cx0','nx'),('cz0','nz'))):
            raise TerrainReadError(400, 'Terrain block extent is unsafe')
        with self._lock:
            world = self._world
            if world is None:
                raise TerrainReadError(503, 'Terrain description is not ready')
            d, overrides, fixed = world
            identity = (d['world_epoch'], d['world_geometry_revision'])
            if identity != (fields['world_epoch'], fields['world_geometry_revision']):
                raise TerrainReadError(409, 'Terrain world changed')
            coords = [(x,z) for z in range(fields['cz0'],fields['cz0']+fields['nz'])
                      for x in range(fields['cx0'],fields['cx0']+fields['nx'])]
            cached = {c:self._cache[c] for c in coords if c in self._cache}
            for c in cached:
                self._cache.move_to_end(c)
            self.hits += len(cached)
        missing = [c for c in coords if c not in cached]
        if missing:
            generated = {c: [] for c in missing}
            if not d['manual_fixed'] and d['density_per_1000m2']:
                blocks = torch.tensor(missing, dtype=torch.int64)
                bank = load_periodic_maps(parameters=d['world_parameters'])
                if bank.asset_hash != d['bank_hash']:
                    raise TerrainReadError(409, 'Terrain asset changed')
                lower = blocks.float()*d['block_size_m']
                centers, radii, keys, active = periodic_query(bank,
                    torch.full((len(blocks),),d['map_id'],dtype=torch.long), lower,
                    lower+d['block_size_m'])
                # Each tile owns centers in its half-open cell; circle geometry is uncut.
                active &= (centers >= lower[:,None]).all(-1) & (centers < (lower+25)[:,None]).all(-1)
                rows, slots = active.nonzero(as_tuple=True)
                centers -= torch.tensor(d['map_origin_xy'])[None,None]
                for row, key, (x,z), radius in zip(rows.tolist(),keys[rows,slots].tolist(),
                                                 centers[rows,slots].tolist(),radii[rows,slots].tolist()):
                    if tuple(key) not in overrides:
                        generated[missing[row]].append(dict(id='t:'+':'.join(map(str,key)),x=x,z=z,r=radius))
            additions = fixed if d['manual_fixed'] else [
                dict(id=('m:'+str(k[3]) if k[0]==-(1<<63) else 't:'+':'.join(map(str,k))),
                     x=v[0][0],z=v[0][1],r=v[1]) for k,v in overrides.items() if v is not None]
            size = d['block_size_m']
            for cx,cz in missing:
                x0,z0 = d['phase'][0]+cx*size,d['phase'][1]+cz*size
                for o in additions:
                    dx=o['x']-max(x0,min(o['x'],x0+size));dz=o['z']-max(z0,min(o['z'],z0+size))
                    if dx*dx+dz*dz <= o['r']**2:
                        generated[cx,cz].append(o)
                cached[cx,cz]=json.dumps(dict(cx=cx,cz=cz,obstacles=generated[cx,cz]),
                    separators=(',',':'),allow_nan=False).encode()
            with self._lock:
                if self._world is not world:
                    raise TerrainReadError(409, 'Terrain world changed during generation')
                for c in missing:
                    self._cache[c]=cached[c]
                self.generated_blocks += len(missing)
                while len(self._cache)>1024:
                    self._cache.popitem(last=False)
        header=json.dumps(dict(schema='viewer_world_tiles_v1',world_epoch=identity[0],
            world_geometry_revision=identity[1],request_id=fields['request_id']),separators=(',',':')).encode()
        result=header[:-1]+b',"blocks":['+b','.join(cached[c] for c in coords)+b']}'
        with self._lock:
            if self._world is not world:
                raise TerrainReadError(409, 'Terrain world changed before response')
        return result

    def preview(self, query):
        fields = {k:self._integer(query,k) for k in ('world_epoch','world_geometry_revision','request_id')}
        with self._lock:
            world = self._world
            if world is None:
                raise TerrainReadError(503, 'Terrain description is not ready')
            d = world[0]
            if any(fields[k] != d[k] for k in ('world_epoch','world_geometry_revision')):
                raise TerrainReadError(409, 'Terrain world changed')
        if d['manual_fixed']:
            raise TerrainReadError(400, 'Fixed geometry has no periodic world preview')
        bank = load_periodic_maps(parameters=d['world_parameters'])
        if bank.asset_hash != d['bank_hash']:
            raise TerrainReadError(409, 'Terrain asset changed')
        family = d['map_id']
        count = int(bank.counts[family])
        centers = bank.centers[family,:count].tolist()
        radii = bank.radii[family,:count].tolist()
        result = dict(**fields, bank_hash=bank.asset_hash, id=str(d['world_parameters']['world_map_seed']),
                      density=d['density_per_1000m2'], length_m=1000., target=None,
                      obstacles=[dict(x=x,z=z,r=r) for (x,z),r in zip(centers,radii)])
        with self._lock:
            if self._world is not world:
                raise TerrainReadError(409, 'Terrain world changed before response')
        return json.dumps(result,separators=(',',':'),allow_nan=False).encode()
