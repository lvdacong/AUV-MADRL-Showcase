"""CPU Viewer randomness and bounded, in-memory records of applied actions."""
from __future__ import annotations

import copy
import hashlib
import sys

import torch


class ViewerPolicyNoise:
    def __init__(self, seed: int, maximum_decisions: int):
        self.seed = seed
        self.maximum_decisions = maximum_decisions
        self.tables = {}

    def at(self, role: str, slots: list[int], decision: int) -> torch.Tensor:
        if type(decision) is not int or decision < 0:
            raise ValueError('Policy decision must be a nonnegative integer')
        block, offset = divmod(decision, self.maximum_decisions + 1)
        values = []
        for slot in slots:
            key = (role, slot)
            cached = self.tables.get(key)
            if cached is None or block < cached[0]:
                seed = int.from_bytes(hashlib.blake2b(f'{self.seed}:{role}:{slot}'.encode(), digest_size=8).digest(), 'little')
                generator = torch.Generator(device='cpu').manual_seed(seed)
                cached = (-1, generator, None)
            index, generator, table = cached
            while index < block:
                table = torch.randn(self.maximum_decisions + 1, 2, generator=generator, dtype=torch.float32)
                index += 1
            self.tables[key] = (index, generator, table)
            values.append(table[offset])
        return torch.stack(values) if values else torch.empty(0, 2)


def payload_bytes(value, seen=None):
    seen = set() if seen is None else seen
    pending = [value]
    size = 0
    pop, extend, add = pending.pop, pending.extend, seen.add
    identity_of, size_of, type_of = id, sys.getsizeof, type
    tensor_type, none_type = torch.Tensor, type(None)
    while pending:
        item = pop()
        identity = identity_of(item)
        if identity in seen:
            continue
        add(identity)
        size += size_of(item)
        kind = type_of(item)
        if kind is dict:
            extend(item.keys())
            extend(item.values())
        elif kind is list or kind is tuple:
            extend(item)
        elif (kind is str or kind is float or kind is int or kind is bool
              or kind is bytes or kind is none_type):
            continue
        elif isinstance(item, tensor_type):
            size += item.numel() * item.element_size()
        elif isinstance(item, dict):
            extend(item.keys())
            extend(item.values())
        elif isinstance(item, (list, tuple)):
            extend(item)
    return size


def physical_digest(env):
    """Exclude wall clock/UI revision; include all environment continuation state."""
    digest = hashlib.sha256()
    def add(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(str(key).encode()); add(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value: add(item)
        else:
            digest.update(repr(value).encode())
    add(env.state_dict())
    return digest.digest()


class ViewerEpisodeRecord:
    BYTE_LIMIT = 32 * 1024 * 1024

    def __init__(self, initial, identity, contract, *, initial_terrain_coverage=None):
        self.initial = copy.deepcopy(initial)
        # Replay needs the exact local terrain footprint as well as its packed
        # obstacles. General checkpoint restore intentionally invalidates this
        # cache; re-querying it on Replay can change obstacle/contact slot layout.
        self.initial_terrain_coverage = copy.deepcopy(initial_terrain_coverage)
        self.identity = dict(identity)
        self.contract = contract
        self.intervals = []
        self.checks = {}
        self.complete = False
        self.error = None
        self.bytes = payload_bytes((self.initial, self.initial_terrain_coverage, self.identity, contract)) + 1024
        self._last_frame_key = None
        self._last_frame = None
        self._limit()

    def _limit(self):
        if self.bytes > self.BYTE_LIMIT:
            self.error = 'Last LIVE episode exceeded the 32 MiB recording limit'
            self.intervals.clear(); self.checks.clear(); self._last_frame = None
            self.initial = {}
            self.initial_terrain_coverage = None

    def append(self, start, end, actions, frame_key, frame, checksum):
        """Retain an owned, read-only frame; same-revision continuations may omit it."""
        if self.error or end <= start:
            return
        expected = self.intervals[-1]['end'] if self.intervals else 0
        if start != expected:
            self.error = f'Missing applied actions at tick {expected}'
            return
        if self.intervals and frame_key == self._last_frame_key:
            self.intervals[-1]['end'] = end
        else:
            if not isinstance(frame, dict):
                self.error = f'Missing applied decision frame at tick {start}'
                return
            self._last_frame = frame
            self._last_frame_key = frame_key
            self.bytes += payload_bytes(self._last_frame)
            interval = dict(start=start, end=end, actions={r:a.detach().clone() for r,a in actions.items()}, frame=self._last_frame)
            # Conservative container allowance includes list/dict growth and references.
            self.bytes += payload_bytes({k:v for k,v in interval.items() if k != 'frame'}) + 512
            self.intervals.append(interval)
        if checksum is not None:
            self.checks[end] = checksum
            self.bytes += payload_bytes((end, checksum))
        self._limit()

    def finish(self, tick):
        if not self.error and self.intervals and self.intervals[-1]['end'] == tick:
            self.complete = True
        elif not self.error:
            self.error = 'Last LIVE episode has no complete action record'
