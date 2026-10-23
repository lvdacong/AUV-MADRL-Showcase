from __future__ import annotations

from dataclasses import dataclass

UINT32_MASK = 0xFFFFFFFF
INT32_MAX = 0x7FFFFFFF


def _u32(value: int) -> int:
    return value & UINT32_MASK


def _i32(value: int) -> int:
    value &= UINT32_MASK
    return value if value <= INT32_MAX else value - (UINT32_MASK + 1)


def compose_episode_seed(worker_seed: int, area_index: int, episode_index: int) -> int:
    """Bit-for-bit port of the Unity episode-seed Compose function."""

    value = _u32(worker_seed)
    value ^= _u32(0x9E3779B9 + _u32(area_index) + _u32(value << 6) + (value >> 2))
    value = _u32(value)
    value ^= _u32(0x85EBCA6B + _u32(episode_index) + _u32(value << 6) + (value >> 2))
    return _u32(value) & INT32_MAX


@dataclass
class DotNetRandomState:
    inext: int
    inextp: int
    seed_array: tuple[int, ...]


class DotNetRandom:
    """Deterministic episode sampler used by GoalReach checkpoints."""

    MBIG = INT32_MAX
    MSEED = 161803398

    def __init__(self, seed: int):
        subtraction = INT32_MAX if seed == -0x80000000 else abs(int(seed))
        mj = _i32(self.MSEED - subtraction)
        seed_array = [0] * 56
        seed_array[55] = mj
        mk = 1
        for i in range(1, 55):
            ii = (21 * i) % 55
            seed_array[ii] = mk
            mk = _i32(mj - mk)
            if mk < 0:
                mk = _i32(mk + self.MBIG)
            mj = seed_array[ii]
        for _ in range(4):
            for i in range(1, 56):
                seed_array[i] = _i32(seed_array[i] - seed_array[1 + (i + 30) % 55])
                if seed_array[i] < 0:
                    seed_array[i] = _i32(seed_array[i] + self.MBIG)
        self._seed_array = seed_array
        self._inext = 0
        self._inextp = 21

    def state_dict(self) -> DotNetRandomState:
        return DotNetRandomState(self._inext, self._inextp, tuple(self._seed_array))

    def load_state_dict(self, state: DotNetRandomState) -> None:
        if len(state.seed_array) != 56:
            raise ValueError("Legacy System.Random state must contain 56 integers")
        self._inext = int(state.inext)
        self._inextp = int(state.inextp)
        self._seed_array = list(state.seed_array)

    def _internal_sample(self) -> int:
        loc_inext = self._inext + 1
        if loc_inext >= 56:
            loc_inext = 1
        loc_inextp = self._inextp + 1
        if loc_inextp >= 56:
            loc_inextp = 1
        result = _i32(self._seed_array[loc_inext] - self._seed_array[loc_inextp])
        if result == self.MBIG:
            result = _i32(result - 1)
        if result < 0:
            result = _i32(result + self.MBIG)
        self._seed_array[loc_inext] = result
        self._inext = loc_inext
        self._inextp = loc_inextp
        return result

    def random(self) -> float:
        return self._internal_sample() * (1.0 / self.MBIG)

    next_double = random

    def _large_range_sample(self) -> float:
        result = self._internal_sample()
        if self._internal_sample() % 2 == 0:
            result = -result
        value = result + (self.MBIG - 1)
        return value / (2 * self.MBIG - 1)

    def next_int(self, minimum: int | None = None, maximum: int | None = None) -> int:
        if minimum is None:
            if maximum is not None:
                raise ValueError("maximum cannot be supplied without minimum")
            return self._internal_sample()
        elif maximum is None:
            maximum = minimum
            minimum = 0
        if minimum > maximum:
            raise ValueError("minimum must be at most maximum")
        span = maximum - minimum
        if span <= self.MBIG:
            return int(self.random() * span) + minimum
        return int(self._large_range_sample() * span) + minimum
